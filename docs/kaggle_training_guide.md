# Kaggle Training Guide (Phase 2 → full-scale run)

This guide covers taking the pipeline built and locally sanity-tested in
Phase 2 to Kaggle Notebooks for the real, full-scale (or as-much-as-quota-allows)
training run on a GPU. **Nothing in this repo has trained a real model yet** —
this document only prepares you to do that run yourself on Kaggle.

Kaggle API/CLI credentials were not configured in this environment, so this
guide uses the manual notebook route (option (a) from the Phase 2 spec), not
`kaggle kernels push`.

## 1. What you're bringing to Kaggle

From this repo (branch `feature/model-training`):
- `model/model.py` — backbone (`convnext_tiny.fb_in22k_ft_in1k` via timm) + regression head + loss
- `model/dataset.py` — `PriceImageDataset` + transforms/augmentation
- `model/train.py` — the training loop, with `--subset-mode full`
- `scripts/download_images.py` — image download/resize (rerun on Kaggle against the full train split, or use a pre-downloaded image dataset if you create one)
- `data/processed/metadata.csv` — the full 73,517-row metadata (train/val/test split column already present)

Model checkpoints and downloaded images are **not** committed to git (see
`.gitignore` — `data/*`, `*.pt`, `*.pth`, `*.onnx`). You'll regenerate/download
them on Kaggle directly.

## 2. Set up the Kaggle notebook

1. Go to https://www.kaggle.com/code → **New Notebook**.
2. In the notebook settings (right sidebar): **Accelerator → GPU** (T4 x2 or
   P100, whichever is available/free-tier). Turn on **Internet** (needed to
   `pip install timm` and to download images from `m.media-amazon.com`, and
   for pretrained-weight downloads from Hugging Face Hub).
3. Add the dataset: **Add Input → Datasets → search `suvroo/amazon-ml`**, add
   it. This mounts the raw `train.csv` at `/kaggle/input/amazon-ml/train.csv`
   (exact path may vary — check the actual mounted path in the notebook's
   file browser after adding it).
4. Either:
   - **(a) Clone this repo's code into the notebook.** Simplest: use a
     notebook cell to `!git clone` your GitHub repo (public URL, or private
     with a token/deploy key you set up yourself — do this only if you're
     comfortable exposing a token inside the Kaggle notebook environment),
     then `%cd amazon-vision-price-predictor`. Or,
   - **(b) Upload the four files above** (`model/model.py`, `model/dataset.py`,
     `model/train.py`, `scripts/download_images.py`) as a **Kaggle Dataset**
     of your own ("Notebook code" dataset) and add it as input — avoids
     needing git credentials in the notebook at all.

## 3. Install dependencies in the notebook

Kaggle's base image already ships a CUDA-enabled `torch`, so **do not**
`pip install torch==2.14.0` there (that would reinstall the CPU-only PyPI
wheel and lose GPU support, exactly the situation to avoid). Only add what's
missing:

```python
!pip install -q "timm>=1.0.27"
```

Then verify GPU is actually visible before doing anything else:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If `torch.cuda.is_available()` is `False`, stop and fix the accelerator
setting (step 2) before proceeding — don't train on CPU on Kaggle, there's
no point.

## 4. Get metadata onto the Kaggle filesystem

Copy (or symlink) `data/processed/metadata.csv` from this repo into the
notebook's working directory, e.g.:

```python
import shutil, os
os.makedirs("data/processed", exist_ok=True)
shutil.copy("/kaggle/input/<your-code-dataset>/metadata.csv", "data/processed/metadata.csv")
```

(Or regenerate it from the raw Kaggle dataset with `scripts/data_cleaning.py`
+ `scripts/data_split.py`, which are also already in this repo — same
approach as Phase 1, just run in the Kaggle environment instead of locally.)

## 5. Download + resize the full image set

```python
!python scripts/download_images.py \
    --metadata data/processed/metadata.csv \
    --out-dir data/processed/images/full \
    --image-size 224 \
    --workers 16 \
    --max-retries 3
```

This is the same script used locally for the dev subset — unmodified. Expect
this step to take a while for ~51k+ images; it's resumable (safe to
re-run/continue if a Kaggle session times out), and it will print/report the
real number of failed downloads at the end — check that count before
training (a high failure rate signals a network or URL problem worth
investigating, not just skipping past).

**Kaggle session limits**: free-tier GPU sessions have a runtime limit
(historically ~9-12 hours per session, subject to change — check the current
limit in the Kaggle UI). If the full download + full training won't fit in
one session, either: split the download into a separate earlier notebook run
and save the resulting images as a Kaggle Dataset output (so a later notebook
just adds it as input rather than re-downloading), or rely on
`scripts/download_images.py`'s resumability across multiple sessions.

## 6. Run training

Sanity-checked locally with `--subset-mode dev`; on Kaggle use `full`:

```python
!python model/train.py \
    --subset-mode full \
    --epochs 15 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 2 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --log-path /kaggle/working/logs/train.log
```

Notes:
- `--epochs`, `--batch-size`, and `--lr` above are reasonable **starting
  points**, not tuned values — nothing in Phase 2 ran a real training job to
  validate them. Expect to adjust based on the actual train/val loss curves
  you observe.
- A checkpoint is saved every epoch (`epoch_N.pt`) plus a `best.pt` (lowest
  val loss so far) in `--checkpoint-dir`, so a run interrupted by a Kaggle
  session timeout can be resumed by loading the last saved epoch checkpoint
  (note: the current `train.py` does not yet have a `--resume-from` flag to
  automatically reload a checkpoint's weights/optimizer state and continue —
  add that if you need seamless multi-session resume, or reload the
  `model_state_dict`/`optimizer_state_dict` manually in a notebook cell).
- `train.log` (or wherever `--log-path` points) has per-batch and per-epoch
  train/val loss — download it via Kaggle's output file browser to inspect
  after the run.

## 7. Save outputs

Before your Kaggle session ends, make sure to persist what you need:
- Commit the notebook itself (Kaggle does this automatically as you save).
- Download `/kaggle/working/checkpoints/best.pt` (and/or the per-epoch
  checkpoints) and `/kaggle/working/logs/train.log` to your local machine, or
  save `/kaggle/working` as a Kaggle Dataset output for the next phase
  (Phase 3 — evaluation against the held-out test set) to consume.

## 8. What Phase 2 did NOT do (do not treat as done)

- No real training run happened — only a 1-2 epoch, small-batch CPU sanity
  check on the ~15k-row dev subset, to prove the pipeline runs end-to-end.
- No evaluation against the held-out `test` split (that's Phase 3).
- No `kaggle kernels push` / Kaggle API automation was used (API/CLI
  credentials were not confirmed as configured in this environment).
