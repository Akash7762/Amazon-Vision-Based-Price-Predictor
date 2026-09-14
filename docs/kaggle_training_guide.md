# Kaggle Training Guide (full-scale GPU run)

Phase 2 built and locally sanity-tested the training pipeline on CPU. This
guide covers the actual full-scale training run, which happens on Kaggle
Notebooks.

**Nothing in this repo has trained a real model yet.** The only checkpoints
produced locally came from a handful of batches on CPU to prove the pipeline
executes; they are not a usable model. Following this guide is what produces
one.

---

## Step 0 — Enable GPU and Internet on your Kaggle account

If **Settings → Accelerator** offers no GPU option, the cause is almost
certainly that your account is not phone-verified. Kaggle gates both GPU/TPU
accelerators *and* internet access behind phone verification.

**Confirm the diagnosis:** check the **Internet** toggle in the same panel.
If Internet is also unavailable, it is phone verification — the two unlock
together.

**Fix:** kaggle.com/settings → Phone Verification → verify your number, then
reopen/restart the notebook.

This pipeline needs Internet regardless of the GPU, because it downloads
images from `m.media-amazon.com`, pip-installs `timm`, and fetches pretrained
weights from Hugging Face Hub.

If you are already verified and still see no GPU, check: weekly GPU quota
exhausted (roughly 30 hrs/week, resets weekly — verify the current figure in
Kaggle's docs, these limits change); you are viewing someone else's notebook
read-only rather than your own copy in edit mode; or the notebook is attached
to a competition whose host disabled accelerators.

---

## Strategy: two notebooks, not one

| | Notebook A — Download | Notebook B — Train |
|---|---|---|
| Accelerator | **None (CPU)** | **GPU** |
| Job | fetch + resize ~73.5k images | train the model |

Downloading needs zero GPU. Running it inside a GPU session spends your
limited weekly GPU quota on network waiting. Notebook A's output becomes a
Kaggle Dataset that Notebook B attaches read-only, so you download **once**
and can then train repeatedly — including after a crash, and again for Phase
3 evaluation — without re-downloading.

Download **all 73,517 rows** (train + val + test), not just the 51k train
split. Phase 3 needs the test images, and one download job is cheaper than
two.

---

## Step 1 — Upload your inputs as Kaggle Datasets

### metadata.csv

Upload `data/processed/metadata.csv` (72 MB) at kaggle.com/datasets → New
Dataset.

**Do not regenerate it on Kaggle** from the raw `train.csv` via
`data_cleaning.py` + `data_split.py`. Those use `train_test_split` under a
specific scikit-learn version; regenerating under a different version risks
producing a *different* train/val/test split. That would silently leak test
rows into training and invalidate every Phase 3 metric. Upload the exact file
the splits were made from.

### Code

- **Public repo:** `!git clone <your repo URL>` in a notebook cell. Easiest,
  and stays in sync with future changes.
- **Private repo:** upload `model/` and `scripts/` as a small Kaggle Dataset
  instead. Avoids putting a GitHub token inside the notebook environment,
  which is worth avoiding.

---

## Step 2 — Notebook A: download the images (CPU)

New Notebook → Accelerator **None**, Internet **On**. Attach the metadata
dataset and your code.

```python
!pip install -q pillow requests pandas
!python scripts/download_images.py \
    --metadata /kaggle/input/<your-metadata-dataset>/metadata.csv \
    --out-dir /kaggle/working/images \
    --image-size 224 --workers 32 --max-retries 3
```

Run this with **Save Version → "Save & Run All (Commit)"**, not
interactively. Interactive sessions shut down after a period of inactivity; a
committed run continues in the background with your browser closed.

Notes:
- Images are **letterbox-padded** to 224x224 (aspect ratio preserved, padded
  to square with white), not stretched. See `model/preprocessing.py`.
  Phase 4 inference must import and apply that same function.
- The script writes each JPEG to a temp file and atomically renames it, so an
  interrupted run never leaves a truncated image. Just re-run to resume;
  completed files are skipped.
- A `_preprocessing.json` manifest is written into the output directory. If
  you later re-run with different resize settings against that directory, the
  script refuses rather than silently mixing incompatible images.
- **Storage:** ~73.5k images at roughly 15 KB each is about 1.1 GB, well
  under the 20 GB `/kaggle/working` cap.
- **Timing:** locally this measured ~6 images/sec at 16 workers. Kaggle's
  bandwidth is better but Amazon may rate-limit at 32 workers. Read the first
  `progress: 1000/73517` line and extrapolate rather than trusting an
  estimate.
- **Check the failure count at the end.** Locally it was 0 out of 18,214. If
  Kaggle reports thousands, that is rate-limiting rather than dead URLs —
  re-run (it resumes) instead of accepting the loss.

## Step 3 — Turn the images into a reusable Dataset

When Notebook A finishes: open its **Output** tab → **New Dataset**. This is
the step that makes everything downstream repeatable.

---

## Step 4 — Notebook B: verify GPU, then smoke-test

New Notebook → Accelerator **GPU**, Internet **On**. Attach the image
dataset, the metadata dataset, and your code.

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If `cuda.is_available()` is `False`, **stop** and fix the accelerator setting.
Do not proceed — you would be training on CPU and burning hours for nothing.

```python
!pip install -q "timm>=1.0.27"
```

**Do not** `pip install torch==2.14.0`. Kaggle's image already ships a
CUDA-enabled torch; installing the PyPI wheel replaces it with the CPU-only
build. (`requirements.txt` pins the CPU wheel because that is what the local
Windows dev machine has — it is not the right pin for Kaggle.)

Then a deliberate smoke test, to catch path and permission errors in two
minutes rather than two hours:

```python
!python model/train.py --subset-mode full \
    --metadata /kaggle/input/<metadata-dataset>/metadata.csv \
    --image-dir /kaggle/input/<images-dataset>/images \
    --epochs 1 --max-batches 5 --batch-size 64 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --log-path /kaggle/working/logs/smoke.log
```

`--metadata` and `--image-dir` are **required on Kaggle**. Without them the
script falls back to the repo-relative defaults (`data/processed/...`), which
do not exist there. It fails fast with an explicit message rather than a
stack trace if a path is wrong.

## Step 5 — The real run

```python
!python model/train.py --subset-mode full \
    --metadata /kaggle/input/<metadata-dataset>/metadata.csv \
    --image-dir /kaggle/input/<images-dataset>/images \
    --epochs 15 --batch-size 64 --lr 1e-4 --num-workers 4 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --log-path /kaggle/working/logs/train.log
```

Again via **Save & Run All**.

- **Time it honestly.** Let epoch 1 finish, read `elapsed_sec` from the log,
  multiply by 15. If that exceeds the session limit (~9 hrs on GPU as of
  writing — verify current limits), reduce epochs and continue from the last
  checkpoint instead of losing the session.
- `--epochs 15`, `--batch-size 64`, `--lr 1e-4` are **starting points, not
  tuned values.** Nothing has validated them. Watch the val loss: still
  falling at epoch 15 means train longer; bottomed out at epoch 5 and rising
  means overfitting, and `best.pt` already holds the best epoch.
- A checkpoint is saved every epoch plus a `best.pt` tracking lowest val
  loss. Note there is still **no `--resume-from` flag** — resuming after a
  session timeout means loading `model_state_dict` / `optimizer_state_dict`
  from a checkpoint manually in a notebook cell.

## Step 6 — Get your results out

Before the session ends, download from the Output tab:

- `checkpoints/best.pt` — the trained model, the actual deliverable
- `logs/train.log` and `checkpoints/history.json` — your loss curves, needed
  for your report and viva

Also save `/kaggle/working` as a Dataset so Phase 3 can attach it directly
rather than re-uploading a ~330 MB checkpoint.

---

## Note on the local dev image cache

`data/processed/images/dev` was downloaded **before** letterbox-padding was
introduced, so those 18,214 images are stretched and inconsistent with
current code. The downloader will refuse to add to that directory. To refresh
it for local work:

```bash
python scripts/download_images.py \
    --metadata data/processed/dev_subset.csv \
    --out-dir data/processed/images/dev \
    --image-size 224 --workers 16 --overwrite
```

This only matters for local sanity checks; the Kaggle run downloads fresh
images with the correct preprocessing either way.

## What is still not done

- No real training run has happened yet — this guide is how you do it.
- No evaluation against the held-out `test` split (Phase 3).
- No `kaggle kernels push` automation; Kaggle API credentials were never
  configured, so this is the manual notebook route.
