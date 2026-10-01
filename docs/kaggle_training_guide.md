# Kaggle Training Guide — Phase 2 end to end

This covers all of Phase 2 on Kaggle, starting from a fresh notebook: download
the images, train, and decide which checkpoint goes to Phase 3.

**No real model has been trained yet.** The only checkpoints so far came from
5-batch CPU sanity runs proving the code executes.

It uses two notebooks:

| Notebook | Job | Accelerator | Output |
|---|---|---|---|
| **A — Download** | fetch and resize ~73.5k images | **None (CPU)** | `images.zip`, made into a Dataset |
| **B — Train** | smoke test, then the real run | **GPU** | `best.pt`, `history.json`, logs |

Downloading needs no GPU. Running it in a GPU session uses up your weekly GPU
quota while the notebook waits on the network.

**Rule for every step: don't start the next one until the check passes.** A
wrong image path or a CPU-only session takes two minutes to catch early and
hours to catch late.

---

## Kaggle limitations this guide is built around

These figures change. Check them against Kaggle's docs or your account page
before relying on them.

| Limit | Roughly | What it means here |
|---|---|---|
| GPU quota | ~30 hrs/week, resets weekly | Download on CPU. Do smoke tests before long runs. |
| Max session length | 12 hrs at time of writing | **Plan runs to finish in ≤ 9 hrs** for margin. Don't assume a run killed at the limit keeps its output. |
| Interactive sessions | stop after inactivity; nothing saved unless you save a version | Do real work only with **Save & Run All (Commit)**. |
| `/kaggle/working` | ~20 GB, **saved** as output | Only put what you want to keep here. |
| `/tmp` | scratch disk, **not saved** | Put working files here. |
| Output with tens of thousands of files | slow to save; turning it into a Dataset was unreliable on this project (Sep 14: the dialog showed 446 of 73k files) | **Zip the images into one file** before saving. |
| `/kaggle/input` | read-only | Checkpoints go to `/kaggle/working`, never next to the inputs. |
| Internet | needs a phone-verified account | Required for `git clone`, `pip`, pretrained weights, and image downloads. |
| CPU cores | ~4 | `--num-workers 4`. Data loading may be the bottleneck, not the GPU. |
| GPU choice | T4 ×2 or P100 | Pick **T4 ×2**. Recent PyTorch builds have been dropping support for older GPUs like the P100. The script uses **one** GPU, so the second T4 sits idle. |
| Committed runs | start from a **fresh** environment and run every cell top to bottom | Every cell must stand alone. Nothing from your interactive session carries over. |
| Concurrent GPU sessions | limited; a saved run waits in **Queued** while another GPU session of yours is open | Stop the interactive session before saving. Watch progress from the notebook page, not the editor, since opening the editor can start a session. |
| `!command` in a cell | a failing command doesn't stop the notebook | Notebook B wraps commands in `run()`, which raises on failure. |

---

## Step 0 — One-time setup

### 0.1 Account

Open any notebook → **Settings → Accelerator**. No GPU option? Check the
**Internet** toggle. If both are unavailable, verify your phone at
kaggle.com/settings, then reopen the notebook.

If you're verified and still don't see a GPU:
- your weekly quota may be used up
- you may be viewing someone else's notebook instead of your own editable copy
- the notebook may be attached to a competition that disables accelerators

**Check:** the GPU option appears, and Internet can be turned on.

### 0.2 Code on GitHub

The notebooks `git clone` the repo. `--resume-from` and `--log-every` only
exist on `feature/train-resume` until that's merged, so clone that branch.
It has to be **pushed** first:

```bash
git push -u origin feature/train-resume
```

### 0.3 metadata.csv as a Dataset

`metadata.csv` is gitignored, so it won't come with the clone. If you already
have a Kaggle Dataset with it, reuse it. Otherwise upload
`data/processed/metadata.csv` (72 MB) at kaggle.com/datasets → New Dataset.

**Don't regenerate it on Kaggle** from raw `train.csv`. The split comes from
`train_test_split`, and a different scikit-learn version can produce a
different split. That would leak test rows into training and make every
Phase 3 number wrong without any error. Upload the exact file.

**Check:** you can attach the metadata dataset to a notebook.

---

## Notebook A — Download the images (CPU)

New Notebook → Accelerator **None**, Internet **On**. Attach the metadata
dataset.

### A1. Clone the code

```python
!rm -rf /tmp/repo && git clone -b feature/train-resume --depth 1 \
    https://github.com/Akash7762/Amazon-Vision-Based-Price-Predictor.git /tmp/repo
%cd /tmp/repo
!pip install -q pillow requests pandas
```

### A2. Find metadata.csv

```python
import glob
hits = glob.glob("/kaggle/input/**/metadata.csv", recursive=True)
print(hits)
META = hits[0]
```

If more than one path prints, set `META` to the one you mean.

### A3. Test with 1,000 rows

```python
%%time
!python scripts/download_images.py --metadata {META} \
    --out-dir /tmp/images_test --image-size 224 --workers 32 --limit 1000
```

(`%%time` has to be the first line of the cell, so put `META` in its own
cell above.)

**Check:** `Failed (after retries)` is close to 0. Work out the rate as
1,000 ÷ the wall time, then how long 73,517 will take. It needs to finish
well under 9 hours. If failures climb, lower `--workers` to 16: that's
usually Amazon rate-limiting, not dead links.

**Run A1–A3 interactively, then stop.** Don't run the full download
interactively. The commit in A6 runs everything again from scratch, so you'd
download twice.

### A4. Full download, to `/tmp`

```python
!python scripts/download_images.py --metadata {META} \
    --out-dir /tmp/images --image-size 224 --workers 32 --max-retries 3
```

This writes to `/tmp`, not `/kaggle/working`. 73k loose files in the output
is what caused trouble last time. Only the zip from A5 goes into the output.

This downloads **all 73,517 rows** (train, val and test). Phase 3 needs the
test images, and one download is cheaper than two.

What the script handles:
- **Letterbox padding.** Images are padded to 224×224 with white borders, not
  stretched (`model/preprocessing.py`). Phase 4 inference must use the same
  function.
- **Safe writes.** Each JPEG is written to a temp file and then renamed, so
  there are never half-written images.
- **A manifest.** `_preprocessing.json` records the settings.
- **A failures file.** `_download_failures.csv` lists what failed.

### A5. Verify, then zip

```python
import os, zipfile, pandas as pd

ids  = set(pd.read_csv(META)["sample_id"].astype(str))
have = {f[:-4] for f in os.listdir("/tmp/images") if f.endswith(".jpg")}
print(f"metadata={len(ids)} images={len(have)} missing={len(ids - have)}")
assert len(ids - have) < 100, "too many missing: rate-limited? re-run A4, it resumes"

# ZIP_STORED: JPEGs are already compressed, so deflating them again only costs time
with zipfile.ZipFile("/kaggle/working/images.zip", "w", zipfile.ZIP_STORED) as z:
    for f in sorted(os.listdir("/tmp/images")):
        z.write(os.path.join("/tmp/images", f), arcname=f"images/{f}")

with zipfile.ZipFile("/kaggle/working/images.zip") as z:
    n = sum(1 for x in z.namelist() if x.endswith(".jpg"))
print(f"zip holds {n} jpgs, {os.path.getsize('/kaggle/working/images.zip')/1e9:.2f} GB")
```

**Check:** `missing` is in single digits (on Sep 14, 1 of 73,517 failed for
good), and the zip holds the same number of JPEGs. The `assert` makes the
committed run fail loudly instead of saving a partial dataset.

### A6. Commit

**Save Version → Save & Run All (Commit).** You can close the browser. Watch
progress from the notebook's **Versions** list.

When it finishes, open that version → **Output**. You should see a single
`images.zip` of about 1 GB. Click **New Dataset** and name it something
distinct, like `amazon-price-images-v2`, so it can't be confused with the old
dataset that has duplicate folders.

**Check:** the new Dataset exists. Kaggle may keep it as `images.zip` or
extract it into an `images/` folder. On Sep 26 it extracted it, under
`/kaggle/input/datasets/<username>/amazon-price-images-v2/images`.

---

## Notebook B — Train (GPU)

The cells live in [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb).
Import that file into Kaggle rather than retyping it.

What the cells do differently from a plain `!python ...` notebook, and why:

- **Every shell command goes through a `run()` helper** that prints output
  live and **raises if the command fails**. A plain `!command` ignores
  failures, so a broken smoke test would still roll on into the 4-hour run.
- **`python -u`** so training lines reach the log as they happen.
- **`--max-hours 8`** on the real run. Before each epoch the script checks
  whether another epoch still fits; if not, it stops cleanly, so the run
  finishes and saves its output instead of being killed at the session
  limit.
- **Checkpoints are written to a temp file and renamed**, so a session dying
  mid-save can't corrupt `best.pt`.
- **Cell 1 runs a real convolution on the GPU**, so a card this torch build
  can't drive (e.g. P100) fails in seconds.
- **Cell 2 refuses to continue if GitHub has old code** (it checks for
  `--max-hours` in `train.py`). Push before you start.
- **Cell 3 uses fixed paths** and only falls back to a shallow search that
  never lists the 73k-image folder.

### B0. Before you start

1. On your PC: `git push`, so GitHub has the latest `feature/train-resume`.
2. On Kaggle: **View Active Events** (bottom left of any editor) → stop
   everything that's running. An open GPU session is the usual reason a saved
   run sits in **Queued**.
3. Note your GPU hours left (right panel → Session options in any
   notebook). One full run uses about 4.

### B1. Create the notebook

1. **Create → New Notebook.** Then **File → Import Notebook** and upload
   `notebooks/kaggle_train.ipynb` (or paste the six cells by hand).
2. Right panel → **Session options**: Accelerator **GPU T4 x2**, Internet
   **On**.
3. **Add Input**: `amazon-price-metadata` and `amazon-price-images-v2`. Only
   those two.

### B2. Rehearse Cells 1–4 interactively

Run Cells 1–4 with Shift + Enter. **Don't run Cells 5 or 6.**

| Cell | Good output |
|---|---|
| 1 | `GPU: Tesla T4` and `GPU works, test output shape: (8, 16, 222, 222)` |
| 2 | a commit line, `code is up to date`, then `torch ... \| timm ... \| cuda OK`. Red pip "dependency resolver" warnings above it are harmless. |
| 3 | `images=73517 missing=0` and the manifest with `"resize_mode": "letterbox_pad"` (the listing takes ~1 min) |
| 4 | `device=cuda`, 51461 train / 11028 val usable rows, a finite `avg_loss`, `=== Training run end ===` |

### B3. Stop the session, then save the version

1. **Stop the interactive session first** (power icon / Stop session), and
   check View Active Events shows nothing running. If it's left open, it
   uses GPU hours and can hold the saved run in the queue.
2. **Save Version** → name `run1 - convnext_tiny lr1e-4 bs64 e15` →
   **Save & Run All (Commit)** → Advanced: always save output → **Save**.
   If Save Version is greyed out without a session, start one, save, and
   stop it straight away.
3. Close the editor tab. **Opening the editor again can start a new
   session.** Follow progress from the notebook's page (Your Work → Code)
   instead.

### B4. While it runs

- **Queued** is normal. No GPU hours are used. If it's still queued after an
  hour, check View Active Events for a session you forgot.
- **Running**: Cells 1–4 take ~5–8 min. Then a `batch=` line roughly every
  minute, and an epoch summary every ~15 min (measured Sep 26: ~900 s per
  epoch). 15 epochs ≈ 3.8 hours.
- At the end the log shows `best_epoch=… best_val_loss=…` and Cell 6 prints
  the summary against the median baseline (14.03).

### B5. When it finishes

Output should hold `checkpoints/best.pt` and `checkpoints/epoch_<last>.pt`
(~334 MB each), `checkpoints/history.json`, `logs/train.log` and
`loss_curve.png`.

1. Download `best.pt`, `history.json`, `train.log`, `loss_curve.png` into
   `model/checkpoints/` and `model/logs/` locally (both gitignored).
2. Output → **New Dataset** → `amazon-price-run1`, so Phase 3 and any resume
   can attach it.

### B6. Only if Cell 6 says "Stopped early by the time budget"

Open the notebook, attach `amazon-price-run1` as a third input, and replace
Cell 5 with this (change `epoch_9.pt` to the file Cell 6 named):

```python
# Cell 5 (resume) - carries on to epoch 15 from the last saved epoch
EPOCHS = 15
PREV = "/kaggle/input/datasets/akashverma7762/amazon-price-run1/checkpoints"
print(sorted(os.listdir(PREV)))
run(f"python -u model/train.py --subset-mode full --metadata {META} --image-dir {IMG} "
    f"--epochs {EPOCHS} --batch-size 64 --lr 1e-4 --num-workers 4 --max-hours 8 "
    f"--resume-from {PREV}/epoch_9.pt "
    f"--checkpoint-dir /kaggle/working/checkpoints --log-path /kaggle/working/logs/train.log")
```

`--epochs` is the final epoch number, not how many more. Weights, optimizer
state, loss history and the best-val-loss are restored; shuffle order and
RNG state are not, so say so in the report. Then repeat B3–B5.

### B7. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| Cell 1 `AssertionError: No GPU` | accelerator not set | Session options → GPU T4 x2, restart session |
| Cell 1 `no kernel image is available` | P100 selected | switch to GPU T4 x2 |
| Cell 2 `Could not resolve host: github.com` | Internet off | Session options → Internet On |
| Cell 2 `OLD CODE on GitHub` | branch not pushed | `git push` on your PC, rerun Cell 2 |
| Cell 3 `Attach exactly two inputs` | wrong / extra datasets | attach only metadata + images-v2 |
| Cell 4/5 `CUDA out of memory` | unexpected at batch 64 on a T4 (the smoke test passed) | tell me before changing batch size, it changes the experiment |
| Cell 4/5 `DataLoader worker ... killed` / `bus error` | shared-memory limits | change `--num-workers 4` to `2` |
| HF `unauthenticated requests` warning | no Hugging Face token | harmless |

### B8. Read the curve and decide

This is roadmap Phase 2, step 4.

**First, the bar to beat.** Always predicting the median train price
($14.00) scores **val SmoothL1 = 14.03** (MAE $14.52). The model has to get
clearly below that to have learned anything from the images. The 100-batch
timing run on Sep 26 already reached 12.86 on part of the val set.

Reading the loss: with `beta=1`, SmoothL1 is roughly MAE − 0.5 once errors
are above $1, so a val loss of 10 means predictions are typically about
$10.50 off.

| What you see | What it means | What to do next |
|---|---|---|
| val loss still falling at the last epoch | undertrained | more epochs via B6 |
| val loss bottoms out, then rises | overfitting | `best.pt` already holds the best epoch. Try stronger augmentation or fewer epochs. |
| val loss flat from epoch 1 | not learning | check `--lr`; re-read the Cell 4 output |
| train loss far below val loss, both flat | memorising | more augmentation or regularisation |

For tuning, change **one** thing per run and keep each run's
`history.json` as its own Dataset. Your weekly GPU hours cap how many runs
you get, so spend them where the curve points.

**Phase 2 is done when** you can say, from the curve, why the checkpoint you
kept is the one you kept. That's where Phase 3 starts, and it's a likely
viva question.

---

## Known gaps

- No evaluation on the held-out `test` split yet. That's Phase 3. Don't
  look at test metrics while tuning in B8.
- No `kaggle kernels push` automation. Kaggle API credentials were never set
  up, so this is the manual notebook route.
- The hyperparameters are unvalidated defaults. No search has been run.
- The local `data/processed/images/dev` cache was downloaded before
  letterbox padding existed, so those images are stretched. That only
  matters for local sanity runs. To refresh it:

  ```bash
  python scripts/download_images.py \
      --metadata data/processed/dev_subset.csv \
      --out-dir data/processed/images/dev \
      --image-size 224 --workers 16 --overwrite
  ```
