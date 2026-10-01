# Kaggle Training Guide — Phase 2 end to end

This covers all of Phase 2 on Kaggle, starting from a fresh notebook: download
the images, train, and decide which checkpoint goes to Phase 3.

The first real model, run1, was trained on Oct 1 (val MAE well below the
median-price baseline). Results for every run are in
[`experiments.md`](experiments.md).

It uses two notebooks:

| Notebook | Job | Accelerator | Output |
|---|---|---|---|
| **A — Download** | fetch and resize ~73.5k images | **None (CPU)** | `images.zip`, made into a Dataset |
| **B — Train** | train one run, score it on val, compare with an earlier run | **GPU** | `best.pt`, `history.json`, `eval/` metrics |

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

## Notebook B — Train and compare (GPU)

The cells live in [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb),
one notebook for every run: **Cell 0 holds the run's settings and is the only
cell you edit.** Import the file into Kaggle rather than retyping it. (run1
used an earlier 6-cell version, still in git history at `fd359e4`.)

| Cell | What it does | Time on a T4 |
|---|---|---|
| 0 | settings: run name, target, epochs, which earlier run to compare with | — |
| 1 | checks the GPU really works (runs a convolution) | seconds |
| 2 | clones the code, refuses to go on if GitHub has old code, installs timm | ~1 min |
| 3 | fixed data paths, image check, finds the earlier run's `best.pt` | ~1 min |
| 4 | scores the earlier run on val with `model/evaluate.py` | ~2 min |
| 5 | smoke test: 5 batches with this run's settings | ~2 min |
| 6 | the training run | ~15 min per epoch |
| 7 | scores this run's `best.pt` on val, then compares the two | ~2 min |
| 8 | summary, loss and dollar-error plots, trims old checkpoints | seconds |

Why the cells look the way they do:

- **Every shell command goes through `run()`**, which shows output live and
  **raises if the command fails**. A plain `!command` ignores failures, so a
  broken smoke test would still roll on into the long run. Paths are quoted
  with `q()`.
- **`python -u`** so training lines reach the log as they happen.
- **`--max-hours 8`**: before each epoch the script checks another epoch
  still fits; if not it stops cleanly, so the run finishes and saves its
  output instead of being killed at the session limit.
- **Checkpoints are written to a temp file and renamed**, so a session dying
  mid-save can't corrupt `best.pt`.
- **`--select-by val_mae`**: `best.pt` is the epoch with the lowest val
  error *in dollars*. The loss can't be compared across targets (with
  `--target log` it's in log-dollars), and the epoch with the lowest loss
  isn't always the one with the lowest dollar error.
- **Cell 4 scores the earlier run before training**, with the same script
  and the same images as Cell 7. That makes the comparison like-for-like,
  and if anything is wrong with the evaluation it shows up in minute 5, not
  after the long run.

### B0. Before you start

1. On your PC: `git push`, so GitHub has the latest `feature/train-resume`.
2. On Kaggle: **View Active Events** (bottom left of any editor) → stop
   everything that's running. An open GPU session is the usual reason a saved
   run sits in **Queued**.
3. Check your GPU hours left (right panel → Session options). A 6-epoch run
   uses about 1¾ hours, a 15-epoch run about 4.

### B1. Create the notebook

1. **Create → New Notebook** → **File → Import Notebook** → upload
   `notebooks/kaggle_train.ipynb`.
2. Right panel → **Session options**: Accelerator **GPU T4 x2**, Internet
   **On**.
3. **Add Input**, three of them:
   - `amazon-price-metadata`
   - `amazon-price-images-v2`
   - the earlier run's output, e.g. `amazon-price-run1`. Attach it either as
     the Dataset you made from its output, or straight from **Add Input →
     Notebooks → Your Work**. Not both, or Cell 3 finds two `best.pt` files
     and stops.

### B2. Set Cell 0

For run2 it already reads:

```python
RUN_NAME = "run2"
TARGET = "log"
EPOCHS = 6
COMPARE_WITH = "auto"
COMPARE_NAME = "run1"
```

For a later run, change the name and the one thing being tested. If you
don't want a comparison, set `COMPARE_WITH = None` and skip the third input.

### B3. Rehearse Cells 0–5 interactively

Shift + Enter through Cells 0–5. **Don't run 6, 7 or 8.**

| Cell | Good output |
|---|---|
| 0 | `run2: target=log, epochs=6, compare with run1 (auto)` |
| 1 | `GPU: Tesla T4` and `GPU works, test output shape: (8, 16, 222, 222)` |
| 2 | a commit line, `code is up to date`, then `torch ... \| timm ... \| cuda OK`. Red pip "dependency resolver" warnings above it are harmless. |
| 3 | `images=73517 missing=0`, the manifest with `"resize_mode": "letterbox_pad"`, and exactly one `earlier-run checkpoints found:` path ending in `checkpoints/best.pt` |
| 4 | `=== run1 on val: 11028 images, epoch 14, target=price ...` and two tables. Its MAE is run1's real val error in dollars. |
| 5 | `target=log`, 51461 train / 11028 val usable rows, `avg_loss=... mae=$...`, `=== Training run end ===` |

### B4. Stop the session, then save the version

1. **Stop the interactive session first** (power icon / Stop session), and
   check View Active Events shows nothing running.
2. **Save Version** → name it after the run (e.g. `run2 - log price e6`) →
   **Save & Run All (Commit)** → Advanced: always save output → **Save**.
   If Save Version is greyed out without a session, start one, save, and
   stop it straight away.
3. Close the editor tab. **Opening the editor again can start a new
   session.** Follow progress from the notebook's page (Your Work → Code).

### B5. While it runs

- **Queued** is normal and uses no GPU hours. Still queued after an hour?
  Check View Active Events for a session you forgot.
- **Running**: Cells 0–5 take ~8–10 min. Then a `batch=` line roughly every
  minute and an epoch summary every ~15 min, now with `mae=$...` on each
  `DONE` line. 6 epochs ≈ 1.5 h, plus ~5 min for Cells 7–8.

### B6. When it finishes

Output holds `checkpoints/best.pt` and `checkpoints/epoch_<last>.pt`
(~334 MB each), `checkpoints/history.json`, `logs/train.log`,
`loss_curve.png`, and `eval/` with each run's metrics (`*_val.json`),
per-image predictions (`*_val_predictions.csv`, worst first) and
`comparison_val.md`.

1. Download `history.json`, `train.log`, `loss_curve.png` and the `eval/`
   folder. `best.pt` too if this run turns out to be the keeper.
2. Output → **New Dataset** → e.g. `amazon-price-run2`, so later runs and
   Phase 3 can attach it.

### B7. Only if Cell 8 says "Stopped early by the time budget"

Make the stopped run's output a Dataset (e.g. `amazon-price-run2`), attach
it, and in Cell 0 set `COMPARE_WITH` to run1's exact path instead of
`"auto"` (with two runs attached, auto finds two). Then replace Cell 6 with
this, changing `epoch_3.pt` to the file Cell 8 named:

```python
# Cell 6 (resume) - continues to EPOCHS from the last saved epoch
PREV = "/kaggle/input/datasets/akashverma7762/amazon-price-run2/checkpoints"
print(sorted(os.listdir(PREV)))
run(f"python -u model/train.py --subset-mode full --metadata {q(META)} --image-dir {q(IMG)} "
    f"--target {TARGET} --select-by val_mae --epochs {EPOCHS} --batch-size 64 --lr 1e-4 "
    f"--num-workers 4 --max-hours 8 --resume-from {q(PREV + '/epoch_3.pt')} "
    f"--checkpoint-dir /kaggle/working/checkpoints --log-path /kaggle/working/logs/train.log")
```

`--epochs` is the final epoch number, not how many more. The target must
match the checkpoint (the script refuses otherwise). Weights, optimizer
state, history and the best-so-far score are restored; shuffle order and RNG
state are not, so say so in the report. Then repeat B4–B6.

### B8. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| Cell 1 `AssertionError: No GPU` | accelerator not set | Session options → GPU T4 x2, restart session |
| Cell 1 `no kernel image is available` | P100 selected | switch to GPU T4 x2 |
| Cell 2 `Could not resolve host: github.com` | Internet off | Session options → Internet On |
| Cell 2 `OLD CODE on GitHub` | branch not pushed | `git push` on your PC, rerun Cell 2 |
| Cell 3 `Attach exactly one metadata and one images dataset` | wrong / extra datasets | attach only metadata + images-v2 (+ the earlier run) |
| Cell 3 `Expected exactly one attached earlier run` | earlier run not attached, or attached twice | attach it once, or set `COMPARE_WITH` to a path / `None` |
| `CUDA out of memory` | unexpected at batch 64 on a T4 | tell me before changing batch size, it changes the experiment |
| `DataLoader worker ... killed` / `bus error` | shared-memory limits | `--num-workers 4` → `2` in Cells 5 and 6 |
| HF `unauthenticated requests` warning | no Hugging Face token | harmless |

### B9. Read the results and decide

This is roadmap Phase 2, step 4. Record every run in
[`experiments.md`](experiments.md), including the ones that lose.

**The bar to beat** is always predicting the median train price ($14.00):
val MAE **$14.52**, val SmoothL1 14.03.

**Compare runs on val MAE in dollars**, from Cell 7's comparison: same
script, same 11,028 images. The table under it gives the paired-bootstrap
95% interval for the difference. If the interval includes 0, the runs
aren't distinguishable on this val set. That interval covers which images
landed in val; it does **not** cover training randomness (seed, shuffle
order), so a small gap is weak evidence even when the interval excludes 0.

Reading a curve:

| What you see | What it means | What to do next |
|---|---|---|
| val still improving at the last epoch | undertrained | more epochs via B7 |
| val bottoms out, then rises | overfitting | `best.pt` already holds the best epoch |
| val flat from epoch 1 | not learning | check `--lr`; re-read the Cell 5 output |
| val flat while train keeps falling (run1) | memorising | regularisation / augmentation, not more epochs |

Change **one** thing per run. Your weekly GPU hours cap how many runs you
get, so spend them where the curve points.

**Phase 2 is done when** you can say why the checkpoint you kept is the one
you kept, from numbers you'd defend in the viva. Then Phase 3 scores it once
on the test split.

---

## Known gaps

- No evaluation on the held-out `test` split yet. That's Phase 3. Don't
  look at test metrics while tuning in B9.
- No `kaggle kernels push` automation. Kaggle API credentials were never set
  up, so this is the manual notebook route.
- Tuning so far is one change at a time (run2: log-price target); no
  systematic hyperparameter search.
- The local `data/processed/images/dev` cache was downloaded before
  letterbox padding existed, so those images are stretched. That only
  matters for local sanity runs. To refresh it:

  ```bash
  python scripts/download_images.py \
      --metadata data/processed/dev_subset.csv \
      --out-dir data/processed/images/dev \
      --image-size 224 --workers 16 --overwrite
  ```
