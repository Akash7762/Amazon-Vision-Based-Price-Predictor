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
extract it into an `images/` folder. Step B2 handles both.

---

## Notebook B — Train (GPU)

New Notebook → Accelerator **GPU T4 ×2**, Internet **On**. Attach
`amazon-price-images-v2` and the metadata dataset.

Every cell below must work in a fresh session, because the commit runs them
all from the top.

### B1. GPU check, then code

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "no GPU: fix Settings -> Accelerator before going further"
```

```python
!rm -rf /tmp/repo && git clone -b feature/train-resume --depth 1 \
    https://github.com/Akash7762/Amazon-Vision-Based-Price-Predictor.git /tmp/repo
%cd /tmp/repo
!pip install -q "timm>=1.0.27"
```

**Don't** `pip install torch` or `pip install -r requirements.txt`. Kaggle
already has a CUDA build of torch. `requirements.txt` pins the CPU-only
wheel for the local Windows machine, and installing it would replace the GPU
build.

### B2. Point at the images

Kaggle now mounts datasets under your username, one level deeper than older
guides assume:

```
/kaggle/input/datasets/<username>/<dataset-name>/...
```

It also extracted `images.zip` on import, so the images sit in a plain
`images/` folder and training reads them straight from the input. No unzip
step needed.

```python
import os, pandas as pd

META = "/kaggle/input/datasets/akashverma7762/amazon-price-metadata/metadata.csv"
IMG  = "/kaggle/input/datasets/akashverma7762/amazon-price-images-v2/images"

ids  = set(pd.read_csv(META)["sample_id"].astype(str))
have = {f[:-4] for f in os.listdir(IMG) if f.endswith(".jpg")}
print(f"META={META}\nIMG={IMG}\nimages={len(have)} missing={len(ids - have)}")
assert len(ids - have) < 100, "images missing - check the attached datasets"

man = os.path.join(IMG, "_preprocessing.json")
print(open(man).read() if os.path.exists(man) else "no manifest found")
```

**Check:** `images=73517 missing=0` (what the Sep 26 run showed) and
`"resize_mode": "letterbox_pad"` in the manifest.

Don't search `/kaggle/input` with a recursive `glob` or `os.walk`. It walks
the 73k-file image folder on a slow mount: a single pass took 64 s, and the
first version of this cell did three of them and looked hung. If the paths
ever change, find them with `!ls /kaggle/input/datasets/<username>/*/`
instead.

### B3. Smoke test (5 batches)

```python
!python model/train.py --subset-mode full --metadata {META} --image-dir {IMG} \
    --epochs 1 --max-batches 5 --batch-size 64 --num-workers 4 \
    --checkpoint-dir /tmp/smoke --log-path /tmp/smoke/smoke.log
```

**Check** these lines in the output:
- `device=cuda cuda_available=True`
- `split='train': ... usable rows` around 51k, and `split='val'` around 11k.
  `0 usable rows` means `IMG` is wrong.
- a finite `avg_loss`

The smoke test writes to `/tmp`, so it doesn't clutter the saved output.

### B4. Measure before choosing `--epochs`

Run this interactively. It does a 100-batch slice and times it:

```python
!python model/train.py --subset-mode full --metadata {META} --image-dir {IMG} \
    --epochs 1 --max-batches 100 --batch-size 64 --num-workers 4 \
    --checkpoint-dir /tmp/timing --log-path /tmp/timing/t.log --log-every 0
```

Read `elapsed_sec` from the `split=train DONE` and `split=val DONE` lines.
A full epoch is 804 train batches (51,461 rows ÷ 64, last partial batch
dropped) and 173 val batches (11,028 ÷ 64, rounded up). So:

```
seconds per epoch  ≈ (train elapsed / 100) × 804  +  (val elapsed / 100) × 173
epochs that fit    ≈ (8 hrs × 3600) / seconds per epoch
```

8 hours rather than 9 leaves room for setup (clone, pretrained weights)
and the final cell.

**Measured on Sep 26 (T4, `--num-workers 4`, images read from
`/kaggle/input`):** train 105.8 s and val 30.4 s per 100 batches, so about
850 + 53 ≈ 900 s (15 min) per epoch, about 3.8 hours for 15 epochs. Roughly
1 s per training batch is slow for ConvNeXt-Tiny on a T4, which points at
data loading (JPEG decode + augmentation on 4 CPUs, reading from the input
mount) rather than the GPU. That's fast enough for 15 epochs, so it's left
alone for now.

If 15 epochs fit, use 15. If not, use what fits and continue in a second
session with step B7. A committed run that hits the session limit is the one
really expensive mistake in this workflow.

Also watch the GPU usage bar in the editor's right-hand panel during the
timing run. If it stays low, the 4 CPUs decoding and augmenting JPEGs are the bottleneck.
That's normal on Kaggle. It means a faster GPU wouldn't help, not that
something is broken.

### B5. The real run

```python
!python model/train.py --subset-mode full --metadata {META} --image-dir {IMG} \
    --epochs 15 --batch-size 64 --lr 1e-4 --num-workers 4 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --log-path /kaggle/working/logs/train.log
```

Set `--epochs` to what B4 said fits. Then **delete or comment out the B4
cell**, since the commit would run it again for nothing. Then **Save Version
→ Save & Run All (Commit)**.

What the script does during the run:
- saves `epoch_N.pt` every epoch, plus `best.pt` for the lowest val loss
- rewrites `history.json` after every epoch, so a crash still leaves a loss
  curve
- logs batch loss every 50 batches (`--log-every`), so the output stays
  readable

`--epochs 15 --batch-size 64 --lr 1e-4` are **starting points, not tuned
values**. Step B9 is where you judge them.

### B6. Trim the output and plot the curve

Add this as the last cell, so it runs inside the commit:

```python
import json, os, glob, re
import matplotlib.pyplot as plt

ck = "/kaggle/working/checkpoints"
h = json.load(open(f"{ck}/history.json"))

# Each checkpoint is roughly 330 MB (weights + AdamW state). Keep best + last only.
epochs = sorted(int(re.search(r"epoch_(\d+)", p).group(1)) for p in glob.glob(f"{ck}/epoch_*.pt"))
for e in epochs[:-1]:
    os.remove(f"{ck}/epoch_{e}.pt")

plt.plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label="train")
plt.plot([r["epoch"] for r in h], [r["val_loss"] for r in h], label="val")
plt.xlabel("epoch"); plt.ylabel("SmoothL1 loss"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("/kaggle/working/loss_curve.png", dpi=150, bbox_inches="tight")
print(json.dumps(h, indent=1))
```

The newest `epoch_N.pt` is kept so B7 can resume from it.

**Check:** the committed version's Output has `checkpoints/best.pt`,
`checkpoints/epoch_<last>.pt`, `checkpoints/history.json`,
`logs/train.log` and `loss_curve.png`.

### B7. If you need more epochs, resume

Make the finished run's output into a Dataset (Output → New Dataset, e.g.
`amazon-price-run1`). Make a copy of Notebook B, attach that dataset, and
swap the B5 cell for:

```python
PREV = "/kaggle/input/datasets/akashverma7762/amazon-price-run1/checkpoints"
print(sorted(os.listdir(PREV)))   # confirm the epoch_N.pt you're resuming from
!python model/train.py --subset-mode full --metadata {META} --image-dir {IMG} \
    --epochs 15 --batch-size 64 --lr 1e-4 --num-workers 4 \
    --resume-from {PREV}/epoch_7.pt \
    --checkpoint-dir /kaggle/working/checkpoints \
    --log-path /kaggle/working/logs/train.log
```

- `--epochs` is the **final epoch number**, not how many more to run. From
  epoch 7 with `--epochs 15`, it runs 8–15. The script refuses to start if
  the checkpoint is already past `--epochs`.
- It restores the weights, optimizer state, loss history and best-val-loss,
  so `best.pt` still means best across both sessions.
- Keep `--batch-size`, `--lr` and pretrained the same. The script warns if
  they differ, because it's no longer one experiment.
- It does **not** restore the shuffle order or RNG state. Mention that in
  the report rather than claiming the run continued exactly.

### B8. Save the results

From the final version's Output, download `checkpoints/best.pt`,
`checkpoints/history.json`, `logs/train.log` and `loss_curve.png`. The last
three are what your report and viva use. Also make the output a Dataset, so
Phase 3 can attach `best.pt` without re-uploading it.

**Check:** `best.pt` is on your machine **and** in a Kaggle Dataset.

### B9. Read the curve and decide

This is roadmap Phase 2, step 4.

**First, the bar to beat.** A "model" that ignores the image and always
predicts the median train price ($14.00) scores **val SmoothL1 = 14.03**
(MAE $14.52). The model has to get clearly below that to have learned
anything from the images. The 100-batch timing run on Sep 26 already reached
12.86 on part of the val set, so it's starting to move. Where the full run
ends up relative to 14.03 is the number worth putting in the report.

Reading the loss: with `beta=1`, SmoothL1 is roughly MAE − 0.5 once errors
are above $1, so a val loss of 10 means the typical prediction is about $10.50
off.

| What you see | What it means | What to do next |
|---|---|---|
| val loss still falling at the last epoch | undertrained | more epochs via B7 |
| val loss bottoms out, then rises | overfitting | `best.pt` already holds the best epoch. Try stronger augmentation or fewer epochs. |
| val loss flat from epoch 1 | not learning | check `--lr`. Re-read the B3 output. |
| train loss far below val loss, and both flat | memorising | more augmentation or regularisation |

For tuning, change **one** thing per run and keep each run's
`history.json` as its own Dataset, so comparisons are real. Your weekly GPU
quota limits how many runs you can do, so use them on what the curve points
to.

**Phase 2 is done when** you can say, from the curve, why the checkpoint you
kept is the one you kept. That's where Phase 3 starts, and it's a likely
viva question.

---

## Known gaps

- No evaluation on the held-out `test` split yet. That's Phase 3. Don't
  look at test metrics while tuning in B9.
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
