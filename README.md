# Amazon Vision-Based Price Prediction System

Estimates the price of a product from a photograph of it, using a fine-tuned
vision model served through an API and an installable web app.

> **Status: in development — Phases 0–2 of 10 complete.**
> A first model is trained. On the validation split its average error is
> **$11.23 per product**, against **$14.52** for always guessing the median
> price. The held-out test split hasn't been used yet; the chosen model is
> scored on it once, in Phase 3. See [Current status](#current-status) for
> exactly what does and does not work today.

---

## Problem

Pricing a product from its image alone is a hard regression problem: visual
features correlate with price only loosely, and the relationship differs
wildly across product categories. This project tests how far a fine-tuned
pretrained vision backbone can get on that task, and packages the result as
a working app rather than a notebook.

## Approach

A pretrained ConvNeXt-Tiny backbone with its classifier replaced by a
single-output regression head, fine-tuned on ~51k Amazon product images
paired with their prices, then exported for inference behind a FastAPI
service and a Next.js PWA.

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Model framework | PyTorch 2.14 |
| Vision backbone | `timm` 1.0.29 — `convnext_tiny.fb_in22k_ft_in1k` |
| Training compute | Kaggle Notebooks (T4 GPU) |
| Model export | TorchScript or ONNX *(Phase 3)* |
| Backend API | FastAPI *(Phase 4)* |
| Frontend | React / Next.js, installable PWA *(Phase 5)* |
| Containerization | Docker *(Phase 7)* |
| Version control | Git + GitHub, Git LFS for weights |

## Dataset

[Amazon ML Challenge 2021](https://www.kaggle.com/datasets/suvroo/amazon-ml)
(`suvroo/amazon-ml`), using `train.csv` only — the dataset provides image
**URLs**, not bundled image files, so images are fetched and cached locally
by `scripts/download_images.py`.

After cleaning (deduplication, missing-value removal, 1st/99th percentile
outlier clipping):

| | |
|---|---|
| Rows | 73,517 (from 75,000 raw) |
| Price range | $1.32 – $145.25 USD |
| Split | 51,461 train / 11,028 val / 11,028 test |
| Stratification | 10 quantile price bins |

Raw and processed data are **not** committed — see [`data/README.md`](data/README.md)
for provenance and regeneration steps.

## Repository structure

```
├── data/          # dataset docs; actual data is gitignored
├── notebooks/     # EDA, and the Kaggle training notebook
├── scripts/       # data cleaning, splitting, image download
├── model/         # model, dataset, preprocessing, training, evaluation
├── backend/       # FastAPI service          (Phase 4)
├── frontend/      # Next.js PWA              (Phase 5)
├── deploy/        # Docker + deploy config   (Phase 7)
└── docs/          # Kaggle guide and the experiment log
```

## Current status

**Working and verified:**

- **Data pipeline** — cleaning, stratified splitting, and EDA over 73,517 rows
- **Image pipeline** — resumable, crash-safe download and letterbox resize to
  224×224. All 73,517 images downloaded and checked on Kaggle, none missing.
- **Training pipeline** — Dataset/DataLoader with train-only augmentation,
  ConvNeXt-Tiny + regression head, SmoothL1 loss, per-epoch checkpoints that
  can't be corrupted by a crash, resume after a Kaggle timeout
  (`--resume-from`), and a time budget that stops cleanly before the session
  limit (`--max-hours`).
- **Trained model (run1)** — 15 epochs on a Kaggle T4 GPU, about 15 minutes
  per epoch.
  Scored on the 11,028 validation images:

  | | run1 | always guess the median ($14) |
  |---|---|---|
  | Average error (MAE) | **$11.23** | $14.52 |
  | Median error | **$5.71** | $8.65 |
  | SMAPE | 53.5% | 70.6% |

- **Evaluation** — `model/evaluate.py` scores any checkpoint in dollars,
  breaks the error down by price range, writes per-image predictions (worst
  first), and compares two runs with a bootstrap confidence interval.
- **One controlled experiment** — run2 trained on log(price) instead of
  price. It did not beat run1: $11.50 average error, a difference of +$0.27
  (95% interval +$0.13 to +$0.40). The decision rule was written down before
  the run. Both runs are in [`docs/experiments.md`](docs/experiments.md).

**Known limits of the current model:**

- Validation error stops improving after 2–4 epochs while training error
  keeps falling: the model memorises the training images.
- Errors are largest at the extremes. For items under $5 the average error
  ($5.89) is bigger than the price itself, and for items over $50 it is
  $42.15. The model hedges toward typical prices, most likely because much of
  what sets a price (brand, pack size, quantity) is hard to see in a photo.

**Not done yet:**

- No score on the held-out test split. That happens once, for run1, in
  Phase 3.
- No model export, backend, frontend, or deployment (Phases 3–7)

## Setup

```bash
git clone https://github.com/Akash7762/Amazon-Vision-Based-Price-Predictor.git
cd Amazon-Vision-Based-Price-Predictor

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` pins the **CPU-only** PyTorch build, which is what the
> local development machine uses. Do not install it on a GPU host such as
> Kaggle — those images already ship a CUDA-enabled torch.

## Usage

Reproduce the data pipeline (requires `data/raw/train.csv` from Kaggle):

```bash
python scripts/data_cleaning.py      # -> data/processed/metadata.csv
python scripts/data_split.py         # adds the train/val/test split column
python scripts/make_dev_subset.py    # -> a small stratified dev subset
```

Cache the images locally:

```bash
python scripts/download_images.py \
    --metadata data/processed/dev_subset.csv \
    --out-dir data/processed/images/dev \
    --image-size 224 --workers 16
```

Run a quick pipeline sanity check (a few batches, CPU):

```bash
python model/train.py --subset-mode dev --epochs 1 --max-batches 5 \
    --batch-size 8 --checkpoint-dir model/checkpoints/sanity
```

Score a trained checkpoint in dollars (here on the dev subset; the real runs
use the full metadata and images on Kaggle):

```bash
python model/evaluate.py --checkpoint model/checkpoints/sanity/best.pt \
    --name sanity --metadata data/processed/dev_subset.csv \
    --image-dir data/processed/images/dev --split val --out-dir model/logs/eval
```

The real training runs happen on a Kaggle GPU with
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb); its first cell
holds the run's settings. Step-by-step instructions are in
[`docs/kaggle_training_guide.md`](docs/kaggle_training_guide.md).

## Roadmap

- [x] **Phase 0** — Repository and environment setup
- [x] **Phase 1** — Data collection and preparation
- [x] **Phase 2** — Model design, training, and one controlled experiment
- [ ] **Phase 3** — Model evaluation, error analysis, export
- [ ] **Phase 4** — FastAPI backend
- [ ] **Phase 5** — Next.js PWA frontend
- [ ] **Phase 6** — Integration and end-to-end testing
- [ ] **Phase 7** — Deployment
- [ ] **Phase 8** — Documentation and repo polish
- [ ] **Phase 9** — Final presentation

## Notes

Any performance figure quoted in this repository comes from a run that was
actually executed and logged. No benchmark numbers are estimated or claimed
ahead of the run that produces them.

This README will be replaced by full documentation — architecture diagram,
screenshots, demo, installation and usage — in Phase 8.

## License

See [LICENSE](LICENSE).
