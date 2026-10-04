# Amazon Vision-Based Price Prediction System

Estimates the price of a product from a photograph of it, using a fine-tuned
vision model served through an API and an installable web app.

> **Status: in development — Phases 0–5 of 10 complete.**
> Take or choose a product photo in the web app, and it shows an estimated
> price with a calibrated range. On 11,028 held-out test images the model's
> average error is **$11.27 per product**, against **$14.51** for always
> guessing the median price (22% lower). See [Current status](#current-status)
> for exactly what does and does not work today.

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
| Model export | ONNX, run with `onnxruntime` (no PyTorch needed to serve) |
| Backend API | FastAPI + `onnxruntime` (no PyTorch) |
| Frontend | Next.js 16 (React 19), installable PWA |
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
├── notebooks/     # EDA, and the Kaggle training and evaluation notebooks
├── scripts/       # data cleaning, splitting, image download
├── model/         # training, evaluation, error analysis, ONNX export
├── backend/       # FastAPI service: /predict, /health
├── frontend/      # Next.js web app (installable PWA)
├── deploy/        # Docker + deploy config   (Phase 7)
└── docs/          # Kaggle guide, experiment log, evaluation report
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
- **Trained model (run1)** — 15 epochs on a Kaggle T4 GPU, 3.8 hours
  (15.1 minutes per epoch). Scored once on the 11,028 held-out test images,
  never used for any decision:

  | | run1, test | always guess the median ($14) |
  |---|---|---|
  | Average error (MAE) | **$11.27** (95% interval $10.99–$11.56) | $14.51 |
  | Median error | **$5.74** | $8.71 |
  | SMAPE | 54.0% | 70.5% |

  The validation estimate was $11.23, so the test confirms it. Removing every
  test item whose photo also appears in training gives $11.23, so the score
  isn't inflated by duplicates.
- **Evaluation and error analysis** — `model/evaluate.py` scores any
  checkpoint in dollars with a bootstrap interval and compares runs;
  `model/error_analysis.py` plots predicted vs actual, shows the worst
  predictions with their photos, and runs a pack-size check. Full results in
  [`docs/evaluation.md`](docs/evaluation.md).
- **One controlled experiment** — run2 trained on log(price) instead of
  price. It did not beat run1: $11.50 average error on validation, a
  difference of +$0.27 (95% interval +$0.13 to +$0.40). The decision rule was
  written down before the run. Both runs are in
  [`docs/experiments.md`](docs/experiments.md).
- **Exported model** — `price_model.onnx` (112 MB), with normalisation and
  the dollar conversion inside the graph. It matches PyTorch to within
  $0.0002 on 256 test images and prices a photo in about 0.1 s on the
  development PC's CPU, without PyTorch (`model/predict_onnx.py`).
  Download it from the
  [`v0.1-model` release](https://github.com/Akash7762/Amazon-Vision-Based-Price-Predictor/releases/tag/v0.1-model).
- **Backend API** — FastAPI service in [`backend/`](backend/README.md).
  `POST /predict` takes a photo and returns the price with a range that held
  79.6% of real test prices (80% target); `GET /health` reports the model.
  It checks the model's sha256 at startup, rejects bad uploads with clear
  errors (400/413/415), and has 27 tests. About 0.3 s per request on the
  development PC's CPU.
- **Web app** — Next.js PWA in [`frontend/`](frontend/README.md). Take a
  photo (phones) or choose, drop or paste one (desktop); it shows the price,
  the range and a plain explanation, with clear messages when something goes
  wrong. Installable from Chrome/Edge; after one visit it opens offline
  (prices always come from the live model). Checked against the real model in
  desktop and phone-sized browsers.

**Known limits of the current model:**

- **It hedges toward typical prices.** It gets the order right (pricier
  items get higher guesses), but its guesses move only about half as far as
  real prices: items under $5 are guessed $5.67 too high on average, items
  over $50 $40.33 too low.
- **It can't see what the photo doesn't show.** The worst misses are bulk
  and case quantities photographed as a single unit, and one tea brand
  (4% of products) whose variants differ only in small label text, which the
  model mostly prices at about $45 whatever the variant. Products sold in packs are guessed
  4.5% lower than single items at the same price.
- **Some labels look wrong.** Several of the worst "over-predictions" are
  multipack listings priced like a single unit (a 24-pack of water at $1.79).
- Validation error stops improving after 2–4 epochs while training error
  keeps falling: the model memorises the training images.

**Not done yet:**

- Installing on a phone: needs HTTPS, which deployment (Phase 7) provides
- End-to-end tests and deployment (Phases 6–7)

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

Run the app: the API first (after `python backend/download_model.py`; full
steps in [`backend/README.md`](backend/README.md)), then the web app (steps in
[`frontend/README.md`](frontend/README.md)), and open http://localhost:3000:

```bash
uvicorn backend.app.main:app --reload       # repo root, Python venv
cd frontend && npm install && npm run dev   # another terminal
```

The API's own interactive docs are at http://localhost:8000/docs.

Or price photos directly with the exported model. This needs only
`onnxruntime`, `numpy` and `Pillow`, not PyTorch:

```bash
python model/predict_onnx.py --model price_model.onnx photo1.jpg photo2.jpg
```

The real training runs happen on a Kaggle GPU with
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) (its first cell
holds the run's settings), and the test scoring, error analysis and export
with [`notebooks/kaggle_phase3.ipynb`](notebooks/kaggle_phase3.ipynb).
Step-by-step instructions are in
[`docs/kaggle_training_guide.md`](docs/kaggle_training_guide.md).

## Roadmap

- [x] **Phase 0** — Repository and environment setup
- [x] **Phase 1** — Data collection and preparation
- [x] **Phase 2** — Model design, training, and one controlled experiment
- [x] **Phase 3** — Model evaluation, error analysis, export
- [x] **Phase 4** — FastAPI backend
- [x] **Phase 5** — Next.js PWA frontend (installing on a phone waits for HTTPS in Phase 7)
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
