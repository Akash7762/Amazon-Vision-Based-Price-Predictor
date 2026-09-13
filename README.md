# Amazon Vision-Based Price Prediction System

Estimates the price of a product from a photograph of it, using a fine-tuned
vision model served through an API and an installable web app.

> **Status: in development — Phases 0–2 of 10 complete.**
> The data pipeline and the model training pipeline are built and verified.
> **No model has been trained yet** — full-scale training runs on Kaggle GPU
> next. See [Current status](#current-status) for exactly what does and does
> not work today.

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
| Training compute | Kaggle Notebooks (GPU) |
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
├── notebooks/     # exploratory data analysis
├── scripts/       # data cleaning, splitting, image download
├── model/         # model definition, dataset, preprocessing, training
├── backend/       # FastAPI service          (Phase 4)
├── frontend/      # Next.js PWA              (Phase 5)
├── deploy/        # Docker + deploy config   (Phase 7)
└── docs/          # guides
```

## Current status

**Working and verified:**

- **Data pipeline** — cleaning, stratified splitting, and EDA over 73,517 rows
- **Image pipeline** — resumable, crash-safe download and letterbox resize to
  224×224. Verified at 18,214/18,214 images with zero failures.
- **Training pipeline** — Dataset/DataLoader with train-only augmentation,
  ConvNeXt-Tiny + regression head, SmoothL1 loss, per-epoch checkpointing and
  file logging. Runs end-to-end.

**Not done yet:**

- **No trained model.** Local runs were CPU sanity checks over a few dozen
  samples, purely to prove the pipeline executes — the resulting checkpoints
  are not usable models. Full training is the immediate next step.
- No evaluation against the held-out test split (Phase 3)
- No backend, frontend, or deployment (Phases 4–7)

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

For the real training run on GPU, see
[`docs/kaggle_training_guide.md`](docs/kaggle_training_guide.md).

## Roadmap

- [x] **Phase 0** — Repository and environment setup
- [x] **Phase 1** — Data collection and preparation
- [x] **Phase 2** — Model design and training pipeline
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
