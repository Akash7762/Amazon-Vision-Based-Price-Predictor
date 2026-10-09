# Backend API (Phase 4)

A FastAPI service that prices a product from a photo. It serves the
`v0.1-model` ONNX model with `onnxruntime`. **PyTorch isn't needed**: the
server depends on six packages ([`requirements.txt`](requirements.txt)).

```
POST /predict   (a product photo)  ->  {"price": 8.57, "currency": "USD",
                                         "range": {"low": 3.55, "high": 25.93, "coverage": 0.8},
                                         "model_version": "v0.1-model"}
```

## Setup

Use **Python 3.12**, the version the project uses. On the development PC the
default `python` is a 3.15 pre-release, for which `onnxruntime` has no build
yet, so installing the requirements fails there. On Windows, `py -3.12`
picks the right one.

```bash
py -3.12 -m venv venv                              # once; skip if the project venv exists
venv\Scripts\activate                              # macOS/Linux: source venv/bin/activate
pip install -r backend/requirements-dev.txt        # runtime + test packages
python backend/download_model.py                   # the model, from the GitHub release
```

`download_model.py` saves `price_model.onnx` (112 MB) and `price_model.json`
into `backend/models/` (ignored by git) and checks the file's sha256 against
the one recorded when the model was verified. If GitHub's download server is
unreachable from your network (some networks reset connections to it), get
both files from the
[release page](https://github.com/Akash7762/Amazon-Vision-Based-Price-Predictor/releases/tag/v0.1-model)
in a browser and install them with:

```bash
python backend/download_model.py --from-file path/to/price_model.onnx
```

## Run

From the **repo root** (the API imports the shared image code in `model/`):

```bash
uvicorn backend.app.main:app --reload --timeout-keep-alive 75
```

`--timeout-keep-alive 75` keeps idle connections open longer than anything
in front of the API does: Next.js (Node.js) drops them after 5 s, cloud load
balancers usually after 60 s. With uvicorn's default of 5 s both ends closed
at the same moment, and now and then the web app sent a request down a
connection that was just being closed: 8 of 432 requests failed with a 500
in a Phase 6 stress test, none with 75 s.

Then open http://localhost:8000/docs to try it in the browser: expand
`POST /predict`, click **Try it out**, choose a photo, **Execute**.

From the command line:

```bash
curl -F "file=@photo.jpg" http://localhost:8000/predict
```

## Endpoints

### `GET /health`

```json
{"status": "ok", "model_version": "v0.1-model",
 "model_sha256": "b7055048f0f059789f740e121c7b739c67b2269845c2e90814273f6cccebf984",
 "input_size": 224, "range_coverage": 0.8}
```

### `POST /predict`

A multipart upload with one field, `file`: a JPEG, PNG or WebP photo.

| Field | Meaning |
|---|---|
| `price` | the estimate, in US dollars |
| `range.low`, `range.high` | the price range (see below) |
| `range.coverage` | 0.8: about 8 in 10 products with a similar estimate were priced inside such a range |
| `model_version` | which model answered |

Errors come back as `{"detail": "..."}` with a message meant for people:

| Status | When |
|---|---|
| 400 | empty file, or not a readable image (corrupt, cut off) |
| 413 | file over 10 MB, or more than 50 million pixels |
| 415 | not a JPEG, PNG or WebP, by declared type or by content |
| 422 | no `file` field in the request |

## How the price range works

The model gives one number, so the range comes from how wrong that number
was on the 11,028 validation images. They are split into 8 equal groups by
predicted price; in each group we keep the 10th and 90th percentile of
*actual ÷ predicted*. A new prediction gets its group's two ratios:

| Predicted price | Range | Coverage on the test set |
|---|---|---|
| $1.32–5.58 | ×0.57 to ×5.62 | 80.9% |
| $5.58–8.02 | ×0.48 to ×4.16 | 80.5% |
| $8.02–10.63 | ×0.41 to ×3.03 | 77.3% |
| $10.63–13.53 | ×0.42 to ×2.75 | 80.8% |
| $13.53–17.07 | ×0.37 to ×2.60 | 80.8% |
| $17.07–22.08 | ×0.36 to ×2.28 | 78.0% |
| $22.08–31.83 | ×0.30 to ×2.04 | 80.2% |
| $31.84–105.06 | ×0.34 to ×1.84 | 78.2% |

Fitted on validation only, the ranges then held **79.6%** of real test prices
against an 80% target, so the stated coverage is honest. They are also wide
(median width $31): a prediction of $10 comes with roughly $4–$28. That's
what a photo-only estimate can honestly claim.

Within each group, the median of actual ÷ predicted is between 0.95 and 1.16.
Given what the model says, the real price is centred on it; the "pull toward
the middle" seen in Phase 3 is the model being cautious when a photo is
ambiguous, not a bias in its answers.

The numbers live in [`calibration.json`](calibration.json), made by
`python model/calibrate_interval.py` (see its docstring). The same code
(`model/price_range.py`) fitted them, checked them on test, and serves them,
so the API returns ranges exactly as they were checked.

## Settings

Environment variables, all optional:

| Variable | Default |
|---|---|
| `MODEL_PATH` | `backend/models/price_model.onnx` |
| `MODEL_CARD_PATH` | the model path with `.json` |
| `CALIBRATION_PATH` | `backend/calibration.json` |
| `MODEL_VERSION` | `v0.1-model` |
| `MAX_UPLOAD_MB` | `10` |
| `MAX_IMAGE_PIXELS` | `50000000` |
| `ALLOWED_ORIGINS` | `http://localhost:3000` (comma-separated; for the Phase 5 frontend) |

## Tests

```bash
python -m pytest
```

54 tests, about 6 seconds.

- **31 use a tiny stand-in model** with the real model's inputs and outputs
  (`tests/conftest.py`), so they run without the 112 MB file. They cover the
  answers, every rejection, CORS, the startup checks, image handling and the
  price range. Each of the subtlest ones (transparent PNGs, phone-photo
  rotation, CORS on error responses, multi-picture JPEGs, 16-bit greyscale)
  was checked by breaking the behaviour and watching the test fail.
- **23 use the real model** (`tests/test_real_model.py`, skipped until it's
  downloaded): three test photos get exactly the answers the Phase 3
  reference code gives, the same pixels get the same price in every format,
  and 15 unusual images all get a valid answer. See
  [`docs/testing.md`](../docs/testing.md), which also covers the end-to-end
  tests.

## Design notes

- **One image pipeline.** `/predict` calls `model/predict_onnx.py`, the code
  verified in Phase 3 to reproduce PyTorch's prices: turn the photo upright
  (EXIF), letterbox it to 224×224 on white (`model/preprocessing.py`), pass
  the raw pixels. Normalisation is inside the ONNX graph.
- **Refuses to start on the wrong model.** At startup the model's sha256 must
  match its model card; a missing or damaged file stops the app with a
  message saying how to fix it, instead of serving wrong prices.
- **Prices stay in the training range**, $1.32–$145.25. The model can output
  a negative or tiny number for an unusual photo; it never saw prices outside
  that range.
- **Transparent PNGs get a white background**, like the catalogue photos the
  model learned from (a plain RGB conversion would make them black).
  16-bit greyscale is scaled down to 8 bits (a plain conversion clips it to
  white), and JPEGs carrying extra images (MPO) are read as their main photo.
- **Large uploads are refused early**: by `Content-Length` before the body
  is read, and by pixel count before an image is decoded.
- **Inference runs in a worker thread** (`/predict` is a plain `def`), so one
  slow request doesn't block others. On the development PC (CPU): a median
  95 ms per photo through the web app, one at a time (Phase 6); about 2
  requests a second with 8 at once (Phase 4).
- **`/health` doesn't queue behind them** (it's `async`): with 48 uploads
  running it answers in a median 44 ms, against 592 ms when it shared the
  worker threads.

## Known limits

- Any photo gets a price, even one that isn't a product. Nothing detects
  "this isn't something for sale".
- HEIC (the iPhone default) isn't accepted; browsers usually convert to JPEG
  when uploading.
- No authentication or rate limiting yet. Those belong with deployment
  (Phase 7).
