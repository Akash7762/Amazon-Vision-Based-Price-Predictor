# Phase 3 — Evaluation, error analysis, export

The model chosen in Phase 2 is **run1** (`best.pt`, epoch 14, raw-price
target): val MAE $11.23, against $14.52 for always guessing the median price.
See [`experiments.md`](experiments.md) for how it was chosen.

Phase 3 runs in Kaggle Notebook C ([`notebooks/kaggle_phase3.ipynb`](../notebooks/kaggle_phase3.ipynb),
steps in the [Kaggle guide](kaggle_training_guide.md#notebook-c--phase-3-test-score-error-analysis-export-cpu)).

## Plan, written before the test run

*Left as written. Results go under "Results" below.*

**1. The test score.** run1 is scored once on the 11,028 test images, which
no run has been scored on. Validation was used for choosing (the epoch, and
run1 over run2), so the val numbers are slightly flattered by that choice;
the test number is the one to report. It comes with a 95% bootstrap interval
over images, and next to the median-price baseline on the same images. No
model gets changed or re-chosen because of it. If it disappoints, that goes
in the report as it is.

**2. Error analysis.**

- Predicted vs actual price across price levels, to see whether run1 pulls
  its guesses toward the middle, as run2's breakdown suggested.
- The 12 items priced furthest too low and the 12 priced most too high by
  ratio, with their photos, to name what goes wrong.
- **The pack-size check.** The working explanation is that much of what sets
  a price (pack size, quantity, brand) isn't visible in a photo. Pack size is
  testable: about a third of products mention a pack size ("Pack of 12",
  "6-Pack"; 3,582 of the test images), and the photo often shows one unit. The check compares multipacks with single items
  *within the same price range* and combines the ranges, with a 95%
  interval. How it will be read:
  - interval entirely below 0: multipacks are guessed low compared with
    single items at the same price, consistent with the photo missing the
    pack size.
  - interval includes 0: no clear difference; the check doesn't support the
    explanation.
  - interval entirely above 0: the opposite of the explanation.

**3. Export.** run1 is exported to ONNX (`price_model.onnx`), with the
ImageNet normalisation and the conversion back to dollars inside the graph,
so the backend only has to letterbox a photo and pass raw pixels. It counts
as done only if:

- on 256 test images, ONNX and PyTorch prices differ by at most $0.01, and
- `model/predict_onnx.py`, which uses only onnxruntime, numpy and Pillow,
  reproduces Cell 4's prices for 8 test photos.

Why ONNX rather than TorchScript: onnxruntime runs the model without
PyTorch, so the Phase 4 backend stays small, and TorchScript is in
maintenance mode in current PyTorch.

**Already checked locally** (CPU, the dev subset, a barely trained test
checkpoint, so only the code was being tested, not the model): both
exporters produce a single ~112 MB file, ONNX matches PyTorch to within
$0.00002, the log-price path includes its `exp()`, and `predict_onnx.py`
runs without importing PyTorch.

## Results

From Notebook C on Kaggle (CPU), Oct 4 2026. Scoring the 11,028 test images
took 27 minutes.

### 1. The test score

**Test MAE $11.27** (95% interval $10.99 to $11.56), against $14.51 for
always guessing the median price on the same images: **22% lower**. The
median error is $5.74 against $8.71 (34% lower).

| | median baseline (test) | run1, val (used for choosing) | **run1, test** |
|---|---|---|---|
| MAE ($) | 14.508 | 11.233 | **11.268** |
| RMSE ($) | 23.657 | 19.280 | 19.161 |
| Median abs error ($) | 8.710 | 5.713 | 5.737 |
| SMAPE (%) | 70.516 | 53.513 | 53.986 |
| Bias ($) | -7.760 | -4.345 | -4.273 |

**Val and test agree.** Test MAE is $0.04 above val, far inside the test
interval. Choosing the epoch and the run on val didn't flatter the val
number by any amount we can measure, so the val results in
[`experiments.md`](experiments.md) were a fair guide.

By price range, on test:

| Price range | n | MAE ($) | baseline MAE | SMAPE (%) | baseline SMAPE | bias ($) |
|---|---|---|---|---|---|---|
| $0-5 | 1841 | 5.83 | 10.67 | 72.8 | 124.4 | +5.67 |
| $5-10 | 2326 | 5.36 | 6.45 | 45.5 | 61.1 | +4.14 |
| $10-20 | 2913 | 6.11 | 2.51 | 39.0 | 17.1 | +0.99 |
| $20-50 | 2844 | 13.16 | 17.61 | 53.5 | 73.3 | -8.99 |
| $50+ | 1104 | 41.53 | 61.54 | 81.1 | 133.9 | -40.33 |

**The model hedges toward typical prices.** Items under $5 are guessed too
high by $5.67 on average, which is 97% of their error. Items over $50 are
guessed too low by $40.33, also 97% of their error. In the $10-20 range,
always guessing $14 beats the model (2.51 vs 6.11), because $14 sits inside
that range; the model gives that back many times over at both ends. This is
the same pattern run2 showed on val, now confirmed for run1 on test.

No prediction was below $0; 8 were below $1. A raw-price head can still
produce a negative price for an unusual photo, so the Phase 4 backend should
clamp its output anyway.

### 2. The pack-size check

**Multipacks are guessed 4.5% lower than single items at the same price
level** (95% interval -7.0% to -1.9%). By the rule written before the run,
the interval is entirely below 0: **consistent with the photo not showing
the pack size.**

It is also **small**. A 4.5% shift is little next to a typical miss of
about 54% (SMAPE). Pack size is one of the things the photo misses, but
not the main reason for the error. The main reason is the hedging above:
from a photo alone, the model can't tell expensive items from mid-priced
ones well enough to commit to a high or low price.

### 3. The export

| | |
|---|---|
| File | `price_model.onnx`, 111.8 MB, one file |
| sha256 | `b7055048f0f059789f740e121c7b739c67b2269845c2e90814273f6cccebf984` |
| Exporter | PyTorch's current (dynamo) exporter |
| ONNX vs PyTorch, 256 test images | max difference **$0.000179**, mean $0.000067: **passed** (limit $0.01) |
| `predict_onnx.py` (no PyTorch) vs Cell 4, 8 test images | all 8 match, largest difference $0.0001: **passed** |

Both acceptance checks set in the plan passed, so `price_model.onnx` is the
model the Phase 4 backend will serve.
