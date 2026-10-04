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

*Pending: filled in from Notebook C's output.*
