# Where each part of the report comes from

A map from the usual chapters of a project report to the files in this
repository that hold the material, the figures and the numbers. Every
number here comes from a logged run.

## Report chapters

| Chapter | Source in the repo | Figures and numbers to use |
|---|---|---|
| **1. Introduction** | README: "Problem statement" | the problem, the goal, the output (a price and an 80% range) |
| **2. Background** | not in the repo: your own reading | topics to cover: transfer learning, CNNs for regression, ConvNeXt, prediction intervals (split-conformal calibration), PWAs |
| **3. Dataset** | [`data/README.md`](../../data/README.md), [`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb) | 75,000 raw rows → 73,517 after cleaning; $1.32–$145.25 after 1st/99th percentile clipping; 51,461 / 11,028 / 11,028, stratified by 10 price bins |
| **4. Method** | [`docs/experiments.md`](../experiments.md), `model/train.py`, `model/preprocessing.py` | ConvNeXt-Tiny (ImageNet-22k), linear head, SmoothL1 on raw price, AdamW lr 1e-4, batch 64, 15 epochs, 224×224 letterbox on white; augmentation: random resized crop (0.8–1.0), horizontal flip, colour jitter; Kaggle T4, 3.79 h. Loss curves: [`run1_loss_curve.png`](../experiments/run1_loss_curve.png), [`run2_loss_curve.png`](../experiments/run2_loss_curve.png) |
| **5. Experiment** | [`docs/experiments.md`](../experiments.md), [`run2_vs_run1_val.md`](../experiments/run2_vs_run1_val.md) | run2 trained on log(price): val MAE $11.50 vs $11.23, a difference of +$0.27 (95% interval +$0.13 to +$0.40), so run1 was kept by the rule set before the run |
| **6. Evaluation** | [`docs/evaluation.md`](../evaluation.md) | test MAE $11.27 (95% CI $10.99–$11.56) vs $14.51 baseline; median error $5.74 vs $8.71; SMAPE 54.0% vs 70.5%; range coverage 79.6%. Figure: [`calibration.png`](../phase3/calibration.png) |
| **7. Error analysis** | [`docs/evaluation.md`](../evaluation.md), [`docs/phase3/error_analysis.md`](../phase3/error_analysis.md) | hedging (under $5: +$5.67, over $50: −$40.33); pack size (−4.5%); one brand at $21.25 MAE; photos of the worst misses: [`worst_under.jpg`](../phase3/worst_under.jpg), [`worst_over.jpg`](../phase3/worst_over.jpg) |
| **8. System design** | README: "Architecture"; [`backend/README.md`](../../backend/README.md), [`frontend/README.md`](../../frontend/README.md) | the architecture diagram (Mermaid in the README); ONNX export matching PyTorch within $0.0002 on 256 test images |
| **9. Testing** | [`docs/testing.md`](../testing.md) | 57 backend tests (23 with the real model), 37 end-to-end tests in 3 browsers, 0 accessibility issues, 7 bugs found and fixed |
| **10. Deployment** | [`deploy/README.md`](../../deploy/README.md) | Render + Vercel, free plans; the memory crash and its fix; about 1 s per photo on the live server |
| **11. Results and discussion** | README: "Features", "Limitations" | screenshots in [`docs/images/`](../images/) |
| **12. Conclusion and future work** | [`viva-qa.md`](viva-qa.md), last questions | text plus photo, rotation augmentation, a "not a product" check, paid hosting |

## Slides

| Slide | Source |
|---|---|
| 1. Cover | the live link |
| 2. The problem | README: "Problem statement" |
| 3. How it was built | README: "Solution overview" |
| 4. Results | `docs/evaluation.md` |
| 5. Live demo | `docs/images/desktop-result.png`, `phone-result.png`; script in [`demo-script.md`](demo-script.md) |
| 6. Testing and deployment | `docs/testing.md`, `deploy/README.md` |
| 7. Limits and future work | README: "Limitations" |
| 8. Thank you | the live link and the repository |
