# run2 vs run1 on val

Output of `model/evaluate.py` from the run2 notebook (Kaggle T4, Cells 4 and
7), copied as printed. Both checkpoints scored by the same script on the same
11,028 val images.

## run2 on its own

=== run2 on val: 11028 images, epoch 5, target=log, 63s ===

| | run2 | median baseline ($14.00) |
|---|---|---|
| MAE ($) | 11.498 | 14.521 |
| RMSE ($) | 19.675 | 23.727 |
| Median abs error ($) | 5.963 | 8.647 |
| SMAPE (%) | 54.915 | 70.561 |
| Bias ($) | -4.966 | -7.762 |

| Price range | n | MAE ($) | baseline MAE | SMAPE (%) | baseline SMAPE | bias ($) |
|---|---|---|---|---|---|---|
| $0-5 | 1822 | 5.42 | 10.68 | 72.6 | 124.5 | +5.29 |
| $5-10 | 2334 | 4.85 | 6.51 | 43.8 | 61.9 | +3.74 |
| $10-20 | 2928 | 5.94 | 2.52 | 38.5 | 17.3 | +0.68 |
| $20-50 | 2841 | 14.04 | 17.57 | 57.7 | 73.4 | -10.05 |
| $50+ | 1103 | 43.81 | 61.82 | 85.6 | 134.1 | -42.24 |

## Comparison on val (11028 images)

| | median baseline | run1 (target=price, epoch 14) | run2 (target=log, epoch 5) |
|---|---|---|---|
| MAE ($) | 14.521 | 11.233 | 11.498 |
| RMSE ($) | 23.727 | 19.280 | 19.675 |
| Median abs error ($) | 8.647 | 5.713 | 5.963 |
| SMAPE (%) | 70.561 | 53.513 | 54.915 |
| Bias ($) | -7.762 | -4.345 | -4.966 |

| Price range | n | baseline MAE | run1 MAE | run2 MAE | run1 SMAPE | run2 SMAPE |
|---|---|---|---|---|---|---|
| $0-5 | 1822 | 10.68 | 5.89 | 5.42 | 73.3 | 72.6 |
| $5-10 | 2334 | 6.51 | 5.06 | 4.85 | 44.3 | 43.8 |
| $10-20 | 2928 | 2.52 | 6.01 | 5.94 | 38.0 | 38.5 |
| $20-50 | 2841 | 17.57 | 13.11 | 14.04 | 53.4 | 57.7 |
| $50+ | 1103 | 61.82 | 42.15 | 43.81 | 82.1 | 85.6 |

**run2 minus run1** (negative = run2 better), paired bootstrap over images, 2000 resamples:

| Metric | difference | 95% interval | reading |
|---|---|---|---|
| MAE ($) | +0.266 | [+0.134, +0.402] | run1 lower, interval excludes 0 |
| SMAPE (%) | +1.402 | [+0.834, +2.012] | run1 lower, interval excludes 0 |

The interval covers which images happened to land in the split. It doesn't
cover run-to-run randomness from training (seed, shuffle order).
