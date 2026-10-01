# Experiments

One row per training run. Val numbers only — the test split is not touched
until Phase 3.

Runs are compared on **val MAE in dollars**, scored by `model/evaluate.py`
on the same 11,028 val images. The loss can't be compared across runs once
the target changes (run2's loss is in log-dollars).

| Run | Date | Change from previous | Epochs | best.pt | Val MAE ($) | Val loss | Notes |
|---|---|---|---|---|---|---|---|
| median price | — | always predict $14.00 | — | — | **14.52** | 14.03 | the bar to beat |
| run1 | 2026-10-01 | — (first full run) | 15 | epoch 14, by val loss | pending: scored in run2's notebook | 10.755 | plateaued by epoch ~4, train kept falling |
| run2 | — | target = log(price) | 6 | by val MAE | pending | log space | planned, see below |

---

## run1 — baseline ConvNeXt-Tiny

**Setup:** `convnext_tiny.fb_in22k_ft_in1k` (ImageNet-22k pretrained), single
linear head, SmoothL1 loss on raw price, AdamW lr 1e-4 (constant, no
schedule), batch 64, 15 epochs, 224×224 letterboxed images, augmentation =
RandomResizedCrop(0.8–1.0) + horizontal flip + colour jitter. Kaggle T4,
~15 min/epoch, ~3.8 h total.

**Files:** [`run1_history.json`](experiments/run1_history.json),
[`run1_loss_curve.png`](experiments/run1_loss_curve.png). Checkpoint is
`best.pt` (epoch 14), kept on Kaggle as the `amazon-price-run1` dataset —
too large for git.

![run1 loss curve](experiments/run1_loss_curve.png)

**What the curve shows**

- The model clearly learns from the images: val loss is 15.8% below the
  baseline after epoch 1 and 23.4% below at best (10.755). With SmoothL1 at
  beta=1 that's roughly a typical error of ~$11 against ~$14.50 for the
  baseline — approximate until Phase 3 measures MAE directly.
- **Val stopped improving around epoch 4.** From epoch 4 to 15 it moves
  between 10.75 and 11.17 (mean 10.99, sd 0.12) with no trend.
- **Train loss kept falling** from 12.5 to 4.6. By epoch 15 val loss is 2.4×
  train loss: the extra epochs went into memorising the training set, not
  into anything that carries over to new images. Val didn't get worse,
  though, so `best.pt` is still a sound checkpoint.
- **"Best = epoch 14" is mostly noise.** Epoch 14 (10.755) beats epoch 4
  (10.917) by 0.16, which is about one standard deviation of the
  epoch-to-epoch wobble. Picking the lowest of 15 noisy val scores also
  flatters the number slightly — one reason the final figure has to come
  from the untouched test split.

**What it suggests for the next run** (change one thing at a time):

1. Epochs: 5–6 is enough to compare ideas, ~1.5 h each instead of ~4 h.
2. Regularisation, since the problem is memorisation: stochastic depth
   (`drop_path_rate`), stronger augmentation, or more weight decay.
3. Predict log(price) instead of price. Prices run from $1.32 to $145 with a
   long tail; raw-price loss is dominated by expensive items. Comparing this
   fairly needs val MAE in dollars, not the loss value.
4. A learning-rate schedule (warmup + cosine decay) instead of a constant
   1e-4.

**Limitation worth stating:** price is only partly visible in a photo — two
near-identical items can differ a lot in price (brand, pack size, quantity).
Some of the remaining error is a ceiling of image-only prediction, not just
the model.

---

## run2 — log-price target (planned)

**The one change from run1:** the head predicts log(price) instead of price
(`--target log`, see [`model/targets.py`](../model/targets.py)); dollars are
exp(output). Same backbone, loss function (SmoothL1, beta=1, now on
log-price), optimizer, learning rate, batch size, augmentation and data.

On the log scale an error is a ratio: guessing $4 for a $2 item and $100 for
a $50 item are both "2× too high" and cost the same. On run1's raw scale the
second is a 25 times bigger error ($50 vs $2), so the loss is dominated by
expensive items.

**Two other differences, neither of which favours run2:**

- **6 epochs instead of 15.** run1's val error stopped improving by about
  epoch 4, so epochs 7–15 bought nothing. If anything, fewer epochs make
  this a harder test for run2.
- **best.pt is picked by val MAE in dollars** (`--select-by val_mae`) instead
  of val loss. For run1 those are nearly the same thing (SmoothL1 in dollars
  is about MAE − 0.5); for a log-target run they aren't, and dollars is what
  the runs are compared on.

**Decision rule, written down before the run:**

- Deciding number: val MAE in dollars of each run's `best.pt`, both scored
  by `model/evaluate.py` on the same 11,028 val images (notebook Cells 4
  and 7).
- run2 replaces run1 only if its val MAE is lower **and** the paired
  bootstrap 95% interval for the difference excludes 0. Otherwise run1
  stays.
- SMAPE, RMSE, bias and the per-price-range table are reported alongside
  but don't decide.
- Either way, the result goes in this file.

**Expectation going in (not a result):** log-price should help cheap items
and relative error (SMAPE) most. Dollar MAE could go either way, since
expensive items dominate it. A log model also tends to guess low on
average (bias below 0), because exp of an average log is below the average
price.
