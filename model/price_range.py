"""
A price range to go with each prediction: the "confidence" the API reports.

The model gives one number. To say how far off it might be, we look at how
far off it actually was on the validation images, separately for cheap and
expensive predictions:

1. Sort the validation predictions into equal-sized groups by predicted price.
2. In each group, compute actual / predicted for every image, and keep the
   10th and 90th percentiles (for 80% coverage).
3. For a new photo, find its group and report
   [prediction x 10th percentile, prediction x 90th percentile].

So the range reads: "of the validation products that got a similar estimate,
8 in 10 were priced between these two numbers". It is fitted on validation
only; its coverage is then checked once on test (model/calibrate_interval.py).

Ratios rather than dollar offsets because errors grow with price. Groups
because the spread depends on the prediction: low guesses are relatively
less certain than high ones.

Both the calibration script and the backend use these functions, so the range
the API returns is computed exactly the way it was checked.
"""
import numpy as np


def clamp(price, floor, ceiling):
    """Keep a price inside the range the model was trained on."""
    return float(min(max(price, floor), ceiling))


def fit(preds, prices, floor, ceiling, n_bins=8, coverage=0.8):
    """Fit the grouped ratio ranges on (prediction, actual price) pairs."""
    preds = np.clip(np.asarray(preds, dtype=float), floor, ceiling)
    prices = np.asarray(prices, dtype=float)
    edges = np.quantile(preds, np.linspace(0, 1, n_bins + 1))[1:-1]
    groups = np.searchsorted(edges, preds, side="right")
    tail = (1.0 - coverage) / 2.0

    bins = []
    for b in range(n_bins):
        in_bin = groups == b
        ratios = prices[in_bin] / preds[in_bin]
        low, median, high = np.quantile(ratios, [tail, 0.5, 1.0 - tail])
        bins.append({
            "n": int(in_bin.sum()),
            "pred_min": float(preds[in_bin].min()),
            "pred_max": float(preds[in_bin].max()),
            "ratio_low": float(low),
            "ratio_median": float(median),
            "ratio_high": float(high),
        })
    return {
        "coverage": coverage,
        "price_floor": float(floor),
        "price_ceiling": float(ceiling),
        "edges": [float(e) for e in edges],
        "bins": bins,
    }


def interval(price, calibration):
    """(low, high) for a prediction that has already been clamped."""
    group = int(np.searchsorted(calibration["edges"], price, side="right"))
    b = calibration["bins"][group]
    floor, ceiling = calibration["price_floor"], calibration["price_ceiling"]
    return (clamp(price * b["ratio_low"], floor, ceiling),
            clamp(price * b["ratio_high"], floor, ceiling))
