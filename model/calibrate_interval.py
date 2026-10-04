"""
Fit the price ranges the API reports (see model/price_range.py) and check
them on the test split.

    python model/calibrate_interval.py \
        --val-predictions ../phase2/run2/eval/run1_val_predictions.csv \
        --test-predictions ../phase3/eval/run1_test_predictions.csv \
        --metadata data/processed/metadata.csv --out backend/calibration.json

Fitted on validation predictions only. The test predictions are used once,
to report how often the real price actually fell inside its range; nothing is
tuned on them. The clamp bounds are the lowest and highest training prices,
since the model never saw anything outside them.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.price_range import clamp, fit, interval


def main():
    ap = argparse.ArgumentParser(description="Fit and check the API's price ranges.")
    ap.add_argument("--val-predictions", required=True, help="evaluate.py output for val")
    ap.add_argument("--test-predictions", required=True, help="evaluate.py output for test")
    ap.add_argument("--metadata", required=True, help="for the training price range")
    ap.add_argument("--out", default="backend/calibration.json")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--coverage", type=float, default=0.8)
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata, usecols=["price", "split"])
    train = meta.loc[meta["split"] == "train", "price"]
    floor, ceiling = float(train.min()), float(train.max())

    val = pd.read_csv(args.val_predictions)
    test = pd.read_csv(args.test_predictions)
    cal = fit(val["pred"], val["price"], floor, ceiling, args.bins, args.coverage)

    # Check on test, through exactly the path the API takes: clamp, then range.
    prices = [clamp(p, floor, ceiling) for p in test["pred"]]
    ranges = np.array([interval(p, cal) for p in prices])
    inside = (test["price"].to_numpy() >= ranges[:, 0]) & (test["price"].to_numpy() <= ranges[:, 1])
    groups = np.searchsorted(cal["edges"], prices, side="right")
    widths = ranges[:, 1] - ranges[:, 0]

    cal["fitted_on"] = f"{os.path.basename(args.val_predictions)} ({len(val)} val images)"
    cal["test_check"] = {
        "file": os.path.basename(args.test_predictions),
        "n": int(len(test)),
        "coverage": float(inside.mean()),
        "median_width_usd": float(np.median(widths)),
        "coverage_by_group": [float(inside[groups == b].mean()) for b in range(args.bins)],
    }
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")

    print(f"Fitted {args.bins} groups on {len(val)} val predictions; clamp ${floor}-${ceiling}")
    print("| predicted price | n (val) | range (x prediction) | median actual/pred | test coverage |")
    print("|---|---|---|---|---|")
    for b, g in enumerate(cal["bins"]):
        print(f"| ${g['pred_min']:.2f}-{g['pred_max']:.2f} | {g['n']} | "
              f"x{g['ratio_low']:.2f} to x{g['ratio_high']:.2f} | {g['ratio_median']:.2f} | "
              f"{cal['test_check']['coverage_by_group'][b]:.1%} |")
    tc = cal["test_check"]
    print(f"\nTest coverage {tc['coverage']:.1%} (target {args.coverage:.0%}), "
          f"median range width ${tc['median_width_usd']:.2f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
