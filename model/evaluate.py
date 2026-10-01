"""
Score a trained checkpoint in dollars, and compare runs.

1. Evaluate one checkpoint on a split:

    python model/evaluate.py --checkpoint .../checkpoints/best.pt --name run1 \
        --metadata .../metadata.csv --image-dir .../images --split val \
        --out-dir /kaggle/working/eval

   Writes <out-dir>/<name>_<split>.json (metrics) and
   <out-dir>/<name>_<split>_predictions.csv (one row per image, worst first).

2. Compare runs that were evaluated on the same split:

    python model/evaluate.py --compare eval/run1_val.json eval/run2_val.json

   Prints side-by-side tables (markdown, so they can go straight into
   docs/experiments.md) and saves them next to the first JSON. With exactly two
   runs it adds a paired bootstrap 95% interval for the difference.

Metrics, all on the original dollar scale whatever the model was trained on:
MAE, RMSE, median absolute error, SMAPE (%), and bias (mean of prediction
minus price: negative means the model guesses low on average). Every report
also scores "always predict the median train price" on the same images, so no
number appears without its reference point.

Compare runs with --split val. --split test is for the final Phase 3 score,
once, after the model has been chosen: choosing between runs by their test
score would make that score optimistic.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import PriceImageDataset
from model.model import build_model, EXPECTED_INPUT_SIZE
from model.targets import checkpoint_target, to_price

# Price ranges for the breakdown. Each holds 1,100+ of the 11,028 val images.
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, float("inf"))]

METRIC_ROWS = [("mae", "MAE ($)"), ("rmse", "RMSE ($)"), ("median_ae", "Median abs error ($)"),
               ("smape", "SMAPE (%)"), ("bias", "Bias ($)")]


def bucket_label(lo, hi):
    return f"${lo}+" if hi == float("inf") else f"${lo}-{hi}"


def smape_terms(y, p):
    """Per-item symmetric percentage error. Their mean is SMAPE (0-200%)."""
    return 200.0 * np.abs(p - y) / (np.abs(y) + np.abs(p))


def metrics(y, p):
    err = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(np.abs(err))),
        "smape": float(np.mean(smape_terms(y, p))),
        "bias": float(np.mean(err)),
    }


@torch.no_grad()
def predict(checkpoint_path, metadata, image_dir, split, batch_size, num_workers,
            max_batches, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    target = checkpoint_target(ckpt)

    # pretrained=False: the weights come from the checkpoint, no download needed.
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = PriceImageDataset(metadata, image_dir, split=split,
                           image_size=EXPECTED_INPUT_SIZE, train=False)
    if len(ds) == 0:
        raise SystemExit(f"ERROR: no usable '{split}' rows for {metadata} / {image_dir}.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")

    preds = []
    for i, (images, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(images.to(device))
        preds.append(to_price(out, target).squeeze(1).float().cpu().numpy())
    p = np.concatenate(preds).astype(np.float64)

    rows = ds.df.iloc[:len(p)]  # shuffle=False, so predictions line up with ds.df
    return ckpt, target, rows["sample_id"].to_numpy(), rows["price"].to_numpy(np.float64), p


def evaluate(args):
    if args.split == "test":
        print("NOTE: scoring the TEST split. Do this once, for the chosen model only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ckpt, target, ids, y, p = predict(args.checkpoint, args.metadata, args.image_dir,
                                      args.split, args.batch_size, args.num_workers,
                                      args.max_batches, device)

    meta = pd.read_csv(args.metadata, usecols=["price", "split"])
    train_median = float(meta.loc[meta["split"] == "train", "price"].median())
    base = np.full_like(y, train_median)

    name = re.sub(r"[^A-Za-z0-9_-]+", "-", args.name).strip("-") or "model"
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{name}_{args.split}")

    preds = pd.DataFrame({"sample_id": ids, "price": y, "pred": p})
    preds["abs_err"] = (preds["pred"] - preds["price"]).abs()
    preds["smape"] = smape_terms(y, p)
    preds.sort_values("abs_err", ascending=False).round(4).to_csv(
        stem + "_predictions.csv", index=False)

    by_range = []
    for lo, hi in BUCKETS:
        m = (y >= lo) & (y < hi)
        if m.any():
            mm, bm = metrics(y[m], p[m]), metrics(y[m], base[m])
            by_range.append({"range": bucket_label(lo, hi), "n": int(m.sum()),
                             "model_mae": mm["mae"], "baseline_mae": bm["mae"],
                             "model_smape": mm["smape"], "baseline_smape": bm["smape"],
                             "model_bias": mm["bias"]})

    report = {
        "name": name,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "target": target,
        "train_median_price": train_median,
        "model": metrics(y, p),
        "baseline_median": metrics(y, base),
        "by_price_range": by_range,
        # Relative, so --compare still finds it if the folder is moved.
        "predictions_csv": os.path.basename(stem + "_predictions.csv"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(stem + ".json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {name} on {args.split}: {report['model']['n']} images, "
          f"epoch {report['epoch']}, target={target}, {report['seconds']:.0f}s ===")
    print(f"| | {name} | median baseline (${train_median:.2f}) |\n|---|---|---|")
    for key, label in METRIC_ROWS:
        print(f"| {label} | {report['model'][key]:.3f} | {report['baseline_median'][key]:.3f} |")
    print("\n| Price range | n | MAE ($) | baseline MAE | SMAPE (%) | baseline SMAPE | bias ($) |")
    print("|---|---|---|---|---|---|---|")
    for r in by_range:
        print(f"| {r['range']} | {r['n']} | {r['model_mae']:.2f} | {r['baseline_mae']:.2f} | "
              f"{r['model_smape']:.1f} | {r['baseline_smape']:.1f} | {r['model_bias']:+.2f} |")
    print(f"\nSaved {stem}.json and {stem}_predictions.csv")


def paired_bootstrap(d, n_boot=2000, seed=0):
    """95% interval for mean(d), resampling images with replacement."""
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(json_paths):
    reports = [json.load(open(p)) for p in json_paths]
    if len({(r["split"], r["model"]["n"]) for r in reports}) > 1:
        raise SystemExit("ERROR: these runs were scored on different splits or image sets.")
    names = [r["name"] for r in reports]
    split = reports[0]["split"]
    lines = []

    lines.append(f"### Comparison on {split} ({reports[0]['model']['n']} images)\n")
    lines.append("| | median baseline | " + " | ".join(
        f"{r['name']} (target={r['target']}, epoch {r['epoch']})" for r in reports) + " |")
    lines.append("|---|---|" + "---|" * len(reports))
    for key, label in METRIC_ROWS:
        lines.append(f"| {label} | {reports[0]['baseline_median'][key]:.3f} | "
                     + " | ".join(f"{r['model'][key]:.3f}" for r in reports) + " |")

    lines.append("\n| Price range | n | baseline MAE | " + " | ".join(
        f"{n} MAE" for n in names) + " | " + " | ".join(f"{n} SMAPE" for n in names) + " |")
    lines.append("|---|---|---|" + "---|" * (2 * len(reports)))
    for i, b in enumerate(reports[0]["by_price_range"]):
        lines.append(f"| {b['range']} | {b['n']} | {b['baseline_mae']:.2f} | "
                     + " | ".join(f"{r['by_price_range'][i]['model_mae']:.2f}" for r in reports)
                     + " | "
                     + " | ".join(f"{r['by_price_range'][i]['model_smape']:.1f}" for r in reports)
                     + " |")

    if len(reports) == 2:
        a, b = reports
        frames = []
        for path, r in zip(json_paths, reports):
            csv = os.path.join(os.path.dirname(os.path.abspath(path)), r["predictions_csv"])
            frames.append(pd.read_csv(csv).set_index("sample_id"))
        if not frames[0].index.sort_values().equals(frames[1].index.sort_values()):
            raise SystemExit("ERROR: the two prediction files cover different images.")
        fa, fb = frames[0], frames[1].loc[frames[0].index]

        lines.append(f"\n**{b['name']} minus {a['name']}** (negative = {b['name']} better), "
                     f"paired bootstrap over images, 2000 resamples:\n")
        lines.append("| Metric | difference | 95% interval | reading |\n|---|---|---|---|")
        for col, label in (("abs_err", "MAE ($)"), ("smape", "SMAPE (%)")):
            diff, lo, hi = paired_bootstrap((fb[col] - fa[col]).to_numpy())
            if hi < 0:
                reading = f"{b['name']} lower, interval excludes 0"
            elif lo > 0:
                reading = f"{a['name']} lower, interval excludes 0"
            else:
                reading = "no clear difference"
            lines.append(f"| {label} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {reading} |")
        lines.append("\nThe interval covers which images happened to land in the split. It "
                     "doesn't cover run-to-run randomness from training (seed, shuffle order).")

    text = "\n".join(lines)
    print(text)
    out = os.path.join(os.path.dirname(os.path.abspath(json_paths[0])),
                       f"comparison_{split}.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description="Score checkpoints in dollars, or compare runs.")
    ap.add_argument("--compare", nargs="+", metavar="JSON",
                    help="Compare already-evaluated runs instead of evaluating a checkpoint.")
    ap.add_argument("--checkpoint")
    ap.add_argument("--name", default="model", help="Label used in file names and tables.")
    ap.add_argument("--metadata")
    ap.add_argument("--image-dir")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Only for quick code checks; the numbers are meaningless with it.")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    missing = [f for f in ("checkpoint", "metadata", "image_dir") if not getattr(args, f)]
    if missing:
        ap.error("missing " + ", ".join("--" + m.replace("_", "-") for m in missing))
    for label, path in (("checkpoint", args.checkpoint), ("metadata", args.metadata),
                        ("image dir", args.image_dir)):
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: {label} not found at '{path}'.")
    evaluate(args)


if __name__ == "__main__":
    main()
