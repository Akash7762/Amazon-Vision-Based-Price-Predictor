"""
Error analysis for a scored split (Phase 3).

Reads the per-image predictions written by model/evaluate.py, joins the
product descriptions from the metadata, and writes to --out-dir:

    error_analysis.md   tables and numbers, ready to copy into docs/
    calibration.png     predicted vs actual price, across price levels
    worst_under.jpg     the items priced furthest too low, with their photos
    worst_over.jpg      the items priced most too high (by ratio), with photos

    python model/error_analysis.py --predictions eval/run1_test_predictions.csv \
        --metadata .../metadata.csv --image-dir .../images --out-dir analysis --name run1

The pack-size check
-------------------
The working explanation for the model's errors is that much of what sets a
price can't be seen in a photo. Pack size is the part of that we can test:
about a third of products mention one ("Pack of 12", "6-Pack"), and the photo
often shows one unit. If the photo misses the pack size, the model should guess
multipacks low compared with single items at the same price level.

The check compares the two inside each price range (so the general habit of
guessing expensive items low doesn't leak in), then combines the ranges,
weighted by size, with a bootstrap 95% interval. Errors are measured as
log(predicted / actual), so a 20% miss counts the same on a $5 item as on a
$50 one.
"""
import argparse
import os
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.evaluate import BUCKETS, bucket_label

# Roughly log-spaced, so cheap items get as much resolution as expensive ones.
CALIBRATION_EDGES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 146]


def item_name(text):
    m = re.search(r"Item Name:\s*([^\n]+)", text or "")
    name = (m.group(1) if m else (text or "")).strip()
    return name.replace("|", "/")


def pack_count(text):
    """N from 'Pack of N' or 'N-Pack' / 'N Pack' in the description, else 1."""
    text = text or ""
    found = [int(n) for n in re.findall(r"[Pp]ack of\s*(\d+)", text)]
    found += [int(n) for n in re.findall(r"\b(\d+)\s*[- ]?[Pp]ack\b", text)]
    return max([n for n in found if n >= 1], default=1)


def pct(log_ratio):
    """Mean log(pred/actual) -> 'predicted x% above/below actual' (geometric)."""
    return 100.0 * (np.exp(log_ratio) - 1.0)


def pack_check(df, n_boot=2000, seed=0):
    """Multipack minus single, mean log-ratio, within price ranges."""
    rng = np.random.default_rng(seed)
    rows, groups = [], []
    for lo, hi in BUCKETS:
        band = df[(df["price"] >= lo) & (df["price"] < hi)]
        single = band.loc[band["pack"] == 1, "log_ratio"].to_numpy()
        multi = band.loc[band["pack"] >= 2, "log_ratio"].to_numpy()
        if len(single) == 0 or len(multi) == 0:
            continue
        rows.append({
            "range": bucket_label(lo, hi), "n_single": len(single), "n_multi": len(multi),
            "single_pct": pct(single.mean()), "multi_pct": pct(multi.mean()),
            "single_mae": float(band.loc[band["pack"] == 1, "abs_err"].mean()),
            "multi_mae": float(band.loc[band["pack"] >= 2, "abs_err"].mean()),
            "diff": float(multi.mean() - single.mean()),
        })
        groups.append((single, multi, len(band)))

    if not groups:  # no price range has both single items and multipacks
        return rows, float("nan"), float("nan"), float("nan")
    weights = np.array([g[2] for g in groups], dtype=float)
    weights /= weights.sum()
    overall = float(sum(w * (m.mean() - s.mean()) for w, (s, m, _) in zip(weights, groups)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = sum(w * (m[rng.integers(0, len(m), len(m))].mean()
                            - s[rng.integers(0, len(s), len(s))].mean())
                       for w, (s, m, _) in zip(weights, groups))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rows, overall, float(lo), float(hi)


def calibration(df):
    rows = []
    for lo, hi in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
        b = df[(df["price"] >= lo) & (df["price"] < hi)]
        if len(b):
            rows.append({"range": f"${lo}-{hi}", "n": len(b),
                         "median_actual": float(b["price"].median()),
                         "median_pred": float(b["pred"].median()),
                         "mae": float(b["abs_err"].mean())})
    return rows


def plot_calibration(df, cal, path, name, seed=0):
    sample = df.sample(min(len(df), 4000), random_state=seed)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(sample["price"], sample["pred"].clip(lower=0.5), s=4, alpha=0.15, c="C0",
               label="individual items (sample)")
    ax.plot([r["median_actual"] for r in cal], [r["median_pred"] for r in cal], "o-", c="C1",
            lw=2, label="median prediction per price level")
    ax.plot([1, 150], [1, 150], "--", c="gray", label="perfect prediction")
    ax.set(xscale="log", yscale="log", xlim=(1, 150), ylim=(0.5, 150),
           xlabel=r"actual price (\$)", ylabel=r"predicted price (\$)",
           title=f"{name}: predicted vs actual price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_grid(rows, image_dir, path, title):
    n = len(rows)
    cols = 4
    nrows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.2, nrows * 3.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), rows.iterrows()):
        img_path = os.path.join(image_dir, f"{r['sample_id']}.jpg")
        if os.path.exists(img_path):
            ax.imshow(Image.open(img_path).convert("RGB"))
        else:
            ax.text(0.5, 0.5, "image missing", ha="center", va="center")
        # Escape $: matplotlib reads a pair of them as a maths formula.
        name = "\n".join(textwrap.wrap(r["name"], 34)[:2]).replace("$", r"\$")
        ax.set_title(f"{name}\nactual \\${r['price']:.2f}  |  predicted \\${r['pred']:.2f}",
                     fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    # JPEG: these are photos, and PNG makes the grid several times larger.
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Error analysis for a scored split.")
    ap.add_argument("--predictions", required=True, help="*_predictions.csv from evaluate.py")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--name", default="model")
    ap.add_argument("--n-worst", type=int, default=12)
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions)
    meta = pd.read_csv(args.metadata, usecols=["sample_id", "catalog_content"])
    df = preds.merge(meta, on="sample_id", how="left")
    if df["catalog_content"].isna().all():
        raise SystemExit("ERROR: no descriptions matched - wrong metadata file?")
    df["name"] = df["catalog_content"].map(item_name)
    df["pack"] = df["catalog_content"].map(pack_count)
    df["log_ratio"] = np.log(df["pred"].clip(lower=0.01) / df["price"])
    os.makedirs(args.out_dir, exist_ok=True)

    cal = calibration(df)
    plot_calibration(df, cal, os.path.join(args.out_dir, "calibration.png"), args.name)

    under = df[df["pred"] < df["price"]].assign(gap=lambda d: d["price"] - d["pred"])
    under = under.sort_values("gap", ascending=False).head(args.n_worst)
    over = df.assign(ratio=df["pred"].clip(lower=0.01) / df["price"])
    over = over.sort_values("ratio", ascending=False).head(args.n_worst)
    plot_grid(under, args.image_dir, os.path.join(args.out_dir, "worst_under.jpg"),
              f"{args.name}: priced furthest too low")
    plot_grid(over, args.image_dir, os.path.join(args.out_dir, "worst_over.jpg"),
              f"{args.name}: priced most too high (by ratio)")

    pack_rows, diff, lo, hi = pack_check(df)
    if np.isnan(diff):
        reading = "not enough single items and multipacks in the same price ranges to compare."
    elif hi < 0:
        reading = ("multipacks are guessed lower than single items at the same price level, "
                   "and the interval excludes 0. Consistent with the photo not showing pack size.")
    elif lo > 0:
        reading = ("multipacks are guessed *higher* than single items at the same price level. "
                   "The opposite of the pack-size explanation.")
    else:
        reading = ("no clear difference between multipacks and single items at the same price "
                   "level. This check doesn't support the pack-size explanation.")

    n = len(df)
    lines = [f"# Error analysis: {args.name} on {os.path.basename(args.predictions)}", "",
             f"{n} images. Errors in dollars unless stated.", "",
             "| | value |", "|---|---|",
             f"| MAE | ${df['abs_err'].mean():.3f} |",
             f"| median absolute error | ${df['abs_err'].median():.3f} |",
             f"| SMAPE | {df['smape'].mean():.2f}% |",
             f"| bias (mean of predicted - actual) | ${(df['pred'] - df['price']).mean():+.3f} |",
             f"| predictions below $0 | {int((df['pred'] < 0).sum())} |",
             f"| predictions below $1 | {int((df['pred'] < 1).sum())} |",
             f"| products sold as packs of 2+ | {int((df['pack'] >= 2).sum())} ({(df['pack'] >= 2).mean():.1%}) |",
             "", "## Predicted vs actual, by price level", "",
             "![calibration](calibration.png)", "",
             "| actual price | n | median actual | median predicted | MAE |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['range']} | {r['n']} | ${r['median_actual']:.2f} | ${r['median_pred']:.2f} | "
              f"${r['mae']:.2f} |" for r in cal]
    lines += ["", "## Pack-size check", "",
              "Mean of log(predicted / actual), shown as a percentage: -20% means predictions "
              "sit 20% below the real price on average.", "",
              "| price range | single items | multipacks | single: predicted vs actual | "
              "multipack: predicted vs actual | single MAE | multipack MAE |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['range']} | {r['n_single']} | {r['n_multi']} | {r['single_pct']:+.1f}% | "
              f"{r['multi_pct']:+.1f}% | ${r['single_mae']:.2f} | ${r['multi_mae']:.2f} |"
              for r in pack_rows]
    lines += ["", f"**Multipacks vs single items at the same price level:** "
              f"{pct(diff):+.1f}% (95% interval {pct(lo):+.1f}% to {pct(hi):+.1f}%). "
              f"Reading: {reading}", ""]
    for title, rows, png in (("Priced furthest too low", under, "worst_under.jpg"),
                             ("Priced most too high (by ratio)", over, "worst_over.jpg")):
        lines += [f"## {title}", "", f"![{title}]({png})", "",
                  "| sample_id | product | actual | predicted |", "|---|---|---|---|"]
        lines += [f"| {r['sample_id']} | {textwrap.shorten(r['name'], 70)} | ${r['price']:.2f} | "
                  f"${r['pred']:.2f} |" for _, r in rows.iterrows()]
        lines.append("")

    md = os.path.join(args.out_dir, "error_analysis.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:16]))
    print(f"\nPack-size check: {pct(diff):+.1f}% (95% interval {pct(lo):+.1f}% to {pct(hi):+.1f}%)"
          f" -> {reading}")
    print(f"\nSaved {md}, calibration.png, worst_under.jpg, worst_over.jpg in {args.out_dir}")


if __name__ == "__main__":
    main()
