"""
Build a small stratified "dev" subset for local pipeline development and
sanity-testing (Phase 2).

Reuses the same price-bin stratification approach already used in
scripts/data_split.py (pd.qcut into 10 quantile bins) so the dev subset's
price distribution matches the full split's distribution.

NOTE beyond the literal Phase-2 spec: the spec only asked for a ~15,000-row
subset of the `train` split. That alone can't locally sanity-test a
train/val training loop (there would be zero validation rows), so this
script *also* draws a small, proportionally-sized stratified sample from the
`val` split (same train:val ratio as the real split, ~4.67:1) and includes
it in the same output CSV with its `split` column preserved as "val". The
train-subset size stays exactly what was asked (~15,000); only a val slice
is added on top. This does not touch the `test` split at all (test is
reserved for Phase 3).

Usage:
    python scripts/make_dev_subset.py \
        --input data/processed/metadata.csv \
        --output data/processed/dev_subset.csv \
        --n 15000
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def _stratified_sample(split_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    split_df = split_df.copy()
    split_df["price_bin"] = pd.qcut(split_df["price"], q=10, labels=False, duplicates="drop")
    frac = n / len(split_df)
    sample_df, _ = train_test_split(
        split_df, train_size=frac, random_state=seed, stratify=split_df["price_bin"],
    )
    return sample_df.drop(columns=["price_bin"])


def make_dev_subset(input_path: str, output_path: str, n: int, seed: int = 42) -> pd.DataFrame:
    print(f"Loading metadata from {input_path}...")
    df = pd.read_csv(input_path)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    print(f"Full train split size: {len(train_df)}")
    print(f"Full val split size: {len(val_df)}")

    if n >= len(train_df):
        raise ValueError(
            f"Requested subset size ({n}) >= train split size ({len(train_df)}); "
            "nothing to subsample."
        )

    # Train subset: exactly what was requested (~n rows).
    train_subset = _stratified_sample(train_df, n, seed)

    # Val subset: proportional to the real train:val ratio, so the dev
    # pipeline can be locally sanity-tested end-to-end (train + validate).
    val_ratio = len(val_df) / len(train_df)
    n_val = max(1, round(n * val_ratio))
    val_subset = _stratified_sample(val_df, n_val, seed)

    subset_df = pd.concat([train_subset, val_subset], ignore_index=True)

    print(f"Dev subset — train rows: {len(train_subset)} (requested ~{n}), "
          f"val rows: {len(val_subset)} (proportional, ratio={val_ratio:.3f})")
    print("Price stats — full train vs dev train-subset:")
    print("  full train : min=%.2f max=%.2f mean=%.2f" % (
        train_df["price"].min(), train_df["price"].max(), train_df["price"].mean()
    ))
    print("  dev train  : min=%.2f max=%.2f mean=%.2f" % (
        train_subset["price"].min(), train_subset["price"].max(), train_subset["price"].mean()
    ))
    print("Price stats — full val vs dev val-subset:")
    print("  full val   : min=%.2f max=%.2f mean=%.2f" % (
        val_df["price"].min(), val_df["price"].max(), val_df["price"].mean()
    ))
    print("  dev val    : min=%.2f max=%.2f mean=%.2f" % (
        val_subset["price"].min(), val_subset["price"].max(), val_subset["price"].mean()
    ))

    subset_df.to_csv(output_path, index=False)
    print(f"Saved dev subset ({len(subset_df)} total rows) to {output_path}")
    return subset_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stratified dev subset (train + proportional val).")
    parser.add_argument("--input", default="data/processed/metadata.csv")
    parser.add_argument("--output", default="data/processed/dev_subset.csv")
    parser.add_argument("--n", type=int, default=15000, help="Target size of the TRAIN portion of the subset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_dev_subset(args.input, args.output, args.n, args.seed)
