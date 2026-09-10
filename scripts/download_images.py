"""
Resumable image download + resize pipeline (Phase 2).

Downloads each image referenced by `image_link` in a metadata CSV, resizes it
to the model's expected input resolution, and saves it locally as a JPEG
named by `sample_id`. Designed to work unmodified on both the ~15k-row dev
subset (locally) and the full ~51k-row train split (on Kaggle) — just point
it at a different --metadata / --out-dir.

Resumable: if an output file already exists, it is skipped (unless
--overwrite is passed), so re-running the script after an interruption only
downloads what's missing.

Failure handling: each image is retried a fixed number of times with a short
backoff. If all retries fail, the sample_id is logged to a failures CSV and
counted — failures are never silently dropped.

Usage:
    python scripts/download_images.py \
        --metadata data/processed/dev_subset.csv \
        --out-dir data/processed/images/dev \
        --image-size 224 \
        --workers 8
"""
import argparse
import csv
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

REQUEST_TIMEOUT_SECS = 10
RETRY_BACKOFF_SECS = 1.5


def download_and_resize_one(sample_id, url, out_dir, image_size, max_retries, overwrite):
    """Download one image, resize to (image_size, image_size), save as JPEG.

    Returns (sample_id, status, detail) where status is one of:
        "ok", "skipped_exists", "failed"
    """
    out_path = os.path.join(out_dir, f"{sample_id}.jpg")

    if not overwrite and os.path.exists(out_path):
        return sample_id, "skipped_exists", None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECS)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img = img.convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img.save(out_path, format="JPEG", quality=90)
            return sample_id, "ok", None
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(RETRY_BACKOFF_SECS * attempt)

    return sample_id, "failed", last_error


def download_images(metadata_path, out_dir, image_size, workers, max_retries, overwrite, limit=None):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    if limit is not None:
        df = df.head(limit)

    print(f"Loaded {len(df)} rows from {metadata_path}")
    print(f"Output dir: {out_dir}")
    print(f"Target image size: {image_size}x{image_size}")

    results = {"ok": 0, "skipped_exists": 0, "failed": 0}
    failures = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_and_resize_one,
                row.sample_id, row.image_link, out_dir, image_size, max_retries, overwrite,
            ): row.sample_id
            for row in df.itertuples()
        }

        done = 0
        total = len(futures)
        for future in as_completed(futures):
            sample_id, status, detail = future.result()
            results[status] += 1
            if status == "failed":
                failures.append((sample_id, detail))
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"  progress: {done}/{total} "
                      f"(ok={results['ok']} skipped={results['skipped_exists']} failed={results['failed']})")

    # Always write the failures log, even if empty, so it's clear failures were checked.
    failures_path = os.path.join(out_dir, "_download_failures.csv")
    with open(failures_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "error"])
        writer.writerows(failures)

    print("\n=== Download summary ===")
    print(f"Total rows requested : {total}")
    print(f"Downloaded OK         : {results['ok']}")
    print(f"Already present       : {results['skipped_exists']}")
    print(f"Failed (after retries): {results['failed']}")
    print(f"Failures logged to    : {failures_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and resize product images.")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV (must have sample_id, image_link).")
    parser.add_argument("--out-dir", required=True, help="Directory to save resized images into.")
    parser.add_argument("--image-size", type=int, default=224, help="Target square resolution (default: 224).")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per image before giving up.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if the local file exists.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows (for quick tests).")
    args = parser.parse_args()

    download_images(
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        image_size=args.image_size,
        workers=args.workers,
        max_retries=args.max_retries,
        overwrite=args.overwrite,
        limit=args.limit,
    )
