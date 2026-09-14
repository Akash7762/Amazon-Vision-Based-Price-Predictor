"""
Resumable image download + resize pipeline (Phase 2).

Downloads each image referenced by `image_link` in a metadata CSV, resizes it
to the model's expected input resolution, and saves it locally as a JPEG
named by `sample_id`. Designed to work unmodified on both the ~15k-row dev
subset (locally) and the full dataset (on Kaggle) — just point it at a
different --metadata / --out-dir.

Resizing uses letterbox-padding (see model/preprocessing.py), NOT a plain
stretch: aspect ratio is preserved and the image is centred on a white
canvas. Phase 4 inference must apply the identical transform.

Resumable and crash-safe: each JPEG is written to a temp file and atomically
renamed into place, so a file at its final path is always complete. An
interrupted run can therefore be resumed simply by re-running (existing
files are skipped unless --overwrite is passed) with no risk of a truncated
image being mistaken for a finished one.

A `_preprocessing.json` manifest is written into the output directory
recording the resize settings used; re-running with different settings
against a populated directory is refused rather than silently mixing
incompatible images.

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
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocessing import letterbox_resize, LETTERBOX_FILL, RESIZE_MODE

REQUEST_TIMEOUT_SECS = 10
RETRY_BACKOFF_SECS = 1.5
JPEG_QUALITY = 90
MANIFEST_NAME = "_preprocessing.json"


def download_and_resize_one(sample_id, url, out_dir, image_size, max_retries, overwrite):
    """Download one image, letterbox-resize it to (image_size, image_size),
    and save it as a JPEG.

    The file is written to a temporary path and then atomically renamed into
    place, so an interrupted run can never leave a truncated JPEG that a
    later resumed run would mistake for a completed download.

    Returns (sample_id, status, detail) where status is one of:
        "ok", "skipped_exists", "failed"
    """
    out_path = os.path.join(out_dir, f"{sample_id}.jpg")

    if not overwrite and os.path.exists(out_path):
        return sample_id, "skipped_exists", None

    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECS)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img = letterbox_resize(img, image_size, fill=LETTERBOX_FILL)
            img.save(tmp_path, format="JPEG", quality=JPEG_QUALITY)
            # Atomic on both POSIX and Windows when src/dst share a filesystem:
            # the final path only ever appears fully written.
            os.replace(tmp_path, out_path)
            return sample_id, "ok", None
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as e:
            last_error = str(e)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt < max_retries:
                time.sleep(RETRY_BACKOFF_SECS * attempt)

    return sample_id, "failed", last_error


def check_cache_consistency(out_dir, image_size, overwrite):
    """Guard against mixing images produced by different preprocessing.

    Silently mixing (say) stretched and letterboxed images inside one cache
    directory produces a training set with two incompatible preprocessing
    conventions and no error to reveal it — the most expensive possible
    mistake here, since the only remedy is a full re-download. So compare the
    current settings against the manifest left by whatever wrote this
    directory before, and refuse to continue on a mismatch.
    """
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    current = {"image_size": image_size, "resize_mode": RESIZE_MODE,
               "fill": list(LETTERBOX_FILL), "jpeg_quality": JPEG_QUALITY}

    existing_images = any(
        name.endswith(".jpg") for name in os.listdir(out_dir)
    ) if os.path.isdir(out_dir) else False

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        if previous != current and existing_images and not overwrite:
            raise SystemExit(
                f"ERROR: {out_dir} already holds images written with different "
                f"preprocessing.\n"
                f"  existing: {previous}\n"
                f"  current : {current}\n"
                f"Mixing them would corrupt the dataset. Re-run with --overwrite "
                f"to regenerate every image with the current settings, or point "
                f"--out-dir at a fresh directory."
            )
    elif existing_images and not overwrite:
        # Refuse rather than warn: we cannot verify how these were produced, and
        # writing the manifest now would falsely certify them as matching the
        # current settings, permanently hiding the inconsistency from later runs.
        raise SystemExit(
            f"ERROR: {out_dir} already contains .jpg files but no {MANIFEST_NAME}, "
            f"so the preprocessing used to create them cannot be verified. They were "
            f"most likely written before letterbox-padding was introduced (i.e. "
            f"stretched to square).\n"
            f"Re-run with --overwrite to regenerate every image with the current "
            f"settings ({current}), or point --out-dir at a fresh directory."
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)


def download_images(metadata_path, out_dir, image_size, workers, max_retries, overwrite, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    check_cache_consistency(out_dir, image_size, overwrite)

    df = pd.read_csv(metadata_path, usecols=["sample_id", "image_link"])
    if limit is not None:
        df = df.head(limit)

    print(f"Loaded {len(df)} rows from {metadata_path}")
    print(f"Output dir: {out_dir}")
    print(f"Target image size: {image_size}x{image_size} (resize mode: {RESIZE_MODE})")

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
