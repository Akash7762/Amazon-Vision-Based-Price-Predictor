"""
Price product photos with the exported ONNX model.

Needs only onnxruntime, numpy and Pillow, not PyTorch. This is the same route
the Phase 4 backend will take: open the photo, letterbox it, pass the raw
pixels to the model.

    python model/predict_onnx.py --model export/price_model.onnx photo1.jpg photo2.jpg

Check mode re-prices a sample of the images in a predictions CSV from
model/evaluate.py and fails if any price differs by more than a cent:

    python model/predict_onnx.py --model export/price_model.onnx \
        --check-predictions eval/run1_test_predictions.csv --image-dir .../images

Phone photos often store their rotation in EXIF instead of in the pixels, so
the photo is turned upright first. Amazon catalogue images have no such tag,
so this changes nothing for them.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocessing import letterbox_resize  # PIL only, no torch


def load_session(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def pixels_from_image(img, size):
    """An opened photo -> the model's input: upright, letterboxed, (1, size, size, 3) uint8."""
    img = ImageOps.exif_transpose(img)
    return np.asarray(letterbox_resize(img, size), dtype=np.uint8)[None]


def predict_image(session, img):
    """Dollars for one opened photo (a PIL Image). The backend calls this."""
    size = session.get_inputs()[0].shape[1]  # (1, size, size, 3)
    return float(session.run(["price"], {"image": pixels_from_image(img, size)})[0][0])


def predict(session, image_path):
    """Dollars for one photo on disk."""
    with Image.open(image_path) as img:
        return predict_image(session, img)


def check(session, predictions_csv, image_dir, n, tolerance=0.01):
    """Re-price a sample of evaluate.py's images; fail if any differs by > tolerance."""
    with open(predictions_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = random.Random(0).sample(rows, min(n, len(rows)))
    print("| sample_id | actual | evaluate.py (PyTorch) | ONNX |\n|---|---|---|---|")
    worst = 0.0
    for r in rows:
        price = predict(session, os.path.join(image_dir, f"{r['sample_id']}.jpg"))
        worst = max(worst, abs(price - float(r["pred"])))
        print(f"| {r['sample_id']} | ${float(r['price']):.2f} | ${float(r['pred']):.2f} | ${price:.2f} |")
    if worst > tolerance:
        raise SystemExit(f"ERROR: ONNX and PyTorch differ by up to ${worst:.4f}")
    print(f"\nAll {len(rows)} match (largest difference ${worst:.4f}).")


def main():
    ap = argparse.ArgumentParser(description="Price product photos with the ONNX model.")
    ap.add_argument("--model", required=True, help="path to price_model.onnx")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--check-predictions", metavar="CSV",
                    help="predictions CSV from evaluate.py: re-price a sample and compare")
    ap.add_argument("--image-dir", help="folder with <sample_id>.jpg, for --check-predictions")
    ap.add_argument("--n", type=int, default=8, help="how many images --check-predictions uses")
    args = ap.parse_args()

    session = load_session(args.model)
    if args.check_predictions:
        if not args.image_dir:
            ap.error("--check-predictions needs --image-dir")
        check(session, args.check_predictions, args.image_dir, args.n)
        return
    if not args.images:
        ap.error("give at least one image, or --check-predictions")
    for path in args.images:
        print(f"${predict(session, path):8.2f}  {path}")


if __name__ == "__main__":
    main()
