"""
Writes expected.json: the answer the API should give for each test photo.

The answers come from the Phase 3 reference code (model/predict_onnx.py and
model/price_range.py), not from the backend, so the tests compare two
separate routes to the same price: browser -> Next.js -> FastAPI -> model,
and the model on its own.

    python e2e/fixtures/make_expected.py     # repo root, after backend/download_model.py

Rerun it only when the model or backend/calibration.json changes.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from backend.download_model import EXPECTED_SHA256, RELEASE  # noqa: E402
from model.predict_onnx import load_session, predict  # noqa: E402
from model.price_range import clamp, interval  # noqa: E402

# Validation photos from the dataset (the model never trained on them), picked
# at three price levels. sample_id and listed_price are from metadata.csv.
PHOTOS = [
    ("mustard.jpg", "French's Honey Dijon mustard, 12 oz", 34988, 3.24),
    ("olive-oil.jpg", "Filippo Berio extra virgin olive oil, 750 ml", 298607, 10.97),
    ("garlic.jpg", "Rani organic ground garlic, 16 oz", 217731, 19.99),
]


def main():
    here = Path(__file__).resolve().parent
    session = load_session(str(REPO / "backend" / "models" / "price_model.onnx"))
    cal = json.loads((REPO / "backend" / "calibration.json").read_text())
    photos = []
    for file, product, sample_id, listed in PHOTOS:
        price = clamp(predict(session, here / "photos" / file), cal["price_floor"], cal["price_ceiling"])
        low, high = interval(price, cal)
        photos.append({"file": file, "product": product, "sample_id": sample_id, "listed_price": listed,
                       "price": round(price, 2), "low": round(low, 2), "high": round(high, 2)})
        print(f"{file:14} ${price:6.2f}  (${low:.2f} - ${high:.2f})   listed ${listed:.2f}")

    out = {"model_version": RELEASE, "model_sha256": EXPECTED_SHA256,
           "range_coverage": cal["coverage"], "photos": photos}
    (here / "expected.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
