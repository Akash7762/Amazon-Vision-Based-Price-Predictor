"""
The model behind the API.

Loads price_model.onnx once at startup and refuses to start if the file isn't
the one its model card describes (sha256), so a truncated download or the
wrong file fails loudly instead of serving wrong prices.

For each photo it uses the same code that was verified in Phase 3
(model/predict_onnx.py: turn upright, letterbox, raw pixels into the model),
then keeps the price inside the training range and adds the calibrated range
from model/price_range.py.
"""
import hashlib
import json
from pathlib import Path

import onnxruntime as ort

from model.predict_onnx import predict_image
from model.price_range import clamp, interval

GET_MODEL = "Download it with: python backend/download_model.py"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Predictor:
    def __init__(self, settings):
        for label, path, hint in (
            ("Model", settings.model_path, GET_MODEL),
            ("Model card", settings.model_card_path, GET_MODEL),
            ("Price-range calibration", settings.calibration_path,
             "It is committed at backend/calibration.json."),
        ):
            if not Path(path).exists():
                raise RuntimeError(f"{label} not found at {path}. {hint}")

        self.card = json.loads(Path(settings.model_card_path).read_text())
        self.sha256 = sha256(settings.model_path)
        if self.sha256 != self.card["sha256"]:
            raise RuntimeError(
                f"{settings.model_path} doesn't match its model card (sha256 {self.sha256[:12]}... "
                f"vs {self.card['sha256'][:12]}...). It may be damaged or a different model. {GET_MODEL}")

        self.session = ort.InferenceSession(str(settings.model_path),
                                            providers=["CPUExecutionProvider"])
        self.input_size = int(self.session.get_inputs()[0].shape[1])
        self.calibration = json.loads(Path(settings.calibration_path).read_text())
        self.version = settings.model_version

    def predict(self, img):
        cal = self.calibration
        # A raw-price head can return a negative or tiny number for an odd
        # photo; the model never saw prices outside the training range.
        price = clamp(predict_image(self.session, img), cal["price_floor"], cal["price_ceiling"])
        low, high = interval(price, cal)
        return {
            "price": round(price, 2),
            "currency": "USD",
            "range": {"low": round(low, 2), "high": round(high, 2), "coverage": cal["coverage"]},
            "model_version": self.version,
        }
