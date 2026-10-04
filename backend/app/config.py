"""
Settings, read from environment variables so the same code runs on a laptop,
in the tests and (Phase 7) in Docker.

    MODEL_PATH          price_model.onnx            default backend/models/price_model.onnx
    MODEL_CARD_PATH     price_model.json            default next to the model
    CALIBRATION_PATH    price-range calibration     default backend/calibration.json
    MODEL_VERSION       reported by the API         default v0.1-model
    MAX_UPLOAD_MB       largest accepted upload     default 10
    MAX_IMAGE_PIXELS    largest accepted image      default 50,000,000 (a 48 MP phone photo fits)
    ALLOWED_ORIGINS     comma-separated, for CORS   default http://localhost:3000 (Next.js dev server)
"""
import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_path: Path = BACKEND_DIR / "models" / "price_model.onnx"
    model_card_path: Path = BACKEND_DIR / "models" / "price_model.json"
    calibration_path: Path = BACKEND_DIR / "calibration.json"
    model_version: str = "v0.1-model"
    max_upload_mb: float = 10.0
    max_image_pixels: int = 50_000_000
    allowed_origins: tuple = ("http://localhost:3000",)

    @property
    def max_upload_bytes(self):
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls):
        env = os.environ.get
        model_path = Path(env("MODEL_PATH", cls.model_path))
        return cls(
            model_path=model_path,
            model_card_path=Path(env("MODEL_CARD_PATH", model_path.with_suffix(".json"))),
            calibration_path=Path(env("CALIBRATION_PATH", cls.calibration_path)),
            model_version=env("MODEL_VERSION", cls.model_version),
            max_upload_mb=float(env("MAX_UPLOAD_MB", cls.max_upload_mb)),
            max_image_pixels=int(env("MAX_IMAGE_PIXELS", cls.max_image_pixels)),
            allowed_origins=tuple(o.strip() for o in env("ALLOWED_ORIGINS", ",".join(cls.allowed_origins)).split(",")
                                  if o.strip()),
        )
