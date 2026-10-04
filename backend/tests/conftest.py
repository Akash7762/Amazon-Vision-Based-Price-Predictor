"""
Test fixtures: a tiny stand-in model with the real model's interface, so the
API tests run in seconds without the 112 MB file.

The stand-in prices a photo at (average pixel value) / 10, so an all-white
letterboxed photo costs exactly $25.50 and an all-black one $0 (clamped to
the $1.32 floor). Its input and output names, shapes and types match
price_model.onnx: uint8 "image" (1, 224, 224, 3) in, float "price" (1,) out.
"""
import io
import json

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.predictor import sha256

# Simple numbers, so expected ranges are easy to work out by hand.
CALIBRATION = {
    "coverage": 0.8,
    "price_floor": 1.32,
    "price_ceiling": 145.25,
    "edges": [10.0, 20.0],
    "bins": [
        {"n": 1, "pred_min": 1.32, "pred_max": 10.0, "ratio_low": 0.5, "ratio_median": 1.0, "ratio_high": 2.0},
        {"n": 1, "pred_min": 10.0, "pred_max": 20.0, "ratio_low": 0.6, "ratio_median": 1.0, "ratio_high": 1.5},
        {"n": 1, "pred_min": 20.0, "pred_max": 145.25, "ratio_low": 0.8, "ratio_median": 1.0, "ratio_high": 1.25},
    ],
}


def build_stand_in_model(path):
    nodes = [
        helper.make_node("Cast", ["image"], ["pixels"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["pixels"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Div", ["mean", "ten"], ["price"]),
    ]
    graph = helper.make_graph(
        nodes, "stand_in_price_model",
        [helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 224, 224, 3])],
        [helper.make_tensor_value_info("price", TensorProto.FLOAT, [1])],
        initializer=[numpy_helper.from_array(np.array([10.0], dtype=np.float32), "ten")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture
def model_files(tmp_path):
    model = tmp_path / "price_model.onnx"
    build_stand_in_model(str(model))
    card = tmp_path / "price_model.json"
    card.write_text(json.dumps({"file": model.name, "sha256": sha256(model)}))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(CALIBRATION))
    return model, card, calibration


@pytest.fixture
def settings(model_files):
    model, card, calibration = model_files
    return Settings(model_path=model, model_card_path=card, calibration_path=calibration,
                    model_version="test-model", max_upload_mb=1, max_image_pixels=10_000_000,
                    allowed_origins=("http://localhost:3000",))


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:  # "with" runs startup, which loads the model
        yield c


def image_bytes(img, fmt, **save_args):
    buf = io.BytesIO()
    img.save(buf, fmt, **save_args)
    return buf.getvalue()


@pytest.fixture
def white_png():
    return image_bytes(Image.new("RGB", (300, 200), (255, 255, 255)), "PNG")
