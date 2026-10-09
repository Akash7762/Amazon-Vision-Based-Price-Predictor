"""
Unit tests for the shared pieces the API is built on: image preparation,
upload checks, the price range, and the committed calibration file.
"""
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.app.images import UploadError, read_upload
from model.predict_onnx import pixels_from_image
from model.preprocessing import letterbox_resize
from model.price_range import clamp, fit, interval

REPO = Path(__file__).resolve().parents[2]


def test_pixels_have_the_model_input_shape():
    px = pixels_from_image(Image.new("RGB", (640, 480), (10, 20, 30)), 224)
    assert px.shape == (1, 224, 224, 3) and px.dtype == np.uint8


def test_phone_photo_rotation_is_applied():
    # Stored landscape 200x100, with an EXIF tag saying "rotate 90" (orientation 6),
    # as phones do. Upright it is portrait, so the white padding goes left and right.
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), (200, 30, 30)).save(buf, "JPEG", exif=exif.tobytes())
    px = pixels_from_image(Image.open(buf), 224)[0]
    assert (px[112, 5] > 240).all()      # left edge: padding
    assert (px[5, 112] < 240).any()      # top edge: photo, so it was turned upright


def test_letterbox_leaves_plain_photos_alone():
    # The transparency handling must not change ordinary RGB photos, which is
    # what the model was trained and verified on.
    img = Image.effect_noise((300, 180), 50).convert("RGB")
    expected = Image.new("RGB", (224, 224), (255, 255, 255))
    expected.paste(img.resize((224, 134), Image.BILINEAR), (0, 45))
    assert np.array_equal(np.asarray(letterbox_resize(img, 224)), np.asarray(expected))


@pytest.mark.parametrize("dtype", ["<u2", ">u2", "<i4"])  # Pillow modes I;16, I;16B, I
def test_16bit_greyscale_is_scaled_not_clipped(dtype):
    # Mid-grey in 16 bits is 128 * 257. Clipped at 255 it would turn white.
    img = Image.fromarray(np.full((200, 300), 128 * 257, dtype=dtype))
    assert tuple(letterbox_resize(img, 224).getpixel((112, 112))) == (128, 128, 128)


def test_read_upload_caps_bodies_without_content_length():
    with pytest.raises(UploadError) as e:
        read_upload(io.BytesIO(b"x" * 101), max_bytes=100)
    assert e.value.status == 413
    assert read_upload(io.BytesIO(b"x" * 100), max_bytes=100) == b"x" * 100


def test_price_range_contains_the_price_and_stays_in_bounds():
    rng = np.random.default_rng(0)
    preds = rng.uniform(2, 100, 2000)
    prices = preds * rng.lognormal(0, 0.5, 2000)
    cal = fit(preds, prices, floor=1.32, ceiling=145.25, n_bins=4, coverage=0.8)
    for p in (1.32, 5.0, 37.0, 145.25):
        low, high = interval(p, cal)
        assert 1.32 <= low <= p <= high <= 145.25
    assert clamp(-3.0, 1.32, 145.25) == 1.32 and clamp(500, 1.32, 145.25) == 145.25


def test_fitted_range_hits_its_coverage():
    rng = np.random.default_rng(1)
    preds = rng.uniform(5, 50, 20000)
    prices = preds * rng.lognormal(0, 0.4, 20000)
    cal = fit(preds[:10000], prices[:10000], floor=0.01, ceiling=1e9, n_bins=8, coverage=0.8)
    ranges = np.array([interval(p, cal) for p in preds[10000:]])
    inside = (prices[10000:] >= ranges[:, 0]) & (prices[10000:] <= ranges[:, 1])
    assert 0.77 < inside.mean() < 0.83


def test_committed_calibration_is_sane():
    cal = json.loads((REPO / "backend" / "calibration.json").read_text())
    assert cal["coverage"] == 0.8
    assert cal["edges"] == sorted(cal["edges"]) and len(cal["bins"]) == len(cal["edges"]) + 1
    for b in cal["bins"]:
        assert b["ratio_low"] < 1 < b["ratio_high"]
    assert 0.75 < cal["test_check"]["coverage"] < 0.85
