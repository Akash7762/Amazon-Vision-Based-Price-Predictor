"""
API tests with the real model: uploaded bytes in, price out, checked against
the Phase 3 reference code (e2e/fixtures/expected.json, written by
e2e/fixtures/make_expected.py).

Skipped when the model hasn't been downloaded (python backend/download_model.py).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.download_model import EXPECTED_SHA256
from backend.tests.conftest import image_bytes

FIXTURES = Path(__file__).resolve().parents[2] / "e2e" / "fixtures"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())
SETTINGS = Settings()  # the real model, at its default place
CENT = 0.015  # a cent either way, for float differences between CPUs

pytestmark = pytest.mark.skipif(not SETTINGS.model_path.exists(),
                                reason="real model not downloaded: python backend/download_model.py")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(SETTINGS)) as c:
        yield c


@pytest.fixture(scope="module")
def olive_oil():
    with Image.open(FIXTURES / "photos" / "olive-oil.jpg") as img:
        return img.convert("RGB")


def answer(client, data, name="photo.png", content_type="image/png"):
    r = client.post("/predict", files={"file": (name, data, content_type)})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_the_released_model(client):
    body = client.get("/health").json()
    assert body["model_version"] == EXPECTED["model_version"]
    assert body["model_sha256"] == EXPECTED_SHA256 == EXPECTED["model_sha256"]


@pytest.mark.parametrize("photo", EXPECTED["photos"], ids=lambda p: p["file"])
def test_photo_gets_the_reference_answer(client, photo):
    body = answer(client, (FIXTURES / "photos" / photo["file"]).read_bytes(), photo["file"], "image/jpeg")
    assert body["price"] == pytest.approx(photo["price"], abs=CENT)
    assert body["range"]["low"] == pytest.approx(photo["low"], abs=CENT)
    assert body["range"]["high"] == pytest.approx(photo["high"], abs=CENT)
    assert body["range"]["coverage"] == EXPECTED["range_coverage"]
    assert body["model_version"] == EXPECTED["model_version"]


def test_same_pixels_same_price_in_every_format(client, olive_oil):
    jpeg = (FIXTURES / "photos" / "olive-oil.jpg").read_bytes()
    prices = {
        "jpeg": answer(client, jpeg, "photo.jpg", "image/jpeg")["price"],
        "png": answer(client, image_bytes(olive_oil, "PNG"))["price"],
        "webp": answer(client, image_bytes(olive_oil, "WEBP", lossless=True), "photo.webp", "image/webp")["price"],
    }
    assert len(set(prices.values())) == 1, prices


def test_rotation_tag_turns_the_photo_upright(client, olive_oil):
    # Stored on its side with EXIF orientation 6, the way phones save photos.
    exif = Image.Exif()
    exif[0x0112] = 6
    sideways = image_bytes(olive_oil.transpose(Image.Transpose.ROTATE_90), "PNG", exif=exif.tobytes())
    assert answer(client, sideways)["price"] == answer(client, image_bytes(olive_oil, "PNG"))["price"]


def test_transparent_background_counts_as_white(client, olive_oil):
    # Cut the product out: the near-white background becomes transparent (and
    # black underneath, which is what a careless RGB conversion would show).
    px = np.asarray(olive_oil)
    background = (px >= 250).all(axis=2)
    rgba = np.dstack([px, np.where(background, 0, 255)]).astype(np.uint8)
    rgba[background, :3] = 0
    on_white = px.copy()
    on_white[background] = 255
    cut_out = answer(client, image_bytes(Image.fromarray(rgba, "RGBA"), "PNG"))["price"]
    assert cut_out == answer(client, image_bytes(Image.fromarray(on_white), "PNG"))["price"]


def test_16bit_greyscale_matches_8bit(client, olive_oil):
    grey = olive_oil.convert("L")
    grey16 = Image.fromarray(np.asarray(grey, dtype=np.uint16) * 257)
    assert answer(client, image_bytes(grey16, "PNG"))["price"] == answer(client, image_bytes(grey, "PNG"))["price"]


def unusual_images(photo):
    exif = Image.Exif()
    exif[0x0112] = 9  # not a valid orientation
    return {
        "1x1 pixel": image_bytes(Image.new("RGB", (1, 1), (200, 30, 30)), "PNG"),
        "5000x20 strip": image_bytes(Image.new("RGB", (5000, 20), (30, 30, 200)), "PNG"),
        "blank white": image_bytes(Image.new("RGB", (640, 480), (255, 255, 255)), "PNG"),
        "blank black": image_bytes(Image.new("RGB", (640, 480), (0, 0, 0)), "PNG"),
        "noise": image_bytes(Image.effect_noise((640, 480), 80).convert("RGB"), "PNG"),
        "greyscale jpeg": image_bytes(photo.convert("L"), "JPEG"),
        "1-bit png": image_bytes(photo.convert("1"), "PNG"),
        "palette png, transparent": image_bytes(photo.convert("P"), "PNG", transparency=0),
        "cmyk jpeg": image_bytes(photo.convert("CMYK"), "JPEG"),
        "progressive jpeg": image_bytes(photo, "JPEG", progressive=True),
        "bad rotation tag": image_bytes(photo, "JPEG", exif=exif.tobytes()),
        "webp with alpha": image_bytes(photo.convert("RGBA"), "WEBP"),
        "animated webp": image_bytes(photo, "WEBP", save_all=True, append_images=[photo.rotate(90)]),
        "animated png": image_bytes(photo, "PNG", save_all=True, append_images=[photo.rotate(90)]),
        "jpeg with extra images (mpo)": image_bytes(photo, "MPO", save_all=True, append_images=[photo]),
    }


@pytest.mark.parametrize("kind", list(unusual_images(Image.new("RGB", (1, 1)))))
def test_unusual_image_gets_a_valid_answer(client, olive_oil, kind):
    # Not all of these are product photos: the model has no way to say "this
    # isn't a product" and prices whatever it's given. The API's job is to
    # answer every valid image with a price and range inside the training range.
    body = answer(client, unusual_images(olive_oil)[kind], "upload", "application/octet-stream")
    price, low, high = body["price"], body["range"]["low"], body["range"]["high"]
    assert 1.32 <= low <= price <= high <= 145.25
