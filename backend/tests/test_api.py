"""API tests against the stand-in model (see conftest.py)."""
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.tests.conftest import image_bytes

ORIGIN = "http://localhost:3000"


def post(client, data, name="photo.png", content_type="image/png", headers=None):
    return client.post("/predict", files={"file": (name, data, content_type)}, headers=headers)


# --- happy path -----------------------------------------------------------------

def test_health(client, settings):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-model"
    assert body["model_sha256"] == json.loads(settings.model_card_path.read_text())["sha256"]
    assert body["input_size"] == 224
    assert body["range_coverage"] == 0.8


def test_predict_white_photo(client, white_png):
    r = post(client, white_png)
    assert r.status_code == 200
    body = r.json()
    assert body["price"] == 25.5  # 255 / 10
    assert body["currency"] == "USD"
    # 25.5 is in the third group ($20+): x0.8 to x1.25
    assert body["range"] == {"low": 20.4, "high": 31.88, "coverage": 0.8}
    assert body["model_version"] == "test-model"


def test_black_photo_is_clamped_to_training_floor(client):
    black = image_bytes(Image.new("RGB", (224, 224), (0, 0, 0)), "JPEG")
    body = post(client, black, "photo.jpg", "image/jpeg").json()
    assert body["price"] == 1.32  # the model says $0; nothing below the cheapest training price
    assert body["range"]["low"] == 1.32  # the range is clamped too
    assert body["range"]["high"] == 2.64  # 1.32 x 2.0


def test_transparent_png_gets_a_white_background(client):
    transparent = image_bytes(Image.new("RGBA", (300, 300), (0, 0, 0, 0)), "PNG")
    # A plain RGB conversion would make this black ($1.32); the training
    # photos had white backgrounds, so transparency becomes white ($25.50).
    assert post(client, transparent).json()["price"] == 25.5


def test_webp_is_accepted(client):
    webp = image_bytes(Image.new("RGB", (100, 100), (255, 255, 255)), "WEBP", lossless=True)
    assert post(client, webp, "photo.webp", "image/webp").json()["price"] == 25.5


def test_large_photo_is_letterboxed(client):
    # 3000x2000 grey: scaled to 224x149 and padded with white above and below,
    # so the average sits between grey (128 -> $12.80) and white ($25.50).
    grey = image_bytes(Image.new("RGB", (3000, 2000), (128, 128, 128)), "PNG")
    r = post(client, grey)
    assert r.status_code == 200
    assert 12.8 < r.json()["price"] < 25.5


def test_generic_content_type_is_checked_by_content(client, white_png):
    # Some clients send application/octet-stream; the bytes decide.
    assert post(client, white_png, "photo", "application/octet-stream").status_code == 200


# --- rejected uploads -------------------------------------------------------------

def test_rejects_non_image_with_image_type(client):
    r = post(client, b"definitely not a picture", "photo.png", "image/png")
    assert r.status_code == 400
    assert "not a readable image" in r.json()["detail"]


def test_rejects_declared_non_image_type(client, white_png):
    r = post(client, white_png, "report.pdf", "application/pdf")
    assert r.status_code == 415


def test_rejects_unsupported_image_format(client):
    gif = image_bytes(Image.new("RGB", (50, 50), (255, 0, 0)), "GIF")
    r = post(client, gif, "anim.gif", "image/gif")
    assert r.status_code == 415
    # Even when it claims to be a JPEG, the bytes say GIF.
    assert post(client, gif, "anim.jpg", "image/jpeg").status_code == 415


def test_rejects_truncated_jpeg(client):
    jpeg = image_bytes(Image.effect_noise((400, 400), 60).convert("RGB"), "JPEG")
    r = post(client, jpeg[: len(jpeg) // 2], "photo.jpg", "image/jpeg")
    assert r.status_code == 400


def test_rejects_empty_file(client):
    assert post(client, b"", "photo.png", "image/png").status_code == 400


def test_rejects_file_over_size_limit(client):
    r = post(client, b"\xff" * (1024 * 1024 + 200 * 1024), "big.jpg", "image/jpeg")  # limit is 1 MB here
    assert r.status_code == 413
    assert "larger than 1 MB" in r.json()["detail"]


def test_rejects_too_many_pixels(client):
    # 4000x3000 = 12 MP, over the 10 MP test limit, but only a few KB as a PNG.
    huge = image_bytes(Image.new("RGB", (4000, 3000), (255, 255, 255)), "PNG")
    assert len(huge) < 1024 * 1024
    r = post(client, huge)
    assert r.status_code == 413
    assert "4000x3000" in r.json()["detail"]


def test_rejects_missing_file_field(client):
    assert client.post("/predict").status_code == 422


# --- CORS --------------------------------------------------------------------------

def test_cors_allows_the_frontend_origin(client):
    r = client.options("/predict", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_cors_ignores_other_origins(client):
    r = client.options("/predict", headers={"Origin": "https://example.com",
                                            "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_size_error_still_carries_cors_headers(client):
    # Otherwise the browser hides the 413 from the frontend as a CORS failure.
    r = post(client, b"\xff" * (2 * 1024 * 1024), "big.jpg", "image/jpeg", headers={"Origin": ORIGIN})
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == ORIGIN


# --- startup checks ----------------------------------------------------------------

def test_refuses_to_start_when_model_does_not_match_card(settings):
    settings.model_card_path.write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="doesn't match its model card"):
        with TestClient(create_app(settings)):
            pass


def test_refuses_to_start_without_model(settings, tmp_path):
    settings.model_path.unlink()
    with pytest.raises(RuntimeError, match="download_model.py"):
        with TestClient(create_app(settings)):
            pass
