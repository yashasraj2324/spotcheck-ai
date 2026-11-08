import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from skin_lesion.api.main import create_app
from skin_lesion.api.service import Predictor
from skin_lesion.data.constants import CLASSES
from skin_lesion.models.factory import build_model


def _png_bytes(size=80) -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tiny_cfg):
    predictor = Predictor(build_model(tiny_cfg), tiny_cfg)
    with TestClient(create_app(predictor)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_contract(client):
    r = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["top_class"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    overlay = Image.open(io.BytesIO(base64.b64decode(body["gradcam_png_base64"])))
    assert overlay.size == (64, 64)
    assert "not" in body["disclaimer"].lower()


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 422


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setenv("SKIN_LESION_CHECKPOINT", "does/not/exist.pth")
    with TestClient(create_app(None)) as c:
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
        assert r.status_code == 503
