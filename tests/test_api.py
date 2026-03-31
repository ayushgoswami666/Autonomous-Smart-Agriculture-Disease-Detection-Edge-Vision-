"""
test_api.py — API Endpoint Tests
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection

Run:
    pytest tests/test_api.py -v
"""

import os
import io
import json
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api.app import app as flask_app


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["UPLOAD_FOLDER"] = "/tmp/test_uploads/"
    os.makedirs("/tmp/test_uploads/", exist_ok=True)
    with flask_app.test_client() as client:
        yield client


def _make_test_image() -> bytes:
    """Create a small 224x224 JPEG in memory."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (224, 224), color=(100, 180, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ── Health ────────────────────────────────────────────────────────────────────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data
    assert data["status"] == "ok"


# ── Upload ────────────────────────────────────────────────────────────────────
def test_upload_valid_image(client):
    img_bytes = _make_test_image()
    data = {"image": (io.BytesIO(img_bytes), "leaf.jpg")}
    resp = client.post("/api/image/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 201
    body = resp.get_json()
    assert "imageId" in body


def test_upload_no_file(client):
    resp = client.post("/api/image/upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_upload_invalid_extension(client):
    data = {"image": (io.BytesIO(b"fake"), "leaf.txt")}
    resp = client.post("/api/image/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


# ── Detect ────────────────────────────────────────────────────────────────────
def test_detect_no_body(client):
    resp = client.post("/api/detect", json={})
    assert resp.status_code == 400


def test_detect_missing_image(client):
    resp = client.post("/api/detect", json={"imageId": "nonexistent-id"})
    assert resp.status_code in (404, 503)   # 404 if model loaded, 503 if not


# ── Recommendation ────────────────────────────────────────────────────────────
def test_recommendation_known_disease(client):
    resp = client.get("/api/recommendation?disease=Tomato___Early_blight")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "treatment" in body
    assert "prevention" in body


def test_recommendation_unknown_disease(client):
    resp = client.get("/api/recommendation?disease=Unknown___XYZ_disease")
    assert resp.status_code == 404


def test_recommendation_missing_param(client):
    resp = client.get("/api/recommendation")
    assert resp.status_code == 400


def test_recommendation_healthy_plant(client):
    resp = client.get("/api/recommendation?disease=Tomato___healthy")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "treatment" in body


# ── History ───────────────────────────────────────────────────────────────────
def test_history_returns_list(client):
    resp = client.get("/api/history")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "records" in body
    assert isinstance(body["records"], list)
