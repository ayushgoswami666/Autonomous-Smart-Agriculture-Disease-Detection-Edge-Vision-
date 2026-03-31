"""
test_model.py — Model Prediction & Preprocessing Tests
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection

Run:
    pytest tests/test_model.py -v
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.predict import preprocess_image, _estimate_severity


# ── Preprocessing ─────────────────────────────────────────────────────────────
def test_preprocess_creates_correct_shape(tmp_path):
    """Preprocessed image should be (1, 224, 224, 3)."""
    from PIL import Image
    img_path = str(tmp_path / "test_leaf.jpg")
    Image.new("RGB", (400, 300), color=(80, 160, 40)).save(img_path)

    arr = preprocess_image(img_path)
    assert arr.shape == (1, 224, 224, 3)


def test_preprocess_normalised(tmp_path):
    """Pixel values should be in [0, 1]."""
    from PIL import Image
    img_path = str(tmp_path / "test_leaf.jpg")
    Image.new("RGB", (224, 224), color=(255, 255, 255)).save(img_path)

    arr = preprocess_image(img_path)
    assert arr.max() <= 1.0
    assert arr.min() >= 0.0


def test_preprocess_handles_png(tmp_path):
    from PIL import Image
    img_path = str(tmp_path / "test_leaf.png")
    Image.new("RGBA", (300, 300), color=(0, 255, 0, 128)).save(img_path)

    arr = preprocess_image(img_path)
    assert arr.shape == (1, 224, 224, 3)   # RGBA → RGB conversion


# ── Severity Estimation ───────────────────────────────────────────────────────
@pytest.mark.parametrize("confidence, expected", [
    (0.95, "high"),
    (0.85, "high"),
    (0.75, "medium"),
    (0.60, "medium"),
    (0.50, "low"),
    (0.10, "low"),
])
def test_severity_levels(confidence, expected):
    assert _estimate_severity(confidence) == expected


# ── Predict (mocked model) ────────────────────────────────────────────────────
def test_predict_returns_expected_keys(tmp_path):
    """predict() should return disease, confidence, severity, top5."""
    from PIL import Image
    from model.predict import predict

    img_path = str(tmp_path / "leaf.jpg")
    Image.new("RGB", (224, 224), color=(60, 120, 40)).save(img_path)

    # Mock model
    num_classes = 38
    mock_probs  = np.zeros(num_classes, dtype=np.float32)
    mock_probs[5] = 0.92

    mock_model = MagicMock()
    mock_model.predict.return_value = np.expand_dims(mock_probs, axis=0)

    class_names = [f"Disease_{i}" for i in range(num_classes)]

    result = predict(img_path, model=mock_model, class_names=class_names)

    assert "disease"    in result
    assert "confidence" in result
    assert "severity"   in result
    assert "top5"       in result
    assert result["disease"]    == "Disease_5"
    assert result["confidence"] == pytest.approx(0.92, abs=1e-4)
    assert result["severity"]   == "high"
    assert len(result["top5"])  == 5
