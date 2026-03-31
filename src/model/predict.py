"""
predict.py — Crop Disease Prediction Utility
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image


IMG_SIZE = (224, 224)


def load_model_and_classes(model_path: str = "models/disease_model.h5",
                           classes_path: str = "models/class_names.npy"):
    """Load trained Keras model and class labels."""
    model = tf.keras.models.load_model(model_path)
    class_names = np.load(classes_path, allow_pickle=True).tolist()
    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess an image for inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


def predict(image_path: str,
            model=None,
            class_names=None,
            model_path: str = "models/disease_model.h5",
            classes_path: str = "models/class_names.npy") -> dict:
    """
    Predict disease from a leaf image.

    Returns:
        {
            "disease":     "Tomato___Early_blight",
            "confidence":  0.94,
            "severity":    "medium",
            "top5":        [("Tomato___Early_blight", 0.94), ...]
        }
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_classes(model_path, classes_path)

    img = preprocess_image(image_path)
    probs = model.predict(img, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    disease = class_names[top_idx]

    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [(class_names[i], float(probs[i])) for i in top5_indices]

    severity = _estimate_severity(confidence)

    return {
        "disease":    disease,
        "confidence": confidence,
        "severity":   severity,
        "top5":       top5,
    }


def _estimate_severity(confidence: float) -> str:
    """Map confidence score to severity level."""
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.60:
        return "medium"
    return "low"


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"\nDisease    : {result['disease']}")
    print(f"Confidence : {result['confidence'] * 100:.1f}%")
    print(f"Severity   : {result['severity']}")
    print("\nTop 5 Predictions:")
    for name, prob in result["top5"]:
        print(f"  {name:<45} {prob * 100:.1f}%")
