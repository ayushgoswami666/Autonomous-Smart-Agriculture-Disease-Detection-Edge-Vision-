"""
app.py — Flask Backend API
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection

Endpoints:
  POST /api/image/upload       — Upload crop leaf image
  POST /api/detect             — Detect disease from imageId
  GET  /api/recommendation     — Get treatment recommendation
  GET  /api/history            — Get detection history
"""

import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.predict import predict, load_model_and_classes
from api.recommendations import get_recommendation

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = "data/uploads/"
ALLOWED_EXT     = {"png", "jpg", "jpeg"}
MODEL_PATH      = "models/disease_model.h5"
CLASSES_PATH    = "models/class_names.npy"
HISTORY_FILE    = "data/history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model once at startup
model, class_names = None, None
try:
    model, class_names = load_model_and_classes(MODEL_PATH, CLASSES_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[WARN] Could not load model: {e}. Run training first.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(records: list):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/image/upload", methods=["POST"])
def upload_image():
    """Upload a crop leaf image. Returns imageId."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

    image_id  = str(uuid.uuid4())
    filename  = secure_filename(f"{image_id}.{file.filename.rsplit('.', 1)[1].lower()}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    return jsonify({"imageId": image_id, "filename": filename}), 201


@app.route("/api/detect", methods=["POST"])
def detect_disease():
    """Detect disease from an uploaded imageId."""
    data = request.get_json()
    if not data or "imageId" not in data:
        return jsonify({"error": "imageId is required"}), 400

    image_id = data["imageId"]

    # Find file
    image_path = None
    for ext in ALLOWED_EXT:
        candidate = os.path.join(UPLOAD_FOLDER, f"{image_id}.{ext}")
        if os.path.exists(candidate):
            image_path = candidate
            break

    if not image_path:
        return jsonify({"error": "Image not found. Upload first."}), 404

    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 503

    try:
        result = predict(image_path, model=model, class_names=class_names)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Save to history
    record = {
        "id":          str(uuid.uuid4()),
        "imageId":     image_id,
        "disease":     result["disease"],
        "confidence":  result["confidence"],
        "severity":    result["severity"],
        "timestamp":   datetime.utcnow().isoformat(),
    }
    history = load_history()
    history.append(record)
    save_history(history)

    return jsonify({
        "disease":    result["disease"],
        "confidence": result["confidence"],
        "severity":   result["severity"],
        "top5":       result["top5"],
    }), 200


@app.route("/api/recommendation", methods=["GET"])
def recommendation():
    """Get treatment recommendation for a disease."""
    disease = request.args.get("disease")
    if not disease:
        return jsonify({"error": "disease query param required"}), 400

    rec = get_recommendation(disease)
    if not rec:
        return jsonify({"error": "No recommendation found for this disease"}), 404

    return jsonify(rec), 200


@app.route("/api/history", methods=["GET"])
def history():
    """Return all past detection records."""
    records = load_history()
    return jsonify({"records": records}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
