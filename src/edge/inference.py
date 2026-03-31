"""
inference.py — Edge Device Inference (Raspberry Pi)
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection

Uses TFLite model for fast, low-power on-device inference.

Run:
    python src/edge/inference.py --model models/disease_model.tflite
    python src/edge/inference.py --model models/disease_model.tflite --image data/test_leaf.jpg
"""

import argparse
import time
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


IMG_SIZE = (224, 224)


def load_tflite_model(model_path: str):
    """Load TFLite interpreter."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image for TFLite input."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_inference(interpreter, image_path: str, class_names: list) -> dict:
    """Run inference on a single image. Returns result dict."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = preprocess_image(image_path)

    # Set input
    interpreter.set_tensor(input_details[0]["index"], img)

    # Run inference — timed
    t0 = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Get output
    probs   = interpreter.get_tensor(output_details[0]["index"])[0]
    top_idx = int(np.argmax(probs))

    return {
        "disease":      class_names[top_idx],
        "confidence":   float(probs[top_idx]),
        "inference_ms": round(elapsed_ms, 2),
        "top5": [
            (class_names[i], float(probs[i]))
            for i in np.argsort(probs)[::-1][:5]
        ],
    }


def convert_to_tflite(keras_model_path: str, output_path: str = "models/disease_model.tflite"):
    """Convert Keras .h5 model to TFLite for edge deployment."""
    import tensorflow as tf
    model      = tf.keras.models.load_model(keras_model_path)
    converter  = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimize for edge (size + speed)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"[INFO] TFLite model saved: {output_path} ({size_kb:.1f} KB)")
    return output_path


def camera_loop(interpreter, class_names: list, camera_index: int = 0):
    """
    Continuous inference loop using PiCamera / USB camera.
    Captures frames and runs inference at ~1 fps.
    """
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV not found. Install: pip install opencv-python")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Camera loop started. Press Ctrl+C to stop.")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Save temp frame
            tmp_path = "/tmp/frame.jpg"
            cv2.imwrite(tmp_path, frame)

            result = run_inference(interpreter, tmp_path, class_names)
            frame_count += 1

            print(
                f"[Frame {frame_count:04d}] "
                f"{result['disease']:<45} "
                f"Conf: {result['confidence'] * 100:.1f}%  "
                f"Time: {result['inference_ms']:.0f} ms"
            )
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Camera loop stopped.")
    finally:
        cap.release()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Inference — Crop Disease Detection")
    parser.add_argument("--model",   required=True,             help="Path to .tflite model")
    parser.add_argument("--classes", default="models/class_names.npy", help="Path to class names .npy")
    parser.add_argument("--image",   default=None,              help="Single image path (optional)")
    parser.add_argument("--camera",  action="store_true",       help="Run camera inference loop")
    parser.add_argument("--convert", default=None,              help="Convert .h5 model to TFLite")
    args = parser.parse_args()

    # Convert mode
    if args.convert:
        convert_to_tflite(args.convert, args.model)
        raise SystemExit(0)

    # Load
    import os
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("        Run convert first: python inference.py --convert models/disease_model.h5 --model models/disease_model.tflite")
        raise SystemExit(1)

    interpreter = load_tflite_model(args.model)
    class_names = np.load(args.classes, allow_pickle=True).tolist() if os.path.exists(args.classes) else [str(i) for i in range(38)]

    print(f"[INFO] Model   : {args.model}")
    print(f"[INFO] Classes : {len(class_names)}")

    if args.camera:
        camera_loop(interpreter, class_names)
    elif args.image:
        result = run_inference(interpreter, args.image, class_names)
        print(f"\nDisease     : {result['disease']}")
        print(f"Confidence  : {result['confidence'] * 100:.1f}%")
        print(f"Inference   : {result['inference_ms']} ms")
        print("\nTop 5:")
        for name, prob in result["top5"]:
            print(f"  {name:<45} {prob * 100:.1f}%")
    else:
        print("[INFO] Specify --image <path> or --camera to run inference.")
