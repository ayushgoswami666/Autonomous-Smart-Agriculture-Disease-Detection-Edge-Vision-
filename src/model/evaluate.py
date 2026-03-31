"""
evaluate.py — Model Evaluation Script
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection

Generates:
  - Validation accuracy / loss
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix saved to docs/confusion_matrix.png

Usage:
    python src/model/evaluate.py --model models/disease_model.h5 --dataset data/processed/
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


IMG_SIZE   = (224, 224)
BATCH_SIZE = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="models/disease_model.h5")
    p.add_argument("--dataset", default="data/processed/")
    p.add_argument("--output",  default="docs/")
    return p.parse_args()


def evaluate(args):
    print(f"[INFO] Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    # Validation generator
    val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2).flow_from_directory(
        args.dataset,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    # Evaluate
    print("[INFO] Evaluating on validation set...")
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"\n[RESULT] Validation Accuracy : {acc * 100:.2f}%")
    print(f"[RESULT] Validation Loss     : {loss:.4f}")

    # Predictions
    val_gen.reset()
    preds       = model.predict(val_gen, verbose=1)
    y_pred      = np.argmax(preds, axis=1)
    y_true      = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    # Classification report
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    os.makedirs(args.output, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Greens",
    )
    plt.title("Confusion Matrix — Crop Disease Detection", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(args.output, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=100)
    print(f"[INFO] Confusion matrix saved: {cm_path}")


if __name__ == "__main__":
    evaluate(parse_args())
