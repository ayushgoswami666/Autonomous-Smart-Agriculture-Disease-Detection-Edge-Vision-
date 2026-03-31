"""
train.py — CNN Disease Detection Model Training
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt


# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 38          # PlantVillage dataset — 38 disease classes
MODEL_DIR   = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train crop disease detection CNN")
    parser.add_argument("--dataset",  type=str, default="data/processed/", help="Path to dataset directory")
    parser.add_argument("--epochs",   type=int, default=20,                help="Number of training epochs")
    parser.add_argument("--lr",       type=float, default=0.001,           help="Learning rate")
    parser.add_argument("--model",    type=str, default="mobilenetv2",     help="Base model: mobilenetv2 | custom")
    parser.add_argument("--output",   type=str, default="models/disease_model.h5")
    return parser.parse_args()


# ── Data Generators ───────────────────────────────────────────────────────────
def build_data_generators(dataset_path: str):
    """Return train and validation ImageDataGenerators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    return train_gen, val_gen


# ── Model Builders ────────────────────────────────────────────────────────────
def build_mobilenetv2(num_classes: int) -> tf.keras.Model:
    """Transfer learning with MobileNetV2 base."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False  # Freeze base initially

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_custom_cnn(num_classes: int) -> tf.keras.Model:
    """Lightweight custom CNN for edge devices."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print(f"\n[INFO] Loading dataset from: {args.dataset}")
    train_gen, val_gen = build_data_generators(args.dataset)

    num_classes = train_gen.num_classes
    print(f"[INFO] Classes detected: {num_classes}")
    print(f"[INFO] Training samples : {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}")

    # Build model
    if args.model == "mobilenetv2":
        model = build_mobilenetv2(num_classes)
        print("[INFO] Using MobileNetV2 (transfer learning)")
    else:
        model = build_custom_cnn(num_classes)
        print("[INFO] Using custom CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    # Train
    print(f"\n[INFO] Training for {args.epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n[RESULT] Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"[RESULT] Validation Loss     : {val_loss:.4f}")
    print(f"[INFO]   Model saved to      : {args.output}")

    # Save class labels
    class_names = list(train_gen.class_indices.keys())
    np.save(os.path.join(MODEL_DIR, "class_names.npy"), class_names)
    print(f"[INFO]   Class names saved   : {MODEL_DIR}class_names.npy")

    _plot_history(history)
    return model, history


def _plot_history(history):
    """Save accuracy/loss curves to models/training_plot.png."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_plot.png"))
    print(f"[INFO]   Training plot saved: {MODEL_DIR}training_plot.png")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
