"""
preprocess.py — Dataset Preprocessing Script
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection

Prepares PlantVillage dataset for CNN training:
  - Resize images to 224x224
  - Augment minority classes
  - Split into train/val
  - Save processed images

Usage:
    python scripts/preprocess.py --input data/raw/ --output data/processed/
"""

import os
import argparse
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np


IMG_SIZE     = (224, 224)
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/raw/",       help="Raw dataset path")
    p.add_argument("--output", default="data/processed/", help="Output processed path")
    p.add_argument("--augment", action="store_true",      help="Apply augmentation to minority classes")
    return p.parse_args()


def augment_image(img: Image.Image) -> Image.Image:
    """Apply random augmentation to a PIL image."""
    ops = [
        lambda x: x.rotate(random.randint(-30, 30)),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.0))),
    ]
    for op in random.sample(ops, k=random.randint(1, 3)):
        img = op(img)
    return img


def process_class(class_dir: Path, output_class_dir: Path, augment: bool = False):
    """Process all images in a class directory."""
    output_class_dir.mkdir(parents=True, exist_ok=True)
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
             list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))

    count = 0
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
            out_path = output_class_dir / img_path.name
            img.save(out_path, "JPEG", quality=90)
            count += 1

            # Augment if requested
            if augment:
                for i in range(2):
                    aug_img  = augment_image(img)
                    aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                    aug_img.save(output_class_dir / aug_name, "JPEG", quality=85)
                    count += 1
        except Exception as e:
            print(f"  [WARN] Skipping {img_path.name}: {e}")

    return count


def preprocess(args):
    input_root  = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}")
        print("        Download PlantVillage dataset and place it in data/raw/")
        return

    class_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"[INFO] Found {len(class_dirs)} classes in {input_root}")

    total = 0
    for cls_dir in class_dirs:
        out_dir = output_root / cls_dir.name
        n = process_class(cls_dir, out_dir, augment=args.augment)
        total += n
        print(f"  ✅  {cls_dir.name:<50} → {n:>5} images")

    print(f"\n[DONE] Total processed images : {total}")
    print(f"[DONE] Output directory       : {output_root}")


if __name__ == "__main__":
    preprocess(parse_args())
