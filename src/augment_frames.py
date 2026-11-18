"""Synthetically expand real-frame variety for the generalization test.

We only have 4 clean real frames, all from one flight pass seconds apart --
not enough variety to claim the agent "generalizes." This script makes 4
augmented variants of each real frame (brightness up/down, small rotation,
horizontal flip) so training sees more lighting/orientation diversity per
scene, while the held-out test in train_rl_generalization.py still holds out
an entire real scene (base frame + all its variants) it never saw in any
form.

Usage:
    python src/augment_frames.py
"""
from pathlib import Path

import cv2
import numpy as np

BASE_FRAMES = {
    "frame_multi": "data/real/tent_mannequin_multi.jpg",
    "frame_mannequin_tent": "data/real/mannequin_tent.jpg",
    "frame_single_tent": "data/real/single_tent.jpg",
    "frame_tent_closeup": "data/real/tent_closeup.jpg",
}

OUT_DIR = Path("data/real/augmented")


def brightness(img, delta):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=delta)


def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def flip_horizontal(img):
    return cv2.flip(img, 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for base_id, path in BASE_FRAMES.items():
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)

        variants = {
            "bright": brightness(img, 35),
            "dark": brightness(img, -35),
            "rot": rotate(img, 6),
            "flip": flip_horizontal(img),
        }

        for variant_name, variant_img in variants.items():
            out_path = OUT_DIR / f"{base_id}_{variant_name}.jpg"
            cv2.imwrite(str(out_path), variant_img)
            manifest.append((f"{base_id}_{variant_name}", str(out_path)))
            print(f"  wrote {out_path}")

    print(f"\n{len(manifest)} synthetic variants written to {OUT_DIR}")


if __name__ == "__main__":
    main()
