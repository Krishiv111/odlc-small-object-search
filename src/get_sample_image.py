"""Grab a test image to validate the tiling harness.

Tries SAHI's own tutorial image (a real aerial-style photo with small vehicles,
public domain, used across their docs) first. Falls back to a synthetic image
with small squares scattered on a noisy background, just so the pipeline has
something to run on if there's no internet access.

This is a stand-in only -- swap data/sample.jpg for the team's real flight
imagery (or a VisDrone/DOTA crop) as soon as that's available.
"""
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SAHI_DEMO_URL = (
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg"
)
OUT_PATH = Path("data/sample.jpg")


def try_download() -> bool:
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAHI_DEMO_URL, OUT_PATH)
        return True
    except Exception as e:
        print(f"Download failed ({e}), falling back to a synthetic image.")
        return False


def make_synthetic(width=1920, height=1080, n_targets=25, seed=42):
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 160, size=(height, width, 3), dtype=np.uint8)

    for _ in range(n_targets):
        size = rng.integers(6, 20)
        x = rng.integers(0, width - size)
        y = rng.integers(0, height - size)
        color = rng.integers(0, 60, size=3)
        img[y : y + size, x : x + size] = color

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(OUT_PATH, quality=90)
    print(f"Wrote synthetic test image with {n_targets} small targets to {OUT_PATH}")


if __name__ == "__main__":
    if not try_download():
        make_synthetic()
    else:
        print(f"Downloaded sample image to {OUT_PATH}")
