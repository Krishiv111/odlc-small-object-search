"""Precompute per-tile detections + features across MULTIPLE images, on a
shared grid, for the generalization experiment (train on some images, test
on a held-out one the agent never saw).

Usage:
    python src/build_oracle_multi.py --out results/tile_oracle_multi.csv
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_prediction

from multi_image_grid import build_grid, crop, load_resized, tile_features

IMAGES = {
    "vehicles_lot": "data/sample.jpg",
    "desert_convoy": "data/multi/terrain2.png",
    "marina_boats": "data/multi/obb_test_image.png",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--out", default="results/tile_oracle_multi.csv")
    args = parser.parse_args()

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolov8n.pt",
        confidence_threshold=args.confidence,
        device="cpu",
    )

    tiles = build_grid()
    rows = []
    tmp_path = Path("data/_tile_tmp.jpg")

    for image_id, path in IMAGES.items():
        img = load_resized(path)
        print(f"\n{image_id} ({path}):")
        for tile in tiles:
            tile_img = crop(img, tile)
            cv2.imwrite(str(tmp_path), tile_img)
            result = get_prediction(str(tmp_path), model)
            n = len(result.object_prediction_list)
            feats = tile_features(tile_img)
            rows.append(
                {
                    "image_id": image_id,
                    "tile_id": tile.tile_id,
                    "x": tile.x, "y": tile.y, "w": tile.w, "h": tile.h,
                    "num_detections": n,
                    "mean_r": feats[0], "mean_g": feats[1], "mean_b": feats[2],
                    "edge_density": feats[3],
                }
            )
            print(f"  tile {tile.tile_id}: {n} detections")
    tmp_path.unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nSaved {len(df)} rows across {len(IMAGES)} images to {args.out}")
    for image_id in IMAGES:
        total = df[df.image_id == image_id]["num_detections"].sum()
        print(f"  {image_id}: {total} total detections across grid")


if __name__ == "__main__":
    main()
