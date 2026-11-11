"""Precompute per-tile detection counts for a fixed non-overlapping grid.

This is the "oracle" the RL env uses as ground truth reward: run the
detector on every tile once, cache how many detections each tile holds, so
env.step() can look up a reward instead of re-running YOLO every RL step
(that would make training painfully slow for no benefit on a fixed image).

Usage:
    python src/build_oracle.py --image data/sample.jpg --tile-size 320 \
        --out results/tile_oracle.csv
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_prediction

from tile_grid import build_grid, crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/sample.jpg")
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--out", default="results/tile_oracle.csv")
    args = parser.parse_args()

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolov8n.pt",
        confidence_threshold=args.confidence,
        device="cpu",
    )

    tiles = build_grid(args.image, tile_size=args.tile_size)
    rows = []
    for tile in tiles:
        tile_img = crop(args.image, tile)
        tmp_path = Path("data/_tile_tmp.jpg")
        cv2.imwrite(str(tmp_path), tile_img)
        result = get_prediction(str(tmp_path), model)
        n = len(result.object_prediction_list)
        rows.append(
            {"tile_id": tile.tile_id, "x": tile.x, "y": tile.y,
             "w": tile.w, "h": tile.h, "num_detections": n}
        )
        print(f"tile {tile.tile_id} ({tile.x},{tile.y}): {n} detections")
    tmp_path.unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} tile rows to {args.out}")
    print(f"Total detections across all tiles (grid, no overlap): {df['num_detections'].sum()}")


if __name__ == "__main__":
    main()
