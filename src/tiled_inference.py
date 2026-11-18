"""Baseline experiment: exhaustive sliced inference (SAHI) vs. full-image inference.

This is the Week-1 baseline for the small-object-at-altitude project: before any
RL search policy, we need numbers for the "check every tile" approach so there's
something to compare against later.

Usage:
    python src/tiled_inference.py --image data/sample.jpg --out results/baseline.csv
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions


def build_model(confidence_threshold: float = 0.25, device: str = "cpu", model_path: str = "yolov8n.pt"):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )


def run_full_image(model, image_path: str):
    t0 = time.perf_counter()
    result = get_prediction(image_path, model)
    elapsed = time.perf_counter() - t0
    return {
        "method": "full_image",
        "slice_size": None,
        "overlap": None,
        "num_detections": len(result.object_prediction_list),
        "num_tiles": 1,
        "time_sec": round(elapsed, 4),
    }, result


def run_sliced(model, image_path: str, slice_size: int, overlap: float):
    t0 = time.perf_counter()
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    step = max(1, int(slice_size * (1 - overlap)))
    num_tiles_x = max(1, -(-max(w - slice_size, 0) // step) + 1)
    num_tiles_y = max(1, -(-max(h - slice_size, 0) // step) + 1)
    num_tiles = num_tiles_x * num_tiles_y

    return {
        "method": "sliced",
        "slice_size": slice_size,
        "overlap": overlap,
        "num_detections": len(result.object_prediction_list),
        "num_tiles": num_tiles,
        "time_sec": round(elapsed, 4),
    }, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="results/baseline.csv")
    parser.add_argument("--slice-sizes", default="320,512,640")
    parser.add_argument("--overlaps", default="0.1,0.2")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vis_dir = Path("results/vis")
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(confidence_threshold=args.confidence, model_path=args.model)

    rows = []

    row, result = run_full_image(model, args.image)
    rows.append(row)
    print(f"[full_image] detections={row['num_detections']} time={row['time_sec']}s")
    if args.save_vis:
        visualize_object_predictions(
            cv2.imread(args.image), object_prediction_list=result.object_prediction_list,
            output_dir=str(vis_dir), file_name="full_image", export_format="jpg",
        )

    slice_sizes = [int(s) for s in args.slice_sizes.split(",")]
    overlaps = [float(o) for o in args.overlaps.split(",")]

    for slice_size in slice_sizes:
        for overlap in overlaps:
            row, result = run_sliced(model, args.image, slice_size, overlap)
            rows.append(row)
            print(
                f"[sliced size={slice_size} overlap={overlap}] "
                f"detections={row['num_detections']} tiles={row['num_tiles']} time={row['time_sec']}s"
            )
            if args.save_vis:
                tag = f"sliced_{slice_size}_{overlap}"
                visualize_object_predictions(
                    cv2.imread(args.image), object_prediction_list=result.object_prediction_list,
                    output_dir=str(vis_dir), file_name=tag, export_format="jpg",
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
