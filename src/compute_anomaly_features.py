"""Turn raw per-tile color/edge stats into anomaly scores: how different is
this tile from what's "typical" for THIS image, instead of raw absolute
numbers. Fixes the flaw raw color features have for SUAS specifically --
raw mean_r/g/b only means something if we've seen that exact color before
in training, but we don't know what color the real competition tent will
be. An anomaly score doesn't care what the color is, only whether it
stands out from its own image's background.

No new YOLO inference needed -- this reads an oracle CSV that already has
mean_r/g/b and edge_density per tile (from build_oracle_multi.py) and adds
two new columns computed from those.

Usage:
    python src/compute_anomaly_features.py --in results/tile_oracle_multi_real.csv \
        --out results/tile_oracle_multi_real_anomaly.csv
"""
import argparse

import numpy as np
import pandas as pd

COLOR_COLS = ["mean_r", "mean_g", "mean_b"]


def add_anomaly_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["color_anomaly"] = 0.0
    df["edge_anomaly"] = 0.0

    for image_id, group in df.groupby("image_id"):
        # Median, not mean: a couple of anomalous (target) tiles shouldn't
        # drag the "typical background" estimate toward them.
        bg_color = group[COLOR_COLS].median().values
        bg_edge = group["edge_density"].median()

        color_dist = np.linalg.norm(group[COLOR_COLS].values - bg_color, axis=1)
        edge_dist = (group["edge_density"] - bg_edge).abs()

        df.loc[group.index, "color_anomaly"] = color_dist
        df.loc[group.index, "edge_anomaly"] = edge_dist

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="results/tile_oracle_multi_real.csv")
    parser.add_argument("--out", default="results/tile_oracle_multi_real_anomaly.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df = add_anomaly_columns(df)
    df.to_csv(args.out, index=False)

    hit = df[df.num_detections > 0]
    miss = df[df.num_detections == 0]
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"\nSanity check -- does the anomaly score actually separate hit vs miss tiles better than raw color did?")
    print(f"  tiles WITH a real detection (n={len(hit)}): "
          f"color_anomaly={hit.color_anomaly.mean():.4f}  edge_anomaly={hit.edge_anomaly.mean():.4f}")
    print(f"  tiles WITHOUT a detection    (n={len(miss)}): "
          f"color_anomaly={miss.color_anomaly.mean():.4f}  edge_anomaly={miss.edge_anomaly.mean():.4f}")


if __name__ == "__main__":
    main()
