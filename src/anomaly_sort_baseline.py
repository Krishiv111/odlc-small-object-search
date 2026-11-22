"""No-RL baseline: for each image, just sort tiles by anomaly score and
check the top `budget` of them. No training, no learning, no memory of
other images -- pure per-image heuristic.

This exists to answer an honest question: once tiles have a good anomaly
score, is the trained RL agent actually adding value over just sorting by
that score directly? If this baseline matches the trained agent, the real
fix was the features, not the RL -- worth knowing either way.

Usage:
    python src/anomaly_sort_baseline.py --oracle results/tile_oracle_multi_real_anomaly.csv --budget 5
"""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="results/tile_oracle_multi_real_anomaly.csv")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--out", default="results/anomaly_sort_baseline.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.oracle)
    rows = []
    for image_id, group in df.groupby("image_id"):
        top_tiles = group.sort_values("color_anomaly", ascending=False).head(args.budget)
        found = float(top_tiles["num_detections"].sum())
        oracle_best = float(group["num_detections"].nlargest(args.budget).sum())
        rows.append({"image_id": image_id, "anomaly_sort_found": found, "oracle_best": oracle_best})
        print(f"[{image_id}] anomaly-sort found {found:.0f} / oracle best {oracle_best:.0f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    total_found = out_df["anomaly_sort_found"].sum()
    total_best = out_df["oracle_best"].sum()
    print(f"\nTotal: anomaly-sort found {total_found:.0f} / oracle best {total_best:.0f} "
          f"({100 * total_found / total_best:.0f}%)")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
