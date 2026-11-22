"""Compare raw-color features vs. anomaly features on the same
leave-one-scene-out generalization test, plus the no-RL anomaly-sort
baseline, so we can tell whether the feature swap (and the RL training on
top of it) actually helped.

Usage:
    python src/compare_feature_sets.py
"""
from pathlib import Path

import pandas as pd

BASES = ["frame_multi", "frame_mannequin_tent", "frame_single_tent", "frame_tent_closeup"]


def held_out_rows(prefix: str) -> pd.DataFrame:
    rows = []
    for base in BASES:
        df = pd.read_csv(f"results/{prefix}_{base}.csv")
        held = df[df["status"].str.contains("HELD OUT")].copy()
        held["held_out_scene"] = base
        rows.append(held)
    return pd.concat(rows, ignore_index=True)


def main():
    raw = held_out_rows("rl_generalization_loso")
    anomaly = held_out_rows("rl_generalization_loso_anomaly")
    anomaly_sort = pd.read_csv("results/anomaly_sort_baseline.csv")

    per_scene = pd.DataFrame({
        "raw_agent": raw.groupby("held_out_scene")["trained_agent"].sum(),
        "anomaly_agent": anomaly.groupby("held_out_scene")["trained_agent"].sum(),
        "oracle_best": raw.groupby("held_out_scene")["oracle_best"].sum(),
    }).reindex(BASES)
    per_scene.to_csv("results/feature_comparison_per_scene.csv")

    print("Per held-out scene (agent score / oracle best):")
    print(per_scene.to_string())

    raw_total, raw_best = raw.trained_agent.sum(), raw.oracle_best.sum()
    anomaly_total, anomaly_best = anomaly.trained_agent.sum(), anomaly.oracle_best.sum()
    sort_total, sort_best = anomaly_sort.anomaly_sort_found.sum(), anomaly_sort.oracle_best.sum()

    summary = pd.DataFrame([
        {"method": "raw features + RL", "found": raw_total, "oracle_best": raw_best,
         "pct_of_oracle": 100 * raw_total / raw_best},
        {"method": "anomaly features + RL", "found": anomaly_total, "oracle_best": anomaly_best,
         "pct_of_oracle": 100 * anomaly_total / anomaly_best},
        {"method": "anomaly features, no RL (just sort)", "found": sort_total, "oracle_best": sort_best,
         "pct_of_oracle": 100 * sort_total / sort_best},
    ])
    summary.to_csv("results/feature_comparison_summary.csv", index=False)
    print("\nOverall:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
