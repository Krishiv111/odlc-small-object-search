"""Aggregate the 4 leave-one-scene-out generalization runs into one summary:
for each run, only the HELD OUT rows matter (that's the actual generalization
test); the "seen" rows are just confirming training worked.

Usage:
    python src/aggregate_loso.py
"""
from pathlib import Path

import pandas as pd

BASES = ["frame_multi", "frame_mannequin_tent", "frame_single_tent", "frame_tent_closeup"]


def main():
    all_held_out = []
    for base in BASES:
        df = pd.read_csv(f"results/rl_generalization_loso_{base}.csv")
        held = df[df["status"].str.contains("HELD OUT")].copy()
        held["held_out_scene"] = base
        all_held_out.append(held)

    combined = pd.concat(all_held_out, ignore_index=True)
    combined.to_csv("results/rl_generalization_loso_combined.csv", index=False)

    print("Per held-out-scene totals (agent / oracle best / random avg, summed across that scene's "
          "base frame + 4 synthetic variants):\n")
    per_scene = combined.groupby("held_out_scene")[["trained_agent", "oracle_best", "random_avg"]].sum()
    print(per_scene.to_string())

    totals = combined[["trained_agent", "oracle_best", "random_avg"]].sum()
    pct_of_oracle = 100 * totals["trained_agent"] / totals["oracle_best"]
    pct_random = 100 * totals["random_avg"] / totals["oracle_best"]
    print(f"\nOverall across all 4 held-out scenes (20 held-out frames total):")
    print(f"  trained agent: {totals['trained_agent']:.0f} / oracle best: {totals['oracle_best']:.0f} "
          f"({pct_of_oracle:.0f}% of oracle) / random avg: {totals['random_avg']:.1f} ({pct_random:.0f}% of oracle)")

    Path("results").mkdir(exist_ok=True)
    print(f"\nSaved combined held-out rows to results/rl_generalization_loso_combined.csv")


if __name__ == "__main__":
    main()
