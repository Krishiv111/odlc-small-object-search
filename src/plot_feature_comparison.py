"""Plot: raw features + RL vs. anomaly features + RL vs. anomaly features
with no RL (just sort), across the 4 held-out leave-one-scene-out tests.

Usage:
    python src/plot_feature_comparison.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASES = ["frame_multi", "frame_mannequin_tent", "frame_single_tent", "frame_tent_closeup"]
DISPLAY_NAMES = {
    "frame_multi": "multi\n(2 tents+1 mannequin)",
    "frame_mannequin_tent": "mannequin+tent",
    "frame_single_tent": "single tent",
    "frame_tent_closeup": "tent closeup",
}


def main():
    per_scene = pd.read_csv("results/feature_comparison_per_scene.csv", index_col=0).reindex(BASES)
    summary = pd.read_csv("results/feature_comparison_summary.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(BASES))
    width = 0.3
    ax1.bar(x - width, per_scene["raw_agent"], width, label="raw features + RL", color="#ff7f0e")
    ax1.bar(x, per_scene["anomaly_agent"], width, label="anomaly features + RL", color="#2ca02c")
    ax1.bar(x + width, per_scene["oracle_best"], width, label="oracle best (ceiling)", color="#9467bd")
    ax1.set_xticks(x)
    ax1.set_xticklabels([DISPLAY_NAMES[b] for b in BASES])
    ax1.set_ylabel("detections found (held-out scene + variants)")
    ax1.set_title("Per held-out scene: raw vs. anomaly features")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    ax2.bar(summary["method"], summary["pct_of_oracle"], color=colors)
    ax2.set_ylabel("% of oracle-best found (held-out totals)")
    ax2.set_title("Does RL training add value over the feature alone?")
    ax2.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax2.grid(alpha=0.3, axis="y")
    ax2.axhline(21, color="gray", linestyle="--", linewidth=1, label="random policy (~21%)")
    ax2.legend()

    fig.tight_layout()
    out_dir = Path("results/plots_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "feature_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir}/feature_comparison.png")


if __name__ == "__main__":
    main()
