"""Plot the leave-one-scene-out generalization results: per held-out scene,
trained agent vs. oracle best vs. random, summed across that scene's base
frame + 4 synthetic variants.

Usage:
    python src/plot_loso.py
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
    df = pd.read_csv("results/rl_generalization_loso_combined.csv")
    per_scene = df.groupby("held_out_scene")[["trained_agent", "oracle_best", "random_avg"]].sum()
    per_scene = per_scene.reindex(BASES)

    x = np.arange(len(BASES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(x - width, per_scene["trained_agent"], width, label="trained DQN (held out)", color="#2ca02c")
    ax.bar(x, per_scene["oracle_best"], width, label="oracle best (ceiling)", color="#9467bd")
    ax.bar(x + width, per_scene["random_avg"], width, label="random policy", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[b] for b in BASES])
    ax.set_ylabel("detections found (summed over base + 4 synthetic variants)")
    ax.set_title("Leave-one-scene-out generalization: 4 independent held-out tests")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    out_dir = Path("results/plots_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rl_generalization_loso.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir}/rl_generalization_loso.png")


if __name__ == "__main__":
    main()
