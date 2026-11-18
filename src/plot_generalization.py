"""Plot the train-vs-held-out generalization comparison.

Usage:
    python src/plot_generalization.py --summary results/rl_generalization_summary.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/rl_generalization_summary.csv")
    parser.add_argument("--out", default="results/plots")
    parser.add_argument("--out-name", default="rl_generalization.png")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)

    policies = ["trained_agent", "random_avg", "oracle_best"]
    colors = {"trained_agent": "#2ca02c", "random_avg": "#ff7f0e", "oracle_best": "#9467bd"}
    labels = {"trained_agent": "trained DQN", "random_avg": "random", "oracle_best": "oracle best"}

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, policy in enumerate(policies):
        ax.bar(x + (i - 1) * width, df[policy], width, label=labels[policy], color=colors[policy])

    image_labels = [
        f"{row.image_id}\n({'HELD OUT' if 'HELD OUT' in row.status else 'trained on'})"
        for row in df.itertuples()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(image_labels)
    ax.set_ylabel("detections found")
    ax.set_title("Generalization check: trained vs. held-out image")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / args.out_name, dpi=150)
    plt.close(fig)

    print(f"Wrote generalization plot to {out_dir}/{args.out_name}")


if __name__ == "__main__":
    main()
