"""Plot the RL training curve and the policy comparison bar chart.

Usage:
    python src/plot_rl.py --training results/rl_training.csv \
        --summary results/rl_training_summary.csv --out results/plots
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="results/rl_training.csv")
    parser.add_argument("--summary", default="results/rl_training_summary.csv")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.training)
    summary_df = pd.read_csv(args.summary)

    # Training curve: reward per episode, with rolling average.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_df["episode"], train_df["reward"], alpha=0.3, color="#1f77b4", label="episode reward")
    rolling = train_df["reward"].rolling(window=15, min_periods=1).mean()
    ax.plot(train_df["episode"], rolling, color="#1f77b4", linewidth=2, label="rolling avg (15 ep)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("detections found (4-tile budget)")
    ax.set_title("DQN tile-selection agent: training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rl_training_curve.png", dpi=150)
    plt.close(fig)

    # Policy comparison bar chart.
    colors = {"trained_dqn": "#2ca02c", "random": "#ff7f0e", "oracle_best": "#9467bd", "full_exhaustive": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        summary_df["policy"],
        summary_df["detections"],
        color=[colors.get(p, "#1f77b4") for p in summary_df["policy"]],
    )
    for bar, row in zip(bars, summary_df.itertuples()):
        ax.annotate(
            f"{row.detections:.0f} det / {row.tiles_checked} tiles",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("detections found")
    ax.set_title("Tile-selection policy comparison (4-tile budget vs. full grid)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "rl_policy_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Wrote 2 RL plots to {out_dir}/")


if __name__ == "__main__":
    main()
