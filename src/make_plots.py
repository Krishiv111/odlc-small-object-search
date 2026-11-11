"""Turn results/baseline.csv into the plots used in RESULTS.md.

Usage:
    python src/make_plots.py --csv results/baseline.csv --out results/plots
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def label_for(row) -> str:
    if row["method"] == "full_image":
        return "full image"
    return f"{int(row['slice_size'])}px / {row['overlap']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/baseline.csv")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["label"] = df.apply(label_for, axis=1)
    df = df.sort_values("num_tiles").reset_index(drop=True)

    colors = ["#d62728" if m == "full_image" else "#1f77b4" for m in df["method"]]

    # Detections vs. tiles checked
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["num_tiles"], df["num_detections"], c=colors, s=70, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(
            row["label"],
            (row["num_tiles"], row["num_detections"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("tiles checked")
    ax.set_ylabel("detections found")
    ax.set_title("Detections vs. tiles checked (1 test image)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "detections_vs_tiles.png", dpi=150)
    plt.close(fig)

    # Detections vs. time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["time_sec"], df["num_detections"], c=colors, s=70, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(
            row["label"],
            (row["time_sec"], row["num_detections"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("inference time (s, CPU)")
    ax.set_ylabel("detections found")
    ax.set_title("Detections vs. inference time (1 test image)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "detections_vs_time.png", dpi=150)
    plt.close(fig)

    # Bar chart: detections per config
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["label"], df["num_detections"], color=colors)
    ax.set_ylabel("detections found")
    ax.set_title("Detections by method/config (1 test image)")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "detections_by_config.png", dpi=150)
    plt.close(fig)

    print(f"Wrote 3 plots to {out_dir}/")


if __name__ == "__main__":
    main()
