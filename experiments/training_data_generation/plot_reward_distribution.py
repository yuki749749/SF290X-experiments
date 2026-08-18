"""
plot_reward_distribution.py
============================
Plots the reward distribution of the processed training dataset.

Usage
-----
    python plot_reward_distribution.py
    python plot_reward_distribution.py \\
        --data results/process_trajectories/run/2026-05-13_10-12-10/training_data.pkl
    python plot_reward_distribution.py \\
        --data path/to/training_data.pkl --output my_plot.pdf

Output
------
    <data_dir>/reward_distribution.pdf   (default)
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style, FIGURE_SIZES

apply_style(grid=False)

DEFAULT_DATA = Path(
    "results/process_trajectories/run/2026-05-13_10-12-10/training_data.pkl"
)


def plot_reward_distribution(data_path: Path, output_path: Path) -> None:
    dataset = joblib.load(data_path)
    rewards = np.array([float(d["r"]) for d in dataset], dtype=np.float32)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"], constrained_layout=True)
    ax.hist(rewards, bins=60, color="#4393C3", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Reward $r$")
    ax.set_ylabel("Count")
    ax.set_title(f"Reward distribution  ($n={len(rewards):,}$)", fontsize=10)

    percentiles = [50, 90, 95]
    colors      = ["#D55E00", "#009E73", "#0072B2"]
    pct_values  = np.percentile(rewards, percentiles)

    for p, c, v in zip(percentiles, colors, pct_values):
        ax.axvline(v, color=c, linestyle="--", linewidth=1.0, label=f"p{p} = {v:.3f}")

    ax.axvline(rewards.mean(), color="black", linestyle=":", linewidth=1.0,
               label=f"mean = {rewards.mean():.3f}")
    ax.legend(frameon=False)

    print("\n--- Reward Percentiles ---")
    print(f"  {'mean':<6}: {rewards.mean():.4f}")
    for p, v in zip(percentiles, pct_values):
        print(f"  p{p:<5}: {v:.4f}")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot reward distribution of the training dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to training_data.pkl (default: latest processed run).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <data_dir>/reward_distribution.pdf).",
    )
    args = parser.parse_args()

    output_path = args.output or (args.data.parent / "reward_distribution.pdf")

    print(f"Data   : {args.data}")
    print(f"Output : {output_path}")

    plot_reward_distribution(args.data, output_path)


if __name__ == "__main__":
    main()
