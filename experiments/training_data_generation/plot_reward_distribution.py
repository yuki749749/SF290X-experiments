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
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style, FIGURE_SIZES

apply_style(grid=False)

# Import window extraction utility and planner mapping
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_trajectories import extract_windows, PLANNER_DIRS

DEFAULT_DATA = Path(
    "results/process_trajectories/run/2026-05-13_10-12-10/training_data.pkl"
)


def plot_reward_distribution(data_path: Path, output_path: Path) -> None:
    # Resolve the companion stats file
    prefix = data_path.name.replace("_data.pkl", "")
    stats_path = data_path.parent / f"{prefix}_data_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Companion stats file not found at {stats_path}")
        
    with open(stats_path, "r") as f:
        stats = json.load(f)
        
    horizon = stats.get("horizon", 16)
    stride = stats.get("stride", 4)
    reward_key = stats.get("reward_key", "rmse_history")
    reward_type = stats.get("reward_type", "rmse")
    max_step = stats.get("max_step", 10.0)
    initial_heading = stats.get("initial_heading", 0.7853981633974483)

    dataset = joblib.load(data_path)
    
    # Extract windows from all trajectories
    windowed_dataset = []
    for traj in dataset:
        history_mock = {
            "position_history": traj["positions"],
            "mean_history": traj["means"],
            "variance_history": traj["variances"],
            reward_key: traj["rmse_history"],
        }
        windows = extract_windows(
            history_mock, horizon, stride, reward_key, reward_type, max_step, initial_heading
        )
        for w in windows:
            w["planner"] = traj.get("planner", "unknown")
        windowed_dataset.extend(windows)

    # Reverse mapping for filenames (e.g. "BayesianOptimizationPlanner" -> "bo")
    planner_keys = {tag: name for name, tag in PLANNER_DIRS.items()}

    # Plot separately for each planner
    planners_found = set(w["planner"] for w in windowed_dataset)
    for planner in planners_found:
        planner_rewards = np.array([float(w["r"]) for w in windowed_dataset if w["planner"] == planner], dtype=np.float32)
        if len(planner_rewards) == 0:
            continue
            
        planner_suffix = planner_keys.get(planner, planner.lower())
        
        # Adjust output path name: e.g. my_plot.pdf -> my_plot_bo.pdf
        if output_path.suffix:
            current_output_path = output_path.parent / f"{output_path.stem}_{planner_suffix}{output_path.suffix}"
        else:
            current_output_path = output_path.parent / f"{output_path.name}_{planner_suffix}.pdf"

        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"], constrained_layout=True)
        ax.hist(planner_rewards, bins=60, color="#4393C3", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Reward $r$")
        ax.set_ylabel("Count")
        
        title_name = planner_suffix.replace("_", " ").upper()
        ax.set_title(f"{title_name} Reward Distribution  ($n={len(planner_rewards):,}$)", fontsize=10)

        percentiles = [50, 90, 95]
        colors      = ["#D55E00", "#009E73", "#0072B2"]
        pct_values  = np.percentile(planner_rewards, percentiles)

        for p, c, v in zip(percentiles, colors, pct_values):
            ax.axvline(v, color=c, linestyle="--", linewidth=1.0, label=f"p{p} = {v:.3f}")

        ax.axvline(planner_rewards.mean(), color="black", linestyle=":", linewidth=1.0,
                   label=f"mean = {planner_rewards.mean():.3f}")
        ax.legend(frameon=False)

        print(f"\n--- {title_name} Reward Percentiles ---")
        print(f"  {'mean':<6}: {planner_rewards.mean():.4f}")
        for p, v in zip(percentiles, pct_values):
            print(f"  p{p:<5}: {v:.4f}")

        fig.savefig(current_output_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {current_output_path}")
        plt.close(fig)


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
