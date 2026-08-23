"""
plot_data_characterization.py
==============================
Density plot characterizing the training data distribution.

Loads the processed training data .pkl and plots GP posterior trace
(sum of b_var over the evaluation grid) vs reward for each window.

Usage
-----
    python plot_data_characterization.py
    python plot_data_characterization.py \\
        --data results/process_trajectories/run/2026-05-13_10-12-10/training_data.pkl
    python plot_data_characterization.py \\
        --data path/to/training_data.pkl --output my_plot.pdf

Output
------
    <data_dir>/trace_vs_reward.pdf   (default)
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from plot_style import apply_style, FIGURE_SIZES
from utils import find_latest_run_dir

apply_style(grid=False)

# Import window extraction utility
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_trajectories import extract_windows

# Auto-detect latest processed run directory
try:
    DEFAULT_DATA_DIR = find_latest_run_dir(
        Path(__file__).resolve().parent.parent.parent / "results",
        "process_trajectories",
        "training_data.pkl",
    )
    DEFAULT_DATA = DEFAULT_DATA_DIR / "training_data.pkl"
except FileNotFoundError:
    DEFAULT_DATA = Path("results/process_trajectories/run/2026-05-13_10-12-10/training_data.pkl")


def load_arrays(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
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
            "trace_history": traj.get("trace_history"),
        }
        windows = extract_windows(
            history_mock, horizon, stride, reward_key, reward_type, max_step, initial_heading
        )
        windowed_dataset.extend(windows)

    trace  = np.array([w["b_var"].sum() for w in windowed_dataset], dtype=np.float32)
    reward = np.array([float(w["r"])    for w in windowed_dataset], dtype=np.float32)
    return trace, reward


def plot_trace_vs_reward(
    trace: np.ndarray,
    reward: np.ndarray,
    output_path: Path,
    gridsize: int = 45,
) -> None:
    # Filter for positive rewards to prevent hexbin shape stretching
    mask = reward >= 0
    trace = trace[mask]
    reward = reward[mask]

    corr  = np.corrcoef(trace, reward)[0, 1]
    n     = len(trace)
    scale = 10 ** int(np.floor(np.log10(trace.mean())))
    trace_scaled = trace / scale

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"], constrained_layout=True)

    hb = ax.hexbin(
        trace_scaled, reward,
        gridsize=gridsize,
        cmap="Blues",
        mincnt=1,
        linewidths=0.2,
    )
    counts = hb.get_array()
    hb.set_clim(vmin=0, vmax=np.percentile(counts[counts > 0], 95))
    cb = fig.colorbar(hb, ax=ax, label="Count (clipped at p95)")
    cb.ax.tick_params(labelsize=8)

    exp = int(np.log10(scale))
    ax.set_xlabel(rf"GP posterior trace  $\mathrm{{tr}}(\Sigma)\ /\ 10^{{{exp}}}$")
    ax.set_ylabel("Reward $r$")
    ax.set_ylim(bottom=0.0)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    if output_path.suffix == ".pdf":
        svg_path = output_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved -> {svg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot GP trace vs reward for the training dataset."
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
        help="Output PDF path (default: <data_dir>/trace_vs_reward.pdf).",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=45,
        help="Hexbin grid resolution (default: 45).",
    )
    args = parser.parse_args()

    output_path = args.output or (args.data.parent / "trace_vs_reward.pdf")

    print(f"Data   : {args.data}")
    print(f"Output : {output_path}")

    trace, reward = load_arrays(args.data)
    print(f"Samples: {len(trace):,}")
    print(f"Trace  : min={trace.min():.3f}  max={trace.max():.3f}  mean={trace.mean():.3f}")
    print(f"Reward : min={reward.min():.3f}  max={reward.max():.3f}  mean={reward.mean():.3f}")

    plot_trace_vs_reward(trace, reward, output_path, gridsize=args.gridsize)


if __name__ == "__main__":
    main()
