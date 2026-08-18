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


def load_arrays(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = joblib.load(data_path)
    trace  = np.array([d["b_var"].sum() for d in dataset], dtype=np.float32)
    reward = np.array([float(d["r"])    for d in dataset], dtype=np.float32)
    return trace, reward


def plot_trace_vs_reward(
    trace: np.ndarray,
    reward: np.ndarray,
    output_path: Path,
    gridsize: int = 30,
) -> None:
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
        default=30,
        help="Hexbin grid resolution (default: 30).",
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
