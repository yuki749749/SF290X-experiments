"""
Visualize sampled trajectories from known_grid.py.

Creates a 1×4 figure — one panel per conditioning variant — where the
background shows the GP posterior variance and each sampled trajectory is
drawn on top.

Usage
-----
    # Auto-select the most recent run:
    python plot_trajectories.py

    # Specific run directory:
    python plot_trajectories.py --run_dir results/single_scenario_analysis/run/2026-05-09_12-00-00
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style, FIGURE_SIZES

apply_style(grid=False)

VARIANTS = [
    ("belief_and_reward", "Belief + Reward", "#0072B2"),
    ("belief_only",       "Belief only",     "#E69F00"),
    ("reward_only",       "Reward only",     "#009E73"),
    ("unconditioned",     "Unconditioned",   "#CC79A7"),
]


# ── Directory resolution ───────────────────────────────────────────────────────

def resolve_run_dir(base: Path) -> Path:
    if (base / "trajectories.pkl").exists():
        return base
    if not base.exists():
        raise FileNotFoundError(
            f"Directory not found: {base}\n"
            "Run known_grid.py first, then pass --run_dir to the exact output folder."
        )
    candidates = sorted(d for d in base.iterdir() if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No subdirectories in {base}")
    latest = candidates[-1]
    print(f"Auto-selected run: {latest.name}")
    return latest


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot(run_dir: Path, output_path: Path) -> None:
    data = joblib.load(run_dir / "trajectories.pkl")

    eval_x     = data["eval_x"]       # (1600, 2)
    belief_var = data["belief_var"]    # (1600,)
    known_x    = data["known_x"]       # (M, 2)
    init_pos   = data["initial_position"]

    # Reconstruct the 2-D evaluation grid
    side = int(round(np.sqrt(len(eval_x))))
    X = eval_x[:, 0].reshape(side, side)
    Y = eval_x[:, 1].reshape(side, side)
    Z = belief_var.reshape(side, side)

    vmin, vmax = Z.min(), Z.max()

    # Bounding box of the known region
    kx_min, kx_max = known_x[:, 0].min(), known_x[:, 0].max()
    ky_min, ky_max = known_x[:, 1].min(), known_x[:, 1].max()

    fig, axes = plt.subplots(
        1, 4, figsize=FIGURE_SIZES["wide"],
        constrained_layout=True,
    )

    pcm = None
    for ax, (key, label, color) in zip(axes, VARIANTS):
        trajectories = data["results"][key]  # (N, horizon, 2)

        # Posterior variance background
        pcm = ax.pcolormesh(
            X, Y, Z,
            cmap="YlOrRd", vmin=vmin, vmax=vmax,
            rasterized=True, shading="auto",
        )

        # Known region outline
        rect = mpatches.FancyBboxPatch(
            (kx_min, ky_min), kx_max - kx_min, ky_max - ky_min,
            boxstyle="square,pad=0",
            linewidth=1.0, edgecolor="blue", facecolor="none",
            linestyle="--", zorder=3,
        )
        ax.add_patch(rect)

        # Sampled trajectories
        for traj in trajectories:
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=color, alpha=0.4, linewidth=0.9,
                solid_capstyle="round",
            )

        # Initial position marker
        ax.scatter(
            init_pos[0], init_pos[1],
            s=40, color="white", edgecolors="black",
            linewidths=0.8, zorder=5,
        )

        ax.set_title(label)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")

    axes[0].set_ylabel("y (m)")

    # Shared colorbar to the right of the last panel
    cbar = fig.colorbar(pcm, ax=axes[-1], shrink=0.85, pad=0.02)
    cbar.set_label("Posterior variance", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot sampled trajectories from single_scenario_analysis/known_grid.py."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("results/known_grid/run"),
        help="Run directory containing trajectories.pkl, or parent to auto-select latest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <run_dir>/trajectories.pdf).",
    )
    args = parser.parse_args()

    run_dir     = resolve_run_dir(args.run_dir)
    output_path = args.output or (run_dir / "trajectories.pdf")

    print(f"Run dir : {run_dir}")
    print(f"Output  : {output_path}")

    plot(run_dir, output_path)


if __name__ == "__main__":
    main()
