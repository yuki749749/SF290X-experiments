"""
plot_trajectory_grid.py
=======================
Plot agent trajectories for a single scenario across all (w_belief, w_return)
configurations in a guidance weights sweep, arranged as a 2-D grid.

    Rows    → w_belief (belief guidance weight), largest at top
    Columns → w_return (return guidance weight), smallest at left

Each cell shows the full trajectory overlaid on the ground-truth plume field.

Expected Hydra multirun layout:
    results/guidance_weights_sweep/sweep/<timestamp>/
        scenario_<i>/w_belief=<w_b>,w_return=<w_r>/history.pkl

Usage
-----
    # Auto-select latest sweep, all scenarios:
    python plot_trajectory_grid.py

    # Specific sweep dir, all scenarios:
    python plot_trajectory_grid.py \\
        --sweep_dir results/guidance_weights_sweep/sweep/2026-05-13_12-00-00

    # Single scenario:
    python plot_trajectory_grid.py --scenario_idx 2

    # Subset of scenarios:
    python plot_trajectory_grid.py --scenario_idx 0 1 3

Output
------
    <sweep_root>/scenario_<i>/trajectory_grid.pdf  (one file per scenario)
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style

apply_style(grid=False)


# ── Directory helpers ──────────────────────────────────────────────────────────

def resolve_sweep_root(sweep_dir: Path) -> Path:
    if any(sweep_dir.glob("scenario_*")):
        return sweep_dir
    subdirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {sweep_dir}")
    latest = subdirs[-1]
    print(f"Auto-selected timestamp: {latest.name}")
    return latest


def _fmt(v: float) -> str:
    s = f"{v:g}"
    return s + ".0" if "." not in s else s


def discover_scenarios(sweep_root: Path) -> list[int]:
    indices = []
    for p in sweep_root.iterdir():
        m = re.fullmatch(r"scenario_(\d+)", p.name)
        if m and p.is_dir():
            indices.append(int(m.group(1)))
    if not indices:
        raise FileNotFoundError(f"No scenario_* directories found under {sweep_root}")
    return sorted(indices)


def discover_grid(
    sweep_root: Path, belief_key: str, return_key: str
) -> tuple[list[float], list[float]]:
    pattern = re.compile(
        rf"{re.escape(belief_key)}=([^,]+),{re.escape(return_key)}=(.+)"
    )
    belief_vals, return_vals = set(), set()
    for p in sweep_root.glob("scenario_*/*"):
        m = pattern.fullmatch(p.name)
        if m:
            belief_vals.add(float(m.group(1)))
            return_vals.add(float(m.group(2)))
    if not belief_vals:
        raise FileNotFoundError(
            f"No grid subdirs matching '{belief_key}=*,{return_key}=*' found under {sweep_root}"
        )
    return sorted(belief_vals), sorted(return_vals)


def load_history(
    sweep_root: Path,
    belief_key: str,
    return_key: str,
    w_b: float,
    w_r: float,
    scenario_idx: int,
) -> dict | None:
    subdir = f"{belief_key}={_fmt(w_b)},{return_key}={_fmt(w_r)}"
    pkl_path = sweep_root / f"scenario_{scenario_idx}" / subdir / "history.pkl"
    if not pkl_path.exists():
        return None
    return joblib.load(pkl_path)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def to_grid(arr, side: int) -> np.ndarray:
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr).reshape(side, side)


def add_domain_box(ax, domain_size, pad):
    W, H = domain_size
    rect = mpatches.Rectangle(
        (0, 0), W, H,
        linewidth=0.8, edgecolor="white", facecolor="none",
        linestyle="--", zorder=5,
    )
    ax.add_patch(rect)
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(-pad, H + pad)


# ── Core plot ──────────────────────────────────────────────────────────────────

def plot_trajectory_grid(
    sweep_root: Path,
    output_path: Path,
    belief_key: str,
    return_key: str,
    belief_vals: list[float],
    return_vals: list[float],
    scenario_idx: int,
    domain_size: tuple[float, float],
    domain_pad: float,
) -> None:
    n_rows = len(belief_vals)
    n_cols = len(return_vals)

    # Determine a shared vmin/vmax for the ground truth across all cells
    gt_vmin, gt_vmax = None, None
    for w_b in belief_vals:
        for w_r in return_vals:
            h = load_history(sweep_root, belief_key, return_key, w_b, w_r, scenario_idx)
            if h is None or h.get("ground_truth") is None:
                continue
            gt = np.asarray(h["ground_truth"])
            gmin, gmax = float(gt.min()), float(gt.max())
            gt_vmin = gmin if gt_vmin is None else min(gt_vmin, gmin)
            gt_vmax = gmax if gt_vmax is None else max(gt_vmax, gmax)

    cell_size = 2.4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size),
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Trajectory grid — scenario {scenario_idx}  "
        r"(rows: $w_b$, cols: $w_r$)",
        fontsize=10,
    )

    # Rows: smallest w_belief at top (row 0)
    for row_i, w_b in enumerate(belief_vals):
        for col_j, w_r in enumerate(return_vals):
            ax = axes[row_i, col_j]
            h = load_history(sweep_root, belief_key, return_key, w_b, w_r, scenario_idx)

            if h is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ground_truth = h.get("ground_truth")
            pos_seq = np.array(h["position_history"])  # (T+1, 2)

            if ground_truth is not None:
                N    = len(np.asarray(ground_truth).flatten())
                side = int(round(np.sqrt(N)))
                bg   = to_grid(ground_truth, side).T
                vmin, vmax = gt_vmin, gt_vmax
            else:
                means = h["mean_history"]
                N    = len(np.asarray(means[-1]).flatten())
                side = int(round(np.sqrt(N)))
                bg   = to_grid(means[-1], side).T
                vmin, vmax = None, None

            W, H = domain_size
            extent = [-domain_pad, W + domain_pad, -domain_pad, H + domain_pad]

            ax.imshow(
                bg,
                origin="lower",
                cmap="viridis",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                zorder=0,
            )
            add_domain_box(ax, domain_size, domain_pad)

            inside = (
                (pos_seq[:, 0] >= 0) & (pos_seq[:, 0] <= W) &
                (pos_seq[:, 1] >= 0) & (pos_seq[:, 1] <= H)
            )

            ax.plot(pos_seq[:, 0], pos_seq[:, 1],
                    color="white", lw=0.8, alpha=0.7, zorder=6)

            if (~inside).any():
                ax.scatter(pos_seq[~inside, 0], pos_seq[~inside, 1],
                           color="orange", s=6, zorder=7, linewidths=0)

            ax.scatter(pos_seq[inside, 0], pos_seq[inside, 1],
                       color="white", s=3, zorder=7, linewidths=0)

            ax.scatter(pos_seq[0, 0], pos_seq[0, 1],
                       marker="*", s=60, color="yellow",
                       edgecolors="black", linewidths=0.4, zorder=8)

            ax.set_xticks([])
            ax.set_yticks([])

            if col_j == 0:
                ax.set_ylabel(f"$w_b = {_fmt(w_b)}$", fontsize=8)
            if row_i == 0:
                ax.set_title(f"$w_r = {_fmt(w_r)}$", fontsize=8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot trajectory grid for a single scenario across guidance weight configs."
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/guidance_weights_sweep/sweep"),
        help="Sweep root or specific timestamp subdir (default: auto-latest).",
    )
    parser.add_argument(
        "--scenario_idx",
        type=int,
        nargs="*",
        default=None,
        help="Scenario index (or indices) to visualize (default: all found in sweep).",
    )
    parser.add_argument(
        "--belief_key",
        type=str,
        default="w_belief",
        help="Subdir key for belief_guidance_w (default: 'w_belief').",
    )
    parser.add_argument(
        "--return_key",
        type=str,
        default="w_return",
        help="Subdir key for return_guidance_w (default: 'w_return').",
    )
    parser.add_argument(
        "--domain_size",
        type=float,
        nargs=2,
        default=[15.0, 15.0],
        metavar=("W", "H"),
        help="Physical domain size (default: 15 15).",
    )
    parser.add_argument(
        "--domain_pad",
        type=float,
        default=5.0,
        metavar="PAD",
        help="Padding around domain boundary in domain units (default: 5.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <sweep_root>/trajectory_grid_scenario_<i>.pdf).",
    )
    args = parser.parse_args()

    sweep_root = resolve_sweep_root(args.sweep_dir)
    belief_vals, return_vals = discover_grid(sweep_root, args.belief_key, args.return_key)

    scenario_indices = args.scenario_idx if args.scenario_idx else discover_scenarios(sweep_root)

    print(f"Sweep root      : {sweep_root}")
    print(f"Scenarios       : {scenario_indices}")
    print(f"w_belief values : {belief_vals}")
    print(f"w_return values : {return_vals}")

    for idx in scenario_indices:
        output_path = args.output or (sweep_root / f"scenario_{idx}" / "trajectory_grid.pdf")
        print(f"\nPlotting scenario {idx} → {output_path}")
        plot_trajectory_grid(
            sweep_root=sweep_root,
            output_path=output_path,
            belief_key=args.belief_key,
            return_key=args.return_key,
            belief_vals=belief_vals,
            return_vals=return_vals,
            scenario_idx=idx,
            domain_size=tuple(args.domain_size),
            domain_pad=args.domain_pad,
        )


if __name__ == "__main__":
    main()
