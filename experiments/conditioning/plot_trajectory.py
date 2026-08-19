"""
plot_sweep.py
=============
Walk a Hydra multirun sweep directory and save evaluation_grid.pdf
alongside each history.pkl.

Expected folder structure (from evaluation.yaml):
    results/evaluation/sweep/<timestamp>/scenario_<i>/<planner>/history.pkl

Usage
-----
    # Plot a specific sweep timestamp:
    python plot_trajectory.py --sweep_dir results/evaluation/sweep/2026-03-23_16-49-31

    # Plot the most recent sweep automatically:
    python plot_trajectory.py --sweep_dir results/evaluation/sweep

    # Override domain size if different from default:
    python plot_trajectory.py --sweep_dir results/evaluation/sweep --domain_size 10 10

    # Control padding around domain shown in plot (in domain units):
    python plot_trajectory.py --sweep_dir results/evaluation/sweep --domain_pad 1.0
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style

apply_style(grid=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_grid(arr, side):
    """Flat tensor or ndarray → (side, side) numpy array."""
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr).reshape(side, side)


def find_history_files(sweep_root: Path) -> list[Path]:
    hits = sorted(sweep_root.glob("*/*/history.pkl"))
    if hits:
        return hits
    subdirs = sorted(d for d in sweep_root.iterdir() if d.is_dir())
    if subdirs:
        hits = sorted(subdirs[-1].glob("*/*/history.pkl"))
    return hits


def label_from_path(pkl_path: Path) -> str:
    parts = pkl_path.parts
    planner  = parts[-2]
    scenario = parts[-3]
    return f"{scenario} / {planner}"


def add_domain_box(ax, domain_size, pad):
    """
    Draw a dashed white rectangle marking the domain boundary [0,W]x[0,H],
    and set axis limits to domain + pad on all sides so out-of-domain
    trajectory segments are visible.
    """
    W, H = domain_size
    rect = mpatches.Rectangle(
        (0, 0), W, H,
        linewidth=1.2,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(rect)
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(-pad, H + pad)


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def plot_history(
    pkl_path: Path,
    domain_size: tuple[float, float],
    domain_pad: float = 1.0,
) -> None:
    try:
        from omegaconf import OmegaConf
        cfg_path = Path(pkl_path).parent.parent.parent / ".hydra" / "config.yaml"
        if cfg_path.exists():
            cfg = OmegaConf.load(cfg_path)
            domain_size = tuple(cfg.get("domain_size", domain_size))
            domain_pad = float(cfg.get("domain_pad", domain_pad))
    except Exception:
        pass

    history = joblib.load(pkl_path)

    means        = history["mean_history"]
    variances    = history["variance_history"]
    positions    = history["position_history"]
    ground_truth = history.get("ground_truth", None)

    T    = len(means) - 1
    N    = len(means[0].flatten())
    side = int(round(np.sqrt(N)))
    assert side * side == N, (
        f"Evaluation grid is not square (N={N}). "
        "Adjust make_evaluation_x to produce a perfect square."
    )

    steps      = [0, T // 2, T]
    row_labels = [rf"$t = {s}$" for s in steps]

    has_gt     = ground_truth is not None
    n_cols     = 3 if has_gt else 2
    col_titles = (
        ["Ground Truth", "Predictive Mean", "Predictive Variance"]
        if has_gt else
        ["Predictive Mean", "Predictive Variance"]
    )
    cmaps = ["viridis", "viridis", "plasma"] if has_gt else ["viridis", "plasma"]

    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(n_cols * 2.8, 7.5),
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(label_from_path(pkl_path), fontsize=9)

    gt_grid = to_grid(ground_truth, side) if has_gt else None

    v_min  = ground_truth.min().item() if has_gt else None
    v_max  = ground_truth.max().item() if has_gt else None
    var_max = max(v.max().item() for v in variances)

    for row, (t, row_label) in enumerate(zip(steps, row_labels)):
        mean_grid = to_grid(means[t], side)
        var_grid  = to_grid(variances[t], side)

        grids = ([gt_grid, mean_grid, var_grid] if has_gt
                 else [mean_grid, var_grid])
        vmins = [v_min,  v_min,  0]
        vmaxs = [v_max,  v_max,  var_max]

        pos_seq = np.array(positions[: t + 1])  # (t+1, 2)

        # Separate in-domain and out-of-domain trajectory segments
        W, H = domain_size
        inside = (
            (pos_seq[:, 0] >= 0) & (pos_seq[:, 0] <= W) &
            (pos_seq[:, 1] >= 0) & (pos_seq[:, 1] <= H)
        )

        for col in range(n_cols):
            ax = axes[row, col]

            # --- background image (clipped to domain extent) --------------
            im = ax.imshow(
                grids[col].T,
                origin="lower",
                cmap=cmaps[col],
                extent=[-domain_pad, domain_size[0] + domain_pad, -domain_pad, domain_size[1] + domain_pad],
                vmin=vmins[col], vmax=vmaxs[col],
                zorder=0,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # --- domain boundary box + expanded axis limits ---------------
            add_domain_box(ax, domain_size, domain_pad)

            # --- trajectory: in-domain segment (white) --------------------
            ax.plot(
                pos_seq[:, 0], pos_seq[:, 1],
                color="white", lw=0.8, alpha=0.5, zorder=6,
            )

            # --- out-of-domain waypoints highlighted (orange) -------------
            out_mask = ~inside[: t + 1]
            if out_mask.any():
                ax.scatter(
                    pos_seq[out_mask, 0], pos_seq[out_mask, 1],
                    color="orange", s=8, zorder=7, linewidths=0,
                )

            # --- waypoint dots (white) ------------------------------------
            ax.scatter(
                pos_seq[inside[: t + 1], 0], pos_seq[inside[: t + 1], 1],
                color="white", s=4, zorder=7, linewidths=0,
            )

            # --- start marker ---------------------------------------------
            ax.scatter(
                pos_seq[0, 0], pos_seq[0, 1],
                marker="*", s=60, color="yellow",
                edgecolors="black", linewidths=0.4, zorder=8,
            )

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    out_path = pkl_path.parent / "evaluation_grid.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Hydra sweep evaluation grids.")
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/conditioning_ablation/sweep"),
        help="Path to the Hydra sweep root or a specific timestamp subdir.",
    )
    parser.add_argument(
        "--domain_size",
        type=float,
        nargs=2,
        default=[300.0, 300.0],
        metavar=("W", "H"),
        help="Physical domain size used during evaluation (default: 300 300).",
    )
    parser.add_argument(
        "--domain_pad",
        type=float,
        default=50.0,
        metavar="PAD",
        help="Padding (in domain units) shown around the domain boundary (default: 50.0).",
    )
    args = parser.parse_args()

    pkl_files = find_history_files(args.sweep_dir)
    if not pkl_files:
        print(f"No history.pkl files found under: {args.sweep_dir}", file=sys.stderr)
        sys.exit(1)

    domain_size = tuple(args.domain_size)
    print(f"Found {len(pkl_files)} history file(s). Domain: {domain_size}, pad: {args.domain_pad}")

    for pkl_path in pkl_files:
        print(f"Processing: {pkl_path}")
        try:
            plot_history(pkl_path, domain_size, domain_pad=args.domain_pad)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()