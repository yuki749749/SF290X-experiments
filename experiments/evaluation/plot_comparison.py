"""
plot_configurations.py
======================
Plot a 1×5 comparison figure across four planner/ablation configurations
for a single scenario:

    [ Ground Truth | Config 1 | Config 2 | Config 3 | Config 4 ]

Each configuration panel shows the predictive mean field at the final
timestep T, with the full AUV trajectory overlaid.

Expected folder structure (same as evaluation.yaml sweep output):
    <sweep_dir>/<timestamp>/scenario_<i>/<config_name>/history.pkl

Usage
-----
    # Auto-discover all (scenario, config) combos under a sweep dir and
    # produce one figure per scenario:
    python plot_configurations.py --sweep_dir results/evaluation/sweep

    # Plot a specific timestamp:
    python plot_configurations.py \\
        --sweep_dir results/evaluation/sweep/2026-03-23_16-49-31

    # Provide four history.pkl paths explicitly (single figure):
    python plot_configurations.py \\
        --pkl cfg1/history.pkl cfg2/history.pkl cfg3/history.pkl cfg4/history.pkl \\
        --out comparison.pdf

    # Control timestep shown (default: final step T):
    python plot_configurations.py --sweep_dir ... --timestep 20

    # Override domain / padding:
    python plot_configurations.py --sweep_dir ... --domain_size 15 15 --domain_pad 5
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style, FIGURE_SIZES
from evaluation_utils import to_grid, infer_side, add_domain_box, overlay_trajectory, inside_mask

apply_style(grid=False)


# Canonical left-to-right order for the 1×5 figure.
# Keys are the planner folder names produced by the sweep.
_CONFIG_ORDER = [
    "diffusion",
    "bo",
    "random",
    "lawnmower",
]

_CONFIG_LABELS = {
    "diffusion": "Diffusion",
    "bo":        "Bayesian Optimization",
    "random":    "Random Walk",
    "lawnmower": "Lawnmower",
}


def config_label(pkl_path: Path) -> str:
    """Human-readable panel title derived from the planner folder name."""
    name = pkl_path.parent.name          # e.g. "diffusion", "bo", "random", "lawnmower"
    return _CONFIG_LABELS.get(name, name.replace("_", " ").title())


# inside_mask imported from evaluation_utils

# ---------------------------------------------------------------------------
# Core: single 1×5 figure
# ---------------------------------------------------------------------------

def plot_1x5(
    pkl_paths: list[Path],
    domain_size: tuple[float, float],
    domain_pad: float = 5.0,
    timestep: int | None = None,
    out_path: Path | None = None,
    suptitle: str = "",
) -> Path:
    """
    Parameters
    ----------
    pkl_paths   : exactly 4 history.pkl files, one per configuration
    domain_size : (W, H) of the physical domain
    domain_pad  : axis padding beyond the domain boundary
    timestep    : which step to show; None → final step T of each run
    out_path    : where to save the PDF; auto-derived if None
    suptitle    : figure-level title
    """
    num_configs = len(pkl_paths)
    assert num_configs >= 1, "At least 1 pkl path required for comparison layout."

    try:
        from omegaconf import OmegaConf
        cfg_path = Path(pkl_paths[0]).parent.parent.parent / ".hydra" / "config.yaml"
        if cfg_path.exists():
            cfg = OmegaConf.load(cfg_path)
            domain_size = tuple(cfg.get("domain_size", domain_size))
            domain_pad = float(cfg.get("domain_pad", domain_pad))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Load all histories
    # ------------------------------------------------------------------
    histories = [joblib.load(p) for p in pkl_paths]

    # Ground truth: take from first file; warn if they differ
    ground_truth = histories[0].get("ground_truth", None)
    if ground_truth is None:
        raise ValueError("history.pkl has no 'ground_truth' key.")

    side = infer_side(ground_truth)
    gt_grid = to_grid(ground_truth, side)

    # Shared color scale for all mean panels (anchored to GT range)
    v_min = float(np.asarray(ground_truth).min())
    v_max = float(np.asarray(ground_truth).max())

    # ------------------------------------------------------------------
    # Figure layout: 1 row × (num_configs + 1) cols
    # ------------------------------------------------------------------
    fig_width = (num_configs + 1) * 2.8
    fig, axes = plt.subplots(
        1, num_configs + 1,
        figsize=(fig_width, 3.2),
        constrained_layout=True,
    )

    # if suptitle:
    #     fig.suptitle(suptitle, fontsize=10, y=1.02)

    extent = [
        -domain_pad, domain_size[0] + domain_pad,
        -domain_pad, domain_size[1] + domain_pad,
    ]

    # ------------------------------------------------------------------
    # Column 0: Ground Truth (no trajectory)
    # ------------------------------------------------------------------
    ax_gt = axes[0]
    im_gt = ax_gt.imshow(
        gt_grid.T,
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=v_min, vmax=v_max,
        zorder=0,
    )
    add_domain_box(ax_gt, domain_size, domain_pad)
    ax_gt.set_title("Ground Truth", fontsize=16)
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])

    # ------------------------------------------------------------------
    # Columns 1–4: Predictive mean + trajectory per config
    # ------------------------------------------------------------------
    for col_idx, (h, pkl_path) in enumerate(zip(histories, pkl_paths), start=1):
        ax = axes[col_idx]

        means     = h["mean_history"]
        positions = h["position_history"]

        T = len(means) - 1
        t = timestep if (timestep is not None and timestep <= T) else T

        mean_grid = to_grid(means[t], side)
        pos_seq   = np.array(positions[: t + 1])   # (t+1, 2)
        mask      = inside_mask(pos_seq, domain_size)

        # Mean field
        ax.imshow(
            mean_grid.T,
            origin="lower",
            cmap="viridis",
            extent=extent,
            vmin=v_min, vmax=v_max,   # same scale as GT
            zorder=0,
        )
        add_domain_box(ax, domain_size, domain_pad)

        # Trajectory overlay
        overlay_trajectory(ax, pos_seq, domain_size, mask)

        # Step annotation (bottom-left corner)
        # ax.text(
        #     0.03, 0.03, f"t = {t}",
        #     transform=ax.transAxes,
        #     fontsize=7, color="white",
        #     va="bottom", ha="left",
        #     bbox=dict(facecolor="black", alpha=0.35, pad=1.5, linewidth=0),
        # )

        ax.set_title(config_label(pkl_path), fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    # ------------------------------------------------------------------
    # Single shared colorbar on the right
    # ------------------------------------------------------------------
    fig.colorbar(im_gt, ax=axes.tolist(), fraction=0.02, pad=0.02, shrink=0.8)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if out_path is None:
        # Save alongside the first pkl's parent directory
        out_path = pkl_paths[0].parent.parent / f"configurations_1x{num_configs + 1}.pdf"

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved -> {out_path}")
    if out_path.suffix == ".pdf":
        svg_path = out_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"  Saved -> {svg_path}")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Discovery: group pkl files by scenario, collect configs
# ---------------------------------------------------------------------------

def find_and_group(sweep_root: Path) -> dict[str, list[Path]]:
    """
    Returns {scenario_key: [pkl_cfg1, pkl_cfg2, pkl_cfg3, pkl_cfg4]}
    sorted consistently by config name so order is reproducible.
    """
    from evaluation_utils import find_and_group as _find_and_group
    return _find_and_group(sweep_root, _CONFIG_ORDER)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="1×5 configuration comparison figure for AUV plume mapping."
    )
    # Option A: explicit pkl paths
    parser.add_argument(
        "--pkl",
        type=Path,
        nargs="+",
        metavar="PKL",
        help="One or more history.pkl paths, one per configuration.",
    )
    # Option B: auto-discover from sweep dir
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/evaluation/sweep"),
        help="Hydra sweep root (or timestamp subdir); one figure per scenario.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF path (used with --pkl; ignored for --sweep_dir).",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Timestep to display (default: final step T).",
    )
    parser.add_argument(
        "--domain_size",
        type=float,
        nargs=2,
        default=[300.0, 300.0],
        metavar=("W", "H"),
    )
    parser.add_argument(
        "--domain_pad",
        type=float,
        default=50.0,
        metavar="PAD",
    )
    args = parser.parse_args()

    domain_size = tuple(args.domain_size)

    # ------------------------------------------------------------------
    # Option A: explicit pkl list
    # ------------------------------------------------------------------
    if args.pkl:
        for p in args.pkl:
            if not p.exists():
                print(f"ERROR: file not found: {p}", file=sys.stderr)
                sys.exit(1)
        plot_1x5(
            list(args.pkl),
            domain_size=domain_size,
            domain_pad=args.domain_pad,
            timestep=args.timestep,
            out_path=args.out,
        )
        return

    # ------------------------------------------------------------------
    # Option B: auto-discover
    # ------------------------------------------------------------------
    groups = find_and_group(args.sweep_dir)
    if not groups:
        print(f"No history.pkl files found under: {args.sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(groups)} scenario(s). Domain: {domain_size}, pad: {args.domain_pad}")

    for scenario, pkls in sorted(groups.items()):
        if not pkls:
            continue

        out = pkls[0].parent.parent / f"{scenario}_configurations_1x{len(pkls) + 1}.pdf"
        print(f"  Processing {scenario}: {[p.parent.name for p in pkls]}")
        try:
            plot_1x5(
                pkls,
                domain_size=domain_size,
                domain_pad=args.domain_pad,
                timestep=args.timestep,
                out_path=out,
                suptitle=scenario.replace("_", " ").title(),
            )
        except Exception as exc:
            print(f"  ERROR ({scenario}): {exc}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()