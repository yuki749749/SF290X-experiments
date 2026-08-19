"""
plot_conditioning_ablation.py
==============================
Plots the 2×2 belief × return conditioning ablation results.

Expected directory layout (Hydra multirun output):

    results/conditioning_ablation/sweep/<timestamp>/
        scenario_0/use_belief=True,use_return=True/history.pkl
        scenario_0/use_belief=True,use_return=False/history.pkl
        scenario_0/use_belief=False,use_return=True/history.pkl
        scenario_0/use_belief=False,use_return=False/history.pkl
        scenario_1/...
        ...

If <sweep_dir> points to the sweep root (containing timestamp subdirs),
the most recent timestamp is selected automatically.  Pass a specific
timestamp dir to override.

Output
------
    <sweep_root>/conditioning_ablation.pdf

Two panels side by side:
    Left : Normalized RMSE over time (lower is better)
    Right: Normalized Trace Reduction over time (higher is better)

Usage
-----
    # Auto-select latest timestamp:
    python plot_conditioning_ablation.py

    # Specific timestamp:
    python plot_conditioning_ablation.py \
        --sweep_dir results/conditioning_ablation/sweep/2026-04-15_12-00-00
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from plot_style import apply_style, FIGURE_SIZES
from utils import resolve_sweep_root
from evaluation_utils import mean_ci_bootstrap as mean_ci

apply_style()

# Variant definitions: (use_belief, use_return, dir_suffix, label, color, linestyle)
# dir_suffix must match the Hydra subdir name exactly.
VARIANTS = [
    (True,  True,  "use_belief=True,use_return=True",   r"Belief + Reward",        "#0072B2", "-"),
    (True,  False, "use_belief=True,use_return=False",  r"Belief only",            "#E69F00", "--"),
    (False, True,  "use_belief=False,use_return=True",  r"Reward only",            "#009E73", "-."),
    (False, False, "use_belief=False,use_return=False", r"Unconditioned",          "#999999", ":"),
]

METRICS = [
    ("rmse_history",                     "Normalized RMSE",            "lower is better"),
]


# ── Directory resolution ───────────────────────────────────────────────────────


# ── Data loading ───────────────────────────────────────────────────────────────

def load_variant(sweep_root: Path, dir_suffix: str, metric_key: str) -> np.ndarray:
    """
    Collect history.pkl files for one variant across all scenario directories.

    Layout:  sweep_root / scenario_* / <dir_suffix> / history.pkl

    Returns
    -------
    np.ndarray of shape (n_scenarios, T), shorter runs padded with last value.
    """
    pkl_files = sorted(sweep_root.glob(f"scenario_*/{dir_suffix}/history.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No history.pkl files found for variant '{dir_suffix}' under {sweep_root}"
        )

    arrays = []
    for pkl in pkl_files:
        h = joblib.load(pkl)
        if metric_key not in h:
            raise KeyError(f"Metric '{metric_key}' not found in {pkl}")
        arrays.append([float(v) for v in h[metric_key]])

    max_len = max(len(a) for a in arrays)
    padded  = [a + [a[-1]] * (max_len - len(a)) for a in arrays]
    return np.array(padded)   # (n_scenarios, T)



# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_ablation(sweep_root: Path, output_path: Path, rmse_only: bool = False) -> None:
    rng = np.random.default_rng()   # shared across all variants and metrics

    metrics = [METRICS[0]] if rmse_only else METRICS

    figsize = FIGURE_SIZES["single"] if len(metrics) == 1 else FIGURE_SIZES["double_col"]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, metric_label, direction) in zip(axes, metrics):
        for (_, _, dir_suffix, label, color, ls) in VARIANTS:
            try:
                matrix = load_variant(sweep_root, dir_suffix, metric_key)
            except (FileNotFoundError, KeyError) as e:
                print(f"  Warning: {e}")
                continue

            mu, lo, hi = mean_ci(matrix, rng=rng)
            t = np.arange(len(mu))
            ax.plot(t, mu, color=color, linestyle=ls)
            ax.fill_between(t, lo, hi, color=color, alpha=0.12)
            print(f"  {label:30s} - {matrix.shape[0]} scenarios, T={len(mu)}")

        ax.set_ylabel(metric_label)
        ax.set_xlabel("Time step")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

    legend_handles = [
        Line2D([0], [0], color=color, linestyle=ls, lw=1.5, label=label)
        for (_, _, _, label, color, ls) in VARIANTS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    if output_path.suffix == ".pdf":
        svg_path = output_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved -> {svg_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2×2 conditioning ablation results from a Hydra multirun."
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/conditioning_ablation/sweep"),
        help="Sweep root or specific timestamp subdir (default: auto-latest).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <sweep_root>/conditioning_ablation.pdf or conditioning_ablation_rmse.pdf).",
    )
    parser.add_argument(
        "--rmse_only",
        "--single",
        dest="rmse_only",
        action="store_true",
        help="Plot only Normalized RMSE (single panel).",
    )
    args = parser.parse_args()

    sweep_root  = resolve_sweep_root(args.sweep_dir)
    if args.output is not None:
        output_path = args.output
    else:
        filename = "conditioning_ablation_rmse.pdf" if args.rmse_only else "conditioning_ablation.pdf"
        output_path = sweep_root / filename

    print(f"Sweep root : {sweep_root}")
    print(f"Output     : {output_path}")

    plot_ablation(sweep_root, output_path, rmse_only=args.rmse_only)


if __name__ == "__main__":
    main()