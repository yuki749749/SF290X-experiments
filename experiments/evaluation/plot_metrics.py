"""
plot_metrics.py — Planner comparison: aggregated metrics over time.

Adapted for the Hydra multirun sweep structure:
    results/evaluation/sweep/<timestamp>/scenario_<i>/<planner>/history.pkl

Usage
-----
    # Specific timestamp:
    python plot_metrics.py --sweep_dir results/evaluation/sweep/2026-03-23_16-49-31

    # Auto-pick the most recent timestamp:
    python plot_metrics.py --sweep_dir results/evaluation/sweep

    # Custom output path:
    python plot_metrics.py --sweep_dir results/evaluation/sweep --output my_comparison.pdf

Output: <sweep_dir>/comparison.pdf
Layout: 1 row × 2 columns (Normalized RMSE | Normalized Trace Reduction)
        Mean ± 95% CI per planner, aggregated across all scenarios.
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D


# ── Publication style ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from plot_style import apply_style, FIGURE_SIZES, PLANNER_COLORS, PLANNER_ORDER, PLANNER_DISPLAY_NAMES
from utils import resolve_sweep_root

apply_style()

METRICS = [
    ("rmse_history",                     r"Normalized RMSE"),
]


# ── Discovery ─────────────────────────────────────────────────────────────────


def discover_results(sweep_root: Path) -> dict[str, list[dict]]:
    """
    Walk sweep_root and collect history dicts keyed by planner name.

    Expected layout:
        sweep_root/scenario_<i>/<planner>/history.pkl

    Returns:
        data[planner_name] -> list of history dicts (one per scenario)
    """
    data: dict[str, list[dict]] = {}
    pkl_files = sorted(sweep_root.glob("*/*/history.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No history.pkl files found under {sweep_root}")

    for pkl_path in pkl_files:
        # Parts relative to sweep_root: scenario_<i> / <planner> / history.pkl
        rel_parts = pkl_path.relative_to(sweep_root).parts
        if len(rel_parts) != 3:
            print(f"  Skipping unexpected path structure: {pkl_path}", file=sys.stderr)
            continue
        planner_name = rel_parts[-2]   # e.g. 'bo', 'lawnmower', 'diffusion'
        history = joblib.load(pkl_path)
        data.setdefault(planner_name, []).append(history)

    return data


# ── Metric helpers ─────────────────────────────────────────────────────────────

def extract_metric(runs: list[dict], key: str) -> np.ndarray:
    """
    Stack metric histories → (n_runs, T).
    Shorter runs are padded to max length by repeating their last value.
    """
    arrays = [[float(v) for v in h[key]] for h in runs]
    max_len = max(len(a) for a in arrays)
    padded = [a + [a[-1]] * (max_len - len(a)) for a in arrays]
    return np.array(padded)


def mean_ci(matrix: np.ndarray, z: float = 1.96):
    """Return (mean, lower_95, upper_95) across axis=0."""
    mu  = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=min(1, matrix.shape[0] - 1)) / np.sqrt(matrix.shape[0])
    return mu, mu - z * sem, mu + z * sem


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(sweep_root: Path, output_path: Path, rmse_only: bool = False) -> None:
    data = discover_results(sweep_root)
    known = [p for p in PLANNER_ORDER if p in data]
    extras = sorted(p for p in data if p not in PLANNER_ORDER)
    planners = known + extras
    _fallback = ["#009E73", "#CC79A7", "#E69F00"]   # green, pink, orange for unknowns
    color_map = {
        p: PLANNER_COLORS.get(p, _fallback[i % len(_fallback)])
        for i, p in enumerate(planners)
    }

    print(f"Planners found : {planners}")
    for p, runs in data.items():
        display = PLANNER_DISPLAY_NAMES.get(p, p)
        print(f"  {display}: {len(runs)} scenario(s)")

    metrics = [METRICS[0]] if rmse_only else METRICS

    figsize = FIGURE_SIZES["single"] if len(metrics) == 1 else FIGURE_SIZES["double_col"]
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=figsize,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for planner in planners:
            runs = data.get(planner, [])
            display = PLANNER_DISPLAY_NAMES.get(planner, planner)
            if not runs or metric_key not in runs[0]:
                print(f"  Warning: '{metric_key}' missing for planner '{display}'", file=sys.stderr)
                continue

            matrix = extract_metric(runs, metric_key)
            mu, lo, hi = mean_ci(matrix)
            t = np.arange(len(mu))
            color = color_map[planner]

            ax.plot(t, mu, color=color, label=display)
            ax.fill_between(t, lo, hi, color=color, alpha=0.15)

        ax.set_ylabel(metric_label)
        ax.set_xlabel(r"Time step")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

    legend_handles = [
        Line2D([0], [0], color=color_map[p], lw=1.5, label=PLANNER_DISPLAY_NAMES.get(p, p))
        for p in planners
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2 if len(metrics) == 1 else len(planners),
        frameon=False,
        bbox_to_anchor=(0.5, -0.18) if len(metrics) == 1 else (0.5, -0.12),
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    if output_path.suffix == ".pdf":
        svg_path = output_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved -> {svg_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot aggregated planner metrics from a Hydra sweep."
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/evaluation/sweep"),
        help="Path to the Hydra sweep root or a specific timestamp subdir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the figure (default: <sweep_dir>/comparison.pdf or comparison_rmse.pdf).",
    )
    parser.add_argument(
        "--rmse_only",
        "--single",
        dest="rmse_only",
        action="store_true",
        help="Plot only Normalized RMSE (single panel).",
    )
    args = parser.parse_args()

    sweep_root = resolve_sweep_root(args.sweep_dir)
    # Default: timestamp folder (sweep_root), not a per-scenario subdir
    if args.output is not None:
        output_path = args.output
    else:
        filename = "comparison_rmse.pdf" if args.rmse_only else "comparison.pdf"
        output_path = sweep_root / filename

    print(f"Sweep root : {sweep_root}")
    print(f"Output     : {output_path}")

    plot_comparison(sweep_root, output_path, rmse_only=args.rmse_only)


if __name__ == "__main__":
    main()