"""
plot_metrics_stacked.py — Planner comparison: stacked metrics for slides.

Same data source as plot_metrics.py, but laid out as 2 rows × 1 column
(Normalized RMSE on top, Normalized Trace Reduction on bottom) with larger
fonts suited for presentation slides.

Usage
-----
    # Auto-pick the most recent timestamp:
    python plot_metrics_stacked.py --sweep_dir results/evaluation/sweep

    # Specific timestamp:
    python plot_metrics_stacked.py --sweep_dir results/evaluation/sweep/2026-03-23_16-49-31

    # Custom output path:
    python plot_metrics_stacked.py --sweep_dir results/evaluation/sweep --output slides_metrics.pdf

Output: <sweep_dir>/comparison_stacked.pdf
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from plot_style import PLANNER_COLORS, PLANNER_ORDER, PLANNER_DISPLAY_NAMES as _BASE_NAMES
from utils import resolve_sweep_root

PLANNER_DISPLAY_NAMES = {**_BASE_NAMES, "bo": "BO"}

# ── Slide-tuned style (larger fonts, no serif) ────────────────────────────────

def apply_slide_style() -> None:
    plt.rcParams.update({
        "text.usetex":       False,
        "font.family":       "sans-serif",
        "axes.labelsize":    20,
        "axes.titlesize":    21,
        "xtick.labelsize":   17,
        "ytick.labelsize":   17,
        "legend.fontsize":   17,
        "axes.linewidth":    1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "lines.linewidth":   2.5,
        "figure.dpi":        150,
        "axes.grid":         False,
    })


METRICS = [
    ("rmse_history",                     "NRMSE"),
    ("normalizedTraceReduction_history", "NTR"),
]


# ── Discovery (shared with plot_metrics.py) ───────────────────────────────────


def discover_results(sweep_root: Path) -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    pkl_files = sorted(sweep_root.glob("*/*/history.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No history.pkl files found under {sweep_root}")
    for pkl_path in pkl_files:
        rel_parts = pkl_path.relative_to(sweep_root).parts
        if len(rel_parts) != 3:
            continue
        planner_name = rel_parts[-2]
        history = joblib.load(pkl_path)
        data.setdefault(planner_name, []).append(history)
    return data


# ── Metric helpers ─────────────────────────────────────────────────────────────

def extract_metric(runs: list[dict], key: str) -> np.ndarray:
    arrays = [[float(v) for v in h[key]] for h in runs]
    max_len = max(len(a) for a in arrays)
    padded = [a + [a[-1]] * (max_len - len(a)) for a in arrays]
    return np.array(padded)


def mean_ci(matrix: np.ndarray, z: float = 1.96):
    mu  = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=min(1, matrix.shape[0] - 1)) / np.sqrt(matrix.shape[0])
    return mu, mu - z * sem, mu + z * sem


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_stacked(sweep_root: Path, output_path: Path) -> None:
    data = discover_results(sweep_root)
    known  = [p for p in PLANNER_ORDER if p in data]
    extras = sorted(p for p in data if p not in PLANNER_ORDER)
    planners = known + extras
    _fallback = ["#009E73", "#CC79A7", "#E69F00"]
    color_map = {
        p: PLANNER_COLORS.get(p, _fallback[i % len(_fallback)])
        for i, p in enumerate(planners)
    }

    print(f"Planners found : {planners}")
    for p, runs in data.items():
        print(f"  {PLANNER_DISPLAY_NAMES.get(p, p)}: {len(runs)} scenario(s)")

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 7),
        constrained_layout=True,
    )

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for planner in planners:
            runs = data.get(planner, [])
            display = PLANNER_DISPLAY_NAMES.get(planner, planner)
            if not runs or metric_key not in runs[0]:
                print(f"  Warning: '{metric_key}' missing for '{display}'", file=sys.stderr)
                continue

            matrix = extract_metric(runs, metric_key)
            mu, lo, hi = mean_ci(matrix)
            t = np.arange(len(mu))
            color = color_map[planner]

            ax.plot(t, mu, color=color, label=display)
            ax.fill_between(t, lo, hi, color=color, alpha=0.15)

        ax.set_ylabel(metric_label)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time step")

    legend_handles = [
        Line2D([0], [0], color=color_map[p], lw=2.5, label=PLANNER_DISPLAY_NAMES.get(p, p))
        for p in planners
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.13),
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot stacked planner metrics (slide layout) from a Hydra sweep."
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
        help="Output path (default: <sweep_dir>/comparison_stacked.pdf).",
    )
    args = parser.parse_args()

    apply_slide_style()

    sweep_root  = resolve_sweep_root(args.sweep_dir)
    output_path = args.output or (sweep_root / "comparison_stacked.pdf")

    print(f"Sweep root : {sweep_root}")
    print(f"Output     : {output_path}")

    plot_stacked(sweep_root, output_path)


if __name__ == "__main__":
    main()
