"""
plotter.py — Planner comparison: aggregated metrics over time.

Usage:
    python plotter.py --results_dir results/ --output comparison.pdf

Layout: 1 row × 3 columns (RMSE | NLPD | Normalized Trace Reduction).
Each subplot shows mean ± 95% CI per planner, aggregated across all scenarios and runs.
"""

import os
import glob
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":          False,
    "mathtext.fontset":     "cm",       # Computer Modern via matplotlib's mathtext
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "serif"],
    "axes.labelsize":       9,
    "axes.titlesize":       10,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      9,
    "figure.titlesize":     11,
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "axes.grid":            True,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.5,
    "lines.linewidth":      1.5,
})

# Colorblind-safe palette (Wong 2011)
PLANNER_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
]

METRICS = [
    ("rmse_history",                     r"Normalized RMSE"),
    # ("nlpd_history",                     r"NLPD"),
    ("normalized_trace_reduction_history", r"Normalized trace reduction"),
]


def discover_results(results_dir: str):
    """Returns planners, and data[planner] -> list of run dicts (one per scenario x seed)."""
    data = {}
    for scenario_dir in sorted(glob.glob(os.path.join(results_dir, "scenario_[0-9][0-9][0-9][0-9]"))):
        for pkl_path in glob.glob(os.path.join(scenario_dir, "*History.pkl")):
            planner_name = os.path.basename(pkl_path).replace("History.pkl", "")
            history = joblib.load(pkl_path)
            data.setdefault(planner_name, []).append(history)

    if not data:
        raise FileNotFoundError(f"No results found in {results_dir}")
    return sorted(data.keys()), data


def extract_metric(runs: list, key: str) -> np.ndarray:
    """Stack metric histories -> (n_runs, T), padding shorter runs with last value."""
    arrays = [[float(v) for v in h[key]] for h in runs]
    max_len = max(len(a) for a in arrays)
    padded = [a + [a[-1]] * (max_len - len(a)) for a in arrays]
    return np.array(padded)


def mean_ci(matrix: np.ndarray, z: float = 1.96):
    mu  = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=min(1, matrix.shape[0] - 1)) / np.sqrt(matrix.shape[0])
    return mu, mu - z * sem, mu + z * sem


def plot_comparison(results_dir: str, output_path: str):
    planners, data = discover_results(results_dir)
    color_map = {p: PLANNER_COLORS[i % len(PLANNER_COLORS)] for i, p in enumerate(planners)}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), constrained_layout=True)  # ~\textwidth

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for planner in planners:
            runs = data.get(planner, [])
            if not runs:
                continue
            matrix = extract_metric(runs, metric_key)
            mu, lo, hi = mean_ci(matrix)
            t = np.arange(len(mu))
            color = color_map[planner]
            ax.plot(t, mu, color=color, label=planner)
            ax.fill_between(t, lo, hi, color=color, alpha=0.15)

        ax.set_title(metric_label)
        ax.set_xlabel(r"Time step")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    legend_handles = [
        Line2D([0], [0], color=color_map[p], lw=1.5, label=p) for p in planners
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(planners),
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/evaluation/")
    parser.add_argument("--output", default="results/evaluation/comparison.png")
    args = parser.parse_args()
    plot_comparison(args.results_dir, args.output)