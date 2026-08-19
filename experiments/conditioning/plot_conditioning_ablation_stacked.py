"""
plot_conditioning_ablation_stacked.py
======================================
Stacked (2×1) slide version of the conditioning ablation plot.

Same data source as plot_conditioning_ablation.py, but laid out as
2 rows × 1 column (Normalized RMSE on top, Normalized Trace Reduction
on bottom) with larger fonts suited for presentation slides.

Usage
-----
    # Auto-select latest timestamp:
    python plot_conditioning_ablation_stacked.py

    # Specific timestamp:
    python plot_conditioning_ablation_stacked.py \
        --sweep_dir results/conditioning_ablation/sweep/2026-04-15_12-00-00

    # Custom output:
    python plot_conditioning_ablation_stacked.py --output slides_ablation.pdf

Output: <sweep_root>/conditioning_ablation_stacked.pdf
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
from utils import resolve_sweep_root

# ── Slide-tuned style ─────────────────────────────────────────────────────────

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


VARIANTS = [
    (True,  True,  "use_belief=True,use_return=True",   "Belief + Reward",  "#0072B2", "-"),
    (True,  False, "use_belief=True,use_return=False",  "Belief only",      "#E69F00", "--"),
    (False, True,  "use_belief=False,use_return=True",  "Reward only",      "#009E73", "-."),
    (False, False, "use_belief=False,use_return=False", "Unconditioned",    "#555555", ":"),
]

METRICS = [
    ("rmse_history",                     "NRMSE"),
]


# ── Directory resolution ───────────────────────────────────────────────────────


# ── Data loading ───────────────────────────────────────────────────────────────

def load_variant(sweep_root: Path, dir_suffix: str, metric_key: str) -> np.ndarray:
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
    return np.array(padded)


def mean_ci(
    matrix: np.ndarray,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    n  = matrix.shape[0]
    mu = matrix.mean(axis=0)
    if n == 1:
        return mu, mu.copy(), mu.copy()
    idx        = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = matrix[idx].mean(axis=1)
    lo, hi     = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return mu, lo, hi


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_stacked(sweep_root: Path, output_path: Path) -> None:
    rng = np.random.default_rng()

    fig, axes = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
    axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
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
            print(f"  {label:30s} — {matrix.shape[0]} scenarios, T={len(mu)}")

        ax.set_ylabel(metric_label)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time step")

    legend_handles = [
        Line2D([0], [0], color=color, linestyle=ls, lw=2.5, label=label)
        for (_, _, _, label, color, ls) in VARIANTS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.13),
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot stacked conditioning ablation results (slide layout)."
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
        help="Output PDF path (default: <sweep_root>/conditioning_ablation_stacked.pdf).",
    )
    args = parser.parse_args()

    apply_slide_style()

    sweep_root  = resolve_sweep_root(args.sweep_dir)
    output_path = args.output or (sweep_root / "conditioning_ablation_stacked.pdf")

    print(f"Sweep root : {sweep_root}")
    print(f"Output     : {output_path}")

    plot_stacked(sweep_root, output_path)


if __name__ == "__main__":
    main()
