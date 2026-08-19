"""
plot_gp_tuning.py
=================
Plot the GP hyperparameter distributions from a GP tuning run and save as PDF.

Usage:
    # Auto-detect latest run under results/gp_tuning/run/
    python experiments/gp_tuning/plot_gp_tuning.py

    # Point to a specific run directory
    python experiments/gp_tuning/plot_gp_tuning.py --dir results/gp_tuning/run/2026-03-20_12-14-17

    # Override the results root
    python experiments/gp_tuning/plot_gp_tuning.py --results-root /abs/path/to/results
"""

import argparse
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from plot_style import apply_style, COLORS, FIGURE_SIZES
from utils import find_latest_run_dir

# Disable grid lines as requested
apply_style(grid=False)


def plot_hyperparameter_distributions(results, output_path: Path):
    # Customize math titles and filter out the noise parameter
    title_mapping = {
        "mean_constant": r"Mean constant $m$",
        "outputscale": r"Outputscale $\sigma^2_f$",
        "lengthscale_0": r"Lengthscale $\ell_1$",
        "lengthscale_1": r"Lengthscale $\ell_2$",
    }
    keys = [k for k in results[0].keys() if k in title_mapping]
    n_keys = len(keys)
    
    # Use standard wide layout for the multi-panel subplot
    fig, axes = plt.subplots(
        1,
        n_keys,
        figsize=FIGURE_SIZES["wide"],
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten()
    
    for i, key in enumerate(keys):
        values = [result[key] for result in results]
        
        # Wong palette blue for histograms
        axes[i].hist(
            values,
            bins=40,
            alpha=0.7,
            color=COLORS["blue"],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[i].set_title(title_mapping[key])
        
        # Wong palette vermillion for median vertical line
        median_val = np.median(values)
        axes[i].axvline(
            median_val,
            color=COLORS["vermillion"],
            linestyle="--",
            linewidth=1.2,
            label=f"Median: {median_val:.3f}",
        )
        axes[i].legend(frameon=False)
        
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GP hyperparameter distributions.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Path to a specific Hydra run directory containing gp_tuning_results.pkl."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Project results root (default: results/). Ignored when --dir is given."
    )
    args = parser.parse_args()

    run_dir: Path = (
        args.dir if args.dir is not None
        else find_latest_run_dir(args.results_root, "gp_tuning", "gp_tuning_results.pkl")
    )

    results_file = run_dir / "gp_tuning_results.pkl"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    print(f"Loading GP tuning results from: {results_file}")
    data = joblib.load(results_file)
    results = data['per_scenario']

    output_path = run_dir / "gp_hyperparameter_distributions.pdf"
    plot_hyperparameter_distributions(results, output_path)


if __name__ == "__main__":
    main()
