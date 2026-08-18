import os
import sys
import joblib
import numpy as np
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def find_latest_run_dir(results_root: Path) -> Path:
    """Return the most recently modified run dir under results/gp_tuning/run/*."""
    search_root = results_root / "gp_tuning" / "run"
    candidates = sorted(
        (p for p in search_root.glob("*") if (p / "gp_tuning_results.pkl").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No gp_tuning_results.pkl found under '{search_root}'. "
            "Pass --dir explicitly or check that gp_tuning.py was run."
        )
    latest = candidates[-1]
    print(f"Auto-detected latest run: {latest}")
    return latest


def plot_hyperparameter_distributions(results, output_path: Path):
    keys = results[0].keys()
    n_keys = len(keys)
    fig, axes = plt.subplots(1, n_keys, figsize=(4 * n_keys, 4))
    for i, key in enumerate(keys):
        values = [result[key] for result in results]
        axes[i].hist(values, bins=20, alpha=0.7, color="#4393C3", edgecolor="white", linewidth=0.4)
        axes[i].set_title(key)
        axes[i].axvline(np.median(values), color='red', linestyle='dashed', linewidth=1, label=f'Median: {np.median(values):.3f}')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot -> {output_path}")
    plt.close()


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
        else find_latest_run_dir(args.results_root)
    )

    results_file = run_dir / "gp_tuning_results.pkl"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    print(f"Loading GP tuning results from: {results_file}")
    data = joblib.load(results_file)
    results = data['per_scenario']

    output_path = run_dir / "gp_hyperparameter_distributions.png"
    plot_hyperparameter_distributions(results, output_path)


if __name__ == "__main__":
    main()
