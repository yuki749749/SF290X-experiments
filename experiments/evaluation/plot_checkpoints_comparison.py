"""
plot_checkpoints_comparison.py
=============================
Standalone plotting script to compare evaluated checkpoints.

Loads aggregated metrics and the name-to-epoch mapping from a completed
evaluate_checkpoints.py run and generates comparison plots.

Usage
-----
    # Auto-detect the latest evaluation run and plot:
    python experiments/evaluation/plot_checkpoints_comparison.py

    # Plot a specific evaluation run:
    python experiments/evaluation/plot_checkpoints_comparison.py --dir results/evaluate_checkpoints/run/2026-08-26_12-44-48
"""

import argparse
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project src and experiments directories to system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_style import apply_style, FIGURE_SIZES, COLORS
from evaluation_utils import discover_results, extract_metric, mean_ci


def plot_perf_vs_epochs(df_agg, name_to_epoch, output_path):
    """Plot Final NRMSE and Fraction in Domain vs. Epochs."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5), constrained_layout=True)

    # Filter epoch checkpoints
    epoch_data = []
    for _, row in df_agg.iterrows():
        name = row['config']
        if name in ['best_checkpoint', 'final_checkpoint']:
            continue
        
        epoch = name_to_epoch.get(name, None)
        if epoch is not None:
            epoch_data.append((epoch, row))

    # Sort by epoch
    epoch_data.sort(key=lambda x: x[0])
    
    if epoch_data:
        epochs = [x[0] for x in epoch_data]
        
        # NRMSE
        nrmse_mean = [x[1]['final_nrmse_mean'] for x in epoch_data]
        nrmse_lo = [x[1]['final_nrmse_ci95_lo'] for x in epoch_data]
        nrmse_hi = [x[1]['final_nrmse_ci95_hi'] for x in epoch_data]
        
        ax1.plot(epochs, nrmse_mean, color=COLORS["blue"], marker='o', label='Checkpoints')
        ax1.fill_between(epochs, nrmse_lo, nrmse_hi, color=COLORS["blue"], alpha=0.15)
        
        # Fraction in Domain
        fid_mean = [x[1]['fraction_in_domain_mean'] for x in epoch_data]
        fid_lo = [x[1]['fraction_in_domain_ci95_lo'] for x in epoch_data]
        fid_hi = [x[1]['fraction_in_domain_ci95_hi'] for x in epoch_data]
        
        ax2.plot(epochs, fid_mean, color=COLORS["blue"], marker='o', label='Checkpoints')
        ax2.fill_between(epochs, fid_lo, fid_hi, color=COLORS["blue"], alpha=0.15)
        
    # Plot special checkpoints
    special_colors = {
        'best_checkpoint': COLORS["vermillion"],
        'final_checkpoint': COLORS["green"],
    }
    special_names = {
        'best_checkpoint': 'Best Checkpoint',
        'final_checkpoint': 'Final Checkpoint',
    }

    for name in ['best_checkpoint', 'final_checkpoint']:
        if name in df_agg['config'].values:
            row = df_agg[df_agg['config'] == name].iloc[0]
            epoch = name_to_epoch.get(name, None)
            if epoch is not None:
                color = special_colors[name]
                display_name = f"{special_names[name]} (epoch {epoch})"
                
                # Plot on ax1 (NRMSE)
                ax1.scatter(epoch, row['final_nrmse_mean'], color=color, marker='*', s=150, zorder=5, label=display_name)
                # Plot on ax2 (FID)
                ax2.scatter(epoch, row['fraction_in_domain_mean'], color=color, marker='*', s=150, zorder=5, label=display_name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Final NRMSE")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Fraction in Domain")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Shared legend below subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        frameon=False,
    )

    # Save PDF and PNG
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Epochs comparison plot -> {output_path}")
    
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved Epochs comparison plot -> {png_path}")


def plot_trajectory_error_over_time(sweep_dir, name_to_epoch, output_path):
    """Plot Trajectory RMSE over time step for different checkpoints."""
    apply_style()
    data = discover_results(sweep_dir)
    
    # Sort planners by epoch
    epoch_checkpoints = []
    special_checkpoints = []
    
    for planner_name in data:
        epoch = name_to_epoch.get(planner_name, None)
        if epoch is not None:
            if planner_name in ['best_checkpoint', 'final_checkpoint']:
                special_checkpoints.append((epoch, planner_name))
            else:
                epoch_checkpoints.append((epoch, planner_name))
                
    # Sort epoch checkpoints
    epoch_checkpoints.sort(key=lambda x: x[0])
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    
    # Create a colormap for epoch checkpoints to show evolution
    import matplotlib.cm as cm
    n_epochs = len(epoch_checkpoints)
    colors = cm.Blues(np.linspace(0.4, 1.0, n_epochs)) if n_epochs > 0 else []
    
    # Plot epoch checkpoints
    for idx, (epoch, name) in enumerate(epoch_checkpoints):
        runs = data[name]
        matrix = extract_metric(runs, "rmse_history")
        mu, lo, hi = mean_ci(matrix)
        t = np.arange(len(mu))
        
        color = colors[idx]
        ax.plot(t, mu, color=color, alpha=0.7, label=f"Epoch {epoch}")
        
    # Plot special checkpoints
    special_styles = {
        'best_checkpoint': (COLORS["vermillion"], '--', 'Best Checkpoint'),
        'final_checkpoint': (COLORS["green"], ':', 'Final Checkpoint'),
    }
    
    for epoch, name in special_checkpoints:
        if name in special_styles:
            color, linestyle, label = special_styles[name]
            runs = data[name]
            matrix = extract_metric(runs, "rmse_history")
            mu, lo, hi = mean_ci(matrix)
            t = np.arange(len(mu))
            
            ax.plot(t, mu, color=color, linestyle=linestyle, linewidth=2.0, label=f"{label} (Epoch {epoch})")
            ax.fill_between(t, lo, hi, color=color, alpha=0.1)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Trajectory Error over Time")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Place legend below the plot to avoid squashing the axes width
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    
    # Save PDF and PNG
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Trajectory error over time plot -> {output_path}")
    
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved Trajectory error over time plot -> {png_path}")


def find_latest_eval_dir() -> Path:
    """Find the most recent evaluation subdirectory that contains results."""
    run_dir = Path("results/evaluate_checkpoints/run")
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Evaluation run directory {run_dir} does not exist. Run evaluate_checkpoints.py first."
        )
    subdirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and (d / "metrics_aggregated.csv").exists()])
    if not subdirs:
        raise FileNotFoundError(f"No completed evaluation runs found under {run_dir}.")
    return subdirs[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from a completed evaluate_checkpoints.py run."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Evaluation run directory. Defaults to the latest run.",
    )
    args = parser.parse_args()

    # Find the target evaluation directory
    if args.dir is None:
        try:
            eval_dir = find_latest_eval_dir()
            print(f"Auto-detected latest evaluation run: {eval_dir}")
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        eval_dir = args.dir
        if not eval_dir.exists():
            print(f"ERROR: Directory '{eval_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)

    # Paths to files
    agg_csv_path = eval_dir / "metrics_aggregated.csv"
    name_to_epoch_json = eval_dir / "name_to_epoch.json"

    if not agg_csv_path.exists():
        print(f"ERROR: 'metrics_aggregated.csv' not found in '{eval_dir}'", file=sys.stderr)
        sys.exit(1)

    if not name_to_epoch_json.exists():
        print(f"ERROR: 'name_to_epoch.json' not found in '{eval_dir}'", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading data...")
    df_agg = pd.read_csv(agg_csv_path)
    with open(name_to_epoch_json, "r") as f:
        name_to_epoch = json.load(f)

    # Plot comparisons
    print("Generating plots...")
    plot_perf_vs_epochs(df_agg, name_to_epoch, eval_dir / "performance_vs_epochs.pdf")
    plot_trajectory_error_over_time(eval_dir, name_to_epoch, eval_dir / "trajectory_error_over_time.pdf")
    
    print("\nPlotting completed successfully!")


if __name__ == "__main__":
    main()
