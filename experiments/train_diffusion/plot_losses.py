"""
plot_training_loss.py

Plot train/val loss from a Hydra training run and save as PDF.

Usage:
    # Auto-detect latest run under results/train_diffusion/run/
    python plot_training_loss.py

    # Point to a specific run directory
    python plot_training_loss.py --dir results/train_diffusion/run/2026-03-20_12-14-17

    # Override the results root
    python plot_training_loss.py --results-root /abs/path/to/results
"""

import argparse
import csv
from pathlib import Path

import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from plot_style import apply_style, COLORS, FIGURE_SIZES
from utils import find_latest_run_dir

apply_style()

TRAIN_COLOR = COLORS["blue"]
VAL_COLOR   = COLORS["vermillion"]


def load_csv(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs, train_losses, val_losses = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return epochs, train_losses, val_losses


def plot_losses(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    out_path: Path,
) -> None:
    best_epoch = epochs[val_losses.index(min(val_losses))]
    best_val = min(val_losses)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])

    ax.plot(epochs, train_losses, color=TRAIN_COLOR, linewidth=1.8, label="Train loss")
    ax.plot(epochs, val_losses, color=VAL_COLOR, linewidth=1.8, linestyle="--", label="Val loss")

    # Mark best validation epoch
    # ax.axvline(best_epoch, color=VAL_COLOR, linewidth=0.8, linestyle=":", alpha=0.7)
    # ax.scatter([best_epoch], [best_val], color=VAL_COLOR, zorder=5, s=40)
    # ax.annotate(
    #     f"best val = {best_val:.4f}\n(epoch {best_epoch})",
    #     xy=(best_epoch, best_val),
    #     xytext=(10, 12),
    #     textcoords="offset points",
    #     fontsize=9,
    #     color=VAL_COLOR,
    # )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot -> {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training/val loss as PDF.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=(
            "Path to a specific Hydra run directory containing training_log.csv. "
            "Example: results/train_diffusion/run/2026-03-20_12-14-17. "
            "Defaults to the latest run found under --results-root."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Project results root (default: results/). Ignored when --dir is given.",
    )
    args = parser.parse_args()

    run_dir: Path = (
        args.dir if args.dir is not None
        else find_latest_run_dir(args.results_root, "train_diffusion", "training_log.csv")
    )

    csv_path = run_dir / "training_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading losses from: {csv_path}")
    epochs, train_losses, val_losses = load_csv(csv_path)

    if not epochs:
        raise ValueError("CSV is empty — has training started yet?")

    out_path = run_dir / "training_loss.pdf"
    plot_losses(epochs, train_losses, val_losses, out_path)


if __name__ == "__main__":
    main()