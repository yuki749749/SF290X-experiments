"""
plot_guidance_scale_ablation.py
================================
Plots RMSE and Trace Reduction curves for a sweep over classifier-free
guidance scale w (condition_guidance_w).

Expected Hydra multirun layout
-------------------------------

    results/guidance_scale_ablation/sweep/<timestamp>/
        scenario_0/guidance_scale=0.0/history.pkl
        scenario_0/guidance_scale=1.0/history.pkl
        scenario_0/guidance_scale=2.0/history.pkl
        ...
        scenario_1/...
        ...

The subdir key name (default: "guidance_scale") must match the Hydra
override key used in your multirun command, e.g.

    python guidance_scale_ablation.py \
        --multirun diffusion.condition_guidance_w=0.0,1.0,2.0,4.0,8.0

with the sweep subdir configured as:

    hydra:
      sweep:
        subdir: scenario_${scenario_idx}/guidance_scale=${diffusion.condition_guidance_w}

If your subdir key differs, pass --subdir_key accordingly.

Output
------
    <sweep_root>/guidance_scale_ablation.pdf

Usage
-----
    # Auto-select latest timestamp, default guidance scales:
    python plot_guidance_scale_ablation.py

    # Specific timestamp + scales:
    python plot_guidance_scale_ablation.py \\
        --sweep_dir results/guidance_scale_ablation/sweep/2026-04-15_12-00-00 \\
        --scales 0.0 1.0 2.0 4.0 8.0

    # Different subdir key:
    python plot_guidance_scale_ablation.py --subdir_key w
"""

import argparse
import itertools
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style, FIGURE_SIZES

apply_style()

# Default guidance scales to plot. Override via --scales.
DEFAULT_SCALES = [0.0, 0.5, 1.0, 1.2, 1.5, 2, 5, 10]

# Linestyles cycle for accessibility (complements the colormap).
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

METRICS = [
    ("rmse_history",                     "Normalized RMSE",            "lower is better"),
    ("normalizedTraceReduction_history", "Normalized Trace Reduction", "higher is better"),
]


# ── Color assignment ───────────────────────────────────────────────────────────

def make_palette(n: int):
    """
    Return n perceptually-ordered colors from a sequential colormap.
    Low w → light/cool, high w → dark/warm, readable on white background.
    """
    cmap = get_cmap("plasma")
    # Sample from 0.15 to 0.85 to avoid extremes that wash out on white.
    return [cmap(0.15 + 0.70 * i / max(n - 1, 1)) for i in range(n)]


# ── Directory resolution ───────────────────────────────────────────────────────

def resolve_sweep_root(sweep_dir: Path) -> Path:
    """
    If sweep_dir already contains scenario_* subdirs, return it directly.
    Otherwise descend into the lexicographically latest subdirectory
    (auto-selects the most recent timestamp).
    """
    if any(sweep_dir.glob("scenario_*")):
        return sweep_dir
    subdirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {sweep_dir}")
    latest = subdirs[-1]
    print(f"Auto-selected timestamp: {latest.name}")
    return latest


def discover_scales(sweep_root: Path, subdir_key: str) -> list[float]:
    """
    Infer available guidance scale values from directory names if --scales
    is not provided.  Matches dirs of the form  <subdir_key>=<number>.
    """
    prefix = f"{subdir_key}="
    found = set()
    for p in sweep_root.glob(f"scenario_*/{prefix}*"):
        suffix = p.name[len(prefix):]
        try:
            found.add(float(suffix))
        except ValueError:
            pass
    if not found:
        raise FileNotFoundError(
            f"Could not discover any '{prefix}<value>' subdirs under {sweep_root}. "
            "Pass --scales explicitly."
        )
    discovered = sorted(found)
    print(f"Discovered guidance scales: {discovered}")
    return discovered


# ── Data loading ───────────────────────────────────────────────────────────────

def format_scale(w: float) -> str:
    """Format a float scale value as it appears in the Hydra subdir name."""
    # Hydra preserves the literal string from the override, so 1.0 → "1.0".
    # Use g-format to strip unnecessary trailing zeros for non-integer values,
    # but keep at least one decimal so it matches typical Hydra output.
    s = f"{w:g}"
    if "." not in s:
        s += ".0"
    return s


def load_variant(
    sweep_root: Path, subdir_key: str, scale: float, metric_key: str
) -> np.ndarray:
    """
    Collect history.pkl files for one guidance scale across all scenarios.

    Layout:  sweep_root / scenario_* / <subdir_key>=<scale> / history.pkl

    Returns np.ndarray of shape (n_scenarios, T).
    """
    dir_suffix = f"{subdir_key}={format_scale(scale)}"
    pkl_files  = sorted(sweep_root.glob(f"scenario_*/{dir_suffix}/history.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No history.pkl files found for '{dir_suffix}' under {sweep_root}"
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


def mean_ci(matrix: np.ndarray, z: float = 1.96):
    mu  = matrix.mean(axis=0)
    sem = (
        matrix.std(axis=0, ddof=min(1, matrix.shape[0] - 1))
        / np.sqrt(matrix.shape[0])
    )
    return mu, mu - z * sem, mu + z * sem


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_ablation(
    sweep_root: Path,
    output_path: Path,
    scales: list[float],
    subdir_key: str,
) -> None:
    colors    = make_palette(len(scales))
    lss       = list(itertools.islice(itertools.cycle(LINESTYLES), len(scales)))

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)

    for ax, (metric_key, metric_label, direction) in zip(axes, METRICS):
        for scale, color, ls in zip(scales, colors, lss):
            label = f"$w = {format_scale(scale)}$"
            try:
                matrix = load_variant(sweep_root, subdir_key, scale, metric_key)
            except (FileNotFoundError, KeyError) as e:
                print(f"  Warning: {e}")
                continue

            mu, lo, hi = mean_ci(matrix)
            t = np.arange(len(mu))
            ax.plot(t, mu, color=color, linestyle=ls, label=label)
            ax.fill_between(t, lo, hi, color=color, alpha=0.12)
            print(
                f"  w={format_scale(scale):>6s}  {matrix.shape[0]} scenarios, "
                f"T={len(mu)},  final {metric_label}: {mu[-1]:.4f}"
            )

        ax.set_title(f"{metric_label}\n({direction})", fontsize=9)
        ax.set_xlabel("Time step")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    legend_handles = [
        Line2D([0], [0], color=c, linestyle=ls, lw=1.5,
               label=f"$w = {format_scale(w)}$")
        for w, c, ls in zip(scales, colors, lss)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(scales), 5),
        frameon=False,
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot guidance-scale ablation results from a Hydra multirun."
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/guidance_scale_ablation/sweep"),
        help="Sweep root or specific timestamp subdir (default: auto-latest).",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Guidance scale values to plot (default: auto-discover from dirs, "
            f"fallback to {DEFAULT_SCALES})."
        ),
    )
    parser.add_argument(
        "--subdir_key",
        type=str,
        default="guidance_scale",
        help=(
            "Hydra subdir key for the guidance scale override "
            "(default: 'guidance_scale'). Must match your sweep subdir template, e.g. "
            "'guidance_scale' if subdirs are named 'guidance_scale=2.0'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <sweep_root>/guidance_scale_ablation.pdf).",
    )
    args = parser.parse_args()

    sweep_root  = resolve_sweep_root(args.sweep_dir)
    output_path = args.output or (sweep_root / "guidance_scale_ablation.pdf")

    # Resolve scales: CLI > auto-discover > hardcoded default
    if args.scales:
        scales = sorted(args.scales)
    else:
        try:
            scales = discover_scales(sweep_root, args.subdir_key)
        except FileNotFoundError:
            print(f"  Could not auto-discover scales; using defaults: {DEFAULT_SCALES}")
            scales = DEFAULT_SCALES

    print(f"Sweep root    : {sweep_root}")
    print(f"Output        : {output_path}")
    print(f"Subdir key    : {args.subdir_key}")
    print(f"Scales        : {scales}")
    print()

    plot_ablation(sweep_root, output_path, scales, args.subdir_key)


if __name__ == "__main__":
    main()