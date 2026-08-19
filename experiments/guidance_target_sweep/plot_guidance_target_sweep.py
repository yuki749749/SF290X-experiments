"""
plot_guidance_target_sweep.py
==============================
Heatmaps of final RMSE and Trace Reduction for a joint sweep over
classifier-free guidance scale (condition_guidance_w) and target return.

Expected Hydra multirun layout
-------------------------------

    results/guidance_target_sweep/sweep/<timestamp>/
        scenario_0/guidance_scale=1.0,target_return=0.5/history.pkl
        scenario_0/guidance_scale=1.0,target_return=1.0/history.pkl
        scenario_0/guidance_scale=2.0,target_return=0.5/history.pkl
        ...
        scenario_1/...
        ...

Run command example
-------------------

    python guidance_target_sweep.py \\
        --multirun \\
        diffusion.condition_guidance_w=0.5,1.0,2.0,5.0,10.0 \\
        planner.target_return=0.5,0.7,1.0,1.2,1.5 \\
        scenario_idx=0,1,2,3,4,6,7,8,9,10

Output
------
    <sweep_root>/guidance_target_heatmap.pdf   — heatmaps of final metric values
    <sweep_root>/guidance_target_curves.pdf    — time-series for every combination

Usage
-----
    # Auto-select latest timestamp:
    python plot_guidance_target_sweep.py

    # Specific sweep directory:
    python plot_guidance_target_sweep.py \\
        --sweep_dir results/guidance_target_sweep/sweep/2026-05-10_12-00-00
"""

import argparse
import itertools
import re
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from plot_style import apply_style, FIGURE_SIZES
from utils import resolve_sweep_root

apply_style()

METRICS = [
    ("rmse_history",                     "Final RMSE",             "lower is better",  True),
    ("normalizedTraceReduction_history", "Final Trace Reduction",  "higher is better", False),
]

SUBDIR_RE = re.compile(
    r"guidance_scale=([\d.eE+\-]+),target_return=([\d.eE+\-]+)"
)

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


# ── Directory helpers ──────────────────────────────────────────────────────────


def discover_combinations(sweep_root: Path) -> tuple[list[float], list[float]]:
    """Return (sorted guidance scales, sorted target returns) from dir names."""
    scales, returns = set(), set()
    for p in sweep_root.glob("scenario_*/guidance_scale=*,target_return=*"):
        m = SUBDIR_RE.search(p.name)
        if m:
            scales.add(float(m.group(1)))
            returns.add(float(m.group(2)))
    if not scales:
        raise FileNotFoundError(
            f"No matching subdirs under {sweep_root}. "
            "Expected: scenario_*/guidance_scale=<w>,target_return=<r>"
        )
    return sorted(scales), sorted(returns)


def fmt(v: float) -> str:
    """Format a float the same way Hydra writes it in the subdir name."""
    s = f"{v:g}"
    return s + ".0" if "." not in s else s


# ── Data loading ───────────────────────────────────────────────────────────────

def load_final_metric(
    sweep_root: Path,
    guidance_scale: float,
    target_return: float,
    metric_key: str,
) -> tuple[float, float] | tuple[None, None]:
    """
    Average the final time-step value of metric_key across all scenarios
    for a given (guidance_scale, target_return) pair.

    Returns (mean, sem) or (None, None) if no data found.
    """
    subdir = f"guidance_scale={fmt(guidance_scale)},target_return={fmt(target_return)}"
    pkl_files = sorted(sweep_root.glob(f"scenario_*/{subdir}/history.pkl"))
    if not pkl_files:
        return None, None

    finals = []
    for pkl in pkl_files:
        h = joblib.load(pkl)
        if metric_key not in h:
            continue
        series = h[metric_key]
        finals.append(float(series[-1]))

    if not finals:
        return None, None

    arr = np.array(finals)
    mu = arr.mean()
    sem = arr.std(ddof=min(1, len(arr) - 1)) / np.sqrt(len(arr))
    return mu, sem


def load_time_series(
    sweep_root: Path,
    guidance_scale: float,
    target_return: float,
    metric_key: str,
) -> np.ndarray | None:
    """Load full time-series (n_scenarios, T) for a given config pair."""
    subdir = f"guidance_scale={fmt(guidance_scale)},target_return={fmt(target_return)}"
    pkl_files = sorted(sweep_root.glob(f"scenario_*/{subdir}/history.pkl"))
    if not pkl_files:
        return None

    arrays = []
    for pkl in pkl_files:
        h = joblib.load(pkl)
        if metric_key not in h:
            continue
        arrays.append([float(v) for v in h[metric_key]])

    if not arrays:
        return None

    max_len = max(len(a) for a in arrays)
    padded = [a + [a[-1]] * (max_len - len(a)) for a in arrays]
    return np.array(padded)


def mean_ci(matrix: np.ndarray, z: float = 1.96):
    mu = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=min(1, matrix.shape[0] - 1)) / np.sqrt(matrix.shape[0])
    return mu, mu - z * sem, mu + z * sem


# ── Heatmap plot ───────────────────────────────────────────────────────────────

def plot_heatmaps(
    sweep_root: Path,
    output_path: Path,
    scales: list[float],
    returns: list[float],
) -> None:
    """2×2 figure: heatmap of mean final value + heatmap of SEM, for both metrics."""
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(3.6 * n_metrics, 3.2), constrained_layout=True
    )

    for ax, (metric_key, metric_label, direction, lower_is_better) in zip(axes, METRICS):
        grid_mu = np.full((len(returns), len(scales)), np.nan)

        for j, w in enumerate(scales):
            for i, r in enumerate(returns):
                mu, _ = load_final_metric(sweep_root, w, r, metric_key)
                if mu is not None:
                    grid_mu[i, j] = mu

        cmap = "RdYlGn_r" if lower_is_better else "RdYlGn"
        im = ax.imshow(
            grid_mu,
            aspect="auto",
            cmap=cmap,
            origin="lower",
        )
        plt.colorbar(im, ax=ax, shrink=0.85)

        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([fmt(w) for w in scales], rotation=45, ha="right")
        ax.set_yticks(range(len(returns)))
        ax.set_yticklabels([fmt(r) for r in returns])
        ax.set_xlabel("Guidance scale $w$")
        ax.set_ylabel("Target return $r$")
        ax.set_title(f"{metric_label}\n({direction})", fontsize=9)

        # Annotate each cell with its value
        for j in range(len(scales)):
            for i in range(len(returns)):
                val = grid_mu[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=6.5,
                        color="black",
                    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap → {output_path}")


# ── Time-series plot ───────────────────────────────────────────────────────────

def plot_curves(
    sweep_root: Path,
    output_path: Path,
    scales: list[float],
    returns: list[float],
) -> None:
    """
    Time-series curves for every (guidance_scale, target_return) combination,
    one subplot per metric.  Lines are colored by guidance scale; linestyle
    cycles over target return values.
    """
    n_combos = len(scales) * len(returns)
    cmap = get_cmap("plasma")
    scale_colors = {w: cmap(0.15 + 0.70 * j / max(len(scales) - 1, 1))
                    for j, w in enumerate(scales)}
    ls_cycle = list(itertools.islice(itertools.cycle(LINESTYLES), len(returns)))
    return_ls = {r: ls_cycle[i] for i, r in enumerate(returns)}

    fig, axes = plt.subplots(
        1, len(METRICS), figsize=FIGURE_SIZES["double_col"], constrained_layout=True
    )

    for ax, (metric_key, metric_label, direction, _) in zip(axes, METRICS):
        for w in scales:
            for r in returns:
                matrix = load_time_series(sweep_root, w, r, metric_key)
                if matrix is None:
                    continue
                mu, lo, hi = mean_ci(matrix)
                t = np.arange(len(mu))
                color = scale_colors[w]
                ls = return_ls[r]
                ax.plot(t, mu, color=color, linestyle=ls, linewidth=1.0)
                ax.fill_between(t, lo, hi, color=color, alpha=0.07)

        ax.set_title(f"{metric_label}\n({direction})", fontsize=9)
        ax.set_xlabel("Time step")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Legend: guidance scale (color) + target return (linestyle)
    scale_handles = [
        Line2D([0], [0], color=scale_colors[w], lw=1.5, label=f"$w={fmt(w)}$")
        for w in scales
    ]
    return_handles = [
        Line2D([0], [0], color="gray", linestyle=return_ls[r], lw=1.5,
               label=f"$r={fmt(r)}$")
        for r in returns
    ]
    fig.legend(
        handles=scale_handles + return_handles,
        loc="lower center",
        ncol=min(n_combos, 6),
        frameon=False,
        bbox_to_anchor=(0.5, -0.22),
        fontsize=7,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved curves  → {output_path}")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(
    sweep_root: Path,
    scales: list[float],
    returns: list[float],
) -> None:
    print("\n── Final-step metric summary (mean ± SEM across scenarios) ──")
    header = f"{'guidance_scale':>15}  {'target_return':>13}"
    for _, label, _, _ in METRICS:
        header += f"  {label:>24}"
    print(header)

    for w in scales:
        for r in returns:
            row = f"{fmt(w):>15}  {fmt(r):>13}"
            for metric_key, _, _, _ in METRICS:
                mu, sem = load_final_metric(sweep_root, w, r, metric_key)
                if mu is None:
                    row += f"  {'—':>24}"
                else:
                    row += f"  {mu:>10.4f} ± {sem:<10.4f}"
            print(row)
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot joint guidance-scale / target-return sweep results."
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/guidance_target_sweep/sweep"),
        help="Sweep root or specific timestamp subdir (default: auto-latest).",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        help="Guidance scale values (default: auto-discover).",
    )
    parser.add_argument(
        "--returns",
        type=float,
        nargs="+",
        default=None,
        help="Target return values (default: auto-discover).",
    )
    parser.add_argument(
        "--no_curves",
        action="store_true",
        help="Skip the time-series curves plot.",
    )
    args = parser.parse_args()

    sweep_root = resolve_sweep_root(args.sweep_dir)
    print(f"Sweep root: {sweep_root}\n")

    if args.scales and args.returns:
        scales = sorted(args.scales)
        returns = sorted(args.returns)
    else:
        disc_scales, disc_returns = discover_combinations(sweep_root)
        scales = sorted(args.scales) if args.scales else disc_scales
        returns = sorted(args.returns) if args.returns else disc_returns

    print(f"Guidance scales : {scales}")
    print(f"Target returns  : {returns}")

    print_summary(sweep_root, scales, returns)

    plot_heatmaps(
        sweep_root,
        sweep_root / "guidance_target_heatmap.pdf",
        scales,
        returns,
    )

    if not args.no_curves:
        plot_curves(
            sweep_root,
            sweep_root / "guidance_target_curves.pdf",
            scales,
            returns,
        )


if __name__ == "__main__":
    main()
