"""
plot_guidance_weights_sweep.py
==============================
Plots final-step RMSE and Trace Reduction as 2-D heatmaps over the
(belief_guidance_w, return_guidance_w) grid produced by guidance_weights_sweep.py.

Expected Hydra multirun layout
-------------------------------
    results/guidance_weights_sweep/sweep/<timestamp>/
        scenario_0/w_belief=0.5,w_return=0.5/history.pkl
        scenario_0/w_belief=0.5,w_return=1.0/history.pkl
        ...
        scenario_1/...

Usage
-----
    # Auto-select latest sweep:
    python plot_guidance_weights_sweep.py

    # Specific timestamp:
    python plot_guidance_weights_sweep.py \\
        --sweep_dir results/guidance_weights_sweep/sweep/2026-05-13_12-00-00

    # Override subdir key names if you used a different config:
    python plot_guidance_weights_sweep.py --belief_key w_b --return_key w_r

Output
------
    <sweep_root>/guidance_weights_sweep.pdf
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style

apply_style(grid=False)

METRICS = [
    ("rmse_history",                     "Final RMSE",            "lower is better",   "Blues_r"),
    ("normalizedTraceReduction_history", "Final Trace Reduction", "higher is better",  "Blues"),
    ("fraction_in_domain",               "Fraction in Domain",    "higher is better",  "Blues"),
]


# ── Directory helpers ──────────────────────────────────────────────────────────

def resolve_sweep_root(sweep_dir: Path) -> Path:
    if any(sweep_dir.glob("scenario_*")):
        return sweep_dir
    subdirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {sweep_dir}")
    latest = subdirs[-1]
    print(f"Auto-selected timestamp: {latest.name}")
    return latest


def parse_weight(s: str) -> float:
    return float(s)


def discover_grid(
    sweep_root: Path, belief_key: str, return_key: str
) -> tuple[list[float], list[float]]:
    """
    Scan scenario_* dirs for subdirs of the form
        w_belief=<float>,w_return=<float>
    and return sorted unique lists of belief and return weight values.
    """
    pattern = re.compile(
        rf"{re.escape(belief_key)}=([^,]+),{re.escape(return_key)}=(.+)"
    )
    belief_vals, return_vals = set(), set()
    for p in sweep_root.glob("scenario_*/*"):
        m = pattern.fullmatch(p.name)
        if m:
            belief_vals.add(parse_weight(m.group(1)))
            return_vals.add(parse_weight(m.group(2)))
    if not belief_vals:
        raise FileNotFoundError(
            f"No grid subdirs matching '{belief_key}=*,{return_key}=*' found under {sweep_root}"
        )
    return sorted(belief_vals), sorted(return_vals)


# ── Data loading ───────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format a float to match Hydra's subdir naming (e.g. 1.0 → '1.0')."""
    s = f"{v:g}"
    if "." not in s:
        s += ".0"
    return s


def load_cell(
    sweep_root: Path,
    belief_key: str,
    return_key: str,
    w_b: float,
    w_r: float,
    metric_key: str,
    domain_size: tuple[float, float] = (15.0, 15.0),
) -> float | None:
    """
    Load all history.pkl files for a given (w_b, w_r) cell, average the
    final-step metric across scenarios, and return the mean.
    Returns None if no files are found.

    ``fraction_in_domain`` is computed from ``position_history`` rather than
    read directly from the stored history arrays.
    """
    subdir = f"{belief_key}={_fmt(w_b)},{return_key}={_fmt(w_r)}"
    pkl_files = sorted(sweep_root.glob(f"scenario_*/{subdir}/history.pkl"))
    if not pkl_files:
        return None

    finals = []
    for pkl in pkl_files:
        h = joblib.load(pkl)
        if metric_key == "fraction_in_domain":
            pos = h.get("position_history")
            if pos is None:
                continue
            W, H = domain_size
            arr = np.array(pos)
            inside = (
                (arr[:, 0] >= 0) & (arr[:, 0] <= W) &
                (arr[:, 1] >= 0) & (arr[:, 1] <= H)
            )
            finals.append(float(inside.mean()))
        else:
            if metric_key not in h:
                continue
            vals = h[metric_key]
            if vals:
                finals.append(float(vals[-1]))

    return float(np.mean(finals)) if finals else None


def build_grid(
    sweep_root: Path,
    belief_key: str,
    return_key: str,
    belief_vals: list[float],
    return_vals: list[float],
    metric_key: str,
    domain_size: tuple[float, float] = (15.0, 15.0),
) -> np.ndarray:
    """
    Returns a (len(belief_vals), len(return_vals)) array of mean final metric
    values.  Missing cells are NaN.
    """
    grid = np.full((len(belief_vals), len(return_vals)), np.nan)
    for i, w_b in enumerate(belief_vals):
        for j, w_r in enumerate(return_vals):
            val = load_cell(sweep_root, belief_key, return_key, w_b, w_r, metric_key, domain_size)
            if val is not None:
                grid[i, j] = val
    return grid


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_sweep(
    sweep_root: Path,
    output_path: Path,
    belief_key: str,
    return_key: str,
    belief_vals: list[float],
    return_vals: list[float],
    domain_size: tuple[float, float] = (15.0, 15.0),
) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.6 * len(METRICS), 3.0), constrained_layout=True)

    for ax, (metric_key, metric_label, direction, cmap) in zip(axes, METRICS):
        grid = build_grid(
            sweep_root, belief_key, return_key,
            belief_vals, return_vals, metric_key, domain_size,
        )

        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, shrink=0.85)

        # annotate cells
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        for i in range(len(belief_vals)):
            for j in range(len(return_vals)):
                val = grid[i, j]
                if not np.isnan(val):
                    brightness = (val - vmin) / (vmax - vmin + 1e-12)
                    text_color = "white" if brightness < 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6, color=text_color)

        ax.set_xticks(range(len(return_vals)))
        ax.set_xticklabels([_fmt(v) for v in return_vals], rotation=45, ha="right")
        ax.set_yticks(range(len(belief_vals)))
        ax.set_yticklabels([_fmt(v) for v in belief_vals])
        ax.set_xlabel(f"$w_r$ (return guidance)")
        ax.set_ylabel(f"$w_b$ (belief guidance)")
        ax.set_title(f"{metric_label}\n({direction})", fontsize=9)

        # mark the best cell
        best_idx = np.nanargmin(grid) if "lower" in direction else np.nanargmax(grid)
        bi, bj = np.unravel_index(best_idx, grid.shape)
        ax.add_patch(plt.Rectangle(
            (bj - 0.5, bi - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=1.5,
        ))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot belief/return guidance weight sweep as 2-D heatmaps."
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/guidance_weights_sweep/sweep"),
        help="Sweep root or specific timestamp subdir (default: auto-latest).",
    )
    parser.add_argument(
        "--belief_key", type=str, default="w_belief",
        help="Subdir key for belief_guidance_w (default: 'w_belief').",
    )
    parser.add_argument(
        "--return_key", type=str, default="w_return",
        help="Subdir key for return_guidance_w (default: 'w_return').",
    )
    parser.add_argument(
        "--domain_size",
        type=float,
        nargs=2,
        default=[15.0, 15.0],
        metavar=("W", "H"),
        help="Physical domain size used during evaluation (default: 15 15).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PDF path (default: <sweep_root>/guidance_weights_sweep.pdf).",
    )
    args = parser.parse_args()

    sweep_root  = resolve_sweep_root(args.sweep_dir)
    output_path = args.output or (sweep_root / "guidance_weights_sweep.pdf")

    belief_vals, return_vals = discover_grid(sweep_root, args.belief_key, args.return_key)
    print(f"Sweep root    : {sweep_root}")
    print(f"Output        : {output_path}")
    print(f"belief_guidance_w values : {belief_vals}")
    print(f"return_guidance_w values : {return_vals}")
    print()

    plot_sweep(sweep_root, output_path, args.belief_key, args.return_key,
               belief_vals, return_vals, tuple(args.domain_size))


if __name__ == "__main__":
    main()
