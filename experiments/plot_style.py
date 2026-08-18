"""
plot_style.py
=============
Shared matplotlib style for all experiment plots. Import and call
apply_style() at the top of each plotting script before creating figures.

Usage
-----
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plot_style import apply_style, FIGURE_SIZES, COLORS

    apply_style()            # time-series / metric plots  (grid on)
    apply_style(grid=False)  # spatial field / image plots (grid off)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Wong colorblind-safe palette ───────────────────────────────────────────────
COLORS = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "amber":      "#E69F00",
    "sky":        "#56B4E9",
    "pink":       "#CC79A7",
    "yellow":     "#F0E442",
    "black":      "#000000",
}

# ── Per-planner appearance (shared across evaluation experiments) ───────────────
PLANNER_COLORS = {
    "diffusion": COLORS["blue"],
    "bo":        COLORS["vermillion"],
    "lawnmower": "#404040",
    "random":    "#999999",
}

PLANNER_ORDER = ["diffusion", "bo", "random", "lawnmower"]

PLANNER_DISPLAY_NAMES = {
    "diffusion": "Diffusion",
    "bo":        "Bayesian Optimization",
    "lawnmower": "Lawnmower",
    "random":    "Random Walk",
}

# ── Standard figure sizes (inches) ────────────────────────────────────────────
# "single"     — small standalone figure (scalar metrics, distributions)
# "double_col" — IEEE double-column textwidth; 1×2 or 1×3 metric panels
# "wide"       — full-width multi-panel comparison (e.g. 1×5 trajectory panels)
FIGURE_SIZES = {
    "single":     (4.5, 3.5),
    "double_col": (7.16, 2.6),
    "wide":       (14.0, 3.2),
}


def apply_style(grid: bool = True) -> None:
    """
    Apply the project-wide matplotlib style.

    Parameters
    ----------
    grid : bool
        True (default) — enable dashed grid; suited for time-series / metric
        plots where horizontal reference lines aid reading.
        False — disable grid; suited for imshow / pcolormesh / diagram plots.
    """
    plt.rcParams.update({
        "text.usetex":       False,
        "mathtext.fontset":  "cm",
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "serif"],
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   11,
        "figure.titlesize":  13,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "lines.linewidth":   1.5,
        "figure.dpi":        150,
        "axes.grid":         grid,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.4,
        "grid.alpha":        0.5,
    })
