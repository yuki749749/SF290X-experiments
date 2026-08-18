"""
plot_beta_tuning.py
-------------------
Load saved beta-tuning histories and plot:
  - Mean RMSE history per beta (with 95% bootstrap CI)
  - Mean normalised trace reduction history per beta

Assumes directory structure written by tune_beta.py:
  results/beta_tuning/scenario_XXXX/beta_<value>_history.pkl
"""

import os
import re
import glob
import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Matplotlib style ──────────────────────────────────────────────────────────
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

# Wong colour-blind palette (skip black; index = 0-based beta order)
WONG = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
]

RESULTS_DIR = "results/beta_tuning"
OUT_DIR = "results/beta_tuning"
N_BOOTSTRAP = 2000
ALPHA = 0.05
FIGWIDTH_IN = 3.5  # IEEE single column


# ── Helpers ───────────────────────────────────────────────────────────────────

def bootstrap_ci(data: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = ALPHA):
    """
    data : (n_scenarios, n_steps)
    Returns mean (n_steps,), lo (n_steps,), hi (n_steps,)
    """
    rng = np.random.default_rng(42)
    n_sc = data.shape[0]
    boot_means = np.stack(
        [data[rng.integers(0, n_sc, n_sc)].mean(axis=0) for _ in range(n)]
    )
    lo = np.percentile(boot_means, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return data.mean(axis=0), lo, hi


def load_beta_histories(results_dir: str):
    """
    Returns dict  beta_value (float) -> {
        'rmse':  np.ndarray (n_scenarios, n_steps),
        'ntr':   np.ndarray (n_scenarios, n_steps),   # normalised trace reduction
    }
    """
    pattern = os.path.join(results_dir, "scenario_*", "beta_*_history.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No history files found under {results_dir!r}")

    # group by beta value
    beta_data: dict[float, dict[str, list]] = {}
    for fp in files:
        m = re.search(r"beta_([\d.eE+\-]+)_history\.pkl", os.path.basename(fp))
        if not m:
            continue
        beta = float(m.group(1))
        hist = joblib.load(fp)
        if beta not in beta_data:
            beta_data[beta] = {"rmse": [], "ntr": []}
        beta_data[beta]["rmse"].append(hist["rmse_history"])
        beta_data[beta]["ntr"].append(hist["normalized_trace_reduction_history"])

    # convert lists → arrays, sort betas
    return {
        b: {
            "rmse": np.array(beta_data[b]["rmse"]),
            "ntr": np.array(beta_data[b]["ntr"]),
        }
        for b in sorted(beta_data)
    }


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_beta_tuning(results_dir: str = RESULTS_DIR, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    data = load_beta_histories(results_dir)
    betas = list(data.keys())
    n_steps = next(iter(data.values()))["rmse"].shape[1]
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIGWIDTH_IN*2, FIGWIDTH_IN),
        sharex=True,
    )
    ax_rmse, ax_ntr = axes

    for idx, beta in enumerate(betas):
        color = WONG[idx % len(WONG)]
        label = rf"$\beta = {beta:g}$"

        # RMSE
        mean, lo, hi = bootstrap_ci(data[beta]["rmse"])
        ax_rmse.plot(steps, mean, color=color, label=label)
        ax_rmse.fill_between(steps, lo, hi, color=color, alpha=0.20, linewidth=0)

        # Normalised trace reduction
        mean, lo, hi = bootstrap_ci(data[beta]["ntr"])
        ax_ntr.plot(steps, mean, color=color, label=label)
        ax_ntr.fill_between(steps, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax_rmse.set_ylabel("Normalized RMSE")
    ax_rmse.set_title(r"RMSE history by $\beta$")
    ax_rmse.grid(True, linewidth=0.4, alpha=0.5)

    ax_ntr.set_ylabel("Normalized trace reduction")
    ax_ntr.set_xlabel("Step $t$")
    ax_ntr.set_title(r"Trace reduction history by $\beta$")
    ax_ntr.set_ylim(bottom=0)
    ax_ntr.grid(True, linewidth=0.4, alpha=0.5)

    # Single shared legend below both panels
    handles = [
        Line2D([0], [0], color=WONG[i % len(WONG)], label=rf"$\beta = {b:g}$")
        for i, b in enumerate(betas)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(betas), 4),
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = os.path.join(out_dir, "beta_tuning.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_beta_tuning()