"""
process_trajectories.py
-----------------------
Post-processes raw trajectory logs from a Hydra multirun sweep into
(τ, r, b_mean, b_var) tuples for diffusion model training.

Expected Hydra multirun layout (produced by generate_raw_trajectories with -m):
    results/generate_raw_trajectories/sweep/<timestamp>/
        scenario_<idx>/
            bo/
                history.pkl
            random/
                history.pkl
            ...

The script auto-discovers the most recent sweep timestamp unless
`sweep_dir` is set explicitly in the config (or via CLI override).

Output (written relative to the project root, not the Hydra output dir):
    <out_dir>/training_data.pkl          ← list[dict]
    <out_dir>/training_data_stats.json   ← dataset statistics
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import hydra
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":      False,
    "mathtext.fontset": "cm",
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Times New Roman", "serif"],
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  9,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "axes.grid":        False,
    "lines.linewidth":  1.5,
})
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from utils import get_output_dir, abs_path

# ── Planner-class → subdirectory name mapping ─────────────────────────────────
# Keys are the directory names Hydra creates; values are the history-file prefix
# that logger.py writes (the planner's class name).
PLANNER_DIRS: dict[str, str] = {
    "bo": "BayesianOptimizationPlanner",
    "random": "RandomPlanner",
}
# ──────────────────────────────────────────────────────────────────────────────

SWEEP_ROOT = "results/generate_raw_trajectories/sweep"

max_step = 10.0
initial_heading = np.pi/4


# ── Discovery helpers ─────────────────────────────────────────────────────────

def find_latest_sweep(sweep_root: str) -> str:
    """Return the path to the most recently modified sweep timestamp directory."""
    candidates = sorted(
        glob.glob(os.path.join(sweep_root, "*")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No sweep directories found under '{sweep_root}'. "
            "Run generate_raw_trajectories with -m first."
        )
    return candidates[-1]


def collect_history_paths(sweep_dir: str) -> list[tuple[str, str, str]]:
    """
    Walk a sweep directory and return all (scenario_dir, planner_tag, pkl_path) triples.

    Directory layout expected:
        sweep_dir/
            scenario_<idx>/
                <planner_dir>/
                    history.pkl
    """
    triples: list[tuple[str, str, str]] = []

    scenario_dirs = sorted(glob.glob(os.path.join(sweep_dir, "scenario_*")))
    if not scenario_dirs:
        raise FileNotFoundError(
            f"No 'scenario_*' subdirectories found in '{sweep_dir}'."
        )

    for sdir in scenario_dirs:
        for planner_dir, planner_tag in PLANNER_DIRS.items():
            pkl = os.path.join(sdir, planner_dir, "history.pkl")
            if os.path.exists(pkl):
                triples.append((sdir, planner_tag, pkl))

    return triples


# ── Core processing ───────────────────────────────────────────────────────────

def extract_windows(
    history: dict,
    horizon: int,
    stride: int,
    reward_key: str,
    reward_type: str = "rmse",
) -> list[dict]:
    """
    Slide a window of length `horizon` over one trajectory.

    tau[0] is the ghost position p_{t-1} (one step before the window start),
    matching the two-point inpainting used by DiffusionPlanner at inference.

    Each window dict contains:
        tau    : (horizon, 2)  float32  — [p_{t-1}, p_t, …, p_{t+H-2}]
                                          tau[0]=ghost, tau[1]=current, tau[2:]=future
        r      : scalar        float32  — reward for the window (see reward_type)
        b_mean : (n_eval,)     float32  — GP posterior mean at step t
        b_var  : (n_eval,)     float32  — GP posterior variance at step t

    reward_type options:
        "rmse"            — (rmse[t] - rmse[t+H-2]) / rmse[t]
                            Positive = RMSE decreased over the window.
        "trace_reduction" — (ntr[t+H-2] - ntr[t]) / (1 - ntr[t])
                            Equivalent to (Tr_t - Tr_{t+H-2}) / Tr_t.
                            Fraction of remaining trace reduction achieved in window.

    A fixed synthetic ghost (matching the inference value) is prepended at index 0
    so that the loop can start at t=1 with a consistent positions[t-1] ghost.
    """
    if reward_type not in ("rmse", "trace_reduction"):
        raise ValueError(f"Unknown reward_type '{reward_type}'. Choose 'rmse' or 'trace_reduction'.")

    positions: list = list(history["position_history"])
    means: list     = list(history["mean_history"])
    variances: list = list(history["variance_history"])
    rewards: list   = list(history[reward_key])

    T = len(positions) - 1  # index of last valid step (before prepend)
    if T < horizon:
        return []  # trajectory too short for even one window

    # Fixed synthetic ghost matching the value used at inference time.
    ghost = [-max_step * np.cos(initial_heading), -max_step * np.sin(initial_heading)]

    # Prepend ghost / null so that positions[0] = p_{-1} and all lists have length T+2.
    # means/variances/rewards get a null placeholder at index 0 (never accessed;
    # the loop starts at t=1).
    positions = [ghost] + positions
    means     = [None]  + means
    variances = [None]  + variances
    rewards   = [None]  + rewards

    windows: list[dict] = []

    if reward_type == "rmse":
        def _reward(t: int) -> np.float32:
            return (np.float32(rewards[t]) - np.float32(rewards[t + horizon - 2])) / (np.float32(rewards[t]) + 1e-8)
    else:  # trace_reduction
        def _reward(t: int) -> np.float32:
            return (np.float32(rewards[t + horizon - 2]) - np.float32(rewards[t])) / (1.0 - np.float32(rewards[t]) + 1e-8)

    for t in range(1, T - horizon + 2, stride):
        tau    = np.asarray(positions[t - 1 : t + horizon - 1], dtype=np.float32)  # (H, 2)
        r      = _reward(t)
        b_mean = np.asarray(means[t],     dtype=np.float32)
        b_var  = np.asarray(variances[t], dtype=np.float32)
        windows.append({"tau": tau, "r": r, "b_mean": b_mean, "b_var": b_var})

    return windows


def collect_dataset(
    sweep_dir: str,
    horizon: int,
    stride: int,
    reward_key: str,
    reward_type: str = "rmse",
) -> tuple[list[dict], dict[str, int]]:
    """
    Gather windows from all (scenario, planner) pairs in the sweep directory.

    Returns:
        dataset   : flat list of window dicts
        counts    : per-planner window counts, for reporting
    """
    triples = collect_history_paths(sweep_dir)
    if not triples:
        raise FileNotFoundError(
            f"No history .pkl files found in '{sweep_dir}'. "
            "Check that generate_raw_trajectories completed successfully."
        )

    dataset: list[dict] = []
    counts: dict[str, int] = {tag: 0 for tag in PLANNER_DIRS.values()}
    skipped = 0

    for sdir, planner_tag, pkl_path in tqdm(triples, desc="Processing trajectories"):
        try:
            history = joblib.load(pkl_path)
        except Exception as exc:
            print(f"  [WARN] Could not load {pkl_path}: {exc}")
            skipped += 1
            continue

        windows = extract_windows(history, horizon, stride, reward_key, reward_type)
        dataset.extend(windows)
        counts[planner_tag] = counts.get(planner_tag, 0) + len(windows)

    n_total = len(triples)
    n_loaded = n_total - skipped
    print(f"  Loaded {n_loaded}/{n_total} trajectories  ({skipped} skipped)")
    for tag, n in counts.items():
        if n:
            print(f"    {tag}: {n:,} windows")
    print(f"  Total windows: {len(dataset):,}")
    return dataset, counts


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(
    dataset: list[dict],
    horizon: int,
    stride: int,
    reward_key: str,
    counts: dict[str, int],
) -> dict:
    all_tau    = np.stack([d["tau"]    for d in dataset])  # (N, H, 2)
    all_r      = np.array([d["r"]      for d in dataset])  # (N,)
    all_b_mean = np.stack([d["b_mean"] for d in dataset])  # (N, n_eval)
    all_b_var  = np.stack([d["b_var"]  for d in dataset])  # (N, n_eval)

    return {
        "n_samples":             len(dataset),
        "horizon":               horizon,
        "stride":                stride,
        "reward_key":            reward_key,
        "windows_per_planner":   counts,
        # trajectory positions
        "tau_mean":              all_tau.mean(axis=(0, 1)).tolist(),
        "tau_std":               all_tau.std(axis=(0, 1)).tolist(),
        "tau_min":               float(all_tau.min()),
        "tau_max":               float(all_tau.max()),
        # reward
        "r_mean":                float(all_r.mean()),
        "r_std":                 float(all_r.std()),
        "r_min":                 float(all_r.min()),
        "r_max":                 float(all_r.max()),
        # belief mean
        "b_mean_global_mean":    float(all_b_mean.mean()),
        "b_mean_global_std":     float(all_b_mean.std()),
        # belief variance
        "b_var_global_mean":     float(all_b_var.mean()),
        "b_var_global_std":      float(all_b_var.std()),
    }


# ── Save ──────────────────────────────────────────────────────────────────────

def plot_reward_distribution(dataset: list[dict], out_dir: str) -> None:
    rewards = np.array([d["r"] for d in dataset], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.hist(rewards, bins=60, color="#4393C3", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Reward $r$")
    ax.set_ylabel("Count")
    ax.set_title(f"Reward distribution  ($n={len(rewards):,}$)", fontsize=10)

    percentiles = [50, 75, 90, 95, 99]
    colors      = ["#D55E00", "#E69F00", "#009E73", "#0072B2", "#CC79A7"]
    pct_values  = np.percentile(rewards, percentiles)

    for p, c, v in zip(percentiles, colors, pct_values):
        ax.axvline(v, color=c, linestyle="--", linewidth=1.0, label=f"p{p} = {v:.3f}")

    ax.axvline(rewards.mean(), color="black", linestyle=":", linewidth=1.0,
               label=f"mean = {rewards.mean():.3f}")
    ax.legend(frameon=False)

    print("\n--- Reward Percentiles ---")
    print(f"  {'mean':<6}: {rewards.mean():.4f}")
    for p, v in zip(percentiles, pct_values):
        print(f"  p{p:<5}: {v:.4f}")

    plot_path = os.path.join(out_dir, "reward_distribution.pdf")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward histogram -> {plot_path}")


def save_dataset(
    dataset: list[dict],
    stats: dict,
    out_dir: str,
    central_data_path: str = None,
    central_stats_path: str = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data_path  = os.path.join(out_dir, "training_data.pkl")
    stats_path = os.path.join(out_dir, "training_data_stats.json")

    joblib.dump(dataset, data_path)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    plot_reward_distribution(dataset, out_dir)

    print(f"\nSaved dataset -> {data_path}")
    print(f"Saved stats   -> {stats_path}")

    if central_data_path is not None:
        os.makedirs(os.path.dirname(central_data_path), exist_ok=True)
        joblib.dump(dataset, central_data_path)
        print(f"Saved copy of dataset to central path -> {central_data_path}")
    if central_stats_path is not None:
        with open(central_stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved copy of stats to central path -> {central_stats_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

@hydra.main(
    config_path="../../config",
    config_name="process_trajectories",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    # Resolve the sweep directory: explicit override > auto-discover latest
    if OmegaConf.select(cfg, "sweep_dir") is not None:
        sweep_dir = str(cfg.sweep_dir)
        print(f"Using explicitly configured sweep_dir:\n  {sweep_dir}")
    else:
        sweep_root = str(OmegaConf.select(cfg, "sweep_root", default=SWEEP_ROOT))
        sweep_dir  = find_latest_sweep(sweep_root)
        print(f"Auto-discovered latest sweep:\n  {sweep_dir}")

    horizon     = int(cfg.horizon)
    stride      = int(cfg.stride)
    reward_key  = str(cfg.reward_key)
    reward_type = str(OmegaConf.select(cfg, "reward_type", default="rmse"))
    out_dir     = get_output_dir()  # save processed data in the Hydra output directory for easy reference
    # out_dir    = str(sweep_dir)  # save processed data in the same sweep directory for easy reference

    print(
        f"\n=== Trajectory Post-Processing ===\n"
        f"  sweep_dir   : {sweep_dir}\n"
        f"  horizon     : {horizon}\n"
        f"  stride      : {stride}\n"
        f"  reward_key  : {reward_key}\n"
        f"  reward_type : {reward_type}\n"
        f"  out_dir     : {out_dir}\n"
    )

    dataset, counts = collect_dataset(sweep_dir, horizon, stride, reward_key, reward_type)
    stats           = compute_stats(dataset, horizon, stride, reward_key, counts)

    print("\n--- Dataset Statistics ---")
    for k, v in stats.items():
        print(f"  {k:<35}: {v}")

    central_processed_path = abs_path(cfg.paths.trajectories.processed)
    if os.path.isdir(central_processed_path) or not central_processed_path.endswith(".pkl"):
        central_data_path = os.path.join(central_processed_path, "training_data.pkl")
        central_stats_path = os.path.join(central_processed_path, "training_data_stats.json")
    else:
        central_data_path = central_processed_path
        central_stats_path = os.path.join(os.path.dirname(central_processed_path), "training_data_stats.json")

    save_dataset(dataset, stats, out_dir, central_data_path, central_stats_path)


if __name__ == "__main__":
    main()