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

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from utils import get_output_dir, find_latest_sweep

# ── Planner-class → subdirectory name mapping ─────────────────────────────────
# Keys are the directory names Hydra creates; values are the history-file prefix
# that logger.py writes (the planner's class name).
PLANNER_DIRS: dict[str, str] = {
    "bo": "BayesianOptimizationPlanner",
    "random": "RandomPlanner",
}
# ──────────────────────────────────────────────────────────────────────────────

# ── Discovery helpers ─────────────────────────────────────────────────────────


def get_scenario_dirs(sweep_dir: str) -> list[str]:
    """
    Find and return all scenario_* subdirectories in the sweep directory,
    sorted numerically.
    """
    scenario_dirs = sorted(
        glob.glob(os.path.join(sweep_dir, "scenario_*")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )
    if not scenario_dirs:
        raise FileNotFoundError(
            f"No 'scenario_*' subdirectories found in '{sweep_dir}'."
        )
    return scenario_dirs


def collect_history_paths_for_dirs(scenario_dirs: list[str]) -> list[tuple[str, str, str]]:
    """
    Walk the provided scenario directories and return all (scenario_dir, planner_tag, pkl_path) triples.
    """
    triples: list[tuple[str, str, str]] = []
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
    max_step: float = 10.0,
    initial_heading: float = np.pi/4,
) -> list[dict]:
    """
    Slide a window of length `horizon` over one trajectory.

    tau[0] is the ghost position p_{t-1} (one step before the window start),
    matching the two-point inpainting used by DiffusionPlanner at inference.

    Each window dict contains:
        tau    : (horizon, 2)  float32  — [p_{t-1}, p_t, …, p_{t+H-2}]
                                          tau[0]=ghost, tau[1]=current, tau[2:]=future
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
    triples: list[tuple[str, str, str]],
    reward_key: str,
) -> tuple[list[dict], dict[str, int]]:
    """
    Gather raw trajectories from specified (scenario, planner) pairs.

    Returns:
        dataset   : flat list of trajectory dicts
        counts    : per-planner trajectory counts, for reporting
    """
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

        trajectory = {
            "positions": np.asarray(history["position_history"], dtype=np.float32),
            "means": np.asarray(history["mean_history"], dtype=np.float32),
            "variances": np.asarray(history["variance_history"], dtype=np.float32),
            "rmse_history": np.asarray(history[reward_key], dtype=np.float32),
            "planner": planner_tag,
        }
        dataset.append(trajectory)
        counts[planner_tag] = counts.get(planner_tag, 0) + 1

    n_total = len(triples)
    n_loaded = n_total - skipped
    print(f"  Loaded {n_loaded}/{n_total} trajectories  ({skipped} skipped)")
    for tag, n in counts.items():
        if n:
            print(f"    {tag}: {n:,} trajectories")
    return dataset, counts


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(
    dataset: list[dict],
    horizon: int,
    stride: int,
    reward_key: str,
    reward_type: str,
    max_step: float,
    initial_heading: float,
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
        "reward_type":           reward_type,
        "max_step":              max_step,
        "initial_heading":       initial_heading,
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

def save_dataset(
    dataset: list[dict],
    stats: dict,
    out_dir: str,
    prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data_path  = os.path.join(out_dir, f"{prefix}_data.pkl")
    stats_path = os.path.join(out_dir, f"{prefix}_data_stats.json")

    joblib.dump(dataset, data_path)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved {prefix} dataset -> {data_path}")
    print(f"Saved {prefix} stats   -> {stats_path}")


def split_scenario_dirs(
    scenario_dirs: list[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    import random
    dirs = list(scenario_dirs)
    rng = random.Random(seed)
    rng.shuffle(dirs)

    n = len(dirs)
    if n == 0:
        return [], [], []
    if n == 1:
        return dirs, [], []
    if n == 2:
        return dirs[:1], dirs[1:], []

    n_train = max(1, int(n * train_frac))
    n_val = max(0, int(n * val_frac))
    if n_train + n_val > n:
        n_train = n - n_val
        if n_train < 1:
            n_train = 1
            n_val = n - 1

    train_dirs = dirs[:n_train]
    val_dirs = dirs[n_train:n_train + n_val]
    test_dirs = dirs[n_train + n_val:]
    return train_dirs, val_dirs, test_dirs


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
        sweep_root = str(OmegaConf.select(cfg, "sweep_root", default="results/generate_raw_trajectories/sweep"))
        sweep_dir  = find_latest_sweep(sweep_root)
        print(f"Auto-discovered latest sweep:\n  {sweep_dir}")

    horizon     = int(cfg.horizon)
    stride      = int(cfg.stride)
    reward_key  = str(cfg.reward_key)
    reward_type = str(OmegaConf.select(cfg, "reward_type", default="rmse"))
    out_dir     = get_output_dir()

    # Retrieve max_step and initial_heading dynamically
    max_step = float(OmegaConf.select(cfg, "planner.max_step", default=10.0))
    initial_heading = float(OmegaConf.select(cfg, "planner.initial_heading", default=np.pi/4))

    # Retrieve split fractions
    train_fraction = float(OmegaConf.select(cfg, "train_fraction", default=0.8))
    val_fraction   = float(OmegaConf.select(cfg, "val_fraction", default=0.1))
    test_fraction  = float(OmegaConf.select(cfg, "test_fraction", default=0.1))

    print(
        f"\n=== Trajectory Post-Processing ===\n"
        f"  sweep_dir   : {sweep_dir}\n"
        f"  horizon     : {horizon}\n"
        f"  stride      : {stride}\n"
        f"  reward_key  : {reward_key}\n"
        f"  reward_type : {reward_type}\n"
        f"  max_step    : {max_step}\n"
        f"  heading     : {initial_heading:.4f}\n"
        f"  out_dir     : {out_dir}\n"
        f"  splits      : train={train_fraction:.2f}, val={val_fraction:.2f}, test={test_fraction:.2f}\n"
    )

    # 1. Discover and sort scenario directories numerically
    scenario_dirs = get_scenario_dirs(sweep_dir)
    print(f"Discovered {len(scenario_dirs)} scenario directories.")

    # 2. Split scenarios
    seed = int(OmegaConf.select(cfg, "seed", default=42))
    train_dirs, val_dirs, test_dirs = split_scenario_dirs(
        scenario_dirs, train_fraction, val_fraction, test_fraction, seed=seed
    )
    print(
        f"Splits:\n"
        f"  Train      : {len(train_dirs)} scenarios\n"
        f"  Validation : {len(val_dirs)} scenarios\n"
        f"  Test       : {len(test_dirs)} scenarios"
    )

    # 3. Process splits
    splits = [
        ("training", train_dirs),
        ("validation", val_dirs),
        ("testing", test_dirs),
    ]

    for prefix, sdirs in splits:
        if not sdirs:
            print(f"\nSkipping empty split '{prefix}'")
            continue
        print(f"\n=== Processing split '{prefix}' ({len(sdirs)} scenarios) ===")
        triples = collect_history_paths_for_dirs(sdirs)
        dataset, counts = collect_dataset(triples, reward_key)
        if not dataset:
            print(f"No trajectories loaded for split '{prefix}'")
            continue

        # Temporarily perform window extraction to compute statistics and plot reward distribution
        windowed_dataset = []
        window_counts = {tag: 0 for tag in PLANNER_DIRS.values()}
        for traj in dataset:
            history_mock = {
                "position_history": traj["positions"],
                "mean_history": traj["means"],
                "variance_history": traj["variances"],
                reward_key: traj["rmse_history"],
            }
            windows = extract_windows(
                history_mock, horizon, stride, reward_key, reward_type, max_step, initial_heading
            )
            for w in windows:
                w["planner"] = traj["planner"]
            windowed_dataset.extend(windows)
            window_counts[traj["planner"]] += len(windows)

        if not windowed_dataset:
            print(f"No windows extracted for split '{prefix}' with current horizon/stride")
            stats = {}
        else:
            stats = compute_stats(
                windowed_dataset,
                horizon,
                stride,
                reward_key,
                reward_type,
                max_step,
                initial_heading,
                window_counts,
            )
            print(f"\n--- {prefix.capitalize()} Dataset Statistics ---")
            for k, v in stats.items():
                print(f"  {k:<35}: {v}")

        save_dataset(dataset, stats, out_dir, prefix)


if __name__ == "__main__":
    main()