"""
evaluate_checkpoints.py
======================
Evaluate all checkpoints (.pt files) in the latest training folder and compare them.

Generates:
1. Per-run and aggregated metrics across checkpoints in CSV and JSON.
2. An Epoch-wise performance plot showing Final NRMSE and Fraction in Domain.
3. An RMSE-over-time plot showing trajectory performance for different checkpoints.

Usage
-----
    # Automatically find latest training folder and run evaluations:
    python experiments/evaluation/evaluate_checkpoints.py --n_scenarios 5

    # Run on a specific training directory:
    python experiments/evaluation/evaluate_checkpoints.py training_dir=results/train_diffusion/run/2026-08-25_01-13-09 --n_scenarios 5
"""

import os
import sys
import re
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from threadpoolctl import threadpool_limits

# Add project src and experiments directories to system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import OmegaConf

from plot_style import apply_style, FIGURE_SIZES, COLORS
from utils import abs_path, get_output_dir, make_grid, find_latest_run_dir, run_scenario, get_planner
from evaluation_utils import discover_results, extract_metric, mean_ci, compute, aggregate
from environment.environment import generate_random_scenario


def run_checkpoint_scenario(
    scenario_idx,
    cfg,
    gp_hyperparams,
    output_dir,
    evaluation_x,
    visualization_x,
    checkpoint_path,
    checkpoint_name,
):
    """
    Run evaluation scenario idx for a specific checkpoint.
    Imports are inside the function for safe joblib Loky worker execution on Windows.
    """
    import torch
    import numpy as np
    import random
    import os
    from environment.environment import generate_random_scenario
    from utils import run_scenario, get_planner
    from omegaconf import OmegaConf

    # Set seeds dynamically for reproducibility of environment noise
    torch.manual_seed(cfg.planner_seed + scenario_idx)
    np.random.seed(cfg.seed + scenario_idx)
    random.seed(cfg.planner_seed + scenario_idx)

    # Generate scenario
    scenario = generate_random_scenario(
        source_range=tuple(cfg.source_range),
        domain_size=tuple(cfg.domain_size),
        intensity_range=tuple(cfg.intensity_range)
    )

    # Output dir for this specific checkpoint and scenario
    scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario_idx}", checkpoint_name)

    # Override checkpoint path in config copy
    import copy
    cfg_copy = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg_copy, False)
    cfg_copy.planner.checkpoint_path = str(checkpoint_path)

    # Run the planner scenario
    logger = run_scenario(
        planner=get_planner(cfg_copy),
        initial_position=tuple(cfg_copy.planner.initial_position),
        initial_heading=cfg_copy.planner.initial_heading,
        n_timesteps=cfg_copy.n_timesteps,
        scenario=scenario,
        scenario_idx=scenario_idx,
        sigma=cfg_copy.sigma,
        evaluation_x=evaluation_x,
        visualization_x=visualization_x,
        gp_hyperparams=gp_hyperparams,
        output_dir=scenario_output_dir,
        diffusion_coefficient=cfg_copy.diffusion_coefficient,
    )
    logger.save_history("history.pkl")





@hydra.main(config_path="../../config", config_name="evaluation", version_base="1.2")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    # 1. Resolve pathways and settings
    if hasattr(cfg, "planner") and "checkpoint_path" in cfg.planner and cfg.planner.checkpoint_path:
        cfg.planner.checkpoint_path = abs_path(cfg.planner.checkpoint_path)

    gp_hyperparams = joblib.load(abs_path(cfg.paths.hyperparameters.gp))

    evaluation_x = make_grid(
        domain_min=cfg.domain_min,
        domain_max=cfg.domain_max,
        n_evaluations=cfg.n_evaluations,
    )

    visualization_x = make_grid(
        domain_min=[
            cfg.domain_min[0] - cfg.domain_pad,
            cfg.domain_min[1] - cfg.domain_pad,
        ],
        domain_max=[
            cfg.domain_max[0] + cfg.domain_pad,
            cfg.domain_max[1] + cfg.domain_pad,
        ],
        n_evaluations=cfg.n_visualizations,
    )

    # Determine scenario indices to run
    if "scenario_idx" in cfg and cfg.scenario_idx != 0:
        scenario_indices = [cfg.scenario_idx]
    else:
        start = cfg.get("start_scenario_idx", 0)
        count = cfg.get("n_scenarios", 1)
        scenario_indices = list(range(start, start + count))

    # 2. Discover Checkpoints
    training_dir = cfg.get("training_dir", None)
    if training_dir is None:
        training_dir = find_latest_run_dir("results", "train_diffusion", "best_checkpoint.pt")
    else:
        training_dir = Path(training_dir)

    print(f"Discovered training directory: {training_dir}")
    checkpoint_files = list(training_dir.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in {training_dir}")

    checkpoints_info = []
    name_to_epoch = {}
    
    print("\nLoading checkpoints and finding epochs...")
    for path in checkpoint_files:
        name = path.stem
        try:
            state = torch.load(path, map_location="cpu")
            epoch = state.get("epoch", None)
        except Exception as exc:
            print(f"  Warning: could not load checkpoint {path.name}: {exc}")
            continue

        if epoch is None:
            # Fallback parsing epoch from filename
            match = re.search(r'epoch_(\d+)', name)
            if match:
                epoch = int(match.group(1))
            else:
                epoch = 999  # Fallback code

        checkpoints_info.append({
            "path": path,
            "name": name,
            "epoch": epoch,
        })
        name_to_epoch[name] = epoch
        print(f"  - {name}: epoch {epoch}")

    # Sort checkpoints to run them in order
    checkpoints_info.sort(key=lambda x: x["epoch"])

    # 3. Setup parallel evaluation sweep
    output_dir = Path(get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEvaluating checkpoints on {len(scenario_indices)} scenario(s)...")
    print(f"Outputs will be saved under: {output_dir}")

    # Construct all tasks (checkpoint, scenario)
    tasks = []
    for info in checkpoints_info:
        for idx in scenario_indices:
            tasks.append((idx, info["path"], info["name"]))

    # Execute all tasks in parallel using a single Parallel session (to reuse Loky workers)
    try:
        with threadpool_limits(limits=1):
            joblib.Parallel(n_jobs=cfg.n_jobs, backend=cfg.get("joblib_backend", "loky"), verbose=10)(
                joblib.delayed(run_checkpoint_scenario)(
                    idx, cfg, gp_hyperparams, output_dir, evaluation_x, visualization_x, path, name
                )
                for idx, path, name in tasks
            )
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except Exception:
            pass

    # 4. Compute Metrics
    print("\nComputing metrics...")
    domain_size = tuple(cfg.domain_size)
    df_runs = compute(output_dir, domain_size)
    df_agg = aggregate(df_runs)

    # Save to CSV files
    per_run_csv = output_dir / "metrics_per_run.csv"
    df_runs.to_csv(per_run_csv, index=False, float_format="%.6f")
    print(f"Saved per-run CSV -> {per_run_csv}")

    agg_csv = output_dir / "metrics_aggregated.csv"
    df_agg.to_csv(agg_csv, index=False, float_format="%.6f")
    print(f"Saved aggregated CSV -> {agg_csv}")

    # 5. Save name_to_epoch mapping for standalone plotting
    import json
    name_to_epoch_json = output_dir / "name_to_epoch.json"
    with open(name_to_epoch_json, "w") as f:
        json.dump(name_to_epoch, f, indent=2)
    print(f"Saved name-to-epoch JSON -> {name_to_epoch_json}")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
