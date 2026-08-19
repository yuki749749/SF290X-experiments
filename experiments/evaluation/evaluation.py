import hydra
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
import joblib
import time

log = logging.getLogger(__name__)

from environment.environment import generate_random_scenario
from utils import abs_path, get_output_dir, make_grid, run_scenario, get_planner


from joblib import Parallel, delayed


def run_single_evaluation(scenario_idx, cfg, gp_hyperparams, output_dir, evaluation_x, visualization_x):
    # Set seeds dynamically for reproducibility
    torch.manual_seed(cfg.planner_seed + scenario_idx)
    np.random.seed(cfg.seed + scenario_idx)

    import random
    random.seed(cfg.planner_seed + scenario_idx)
    scenario = generate_random_scenario(
        source_range=tuple(cfg.source_range),
        domain_size=tuple(cfg.domain_size),
        intensity_range=tuple(cfg.intensity_range)
    )

    scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario_idx}", cfg.planner.type)

    logger = run_scenario(
        planner=get_planner(cfg),
        initial_position=tuple(cfg.planner.initial_position),
        initial_heading=cfg.planner.initial_heading,
        n_timesteps=cfg.n_timesteps,
        scenario=scenario,
        scenario_idx=scenario_idx,
        sigma=cfg.sigma,
        evaluation_x=evaluation_x,
        visualization_x=visualization_x,
        gp_hyperparams=gp_hyperparams,
        output_dir=scenario_output_dir,
        diffusion_coefficient=cfg.diffusion_coefficient,
    )
    logger.save_history("history.pkl")


@hydra.main(config_path="../../config", config_name="evaluation", version_base="1.2")
def main(cfg):
    output_dir = get_output_dir()
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

    # Run in parallel using joblib
    Parallel(n_jobs=cfg.n_jobs)(
        delayed(run_single_evaluation)(
            idx, cfg, gp_hyperparams, output_dir, evaluation_x, visualization_x
        )
        for idx in scenario_indices
    )


if __name__ == "__main__":
    main()
