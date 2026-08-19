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


@hydra.main(config_path="../../config", config_name="evaluation", version_base="1.2")
def main(cfg):
    torch.manual_seed(cfg.planner_seed + cfg.scenario_idx)
    np.random.seed(cfg.seed + cfg.scenario_idx)
    output_dir = get_output_dir()
    planner = get_planner(cfg)

    gp_hyperparams = joblib.load(abs_path(cfg.paths.hyperparameters.gp))
    # print("GP Hyperparameters:", gp_hyperparams)

    import random
    random.seed(cfg.planner_seed + cfg.scenario_idx)
    scenario = generate_random_scenario(
        source_range=tuple(cfg.source_range),
        domain_size=tuple(cfg.domain_size),
        intensity_range=tuple(cfg.intensity_range)
    )
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

    logger = run_scenario(
        planner=planner,
        initial_position=tuple(cfg.planner.initial_position),
        initial_heading=cfg.planner.initial_heading,
        n_timesteps=cfg.n_timesteps,
        scenario=scenario,
        scenario_idx=cfg.scenario_idx,
        sigma=cfg.sigma,
        evaluation_x=evaluation_x,
        visualization_x=visualization_x,
        gp_hyperparams=gp_hyperparams,
        output_dir=output_dir,
        diffusion_coefficient=cfg.diffusion_coefficient,
    )
    logger.save_history("history.pkl")


if __name__ == "__main__":
    main()
