import hydra
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
import joblib
import os

from environment.environment import plume, generate_random_scenario
from belief.belief import Belief, ExactGPModel
from planners.planners import (
    RandomPlanner,
    BayesianOptimizationPlanner,
)
from logger import Logger
from utils import abs_path, get_output_dir


def get_planner(cfg):
    if cfg.planner.type == "random":
        return RandomPlanner(
            domain_size=cfg.domain_size,
            max_step=cfg.planner.max_step,
            min_step=cfg.planner.min_step,
            max_turn=cfg.planner.max_turn,
            boundary_behavior=cfg.planner.boundary_behavior,
        )
    elif cfg.planner.type == "bo":
        return BayesianOptimizationPlanner(
            domain_size=cfg.domain_size,
            max_step=cfg.planner.max_step,
            min_step=cfg.planner.min_step,
            max_turn=cfg.planner.max_turn,
            boundary_behavior=cfg.planner.boundary_behavior,
            acquisition_function=cfg.planner.acquisition_function,
            beta=cfg.planner.beta,
            num_steps=cfg.planner.num_steps,
            num_turns=cfg.planner.num_turns,
            randomize_first_step=cfg.planner.randomize_first_step,
        )
    else:
        raise ValueError(f"Unknown planner type: {cfg.planner}")


def initialize_belief(position, measurement, gp_hyperparams, sigma, eval_x, diffusion_coefficient=2.0):
    train_x = torch.tensor(position, dtype=torch.float32).unsqueeze(0)  # (1, 2)
    train_y = torch.tensor([measurement], dtype=torch.float32)  # (1,)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    prior = gpytorch.priors.GammaPrior(6.0, 6.0 / diffusion_coefficient)
    model = ExactGPModel(train_x, train_y, likelihood, lengthscale_prior=prior)
    with torch.no_grad():
        model.mean_module.constant = torch.tensor(gp_hyperparams["mean_constant"])
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [gp_hyperparams["lengthscale_0"], gp_hyperparams["lengthscale_1"]]
        )
        model.covar_module.outputscale = torch.tensor(gp_hyperparams["outputscale"])
        model.likelihood.noise = torch.tensor(sigma**2)
    for param in model.parameters():
        param.requires_grad_(False)

    return Belief(model, eval_x=eval_x)


def run_scenario(
    planner,
    initial_position,
    initial_heading,
    n_timesteps,
    scenario,
    scenario_idx,
    sigma,
    evaluation_x,
    visualization_x,
    gp_hyperparams,
    output_dir,
    diffusion_coefficient,
):
    planner.reset()
    ground_truth_eval_np = np.array([plume(scenario, p, diffusion_coefficient) for p in evaluation_x.numpy()])
    ground_truth_eval = torch.tensor(ground_truth_eval_np, dtype=torch.float32)

    ground_truth_vis_np = np.array(
        [plume(scenario, p, diffusion_coefficient) for p in visualization_x.numpy()]
    )
    ground_truth_vis = torch.tensor(ground_truth_vis_np, dtype=torch.float32)

    initial_measurement = plume(scenario, initial_position, diffusion_coefficient) + np.random.normal(0, sigma)

    belief = initialize_belief(
        position=initial_position,
        measurement=initial_measurement,
        gp_hyperparams=gp_hyperparams,
        sigma=sigma,
        eval_x=evaluation_x,
        diffusion_coefficient=diffusion_coefficient,
    )

    logger = Logger(
        output_dir,
        eval_x=evaluation_x,
        vis_x=visualization_x,
        ground_truth_eval=ground_truth_eval,
        ground_truth_vis=ground_truth_vis,
    )
    logger.log_step(initial_position, belief)

    current_position = initial_position
    current_heading = initial_heading

    W, H = planner.domain_size

    for t in range(1, n_timesteps + 1):
        current_position, current_heading = planner.compute_next_pose(
            current_position, current_heading, belief
        )

        x, y = current_position
        in_domain = (0 <= x <= W) and (0 <= y <= H)

        if in_domain:
            current_position_tensor = torch.tensor(
                current_position, dtype=torch.float32
            ).unsqueeze(0)
            measurement = plume(scenario, current_position, diffusion_coefficient) + np.random.normal(0, sigma)
            measurement_tensor = torch.tensor([measurement], dtype=torch.float32)

            belief.update(current_position_tensor, measurement_tensor)
        logger.log_step(current_position, belief)
    return logger


def make_grid(domain_min, domain_max, n_evaluations):
    """Uniform grid of evaluation points."""
    side = int(np.sqrt(n_evaluations))
    t1 = torch.linspace(domain_min[0], domain_max[0], side)
    t2 = torch.linspace(domain_min[1], domain_max[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)


@hydra.main(config_path="../../config", config_name="generate_raw_trajectories", version_base="1.2")
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    output_dir = get_output_dir()
    planner = get_planner(cfg)

    gp_hyperparams = joblib.load(abs_path(cfg.paths.hyperparameters.gp))

    import random
    random.seed(cfg.seed + cfg.scenario_idx)
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
        domain_min=[cfg.domain_min[0] - cfg.domain_pad, cfg.domain_min[1] - cfg.domain_pad],
        domain_max=[cfg.domain_max[0] + cfg.domain_pad, cfg.domain_max[1] + cfg.domain_pad],
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
