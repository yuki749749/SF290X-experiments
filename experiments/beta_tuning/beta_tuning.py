import os
import joblib
import numpy as np
import torch
import gpytorch

from environment.environment import plume
from belief.belief import ExactGPModel, Belief
from planners.planners import BayesianOptimizationPlanner
from logger import Logger


domain_size = (10.0, 10.0)
source_range = (1, 3)
intensity_range = (50, 200)
sigma = 0.5

n_evaluations = 40**2
n_steps = 100

max_step = 1.0
max_turn = np.pi / 8
num_steps = 20
num_turns = 11
beta_sweep = [0.1, 1.0, 10.0, 100.0, 300.0, 500.0, 1000.0]
# beta_sweep = [100.0, 300.0]
beta_sweep = [20.0, 30.0, 40.0]

gp_hyperparams = joblib.load("data/hyperparameters/gp_hyperparameters.pkl")
seed = 42

from environment.environment import generate_random_scenario
import random

beta_tuning_scenarios = 100
beta_tuning_seed = 20000
scenarios = []
for idx in range(beta_tuning_scenarios):
    random.seed(beta_tuning_seed + idx)
    scenarios.append(
        generate_random_scenario(
            source_range=source_range,
            domain_size=domain_size,
            intensity_range=intensity_range
        )
    )


def make_evaluation_x(domain_size, n_evaluations):
    """Uniform grid of evaluation points."""
    side = int(np.sqrt(n_evaluations))
    t1 = torch.linspace(0, domain_size[0], side)
    t2 = torch.linspace(0, domain_size[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)  # (side², 2)


def initialize_belief(position, measurement, eval_x, gp_hyperparams, sigma):
    train_x = torch.tensor(position, dtype=torch.float32).unsqueeze(0)  # (1, 2)
    train_y = torch.tensor([measurement], dtype=torch.float32)  # (1,)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    with torch.no_grad():
        model.mean_module.constant = torch.tensor(gp_hyperparams["mean_constant"])
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [gp_hyperparams["lengthscale_0"], gp_hyperparams["lengthscale_1"]]
        )
        model.covar_module.outputscale = torch.tensor(gp_hyperparams["outputscale"])
        model.likelihood.noise = torch.tensor(sigma**2)
    for param in model.parameters():
        param.requires_grad_(False)

    return Belief(model, eval_x)


def run_scenario(scenario, eval_x, gp_hyperparams, sigma, planner):

    ground_truth = np.array(
        [plume(scenario, p) for p in eval_x.numpy()], dtype=np.float32
    )

    current_position = (0.0, 0.0)
    current_heading = np.pi / 4
    measurement = plume(scenario, current_position) + np.random.normal(0, sigma)
    belief = initialize_belief(
        current_position, measurement, eval_x, gp_hyperparams, sigma
    )

    logger = Logger(
        f"results/beta_tuning/scenario_{scenarios.index(scenario):04d}",
        eval_x=eval_x,
        vis_x=eval_x,
        ground_truth_eval=torch.tensor(ground_truth),
        ground_truth_vis=torch.tensor(ground_truth),
    )
    logger.log_step(current_position, belief)

    for t in range(1, n_steps):
        current_position, current_heading = planner.compute_next_pose(
            current_position, current_heading, belief
        )
        measurement = plume(scenario, current_position) + np.random.normal(0, sigma)
        current_position_tensor = torch.tensor(
            current_position, dtype=torch.float32
        ).unsqueeze(0)
        measurement_tensor = torch.tensor([measurement], dtype=torch.float32)
        belief.update(current_position_tensor, measurement_tensor)
        logger.log_step(current_position, belief)
    logger.save_history(f"beta_{planner.beta}_history.pkl")


def main():
    eval_x = make_evaluation_x(domain_size, n_evaluations)
    for beta in beta_sweep:
        print(f"Running beta: {beta}")
        planner = BayesianOptimizationPlanner(
            domain_size,
            max_step,
            max_turn=max_turn,
            beta=beta,
            num_steps=num_steps,
            num_turns=num_turns,
        )
        for scenario in scenarios:
            print(f"Running scenario: {scenarios.index(scenario)}")
            run_scenario(scenario, eval_x, gp_hyperparams, sigma, planner)


if __name__ == "__main__":
    main()
