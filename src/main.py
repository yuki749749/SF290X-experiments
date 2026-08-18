import logging
import os

import gpytorch
import torch

log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
import joblib

from environment.environment import plume, generate_random_scenario
from belief.belief import Belief, ExactGPModel
from planners.planners import (
    LawnmowerPlanner,
    RandomPlanner,
    BayesianOptimizationPlanner,
)
from planners.diffusion_planner import DiffusionPlanner
from diffusion.model import TemporalUnet
from diffusion.diffusion import GaussianDiffusion
from logger import Logger


scenarios = joblib.load("data/scenarios/test_scenarios_025.pkl")


def plot_simulation_results(
    planner, scenario, position_history, mean_history, variance_history, grid, n_grid
):
    # Select specific timesteps to show (e.g., start, middle, end)
    steps_to_plot = [0, len(position_history) // 2, len(position_history) - 1]
    num_steps = len(steps_to_plot)

    fig, axes = plt.subplots(num_steps, 3, figsize=(15, 5 * num_steps))

    # 1. Pre-calculate Ground Truth for the grid
    ground_truth_grid = np.array([plume(scenario, p) for p in grid.numpy()]).reshape(
        n_grid, n_grid
    )

    # 2. Determine global min/max for consistent colorbars
    v_min, v_max = ground_truth_grid.min(), ground_truth_grid.max()
    var_max = max([var.max().item() for var in variance_history])

    for row, t in enumerate(steps_to_plot):
        # Extract data for current timestep
        pos = np.array(position_history[: t + 1])  # Path taken up to time t
        mean = mean_history[t].reshape(n_grid, n_grid).numpy()
        var = variance_history[t].reshape(n_grid, n_grid).numpy()

        # Column 1: Ground Truth + Path
        im0 = axes[row, 0].imshow(
            ground_truth_grid.T,
            origin="lower",
            extent=[0, 10, 0, 10],
            vmin=v_min,
            vmax=v_max,
        )
        axes[row, 0].plot(pos[:, 0], pos[:, 1], "r.-", markersize=5, label="Path")
        axes[row, 0].set_title(f"T={t}: Ground Truth & Path")
        fig.colorbar(im0, ax=axes[row, 0])

        # Column 2: Predictive Mean
        im1 = axes[row, 1].imshow(
            mean.T, origin="lower", extent=[0, 10, 0, 10], vmin=v_min, vmax=v_max
        )
        axes[row, 1].set_title(f"T={t}: GP Mean")
        fig.colorbar(im1, ax=axes[row, 1])

        # Column 3: Predictive Variance (Uncertainty)
        im2 = axes[row, 2].imshow(
            var.T,
            origin="lower",
            extent=[0, 10, 0, 10],
            vmin=0,
            vmax=var_max,
            cmap="viridis",
        )
        axes[row, 2].set_title(f"T={t}: GP Variance")
        fig.colorbar(im2, ax=axes[row, 2])

        path = np.array(position_history[: t + 1])
        for col in range(3):
            axes[row, col].plot(
                path[:, 0], path[:, 1], "r.-", markersize=5, label="Path"
            )

    planner_name = planner.__class__.__name__
    output_dir = f"plots/scenario_{scenarios.index(scenario):04d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/{planner_name}.png"
    plt.savefig(output_path)
    plt.close()


# AUV state
current_heading = np.pi / 4
current_position = (0, 0)
current_belief = None
max_step = 1.0
max_turn = np.pi / 8


domain_size = (10, 10)
sigma = 0.5  # measurement noise standard deviation

gp_hyperparams_path = "data/hyperparameters/gp_hyperparameters.pkl"
gp_hyperparams = joblib.load(gp_hyperparams_path)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 16  # must match training
REPLAN_EVERY = 15  # replan after 15 executed waypoints
TARGET_RETURN = 1.0  # upper-percentile target; tune to training data

unet = TemporalUnet(
    horizon=HORIZON,
    transition_dim=2,
    belief_dim=1600,
    dim=128,
    dim_mults=(1, 2, 4, 8),
    condition_dropout=0.1,
).to(DEVICE)

diffusion_model = GaussianDiffusion(
    model=unet,
    horizon=HORIZON,
    observation_dim=2,
    n_timesteps=100,
    loss_type="l2",
    clip_denoised=True,
    predict_epsilon=True,
    condition_guidance_w=1.2,
).to(DEVICE)

checkpoint_path = "models/best_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
weights_key = "ema_shadow" if "ema_shadow" in checkpoint else "model"
diffusion_model.load_state_dict(checkpoint[weights_key])
log.info(f"Loaded {weights_key} weights from checkpoint")
diffusion_model.eval()

planners = [
    DiffusionPlanner(
        diffusion_model,
        domain_size=domain_size,
        horizon=HORIZON,
        replan_every=REPLAN_EVERY,
        target_return=TARGET_RETURN,
        device=DEVICE,
    )
]


def main():
    print("Starting simulation...")
    n_grid = 40
    t1 = torch.linspace(0, domain_size[0], n_grid)
    t2 = torch.linspace(0, domain_size[1], n_grid)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    test_grid = torch.stack([g1.flatten(), g2.flatten()], dim=-1)

    for planner in planners:
        print(f"Running planner: {planner}")
        for scenario in scenarios:
            print(f"Running scenario: {scenario}")
            planner.reset()
            ground_truth_np = np.array([plume(scenario, p) for p in test_grid.numpy()])
            ground_truth = torch.tensor(ground_truth_np, dtype=torch.float32)

            logger = Logger(
                f"results/evaluation/scenario_{scenarios.index(scenario):04d}",
                eval_x=test_grid,
                vis_x=test_grid,
                ground_truth_eval=ground_truth,
                ground_truth_vis=ground_truth,
            )
            current_heading = np.pi / 4
            current_position = (0, 0)

            measurement = plume(scenario, current_position) + np.random.normal(0, sigma)
            current_position_tensor = torch.tensor(
                current_position, dtype=torch.float32
            ).unsqueeze(0)
            measurement_tensor = torch.tensor([measurement], dtype=torch.float32)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            gp_model = ExactGPModel(current_position_tensor, measurement_tensor, likelihood)
            if os.path.exists(gp_hyperparams_path):
                with torch.no_grad():
                    gp_model.mean_module.constant = torch.tensor(
                        gp_hyperparams["mean_constant"]
                    )
                    gp_model.covar_module.base_kernel.lengthscale = torch.tensor(
                        [
                            gp_hyperparams["lengthscale_0"],
                            gp_hyperparams["lengthscale_1"],
                        ]
                    )
                    gp_model.covar_module.outputscale = torch.tensor(
                        gp_hyperparams["outputscale"]
                    )
                    gp_model.likelihood.noise = torch.tensor(gp_hyperparams["noise"])
                for param in gp_model.parameters():
                    param.requires_grad_(False)
            else:
                gp_model.likelihood.noise_covar.noise = sigma**2
                gp_model.likelihood.noise_covar.raw_noise.requires_grad_(False)

            current_belief = Belief(gp_model, test_grid)

            logger.log_step(current_position, current_belief)
            print("Initial RMSE:", logger.rmse_history[-1])
            print(
                "Initial normalized trace reduction:",
                logger.normalized_trace_reduction_history[-1],
            )

            for t in range(1, 50):
                print(t)
                new_position, new_heading = planner.compute_next_pose(
                    current_position, current_heading, current_belief
                )
                current_position = new_position
                current_heading = new_heading
                measurement = plume(scenario, current_position) + np.random.normal(
                    0, sigma
                )

                current_position_tensor = torch.tensor(
                    current_position, dtype=torch.float32
                ).unsqueeze(0)
                measurement_tensor = torch.tensor([measurement], dtype=torch.float32)

                current_belief.update(current_position_tensor, measurement_tensor)
                logger.log_step(current_position, current_belief)

            print("Final RMSE:", logger.rmse_history[-1])
            print(
                "Final normalized trace reduction:",
                logger.normalized_trace_reduction_history[-1],
            )

            plot_simulation_results(
                planner,
                scenario,
                logger.position_history,
                logger.mean_history,
                logger.variance_history,
                test_grid,
                n_grid,
            )
            logger.save_history(filename=f"{planner.__class__.__name__}History.pkl")

        print(f"Finished planner: {planner}")


if __name__ == "__main__":
    main()
