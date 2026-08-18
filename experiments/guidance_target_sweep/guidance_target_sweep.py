import hydra
import logging
import os
import numpy as np
import gpytorch
import torch
import joblib

log = logging.getLogger(__name__)


from environment.environment import plume, generateRandomScenario
from belief.belief import Belief, ExactGPModel

from planners.diffusion_planner import DiffusionPlanner
from diffusion.model import TemporalUnet
from diffusion.diffusion import GaussianDiffusion
from diffusion.projection import SequentialProjector
from diffusion.projected_diffusion import ProjectedGaussianDiffusion
from logger import Logger
from utils import abs_path, get_output_dir


def get_planner(cfg):
    if cfg.planner.type == "diffusion":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        unet = TemporalUnet(
            horizon=cfg.architecture.horizon,
            transition_dim=cfg.architecture.transition_dim,
            belief_dim=cfg.architecture.belief_dim,
            dim=cfg.architecture.dim,
            dim_mults=tuple(cfg.architecture.dim_mults),
            condition_dropout=cfg.architecture.condition_dropout,
            kernel_size=cfg.architecture.kernel_size,
            crop_size=cfg.architecture.get("crop_size", 40),
            belief_encoder_pooling=cfg.architecture.get("belief_encoder_pooling", False),
        ).to(device)

        projector = SequentialProjector(
            v=1.0,
            max_turn=np.pi / 4,
            domain_size=tuple(cfg.domain_size),
            domain_pad=cfg.domain_pad,
        )

        diffusion_model = ProjectedGaussianDiffusion(
            projector=projector,
            model=unet,
            horizon=cfg.diffusion.horizon,
            observation_dim=cfg.diffusion.observation_dim,
            n_timesteps=cfg.diffusion.n_timesteps,
            loss_type=cfg.diffusion.loss_type,
            clip_denoised=cfg.diffusion.clip_denoised,
            predict_epsilon=cfg.diffusion.predict_epsilon,
            condition_guidance_w=cfg.diffusion.condition_guidance_w,
            belief_guidance_w=cfg.diffusion.get("belief_guidance_w", None),
            return_guidance_w=cfg.diffusion.get("return_guidance_w", None),
        ).to(device)

        checkpoint = torch.load(
            abs_path(cfg.planner.checkpoint_path), map_location=device
        )
        weights_key = "ema_shadow" if "ema_shadow" in checkpoint else "model"
        diffusion_model.load_state_dict(checkpoint[weights_key])
        log.info(f"Loaded {weights_key} weights from checkpoint")
        diffusion_model.eval()

        return DiffusionPlanner(
            diffusion=diffusion_model,
            domain_size=tuple(cfg.domain_size),
            domain_pad=cfg.domain_pad,
            horizon=cfg.diffusion.horizon,
            replan_every=cfg.planner.replan_every,
            target_return=cfg.planner.target_return,
            device=device,
            use_ddim=cfg.planner.use_ddim,
            ddim_steps=cfg.planner.ddim_steps,
            ddim_eta=cfg.planner.ddim_eta,
            warm_start=cfg.planner.warm_start,
            noise_steps=cfg.planner.noise_steps,
            use_belief=cfg.planner.use_belief,
            use_return=cfg.planner.use_return,
            crop_size=cfg.architecture.get("crop_size", 40),
            domain_min=list(cfg.domain_min),
            domain_max=list(cfg.domain_max),
        )
    else:
        raise ValueError(f"Unknown planner type: {cfg.planner}")


def initialize_belief(position, measurement, gp_hyperparams, sigma, eval_x):
    train_x = torch.tensor(position, dtype=torch.float32).unsqueeze(0)
    train_y = torch.tensor([measurement], dtype=torch.float32)
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
):
    planner.reset(initial_position=initial_position, initial_heading=initial_heading)
    ground_truth_eval_np = np.array([plume(scenario, p) for p in evaluation_x.numpy()])
    ground_truth_eval = torch.tensor(ground_truth_eval_np, dtype=torch.float32)

    ground_truth_vis_np = np.array(
        [plume(scenario, p) for p in visualization_x.numpy()]
    )
    ground_truth_vis = torch.tensor(ground_truth_vis_np, dtype=torch.float32)

    initial_measurement = plume(scenario, initial_position) + np.random.normal(0, sigma)

    belief = initialize_belief(
        position=initial_position,
        measurement=initial_measurement,
        gp_hyperparams=gp_hyperparams,
        sigma=sigma,
        eval_x=evaluation_x,
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

    for t in range(1, n_timesteps + 1):
        current_position, current_heading = planner.compute_next_pose(
            current_position, current_heading, belief
        )

        x, y = current_position
        in_domain = True

        if in_domain:
            current_position_tensor = torch.tensor(
                current_position, dtype=torch.float32
            ).unsqueeze(0)
            measurement = plume(scenario, current_position) + np.random.normal(0, sigma)
            measurement_tensor = torch.tensor([measurement], dtype=torch.float32)

            belief.update(current_position_tensor, measurement_tensor)
        logger.log_step(current_position, belief)
    return logger


def make_grid(domain_min, domain_max, n_evaluations):
    side = int(np.sqrt(n_evaluations))
    t1 = torch.linspace(domain_min[0], domain_max[0], side)
    t2 = torch.linspace(domain_min[1], domain_max[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)


@hydra.main(config_path="../../config", config_name="guidance_target_sweep", version_base="1.2")
def main(cfg):
    torch.manual_seed(cfg.planner_seed + cfg.scenario_idx)
    np.random.seed(cfg.seed + cfg.scenario_idx)
    output_dir = get_output_dir()
    planner = get_planner(cfg)

    gp_hyperparams = joblib.load(abs_path(cfg.paths.hyperparameters.gp))

    import random
    random.seed(cfg.planner_seed + cfg.scenario_idx)
    scenario = generateRandomScenario(
        sourceRange=tuple(cfg.source_range),
        domainSize=tuple(cfg.domain_size),
        intensityRange=tuple(cfg.intensity_range)
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
    )
    logger.save_history("history.pkl")


if __name__ == "__main__":
    main()
