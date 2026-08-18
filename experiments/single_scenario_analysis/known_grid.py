"""
Single scenario analysis: sample N trajectories from the diffusion planner
for each conditioning variant (belief+reward, belief-only, reward-only,
unconditioned), given a belief initialized from observations within a
user-specified known region of the field.
"""

import logging
import os
import numpy as np
import gpytorch
import torch
import joblib
import hydra

log = logging.getLogger(__name__)

from environment.environment import plume
from belief.belief import Belief, ExactGPModel
from diffusion.model import TemporalUnet
from diffusion.projection import SequentialProjector
from diffusion.projected_diffusion import ProjectedGaussianDiffusion
from diffusion.helpers import crop_belief_map
from utils import abs_path, get_output_dir


VARIANTS = [
    ("belief_and_reward", True, True),
    ("belief_only",        True, False),
    ("reward_only",        False, True),
    ("unconditioned",      False, False),
]


def make_grid(domain_min, domain_max, n_evaluations):
    side = int(np.sqrt(n_evaluations))
    t1 = torch.linspace(domain_min[0], domain_max[0], side)
    t2 = torch.linspace(domain_min[1], domain_max[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)


def make_known_region_grid(region_min, region_max, n_side):
    """Uniform grid of training points inside the known region."""
    t1 = torch.linspace(region_min[0], region_max[0], n_side)
    t2 = torch.linspace(region_min[1], region_max[1], n_side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)  # (n_side², 2)


def initialize_belief_from_region(known_x, eval_x, scenario, gp_hyperparams, sigma):
    """Seed the GP with noisy observations at the known-region points.

    known_x : (M, 2) grid of points where the field value is revealed.
    eval_x  : (N, 2) full evaluation grid stored on the Belief for prediction.
    """
    positions_np = known_x.numpy()
    measurements = np.array(
        [plume(scenario, p) + np.random.normal(0, sigma) for p in positions_np],
        dtype=np.float32,
    )

    train_x = known_x.float()
    train_y = torch.tensor(measurements, dtype=torch.float32)

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


def build_diffusion_model(cfg, device):
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

    diffusion = ProjectedGaussianDiffusion(
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

    checkpoint = torch.load(abs_path(cfg.planner.checkpoint_path), map_location=device)
    weights_key = "ema_shadow" if "ema_shadow" in checkpoint else "model"
    diffusion.load_state_dict(checkpoint[weights_key])
    log.info(f"Loaded {weights_key} weights from checkpoint")
    diffusion.eval()
    return diffusion


def _normalise(pos_np, domain_size, domain_pad):
    t = torch.tensor(pos_np, dtype=torch.float32)
    ds = torch.tensor(domain_size, dtype=torch.float32)
    dp = torch.tensor(domain_pad, dtype=torch.float32)
    return (t + dp) / (ds + 2 * dp) * 2.0 - 1.0


def _denormalise(tau_norm, domain_size, domain_pad):
    ds = torch.tensor(domain_size, dtype=torch.float32)
    dp = torch.tensor(domain_pad, dtype=torch.float32)
    return ((tau_norm + 1.0) / 2.0 * (ds + 2 * dp) - dp).cpu().numpy()


def sample_trajectories(
    diffusion,
    belief,
    initial_position,
    initial_heading,
    n_samples,
    target_return,
    domain_size,
    domain_pad,
    use_belief,
    use_return,
    device,
    use_ddim=False,
    ddim_steps=25,
    ddim_eta=0.0,
    crop_size=40,
    domain_min=None,
    domain_max=None,
    grid_size=40,
):
    """Return (n_samples, horizon, 2) numpy array of denormalised trajectories."""
    b_mean, b_var = belief.predict_eval_x()  # (grid_size²,) CPU
    if crop_size < grid_size:
        agent_pos = torch.tensor(initial_position, dtype=torch.float32)
        dmin = torch.tensor(domain_min or [0.0, 0.0], dtype=torch.float32)
        dmax = torch.tensor(domain_max or list(domain_size[:2]), dtype=torch.float32)
        b_mean, b_var = crop_belief_map(
            b_mean, b_var, agent_pos, dmin, dmax, crop_size, grid_size
        )

    init_norm = _normalise(np.array(initial_position, dtype=np.float32), domain_size, domain_pad)

    # Ghost waypoint one step behind initial position along initial heading,
    # matching DiffusionPlanner.reset() with max_step=1.0.
    ghost_np = np.array(
        [
            initial_position[0] - np.cos(initial_heading),
            initial_position[1] - np.sin(initial_heading),
        ],
        dtype=np.float32,
    )
    ghost_norm = _normalise(ghost_np, domain_size, domain_pad)

    cond = {
        0: ghost_norm.unsqueeze(0).expand(n_samples, -1).contiguous().to(device),
        1: init_norm.unsqueeze(0).expand(n_samples, -1).contiguous().to(device),
    }
    b_mean_b = b_mean.unsqueeze(0).expand(n_samples, -1).contiguous().to(device)
    b_var_b  = b_var.unsqueeze(0).expand(n_samples, -1).contiguous().to(device)
    returns  = torch.full((n_samples, 1), target_return, dtype=torch.float32, device=device)

    with torch.no_grad():
        tau_norm = diffusion.conditional_sample(
            cond,
            b_mean_b,
            b_var_b,
            returns,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            use_belief=use_belief,
            use_return=use_return,
        )  # (N, horizon, 2)

    return _denormalise(tau_norm, domain_size, domain_pad)  # (N, horizon, 2)


@hydra.main(
    config_path="../../config",
    config_name="known_grid",
    version_base="1.2",
)
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = get_output_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    evaluation_x = make_grid(cfg.domain_min, cfg.domain_max, cfg.n_evaluations)
    gp_hyperparams = joblib.load(abs_path(cfg.paths.hyperparameters.gp))

    scenario = [((cfg.scenario.source_x, cfg.scenario.source_y), cfg.scenario.intensity)]

    known_x = make_known_region_grid(
        cfg.known_region.min, cfg.known_region.max, cfg.known_region.n_side
    )
    n_known = known_x.shape[0]
    print(f"Initializing belief from {n_known} points in known region "
          f"{cfg.known_region.min} – {cfg.known_region.max} ...")
    belief = initialize_belief_from_region(known_x, evaluation_x, scenario, gp_hyperparams, cfg.sigma)
    belief.predict_eval_x()  # populate GPyTorch caches before fantasy updates

    initial_position = tuple(cfg.initial_position)
    initial_heading = float(cfg.initial_heading)

    initial_measurement = plume(scenario, initial_position) + np.random.normal(0, cfg.sigma)
    belief.update(
        torch.tensor(initial_position, dtype=torch.float32).unsqueeze(0),
        torch.tensor([initial_measurement], dtype=torch.float32),
    )
    print(f"Initial position: {initial_position}, heading: {initial_heading:.4f} rad, "
          f"measurement: {initial_measurement:.4f}")

    print("Building diffusion model...")
    diffusion = build_diffusion_model(cfg, device)

    results = {}
    for name, use_belief_flag, use_return_flag in VARIANTS:
        print(f"Sampling {cfg.n_samples} trajectories [{name}] ...")
        trajectories = sample_trajectories(
            diffusion=diffusion,
            belief=belief,
            initial_position=initial_position,
            initial_heading=initial_heading,
            n_samples=cfg.n_samples,
            target_return=cfg.planner.target_return,
            domain_size=tuple(cfg.domain_size),
            domain_pad=cfg.domain_pad,
            use_belief=use_belief_flag,
            use_return=use_return_flag,
            device=device,
            use_ddim=cfg.planner.use_ddim,
            ddim_steps=cfg.planner.ddim_steps,
            ddim_eta=cfg.planner.ddim_eta,
            crop_size=cfg.architecture.get("crop_size", 40),
            domain_min=list(cfg.domain_min),
            domain_max=list(cfg.domain_max),
        )
        results[name] = trajectories  # (N, horizon, 2)
        print(f"  done — trajectory shape: {trajectories.shape}")

    b_mean, b_var = belief.predict_eval_x()
    output = {
        "results": results,           # dict[str -> (N, horizon, 2) ndarray]
        "belief_mean": b_mean.numpy(),
        "belief_var": b_var.numpy(),
        "eval_x": evaluation_x.numpy(),
        "known_x": known_x.numpy(),
        "initial_position": initial_position,
        "initial_heading": initial_heading,
        "scenario": scenario,
    }

    save_path = os.path.join(output_dir, "trajectories.pkl")
    joblib.dump(output, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
