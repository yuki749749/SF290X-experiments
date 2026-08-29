from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
import os
from pathlib import Path

def abs_path(relative: str) -> str:
    if os.path.isabs(relative):
        return relative
    try:
        return os.path.join(get_original_cwd(), relative)
    except ValueError:
        return os.path.abspath(relative)

def get_output_dir() -> str:
    """Return Hydra's resolved output directory for the current run."""
    return HydraConfig.get().runtime.output_dir

def make_grid(domain_min, domain_max, n_evaluations):
    """Uniform grid of evaluation points."""
    import torch
    import numpy as np
    side = int(np.sqrt(n_evaluations))
    t1 = torch.linspace(domain_min[0], domain_max[0], side)
    t2 = torch.linspace(domain_min[1], domain_max[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)

def initialize_belief(position, measurement, gp_hyperparams, sigma, eval_x, diffusion_coefficient=2.0):
    """Initialize SimpleMaternGP model and Belief state."""
    import torch
    from belief.belief import Belief, SimpleMaternGP
    
    train_x = torch.tensor(position, dtype=torch.float32).unsqueeze(0)  # (1, 2)
    train_y = torch.tensor([measurement], dtype=torch.float32)  # (1,)
    
    model = SimpleMaternGP(
        mean_constant=gp_hyperparams["mean_constant"],
        lengthscale=[gp_hyperparams["lengthscale_0"], gp_hyperparams["lengthscale_1"]],
        outputscale=gp_hyperparams["outputscale"],
        noise=sigma**2,
        train_x=train_x,
        train_y=train_y,
    )
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
    update_in_domain_only=False,
    stride=None,
    horizon=None,
):
    """Simulate a planner trajectory step-by-step, updating the belief and logging."""
    import numpy as np
    import torch
    from environment.environment import plume
    from logger import Logger

    planner.reset(initial_position=initial_position, initial_heading=initial_heading)
    
    # Vectorized plume computation for evaluation and visualization grids
    ground_truth_eval = torch.zeros(evaluation_x.shape[0], dtype=torch.float32, device=evaluation_x.device)
    for source_pos, source_intensity in scenario:
        source_pos_t = torch.tensor(source_pos, dtype=torch.float32, device=evaluation_x.device)
        dists = torch.norm(evaluation_x - source_pos_t, p=2, dim=-1)
        ground_truth_eval += source_intensity * torch.exp(-dists / diffusion_coefficient)

    ground_truth_vis = torch.zeros(visualization_x.shape[0], dtype=torch.float32, device=visualization_x.device)
    for source_pos, source_intensity in scenario:
        source_pos_t = torch.tensor(source_pos, dtype=torch.float32, device=visualization_x.device)
        dists = torch.norm(visualization_x - source_pos_t, p=2, dim=-1)
        ground_truth_vis += source_intensity * torch.exp(-dists / diffusion_coefficient)

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
        stride=stride,
        horizon=horizon,
        n_timesteps=n_timesteps,
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
        if update_in_domain_only:
            in_domain = (0 <= x <= W) and (0 <= y <= H)
        else:
            in_domain = True

        if in_domain:
            current_position_tensor = torch.tensor(
                current_position, dtype=torch.float32
            ).unsqueeze(0)
            measurement = plume(scenario, current_position, diffusion_coefficient) + np.random.normal(0, sigma)
            measurement_tensor = torch.tensor([measurement], dtype=torch.float32)

            belief.update(current_position_tensor, measurement_tensor)
        logger.log_step(current_position, belief)
    return logger

def get_planner(cfg):
    """Instantiate a planner based on cfg."""
    import numpy as np
    import torch
    import logging
    from planners.planners import (
        LawnmowerPlanner,
        RandomPlanner,
        BayesianOptimizationPlanner,
    )
    
    log = logging.getLogger(__name__)

    if cfg.planner.type == "lawnmower":
        return LawnmowerPlanner(
            domain_size=cfg.domain_size,
            domain_pad=cfg.domain_pad,
            max_step=cfg.planner.max_step,
            min_step=cfg.planner.min_step,
            max_turn=cfg.planner.max_turn,
            boundary_behavior=cfg.planner.boundary_behavior,
            orientation=cfg.planner.orientation,
            edge_turn_steps=cfg.planner.edge_turn_steps,
        )
    elif cfg.planner.type == "random":
        return RandomPlanner(
            domain_size=cfg.domain_size,
            domain_pad=cfg.get("domain_pad", 5.0),
            max_step=cfg.planner.max_step,
            min_step=cfg.planner.min_step,
            max_turn=cfg.planner.max_turn,
            boundary_behavior=cfg.planner.boundary_behavior,
        )
    elif cfg.planner.type == "bo":
        return BayesianOptimizationPlanner(
            domain_size=cfg.domain_size,
            domain_pad=cfg.get("domain_pad", 5.0),
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
    elif cfg.planner.type == "diffusion":
        from planners.diffusion_planner import DiffusionPlanner
        from diffusion.model import TemporalUnet
        from diffusion.projection import SequentialProjector
        from diffusion.projected_diffusion import ProjectedGaussianDiffusion

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
            conditioning_type=cfg.architecture.get("conditioning_type", "cnn"),
        ).to(device)

        projector = SequentialProjector(
            v=cfg.planner.max_step,
            max_turn=cfg.planner.max_turn,
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
            n_cond_steps=cfg.diffusion.get("n_cond_steps", 2),
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

        # Check if normalization stats exist in the checkpoint
        stats = checkpoint.get("stats", None)
        # If not, try to load from the same folder as the checkpoint
        if stats is None:
            checkpoint_dir = os.path.dirname(abs_path(cfg.planner.checkpoint_path))
            stats_path = os.path.join(checkpoint_dir, "training_data_stats.json")
            if os.path.exists(stats_path):
                try:
                    import json
                    with open(stats_path, "r") as f:
                        stats = json.load(f)
                    log.info(f"Loaded normalization stats from: {stats_path}")
                except Exception as exc:
                    log.warning(f"Error loading stats from {stats_path}: {exc}")

        return DiffusionPlanner(
            diffusion=diffusion_model,
            domain_size=tuple(cfg.domain_size),
            domain_pad=cfg.domain_pad,
            horizon=cfg.diffusion.horizon,
            replan_every=cfg.planner.replan_every,
            target_return=cfg.planner.target_return,
            device=device,
            warm_start=cfg.planner.get("warm_start", False),
            noise_steps=cfg.planner.get("noise_steps", 20),
            use_belief=cfg.planner.get("use_belief", True),
            use_return=cfg.planner.get("use_return", True),
            crop_size=cfg.architecture.get("crop_size", 40),
            domain_min=list(cfg.domain_min),
            domain_max=list(cfg.domain_max),
            max_step=cfg.planner.max_step,
            stats=stats,
            use_egocentric=cfg.planner.get("use_egocentric", False) or cfg.architecture.get("use_egocentric", False),
            max_turn=cfg.planner.max_turn,
        )
    elif cfg.planner.type == "diffusion_bo":
        from planners.diffusion_bo_planner import DiffusionBOPlanner
        from diffusion.model import TemporalUnet
        from diffusion.projection import SequentialProjector
        from diffusion.projected_diffusion import ProjectedGaussianDiffusion

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
            conditioning_type=cfg.architecture.get("conditioning_type", "cnn"),
        ).to(device)

        projector = SequentialProjector(
            v=cfg.planner.max_step,
            max_turn=cfg.planner.max_turn,
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
            n_cond_steps=cfg.diffusion.get("n_cond_steps", 2),
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

        stats = checkpoint.get("stats", None)
        if stats is None:
            checkpoint_dir = os.path.dirname(abs_path(cfg.planner.checkpoint_path))
            stats_path = os.path.join(checkpoint_dir, "training_data_stats.json")
            if os.path.exists(stats_path):
                try:
                    import json
                    with open(stats_path, "r") as f:
                        stats = json.load(f)
                    log.info(f"Loaded normalisation stats from: {stats_path}")
                except Exception as exc:
                    log.warning(f"Error loading stats from {stats_path}: {exc}")

        return DiffusionBOPlanner(
            diffusion=diffusion_model,
            domain_size=tuple(cfg.domain_size),
            domain_pad=cfg.domain_pad,
            horizon=cfg.diffusion.horizon,
            replan_every=cfg.planner.replan_every,
            n_samples=cfg.planner.get("n_samples", 10),
            acquisition=cfg.planner.get("acquisition", "ucb"),
            beta=cfg.planner.get("beta", 20.0),
            gamma=cfg.planner.get("gamma", 1.0),
            device=device,
            warm_start=cfg.planner.get("warm_start", True),
            noise_steps=cfg.planner.get("noise_steps", 20),
            domain_min=list(cfg.domain_min),
            domain_max=list(cfg.domain_max),
            grid_size=int(cfg.n_evaluations ** 0.5),
            max_step=cfg.planner.max_step,
            max_turn=cfg.planner.max_turn,
            stats=stats,
            crop_size=cfg.architecture.get("crop_size", 40),
            acq_mask_sharpness=cfg.planner.get("acq_mask_sharpness", 3.0),
            boundary_penalty=cfg.planner.get("boundary_penalty", 1e4),
            wivr_weighting=cfg.planner.get("wivr_weighting", "hybrid"),
            wivr_grid_size=cfg.planner.get("wivr_grid_size", 20),
            guidance=cfg.planner.get("guidance", False),
            guidance_scale=cfg.planner.get("guidance_scale", 1.0),
            guidance_max_disp=cfg.planner.get("guidance_max_disp", 5.0),
        )
    elif cfg.planner.type == "random_bo":
        from planners.random_bo_planner import RandomBOPlanner
        device = "cuda" if torch.cuda.is_available() else "cpu"
        horizon_val = cfg.diffusion.horizon if hasattr(cfg, "diffusion") and hasattr(cfg.diffusion, "horizon") else cfg.planner.get("horizon", 16)
        return RandomBOPlanner(
            domain_size=tuple(cfg.domain_size),
            domain_pad=cfg.domain_pad,
            horizon=horizon_val,
            replan_every=cfg.planner.replan_every,
            n_samples=cfg.planner.get("n_samples", 100),
            acquisition=cfg.planner.get("acquisition", "wivr"),
            beta=cfg.planner.get("beta", 10.0),
            gamma=cfg.planner.get("gamma", 1.0),
            device=device,
            domain_min=list(cfg.domain_min),
            domain_max=list(cfg.domain_max),
            grid_size=int(cfg.n_evaluations ** 0.5),
            max_step=cfg.planner.max_step,
            max_turn=cfg.planner.max_turn,
            boundary_penalty=cfg.planner.get("boundary_penalty", 0.0),
            wivr_weighting=cfg.planner.get("wivr_weighting", "plume"),
            wivr_grid_size=cfg.planner.get("wivr_grid_size", 20),
            arc_prob=cfg.planner.get("arc_prob", 0.5),
        )
    else:
        raise ValueError(f"Unknown planner type: {cfg.planner}")

def resolve_sweep_root(sweep_dir: str | Path) -> Path:
    """
    If sweep_dir already contains scenario_* subdirs, return it directly.
    Otherwise descend into the lexicographically latest subdirectory
    (auto-selects the most recent timestamp).
    """
    sweep_dir = Path(sweep_dir)
    if any(sweep_dir.glob("scenario_*")):
        return sweep_dir
    subdirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {sweep_dir}")
    latest = subdirs[-1]
    print(f"Auto-selected timestamp: {latest.name}")
    return latest

def find_latest_run_dir(results_root: str | Path, experiment_name: str, required_file: str) -> Path:
    """Return the most recently modified run dir under results_root/experiment_name/run/* containing required_file."""
    search_root = Path(results_root) / experiment_name / "run"
    candidates = sorted(
        (p for p in search_root.glob("*") if (p / required_file).exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {required_file} found under '{search_root}'. "
            "Check that the experiment was run or pass the directory explicitly."
        )
    latest = candidates[-1]
    print(f"Auto-detected latest run: {latest}")
    return latest

def find_latest_sweep(sweep_root: str | Path) -> str:
    """Return the path to the most recently modified sweep timestamp directory as a string."""
    sweep_root = Path(sweep_root)
    candidates = sorted(
        [d for d in sweep_root.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No sweep directories found under '{sweep_root}'. "
            "Run generate_raw_trajectories with -m first."
        )
    return str(candidates[-1])