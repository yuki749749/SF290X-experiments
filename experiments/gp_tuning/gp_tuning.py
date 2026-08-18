import os
import sys
import numpy as np
import joblib
import gpytorch
import torch
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig

# Add src folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from belief.belief import ExactGPModel
from environment.environment import plume, generate_random_scenario
from utils import abs_path, get_output_dir

log = logging.getLogger(__name__)


def make_train_x(domain_size, n_grid):
    """Regular grid of training inputs."""
    side = int(np.sqrt(n_grid))
    t1 = torch.linspace(0, domain_size[0], side)
    t2 = torch.linspace(0, domain_size[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)  # (side², 2)


def train_model(model, train_x, train_y, n_iter=10):
    """Train the model using LBFGS optimizer."""
    model.train()
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    likelihood = model.likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss

    for i in range(n_iter):
        loss = optimizer.step(closure)
        log.info(f"  Iter {i + 1:3d}/{n_iter} - Loss: {loss.item():.3f}")
    model.eval()


def fit_scenario(scenario, train_x, sigma, n_iter):
    train_y = torch.tensor(
        [plume(scenario, p) for p in train_x.numpy()], dtype=torch.float32
    ) + sigma * torch.randn(train_x.shape[0])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model.likelihood.noise_covar.noise = sigma**2
    model.likelihood.noise_covar.raw_noise.requires_grad_(False)

    train_model(model, train_x, train_y, n_iter=n_iter)

    return {
        "mean_constant": model.mean_module.constant.item(),
        "outputscale": model.covar_module.outputscale.item(),
        "lengthscale_0": model.covar_module.base_kernel.lengthscale[0, 0].item(),
        "lengthscale_1": model.covar_module.base_kernel.lengthscale[0, 1].item(),
        "noise": model.likelihood.noise.item(),
    }


def aggregate_results(results):
    """Aggregate results across scenarios by taking the median."""
    aggregated = {}
    for key in results[0].keys():
        aggregated[key] = np.median([result[key] for result in results])
    return aggregated


@hydra.main(config_path="../../config", config_name="gp_tuning", version_base="1.2")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = get_output_dir()
    log.info(f"GP Tuning output directory: {output_dir}")

    log.info(f"Generating {cfg.gp_tuning_scenarios} GP tuning scenarios on-the-fly...")
    import random
    scenarios = []
    for idx in range(cfg.gp_tuning_scenarios):
        random.seed(cfg.gp_tuning_seed + idx)
        scenarios.append(
            generate_random_scenario(
                source_range=tuple(cfg.source_range),
                domain_size=tuple(cfg.domain_size),
                intensity_range=tuple(cfg.intensity_range)
            )
        )
    train_x = make_train_x(tuple(cfg.domain_size), cfg.n_observations)

    results = []
    log.info(f"Fitting GP hyperparameters over {len(scenarios)} scenarios...")
    for idx, scenario in enumerate(scenarios):
        log.info(f"Fitting scenario {idx + 1}/{len(scenarios)}...")
        hyperparameters = fit_scenario(scenario, train_x, cfg.sigma, cfg.n_iter)
        results.append(hyperparameters)

    aggregated = aggregate_results(results)

    # 1. Save aggregated hyperparameters in data directory
    gp_output_path = abs_path(cfg.paths.hyperparameters.gp)
    os.makedirs(os.path.dirname(gp_output_path), exist_ok=True)
    joblib.dump(aggregated, gp_output_path)
    log.info(f"Saved aggregated GP hyperparameters to central path: {gp_output_path}")

    # 2. Save both aggregated and full results inside Hydra run folder for reference
    results_path = os.path.join(output_dir, "gp_tuning_results.pkl")
    run_hyperparams_path = os.path.join(output_dir, "gp_hyperparameters.pkl")
    
    output = {
        "per_scenario": results,
        "aggregated": aggregated,
        "config": {
            "domain_size": list(cfg.domain_size),
            "source_range": list(cfg.source_range),
            "intensity_range": list(cfg.intensity_range),
            "sigma": cfg.sigma,
            "n_observations": cfg.n_observations,
            "n_iter": cfg.n_iter,
            "seed": cfg.seed,
        }
    }
    joblib.dump(output, results_path)
    joblib.dump(aggregated, run_hyperparams_path)
    log.info(f"Saved run results to: {results_path}")
    log.info(f"Saved run hyperparameters to: {run_hyperparams_path}")

    log.info("\nAggregated hyperparameters (median across scenarios):")
    for k, v in aggregated.items():
        vals = [r[k] for r in results]
        log.info(f"  {k:<20s}  median={v:.4f}   "
                 f"[{np.min(vals):.4f}, {np.max(vals):.4f}]  "
                 f"std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
