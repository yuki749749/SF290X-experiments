import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gpytorch
import torch

from belief.belief import ExactGPModel
from environment.environment import plume


domain_size = (15.0, 15.0)
source_range = (1, 3)
intensity_range = (50, 200)
sigma = 0.5
n_observations = 64
n_iter = 5
seed = 42

scenario_path = "data/scenarios/gp_tuning_scenarios.pkl"
results_dir = "results/gp_tuning/"
hyperparameters_dir = "data/hyperparameters/"


def make_train_x(domain_size, n_grid):
    """Regular grid of training inputs."""
    side = int(np.sqrt(n_grid))
    t1 = torch.linspace(0, domain_size[0], side)
    t2 = torch.linspace(0, domain_size[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    return torch.stack([g1.flatten(), g2.flatten()], dim=-1)  # (side², 2)


def train(model, train_x, train_y, n_iter=10, lr=0.1):
    """Train the model.

    Arguments
    model   --  The model to train.
    train_x --  The training inputs.
    train_y --  The training labels.
    n_iter  --  The number of iterations.
    """
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
        if (i + 1) % 1 == 0:
            print(f"Iter {i + 1:3d}/{n_iter} - Loss: {loss.item():.3f}")
    model.eval()

def fit_scenario(scenario, train_x, sigma, n_iter):
    train_y = torch.tensor(
            [plume(scenario, p) for p in train_x.numpy()], dtype=torch.float32
        ) + sigma * torch.randn(train_x.shape[0])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model.likelihood.noise_covar.noise = sigma**2
    model.likelihood.noise_covar.raw_noise.requires_grad_(False)

    train(model, train_x, train_y, n_iter=n_iter)

    return {
        "mean_constant": model.mean_module.constant.item(),
        "outputscale": model.covar_module.outputscale.item(),
        "lengthscale_0": model.covar_module.base_kernel.lengthscale[0, 0].item(),
        "lengthscale_1": model.covar_module.base_kernel.lengthscale[0, 1].item(),
        "noise": model.likelihood.noise.item(),
    }


def aggregate_results(results):
    """Aggregate results across scenarios by taking the mean."""
    aggregated = {}
    for key in results[0].keys():
        aggregated[key] = np.median([result[key] for result in results])
    return aggregated

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    scenarios = joblib.load(scenario_path)
    train_x = make_train_x(domain_size, n_observations)

    for scenario in scenarios:
        hyperparameters = fit_scenario(scenario, train_x, sigma, n_iter)
        results.append(hyperparameters)

    aggregated = aggregate_results(results)
    joblib.dump(aggregated, os.path.join(hyperparameters_dir, "gp_hyperparameters.pkl"))

    output = {
        "per_scenario": results,
        "aggregated": aggregated,
        "config": {
            "domain_size": domain_size,
            "source_range": source_range,
            "intensity_range": intensity_range,
            "sigma": sigma,
            "n_observations": n_observations,
            "n_iter": n_iter,
            "seed": seed,
        }
    }
    joblib.dump(output, os.path.join(results_dir, "gp_tuning_results.pkl"))

    print("Aggregated hyperparameters (median across scenarios):")
    for k, v in aggregated.items():
        vals = [r[k] for r in results]
        print(f"  {k:<20s}  median={v:.4f}   "
              f"[{np.min(vals):.4f}, {np.max(vals):.4f}]  "
              f"std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
