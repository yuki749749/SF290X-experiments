"""
tune_hyperparams.py — Offline GP hyperparameter calibration.

Strategy
--------
For each scenario in a held-out training set, place a dense observation grid,
fit an ExactGP by maximising the marginal log-likelihood (MLL), and record the
optimal hyperparameters.  The per-scenario estimates are aggregated via the
median (robust to outlier scenarios) and saved to a .pkl file that main.py
can load and inject before any planner runs.

Noise variance is treated as *known* (sigma² from the sensor model) and is
fixed throughout — only the GP prior parameters are learned.

Usage
-----
    python tune_hyperparams.py                         # uses defaults
    python tune_hyperparams.py --n_scenarios 50 \
                                --n_obs_per_scenario 64 \
                                --n_iter 30 \
                                --output hyperparams.pkl
"""

import argparse
import os
import sys

import gpytorch
import joblib
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))  # make local imports work

from belief.belief import ExactGPModel
from environment.environment import generate_random_scenario, plume

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Offline GP hyperparameter tuning")
    p.add_argument("--domain_size",          type=float, nargs=2, default=[10.0, 10.0])
    p.add_argument("--source_range",         type=int,   nargs=2, default=[1, 3])
    p.add_argument("--intensity_range",      type=float, nargs=2, default=[50.0, 200.0])
    p.add_argument("--sigma",                type=float, default=0.5,
                   help="Known sensor noise std-dev (fixes likelihood noise to sigma²)")
    p.add_argument("--n_scenarios",          type=int,   default=100,
                   help="Number of training scenarios to optimise over")
    p.add_argument("--n_obs_per_scenario",   type=int,   default=64,
                   help="Sqrt gives grid side — 64 → 8×8 dense grid per scenario")
    p.add_argument("--n_iter",               type=int,   default=20,
                   help="LBFGS iterations per scenario")
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--output",               type=str,   default="data/hyperparameters/gp_tuning_results.pkl")
    p.add_argument("--scenarios_file",       type=str,   default=None,
                   help="Optional path to pre-saved scenarios .pkl (overrides random generation)")
    return p.parse_args()

# ── Dense observation grid for a scenario ────────────────────────────────────

def make_obs_grid(domain_size, n_obs, sigma):
    """Regular grid of observations with additive Gaussian noise."""
    side = int(np.sqrt(n_obs))
    t1 = torch.linspace(0, domain_size[0], side)
    t2 = torch.linspace(0, domain_size[1], side)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    grid = torch.stack([g1.flatten(), g2.flatten()], dim=-1)  # (side², 2)
    return grid

# ── Per-scenario MLL optimisation ────────────────────────────────────────────

def fit_scenario(scenario, grid, sigma, n_iter):
    """
    Fit an ExactGP to one scenario by maximising MLL.

    Returns
    -------
    dict of scalar hyperparameter values (floats).
    """
    # --- observations ---
    truth_np = np.array([plume(scenario, p) for p in grid.numpy()])
    truth    = torch.tensor(truth_np, dtype=torch.float32)
    noisy_y  = truth + sigma * torch.randn_like(truth)

    # --- model ---
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model      = ExactGPModel(grid, noisy_y, likelihood)

    # Fix noise to known value — sensor model is trusted
    model.likelihood.noise_covar.noise = sigma ** 2
    model.likelihood.noise_covar.raw_noise.requires_grad_(False)

    # --- train ---
    model.train()
    likelihood.train()
    optimizer = torch.optim.LBFGS(
        [p for p in model.parameters() if p.requires_grad],
        line_search_fn="strong_wolfe",
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        optimizer.zero_grad()
        loss = -mll(model(grid), noisy_y)
        loss.backward()
        return loss

    for _ in range(n_iter):
        optimizer.step(closure)

    model.eval()

    return {
        "mean_constant": model.mean_module.constant.item(),
        "outputscale":   model.covar_module.outputscale.item(),
        "lengthscale_0": model.covar_module.base_kernel.lengthscale[0, 0].item(),
        "lengthscale_1": model.covar_module.base_kernel.lengthscale[0, 1].item(),
        "noise":         model.likelihood.noise_covar.noise.item(),
    }

# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(records: list[dict], method: str = "median") -> dict:
    """Aggregate per-scenario dicts via median (default) or mean."""
    keys = records[0].keys()
    fn   = np.median if method == "median" else np.mean
    return {k: float(fn([r[k] for r in records])) for k in keys}

# ── Inject into model ─────────────────────────────────────────────────────────

def apply_hyperparams(model: ExactGPModel, hp: dict, freeze: bool = True):
    """
    Inject aggregated hyperparameters into an ExactGPModel.

    Call this in main.py after constructing the model and before any planner.

    Parameters
    ----------
    model  : freshly constructed ExactGPModel (in eval mode is fine)
    hp     : dict returned by joblib.load("hyperparams.pkl")["aggregated"]
    freeze : if True, disable gradient on all parameters (recommended)
    """
    with torch.no_grad():
        model.mean_module.constant.fill_(hp["mean_constant"])
        model.covar_module.outputscale = hp["outputscale"]
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [[hp["lengthscale_0"], hp["lengthscale_1"]]]
        )
        model.likelihood.noise_covar.noise = hp["noise"]

    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    domain_size     = tuple(args.domain_size)
    source_range    = tuple(args.source_range)
    intensity_range = tuple(args.intensity_range)

    # --- scenarios ---
    if args.scenarios_file and os.path.exists(args.scenarios_file):
        scenarios = joblib.load(args.scenarios_file)
        print(f"Loaded {len(scenarios)} scenarios from {args.scenarios_file}")
    else:
        scenarios = [
            generate_random_scenario(source_range, domain_size, intensity_range)
            for _ in range(args.n_scenarios)
        ]
        print(f"Generated {len(scenarios)} random scenarios")

    grid = make_obs_grid(domain_size, args.n_obs_per_scenario, args.sigma)
    print(f"Observation grid: {grid.shape[0]} points per scenario\n")

    # --- optimise per scenario ---
    records = []
    for i, scenario in enumerate(scenarios):
        try:
            hp = fit_scenario(scenario, grid, args.sigma, args.n_iter)
            records.append(hp)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:>4d}/{len(scenarios)}]  "
                      f"outputscale={hp['outputscale']:.3f}  "
                      f"ls=({hp['lengthscale_0']:.3f}, {hp['lengthscale_1']:.3f})  "
                      f"mean={hp['mean_constant']:.3f}")
        except Exception as e:
            print(f"  [{i+1:>4d}] FAILED: {e}")

    if not records:
        raise RuntimeError("All scenario fits failed — check your environment setup.")

    # --- aggregate ---
    aggregated = aggregate(records, method="median")

    print("\n── Aggregated hyperparameters (median across scenarios) ──")
    for k, v in aggregated.items():
        vals = [r[k] for r in records]
        print(f"  {k:<20s}  median={v:.4f}   "
              f"[{np.min(vals):.4f}, {np.max(vals):.4f}]  "
              f"std={np.std(vals):.4f}")

    # --- save ---
    output = {
        "aggregated":   aggregated,
        "per_scenario": records,
        "config": {
            "domain_size":           domain_size,
            "source_range":          source_range,
            "intensity_range":       intensity_range,
            "sigma":                 args.sigma,
            "n_scenarios":           len(scenarios),
            "n_obs_per_scenario":    args.n_obs_per_scenario,
            "n_iter":                args.n_iter,
            "aggregation_method":    "median",
        },
    }
    joblib.dump(output, args.output)
    print(f"\nSaved → {args.output}")
    print(
        "\nTo use in main.py, after constructing ExactGPModel call:\n"
        "  from tune_hyperparams import apply_hyperparams\n"
        "  hp = joblib.load('hyperparams.pkl')['aggregated']\n"
        "  apply_hyperparams(GPModel, hp, freeze=True)"
    )


if __name__ == "__main__":
    main()
