import logging
import os

# from xml.parsers.expat import model
import gpytorch
import torch

log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time

from environment.environment import plume, generateRandomScenario
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


# scenarios = [[((5,5),100)],
#              [((3,3),100), ((8,8),150)]]

# scenarios = [[((5,5),100)]]

scenarios = joblib.load("data/scenarios/test_scenarios_025.pkl")
# scenarios = [generateRandomScenario(sourceRange=(1, 3), domainSize=(10, 10), intensityRange=(50, 200)) for _ in range(25)]


def plotSimulationResults(
    planner, scenario, positionHistory, meanHistory, varianceHistory, grid, nGrid
):
    # Select specific timesteps to show (e.g., start, middle, end)
    stepsToPlot = [0, len(positionHistory) // 2, len(positionHistory) - 1]
    numSteps = len(stepsToPlot)

    fig, axes = plt.subplots(numSteps, 3, figsize=(15, 5 * numSteps))

    # 1. Pre-calculate Ground Truth for the grid
    groundTruthGrid = np.array([plume(scenario, p) for p in grid.numpy()]).reshape(
        nGrid, nGrid
    )

    # 2. Determine global min/max for consistent colorbars
    v_min, v_max = groundTruthGrid.min(), groundTruthGrid.max()
    var_max = max([var.max().item() for var in varianceHistory])

    for row, t in enumerate(stepsToPlot):
        # Extract data for current timestep
        pos = np.array(positionHistory[: t + 1])  # Path taken up to time t
        mean = meanHistory[t].reshape(nGrid, nGrid).numpy()
        var = varianceHistory[t].reshape(nGrid, nGrid).numpy()

        # Column 1: Ground Truth + Path
        im0 = axes[row, 0].imshow(
            groundTruthGrid.T,
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

        path = np.array(positionHistory[: t + 1])
        for col in range(3):
            axes[row, col].plot(
                path[:, 0], path[:, 1], "r.-", markersize=5, label="Path"
            )

    plannerName = planner.__class__.__name__
    outputDir = f"plots/scenario_{scenarios.index(scenario):04d}"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputPath = f"{outputDir}/{plannerName}.png"
    plt.savefig(outputPath)
    plt.close()


# AUV state
currentHeading = np.pi / 4
currentPosition = (0, 0)
currentBelief = None
maxStep = 1.0
maxTurn = np.pi / 8


domainSize = (10, 10)
sigma = 0.5  # measurement noise standard deviation

gp_hyperparams_path = "data/hyperparameters/gp_hyperparameters.pkl"
gp_hyperparams = joblib.load(gp_hyperparams_path)
# betaHyperparametersPath = "data/hyperparameters/betaTuningResults.pkl"


# planners = [BayesianOptimizationPlanner(domainSize,
#                                         maxStep,
#                                         maxTurn,
#                                         boundaryBehavior='clamp',
#                                         acquisitionFunction='ucb',
#                                         beta=10.0,
#                                         numSteps=20,
#                                         numTurns=11),
#             RandomPlanner(domainSize,
#                           maxStep,
#                           maxTurn,
#                           boundaryBehavior='clamp'),
#             LawnmowerPlanner(domainSize,
#                             maxStep,
#                             maxTurn=np.pi,
#                             laneSpacing=1.0,
#                             orientation='horizontal',
#                             boundaryBehavior='clamp')]

# planners = [BayesianOptimizationPlanner(domainSize,
#                                         maxStep,
#                                         maxTurn,
#                                         boundaryBehavior='clamp',
#                                         acquisitionFunction='ucb',
#                                         beta=10.0,
#                                         numSteps=20,
#                                         numTurns=11)]
# planners = [RandomPlanner(domainSize, maxStep, maxTurn)]
# planners = [LawnmowerPlanner(domainSize, maxStep, maxTurn=np.pi)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 16  # must match training
REPLAN_EVERY = 15  # replan after 5 executed waypoints
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
        domain_size=domainSize,
        horizon=HORIZON,
        replan_every=REPLAN_EVERY,
        target_return=TARGET_RETURN,
        device=DEVICE,
    )
]


def main():
    print("Starting simulation...")
    nGrid = 40
    t1 = torch.linspace(0, domainSize[0], nGrid)
    t2 = torch.linspace(0, domainSize[1], nGrid)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    testGrid = torch.stack([g1.flatten(), g2.flatten()], dim=-1)

    for planner in planners:
        print(f"Running planner: {planner}")
        for scenario in scenarios:
            print(f"Running scenario: {scenario}")
            # start = time.time()
            planner.reset()
            groundTruth_np = np.array([plume(scenario, p) for p in testGrid.numpy()])
            groundTruth = torch.tensor(groundTruth_np, dtype=torch.float32)

            logger = Logger(
                outputDirectory=f"results/evaluation/scenario_{scenarios.index(scenario):04d}",
                groundTruth=groundTruth,
            )
            currentHeading = np.pi / 4
            currentPosition = (0, 0)

            measurement = plume(scenario, currentPosition) + np.random.normal(0, sigma)
            currentPositionTensor = torch.tensor(
                currentPosition, dtype=torch.float32
            ).unsqueeze(0)
            measurementTensor = torch.tensor([measurement], dtype=torch.float32)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            GPModel = ExactGPModel(currentPositionTensor, measurementTensor, likelihood)
            if os.path.exists(gp_hyperparams_path):
                with torch.no_grad():
                    GPModel.mean_module.constant = torch.tensor(
                        gp_hyperparams["mean_constant"]
                    )
                    GPModel.covar_module.base_kernel.lengthscale = torch.tensor(
                        [
                            gp_hyperparams["lengthscale_0"],
                            gp_hyperparams["lengthscale_1"],
                        ]
                    )
                    GPModel.covar_module.outputscale = torch.tensor(
                        gp_hyperparams["outputscale"]
                    )
                    GPModel.likelihood.noise = torch.tensor(gp_hyperparams["noise"])
                for param in GPModel.parameters():
                    param.requires_grad_(False)
                # GPModel.likelihood.noise_covar.noise = sigma ** 2
                # GPModel.likelihood.noise_covar.raw_noise.requires_grad_(False)
            else:
                GPModel.likelihood.noise_covar.noise = sigma**2
                GPModel.likelihood.noise_covar.raw_noise.requires_grad_(False)
            # for name, param in GPModel.named_parameters():
            #     print(name, param.data)
            # print(f"mean constant:\t{GPModel.mean_module.constant.item():.3f}")
            # print(f"output scale:\t{GPModel.covar_module.outputscale.item():.3f}")
            # for i in range(2):
            #     print(f"length scale {i}:\t{GPModel.covar_module.base_kernel.lengthscale[0, i].item():.3f}")
            # print(f"noise:\t\t{GPModel.likelihood.noise.item():.3f}")

            currentBelief = Belief(GPModel, testGrid)

            logger.logStep(currentPosition, currentBelief)
            # t = time.time() - start
            # print(f"Initialisation time: {t:.4f} seconds")
            print("Initial RMSE:", logger.rmseHistory[-1])
            # print("Initial NLPD:", logger.nlpdHistory[-1])
            print(
                "Initial normalized trace reduction:",
                logger.normalizedTraceReductionHistory[-1],
            )

            for t in range(1, 50):
                print(t)
                # t = time.time() - start
                # print(f"Time at step {t}: {t:.4f} seconds")
                newPosition, newHeading = planner.computeNextPose(
                    currentPosition, currentHeading, currentBelief
                )
                # t = time.time() - start
                # print(f"Time after computing next pose: {t:.4f} seconds")
                currentPosition = newPosition
                currentHeading = newHeading
                measurement = plume(scenario, currentPosition) + np.random.normal(
                    0, sigma
                )

                currentPositionTensor = torch.tensor(
                    currentPosition, dtype=torch.float32
                ).unsqueeze(0)
                measurementTensor = torch.tensor([measurement], dtype=torch.float32)

                currentBelief.update(currentPositionTensor, measurementTensor)
                # t = time.time() - start
                # print(f"Time after belief update: {t:.4f} seconds")

                logger.logStep(currentPosition, currentBelief)
                # t = time.time() - start
                # print(f"Time after logging step: {t:.4f} seconds")

            print("Final RMSE:", logger.rmseHistory[-1])
            # print("Final NLPD:", logger.nlpdHistory[-1])
            print(
                "Final normalized trace reduction:",
                logger.normalizedTraceReductionHistory[-1],
            )

            plotSimulationResults(
                planner,
                scenario,
                logger.positionHistory,
                logger.meanHistory,
                logger.varianceHistory,
                testGrid,
                nGrid,
            )
            logger.saveHistory(filename=f"{planner.__class__.__name__}History.pkl")

        print(f"Finished planner: {planner}")


if __name__ == "__main__":
    main()
