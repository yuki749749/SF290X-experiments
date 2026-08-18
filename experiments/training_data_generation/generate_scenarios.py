import os
import sys
import joblib
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig

# Add src folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from environment.environment import generateRandomScenario
from utils import abs_path

log = logging.getLogger(__name__)

@hydra.main(config_path="../../config", config_name="generate_scenarios", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("Generating scenarios...")
    
    # 1. GP tuning scenarios
    gp_tuning_path = abs_path(cfg.paths.scenarios.gp_tuning)
    os.makedirs(os.path.dirname(gp_tuning_path), exist_ok=True)
    gp_scenarios = [
        generateRandomScenario(
            sourceRange=tuple(cfg.source_range),
            domainSize=tuple(cfg.domain_size),
            intensityRange=tuple(cfg.intensity_range)
        )
        for _ in range(cfg.gp_tuning_scenarios)
    ]
    joblib.dump(gp_scenarios, gp_tuning_path)
    log.info(f"Saved {cfg.gp_tuning_scenarios} scenarios to {gp_tuning_path}")

    # 2. Beta tuning scenarios
    beta_tuning_path = abs_path(cfg.paths.scenarios.beta_tuning)
    os.makedirs(os.path.dirname(beta_tuning_path), exist_ok=True)
    beta_scenarios = [
        generateRandomScenario(
            sourceRange=tuple(cfg.source_range),
            domainSize=tuple(cfg.domain_size),
            intensityRange=tuple(cfg.intensity_range)
        )
        for _ in range(cfg.beta_tuning_scenarios)
    ]
    joblib.dump(beta_scenarios, beta_tuning_path)
    log.info(f"Saved {cfg.beta_tuning_scenarios} scenarios to {beta_tuning_path}")

    # 3. Training generation scenarios
    generation_path = abs_path(cfg.paths.scenarios.generation)
    os.makedirs(os.path.dirname(generation_path), exist_ok=True)
    gen_scenarios = [
        generateRandomScenario(
            sourceRange=tuple(cfg.source_range),
            domainSize=tuple(cfg.domain_size),
            intensityRange=tuple(cfg.intensity_range)
        )
        for _ in range(cfg.generation_scenarios)
    ]
    joblib.dump(gen_scenarios, generation_path)
    log.info(f"Saved {cfg.generation_scenarios} scenarios to {generation_path}")

    # 4. Evaluation scenarios
    evaluation_path = abs_path(cfg.paths.scenarios.evaluation)
    os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)
    eval_scenarios = [
        generateRandomScenario(
            sourceRange=tuple(cfg.source_range),
            domainSize=tuple(cfg.domain_size),
            intensityRange=tuple(cfg.intensity_range)
        )
        for _ in range(cfg.evaluation_scenarios)
    ]
    joblib.dump(eval_scenarios, evaluation_path)
    log.info(f"Saved {cfg.evaluation_scenarios} scenarios to {evaluation_path}")

    log.info("Scenario generation completed successfully!")

if __name__ == "__main__":
    main()
