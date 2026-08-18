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
    
    scenarios = [
        generateRandomScenario(
            sourceRange=tuple(cfg.source_range),
            domainSize=tuple(cfg.domain_size),
            intensityRange=tuple(cfg.intensity_range)
        )
        for _ in range(cfg.num_scenarios)
    ]
    
    if cfg.get("output_path", None) is not None:
        target_path = abs_path(cfg.output_path)
    else:
        from utils import get_output_dir
        target_path = os.path.join(get_output_dir(), cfg.output_filename)
        
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    joblib.dump(scenarios, target_path)
    log.info(f"Saved {cfg.num_scenarios} scenarios to {target_path}")

    log.info("Scenario generation completed successfully!")

if __name__ == "__main__":
    main()
