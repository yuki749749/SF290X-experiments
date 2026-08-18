from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
import os

def abs_path(relative: str) -> str:
    return os.path.join(get_original_cwd(), relative)

def get_output_dir():
    """Return Hydra's resolved output directory for the current run."""
    return HydraConfig.get().runtime.output_dir