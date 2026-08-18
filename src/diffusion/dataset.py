import joblib
from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import Dataset

from diffusion.helpers import crop_belief_map


class TrajectoryDataset(Dataset):
    """
    Wraps the joblib-serialised list of scenario dicts.

    Each item
    ---------
    tau    : (k+1, 2)         - raw AUV waypoints in grid coordinates
    r      : ()               - scalar return / reward
    b_mean : (crop_size²,)    - GP posterior mean (agent-centric crop)
    b_var  : (crop_size²,)    - GP posterior variance (agent-centric crop)

    tau is normalised to [-1, 1] using fixed domain bounds.
    When crop_size == grid_size (default 40) the full map is returned
    unchanged, preserving backwards compatibility.
    """

    def __init__(
        self,
        path: str,
        domain_size,
        domain_pad: float,
        crop_size: int = 40,
        domain_min: Optional[Sequence[float]] = None,
        domain_max: Optional[Sequence[float]] = None,
        grid_size: int = 40,
    ):
        self.data: list = joblib.load(path)
        self.domain_size = torch.tensor(domain_size, dtype=torch.float32)
        self.domain_pad = torch.tensor(domain_pad, dtype=torch.float32)
        self.crop_size = crop_size
        self.grid_size = grid_size
        _dmin = list(domain_min) if domain_min is not None else [0.0, 0.0]
        _dmax = list(domain_max) if domain_max is not None else list(domain_size[:2])
        self.domain_min = torch.tensor(_dmin, dtype=torch.float32)
        self.domain_max = torch.tensor(_dmax, dtype=torch.float32)

    # ── normalisation ────────────────────────────────────────────────────────

    def normalise_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """(k+1, 2) in [0, domain_size] → [-1, 1]."""
        return (tau + self.domain_pad) / (self.domain_size + 2 * self.domain_pad) * 2.0 - 1.0

    def denormalise_tau(self, tau_norm: torch.Tensor) -> torch.Tensor:
        """Inverse. Works on any batch shape (..., 2)."""
        return (tau_norm + 1.0) / 2.0 * (self.domain_size + 2 * self.domain_pad) - self.domain_pad
    # ── dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.data[idx]
        tau    = torch.from_numpy(s["tau"]).float()
        tau    = self.normalise_tau(tau)
        r      = torch.tensor(s["r"], dtype=torch.float32).unsqueeze(-1)
        b_mean = torch.from_numpy(s["b_mean"]).float()
        b_var  = torch.from_numpy(s["b_var"]).float()
        if self.crop_size < self.grid_size:
            agent_pos = self.denormalise_tau(tau[0])  # (2,) physical coords
            b_mean, b_var = crop_belief_map(
                b_mean, b_var, agent_pos,
                self.domain_min, self.domain_max,
                self.crop_size, self.grid_size,
            )
        return {"tau": tau, "r": r, "b_mean": b_mean, "b_var": b_var}