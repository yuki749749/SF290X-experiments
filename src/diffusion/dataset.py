import os
import json
import joblib
import numpy as np
from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import Dataset

from diffusion.helpers import crop_belief_map


class TrajectoryDataset(Dataset):
    """
    Loads raw trajectory sequences and extracts windows in-memory on the fly.

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
        horizon: int = 16,
        stride: int = 4,
        reward_key: str = "rmse_history",
        reward_type: str = "rmse",
        crop_size: int = 40,
        domain_min: Optional[Sequence[float]] = None,
        domain_max: Optional[Sequence[float]] = None,
        grid_size: int = 40,
        max_step: float = 10.0,
        initial_heading: float = np.pi / 4,
    ):
        self.raw_trajectories = joblib.load(path)
        self.horizon = horizon
        self.stride = stride
        self.reward_key = reward_key
        self.reward_type = reward_type
        self.max_step = max_step
        self.initial_heading = initial_heading

        self.domain_size = torch.tensor(domain_size, dtype=torch.float32)
        self.domain_pad = torch.tensor(domain_pad, dtype=torch.float32)
        self.crop_size = crop_size
        self.grid_size = grid_size
        _dmin = list(domain_min) if domain_min is not None else [0.0, 0.0]
        _dmax = list(domain_max) if domain_max is not None else list(domain_size[:2])
        self.domain_min = torch.tensor(_dmin, dtype=torch.float32)
        self.domain_max = torch.tensor(_dmax, dtype=torch.float32)

        # Pre-compute valid window starting steps for all trajectories
        self.window_lookup = []
        for traj_idx, traj in enumerate(self.raw_trajectories):
            T = len(traj["positions"]) - 1  # Last index of valid step
            for t in range(1, T - self.horizon + 2, self.stride):
                self.window_lookup.append((traj_idx, t))

        # Load normalization stats from training stats file in the same directory
        training_stats_path = os.path.join(os.path.dirname(path), "training_data_stats.json")
        if os.path.exists(training_stats_path):
            try:
                with open(training_stats_path, "r") as f:
                    self.stats = json.load(f)
                self.b_mean_mean = torch.tensor(self.stats.get("b_mean_global_mean", 0.0), dtype=torch.float32)
                self.b_mean_std  = torch.tensor(self.stats.get("b_mean_global_std", 1.0), dtype=torch.float32)
                self.b_var_mean  = torch.tensor(self.stats.get("b_var_global_mean", 0.0), dtype=torch.float32)
                self.b_var_std   = torch.tensor(self.stats.get("b_var_global_std", 1.0), dtype=torch.float32)
                self.normalize_beliefs = True
            except Exception as exc:
                print(f"[WARN] Error loading training stats from {training_stats_path}: {exc}")
                self.normalize_beliefs = False
        else:
            self.normalize_beliefs = False

    # ── normalisation ────────────────────────────────────────────────────────

    def normalise_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """(k+1, 2) in [0, domain_size] → [-1, 1]."""
        return (tau + self.domain_pad) / (self.domain_size + 2 * self.domain_pad) * 2.0 - 1.0

    def denormalise_tau(self, tau_norm: torch.Tensor) -> torch.Tensor:
        """Inverse. Works on any batch shape (..., 2)."""
        return (tau_norm + 1.0) / 2.0 * (self.domain_size + 2 * self.domain_pad) - self.domain_pad

    # ── dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.window_lookup)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, t = self.window_lookup[idx]
        traj = self.raw_trajectories[traj_idx]

        # Construct positions with the synthetic ghost prepend (matches inference)
        ghost = [-self.max_step * np.cos(self.initial_heading), -self.max_step * np.sin(self.initial_heading)]
        positions = [ghost] + list(traj["positions"])
        
        # Slice tau for the window [t-1 : t+H-1]
        tau_np = np.asarray(positions[t - 1 : t + self.horizon - 1], dtype=np.float32)
        tau = torch.from_numpy(tau_np).float()
        tau = self.normalise_tau(tau)

        # Compute reward on-the-fly
        # Fall back to "rmse_history" if reward_key is not in traj
        rewards = traj[self.reward_key] if self.reward_key in traj else traj["rmse_history"]
        if self.reward_type == "rmse":
            r_val = (rewards[t - 1] - rewards[t + self.horizon - 3]) / (rewards[t - 1] + 1e-8)
        else:  # trace_reduction
            r_val = (rewards[t + self.horizon - 3] - rewards[t - 1]) / (1.0 - rewards[t - 1] + 1e-8)
            
        r = torch.tensor([r_val], dtype=torch.float32)

        # Get belief states at step t (index t-1 in raw lists)
        b_mean = torch.from_numpy(traj["means"][t - 1]).float()
        b_var  = torch.from_numpy(traj["variances"][t - 1]).float()

        # Crop if necessary (local belief crop around starting position)
        if self.crop_size < self.grid_size:
            agent_pos = self.denormalise_tau(tau[0])  # (2,) physical coords
            b_mean, b_var = crop_belief_map(
                b_mean, b_var, agent_pos,
                self.domain_min, self.domain_max,
                self.crop_size, self.grid_size,
            )

        # Apply normalization if stats were loaded successfully
        if self.normalize_beliefs:
            b_mean = (b_mean - self.b_mean_mean) / (self.b_mean_std + 1e-8)
            b_var  = (b_var - self.b_var_mean) / (self.b_var_std + 1e-8)

        return {"tau": tau, "r": r, "b_mean": b_mean, "b_var": b_var}