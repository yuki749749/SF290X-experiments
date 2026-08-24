import os
import json
import joblib
import math
import numpy as np
from typing import Dict, Optional, Sequence
import torch
import torch.nn.functional as F
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
        horizon: Optional[int] = None,
        stride: Optional[int] = None,
        reward_key: Optional[str] = None,
        reward_type: Optional[str] = None,
        crop_size: int = 40,
        domain_min: Optional[Sequence[float]] = None,
        domain_max: Optional[Sequence[float]] = None,
        grid_size: int = 40,
        max_step: Optional[float] = None,
        initial_heading: Optional[float] = None,
        use_egocentric: bool = False,
    ):
        self.raw_trajectories = joblib.load(path)

        # Load normalization stats from training stats file in the same directory
        self.stats = {}
        training_stats_path = os.path.join(os.path.dirname(path), "training_data_stats.json")
        if os.path.exists(training_stats_path):
            try:
                with open(training_stats_path, "r") as f:
                    self.stats = json.load(f)
            except Exception as exc:
                print(f"[WARN] Error loading training stats from {training_stats_path}: {exc}")

        # Resolve parameters from stats if not explicitly passed, falling back to defaults
        self.horizon = self.stats.get("horizon", horizon)
        if self.horizon is None:
            self.horizon = 16
        self.stride = self.stats.get("stride", stride)
        if self.stride is None:
            self.stride = 4
        self.reward_key = self.stats.get("reward_key", reward_key)
        if self.reward_key is None:
            self.reward_key = "rmse_history"
        self.reward_type = self.stats.get("reward_type", reward_type)
        if self.reward_type is None:
            self.reward_type = "rmse"
        self.max_step = self.stats.get("max_step", max_step)
        if self.max_step is None:
            self.max_step = 10.0
        self.initial_heading = self.stats.get("initial_heading", initial_heading)
        if self.initial_heading is None:
            self.initial_heading = np.pi / 4

        self.use_egocentric = use_egocentric

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

        if self.stats:
            self.b_mean_mean = torch.tensor(self.stats.get("b_mean_global_mean", 0.0), dtype=torch.float32)
            self.b_mean_std  = torch.tensor(self.stats.get("b_mean_global_std", 1.0), dtype=torch.float32)
            self.b_var_mean  = torch.tensor(self.stats.get("b_var_global_mean", 0.0), dtype=torch.float32)
            self.b_var_std   = torch.tensor(self.stats.get("b_var_global_std", 1.0), dtype=torch.float32)
            
            # Select appropriate min/max based on selected reward type with backward-compatible fallbacks
            if self.reward_type == "rmse":
                self.r_min = torch.tensor(self.stats.get("rmse_r_min", self.stats.get("r_min", 0.0)), dtype=torch.float32)
                self.r_max = torch.tensor(self.stats.get("rmse_r_max", self.stats.get("r_max", 1.0)), dtype=torch.float32)
            else:  # trace_reduction
                self.r_min = torch.tensor(self.stats.get("trace_r_min", self.stats.get("r_min", 0.0)), dtype=torch.float32)
                self.r_max = torch.tensor(self.stats.get("trace_r_max", self.stats.get("r_max", 1.0)), dtype=torch.float32)
                
            self.normalize_beliefs = True
            self.normalize_reward = True
        else:
            self.normalize_beliefs = False
            self.normalize_reward = False

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
        tau_raw = torch.from_numpy(tau_np).float()

        # Get current position and heading
        p_curr = positions[t]
        p_prev = positions[t - 1]
        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]
        heading = math.atan2(dy, dx)

        if self.use_egocentric:
            p_curr_tensor = torch.tensor(p_curr, dtype=torch.float32)
            tau_trans = tau_raw - p_curr_tensor
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            rot_mat = torch.tensor([
                [cos_h, sin_h],
                [-sin_h, cos_h]
            ], dtype=torch.float32)
            tau_rot = tau_trans @ rot_mat.t()
            scale_factor = 2.0 / (self.domain_size + 2 * self.domain_pad)
            tau = tau_rot * scale_factor
        else:
            tau = self.normalise_tau(tau_raw)

        # Compute reward on-the-fly
        if self.reward_type == "rmse":
            rewards = traj[self.reward_key] if self.reward_key in traj else traj["rmse_history"]
            rmse_0 = rewards[0]
            r_val = (rewards[t - 1] - rewards[t + self.horizon - 3]) / (rmse_0 + 1e-8)
        else:  # trace_reduction
            if "trace_history" in traj:
                trace_0 = traj["trace_history"][0]
                trace_start = traj["trace_history"][t - 1]
                trace_end = traj["trace_history"][t + self.horizon - 3]
            else:
                if isinstance(traj["variances"], dict):
                    v0 = traj["variances"].get(0, np.zeros(self.grid_size * self.grid_size, dtype=np.float32))
                    vt = traj["variances"].get(t - 1, np.zeros(self.grid_size * self.grid_size, dtype=np.float32))
                    ve = traj["variances"].get(t + self.horizon - 3, np.zeros(self.grid_size * self.grid_size, dtype=np.float32))
                    trace_0 = v0.sum()
                    trace_start = vt.sum()
                    trace_end = ve.sum()
                else:
                    trace_0 = traj["variances"][0].sum()
                    trace_start = traj["variances"][t - 1].sum()
                    trace_end = traj["variances"][t + self.horizon - 3].sum()
            r_val = (trace_start - trace_end) / (trace_0 + 1e-8)
            
        r = torch.tensor([r_val], dtype=torch.float32)
        if self.normalize_reward:
            r = torch.clamp((r - self.r_min) / (self.r_max - self.r_min + 1e-8), 0.0, 1.0)

        # Get belief states at step t (index t-1 in raw lists)
        if isinstance(traj["means"], dict):
            b_mean_np = traj["means"].get(t - 1)
            if b_mean_np is None:
                b_mean_np = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
            b_mean = torch.from_numpy(b_mean_np).float()

            b_var_np = traj["variances"].get(t - 1)
            if b_var_np is None:
                b_var_np = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
            b_var = torch.from_numpy(b_var_np).float()
        else:
            b_mean = torch.from_numpy(traj["means"][t - 1]).float()
            b_var  = torch.from_numpy(traj["variances"][t - 1]).float()

        # Crop if necessary (local belief crop around starting position)
        if self.crop_size < self.grid_size or self.use_egocentric:
            if self.use_egocentric:
                agent_pos = torch.tensor(p_curr, dtype=torch.float32)
            else:
                agent_pos = self.denormalise_tau(tau[0])  # (2,) physical coords
            b_mean, b_var = crop_belief_map(
                b_mean, b_var, agent_pos,
                self.domain_min, self.domain_max,
                self.crop_size, self.grid_size,
            )

            # Rotate crop for egocentric frame
            if self.use_egocentric:
                grid_img = torch.stack([
                    b_mean.view(self.crop_size, self.crop_size),
                    b_var.view(self.crop_size, self.crop_size)
                ], dim=0) # (2, H, W)
                
                cos_a = math.cos(heading)
                sin_a = math.sin(heading)
                
                # Rotation matrix (we rotate the sampling coordinates by -heading)
                rot_mat = torch.tensor([[
                    [cos_a,  sin_a, 0.0],
                    [-sin_a, cos_a, 0.0]
                ]], dtype=torch.float32)
                
                x_batch = grid_img.unsqueeze(0)
                grid = F.affine_grid(rot_mat, x_batch.size(), align_corners=True)
                rotated = F.grid_sample(x_batch, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
                rotated = rotated.squeeze(0)
                b_mean = rotated[0].flatten()
                b_var  = rotated[1].flatten()

        # Apply normalization if stats were loaded successfully
        if self.normalize_beliefs:
            b_mean = (b_mean - self.b_mean_mean) / (self.b_mean_std + 1e-8)
            b_var  = (b_var - self.b_var_mean) / (self.b_var_std + 1e-8)

        return {"tau": tau, "r": r, "b_mean": b_mean, "b_var": b_var}