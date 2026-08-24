"""
helpers.py
Low-level building blocks for the AUV diffusion planner.

Kept API-compatible with the original Janner et al. (2022) helpers so that
GaussianDiffusion and TemporalUnet can be swapped in with minimal changes.
New addition: BeliefEncoder – a CNN that maps the GP belief maps
(mean + variance) to a fixed-length embedding consumed by TemporalUnet.
New addition: crop_belief_map – extracts an agent-centric patch from the
full belief map so the encoder sees the world in agent-relative coordinates.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import einops 
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Positional / temporal embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal embedding for diffusion timestep t.
    Output shape: (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]                      # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)      # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# 1-D U-Net building blocks (unchanged from original)
# ---------------------------------------------------------------------------

class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish/SiLU"""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        mish: bool = True,
        n_groups: int = 8,
    ):
        super().__init__()
        act_fn = nn.Mish() if mish else nn.SiLU()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            act_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Agent-centric belief crop
# ---------------------------------------------------------------------------

def crop_belief_map(
    b_mean: torch.Tensor,       # (grid_size²,) flattened
    b_var: torch.Tensor,        # (grid_size²,) flattened
    boundary_map: torch.Tensor, # (grid_size²,) flattened
    agent_pos: torch.Tensor,    # (2,) physical coordinates
    domain_min: torch.Tensor,   # (2,)
    domain_max: torch.Tensor,   # (2,)
    crop_size: int,
    grid_size: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a crop_size×crop_size patch from the belief maps and boundary map
    centred on the agent's grid cell. Pads where the crop extends outside the
    grid boundary (mean with 0.0, var with 1.0 (GP prior variance), boundary with 1.0).

    Returns flattened (crop_size²,) tensors on the same device as b_mean.
    """
    norm = (agent_pos - domain_min) / (domain_max - domain_min)  # [0, 1]
    ci = int(round(norm[0].item() * (grid_size - 1)))
    cj = int(round(norm[1].item() * (grid_size - 1)))

    half = crop_size // 2
    src_i0, src_i1 = ci - half, ci - half + crop_size
    src_j0, src_j1 = cj - half, cj - half + crop_size

    # Clamp source indices to grid limits for slice boundaries
    si0 = max(0, min(grid_size, src_i0))
    si1 = max(0, min(grid_size, src_i1))
    sj0 = max(0, min(grid_size, src_j0))
    sj1 = max(0, min(grid_size, src_j1))

    # Compute destination coordinates inside the crop patch
    dst_i0 = max(0, -src_i0)
    dst_i1 = dst_i0 + (si1 - si0)
    dst_j0 = max(0, -src_j0)
    dst_j1 = dst_j0 + (sj1 - sj0)

    mean_map = b_mean.view(grid_size, grid_size)
    var_map  = b_var.view(grid_size, grid_size)
    bound_map = boundary_map.view(grid_size, grid_size)

    mean_crop = torch.zeros(crop_size, crop_size, dtype=b_mean.dtype, device=b_mean.device)
    # Pad out-of-bounds variance with 1.0 (prior variance)
    var_crop  = torch.ones(crop_size, crop_size, dtype=b_var.dtype,  device=b_var.device)
    # Pad out-of-bounds boundary map with 1.0 (outside domain indicator)
    bound_crop = torch.ones(crop_size, crop_size, dtype=boundary_map.dtype, device=boundary_map.device)

    # Apply copy only if there is a valid overlapping region
    if si1 > si0 and sj1 > sj0:
        mean_crop[dst_i0:dst_i1, dst_j0:dst_j1] = mean_map[si0:si1, sj0:sj1]
        var_crop[dst_i0:dst_i1,  dst_j0:dst_j1] = var_map[si0:si1,  sj0:sj1]
        bound_crop[dst_i0:dst_i1, dst_j0:dst_j1] = bound_map[si0:si1, sj0:sj1]

    return mean_crop.flatten(), var_crop.flatten(), bound_crop.flatten()


# ---------------------------------------------------------------------------
# CNN Belief Encoder
# ---------------------------------------------------------------------------

class BeliefEncoder(nn.Module):
    """
    Encodes the GP belief state (mean map + variance map + boundary mask) into a fixed-size
    embedding vector that is later concatenated with the diffusion timestep
    embedding inside TemporalUnet.

    Input
    -----
    b_mean  : (B, grid_size²)  – flattened GP posterior mean
    b_var   : (B, grid_size²)  – flattened GP posterior variance
    b_bound : (B, grid_size²)  – flattened boundary mask

    The three maps are stacked as a 3-channel grid_size×grid_size image and
    processed by a lightweight CNN, then projected to `out_dim`.
    AdaptiveAvgPool2d(1) makes the CNN agnostic to the spatial input size,
    so the same architecture works for the full 40×40 map or any crop.
    """

    def __init__(self, out_dim: int = 128, grid_size: int = 40, pooling: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.pooling = pooling

        def _block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.Mish(),
            )

        self.cnn = nn.Sequential(
            _block(3,   16, stride=1),  # changed to 3 input channels
            _block(16,  32, stride=2),
            _block(32,  64, stride=2),
            _block(64, 128, stride=2),
        )

        if pooling:
            self.pool = nn.AdaptiveAvgPool2d(1)
            proj_in = 128
        else:
            self.pool = None
            with torch.no_grad():
                _dummy = torch.zeros(1, 3, grid_size, grid_size)
                proj_in = self.cnn(_dummy).flatten(1).shape[1]

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(proj_in, out_dim),
            nn.Mish(),
        )

    def forward(self, b_mean: torch.Tensor, b_var: torch.Tensor, b_bound: torch.Tensor) -> torch.Tensor:
        B = b_mean.shape[0]
        mean_map = b_mean.view(B, 1, self.grid_size, self.grid_size)
        var_map  = b_var.view( B, 1, self.grid_size, self.grid_size)
        bound_map = b_bound.view(B, 1, self.grid_size, self.grid_size)
        x = torch.cat([mean_map, var_map, bound_map], dim=1)
        x = self.cnn(x)
        if self.pool is not None:
            x = self.pool(x)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Diffusion schedule helpers (unchanged from original)
# ---------------------------------------------------------------------------

def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, dtype=torch.float32
) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal (2021).
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def extract(
    a: torch.Tensor, t: torch.Tensor, x_shape: tuple
) -> torch.Tensor:
    """
    Gather schedule coefficients at indices t and broadcast to x_shape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def apply_conditioning(
    x: torch.Tensor, conditions: dict, action_dim: int
) -> torch.Tensor:
    """
    Pin observation dimensions at specific horizon steps.
    conditions : { t_step : (B, obs_dim) }
    For AUV use action_dim=0 (trajectory is pure state).
    """
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


# ---------------------------------------------------------------------------
# Loss modules (unchanged from original, unused ones pruned for clarity)
# ---------------------------------------------------------------------------

class WeightedLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, action_dim: int):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        """pred, targ : (B, horizon, transition_dim)"""
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        if self.action_dim > 0:
            a0_loss = (
                loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
            ).mean()
        else:
            a0_loss = torch.tensor(0.0, device=pred.device)
        return weighted_loss, {"a0_loss": a0_loss}


class WeightedStateLoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {"a0_loss": weighted_loss}


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class WeightedStateL2(WeightedStateLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1":       WeightedL1,
    "l2":       WeightedL2,
    "state_l2": WeightedStateL2,
}


# ---------------------------------------------------------------------------
# Diffusion schedule helpers (unchanged from original)
# ---------------------------------------------------------------------------