"""
model.py
Denoising backbone for the AUV trajectory diffusion planner.

AUVTemporalUnet is a lightly adapted version of TemporalUnet (Janner et al.,
2022).  The two changes relative to the original are:

  1. The scalar-return conditioning branch is replaced by a combined
     conditioning embedding  φ_t = [time_embed ‖ belief_embed ‖ return_embed]
     where belief_embed comes from a CNN (BeliefEncoder) and return_embed is
     an MLP over the scalar reward r.  Both conditioning signals independently
     support CFG dropout via separate Bernoulli masks.

  2. action_dim is always 0 for the AUV problem – the trajectory τ is a pure
     state sequence of shape (B, k+1, 2).

Conditioning embedding layout
------------------------------
  time_embed    : (B, dim)         – sinusoidal + MLP
  belief_embed  : (B, dim)         – BeliefEncoder output (CNN)
  return_embed  : (B, dim)         – MLP over scalar r
  ──────────────────────────────────
  embed_dim     : 3 × dim          – concatenated, fed to every ResBlock
"""

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from diffusion.helpers import (
    SinusoidalPosEmb,
    BeliefEncoder,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    crop_belief_map,
)


# ---------------------------------------------------------------------------
# Core residual block (unchanged from original)
# ---------------------------------------------------------------------------

class ResidualTemporalBlock(nn.Module):
    """
    Two Conv1dBlocks with a time (+ belief + return) embedding injected
    additively after the first block, plus a residual projection.

    x   : (B, inp_channels, horizon)
    t   : (B, embed_dim)
    out : (B, out_channels, horizon)
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        horizon: int,
        kernel_size: int = 5,
        mish: bool = True,
    ):
        super().__init__()
        act_fn = nn.Mish() if mish else nn.SiLU()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        # project the full conditioning embedding to out_channels
        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# ---------------------------------------------------------------------------
# Main U-Net backbone
# ---------------------------------------------------------------------------

class TemporalUnet(nn.Module):
    """
    Temporal U-Net denoising backbone conditioned on GP belief and scalar
    return via Classifier-Free Guidance (CFG).

    Parameters
    ----------
    horizon : int
        Number of waypoints in the trajectory (k+1).
    transition_dim : int
        Dimensionality of each waypoint, always 2 (x, y) for AUV.
    belief_dim : int
        Flattened GP belief map length (1600 for a 40×40 grid).
    dim : int
        Base channel width.  All dim_mults scale from this.
    dim_mults : tuple
        U-Net depth multipliers.  (1,2,4,8) → 4 resolution levels.
    condition_dropout : float
        Bernoulli drop probability applied independently to belief_embed
        and return_embed during training to enable CFG.
    kernel_size : int
        Conv1d kernel size in every ResBlock.
    """

    def __init__(
        self,
        horizon: int,
        transition_dim: int = 2,
        belief_dim: int = 1600,
        dim: int = 128,
        dim_mults: tuple = (1, 2, 4, 8),
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        crop_size: int = 40,
        belief_encoder_pooling: bool = False,
    ):
        super().__init__()

        # ── channel layout ──────────────────────────────────────────────────
        dims   = [transition_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        # embed_dim = time(dim) + belief(dim) + return(dim)
        embed_dim = 3 * dim
        self.embed_dim        = embed_dim
        self.condition_dropout = condition_dropout

        # ── diffusion timestep embedding ─────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # ── belief embedding (CNN) ───────────────────────────────────────────
        self.belief_encoder = BeliefEncoder(out_dim=dim, grid_size=crop_size, pooling=belief_encoder_pooling)
        self.belief_mask_dist = Bernoulli(probs=1.0 - condition_dropout)

        # ── scalar return embedding (MLP) ────────────────────────────────────
        self.returns_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.Mish(),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.return_mask_dist = Bernoulli(probs=1.0 - condition_dropout)

        # ── U-Net encoder (down) ─────────────────────────────────────────────
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        h = horizon
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in,  dim_out, embed_dim, h, kernel_size),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim, h, kernel_size),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                h = h // 2

        # ── bottleneck ───────────────────────────────────────────────────────
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, h, kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, h, kernel_size)

        # ── U-Net decoder (up) ───────────────────────────────────────────────
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                # skip-connection doubles channels on input
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim, h, kernel_size),
                ResidualTemporalBlock(dim_in,      dim_in, embed_dim, h, kernel_size),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                h = h * 2

        # ── output projection ────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

    # ── conditioning embedding builder ──────────────────────────────────────

    def _build_embedding(
        self,
        time:                 torch.Tensor,   # (B,)
        b_mean:               torch.Tensor,   # (B, 1600)
        b_var:                torch.Tensor,   # (B, 1600)
        returns:              torch.Tensor,   # (B, 1)
        use_dropout:          bool,
        force_dropout:        bool,
        use_belief:           bool = True,
        use_return:           bool = True,
        force_belief_dropout: bool = False,
        force_return_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Returns the full conditioning vector (B, 3·dim) formed by:
            [time_embed | belief_embed | return_embed]
        During training, belief_embed and return_embed are each independently
        zeroed with probability `condition_dropout` (CFG training).
        During inference with force_dropout=True both are zeroed to obtain the
        unconditional noise estimate required for CFG combination.
        force_belief_dropout / force_return_dropout zero each signal independently,
        used by the 3-pass composed CFG to isolate per-signal guidance directions.
        """
        time_embed   = self.time_mlp(time)                             # (B, dim)
        belief_embed = self.belief_encoder(b_mean, b_var)              # (B, dim)
        return_embed = self.returns_mlp(returns)                       # (B, dim)

        # ablation: zero out belief and return embeddings independently to test their contribution to performance
        if not use_belief:
            belief_embed = torch.zeros_like(belief_embed)
        if not use_return:
            return_embed = torch.zeros_like(return_embed)

        if use_dropout:
            B = time_embed.shape[0]
            b_mask = self.belief_mask_dist.sample((B, 1)).to(belief_embed.device)
            r_mask = self.return_mask_dist.sample((B, 1)).to(return_embed.device)
            belief_embed = b_mask * belief_embed
            return_embed = r_mask * return_embed

        if force_dropout or force_belief_dropout:
            belief_embed = torch.zeros_like(belief_embed)
        if force_dropout or force_return_dropout:
            return_embed = torch.zeros_like(return_embed)

        return torch.cat([time_embed, belief_embed, return_embed], dim=-1)  # (B, 3·dim)

    # ── forward pass ────────────────────────────────────────────────────────

    def forward(
        self,
        x:                    torch.Tensor,     # (B, horizon, 2)
        cond:                 dict,             # {0: (B, 2)}  – start position
        time:                 torch.Tensor,     # (B,)
        b_mean:               torch.Tensor,     # (B, 1600)
        b_var:                torch.Tensor,     # (B, 1600)
        returns:              torch.Tensor,     # (B, 1)
        use_dropout:          bool = True,
        force_dropout:        bool = False,
        use_belief:           bool = True,
        use_return:           bool = True,
        force_belief_dropout: bool = False,
        force_return_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Returns the predicted noise (or x0) with shape (B, horizon, 2).
        """
        # (B, horizon, 2) → (B, 2, horizon)  – conv expects channels first
        x = einops.rearrange(x, "b h t -> b t h")

        t = self._build_embedding(time, b_mean, b_var, returns,
                                  use_dropout, force_dropout, use_belief, use_return,
                                  force_belief_dropout, force_return_dropout)

        # ── encoder ─────────────────────────────────────────────────────────
        skips = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            skips.append(x)
            x = downsample(x)
        

        # ── bottleneck ───────────────────────────────────────────────────────
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # ── decoder ──────────────────────────────────────────────────────────
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, 2, horizon) → (B, horizon, 2)
        return einops.rearrange(x, "b t h -> b h t")