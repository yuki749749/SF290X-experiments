"""
diffusion_bo_planner.py
Diffusion model as a kinematic trajectory prior + BO acquisition selection.

Motivation
----------
The standard DiffusionPlanner uses CFG to steer trajectory generation via a
belief map and return scalar.  The conditioning ablation showed that both
signals *hurt* performance — the model's unconditioned samples already produce
smooth, kinematically feasible paths, and CFG distorts them.

This planner separates the two concerns cleanly:

  • Diffusion model  →  kinematic prior: generates K diverse, smooth,
                        feasible H-step trajectories from the current position.

  • BO acquisition   →  information seeking: scores each candidate by the
                        total acquisition value along all H waypoints, then
                        executes the best one.

The H-step lookahead is the key advantage over greedy single-step BO: we pick
the globally best *trajectory*, not just the best next step.

Acquisition functions
---------------------
  "variance"  :  sum σ²(x_i)  — pure exploration, best for field estimation
  "ucb"       :  sum (μ(x_i) + β·σ(x_i))  — exploration + exploitation,
                 consistent with the existing BayesianOptimizationPlanner

Interface
---------
Identical to DiffusionPlanner and BasePlanner; drop-in replacement.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from planners.planners import BasePlanner
from diffusion.helpers import crop_belief_map

log = logging.getLogger(__name__)


class DiffusionBOPlanner(BasePlanner):
    """
    Parameters
    ----------
    diffusion : GaussianDiffusion
        Trained diffusion model (any checkpoint; conditioning unused).
    domain_size : tuple[float, float]
        Physical domain extent (W, H).
    domain_pad : float
        Padding around domain used during training normalisation.
    horizon : int
        Trajectory length in waypoints (must match the trained model).
    replan_every : int
        Replan after this many executed steps (receding-horizon control).
    n_samples : int
        Number of candidate trajectories sampled per replan (K).
    acquisition : str
        "variance" or "ucb".
    beta : float
        UCB exploration weight (only used when acquisition="ucb").
    device : str | torch.device
    warm_start : bool
        Seed each diffusion run with a noised version of the previous best
        trajectory for temporal consistency.
    noise_steps : int
        Partial-noising steps for warm start (ignored when warm_start=False).
    domain_min, domain_max : list[float]
        Physical domain corners (before padding).
    grid_size : int
        Side length of the GP evaluation grid (sqrt of n_evaluations).
    max_step, max_turn : float
        Kinematic constraints (must match training data).
    stats : dict | None
        Normalisation statistics from training_data_stats.json.
        If None, no belief normalisation is applied.
    """

    def __init__(
        self,
        diffusion,
        domain_size: tuple,
        domain_pad: float,
        horizon: int,
        replan_every: int = 4,
        n_samples: int = 10,
        acquisition: str = "ucb",
        beta: float = 20.0,
        gamma: float = 1.0,
        device: str | torch.device = "cpu",
        warm_start: bool = True,
        noise_steps: int = 20,
        domain_min: Optional[Sequence[float]] = None,
        domain_max: Optional[Sequence[float]] = None,
        grid_size: int = 40,
        max_step: float = 10.0,
        max_turn: float = np.pi / 8,
        stats: Optional[dict] = None,
        crop_size: int = 24,
        acq_mask_sharpness: float = 3.0,
        boundary_penalty: float = 1e4,
    ):
        super().__init__(
            domain_size=domain_size,
            domain_pad=domain_pad,
            max_step=max_step,
            min_step=max_step,
            max_turn=max_turn,
            boundary_behavior="clamp",
        )
        self.diffusion          = diffusion
        self.horizon            = horizon
        self.replan_every       = replan_every
        self.n_samples          = n_samples
        self.acquisition        = acquisition
        self.beta               = beta
        self.gamma              = gamma
        self.device             = torch.device(device)
        self.warm_start         = warm_start
        self.noise_steps        = noise_steps
        self.grid_size          = grid_size
        self.crop_size          = crop_size
        self.acq_mask_sharpness = acq_mask_sharpness
        self.boundary_penalty   = boundary_penalty

        self.domain_size_t = torch.tensor(domain_size, dtype=torch.float32)
        self.domain_pad_t  = torch.tensor(domain_pad,  dtype=torch.float32)

        _dmin = list(domain_min) if domain_min is not None else [0.0, 0.0]
        _dmax = list(domain_max) if domain_max is not None else list(domain_size[:2])

        # Padded domain bounds (match training normalisation)
        self.domain_min = torch.tensor(
            [_dmin[0] - domain_pad, _dmin[1] - domain_pad], dtype=torch.float32
        )
        self.domain_max = torch.tensor(
            [_dmax[0] + domain_pad, _dmax[1] + domain_pad], dtype=torch.float32
        )

        # GP evaluation grid (for belief.predict calls)
        xs = torch.linspace(self.domain_min[0].item(), self.domain_max[0].item(), grid_size)
        ys = torch.linspace(self.domain_min[1].item(), self.domain_max[1].item(), grid_size)
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing="ij")
        self.planner_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)

        # Normalisation stats (optional)
        self.stats = stats
        if stats is not None:
            self.b_mean_mean = torch.tensor(stats.get("b_mean_global_mean", 0.0), dtype=torch.float32, device=self.device)
            self.b_mean_std  = torch.tensor(stats.get("b_mean_global_std",  1.0), dtype=torch.float32, device=self.device)
            self.b_var_mean  = torch.tensor(stats.get("b_var_global_mean",  0.0), dtype=torch.float32, device=self.device)
            self.b_var_std   = torch.tensor(stats.get("b_var_global_std",   1.0), dtype=torch.float32, device=self.device)
            self.normalize_beliefs = True
        else:
            self.normalize_beliefs = False

        self.diffusion.eval()

        # Internal state
        self._waypoint_buffer: list[tuple[float, float]] = []
        self._steps_since_replan: int = 0
        self._last_position: np.ndarray | None = None
        self._last_tau_norm: torch.Tensor | None = None   # best trajectory from previous replan
        self._is_steering_back: bool = False

    # ── normalisation ─────────────────────────────────────────────────────────

    def _normalise(self, pos_np: np.ndarray) -> torch.Tensor:
        """(2,) numpy → (2,) tensor in [-1, 1]."""
        t = torch.tensor(pos_np, dtype=torch.float32)
        return (t + self.domain_pad_t) / (self.domain_size_t + 2 * self.domain_pad_t) * 2.0 - 1.0

    def _denormalise(self, tau_norm: torch.Tensor) -> np.ndarray:
        """(..., 2) normalised tensor → numpy in physical coordinates."""
        domain_size_t = self.domain_size_t.to(tau_norm.device)
        domain_pad_t  = self.domain_pad_t.to(tau_norm.device)
        return (
            ((tau_norm + 1.0) / 2.0 * (domain_size_t + 2 * domain_pad_t) - domain_pad_t)
            .cpu()
            .numpy()
        )

    # ── acquisition scoring ───────────────────────────────────────────────────

    def _score_trajectories(
        self,
        physical_trajs: list[np.ndarray],
        belief,
        buffer_start: int = 2,
    ) -> np.ndarray:
        """
        Score K candidate trajectories with the chosen acquisition function.
        Only evaluates future waypoints [buffer_start:].
        Applies a boundary mask and penalty so out-of-domain waypoints do not
        exploit unobserved GP prior variance.

        Parameters
        ----------
        physical_trajs : list of (horizon, 2) arrays in physical coordinates
        belief         : GP Belief object with .predict(positions) → (mean, var)
        buffer_start   : Index of first future waypoint to evaluate

        Returns
        -------
        scores : (K,) numpy array
        """
        K = len(physical_trajs)
        trajs_np = np.stack(physical_trajs, axis=0)  # (K, H, 2)
        future_trajs = trajs_np[:, buffer_start:, :]  # (K, H_future, 2)
        H_future = future_trajs.shape[1]

        all_positions = torch.tensor(
            future_trajs.reshape(K * H_future, 2), dtype=torch.float32
        )  # (K * H_future, 2)

        with torch.no_grad():
            means, variances = belief.predict(all_positions)  # (K * H_future,) each

        means     = means.cpu().view(K, H_future)      # (K, H_future)
        variances = variances.cpu().view(K, H_future)  # (K, H_future)
        stds      = variances.sqrt()

        # Position-level in-bounds mask (1.0 inside domain, 0.0 outside)
        W, H_dim = self.domain_size
        x_pts = future_trajs[..., 0]
        y_pts = future_trajs[..., 1]
        in_bounds = (x_pts >= 0.0) & (x_pts <= W) & (y_pts >= 0.0) & (y_pts <= H_dim)
        mask = torch.from_numpy(in_bounds).float()

        if self.acquisition == "variance":
            point_acq = variances * mask
        elif self.acquisition == "std":
            point_acq = stds * mask
        elif self.acquisition == "ucb":
            point_acq = (means + self.beta * stds) * mask
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition!r}. Use 'variance', 'std', or 'ucb'.")

        # Temporal discounting gamma^t (earlier waypoints weigh more)
        if self.gamma < 1.0:
            discounts = (self.gamma ** torch.arange(H_future, dtype=torch.float32)).unsqueeze(0)
            point_acq = point_acq * discounts

        scores = point_acq.sum(dim=1)  # (K,)

        # Optional soft penalty if boundary_penalty > 0
        if self.boundary_penalty > 0.0:
            out_of_bounds_count = (~in_bounds).sum(axis=1)  # (K,)
            scores = scores - torch.from_numpy(out_of_bounds_count).float() * self.boundary_penalty

        return scores.numpy()

    # ── warm-start seed ───────────────────────────────────────────────────────

    def _build_warm_start_seed(self, cond: dict) -> torch.Tensor:
        """
        Shift the previously generated normalised trajectory to align with the
        new conditioning, producing a (1, horizon, 2) seed for partial noising.
        """
        H = self.horizon
        seed = torch.zeros((1, H, 2), dtype=torch.float32, device=self.device)
        for idx, val in cond.items():
            seed[0, idx] = val[0]

        # The unexecuted future waypoints start at index (2 + steps_since_replan)
        shift_idx = 2 + self._steps_since_replan if (0 in cond and 1 in cond) else 1 + self._steps_since_replan
        suffix     = self._last_tau_norm[0, shift_idx:]
        suffix_len = suffix.shape[0]
        copy_len   = min(suffix_len, H - 2)
        if copy_len > 0:
            seed[0, 2 : 2 + copy_len] = suffix[:copy_len]

        tail_start = 2 + copy_len
        if tail_start < H:
            if tail_start >= 2:
                last_disp = seed[0, tail_start - 1] - seed[0, tail_start - 2]
            else:
                last_disp = self._last_tau_norm[0, -1] - self._last_tau_norm[0, -2]
            last_pt   = seed[0, tail_start - 1]
            for j in range(H - tail_start):
                seed[0, tail_start + j] = (last_pt + (j + 1) * last_disp).clamp(-1.0, 1.0)

        return seed

    # ── replanning ────────────────────────────────────────────────────────────

    def _replan(self, current_position: tuple[float, float], belief) -> None:
        """
        Sample K trajectories in a single batched pass from the unconditioned
        diffusion model and execute the one with the highest acquisition score.
        """
        p_curr = np.array(current_position, dtype=np.float32)

        # Two-point conditioning (position inpainting only, no belief/return CFG)
        current_position_norm = self._normalise(p_curr)
        if self._last_position is not None:
            last_position_norm = self._normalise(self._last_position)
            cond = {
                0: last_position_norm.unsqueeze(0).to(self.device),    # (1, 2)
                1: current_position_norm.unsqueeze(0).to(self.device),  # (1, 2)
            }
            buffer_start = 2
        else:
            cond = {0: current_position_norm.unsqueeze(0).to(self.device)}
            buffer_start = 1

        K = self.n_samples
        cond_batched = {
            step_idx: pos_tensor.repeat(K, 1)
            for step_idx, pos_tensor in cond.items()
        }

        # Dummy belief/return tensors matching batch size K
        N_crop  = self.crop_size ** 2
        b_dummy = torch.zeros((K, N_crop), dtype=torch.float32, device=self.device)
        r_dummy = torch.zeros((K, 1),      dtype=torch.float32, device=self.device)

        # Warm-start seed (broadcast to K so q_sample adds independent noise per candidate)
        x_init      = None
        noise_steps = None
        if self.warm_start and self._last_tau_norm is not None:
            seed_1      = self._build_warm_start_seed(cond)  # (1, horizon, 2)
            x_init      = seed_1.repeat(K, 1, 1)             # (K, horizon, 2)
            noise_steps = self.noise_steps

        # Sample K candidate trajectories in a single batched pass
        with torch.no_grad():
            candidate_norms = self.diffusion.conditional_sample(
                cond_batched,
                b_dummy,
                b_dummy,
                b_dummy,
                r_dummy,
                x_init=x_init,
                noise_steps=noise_steps,
                use_belief=False,
                use_return=False,
            )  # (K, horizon, 2)

        # Decode to physical coordinates
        physical_trajs = [self._denormalise(candidate_norms[k]) for k in range(K)]

        # Score with BO acquisition
        scores = self._score_trajectories(physical_trajs, belief, buffer_start=buffer_start)
        best_k = int(np.argmax(scores))

        log.debug(
            "DiffusionBOPlanner replan: K=%d, best=%d, acq=%.4f (acq=%s)",
            self.n_samples, best_k, scores[best_k], self.acquisition,
        )

        # Store best normalised trajectory for warm start
        self._last_tau_norm = candidate_norms[best_k : best_k + 1]  # (1, horizon, 2)

        # Fill waypoint buffer from the best trajectory
        best_traj = physical_trajs[best_k]
        self._waypoint_buffer = [
            (float(best_traj[i, 0]), float(best_traj[i, 1]))
            for i in range(buffer_start, self.horizon)
        ]
        self._steps_since_replan = 0

    # ── BasePlanner interface ─────────────────────────────────────────────────

    def reset(
        self,
        initial_position: tuple[float, float] | None = None,
        initial_heading: float | None = None,
    ) -> None:
        """Reset planner state between episodes."""
        self._waypoint_buffer    = []
        self._steps_since_replan = 0
        self._last_tau_norm      = None
        self._is_steering_back   = False

        if initial_position is not None and initial_heading is not None:
            self._last_position = np.array(
                [
                    initial_position[0] - self.max_step * np.cos(initial_heading),
                    initial_position[1] - self.max_step * np.sin(initial_heading),
                ],
                dtype=np.float32,
            )
        else:
            self._last_position = None

    def compute_next_pose(
        self,
        current_position: tuple[float, float],
        current_heading: float,
        current_belief,
    ) -> tuple[tuple[float, float], float]:
        """Returns (new_position, new_heading) — identical API to BasePlanner."""
        x, y = current_position
        W, H = self.domain_size

        # Out-of-domain: steer back, do not consume buffer
        if not (0 <= x <= W and 0 <= y <= H):
            new_heading  = self.steer_back_heading(current_position, current_heading)
            new_position = (
                current_position[0] + self.max_step * np.cos(new_heading),
                current_position[1] + self.max_step * np.sin(new_heading),
            )
            new_position, new_heading = self._clamp_outer(new_position, new_heading)
            self._last_position    = np.array(current_position, dtype=np.float32)
            self._is_steering_back = True
            return new_position, new_heading

        # Back in domain after steering: clear stale buffer and replan cleanly
        if self._is_steering_back:
            self._waypoint_buffer = []
            self._last_tau_norm   = None
            self._is_steering_back = False

        should_replan = (
            len(self._waypoint_buffer) == 0
            or self._steps_since_replan >= self.replan_every
        )

        if should_replan:
            self._replan(current_position, current_belief)

        new_position = self._waypoint_buffer.pop(0)
        self._steps_since_replan += 1
        self._last_position = np.array(current_position, dtype=np.float32)

        dx = new_position[0] - current_position[0]
        dy = new_position[1] - current_position[1]
        new_heading = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else current_heading

        return new_position, new_heading
