"""
diffusion_planner.py
Wraps GaussianDiffusion as a drop-in BasePlanner.

Design
------
The diffusion model generates a full trajectory of `horizon` waypoints in one
call.  The planner keeps an internal waypoint buffer and replans when either
(a) the buffer is exhausted, or
(b) `replan_every` steps have elapsed since the last plan (receding horizon).

Interface is identical to the other planners:
    new_position, new_heading = planner.compute_next_pose(pos, heading, belief)

so the outer simulation loop in main.py needs no structural change.

Replanning predicate
--------------------
`replan_every` (int)  – replan after this many steps regardless of belief.
                         Set to `horizon - 1` for full open-loop execution,
                         or to a small integer (e.g. 5) for more responsive
                         receding-horizon behaviour.

Target return
-------------
`target_return` is the scalar reward signal fed to the CFG conditioning.
During evaluation pass the 95th-percentile return observed in training data,
or simply 1.0 (normalised maximum) to bias the model toward high-reward plans.
"""

from typing import Optional, Sequence

import math
import numpy as np
import torch
import torch.nn.functional as F

from planners.planners import BasePlanner
from diffusion.diffusion import GaussianDiffusion
from diffusion.helpers import crop_belief_map


class DiffusionPlanner(BasePlanner):
    """
    Parameters
    ----------
    diffusion : GaussianDiffusion
        Fully trained diffusion model (on the correct device).
    domain_size : tuple[float, float]
        Physical domain extent, e.g. (10.0, 10.0).  Used for tau
        normalisation to match TrajectoryDataset.normalise_tau.
    horizon : int
        Number of waypoints per planned trajectory (must match training).
    replan_every : int
        Execute this many buffered waypoints before replanning.
        Sensible range: 1 (replan every step, expensive) to horizon-1
        (fully open-loop, cheapest).  Default 5 is a good starting point.
    target_return : float
        Scalar return fed to the CFG return-conditioning branch.
    device : str | torch.device
    """

    def __init__(
        self,
        diffusion,
        domain_size: tuple,
        domain_pad: float,
        horizon: int,
        replan_every: int = 5,
        target_return: float = 1.0,
        device: str | torch.device = "cpu",
        warm_start: bool = False,
        noise_steps: int = 20,
        use_belief: bool = True,
        use_return: bool = True,
        crop_size: int = 40,
        domain_min: Optional[Sequence[float]] = None,
        domain_max: Optional[Sequence[float]] = None,
        grid_size: int = 40,
        max_step: float = 1.0,
        stats: Optional[dict] = None,
        use_egocentric: bool = False,
    ):
        super().__init__(
            domain_size=domain_size,
            domain_pad=domain_pad,
            max_step=max_step,
            min_step=max_step,
            max_turn=np.pi / 4,  # not used by this planner
            boundary_behavior="clamp",
        )
        self.diffusion = diffusion
        self.domain_size = domain_size
        self.domain_size_t = torch.tensor(domain_size, dtype=torch.float32)
        self.domain_pad_t = torch.tensor(domain_pad, dtype=torch.float32)
        self.horizon = horizon
        self.replan_every = replan_every
        self.target_return = target_return
        self.device = torch.device(device)
        self.warm_start = warm_start
        self.noise_steps = noise_steps
        self.use_belief = use_belief
        self.use_return = use_return
        self.crop_size = crop_size
        self.grid_size = grid_size
        _dmin = list(domain_min) if domain_min is not None else [0.0, 0.0]
        _dmax = list(domain_max) if domain_max is not None else list(domain_size[:2])
        
        # Shift domain bounds to include domain_pad so that crop_belief_map aligns with the grid covering [-pad, size + pad]
        self.domain_min = torch.tensor([_dmin[0] - domain_pad, _dmin[1] - domain_pad], dtype=torch.float32)
        self.domain_max = torch.tensor([_dmax[0] + domain_pad, _dmax[1] + domain_pad], dtype=torch.float32)
        self.use_egocentric = use_egocentric

        # Build global boundary map of size (grid_size, grid_size)
        # where inside [0, W] x [0, H] is 0.0, and outside (padded area) is 1.0.
        xs = torch.linspace(self.domain_min[0].item(), self.domain_max[0].item(), grid_size)
        ys = torch.linspace(self.domain_min[1].item(), self.domain_max[1].item(), grid_size)
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
        
        inner_min_x, inner_min_y = _dmin
        inner_max_x, inner_max_y = _dmax
        boundary_mask = ~((inner_min_x <= x_grid) & (x_grid <= inner_max_x) & 
                          (inner_min_y <= y_grid) & (y_grid <= inner_max_y))
        self.boundary_map = boundary_mask.float().flatten()  # (grid_size^2,)

        # Pre-construct padded evaluation grid for planner predictions
        self.planner_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)

        self.diffusion.eval()

        self.stats = stats
        if stats is not None:
            self.b_mean_mean = torch.tensor(stats.get("b_mean_global_mean", 0.0), dtype=torch.float32, device=self.device)
            self.b_mean_std  = torch.tensor(stats.get("b_mean_global_std", 1.0), dtype=torch.float32, device=self.device)
            self.b_var_mean  = torch.tensor(stats.get("b_var_global_mean", 0.0), dtype=torch.float32, device=self.device)
            self.b_var_std   = torch.tensor(stats.get("b_var_global_std", 1.0), dtype=torch.float32, device=self.device)
            self.normalize_beliefs = True
        else:
            self.normalize_beliefs = False

        # internal state
        self._waypoint_buffer: list[tuple[float, float]] = []
        self._steps_since_replan: int = 0
        self._last_position: np.ndarray | None = None # for two-point inpainting
        self._last_tau_norm: torch.Tensor | None = None  # for warm start seed
        self._is_steering_back: bool = False  # flag: were we out-of-domain last step

    # ── normalisation (mirrors TrajectoryDataset) ────────────────────────────

    def _normalise(self, pos_np: np.ndarray) -> torch.Tensor:
        """(2,) numpy → (2,) tensor in [-1, 1]."""
        t = torch.tensor(pos_np, dtype=torch.float32)
        return (t + self.domain_pad_t) / (self.domain_size_t + 2 * self.domain_pad_t) * 2.0 - 1.0

    def _denormalise(self, tau_norm: torch.Tensor) -> np.ndarray:
        """(..., 2) normalised tensor → numpy in physical units."""
        return (
            ((tau_norm + 1.0) / 2.0 * (self.domain_size_t + 2 * self.domain_pad_t) - self.domain_pad_t)
            .cpu()
            .numpy()
        )

    # # ── belief extraction ────────────────────────────────────────────────────

    # @staticmethod
    # def _extract_belief(belief) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Pull b_mean, b_var from the Belief object.

    #     Belief.predict returns (mean, variance) on the full testGrid which
    #     has shape (1600,) for a 40×40 grid – exactly what the model expects.
    #     """
    #     with torch.no_grad():
    #         mean, var = belief.predict(belief.testGrid)
    #     # ensure 1-D flat tensors
    #     return mean.flatten(), var.flatten()


    def _build_warm_start_seed(self, cond: dict) -> torch.Tensor:
        """
        Shift the previously generated normalised trajectory to align with the
        new two-point conditioning, producing a (1, horizon, 2) seed tensor
        that will be partially noised before denoising.
 
        Layout of the returned seed
        ---------------------------
        Index 0   : cond[0]  (last_position, pinned)
        Index 1   : cond[1]  (current_position, pinned)
        Index 2 … 2+suffix_len-1 : unexecuted suffix of the last trajectory
        Index 2+suffix_len … H-1 : linear extrapolation of the final step
 
        The extrapolation extends the last displacement vector of the old
        trajectory, which is both kinematically plausible and avoids the
        zero-padding artefact that would otherwise introduce a spurious
        deceleration signal.
 
        Parameters
        ----------
        cond : {0: (1,2), 1: (1,2)} tensors on self.device
 
        Returns
        -------
        seed : (1, horizon, 2) on self.device
        """
        H = self.horizon
        seed = torch.zeros((1, H, 2), dtype=torch.float32, device=self.device)

        # pin the two conditioning waypoints
        seed[0, 0] = cond[0][0]  # last_position
        seed[0, 1] = cond[1][0]  # current_position

        # unexecuted suffix: everything after the steps already consumed
        if self.use_egocentric:
            # Reconstruct previous global trajectory from previous body frame
            cos_prev = math.cos(self._last_heading)
            sin_prev = math.sin(self._last_heading)
            R_prev = torch.tensor([
                [cos_prev, -sin_prev],
                [sin_prev,  cos_prev]
            ], dtype=torch.float32, device=self.device)
            
            scale_factor = 2.0 / (self.domain_size_t + 2 * self.domain_pad_t)
            scale_factor = scale_factor.to(self.device)
            last_tau_scaled = self._last_tau_norm[0] / scale_factor
            p_prev_tensor = torch.tensor(self._last_pos_for_warm_start, dtype=torch.float32, device=self.device)
            tau_prev_global = last_tau_scaled @ R_prev.t() + p_prev_tensor
            
            # Transform to the new body frame centered at current_position and rotated by heading
            p_curr_tensor = torch.tensor(self._current_pos_for_warm_start, dtype=torch.float32, device=self.device)
            tau_new_trans = tau_prev_global - p_curr_tensor
            
            cos_curr = math.cos(self._current_heading)
            sin_curr = math.sin(self._current_heading)
            R_curr_T = torch.tensor([
                [cos_curr, sin_curr],
                [-sin_curr, cos_curr]
            ], dtype=torch.float32, device=self.device)
            
            tau_new_body = tau_new_trans @ R_curr_T.t()
            last_tau_norm_shifted = (tau_new_body * scale_factor).unsqueeze(0)
            
            suffix = last_tau_norm_shifted[0, self._steps_since_replan:]
        else:
            suffix = self._last_tau_norm[0, self._steps_since_replan:]
            
        suffix_len = suffix.shape[0]
        copy_len   = min(suffix_len, H - 2)
        if copy_len > 0:
            seed[0, 2 : 2 + copy_len] = suffix[:copy_len]

        # linear extrapolation: extend the last displacement vector for the remaining horizon
        tail_start = 2 + copy_len
        if tail_start < H:
            if self.use_egocentric:
                last_disp = (
                    last_tau_norm_shifted[0, -1] - last_tau_norm_shifted[0, -2]
                )
            else:
                last_disp = (
                    self._last_tau_norm[0, -1] - self._last_tau_norm[0, -2]
                )
            last_pt = seed[0, tail_start - 1]
            for j in range(H - tail_start):
                seed[0, tail_start + j] = (
                    last_pt + (j + 1) * last_disp
                ).clamp(-1.0, 1.0)
        
        return seed


    # ── replanning ───────────────────────────────────────────────────────────

    def _replan(
        self,
        current_position: tuple[float, float],
        belief,
    ) -> None:
        """
        Run one full denoising pass to populate self._waypoint_buffer.
        The current position is used as the conditioning start (cond[0]).
        """
        b_mean, b_var = belief.predict(self.planner_grid)  # predict on padded planner grid

        # Compute heading
        p_curr = np.array(current_position, dtype=np.float32)
        if self._last_position is not None:
            p_prev = self._last_position
            dx = p_curr[0] - p_prev[0]
            dy = p_curr[1] - p_prev[1]
            heading = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else 0.0
        else:
            heading = 0.0

        # Save current state for warm start reconstruction
        self._current_pos_for_warm_start = p_curr
        self._current_heading = heading

        # Pre-initialize boundary map (in case cropping is not used)
        b_bound = self.boundary_map.clone()

        if self.crop_size < self.grid_size or self.use_egocentric:
            if self.use_egocentric:
                agent_pos = torch.tensor(current_position, dtype=torch.float32)
            else:
                agent_pos = torch.tensor(current_position, dtype=torch.float32)
            b_mean, b_var, b_bound = crop_belief_map(
                b_mean, b_var, self.boundary_map, agent_pos,
                self.domain_min, self.domain_max,
                self.crop_size, self.grid_size,
            )

            # Rotate crop if egocentric
            if self.use_egocentric:
                grid_img = torch.stack([
                    b_mean.view(self.crop_size, self.crop_size),
                    b_var.view(self.crop_size, self.crop_size),
                    b_bound.view(self.crop_size, self.crop_size)
                ], dim=0) # (3, H, W)
                
                cos_a = math.cos(heading)
                sin_a = math.sin(heading)
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
                b_bound = rotated[2].flatten()

        # batch size 1
        b_mean = b_mean.unsqueeze(0).to(self.device)  # (1, N)
        b_var = b_var.unsqueeze(0).to(self.device)  # (1, N)
        b_bound = b_bound.unsqueeze(0).to(self.device)  # (1, N)

        # Apply normalization if stats were loaded
        if self.normalize_beliefs:
            b_mean = (b_mean - self.b_mean_mean) / (self.b_mean_std + 1e-8)
            b_var  = (b_var - self.b_var_mean) / (self.b_var_std + 1e-8)

        # Set conditioning constraints
        if self.use_egocentric:
            cond = {
                1: torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)
            }
            if self._last_position is not None:
                scale_factor = 2.0 / (self.domain_size_t + 2 * self.domain_pad_t)
                v_norm_x = -self.max_step * scale_factor[0]
                cond[0] = torch.tensor([[v_norm_x.item(), 0.0]], dtype=torch.float32, device=self.device)
                buffer_start = 2
            else:
                buffer_start = 1
        else:
            current_position_norm = self._normalise(
                np.array(current_position, dtype=np.float32)
            )  # (2,)
            if self._last_position is not None:
                last_position_norm = self._normalise(self._last_position)  # (2,)
                cond = {
                    0: last_position_norm.unsqueeze(0).to(self.device),  # (1, 2)
                    1: current_position_norm.unsqueeze(0).to(self.device),  # (1, 2)
                }
                buffer_start = 2
            else:
                cond = {0: current_position_norm.unsqueeze(0).to(self.device)}  # (1, 2)
                buffer_start = 1

        returns = torch.tensor(
            [[self.target_return]], dtype=torch.float32, device=self.device
        )  # (1, 1)

        # build the warm start seed if enabled and we have a previous trajectory to draw from
        x_init = None
        noise_steps = None
        if self.warm_start and self._last_tau_norm is not None:
            x_init = self._build_warm_start_seed(cond)  # (1, horizon, 2)
            noise_steps = self.noise_steps

        with torch.no_grad():
            tau_norm = self.diffusion.conditional_sample(
                cond,
                b_mean,
                b_var,
                b_bound,
                returns,
                x_init=x_init,
                noise_steps=noise_steps,
                use_belief=self.use_belief,
                use_return=self.use_return,
            )  # (1, horizon, 2)

        # store full normalised trajectory for the next warm start 
        self._last_tau_norm = tau_norm 

        if self.use_egocentric:
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            rot_mat = torch.tensor([
                [cos_h, -sin_h],
                [sin_h, cos_h]
            ], dtype=torch.float32, device=self.device)
            scale_factor = 2.0 / (self.domain_size_t + 2 * self.domain_pad_t)
            scale_factor = scale_factor.to(self.device)
            tau_body_phys = tau_norm[0] / scale_factor
            tau = (tau_body_phys @ rot_mat.t() + torch.tensor(current_position, dtype=torch.float32, device=self.device)).cpu().numpy()
        else:
            tau = self._denormalise(tau_norm[0])  # (horizon, 2)

        # skip the first 1 or 2 positions which are pinned by conditioning, and buffer the rest
        self._waypoint_buffer = [
            (float(tau[i, 0]), float(tau[i, 1]))
            for i in range(buffer_start, self.horizon)
        ]
        self._steps_since_replan = 0

        # Save for next warm start
        self._last_pos_for_warm_start = p_curr
        self._last_heading = heading

    # ── BasePlanner interface ────────────────────────────────────────────────

    def reset(
        self,
        initial_position: tuple[float, float] | None = None,
        initial_heading: float | None = None,
    ) -> None:
        """
        Reset planner state between episodes.
 
        Parameters
        ----------
        initial_position : (x, y) physical coordinates at t=0.
        initial_heading  : heading in radians at t=0.
 
        When both are provided a ghost "last position" is back-projected one
        step along the heading direction, so that the very first replan uses
        two-point inpainting and the generated trajectory respects the initial
        heading.  If either argument is omitted the first replan falls back to
        single-point conditioning (original behaviour).
        """
        self._waypoint_buffer = []
        self._steps_since_replan = 0
        self._last_tau_norm = None
        self._is_steering_back = False
        self._last_pos_for_warm_start = None
        self._last_heading = None
 
        if initial_position is not None and initial_heading is not None:
            # Synthesise a ghost waypoint one step behind the start so that
            # cond[0]=ghost, cond[1]=initial_position encodes the correct
            # heading as a displacement vector for two-point inpainting.
            self._last_position = np.array([
                initial_position[0] - self.max_step * np.cos(initial_heading),
                initial_position[1] - self.max_step * np.sin(initial_heading),
            ], dtype=np.float32)
        else:
            self._last_position = None

    def compute_next_pose(
        self,
        current_position: tuple[float, float],
        current_heading: float,
        current_belief,
    ) -> tuple[tuple[float, float], float]:
        """
        Returns (new_position, new_heading) consistent with BasePlanner API.

        Replanning is triggered when:
          - the waypoint buffer is empty (first call, or buffer exhausted), OR
          - replan_every steps have elapsed since the last plan.
        """

        x, y = current_position
        W, H = self.domain_size
        tolerance_rate = 1.0  # allow some tolerance beyond the domain boundary before steering back

        if not (-self.domain_pad * tolerance_rate <= x <= W + self.domain_pad * tolerance_rate and -self.domain_pad * tolerance_rate <= y <= H + self.domain_pad * tolerance_rate):
            # Out-of-domain: steer back, do not consume buffer
            new_heading = self.steer_back_heading(current_position, current_heading)
            new_position = (
                current_position[0] + self.max_step * np.cos(new_heading),
                current_position[1] + self.max_step * np.sin(new_heading),
            )
            new_position, new_heading = self._clamp_outer(new_position, new_heading)
            self._last_position = np.array(current_position, dtype=np.float32)
            self._is_steering_back = True
            return new_position, new_heading
        
        # Back in domain after steering: force a replan so the diffusion model
        # takes over cleanly from the current position rather than resuming a
        # stale buffer that was planned before the excursion.
        if self._is_steering_back:
            self._waypoint_buffer = []
            self._is_steering_back = False


        should_replan = (
            len(self._waypoint_buffer) == 0
            or self._steps_since_replan >= self.replan_every
        )

        if should_replan:
            self._replan(current_position, current_belief)

        # consume the next buffered waypoint
        new_position = self._waypoint_buffer.pop(0)
        self._steps_since_replan += 1

        # update last_position for the next replan's two-point conditioning
        self._last_position = np.array(current_position, dtype=np.float32)

        # derive heading from displacement (used for logging / BO-compatibility)
        dx = new_position[0] - current_position[0]
        dy = new_position[1] - current_position[1]
        new_heading = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else current_heading

        return new_position, new_heading
