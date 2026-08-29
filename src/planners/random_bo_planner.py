"""
random_bo_planner.py
Random Kinematic Dubins Trajectory Prior + BO / WIVR Acquisition Selection.

This planner serves as the rigorous scientific ablation baseline for DiffusionBOPlanner:
It shares the EXACT same:
  • Horizon H (16 steps)
  • Kinematic constraints (v = 10m, max_turn = pi/8)
  • Re-planning frequency (replan_every)
  • Sample budget (K = n_samples)
  • BO / WIVR acquisition scoring function and grid Z
  • Domain handling and execution buffer

The ONLY difference:
  • DiffusionBOPlanner proposes candidates sampled from the learned diffusion prior p(tau).
  • RandomBOPlanner proposes candidates sampled randomly from the kinematically feasible Dubins manifold.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Sequence

import numpy as np
import torch

from planners.diffusion_bo_planner import DiffusionBOPlanner

log = logging.getLogger(__name__)


class RandomBOPlanner(DiffusionBOPlanner):
    """
    Random Kinematic Rollout Planner with BO / WIVR acquisition selection.

    Inherits all scoring, boundary handling, and BasePlanner execution logic
    from DiffusionBOPlanner.
    """

    def __init__(
        self,
        domain_size: tuple,
        domain_pad: float,
        horizon: int = 16,
        replan_every: int = 8,
        n_samples: int = 100,
        acquisition: str = "wivr",
        wivr_weighting: str = "plume",
        wivr_grid_size: int = 20,
        beta: float = 10.0,
        gamma: float = 1.0,
        device: str | torch.device = "cpu",
        domain_min: Optional[Sequence[float]] = None,
        domain_max: Optional[Sequence[float]] = None,
        grid_size: int = 40,
        max_step: float = 10.0,
        max_turn: float = np.pi / 8,
        boundary_penalty: float = 0.0,
        arc_prob: float = 0.5,
    ):
        # Instantiate parent with a dummy object for diffusion
        class _DummyDiffusion:
            def eval(self):
                pass

        super().__init__(
            diffusion=_DummyDiffusion(),
            domain_size=domain_size,
            domain_pad=domain_pad,
            horizon=horizon,
            replan_every=replan_every,
            n_samples=n_samples,
            acquisition=acquisition,
            beta=beta,
            gamma=gamma,
            device=device,
            warm_start=False,
            noise_steps=0,
            domain_min=domain_min,
            domain_max=domain_max,
            grid_size=grid_size,
            max_step=max_step,
            max_turn=max_turn,
            stats=None,
            crop_size=24,
            acq_mask_sharpness=3.0,
            boundary_penalty=boundary_penalty,
            wivr_weighting=wivr_weighting,
            wivr_grid_size=wivr_grid_size,
        )
        self.arc_prob = arc_prob

    def _sample_kinematic_candidates(
        self,
        p_curr: np.ndarray,
        p_prev: np.ndarray,
        K: int,
    ) -> list[np.ndarray]:
        """
        Generate K diverse, kinematically valid Dubins trajectories of length H.

        Each candidate satisfies:
          - Step length ||x_{t+1} - x_t|| == max_step (10m)
          - Turn constraint |theta_{t+1} - theta_t| <= max_turn (pi/8)
          - Smooth continuation from incoming heading p_prev -> p_curr
        """
        H = self.horizon
        v = self.max_step
        max_turn = self.max_turn

        # Incoming heading
        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]
        theta_0 = np.arctan2(dy, dx)

        trajs = np.zeros((K, H, 2), dtype=np.float32)
        trajs[:, 0] = p_prev
        trajs[:, 1] = p_curr

        for k in range(K):
            curr_pos = p_curr.copy()
            curr_heading = theta_0

            # Mixture: Constant-curvature arcs (clean circles/sweeps) vs. Stochastic curvature walk
            is_arc = (k < int(K * self.arc_prob))
            if is_arc:
                turn_rate = (2.0 * (k / max(1, int(K * self.arc_prob) - 1)) - 1.0) * max_turn
            else:
                turn_rate = None

            for t in range(2, H):
                if is_arc:
                    dtheta = turn_rate
                else:
                    dtheta = np.random.uniform(-max_turn, max_turn)

                curr_heading = curr_heading + dtheta
                curr_pos = curr_pos + v * np.array([np.cos(curr_heading), np.sin(curr_heading)], dtype=np.float32)
                trajs[k, t] = curr_pos

        return [trajs[k] for k in range(K)]

    def _replan(self, current_position: tuple[float, float], belief) -> None:
        """
        Sample K kinematically feasible Dubins trajectories and execute the one
        with the highest acquisition score.
        """
        p_curr = np.array(current_position, dtype=np.float32)

        if self._last_position is not None:
            p_prev = self._last_position
            buffer_start = 2
        else:
            # Ghost point behind current position along default heading
            p_prev = np.array([p_curr[0] - self.max_step, p_curr[1]], dtype=np.float32)
            buffer_start = 1

        K = self.n_samples
        physical_trajs = self._sample_kinematic_candidates(p_curr, p_prev, K)

        # Score with identical BO / WIVR acquisition function
        scores = self._score_trajectories(physical_trajs, belief, buffer_start=buffer_start)
        best_k = int(np.argmax(scores))

        log.debug(
            "RandomBOPlanner replan: K=%d, best=%d, acq=%.4f (acq=%s, weighting=%s)",
            self.n_samples, best_k, scores[best_k], self.acquisition, self.wivr_weighting,
        )

        # Fill waypoint buffer from best candidate
        best_traj = physical_trajs[best_k]
        self._waypoint_buffer = [
            (float(best_traj[i, 0]), float(best_traj[i, 1]))
            for i in range(buffer_start, self.horizon)
        ]
        self._steps_since_replan = 0
