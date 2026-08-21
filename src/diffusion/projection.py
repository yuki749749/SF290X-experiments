import numpy as np
import torch


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to (-π, π]. Works on any shape."""
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


# class SequentialProjector:
#     def __init__(
#         self,
#         v: float = 1.0,
#         max_turn: float = np.pi / 4,
#         n_iters: int = 1,
#         domain_size: tuple = (10.0, 10.0),
#         domain_pad: float = 5.0,
#     ):
#         self.v = v
#         self.max_turn = max_turn
#         self.n_iters = n_iters
#         self.domain_size = domain_size
#         self.domain_pad = domain_pad
#         self.v_norm = v * 2 / (domain_size[0] + 2 * domain_pad)

#     def project_speed(
#         self,
#         p_prev: torch.Tensor,  # (B, 2)
#         p_curr: torch.Tensor,  # (B, 2)
#         p_next: torch.Tensor,  # (B, 2)
#     ) -> torch.Tensor:
#         """Project p_next so that ||p_next - p_curr|| == v for all batch items."""
#         direction = p_next - p_curr  # (B, 2)
#         distance = torch.norm(direction, dim=-1, keepdim=True)  # (B, 1)

#         # fallback to incoming heading where direction is degenerate
#         fallback = p_curr - p_prev  # (B, 2)
#         fallback_dist = torch.norm(fallback, dim=-1, keepdim=True)  # (B, 1)

#         default = torch.zeros_like(direction)
#         default[:, 0] = 1.0  # +x axis for doubly-degenerate case

#         direction_normalized = torch.where(
#             distance > 1e-6,
#             direction / distance,
#             torch.where(
#                 fallback_dist > 1e-6,
#                 fallback / fallback_dist,
#                 default,
#             ),
#         )
#         return p_curr + direction_normalized * self.v_norm

#     def project_turn(
#         self,
#         p_prev: torch.Tensor,  # (B, 2)
#         p_curr: torch.Tensor,  # (B, 2)
#         p_next: torch.Tensor,  # (B, 2)
#     ) -> torch.Tensor:
#         """Clamp the turn angle at p_curr to [-max_turn, max_turn]."""
#         diff_curr = p_curr - p_prev  # (B, 2)
#         diff_next = p_next - p_curr  # (B, 2)

#         theta_curr = torch.atan2(diff_curr[:, 1], diff_curr[:, 0])  # (B,)
#         theta_next = torch.atan2(diff_next[:, 1], diff_next[:, 0])  # (B,)

#         turn = wrap_angle(theta_next - theta_curr)  # (B,)
#         turn = torch.clamp(turn, -self.max_turn, self.max_turn)
#         theta_next = theta_curr + turn  # (B,)

#         direction = torch.stack(
#             [torch.cos(theta_next), torch.sin(theta_next)], dim=-1
#         )  # (B, 2)
#         return p_curr + self.v_norm * direction

#     def project_domain(
#         self,
#         p_next: torch.Tensor,  # (B, 2)
#     ) -> torch.Tensor:
#         return torch.clamp(
#             p_next,
#             -1.0,
#             1.0,  # since we assume input is already normalised to [-1, 1]
#         )

#     def project(self, tau: torch.Tensor) -> torch.Tensor:
#         """
#         Project a batch of trajectories onto the feasible set.

#         Parameters
#         ----------
#         tau : (B, T, 2)

#         Returns
#         -------
#         tau : (B, T, 2)  — feasibility-projected copy
#         """
#         tau = tau.clone()
#         T = tau.shape[1]

#         for _ in range(self.n_iters):
#             for t in range(1, T):
#                 p_prev = tau[:, t - 2] if t >= 2 else tau[:, t - 1]  # (B, 2)
#                 p_curr = tau[:, t - 1]  # (B, 2)
#                 p_next = tau[:, t]  # (B, 2)

#                 p_next = self.project_speed(p_prev, p_curr, p_next)
#                 if t >= 2:
#                     p_next = self.project_turn(p_prev, p_curr, p_next)
#                 p_next = self.project_domain(p_next)

#                 tau[:, t] = p_next

#         return tau

class SequentialProjector:
    def __init__(
        self,
        v: float = 1.0,
        max_turn: float = np.pi / 4,
        domain_size: tuple = (10.0, 10.0),
        domain_pad: float = 5.0,
    ):
        self.v = v
        self.max_turn = max_turn
        self.domain_size = domain_size
        self.domain_pad = domain_pad
        self.v_norm = v * 2 / (domain_size[0] + 2 * domain_pad)

    def _get_heading(
        self,
        p_from: torch.Tensor,  # (B, 2)
        p_to: torch.Tensor,    # (B, 2)
        p_fallback_from: torch.Tensor | None = None,  # (B, 2)
        p_fallback_to: torch.Tensor | None = None,    # (B, 2)
    ) -> torch.Tensor:
        """
        Compute heading angle from p_from to p_to.
        Falls back to the direction p_fallback_from -> p_fallback_to if
        the primary direction is degenerate, and to the +x axis as a last resort.

        Returns
        -------
        theta : (B,)
        """
        diff = p_to - p_from  # (B, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (B, 1)

        if p_fallback_from is not None and p_fallback_to is not None:
            fallback = p_fallback_to - p_fallback_from  # (B, 2)
            fallback_dist = torch.norm(fallback, dim=-1, keepdim=True)  # (B, 1)
        else:
            fallback = torch.zeros_like(diff)
            fallback_dist = torch.zeros(diff.shape[0], 1, device=diff.device)

        default = torch.zeros_like(diff)
        default[:, 0] = 1.0  # +x axis

        direction = torch.where(
            dist > 1e-6,
            diff / (dist + 1e-12),
            torch.where(
                fallback_dist > 1e-6,
                fallback / (fallback_dist + 1e-12),
                default,
            ),
        )  # (B, 2)

        return torch.atan2(direction[:, 1], direction[:, 0])  # (B,)

    def _project_step(
        self,
        p_prev: torch.Tensor,   # (B, 2) — projected, for incoming heading
        p_curr: torch.Tensor,   # (B, 2) — projected, anchor point
        p_hat_next: torch.Tensor,  # (B, 2) — diffusion proposal for next point
    ) -> torch.Tensor:
        """
        Unified speed + turn-rate projection for a single waypoint.

        Computes the proposed heading from p_curr to p_hat_next (Option A:
        always from the original diffusion output), clamps the turn relative
        to the incoming heading p_prev -> p_curr, and places the projected
        point at exactly v_norm from p_curr.

        Parameters
        ----------
        p_prev     : (B, 2)  previously projected waypoint (or inpainted)
        p_curr     : (B, 2)  current projected waypoint (or inpainted)
        p_hat_next : (B, 2)  diffusion proposal for the next waypoint

        Returns
        -------
        p_next : (B, 2)  projected next waypoint
        """
        # incoming heading: p_prev -> p_curr
        theta_in = self._get_heading(p_prev, p_curr)  # (B,)

        # proposed heading: p_curr -> p_hat_next (always from original plan)
        theta_raw = self._get_heading(
            p_curr, p_hat_next,
            p_fallback_from=p_prev, p_fallback_to=p_curr,
        )  # (B,)

        # clamp turn
        turn = wrap_angle(theta_raw - theta_in)  # (B,)
        turn = torch.clamp(turn, -self.max_turn, self.max_turn)
        theta_out = theta_in + turn  # (B,)

        # place next point at exactly v_norm from p_curr
        direction = torch.stack(
            [torch.cos(theta_out), torch.sin(theta_out)], dim=-1
        )  # (B, 2)
        return p_curr + self.v_norm * direction

    def _project_domain(self, p: torch.Tensor) -> torch.Tensor:
        """Clip to normalised domain [-1, 1]^2."""
        return torch.clamp(p, -1.0, 1.0)

    def project(self, tau: torch.Tensor, cond: dict | None = None) -> torch.Tensor:
        """
        Project a batch of partial trajectories onto the feasible set C.

        Parameters
        ----------
        tau : (B, H, 2)
        cond : dict or None
            Conditioning dictionary. If a step index is in cond.keys(),
            it is considered pinned (inpainted) and will not be modified.

        Returns
        -------
        tau : (B, H, 2)  feasibility-projected copy.
        """
        tau_hat = tau.clone()  # original diffusion output — read-only
        tau = tau.clone()      # projected trajectory — write target
        H = tau.shape[1]

        # Determine which steps are pinned
        pinned_steps = set(cond.keys()) if cond is not None else {0}

        # --- index 1 ---
        if 1 not in pinned_steps:
            # speed projection only, no incoming heading
            p_curr     = tau[:, 0]      # (B, 2)
            p_hat_next = tau_hat[:, 1]  # (B, 2)

            diff = p_hat_next - p_curr
            dist = torch.norm(diff, dim=-1, keepdim=True)
            default = torch.zeros_like(diff)
            default[:, 0] = 1.0  # +x axis fallback
            direction = torch.where(dist > 1e-6, diff / (dist + 1e-12), default)
            tau[:, 1] = self._project_domain(p_curr + self.v_norm * direction)

        # --- indices 2..H-1: full speed + turn projection ---
        for k in range(2, H):
            if k in pinned_steps:
                continue
            p_prev     = tau[:, k - 2] if k >= 2 else tau[:, 0]  # (B, 2)
            p_curr     = tau[:, k - 1]                            # (B, 2)
            p_hat_next = tau_hat[:, k]                            # (B, 2)

            p_next = self._project_step(p_prev, p_curr, p_hat_next)
            tau[:, k] = self._project_domain(p_next)

        return tau