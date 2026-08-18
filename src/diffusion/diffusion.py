"""
diffusion.py
DDPM process wrapper for the AUV trajectory planner.

GaussianDiffusion wraps TemporalUnet and handles:
  - forward process  q(x_t | x_0)         via q_sample
  - reverse process  p(x_{t-1} | x_t)     via p_sample / p_sample_loop
  - CFG combination  ε = ε_u + w(ε_c - ε_u) in p_mean_variance
  - training loss    p_losses / loss

Key differences from the original GaussianDiffusion
----------------------------------------------------
  1. action_dim is always 0; the trajectory τ is a pure (k+1, 2) state
     sequence.  apply_conditioning pins t=0 to the AUV start position.
  2. The model signature includes b_mean / b_var for the GP belief maps.
  3. CFG runs two forward passes: one with (use_dropout=False) for the
     conditional estimate and one with (force_dropout=True) for the
     unconditional estimate, then combines them with guidance weight w.
  4. clip_denoised defaults to True (safe for normalised coordinates).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


class GaussianDiffusion(nn.Module):
    """
    Parameters
    ----------
    model : AUVTemporalUnet
    horizon : int
        Trajectory length k+1.
    observation_dim : int
        Waypoint dimensionality, always 2.
    n_timesteps : int
        Number of DDPM diffusion steps T.
    loss_type : str
        One of 'l1', 'l2', 'state_l2'.
    clip_denoised : bool
        Clamp x_recon to [-1, 1] after each reverse step.
    predict_epsilon : bool
        True  → model predicts ε (noise), standard DDPM.
        False → model predicts x_0 directly.
    loss_discount : float
        Exponential decay applied along the horizon dimension of the loss.
    condition_guidance_w : float
        CFG guidance weight w.  0 = unconditional; higher = stronger guidance.
        Used for both signals when belief_guidance_w / return_guidance_w are None.
    belief_guidance_w : float or None
        Per-signal guidance weight for the belief conditioning.
        When set (along with return_guidance_w), enables 3-pass composed CFG:
            ε̂ = ε_∅ + w_b*(ε_b − ε_∅) + w_r*(ε_r − ε_∅)
        Falls back to condition_guidance_w when None.
    return_guidance_w : float or None
        Per-signal guidance weight for the return conditioning.
        Falls back to condition_guidance_w when None.
    n_cond_steps : int
        Number of waypoints pinned by inpainting conditioning (1 or 2).
        Determines how many leading loss weights are zeroed during training.
        Set to 2 when using two-point heading conditioning.
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        observation_dim: int = 2,
        n_timesteps: int = 200,
        loss_type: str = "l2",
        clip_denoised: bool = True,
        predict_epsilon: bool = True,
        loss_discount: float = 1.0,
        condition_guidance_w: float = 1.2,
        n_cond_steps: int = 2,
        belief_guidance_w: float | None = None,
        return_guidance_w: float | None = None,
    ):
        super().__init__()

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.transition_dim = observation_dim  # action_dim = 0 for AUV
        self.model = model
        self.condition_guidance_w = condition_guidance_w
        self.belief_guidance_w = belief_guidance_w
        self.return_guidance_w = return_guidance_w
        self.n_cond_steps = n_cond_steps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # ── noise schedule ───────────────────────────────────────────────────
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # forward process coefficients
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # posterior q(x_{t-1} | x_t, x_0) coefficients
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # ── loss ─────────────────────────────────────────────────────────────
        loss_weights = self._get_loss_weights(loss_discount)
        # action_dim=0 → WeightedStateL2 is equivalent but we use WeightedL2
        # with the same weights for API consistency
        self.loss_fn = Losses[loss_type](loss_weights, action_dim=0)

    # ── loss weight construction ─────────────────────────────────────────────

    def _get_loss_weights(self, discount: float) -> torch.Tensor:
        """
        Shape: (horizon, 2).
        First waypoint weight is 0 because it is pinned by conditioning.
        """
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        weights = torch.einsum("h,t->ht", discounts, dim_weights)
        if self.predict_epsilon:
            weights[: self.n_cond_steps, :] = (
                0.0  # conditioned position, don't penalise
            )
        return weights

    # ── schedule helpers ─────────────────────────────────────────────────────

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise  # model directly predicts x_0

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract(self.posterior_variance, t, x_t.shape)
        log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return mean, variance, log_variance_clipped

    # ── reverse process ──────────────────────────────────────────────────────

    def _predict_eps_cfg(
        self,
        x: torch.Tensor,  # (B, horizon, 2)
        cond: dict,
        t: torch.Tensor,  # (B,)  int64
        b_mean: torch.Tensor,
        b_var: torch.Tensor,
        returns: torch.Tensor,
        use_belief: bool = True,
        use_return: bool = True,
    ) -> torch.Tensor:
        """
        CFG noise estimate. Two modes:

        Original (2-pass) — when belief_guidance_w and return_guidance_w are both None:
            ε̂ = ε_∅ + w * (ε_{b,r} − ε_∅)

        Composed (3-pass) — when per-signal weights are set:
            ε̂ = ε_∅ + w_b * (ε_b − ε_∅) + w_r * (ε_r − ε_∅)
        where ε_b has only belief active and ε_r has only return active.
        """
        _kw = dict(use_dropout=False, use_belief=use_belief, use_return=use_return)

        eps_uncond = self.model(x, cond, t, b_mean, b_var, returns,
                                force_dropout=True, **_kw)

        if self.belief_guidance_w is None and self.return_guidance_w is None:
            # original 2-pass CFG
            eps_cond = self.model(x, cond, t, b_mean, b_var, returns,
                                  force_dropout=False, **_kw)
            return eps_uncond + self.condition_guidance_w * (eps_cond - eps_uncond)

        # composed 3-pass CFG — isolate each signal's guidance direction
        w_b = self.belief_guidance_w if self.belief_guidance_w is not None else self.condition_guidance_w
        w_r = self.return_guidance_w if self.return_guidance_w is not None else self.condition_guidance_w

        eps_belief = self.model(x, cond, t, b_mean, b_var, returns,
                                force_dropout=False,
                                force_return_dropout=True, **_kw)
        eps_return = self.model(x, cond, t, b_mean, b_var, returns,
                                force_dropout=False,
                                force_belief_dropout=True, **_kw)

        return (eps_uncond
                + w_b * (eps_belief - eps_uncond)
                + w_r * (eps_return - eps_uncond))

    def p_mean_variance(
        self,
        x: torch.Tensor,  # (B, horizon, 2)
        cond: dict,
        t: torch.Tensor,  # (B,)
        b_mean: torch.Tensor,  # (B, 1600)
        b_var: torch.Tensor,  # (B, 1600)
        returns: torch.Tensor,  # (B, 1)
        use_belief: bool = True,
        use_return: bool = True,
    ):
        """
        CFG combination:
            ε = ε_uncond + w * (ε_cond − ε_uncond)
        """
        # # conditional prediction (belief + return active)
        # eps_cond = self.model(
        #     x, cond, t, b_mean, b_var, returns,
        #     use_dropout=False, force_dropout=False,
        # )
        # # unconditional prediction (belief + return zeroed)
        # eps_uncond = self.model(
        #     x, cond, t, b_mean, b_var, returns,
        #     use_dropout=False, force_dropout=True,
        # )
        # eps = eps_uncond + self.condition_guidance_w * (eps_cond - eps_uncond)

        eps = self._predict_eps_cfg(x, cond, t, b_mean, b_var, returns, use_belief, use_return)

        t_int = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t_int, noise=eps)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t_int
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        cond: dict,
        t: torch.Tensor,
        b_mean: torch.Tensor,
        b_var: torch.Tensor,
        returns: torch.Tensor,
        use_belief: bool = True,
        use_return: bool = True,
    ) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x, cond, t, b_mean, b_var, returns, use_belief, use_return
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise at t=0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple,
        cond: dict,
        b_mean: torch.Tensor,
        b_var: torch.Tensor,
        returns: torch.Tensor,
        verbose: bool = False,
        return_diffusion: bool = False,
        x_init: torch.Tensor | None = None,
        noise_steps: int | None = None,
        use_belief: bool = True,
        use_return: bool = True,
    ):
        """
        Full denoising chain from x_T ~ N(0, I) to x_0.
        """
        device = self.betas.device
        B = shape[0]

        if x_init is not None and noise_steps is not None:
            # warm start from x_init with noise corresponding to noise_steps
            # print("Warm starting diffusion with x_init and noise_steps =", noise_steps)
            t_warm = torch.full((B,), noise_steps - 1, device=device, dtype=torch.long)
            x = self.q_sample(x_init, t_warm)
            reverse_range = range(noise_steps)
        else:
            # cold start from pure noise
            # print("Cold starting diffusion from pure noise")
            x = 0.5 * torch.randn(shape, device=device)
            reverse_range = range(self.n_timesteps)

        x = apply_conditioning(x, cond, action_dim=0)

        diffusion_steps = [x] if return_diffusion else None

        for i in reversed(reverse_range):
            t_batch = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t_batch, b_mean, b_var, returns, use_belief, use_return)
            x = apply_conditioning(x, cond, action_dim=0)
            if return_diffusion:
                diffusion_steps.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion_steps, dim=1)
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        shape: tuple,
        cond: dict,
        b_mean: torch.Tensor,
        b_var: torch.Tensor,
        returns: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,  # 0 = fully deterministic DDIM
        return_diffusion: bool = False,
        x_init: torch.Tensor | None = None,
        noise_steps: int | None = None,
        use_belief: bool = True,
        use_return: bool = True,
    ):
        """
        DDIM reverse process (Song et al., 2021).

        Uses a strided subsequence of the trained DDPM schedule so the
        same model weights work at any step budget.

        Parameters
        ----------
        ddim_steps : int
            Number of denoising steps.  Sensible range 10–100.
            Full quality is typically recovered at ~50 for trajectories.
        eta : float
            Controls stochasticity.  eta=0 → deterministic ODE (classic
            DDIM).  eta=1 → variance matches DDPM.
        """
        device = self.betas.device
        B = shape[0]

        # ── build strided timestep subsequence ──────────────────────────────
        # # linspace from T-1 → 0, integer indices into alphas_cumprod
        # step_ratio = self.n_timesteps // ddim_steps
        # # e.g. n_timesteps=200, ddim_steps=50 → [199, 195, ..., 3]
        # timesteps = (
        #     torch.arange(ddim_steps, device=device) * step_ratio
        # ).long().flip(0)                                    # (ddim_steps,) descending

        # # paired (t_now, t_prev) — t_prev is the *earlier* (less noisy) index
        # timesteps_prev = torch.cat(
        #     [timesteps[1:], torch.zeros(1, device=device, dtype=torch.long)]
        # )                                                   # shifted by one, ends at 0
        if x_init is not None and noise_steps is not None:
            timesteps = torch.linspace(-1, noise_steps - 1, steps=ddim_steps + 1)
        else:
            timesteps = torch.linspace(-1, self.n_timesteps - 1, steps=ddim_steps + 1)
        timesteps = torch.round(timesteps).long().flip(0)

        # ── initialise ──────────────────────────────────────────
        if x_init is not None and noise_steps is not None:
            # warm start from x_init with noise corresponding to noise_steps
            t_warm = torch.full((B,), noise_steps - 1, device=device, dtype=torch.long)
            x = self.q_sample(x_init, t_warm)
        else:
            # cold start from pure noise
            x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, action_dim=0)

        diffusion_steps = [x] if return_diffusion else None

        # for t_now, t_prev in zip(timesteps, timesteps_prev):
        #     t_batch      = t_now.expand(B)                  # (B,)
        #     t_prev_batch = t_prev.expand(B)                 # (B,)

        for i in range(len(timesteps) - 1):
            t_now = timesteps[i]
            t_prev = timesteps[i + 1]
            t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

            # ── CFG noise prediction ─────────────────────────────────────────
            eps = self._predict_eps_cfg(x, cond, t_batch, b_mean, b_var, returns, use_belief, use_return)

            # ── predict x_0 from (x_t, ε) ───────────────────────────────────
            a_t = extract(self.alphas_cumprod, t_batch, x.shape)
            if t_prev >= 0:
                t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
                a_prev = extract(self.alphas_cumprod, t_prev_batch, x.shape)
            else:
                a_prev = torch.ones_like(a_t)  # at t_prev=-1, we want a_prev=1

            x0_pred = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
            if self.clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # ── DDIM update ──────────────────────────────────────────────────
            # σ_t = eta * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1})
            sigma = (
                eta
                * ((1.0 - a_prev) / (1.0 - a_t)).sqrt()
                * (1.0 - a_t / a_prev).sqrt()
            )
            # direction pointing to x_t (deterministic part)
            dir_xt = (1.0 - a_prev - sigma**2).clamp(min=0.0).sqrt() * eps

            noise = sigma * torch.randn_like(x) if eta > 0.0 else 0.0
            x = a_prev.sqrt() * x0_pred + dir_xt + noise

            x = apply_conditioning(x, cond, action_dim=0)
            if return_diffusion:
                diffusion_steps.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion_steps, dim=1)
        return x

    @torch.no_grad()
    def conditional_sample(
        self,
        cond: dict,
        b_mean: torch.Tensor,
        b_var: torch.Tensor,
        returns: torch.Tensor,
        horizon: int | None = None,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        x_init: torch.Tensor | None = None,
        noise_steps: int | None = None,
        use_belief: bool = True,
        use_return: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate a batch of trajectories.

        Parameters
        ----------
        cond    : {0: (B, 2)}   – AUV start position
        b_mean  : (B, 1600)     – GP posterior mean (flattened)
        b_var   : (B, 1600)     – GP posterior variance (flattened)
        returns : (B, 1)        – target return / reward signal

        Returns
        -------
        tau : (B, horizon, 2)
        """
        device = self.betas.device
        B = b_mean.shape[0]
        horizon = horizon or self.horizon
        shape = (B, horizon, self.observation_dim)

        if use_ddim:
            return self.ddim_sample_loop(
                shape,
                cond,
                b_mean,
                b_var,
                returns,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                x_init=x_init,
                noise_steps=noise_steps,
                use_belief=use_belief,
                use_return=use_return,
                **kwargs,
            )

        return self.p_sample_loop(
            shape,
            cond,
            b_mean,
            b_var,
            returns,
            x_init=x_init,
            noise_steps=noise_steps,
            use_belief=use_belief,
            use_return=use_return,
            **kwargs,
        )

    # ── training ─────────────────────────────────────────────────────────────

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) using the reparameterisation trick."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,  # (B, horizon, 2)
        cond: dict,
        t: torch.Tensor,  # (B,)
        b_mean: torch.Tensor,  # (B, 1600)
        b_var: torch.Tensor,  # (B, 1600)
        returns: torch.Tensor,  # (B, 1)
    ):
        noise = torch.randn_like(x_start)

        # zero the noise at the conditioned position so we never penalise it
        if self.predict_epsilon:
            for idx in cond:
                noise[:, idx, :] = 0.0

        x_noisy = self.q_sample(x_start, t, noise)
        x_noisy = apply_conditioning(x_noisy, cond, action_dim=0)

        # model forward with CFG dropout active (training mode)
        x_recon = self.model(
            x_noisy,
            cond,
            t,
            b_mean,
            b_var,
            returns,
            use_dropout=True,
            force_dropout=False,
        )

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, action_dim=0)

        assert noise.shape == x_recon.shape, (
            f"Shape mismatch: noise {noise.shape} vs pred {x_recon.shape}"
        )

        target = noise if self.predict_epsilon else x_start
        loss, info = self.loss_fn(x_recon, target)
        return loss, info

    def loss(
        self,
        x: torch.Tensor,  # (B, horizon, 2)   – normalised trajectory
        cond: dict,  # {0: (B, 2)}
        b_mean: torch.Tensor,  # (B, 1600)
        b_var: torch.Tensor,  # (B, 1600)
        returns: torch.Tensor,  # (B, 1)
    ):
        B = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x.device).long()
        return self.p_losses(x, cond, t, b_mean, b_var, returns)

    def forward(self, cond, b_mean, b_var, returns, **kwargs):
        """Alias for conditional_sample for use during evaluation."""
        return self.conditional_sample(cond, b_mean, b_var, returns, **kwargs)
