import torch

from diffusion.diffusion import GaussianDiffusion
from diffusion.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class ProjectedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, projector, **kwargs):
        super().__init__(**kwargs)
        self.projector = projector

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple,
        cond: dict,
        b_mean: torch.Tensor,
        b_var: torch.Tensor,
        b_bound: torch.Tensor,
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

        k_max = self.n_timesteps if noise_steps is None else noise_steps
        for k in reversed(reverse_range):
            t_batch = torch.full((B,), k, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t_batch, b_mean, b_var, b_bound, returns, use_belief, use_return)
            x = apply_conditioning(x, cond, action_dim=0)
            x_proj = self.projector.project(x.clone(), cond=cond)
            # print(torch.norm(x - x_proj).item())  # print projection difference at each step
            x = k/k_max * x + (1 - k/k_max) * x_proj  # blend projection with original step
            x = apply_conditioning(x, cond, action_dim=0)
            if return_diffusion:
                diffusion_steps.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion_steps, dim=1)
        return x

