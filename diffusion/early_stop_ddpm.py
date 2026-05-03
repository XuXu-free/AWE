import torch
from .ddpm import DDPMScheduler
from .guided_ddpm import GuidedDDPMScheduler


class EarlyStopNoiseDDPMScheduler(DDPMScheduler):
    """
    DDPM scheduler with early stopping of noise injection.

    Based on 'Reduced Jerk by Stopping Noise Injection Early' (Eq. 14).
    The idea is to only add stochastic noise during the lower-SNR (larger t)
    denoising steps, and switch to deterministic mean-only denoising once
    the signal-to-noise ratio becomes high enough.

    This reduces jerk in closed-loop control while still allowing the model
    to explore multi-modality during the early denoising phase.

    Args:
        noise_stop_timestep: denoising steps with t_index <= this value
                             will NOT inject random noise (deterministic).
    """

    def __init__(self, noise_stop_timestep=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_stop_timestep = noise_stop_timestep

    @torch.no_grad()
    def sample(self, model, cond, shape):
        device = self.betas.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, cond, i)

        return img

    def p_sample(self, model, x, t, cond, t_index):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(
            self.sqrt_recip_alphas, t, x.shape
        )

        # 1. Model prediction (noise)
        eps = model(x, t, cond)

        # 2. Denoising mean (standard DDPM)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )

        # 3. Noise injection only when t_index > noise_stop_timestep
        if t_index > self.noise_stop_timestep:
            posterior_variance_t = self._extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            # Deterministic: no noise injected
            return model_mean


class EarlyStopNoiseGuidedDDPMScheduler(GuidedDDPMScheduler):
    """
    Guided DDPM scheduler with early stopping of noise injection.

    Combines gradient guidance toward the previous action with early
    noise stopping for smoother closed-loop trajectories.
    """

    def __init__(self, noise_stop_timestep=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_stop_timestep = noise_stop_timestep

    @torch.no_grad()
    def sample(self, model, cond, shape, prev_action_norm=None):
        device = self.betas.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, cond, i, prev_action_norm)

        return img

    def p_sample(self, model, x, t, cond, t_index, prev_action_norm=None):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(
            self.sqrt_recip_alphas, t, x.shape
        )

        # 1. Model prediction (noise)
        eps = model(x, t, cond)

        # 2. Gradient guidance toward previous action
        if prev_action_norm is not None and self.guidance_weight > 0:
            variance = self._extract(
                1.0 - self.alphas_cumprod, t, x.shape
            )

            if prev_action_norm.dim() == 2:
                prev_action_norm = prev_action_norm.unsqueeze(-1)

            if prev_action_norm.shape[-1] == 1 and x.shape[-1] > 1:
                prev_action_norm = prev_action_norm.expand(-1, -1, x.shape[-1])

            grad_guide = (x - prev_action_norm) / variance
            eps = eps + self.guidance_weight * grad_guide

        # 3. Denoising mean (standard DDPM)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )

        # 4. Noise injection only when t_index > noise_stop_timestep
        if t_index > self.noise_stop_timestep:
            posterior_variance_t = self._extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            # Deterministic: no noise injected
            return model_mean
