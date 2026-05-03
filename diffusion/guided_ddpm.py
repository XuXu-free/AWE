import torch
from .ddpm import DDPMScheduler


class GuidedDDPMScheduler(DDPMScheduler):
    """
    DDPM scheduler with gradient guidance toward the previously executed action.

    Based on 'Consistent Mode Selection through Gradient Guidance' [Eq. 13].
    The idea is to condition the sampled distribution on the previous control
    command u_{t-1} so that closed-loop mode selection remains consistent
    across time steps, avoiding jerky commands.

    At each denoising step we replace the model-predicted noise eps by:

        eps_guided = eps + w * (x_t - x_prev) / (1 - alpha_bar_t)

    where w is the guidance weight and x_prev is the normalized previous action.
    """

    def __init__(self, guidance_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance_weight = guidance_weight

    @torch.no_grad()
    def sample(self, model, cond, shape, prev_action_norm=None):
        """
        Reverse diffusion with optional gradient guidance.

        Args:
            model: noise-prediction network.
            cond: condition vector.
            shape: output shape (batch, action_dim, horizon).
            prev_action_norm: (batch, action_dim) or (batch, action_dim, 1)
                              normalized previous action used as guidance target.
        """
        device = self.betas.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, cond, i, prev_action_norm)

        return img

    def p_sample(self, model, x, t, cond, t_index, prev_action_norm=None):
        """
        Single denoising step with gradient guidance.
        """
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
            # Variance schedule from the paper: (1 - alpha_bar_t)
            variance = self._extract(
                1.0 - self.alphas_cumprod, t, x.shape
            )

            # Match shape for broadcasting
            if prev_action_norm.dim() == 2:
                # (B, action_dim) -> (B, action_dim, 1)
                prev_action_norm = prev_action_norm.unsqueeze(-1)

            # Expand to horizon if needed
            if prev_action_norm.shape[-1] == 1 and x.shape[-1] > 1:
                prev_action_norm = prev_action_norm.expand(-1, -1, x.shape[-1])

            # Eq. (13): f := f - grad_log N(x; prev, variance*I)
            # grad_log N = - (x - prev) / variance
            # Therefore: f := f + (x - prev) / variance
            grad_guide = (x - prev_action_norm) / variance
            eps = eps + self.guidance_weight * grad_guide

        # 3. Denoising mean (standard DDPM)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
