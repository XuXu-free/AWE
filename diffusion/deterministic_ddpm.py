import torch
from .ddpm import DDPMScheduler
from .guided_ddpm import GuidedDDPMScheduler


class DeterministicDDPMScheduler(DDPMScheduler):
    """
    Deterministic DDPM scheduler without any stochastic noise injection.

    At each denoising step, only the model-predicted mean is used:
        x_{t-1} = sqrt_recip_alpha_t * (x_t - beta_t * eps / sqrt_one_minus_alpha_bar_t)

    This makes sampling fully deterministic: the same condition always
    produces the same action sequence, which can be desirable for
    reproducible closed-loop control evaluation.
    """

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

        # 2. Deterministic mean only — no random noise added
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )

        return model_mean


class DeterministicGuidedDDPMScheduler(GuidedDDPMScheduler):
    """
    Guided DDPM scheduler without stochastic noise injection.

    Combines gradient guidance toward the previous action (as in
    GuidedDDPMScheduler) with fully deterministic denoising.

    Useful for evaluating the effect of guidance in isolation,
    without the randomness of diffusion sampling.
    """

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

        # 3. Deterministic mean only — no random noise added
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )

        return model_mean
