import torch
import numpy as np

class HardFlowScheduler:
    def __init__(self, device='cpu'):
        self.device = device

    def _penalty_bounds(self, y, lb, ub):
        p1 = torch.relu(y - ub)
        p2 = torch.relu(lb - y)
        return (p1.pow(2) + p2.pow(2)).sum()

    def sample(self, model, cond, shape, steps=20, num_iters=10, lambda_oc=1.0, bounds=None, smooth_ref=None, smooth_w=0.0, step_size=0.2, noise_scale=0.0):
        batch_size = cond.shape[0]
        x = torch.randn(shape, device=self.device)
        dt = 1.0 / steps
        if bounds is None:
            lb = torch.full_like(x, -1.0)
            ub = torch.full_like(x, 1.0)
        else:
            lb, ub = bounds
            lb = lb.to(self.device)
            ub = ub.to(self.device)
            lb = lb.expand_as(x)
            ub = ub.expand_as(x)
        if smooth_ref is not None:
            sref = smooth_ref.to(self.device).view(batch_size, -1, 1)
        else:
            sref = None
        for i in range(steps):
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=self.device)
            v = model(x, t, cond)
            x_bar = x + v * dt
            y = x_bar.clone().detach().requires_grad_(True)
            for _ in range(num_iters):
                obj = 0.5 * (lambda_oc / dt) * ((y - x_bar).pow(2)).sum()
                obj = obj + 1000.0 * self._penalty_bounds(y, lb, ub)
                if sref is not None and y.dim() == 3:
                    obj = obj + smooth_w * ((y[:, :, 0] - sref[:, :, 0]).pow(2)).sum()
                obj.backward()
                with torch.no_grad():
                    y -= step_size * y.grad
                    y[:] = torch.max(torch.min(y, ub), lb)
                y.grad = None
            x = y.detach()
            if noise_scale > 0:
                noise = torch.randn_like(x) * noise_scale * np.sqrt(dt)
                x = x + noise
        return x
