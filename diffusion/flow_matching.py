import torch
import torch.nn as nn
import numpy as np

class FlowMatchingScheduler:
    def __init__(self, sigma_min=1e-4, device='cpu'):
        self.sigma_min = sigma_min
        self.device = device

    def compute_loss(self, model, x_1, cond):
        """
        Computes the Flow Matching loss.
        x_1: Target data (batch, action_dim, horizon) or (batch, action_dim)
        cond: Condition (batch, obs_dim)
        """
        batch_size = x_1.shape[0]
        
        # 1. Sample t uniform [0, 1]
        t = torch.rand(batch_size, device=self.device)
        
        # 2. Sample x_0 (Noise) ~ N(0, 1)
        x_0 = torch.randn_like(x_1)
        
        # 3. Conditional Flow Matching (CFM) with Optimal Transport and sigma_min:
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        # Broadcasting t to match x shape
        if x_1.dim() == 3: # (B, C, L)
            t_b = t.view(batch_size, 1, 1)
        else: # (B, C)
            t_b = t.view(batch_size, 1)
            
        term_noise = 1 - (1 - self.sigma_min) * t_b
        x_t = term_noise * x_0 + t_b * x_1
        
        # 4. Target Velocity u_t
        # dx_t/dt = x_1 - (1 - sigma_min) * x_0
        u_t = x_1 - (1 - self.sigma_min) * x_0
        
        # 5. Model Prediction v_theta
        # Model takes (x, t, cond)
        
        v_pred = model(x_t, t, cond)
        
        # 6. Loss
        loss = torch.mean((v_pred - u_t) ** 2)
        
        return loss

    @torch.no_grad()
    def sample(self, model, cond, shape, steps=50, noise_scale=0.0):
        """
        Generate samples using Euler integration.
        shape: tuple of output shape (e.g. (batch_size, action_dim) or (batch_size, action_dim, horizon))
        """
        batch_size = cond.shape[0] # Should match shape[0]
        
        # Start from Noise x_0
        x = torch.randn(shape, device=self.device)
            
        dt = 1.0 / steps
        
        # Integration Loop t: 0 -> 1
        for i in range(steps):
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=self.device)
            
            # Predict velocity
            v = model(x, t, cond)
            
            # Euler Step
            x = x + v * dt
            
            # Add Noise (Langevin-like heuristic)
            if noise_scale > 0:
                # Add noise scaled by sqrt(dt)
                noise = torch.randn_like(x) * noise_scale * np.sqrt(dt)
                x = x + noise
            
        return x
