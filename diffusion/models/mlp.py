import torch
import torch.nn as nn
from .common import SinusoidalPosEmb, ResidualBlock

class DiffusionMLP(nn.Module):
    def __init__(
        self, 
        action_dim, 
        obs_dim, 
        horizon=1,
        hidden_dim=256, 
        num_res_blocks=3, 
        dropout=0.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        
        flat_dim = action_dim * horizon
        
        # Timestep embedding
        self.time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(flat_dim, hidden_dim)
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Main trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        
        # Final output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, flat_dim)

    def forward(self, x, t, cond):
        """
        x: (batch, action_dim, horizon) or (batch, flattened_dim) - Noisy action
        t: (batch,) - Timestep
        cond: (batch, obs_dim) - Observation/Condition
        """
        # Flatten input if needed
        is_seq = False
        if x.dim() == 3:
            is_seq = True
            B, C, H = x.shape
            # Assuming C=action_dim, H=horizon. 
            # We transpose to (B, C, H) -> (B, C*H) if we assume channel-first.
            # But wait, typically MLP treats vector.
            # If input is (B, C, H), we usually want to preserve temporal structure or just flatten.
            # Since this is MLP, we just flatten everything.
            # But we must be consistent with how we reshape back.
            # If we flatten as x.reshape(B, -1), it becomes (B, C*H).
            # But `transpose(1, 2)` was used in the training script: `action_seq.transpose(1, 2).reshape(..., -1)`.
            # That means (B, C, H) -> (B, H, C) -> (B, H*C).
            # The previous code: `action_seq.transpose(1, 2).reshape(action_seq.shape[0], -1)`
            # That implies we group by time step: [a1_t1, a2_t1, ..., a1_t2, a2_t2, ...]
            #
            # However, here I should decide on a convention.
            # If I use `x.reshape(B, -1)` on `(B, C, H)`, it results in `[c1_h1, c1_h2, ..., c2_h1, ...]`.
            # This is different order.
            #
            # Let's stick to simple flattening `x.reshape(B, -1)` unless user specifically asked for (B, H, C).
            # But `DiffusionTCN` takes `(B, C, H)`.
            # If I use `reshape(B, -1)` on `(B, C, H)`, it get channel-first flattening.
            # If I reshape output back using `reshape(B, C, H)`, it restores correctly.
            # So `x.reshape(B, -1)` is fine as long as consistent.
            
            x = x.reshape(x.shape[0], -1)
        
        # Embeddings
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        cond_emb = self.cond_proj(cond)
        
        # Combine
        h = x_emb + t_emb + cond_emb
        
        # Residual Blocks
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h)
        output = self.output_proj(h)
        
        if is_seq:
            output = output.reshape(output.shape[0], self.action_dim, self.horizon)
            
        return output

class DiffusionPureMLP(nn.Module):
    """
    Diffusion model with a simple MLP trunk (no residual blocks).
    Used as an ablation to isolate the effect of residual connections
    vs. the diffusion training framework.
    """
    def __init__(
        self,
        action_dim,
        obs_dim,
        horizon=1,
        hidden_dim=256,
        num_layers=4,
        dropout=0.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon

        flat_dim = action_dim * horizon

        # Timestep embedding
        self.time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(flat_dim, hidden_dim)
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)

        # Simple MLP trunk (no residual blocks)
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)

        # Final output
        self.output_proj = nn.Linear(hidden_dim, flat_dim)

    def forward(self, x, t, cond):
        """
        x: (batch, action_dim, horizon) or (batch, flattened_dim) - Noisy action
        t: (batch,) - Timestep
        cond: (batch, obs_dim) - Observation/Condition
        """
        is_seq = False
        if x.dim() == 3:
            is_seq = True
            x = x.reshape(x.shape[0], -1)

        # Embeddings
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        cond_emb = self.cond_proj(cond)

        # Combine
        h = x_emb + t_emb + cond_emb

        # Simple MLP trunk
        h = self.trunk(h)

        output = self.output_proj(h)

        if is_seq:
            output = output.reshape(output.shape[0], self.action_dim, self.horizon)

        return output


class PureMLP(nn.Module):
    """
    Standard MLP for direct prediction without diffusion/flow matching.
    Input: Condition (State)
    Output: Action Sequence
    """
    def __init__(
        self, 
        action_dim, 
        obs_dim, 
        horizon=1,
        hidden_dim=256, 
        num_res_blocks=3, 
        dropout=0.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        
        flat_dim = action_dim * horizon
        
        # Input projection (Only Condition)
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Main trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        
        # Final output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, flat_dim)

    def forward(self, cond):
        """
        cond: (batch, obs_dim)
        Returns: (batch, action_dim, horizon)
        """
        # Embeddings
        h = self.cond_proj(cond)
        
        # Residual Blocks
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h)
        output = self.output_proj(h)
        
        # Reshape to (B, C, H)
        output = output.reshape(output.shape[0], self.action_dim, self.horizon)
            
        return output

class FlowMatchingMLP(nn.Module):
    def __init__(
        self, 
        action_dim, 
        obs_dim, 
        horizon=1,
        hidden_dim=256, 
        num_res_blocks=3, 
        dropout=0.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        
        flat_dim = action_dim * horizon
        
        # Timestep embedding
        # Flow Matching passes t in [0, 1].
        # We need to scale it to make SinusoidalPosEmb effective.
        self.time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(flat_dim, hidden_dim)
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Main trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        
        # Final output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, flat_dim)

    def forward(self, x, t, cond):
        """
        x: (batch, action_dim, horizon) or (batch, flattened_dim) - Noisy action
        t: (batch,) - Timestep [0, 1]
        cond: (batch, obs_dim) - Observation/Condition
        """
        # Flatten input if needed
        is_seq = False
        if x.dim() == 3:
            is_seq = True
            B, C, H = x.shape
            x = x.reshape(x.shape[0], -1)

        # Internal scaling for SinusoidalPosEmb
        t = t * 1000.0

        # Embeddings
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        cond_emb = self.cond_proj(cond)
        
        # Combine
        h = x_emb + t_emb + cond_emb
        
        # Residual Blocks
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h)
        output = self.output_proj(h)
        
        if is_seq:
            output = output.reshape(output.shape[0], self.action_dim, self.horizon)

        return output