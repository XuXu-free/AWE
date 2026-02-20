
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # This is a common way to embed time steps in diffusion models.
        # It maps a scalar t to a vector of dimension dim.
        # The embedding is periodic with a period of 10000.
        #
        # Implements the standard sinusoidal position embedding:
        # PE(t, 2i)   = sin(t / 10000^(2i / dim))
        # PE(t, 2i+1) = cos(t / 10000^(2i / dim))
        #
        # Where:
        # t is the time step (scalar)
        # i is the dimension index (0 <= i < dim/2)
        # dim is the embedding dimension
        #
        # The term `emb` calculated below corresponds to: 1 / 10000^(2i / dim)
        # which is equivalent to: exp(-2i * log(10000) / dim)
        # Actually, code uses (dim//2 - 1) in denominator, slightly adjusting the frequency spread.

        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Mish()  # Mish is commonly used in diffusion models
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h = self.norm1(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h

class DiffusionMLP(nn.Module):
    def __init__(
        self, 
        action_dim, 
        obs_dim, 
        hidden_dim=256, 
        num_res_blocks=3, 
        dropout=0.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        # Timestep embedding
        self.time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Main trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        
        # Final output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, t, cond):
        """
        x: (batch, action_dim) - Noisy action
        t: (batch,) - Timestep
        cond: (batch, obs_dim) - Observation/Condition
        """
        # Embeddings
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        cond_emb = self.cond_proj(cond)
        
        # Combine: usually simple addition or concatenation. 
        # Here we add them to the hidden state.
        h = x_emb + t_emb + cond_emb
        
        # Residual Blocks
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h)
        output = self.output_proj(h)
        return output

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.Mish()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.Mish()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.Mish()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class DiffusionTCN(nn.Module):
    def __init__(
        self, 
        action_dim, 
        obs_dim, 
        horizon=16, 
        hidden_dim=256, 
        levels=3, 
        kernel_size=3, 
        dropout=0.0,
        ):
        super(DiffusionTCN, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Condition projection
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)

        # Input projection (Conv1d for sequence)
        # Input shape: (Batch, Action_Dim, Horizon)
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 1)

        layers = []
        num_channels = [hidden_dim] * (levels + 1)
        for i in range(levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.ModuleList(layers)
        
        # Final output projection
        self.output_proj = nn.Conv1d(hidden_dim, action_dim, 1)

    def forward(self, x, t, cond):
        """
        x: (batch, action_dim, horizon) or (batch, action_dim) - Noisy action sequence
        t: (batch,) - Timestep
        cond: (batch, obs_dim) - Observation/Condition
        """
        # Handle non-sequence input by expanding
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (B, C, 1)
            is_seq = False
        else:
            # Assume (B, C, L)
            is_seq = True
            
        # Embeddings
        t_emb = self.time_mlp(t) # (B, Hidden)
        cond_emb = self.cond_proj(cond) # (B, Hidden)
        
        # Combine global conditioning
        global_cond = t_emb + cond_emb # (B, Hidden)
        global_cond = global_cond.unsqueeze(-1) # (B, Hidden, 1)
        
        # Input Projection
        h = self.input_proj(x) # (B, Hidden, L)
        
        # Add conditioning to feature map (broadcasting over time)
        h = h + global_cond
        
        # TCN Blocks
        for layer in self.tcn:
            h = layer(h)
            
        # Output Projection
        output = self.output_proj(h) # (B, Action, L)
        
        if not is_seq:
            output = output.squeeze(-1)
            
        return output

class FlowMatchingTCN(nn.Module):
    def __init__(
        self, 
        action_dim, 
        obs_dim, 
        horizon=16, 
        hidden_dim=256, 
        levels=4, 
        kernel_size=3, 
        dropout=0.0,
        ):
        super(FlowMatchingTCN, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Timestep embedding
        # Flow Matching passes t in [0, 1].
        # We need to scale it to make SinusoidalPosEmb effective.
        # Scaling by 1000 inside the model is a robust way to reuse the architecture.
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Condition projection
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)

        # Input projection (Conv1d for sequence)
        # Input shape: (Batch, Action_Dim, Horizon)
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 1)

        layers = []
        num_channels = [hidden_dim] * (levels + 1)
        for i in range(levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.ModuleList(layers)
        
        # Final output projection
        self.output_proj = nn.Conv1d(hidden_dim, action_dim, 1)

    def forward(self, x, t, cond):
        """
        x: (batch, action_dim, horizon) or (batch, action_dim) - Noisy action sequence
        t: (batch,) - Timestep [0, 1]
        cond: (batch, obs_dim) - Observation/Condition
        """
        # Internal scaling for SinusoidalPosEmb
        t = t * 1000.0
        
        # Handle non-sequence input by expanding
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (B, C, 1)
            is_seq = False
        else:
            # Assume (B, C, L)
            is_seq = True
            
        # Embeddings
        t_emb = self.time_mlp(t) # (B, Hidden)
        cond_emb = self.cond_proj(cond) # (B, Hidden)
        
        # Combine global conditioning
        global_cond = t_emb + cond_emb # (B, Hidden)
        global_cond = global_cond.unsqueeze(-1) # (B, Hidden, 1)
        
        # Input Projection
        h = self.input_proj(x) # (B, Hidden, L)
        
        # Add conditioning to feature map (broadcasting over time)
        h = h + global_cond
        
        # TCN Blocks
        for layer in self.tcn:
            h = layer(h)
            
        # Output Projection
        output = self.output_proj(h) # (B, Action, L)
        
        if not is_seq:
            output = output.squeeze(-1)
            
        return output
