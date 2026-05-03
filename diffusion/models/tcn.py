import torch
import torch.nn as nn
from .common import SinusoidalPosEmb

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
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
        output_dim, 
        cond_dim, 
        output_num=16, 
        hidden_dim=256, 
        levels=3, 
        kernel_size=3, 
        dropout=0.0,
        ):
        super(DiffusionTCN, self).__init__()
        self.action_dim = output_dim
        self.obs_dim = cond_dim
        self.horizon = output_num
        self.hidden_dim = hidden_dim
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Input projection (Conv1d for sequence)
        # Input shape: (Batch, Action_Dim, Horizon)
        self.input_proj = nn.Conv1d(output_dim, hidden_dim, 1)

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
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1)

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

class PureTCN(nn.Module):
    """
    Standard TCN for direct prediction without diffusion/flow matching.
    Input: Condition (State)
    Output: Action Sequence
    """
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
        super(PureTCN, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # Condition projection
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Conv1d(hidden_dim, hidden_dim, 1)

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

    def forward(self, cond):
        """
        cond: (batch, obs_dim)
        Returns: (batch, action_dim, horizon)
        """
        cond_emb = self.cond_proj(cond)
        h = cond_emb.unsqueeze(-1).repeat(1, 1, self.horizon)
        h = self.input_proj(h)

        for layer in self.tcn:
            h = layer(h)

        output = self.output_proj(h)
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