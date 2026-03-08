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