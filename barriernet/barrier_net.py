import torch
import torch.nn as nn
import torch.nn.functional as F


class BarrierNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden1: int,
        n_hidden21: int,
        n_hidden22: int,
        n_cls: int,
        mean,
        std,
        device=None,
        bn: bool = False,
        obs_x: float = 40.0,
        obs_y: float = 15.0,
        obs_r: float = 6.0,
        u_min=None,
        u_max=None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden1 = n_hidden1
        self.n_hidden21 = n_hidden21
        self.n_hidden22 = n_hidden22
        self.n_cls = n_cls
        self.bn = bn

        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

        self.obs_x = float(obs_x)
        self.obs_y = float(obs_y)
        self.obs_r = float(obs_r)

        if u_min is None:
            u_min_t = torch.full((n_cls,), -float("inf"), dtype=torch.float32)
        else:
            u_min_t = torch.as_tensor(u_min, dtype=torch.float32).view(-1)
        if u_max is None:
            u_max_t = torch.full((n_cls,), float("inf"), dtype=torch.float32)
        else:
            u_max_t = torch.as_tensor(u_max, dtype=torch.float32).view(-1)
        if u_min_t.numel() != n_cls or u_max_t.numel() != n_cls:
            raise ValueError("u_min/u_max must have shape (n_cls,)")
        self.register_buffer("u_min", u_min_t)
        self.register_buffer("u_max", u_max_t)

        self.fc1 = nn.Linear(n_features, n_hidden1)
        self.fc21 = nn.Linear(n_hidden1, n_hidden21)
        self.fc22 = nn.Linear(n_hidden1, n_hidden22)
        self.fc31 = nn.Linear(n_hidden21, n_cls)
        self.fc32 = nn.Linear(n_hidden22, 2)

        if bn:
            self.bn1 = nn.BatchNorm1d(n_hidden1)
            self.bn21 = nn.BatchNorm1d(n_hidden21)
            self.bn22 = nn.BatchNorm1d(n_hidden22)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sgn: int = 1) -> torch.Tensor:
        n_batch = x.size(0)
        x = x.view(n_batch, -1)
        x0 = x * self.std + self.mean

        z = F.relu(self.fc1(x))
        if self.bn:
            z = self.bn1(z)

        z21 = F.relu(self.fc21(z))
        if self.bn:
            z21 = self.bn21(z21)

        z22 = F.relu(self.fc22(z))
        if self.bn:
            z22 = self.bn22(z22)

        u_ref = self.fc31(z21)
        cbf_params = 4.0 * torch.sigmoid(self.fc32(z22))

        u_safe = self.dcbf(x0, u_ref, cbf_params, sgn)
        return u_safe

    def dcbf(self, x0: torch.Tensor, u_ref: torch.Tensor, cbf_params: torch.Tensor, sgn: int) -> torch.Tensor:
        u = u_ref
        u = torch.max(torch.min(u, self.u_max), self.u_min)
        if x0.size(1) < 4 or self.n_cls < 2:
            return u

        px = x0[:, 0]
        py = x0[:, 1]
        theta = x0[:, 2]
        v = x0[:, 3]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        barrier = (px - self.obs_x).pow(2) + (py - self.obs_y).pow(2) - self.obs_r**2
        barrier_dot = 2 * (px - self.obs_x) * v * cos_theta + 2 * (py - self.obs_y) * v * sin_theta
        lf2b = 2 * v.pow(2)

        lg_lfbu1 = -2 * (px - self.obs_x) * v * sin_theta + 2 * (py - self.obs_y) * v * cos_theta
        lg_lfbu2 = 2 * (px - self.obs_x) * cos_theta + 2 * (py - self.obs_y) * sin_theta

        g = torch.stack([-lg_lfbu1, -lg_lfbu2], dim=1)
        h = lf2b + (cbf_params[:, 0] + cbf_params[:, 1]) * barrier_dot + (cbf_params[:, 0] * cbf_params[:, 1]) * barrier

        u2_ref = u[:, :2]
        u2_safe = self._project_halfspace(u2_ref, g, h)

        u_out = u.clone()
        u_out[:, :2] = u2_safe
        u_out = torch.max(torch.min(u_out, self.u_max), self.u_min)
        return u_out

    def _project_halfspace(self, u: torch.Tensor, g: torch.Tensor, h: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        gu = (g * u).sum(dim=1)
        violated = gu > h
        if not torch.any(violated):
            return u

        g_v = g[violated]
        u_v = u[violated]
        h_v = h[violated]
        gu_v = gu[violated]

        denom = (g_v * g_v).sum(dim=1).clamp_min(eps)
        alpha = (gu_v - h_v) / denom
        u_proj = u_v - alpha.unsqueeze(1) * g_v

        out = u.clone()
        out[violated] = u_proj
        return out
