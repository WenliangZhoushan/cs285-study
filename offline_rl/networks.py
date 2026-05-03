"""Shared networks: tanh-Gaussian actor, twin Q critic, value critic."""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(sizes: Sequence[int], activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    """SAC-style squashed Gaussian policy with tanh transform."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), action_scale: float = 1.0):
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden])
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)
        self.action_scale = action_scale

    def forward(self, obs):
        h = F.relu(self.trunk(obs))
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        """Sample action with reparameterization. Returns (action, log_prob)."""
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mu, std)
        u = normal.rsample()
        a = torch.tanh(u) * self.action_scale
        # log prob with tanh correction
        log_prob = normal.log_prob(u) - torch.log(self.action_scale * (1 - torch.tanh(u).pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return a, log_prob

    def log_prob(self, obs, action):
        """Log prob of a given action under current policy (used by IQL/CQL)."""
        mu, log_std = self(obs)
        std = log_std.exp()
        # invert tanh to recover pre-squash sample
        a = (action / self.action_scale).clamp(-0.999999, 0.999999)
        u = torch.atanh(a)
        log_prob = Normal(mu, std).log_prob(u) - torch.log(
            self.action_scale * (1 - a.pow(2)) + 1e-6
        )
        return log_prob.sum(-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mu, log_std = self(obs)
        if deterministic:
            return torch.tanh(mu) * self.action_scale
        std = log_std.exp()
        u = Normal(mu, std).sample()
        return torch.tanh(u) * self.action_scale


class TwinQ(nn.Module):
    """Two Q networks for clipped double-Q learning (SAC/CQL/IQL)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs, action):
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


class ValueCritic(nn.Module):
    """V(s) network used by IQL."""

    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])

    def forward(self, obs):
        return self.net(obs)
