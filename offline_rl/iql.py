"""Implicit Q-Learning (Kostrikov et al., 2021).

Three networks:
  Q(s, a)   - twin Q + target Q
  V(s)      - value network, trained with expectile regression
  pi(a|s)   - tanh-Gaussian policy, trained with advantage-weighted regression

Losses:
  L_V = E[ L_tau(Q_target(s, a) - V(s)) ]                with L_tau(u) = |tau - 1{u<0}| u^2
  L_Q = E[ ( Q(s, a) - (r + gamma * V(s')) )^2 ]
  L_pi = - E[ exp(beta * (Q_target(s, a) - V(s))) * log pi(a|s) ]   (weights clipped)
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from networks import TanhGaussianActor, TwinQ, ValueCritic
from utils import soft_update


@dataclass
class IQLConfig:
    obs_dim: int
    act_dim: int
    action_scale: float = 1.0
    discount: float = 0.99
    tau: float = 0.005           # target network polyak rate
    expectile: float = 0.7       # tau in IQL paper (asymmetric L2)
    beta: float = 3.0            # AWR temperature (1/temperature)
    adv_clip: float = 100.0      # clip exp(beta*adv) for stability
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden: tuple = (256, 256)


class IQLAgent:
    def __init__(self, cfg: IQLConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

        self.actor = TanhGaussianActor(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.action_scale).to(device)
        self.q = TwinQ(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(device)
        self.q_target = TwinQ(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.v = ValueCritic(cfg.obs_dim, cfg.hidden).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.value_lr)

    @staticmethod
    def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff.pow(2)).mean()

    def update_v(self, batch):
        with torch.no_grad():
            q1, q2 = self.q_target(batch["obs"], batch["act"])
            target_q = torch.min(q1, q2)

        v = self.v(batch["obs"])
        loss = self.expectile_loss(target_q - v, self.cfg.expectile)

        self.v_opt.zero_grad()
        loss.backward()
        self.v_opt.step()
        return {"v_loss": loss.item(), "v_mean": v.mean().item()}

    def update_q(self, batch):
        with torch.no_grad():
            next_v = self.v(batch["next_obs"])
            target = batch["rew"] + self.cfg.discount * (1.0 - batch["done"]) * next_v

        q1, q2 = self.q(batch["obs"], batch["act"])
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        return {"q_loss": loss.item(), "q_mean": ((q1 + q2) / 2).mean().item()}

    def update_actor(self, batch):
        with torch.no_grad():
            q1, q2 = self.q_target(batch["obs"], batch["act"])
            target_q = torch.min(q1, q2)
            v = self.v(batch["obs"])
            adv = target_q - v
            weight = torch.exp(self.cfg.beta * adv).clamp(max=self.cfg.adv_clip)

        log_prob = self.actor.log_prob(batch["obs"], batch["act"])
        loss = -(weight * log_prob).mean()

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return {
            "actor_loss": loss.item(),
            "adv_mean": adv.mean().item(),
            "weight_mean": weight.mean().item(),
        }

    def update(self, batch):
        # Order matters: V uses target Q; Q uses V (current); actor uses target Q & V.
        info = {}
        info.update(self.update_v(batch))
        info.update(self.update_q(batch))
        info.update(self.update_actor(batch))
        soft_update(self.q_target, self.q, self.cfg.tau)
        return info
