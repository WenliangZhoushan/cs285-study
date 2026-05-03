"""Conservative Q-Learning (Kumar et al., 2020), continuous-action SAC-CQL.

Critic loss = SAC TD loss + alpha_cql * (logsumexp_a Q(s, a)  -  Q(s, a_data))

The conservative penalty pushes Q down on out-of-distribution actions and up
on data actions. We approximate logsumexp_a by importance sampling actions
from { uniform[-A, A], pi(.|s), pi(.|s') } and use the standard
log-mean-exp(Q - log_pi) estimator (CQL paper, eqn. 4 / Appendix F).

Actor + entropy temperature use the standard SAC objectives.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from offline_rl.networks import TanhGaussianActor, TwinQ
from offline_rl.utils import soft_update


@dataclass
class CQLConfig:
    obs_dim: int
    act_dim: int
    action_scale: float = 1.0
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    cql_alpha: float = 1.0       # weight on conservatism penalty
    cql_n_samples: int = 10      # actions sampled per state for logsumexp estimate
    target_entropy: float = None  # default: -act_dim
    hidden: tuple = (256, 256)


class CQLAgent:
    def __init__(self, cfg: CQLConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

        self.actor = TanhGaussianActor(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.action_scale).to(device)
        self.q = TwinQ(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(device)
        self.q_target = TwinQ(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
        self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(cfg.act_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _sample_actions_with_logp(self, obs, n: int):
        """Repeat obs n times along batch dim, sample policy actions + log probs."""
        b = obs.shape[0]
        obs_rep = obs.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
        act, logp = self.actor.sample(obs_rep)
        act = act.view(b, n, -1)
        logp = logp.view(b, n, 1)
        return act, logp, obs_rep.view(b, n, -1)

    def _q_for_actions(self, obs_rep, actions):
        """Compute (Q1, Q2) for batched obs and a batch of actions per obs.
        obs_rep: (B, N, obs_dim)   actions: (B, N, act_dim)
        returns: q1, q2 of shape (B, N, 1)
        """
        b, n, _ = actions.shape
        flat_obs = obs_rep.reshape(b * n, -1)
        flat_act = actions.reshape(b * n, -1)
        q1, q2 = self.q(flat_obs, flat_act)
        return q1.view(b, n, 1), q2.view(b, n, 1)

    def update_critic(self, batch):
        cfg = self.cfg
        obs, act, rew, next_obs, done = (
            batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["done"],
        )
        b = obs.shape[0]

        # ------- standard SAC TD target -------
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
            tq1, tq2 = self.q_target(next_obs, next_act)
            target_q = torch.min(tq1, tq2) - self.alpha * next_logp
            target = rew + cfg.discount * (1.0 - done) * target_q

        q1, q2 = self.q(obs, act)
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # ------- conservative penalty: log sum exp Q(s, a') - Q(s, a) -------
        n = cfg.cql_n_samples

        # 1) actions ~ Uniform[-A, A]
        unif_act = (torch.rand(b, n, cfg.act_dim, device=self.device) * 2.0 - 1.0) * cfg.action_scale
        unif_logp = math.log(0.5 / cfg.action_scale) * cfg.act_dim  # log density of uniform
        # 2) actions ~ pi(.|s)
        pi_act, pi_logp, obs_rep = self._sample_actions_with_logp(obs, n)
        # 3) actions ~ pi(.|s')
        pi_next_act, pi_next_logp, _ = self._sample_actions_with_logp(next_obs, n)

        obs_rep_for_unif = obs.unsqueeze(1).expand(-1, n, -1)

        q1_u, q2_u = self._q_for_actions(obs_rep_for_unif, unif_act)
        q1_p, q2_p = self._q_for_actions(obs_rep, pi_act)
        q1_pn, q2_pn = self._q_for_actions(obs_rep, pi_next_act)

        def cat_logsumexp(q_u, q_p, q_pn):
            # Importance-corrected logsumexp over the three sources (CQL eq. 4).
            cat = torch.cat(
                [q_u - unif_logp, q_p - pi_logp.detach(), q_pn - pi_next_logp.detach()], dim=1
            )
            return torch.logsumexp(cat, dim=1)  # (B, 1)

        lse1 = cat_logsumexp(q1_u, q1_p, q1_pn)
        lse2 = cat_logsumexp(q2_u, q2_p, q2_pn)

        cql_loss = ((lse1 - q1).mean() + (lse2 - q2).mean()) * cfg.cql_alpha

        loss = td_loss + cql_loss

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

        return {
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "q_mean": ((q1 + q2) / 2).mean().item(),
            "q_ood_mean": ((q1_p + q2_p) / 2).mean().item(),
        }

    def update_actor_and_alpha(self, batch):
        obs = batch["obs"]
        act, logp = self.actor.sample(obs)
        q1, q2 = self.q(obs, act)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * logp - q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -logp.mean().item(),
        }

    def update(self, batch):
        info = {}
        info.update(self.update_critic(batch))
        info.update(self.update_actor_and_alpha(batch))
        soft_update(self.q_target, self.q, self.cfg.tau)
        return info
