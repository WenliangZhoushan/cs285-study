"""Eval + misc helpers."""
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1 - tau) + sp.data * tau)


@torch.no_grad()
def evaluate(env, actor, n_episodes: int = 10, device: str = "cpu") -> float:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor.act(obs_t, deterministic=True).cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))
