"""Minimal replay buffer for offline data."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device="cpu"):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.idx = 0
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.idx] = obs
        self.act[self.idx] = act
        self.rew[self.idx] = rew
        self.next_obs[self.idx] = next_obs
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def load(self, data: dict):
        n = len(data["obs"])
        assert n <= self.capacity
        self.obs[:n] = data["obs"]
        self.act[:n] = data["act"]
        self.rew[:n] = data["rew"]
        self.next_obs[:n] = data["next_obs"]
        self.done[:n] = data["done"]
        self.size = n
        self.idx = n % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x, device=self.device)
        return {
            "obs": to_t(self.obs[idx]),
            "act": to_t(self.act[idx]),
            "rew": to_t(self.rew[idx]),
            "next_obs": to_t(self.next_obs[idx]),
            "done": to_t(self.done[idx]),
        }
