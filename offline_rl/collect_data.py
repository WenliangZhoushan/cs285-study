"""Generate a small offline dataset for Pendulum-v1.

Uses a simple energy-based swing-up controller as a "noisy expert" plus some
random rollouts, so IQL/CQL have a mix of high- and low-quality transitions.

Pendulum-v1 obs = [cos(theta), sin(theta), theta_dot]; action in [-2, 2].
"""
import argparse
import os

import gymnasium as gym
import numpy as np


def heuristic_action(obs: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Energy-pumping swing-up + PD stabilization near the top.

    Pendulum-v1 obs convention: cos_t = 1 at upright, -1 hanging.
    To pump energy, apply torque in the direction of motion (dE/dt = u * theta_dot).
    """
    cos_t, sin_t, theta_dot = obs
    theta = np.arctan2(sin_t, cos_t)
    if cos_t > 0.85 and abs(theta_dot) < 4.0:
        a = -8.0 * theta - 1.0 * theta_dot
    else:
        a = 2.0 * np.sign(theta_dot) if theta_dot != 0 else 2.0
    a = a + rng.normal(0.0, noise)
    return np.clip(np.array([a], dtype=np.float32), -2.0, 2.0)


def collect(n_transitions: int, expert_frac: float, expert_noise: float, seed: int, out_path: str):
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)

    obs_buf, act_buf, rew_buf, next_obs_buf, done_buf = [], [], [], [], []
    n_expert = int(n_transitions * expert_frac)
    n_random = n_transitions - n_expert

    def rollout(n_steps: int, policy):
        nonlocal obs_buf, act_buf, rew_buf, next_obs_buf, done_buf
        steps = 0
        ep_ret = 0.0
        ep_returns = []
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        while steps < n_steps:
            a = policy(obs)
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            obs_buf.append(obs)
            act_buf.append(a)
            rew_buf.append([r])
            next_obs_buf.append(next_obs)
            # We treat truncation as non-terminal for bootstrapping (Pendulum has no real terminal).
            done_buf.append([float(terminated)])
            ep_ret += r
            steps += 1
            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            else:
                obs = next_obs
        return ep_returns

    expert_returns = rollout(n_expert, lambda o: heuristic_action(o, expert_noise, rng))
    random_returns = rollout(n_random, lambda o: env.action_space.sample())

    data = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "act": np.asarray(act_buf, dtype=np.float32),
        "rew": np.asarray(rew_buf, dtype=np.float32),
        "next_obs": np.asarray(next_obs_buf, dtype=np.float32),
        "done": np.asarray(done_buf, dtype=np.float32),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **data)

    print(f"Saved {data['obs'].shape[0]} transitions -> {out_path}")
    if expert_returns:
        print(f"  expert  episodes: {len(expert_returns)}  mean return = {np.mean(expert_returns):.1f}")
    if random_returns:
        print(f"  random  episodes: {len(random_returns)}  mean return = {np.mean(random_returns):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--expert-frac", type=float, default=0.7)
    parser.add_argument("--expert-noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/pendulum_mixed.npz")
    args = parser.parse_args()
    collect(args.n, args.expert_frac, args.expert_noise, args.seed, args.out)
