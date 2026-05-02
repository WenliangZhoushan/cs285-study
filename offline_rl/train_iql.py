"""Train IQL on an offline Pendulum-v1 dataset."""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from offline_rl.iql import IQLAgent, IQLConfig
from offline_rl.replay_buffer import ReplayBuffer
from offline_rl.utils import evaluate, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="offline_rl/data/pendulum_mixed.npz")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    # IQL hyperparameters (paper defaults for D4RL Mujoco)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    data = dict(np.load(args.data))
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=data["obs"].shape[0], device=args.device)
    buffer.load(data)
    print(f"Loaded {buffer.size} transitions from {args.data}")

    cfg = IQLConfig(
        obs_dim=obs_dim, act_dim=act_dim, action_scale=action_scale,
        expectile=args.expectile, beta=args.beta,
    )
    agent = IQLAgent(cfg, device=args.device)

    start = time.time()
    for step in range(1, args.steps + 1):
        batch = buffer.sample(args.batch_size)
        info = agent.update(batch)

        if step % args.eval_every == 0 or step == args.steps:
            eval_ret = evaluate(env, agent.actor, args.eval_episodes, device=args.device)
            elapsed = time.time() - start
            print(
                f"step {step:6d} | eval_return {eval_ret:7.2f} | "
                f"q {info['q_mean']:.2f} v {info['v_mean']:.2f} "
                f"adv {info['adv_mean']:.2f} w {info['weight_mean']:.2f} | "
                f"{elapsed:.1f}s"
            )


if __name__ == "__main__":
    main()
