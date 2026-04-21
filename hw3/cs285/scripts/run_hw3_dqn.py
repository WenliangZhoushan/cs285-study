import time
import argparse
from collections import deque

from cs285.agents.dqn_agent import DQNAgent
import cs285.env_configs

import os
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

from scripting_utils import make_logger, make_config

MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps

    observation = None

    # Replay buffer
    if len(env.observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(
            frame_history_len=frame_history_len
        )
    elif len(env.observation_space.shape) == 1:
        stacked_frames = False
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    reset_env_training()

    # Training bookkeeping (rolling stats for nicer wandb plots)
    recent_returns = deque(maxlen=100)
    recent_ep_lens = deque(maxlen=100)
    num_episodes = 0
    start_time = time.time()
    last_sps_step = 0
    last_sps_time = start_time

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        epsilon = exploration_schedule.value(step)

        action = agent.get_action(observation, epsilon)
        next_observation, reward, done, info = env.step(action)

        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.insert(
                action=action,
                reward=reward,
                next_observation=next_observation[-1, ...],
                done=done and not truncated,
            )
        else:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done and not truncated,
            )

        if done:
            reset_env_training()
            ep_return = float(info["episode"]["r"])
            ep_length = float(info["episode"]["l"])
            recent_returns.append(ep_return)
            recent_ep_lens.append(ep_length)
            num_episodes += 1

            logger.log_scalar(ep_return, "train_return", step)
            logger.log_scalar(ep_length, "train_ep_len", step)
            logger.log_scalar(num_episodes, "num_episodes", step)
            logger.log_scalar(np.mean(recent_returns), "train_return_avg100", step)
            logger.log_scalar(np.mean(recent_ep_lens), "train_ep_len_avg100", step)
        else:
            observation = next_observation

        # Main DQN training loop
        if step >= config["learning_starts"]:
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)
            update_info = agent.update(**batch, step=step)

            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                # Extra diagnostics
                logger.log_scalar(len(replay_buffer), "replay_buffer_size", step)
                now = time.time()
                sps = (step - last_sps_step) / max(now - last_sps_time, 1e-6)
                logger.log_scalar(sps, "steps_per_sec", step)
                logger.log_scalar(now - start_time, "wall_time_sec", step)
                last_sps_step = step
                last_sps_time = now
                logger.flush()

        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )

    logger.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=7)
    parser.add_argument("--log_interval", type=int, default=1000)

    # Weights & Biases integration
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs285-hw3-dqn")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Optional group name for organising related runs.")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config, args=args, config_file=args.config_file)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
