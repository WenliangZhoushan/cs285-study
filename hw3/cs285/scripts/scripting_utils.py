import yaml
import os
import time
import argparse
import shutil
from typing import Optional

import cs285.env_configs
from cs285.infrastructure.logger import Logger


def _make_serializable(obj):
    """Best-effort conversion of a config dict into something wandb can serialize."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if callable(obj):
        return f"<callable {getattr(obj, '__name__', repr(obj))}>"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    base_config_name = config_kwargs.pop("base_config")
    return cs285.env_configs.configs[base_config_name](**config_kwargs)


def _read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def make_logger(
    logdir_prefix: str,
    config: dict,
    args: Optional[argparse.Namespace] = None,
    config_file: Optional[str] = None,
) -> Logger:
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    run_name = logdir_prefix + config["log_name"]
    if args is not None and getattr(args, "seed", None) is not None:
        run_name += f"_seed{args.seed}"
    run_name_ts = run_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, run_name_ts)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    # Save a copy of the raw yaml config alongside the logs (helpful for reproducibility).
    raw_cfg = {}
    if config_file is not None and os.path.exists(config_file):
        try:
            shutil.copy(config_file, os.path.join(logdir, os.path.basename(config_file)))
            raw_cfg = _read_yaml(config_file) or {}
        except Exception as e:
            print(f"[make_logger] Could not copy config file: {e}")

    logger = Logger(logdir)

    # Optional wandb init
    if args is not None and getattr(args, "use_wandb", False):
        wandb_config = {
            "seed": getattr(args, "seed", None),
            "algo": logdir_prefix.strip("_"),
            "log_name": config.get("log_name"),
            "config_file": config_file,
            "raw_config": raw_cfg,
            # serializable view of the fully-resolved config (drops callables)
            "resolved_config": _make_serializable(config),
        }
        group = getattr(args, "wandb_group", None) or config.get("log_name")
        tags = [
            logdir_prefix.strip("_"),
            raw_cfg.get("env_name", "unknown_env"),
        ]
        logger.init_wandb(
            project=getattr(args, "wandb_project", "cs285-hw3"),
            entity=getattr(args, "wandb_entity", None),
            run_name=run_name_ts,
            config=wandb_config,
            group=group,
            tags=tags,
            mode=getattr(args, "wandb_mode", "online"),
        )

    return logger
