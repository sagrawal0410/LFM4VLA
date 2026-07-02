"""Experiment path layout, loggers (wandb), and run naming."""

from __future__ import annotations

import copy
import datetime
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger


def _build_exp_name(variant: Dict[str, Any]) -> str:
    act = variant.get("act_head") or {}
    bs = variant.get("batch_size", 1)
    accum = variant.get("trainer", {}).get("accumulate_grad_batches", 1)
    parts = [
        variant.get("task_name", "run"),
        variant.get("model", "model"),
        f"bs{bs}x{accum}",
        f"lr{variant.get('learning_rate', 0)}",
        f"ws{variant.get('window_size', 1)}",
        f"{act.get('type', 'head')}",
        f"lat{act.get('latent', 1)}",
    ]
    setup = variant.get("train_setup", {})
    if not setup.get("train_vision", True):
        parts.append("freeze_vision")
    if not setup.get("train_text_embedding", True):
        parts.append("freeze_textemb")
    return "-".join(str(p) for p in parts)


def _new_run_id() -> str:
    """Unique, sortable id so identical configs never share ckpt/wandb dirs."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def prepare_experiment(variant: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str, str]:
    """Set seed, create log/checkpoint dirs, attach paths to variant."""
    variant = copy.deepcopy(variant)
    seed = variant.get("seed", 42)
    rank = int(os.environ.get("RANK", "0"))
    seed_everything(seed + rank, workers=True)

    base_name = _build_exp_name(variant)
    run_id = _new_run_id()
    exp_name = f"{base_name}-{run_id}"
    date_str = str(datetime.date.today())
    log_dir = Path(variant["log_root"]) / date_str / exp_name
    ckpt_dir = Path(variant["output_root"]) / date_str / exp_name
    cache_dir = Path(variant.get("cache_root", "runs/cache"))

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    variant["log_dir"] = log_dir.as_posix()
    variant["output_dir"] = ckpt_dir.as_posix()
    variant["cache_root"] = cache_dir.as_posix()
    variant["exp_base_name"] = base_name
    variant["run_id"] = run_id
    variant["exp_name"] = exp_name

    return variant, exp_name, log_dir.as_posix(), ckpt_dir.as_posix()


def build_loggers(
    variant: Dict[str, Any],
    exp_name: str,
    log_dir: str,
) -> Union[bool, List[Logger]]:
    logger_cfg = variant.get("trainer", {}).get("logger", True)

    if logger_cfg is True:
        return True
    if logger_cfg is False:
        return False
    if not isinstance(logger_cfg, list):
        raise ValueError(f"trainer.logger must be true, false, or a list; got {logger_cfg!r}")

    loggers: List[Logger] = []
    for name in logger_cfg:
        if name == "tensorboard":
            loggers.append(TensorBoardLogger(save_dir=log_dir, name=exp_name))
        elif name == "csv":
            loggers.append(CSVLogger(save_dir=log_dir, name=exp_name))
        elif name == "wandb":
            loggers.append(
                WandbLogger(
                    project=variant.get("wandb_project", "lfm4vla"),
                    name=exp_name,
                    id=variant.get("run_id"),
                    save_dir=log_dir,
                    config=variant,
                )
            )
        else:
            raise ValueError(f"Unknown logger {name!r}; use tensorboard, csv, or wandb")

    return loggers
