#!/usr/bin/env python3
"""Offline expert vs predicted actions on random close_drawer val rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.build_dataset import build_dataset
from models.model_backbone import load_config
from train.base_trainer import BaseTrainer

ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
DEFAULT_TASK = "close_drawer"


def _resolve_ckpt_path(ckpt: str) -> Path:
    path = Path(ckpt).expanduser().resolve()
    if path.is_dir():
        for name in ("last.ckpt",):
            candidate = path / name
            if candidate.is_file():
                return candidate
        ckpts = sorted(path.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt file in directory: {path}")
        return ckpts[-1]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _extract_pred_arm_grip(pred_action) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pred_action, dict):
        pred_action = pred_action.get("action", pred_action)
    if isinstance(pred_action, (tuple, list)):
        arm, grip = pred_action
    else:
        arm = pred_action[..., :6]
        grip = pred_action[..., 6:]
    if isinstance(grip, torch.Tensor):
        grip = torch.sigmoid(grip)
    arm = _to_numpy(arm)
    grip = _to_numpy(grip)
    if grip.ndim > arm.ndim - 1:
        grip = grip.squeeze(-1)
    return arm, grip


def _extract_gt(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    arm = _to_numpy(inputs["arm_action_chunck"])
    grip = _to_numpy(inputs["gripper_action_chunck"])
    return arm, grip


def _flatten_chunk(arm: np.ndarray, grip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arm = arm.reshape(-1, arm.shape[-1])
    grip = grip.reshape(-1)
    return arm, grip


def _denorm_arm(arm: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    return 0.5 * (arm + 1.0) * (norm_max - norm_min) + norm_min


def _plot_sample(
    expert_arm: np.ndarray,
    expert_grip: np.ndarray,
    pred_arm: np.ndarray,
    pred_grip: np.ndarray,
    instruction: str,
    out_path: Path,
    denorm: bool,
    norm_min: float,
    norm_max: float,
) -> dict:
    if denorm:
        expert_arm = _denorm_arm(expert_arm, norm_min, norm_max)
        pred_arm = _denorm_arm(pred_arm, norm_min, norm_max)

    steps = np.arange(expert_arm.shape[0])
    expert = np.concatenate([expert_arm, expert_grip[:, None]], axis=-1)
    pred = np.concatenate([pred_arm, pred_grip[:, None]], axis=-1)

    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(instruction[:120], fontsize=10)
    mae = {}
    for dim, (ax, label) in enumerate(zip(axes, ACTION_LABELS)):
        ax.plot(steps, expert[:, dim], "g-", linewidth=2, label="expert")
        ax.plot(steps, pred[:, dim], "r--", linewidth=2, label="predicted")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        mae[label] = float(np.mean(np.abs(expert[:, dim] - pred[:, dim])))
        if dim == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("chunk step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return mae


def _sample_rollout_indices(dataset, task: str, num_rollouts: int, seed: int) -> list[tuple[int, int, str]]:
    """Pick one random timestep per lang segment for ``task``.

    Returns list of (dataset_idx, lang_segment_idx, instruction).
    """
    segments: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        lang_idx = int(dataset.lang_lookup[idx])
        if dataset.lang_task[lang_idx] != task:
            continue
        segments[lang_idx].append(idx)

    if not segments:
        raise ValueError(
            f"No val samples found for task '{task}'. "
            f"Available tasks: {sorted(set(dataset.lang_task))}"
        )

    rng = np.random.default_rng(seed)
    segment_ids = list(segments.keys())
    rng.shuffle(segment_ids)
    segment_ids = segment_ids[:num_rollouts]

    chosen: list[tuple[int, int, str]] = []
    for lang_idx in segment_ids:
        dataset_idx = int(rng.choice(segments[lang_idx]))
        instruction = dataset.lang_ann[lang_idx]
        chosen.append((dataset_idx, lang_idx, instruction))
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Compare expert vs predicted actions on random close_drawer rollouts",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to run config JSON")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning .ckpt file or checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="runs/action_compare/close_drawer")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Number of random lang segments to plot")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="CALVIN lang task name to filter on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--denorm",
        action="store_true",
        help="Plot arm dims in physical CALVIN rel_action units (inverse of norm_min/max)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    configs = load_config(args.config)
    ckpt_path = _resolve_ckpt_path(args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    trainer = BaseTrainer.from_checkpoint(str(ckpt_path), configs=configs)
    trainer.eval()
    trainer.to(device)

    val_cfg = copy.deepcopy(configs["val_dataset"])
    val_cfg.pop("exclude_tasks", None)
    val_dataset = build_dataset(val_cfg, configs, trainer.model)

    rollout_plan = _sample_rollout_indices(
        val_dataset, args.task, args.num_rollouts, args.seed
    )
    print(f"Selected {len(rollout_plan)} '{args.task}' rollouts from {len(val_dataset)} val samples")

    loader = DataLoader(
        Subset(val_dataset, [idx for idx, _, _ in rollout_plan]),
        batch_size=1,
        shuffle=False,
        collate_fn=val_dataset.collater,
        num_workers=0,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_min = float(configs.get("norm_min", -0.65))
    norm_max = float(configs.get("norm_max", 0.65))
    all_mae = []

    for i, batch in enumerate(loader):
        dataset_idx, lang_idx, instruction = rollout_plan[i]

        with torch.no_grad():
            pred = trainer.inference_step(batch)
            inputs = trainer._process_batch(batch)

        pred_arm, pred_grip = _extract_pred_arm_grip(pred["action"])
        gt_arm, gt_grip = _extract_gt(inputs)
        pred_arm, pred_grip = _flatten_chunk(pred_arm, pred_grip)
        gt_arm, gt_grip = _flatten_chunk(gt_arm, gt_grip)

        mae = _plot_sample(
            gt_arm,
            gt_grip,
            pred_arm,
            pred_grip,
            instruction,
            out_dir / f"rollout_{i:03d}.png",
            denorm=args.denorm,
            norm_min=norm_min,
            norm_max=norm_max,
        )
        mae["instruction"] = instruction
        mae["task"] = args.task
        mae["rollout_idx"] = i
        mae["dataset_idx"] = dataset_idx
        mae["lang_segment_idx"] = lang_idx
        mae["start_frame"] = int(val_dataset.episode_lookup[dataset_idx])
        all_mae.append(mae)
        print(
            f"[{i}] lang_seg={lang_idx} frame={mae['start_frame']}  "
            f"saved {out_dir / f'rollout_{i:03d}.png'}  MAE gripper={mae['gripper']:.4f}"
        )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_mae, f, indent=2)
    print(f"Wrote {len(all_mae)} plots to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
