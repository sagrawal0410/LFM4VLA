#!/usr/bin/env python3
"""Plot expert vs predicted action chunks on CALVIN val batches (offline, no sim)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.build_dataset import build_dataset
from models.model_backbone import load_config
from train.base_trainer import BaseTrainer

ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


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
    """Collapse batch/window dims → [T, 6] and [T]."""
    arm = arm.reshape(-1, arm.shape[-1])
    grip = grip.reshape(-1)
    return arm, grip


def _denorm_arm(arm: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    """Invert training normalize_action for arm dims (still in [-1, 1] input)."""
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


def main():
    parser = argparse.ArgumentParser(description="Compare expert vs predicted actions offline")
    parser.add_argument("--config", type=str, required=True, help="Path to run config JSON")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="runs/action_compare")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of val batches to plot")
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
    trainer = BaseTrainer.from_checkpoint(args.ckpt, configs=configs)
    trainer.eval()
    trainer.to(device)

    val_dataset = build_dataset(configs["val_dataset"], configs, trainer.model)
    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=val_dataset.collater,
        num_workers=0,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_min = float(configs.get("norm_min", -0.65))
    norm_max = float(configs.get("norm_max", 0.65))
    all_mae = []

    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break

        with torch.no_grad():
            pred = trainer.inference_step(batch)
            inputs = trainer._process_batch(batch)

        pred_arm, pred_grip = _extract_pred_arm_grip(pred["action"])
        gt_arm, gt_grip = _extract_gt(inputs)
        pred_arm, pred_grip = _flatten_chunk(pred_arm, pred_grip)
        gt_arm, gt_grip = _flatten_chunk(gt_arm, gt_grip)

        raw_text = batch.get("raw_text", ["?"])
        instruction = raw_text[0] if isinstance(raw_text, list) else str(raw_text)

        mae = _plot_sample(
            gt_arm,
            gt_grip,
            pred_arm,
            pred_grip,
            instruction,
            out_dir / f"sample_{i:03d}.png",
            denorm=args.denorm,
            norm_min=norm_min,
            norm_max=norm_max,
        )
        mae["instruction"] = instruction
        mae["sample_idx"] = i
        all_mae.append(mae)
        print(f"[{i}] saved {out_dir / f'sample_{i:03d}.png'}  MAE gripper={mae['gripper']:.4f}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_mae, f, indent=2)
    print(f"Wrote {len(all_mae)} plots to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
