"""LFM4VLA policy wrapper for Humanoid Everyday / G1 MuJoCo eval.

Loads a checkpoint, runs ``inference_step`` on egocentric RGB + language, and
returns denormalized 28-D absolute joint targets (q01/q99).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.base_trainer import BaseTrainer  # noqa: E402


def load_he_action_stats(data_root_dir: str, robot_type: str = "g1") -> dict:
    """Load q01/q99 written by ``HumanoidEverydayDataset`` under ``meta/``."""
    patterns = [
        os.path.join(data_root_dir, "meta", f"action_stats_{robot_type}_*.json"),
        os.path.join(data_root_dir, "meta", "action_stats_*.json"),
    ]
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    if not matches:
        raise FileNotFoundError(
            f"No action_stats_{robot_type}_*.json under {data_root_dir}/meta/. "
            "Train once (or run the dataset once) so stats are cached, or copy "
            "them from the training node."
        )
    path = sorted(matches)[0]
    with open(path) as f:
        stats = json.load(f)
    return {
        "q01": np.asarray(stats["q01"], dtype=np.float32),
        "q99": np.asarray(stats["q99"], dtype=np.float32),
        "path": path,
    }


def resolve_ckpt(ckpt: str | Path) -> Path:
    """Prefer ``last.ckpt``, else newest ``*.ckpt`` under a directory."""
    p = Path(ckpt)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"checkpoint path not found: {ckpt}")
    last = p / "last.ckpt"
    if last.is_file():
        return last
    ckpts = sorted(p.glob("*.ckpt"), key=lambda x: x.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"no *.ckpt under {p}")
    return ckpts[-1]


def apply_he_layout_overrides(
    configs: dict,
    num_action_tokens: Optional[int] = None,
    latent: Optional[int] = None,
) -> dict:
    """Force FCContinuousDecoder + optional token/latent overrides."""
    cfg = json.loads(json.dumps(configs))  # deep copy
    head = cfg.setdefault("act_head", {})
    head["type"] = "FCContinuousDecoder"
    head["action_dim"] = 28
    head["down_sample"] = "none"
    if num_action_tokens is not None:
        head["num_action_tokens"] = int(num_action_tokens)
    if latent is not None:
        head["latent"] = int(latent)
    return cfg


class LFMHumanoidModel:
    """Closed-loop wrapper: egocentric RGB + instruction → 28-D joint targets."""

    def __init__(
        self,
        ckpt_path: str | Path,
        configs: dict,
        action_stats: dict,
        device: str = "cuda:0",
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        keep_native_resolution: bool = True,
        image_size: int = 480,
    ):
        self.configs = configs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.trainer = BaseTrainer.from_checkpoint(str(ckpt_path), configs=configs)
        self.trainer.eval()
        self.trainer.to(self.device)

        self.window_size = configs["window_size"]
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]
        self.norm_min = float(configs.get("norm_min", norm_min))
        self.norm_max = float(configs.get("norm_max", norm_max))
        self.keep_native_resolution = keep_native_resolution
        self.image_size = image_size

        self.q01 = np.asarray(action_stats["q01"], dtype=np.float32).reshape(-1)
        self.q99 = np.asarray(action_stats["q99"], dtype=np.float32).reshape(-1)
        if self.q01.shape[0] != 28 or self.q99.shape[0] != 28:
            raise ValueError(
                f"action stats must be 28-D, got q01={self.q01.shape} q99={self.q99.shape} "
                f"from {action_stats.get('path')}"
            )

        self._chunk_buffer: List[np.ndarray] = []
        self._steps_since_replan = 0
        self.emitted_actions: List[np.ndarray] = []
        self._inference_calls = 0

    def reset(self) -> None:
        self._chunk_buffer = []
        self._steps_since_replan = 0
        self.emitted_actions = []

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        execute_step: int = 1,
    ) -> np.ndarray:
        """Return one denormalized 28-D absolute joint target."""
        assert 1 <= execute_step <= self.fwd_pred_next_n
        if execute_step == 1:
            chunk = self._predict_chunk(image, instruction)
            action = chunk[0]
        else:
            if not self._chunk_buffer or self._steps_since_replan >= execute_step:
                chunk = self._predict_chunk(image, instruction)
                self._chunk_buffer = list(chunk[:execute_step])
                self._steps_since_replan = 0
            action = self._chunk_buffer.pop(0)
            self._steps_since_replan += 1
        physical = self._denorm(action)
        self.emitted_actions.append(physical.copy())
        return physical

    def _denorm(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 28:
            raise ValueError(f"expected 28-D normalized action, got {a.shape}")
        unit = (a - self.norm_min) / (self.norm_max - self.norm_min)
        return unit * (self.q99 - self.q01) + self.q01

    def _predict_chunk(self, image: np.ndarray, instruction: str) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img).convert("RGB")
        if not self.keep_native_resolution:
            pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        rgb = self.trainer.model.image_processor(pil)  # [C,H,W] float [0,255]
        rgb = rgb.unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
        batch: Dict[str, Any] = {"rgb": rgb, "text": [instruction]}
        with torch.no_grad():
            pred = self.trainer.inference_step(batch)["action"]
        self._inference_calls += 1
        if isinstance(pred, (tuple, list)):
            # Should not happen for FCContinuousDecoder; keep defensive.
            pred = pred[0]
        action = pred.detach().float().cpu().reshape(-1, pred.shape[-1]).numpy()
        if action.shape[-1] != 28:
            raise RuntimeError(
                f"policy emitted action_dim={action.shape[-1]}, expected 28. "
                "Check act_head.type=FCContinuousDecoder and action_dim=28."
            )
        return action

    def action_stats_summary(self) -> dict:
        if not self.emitted_actions:
            return {}
        arr = np.stack(self.emitted_actions, axis=0)
        return {
            "num_steps": int(arr.shape[0]),
            "mean_abs": round(float(np.mean(np.abs(arr))), 4),
            "std_mean": round(float(arr.std(axis=0).mean()), 4),
        }
