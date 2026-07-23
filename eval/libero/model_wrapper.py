"""Closed-loop LIBERO policy wrapper for LFM4VLA checkpoints.

Adapts a trained ``RoboLFM2.5`` policy (via ``BaseTrainer.inference_step``) to LIBERO's
robosuite/MuJoCo control loop: ``reset()`` then ``step(image, instruction)`` returning a
7-D LIBERO action ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``.

Action contract (must mirror training exactly — see data/libero_dataset.py and the OXE
RLDS pipeline):

* Arm (dims 0-5): the RLDS pipeline normalized these to ``[-1, 1]`` with BOUNDS_Q99
  (``action_normalization_mask = [True]*6 + [False]``). The policy therefore emits
  normalized arm deltas; we **denormalize** with the dataset's q01/q99 before sending
  to the sim.
* Gripper (dim 6): NOT normalized by RLDS. ``libero_dataset_transform`` maps it to
  ``{1 = open, 0 = close}`` and the collater binarizes it, so the head's sigmoid output
  is ``P(open)``. robosuite's OSC gripper wants ``+1 = close, -1 = open``, so we send
  ``-1`` when open and ``+1`` when closed (matches OpenVLA's normalize+invert).

Only ``window_size=1`` / ``history_type="post"`` is supported (LFM's current config).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.base_trainer import BaseTrainer


def load_action_stats(data_root_dir: str, dataset_name: str) -> dict:
    """Load q01/q99 action bounds saved by the RLDS pipeline for denormalization.

    The stats file is written as ``dataset_statistics_<hash>.json`` under
    ``<data_root_dir>/<dataset_name>/<version>/``. We glob for it so the caller does
    not need to know the content hash.
    """
    patterns = [
        os.path.join(data_root_dir, dataset_name, "**", "dataset_statistics*.json"),
        os.path.join(data_root_dir, dataset_name, "dataset_statistics*.json"),
    ]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No dataset_statistics*.json under {data_root_dir}/{dataset_name}. "
            "Needed to denormalize arm actions at eval. Point --data_root_dir at the "
            "same RLDS directory used for training."
        )
    with open(sorted(matches)[0]) as f:
        stats = json.load(f)
    action = stats["action"] if "action" in stats else stats
    return {
        "q01": np.asarray(action["q01"], dtype=np.float32),
        "q99": np.asarray(action["q99"], dtype=np.float32),
        "path": sorted(matches)[0],
    }


class LFMLiberoModel:
    """LIBERO-compatible wrapper around an LFM4VLA policy."""

    def __init__(
        self,
        ckpt_path,
        configs,
        action_stats: dict,
        device: str = "cuda:0",
        image_size: int = 224,
        image_transform: str = "rot180",
        gripper_open_is_negative: bool = True,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
    ):
        self.configs = configs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.trainer = BaseTrainer.from_checkpoint(str(ckpt_path), configs=configs)
        self.trainer.eval()
        self.trainer.to(self.device)

        self.window_size = configs["window_size"]
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]
        self.image_size = image_size
        self.image_transform = image_transform
        self.gripper_open_is_negative = gripper_open_is_negative

        self.q01 = action_stats["q01"]
        self.q99 = action_stats["q99"]
        # Model emits arm in [norm_min, norm_max]; RLDS used [-1, 1].
        self.norm_min = norm_min
        self.norm_max = norm_max

        self._chunk_buffer: list[np.ndarray] = []
        self._steps_since_replan = 0
        self.emitted_actions: list[np.ndarray] = []
        self._inference_calls = 0

    def reset(self):
        self._chunk_buffer = []
        self._steps_since_replan = 0
        self.emitted_actions = []

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------
    def step(self, image: np.ndarray, instruction: str, execute_step: int = 1) -> np.ndarray:
        """Return one 7-D LIBERO action for the current agentview frame.

        ``execute_step == 1`` re-queries the policy every env step (closed-loop). Larger
        values replay that many actions from the predicted chunk before re-querying
        (open-loop chunking: faster, less reactive).
        """
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

        self.emitted_actions.append(action.copy())
        return self._to_libero_action(action)

    def action_stats(self) -> dict:
        """Per-dim std of raw (normalized) emitted actions — flags a frozen policy."""
        if not self.emitted_actions:
            return {}
        arr = np.stack(self.emitted_actions, axis=0)  # [T, 7]
        gripper_open = arr[:, 6] > 0.5
        return {
            "num_steps": int(arr.shape[0]),
            "arm_std_per_dim": [round(float(s), 4) for s in arr[:, :6].std(axis=0)],
            "arm_mean_per_dim": [round(float(m), 4) for m in arr[:, :6].mean(axis=0)],
            "gripper_open_frac": round(float(gripper_open.mean()), 3),
            "gripper_switches": int(np.count_nonzero(np.diff(gripper_open))),
        }

    # ------------------------------------------------------------------
    # Inference + post-processing
    # ------------------------------------------------------------------
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # robosuite renders upside-down; the RLDS demos were stored rotated 180 (OpenVLA).
        # Match that geometry so eval frames land in the training distribution.
        if self.image_transform == "rot180":
            img = img[::-1, ::-1]
        elif self.image_transform == "flip_vertical":
            img = img[::-1]
        elif self.image_transform == "none":
            pass
        else:
            raise ValueError(f"Unknown image_transform: {self.image_transform}")
        return np.ascontiguousarray(img)

    def _predict_chunk(self, image: np.ndarray, instruction: str) -> np.ndarray:
        img = self._preprocess_image(image)
        pil = Image.fromarray(img).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        rgb = self.trainer.model.image_processor(pil)  # [C, H, W] float in [0, 255]
        rgb = rgb.unsqueeze(0).unsqueeze(0)  # [B=1, T=1, C, H, W]

        batch = {"rgb": rgb, "text": [instruction]}
        with torch.no_grad():
            pred = self.trainer.inference_step(batch)["action"]

        self._inference_calls += 1
        if self.device.type == "cuda" and self._inference_calls % 100 == 0:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        if isinstance(pred, (tuple, list)):
            arm, grip = pred
            grip = torch.sigmoid(grip)  # head emits logits; -> P(open)
            if grip.ndim == arm.ndim - 1:
                grip = grip.unsqueeze(-1)
            action = torch.cat([arm, grip], dim=-1)
        else:
            action = pred

        action = action.detach().float().cpu().reshape(-1, action.shape[-1])  # [chunk, 7]
        return action.numpy()

    def _to_libero_action(self, action: np.ndarray) -> np.ndarray:
        action = action.astype(np.float32).copy()

        # Arm: denormalize [norm_min, norm_max] -> physical delta via q01/q99.
        arm = action[:6]
        arm = (arm - self.norm_min) / (self.norm_max - self.norm_min)  # -> [0, 1]
        arm = arm * (self.q99[:6] - self.q01[:6]) + self.q01[:6]
        out = np.empty(7, dtype=np.float32)
        out[:6] = arm

        # Gripper: P(open) -> robosuite command. Default: open -> -1, close -> +1.
        prob_open = action[6]
        if self.gripper_open_is_negative:
            out[6] = -1.0 if prob_open > 0.5 else 1.0
        else:
            out[6] = 1.0 if prob_open > 0.5 else -1.0
        return out
