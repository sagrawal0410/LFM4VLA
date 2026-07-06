"""Closed-loop CALVIN policy wrapper for LFM4VLA checkpoints.

Adapts a trained ``RoboLFM2.5`` policy (via ``BaseTrainer.inference_step``) to the
``CalvinBaseModel`` interface expected by calvin_agent's rollout loop: ``reset()`` and
``step(obs, goal)`` returning a 7-D CALVIN relative action ``[dx,dy,dz,droll,dpitch,dyaw,gripper]``.

Only ``window_size=1`` / ``history_type="post"`` is supported (LFM's current config).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.calvin.obs_utils import obs_to_uint8_rgb
from train.base_trainer import BaseTrainer


class LFMCalvinModel:
    """CalvinBaseModel-compatible wrapper around an LFM4VLA policy."""

    def __init__(self, ckpt_path, configs, device="cuda:0"):
        self.configs = configs
        self.device = torch.device(device)

        precision = configs.get("trainer", {}).get("precision", "bf16-mixed")
        self.dtype = torch.bfloat16 if "bf16" in str(precision) else torch.float32

        self.trainer = BaseTrainer.from_checkpoint(str(ckpt_path), configs=configs)
        self.trainer.eval()
        self.trainer.to(self.device)

        self.window_size = configs["window_size"]
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]
        self.action_space = configs["act_head"].get("action_space", "continuous")

        self.norm_action = configs.get("norm_action", False)
        self.norm_min = float(configs.get("norm_min", -1.0))
        self.norm_max = float(configs.get("norm_max", 1.0))

        # Buffer of remaining actions in the current predicted chunk.
        self._chunk_buffer: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # CalvinBaseModel interface
    # ------------------------------------------------------------------
    def reset(self):
        self._chunk_buffer = []

    def step(self, obs, goal, execute_step: int = 10):
        """Return a single 7-D CALVIN action for the current observation.

        ``execute_step`` controls open-loop chunk execution: predict a fresh chunk,
        then play back that many steps before re-querying the model. ``execute_step=1``
        re-plans every env step (slow, most reactive); ``execute_step=fwd_pred_next_n``
        plays the whole chunk (fast).
        """
        assert 1 <= execute_step <= self.fwd_pred_next_n

        if not self._chunk_buffer:
            chunk = self._predict_chunk(obs, goal)  # [fwd_pred_next_n, 7], normalized arm + prob gripper
            self._chunk_buffer = list(chunk[:execute_step])

        action = self._chunk_buffer.pop(0)
        return self._to_calvin_action(action)

    # ------------------------------------------------------------------
    # Inference + post-processing
    # ------------------------------------------------------------------
    def _predict_chunk(self, obs, goal) -> torch.Tensor:
        image = Image.fromarray(obs_to_uint8_rgb(obs))
        rgb = self.trainer.model.image_processor(image)  # [C, H, W] float in [0, 255]
        rgb = rgb.unsqueeze(0).unsqueeze(0)  # [B=1, T=1, C, H, W]

        batch = {"rgb": rgb, "text": [goal]}

        with torch.no_grad():
            pred = self.trainer.inference_step(batch)["action"]

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if isinstance(pred, (tuple, list)):
            arm, grip = pred
            grip = torch.sigmoid(grip)  # head emits logits; map to open probability
            if grip.ndim == arm.ndim - 1:
                grip = grip.unsqueeze(-1)
            action = torch.cat([arm, grip], dim=-1)
        else:
            action = pred

        action = action.detach().float().cpu().reshape(-1, action.shape[-1])  # [fwd_pred_next_n, 7]
        return action

    def _to_calvin_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.clone()

        if self.norm_action:
            arm = action[:6]
            arm = 0.5 * (arm + 1.0) * (self.norm_max - self.norm_min) + self.norm_min
            action[:6] = arm

        # Gripper: probability in [0, 1] -> CALVIN {-1 (close), +1 (open)}.
        action[6] = 1.0 if action[6] > 0.5 else -1.0

        return action.numpy()
