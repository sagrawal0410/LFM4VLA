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

        # Remaining actions in the current predicted chunk (only used if execute_step > 1).
        self._chunk_buffer: list[torch.Tensor] = []
        self._steps_since_replan = 0
        # Diagnostics: raw normalized actions the policy emitted this episode.
        self.emitted_actions: list[np.ndarray] = []
        self._inference_calls = 0

    # ------------------------------------------------------------------
    # CalvinBaseModel interface
    # ------------------------------------------------------------------
    def reset(self):
        self._chunk_buffer = []
        self._steps_since_replan = 0
        self.emitted_actions = []

    def step(self, obs, goal, execute_step: int = 1):
        """Return one 7-D CALVIN action for the *current* observation.

        Closed-loop control: with ``execute_step == 1`` (default) the model is queried
        on every env step using the latest camera frame, so each control cycle gets a
        fresh, image-conditioned action. ``execute_step > 1`` replays that many actions
        from a predicted chunk before re-querying (open-loop chunking, less reactive).
        """
        assert 1 <= execute_step <= self.fwd_pred_next_n

        if execute_step == 1:
            # Always re-query the policy with the newest observation.
            chunk = self._predict_chunk(obs, goal)
            action = chunk[0]
        else:
            if not self._chunk_buffer or self._steps_since_replan >= execute_step:
                chunk = self._predict_chunk(obs, goal)
                self._chunk_buffer = list(chunk[:execute_step])
                self._steps_since_replan = 0
            action = self._chunk_buffer.pop(0)
            self._steps_since_replan += 1

        self.emitted_actions.append(action.detach().cpu().numpy().copy())
        return self._to_calvin_action(action)

    def action_stats(self) -> dict:
        """Per-dimension std of the raw normalized actions emitted this episode.

        If std ~ 0 the policy is emitting a constant action (frozen / not reacting to
        the image); healthy closed-loop control should show non-trivial variation.
        """
        if not self.emitted_actions:
            return {}
        arr = np.stack(self.emitted_actions, axis=0)  # [T, 7]
        gripper_open = arr[:, 6] > 0.5  # binarized command actually sent to the env
        return {
            "num_steps": int(arr.shape[0]),
            "arm_std_per_dim": [round(float(s), 4) for s in arr[:, :6].std(axis=0)],
            "arm_mean_per_dim": [round(float(m), 4) for m in arr[:, :6].mean(axis=0)],
            "gripper_mean": round(float(arr[:, 6].mean()), 4),
            "gripper_open_frac": round(float(gripper_open.mean()), 3),
            "gripper_switches": int(np.count_nonzero(np.diff(gripper_open))),
        }

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

        # At execute_step=1 this runs every env step (~350+ calls per sequence).
        # Without periodic cache release, allocator growth eventually aborts the
        # process (native SIGABRT, no Python traceback). Cheap at this frequency.
        self._inference_calls += 1
        if self.device.type == "cuda" and self._inference_calls % 50 == 0:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated(self.device) / 1e9
            reserv = torch.cuda.memory_reserved(self.device) / 1e9
            print(f"  [mem] inference #{self._inference_calls}: "
                  f"cuda alloc={alloc:.2f}GB reserved={reserv:.2f}GB", flush=True)

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
