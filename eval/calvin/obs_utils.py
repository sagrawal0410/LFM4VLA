"""Observation → RGB frame helpers for CALVIN sim eval."""

from __future__ import annotations

import copy

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def obs_to_uint8_rgb(obs, env=None) -> np.ndarray:
    """Convert a CALVIN ``rgb_static`` observation to uint8 HWC RGB for saving/display."""
    rgb = obs["rgb_obs"]["rgb_static"]
    return _to_uint8_rgb(rgb)


def capture_rgb_static(obs, env=None) -> np.ndarray:
    """Copy the static camera frame from a CALVIN observation.

    CALVIN returns ``rgb_static`` as uint8 HWC numpy from PyBullet. We deepcopy
    immediately so later CUDA work cannot invalidate the EGL readback buffer.
    """
    rgb = obs["rgb_obs"]["rgb_static"]
    if torch is not None and isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    return copy.deepcopy(np.asarray(rgb))


def _to_uint8_rgb(frame) -> np.ndarray:
    if torch is not None and isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    frame = copy.deepcopy(frame)
    frame = np.asarray(frame)

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype.kind == "f":
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255)

    frame = frame.astype(np.uint8, copy=False)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]

    return frame.copy()
