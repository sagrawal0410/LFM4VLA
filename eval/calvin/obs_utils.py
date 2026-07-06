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
    if env is not None:
        try:
            frame = env.render(mode="rgb_array")
            return _to_uint8_rgb(frame)
        except Exception:
            pass

    rgb = obs["rgb_obs"]["rgb_static"]
    return _to_uint8_rgb(rgb)


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
