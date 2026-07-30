"""Helpers for LIBERO / robosuite ground-truth depth maps."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def extract_agentview_depth(
    obs: dict,
    env: Optional[Any] = None,
    *,
    normalize: bool = True,
    depth_clip_m: float = 2.0,
) -> np.ndarray:
    """Return agentview depth as ``float32`` ``[H, W]``.

    LIBERO/robosuite expose ``agentview_depth`` when ``camera_depths=True``.
    Raw values are typically MuJoCo-normalized ``[0, 1]``; if ``env`` is given we
    convert to meters via ``get_real_depth_map``, then min-max (or clip) normalize
    for the depth CNN.
    """
    if "agentview_depth" not in obs:
        raise KeyError(
            "obs missing 'agentview_depth'. Create the env with camera_depths=True."
        )
    depth = np.asarray(obs["agentview_depth"])
    # robosuite may return [H, W, 1]
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)

    if env is not None:
        try:
            from robosuite.utils.camera_utils import get_real_depth_map

            sim = getattr(env, "sim", None)
            if sim is None and hasattr(env, "env"):
                sim = getattr(env.env, "sim", None)
            if sim is not None:
                # get_real_depth_map expects normalized depth; returns meters.
                metric = get_real_depth_map(sim, depth)
                if isinstance(metric, np.ndarray) and metric.ndim == 3:
                    metric = metric[..., 0]
                depth = np.asarray(metric, dtype=np.float32)
                if depth_clip_m is not None:
                    depth = np.clip(depth, 0.0, float(depth_clip_m))
        except Exception:
            # Fall back to raw normalized depth if conversion fails.
            pass

    if normalize:
        dmin = float(depth.min())
        dmax = float(depth.max())
        depth = (depth - dmin) / (dmax - dmin + 1e-6)

    return np.ascontiguousarray(depth, dtype=np.float32)


def preprocess_depth_like_rgb(
    depth: np.ndarray,
    image_transform: str = "rot180",
    image_size: int = 224,
) -> np.ndarray:
    """Apply the same geometry as RGB demos (rot180) and resize to ``image_size``."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if image_transform == "rot180":
        depth = depth[::-1, ::-1]
    elif image_transform == "flip_vertical":
        depth = depth[::-1]
    elif image_transform == "none":
        pass
    else:
        raise ValueError(f"Unknown image_transform: {image_transform}")
    depth = np.ascontiguousarray(depth)

    if depth.shape[0] != image_size or depth.shape[1] != image_size:
        # Lazy import so TF-free eval/scripts can use extract without torchvision.
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        depth = t.squeeze(0).squeeze(0).numpy()
    return depth.astype(np.float32)
