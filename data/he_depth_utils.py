"""Shared Humanoid Everyday depth sanitization / normalization.

HE egocentric depth is stored in **raw RealSense-like units (≈ mm)**, not meters.
Audit of the 4-skill G1 subset showed:

  * ~86% finite positive pixels
  * ~14% zeros (holes) — normal
  * ≪0.01% Inf speckles (from float16 overflow / bad pixels)
  * frequent near-uint16 saturations around **62272** that are *not* real range

Training must zero Inf/holes/saturations and normalize robustly, otherwise
min-max is dominated by ~62k sentinels or NaNs poison the depth CNN.
"""

from __future__ import annotations

import numpy as np

# Drop depths beyond this (10 m if units are mm). Indoor HE egocentric is << this.
HE_DEPTH_MAX_MM = 10_000.0
# Near uint16 max (65535); audit peaks at 62272 — treat as invalid saturation.
HE_DEPTH_SAT_FLOOR = 60_000.0
# Robust normalize percentiles over valid pixels (resists residual outliers).
HE_DEPTH_Q_LOW = 1.0
HE_DEPTH_Q_HIGH = 99.0


def sanitize_he_depth_hw(
    frame: np.ndarray,
    *,
    max_depth: float = HE_DEPTH_MAX_MM,
    sat_floor: float = HE_DEPTH_SAT_FLOOR,
) -> np.ndarray:
    """Return ``[H, W]`` float32 with invalid pixels set to 0.

    Invalid = non-finite, ``<= 0``, ``>= max_depth``, or ``>= sat_floor``.
    """
    d = np.asarray(frame, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected depth [H,W], got {d.shape}")
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (d > 0.0) & (d < float(max_depth)) & (d < float(sat_floor))
    return np.where(valid, d, 0.0).astype(np.float32, copy=False)


def sanitize_he_depth_volume(
    arr: np.ndarray,
    *,
    max_depth: float = HE_DEPTH_MAX_MM,
    sat_floor: float = HE_DEPTH_SAT_FLOOR,
) -> np.ndarray:
    """Sanitize ``[T,H,W]`` (or ``[T,H,W,1]``) the same way as a single frame."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    if a.ndim != 3:
        raise ValueError(f"Expected depth [T,H,W], got {a.shape}")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (a > 0.0) & (a < float(max_depth)) & (a < float(sat_floor))
    return np.where(valid, a, 0.0).astype(np.float32, copy=False)


def normalize_he_depth_01(
    frame: np.ndarray,
    *,
    q_low: float = HE_DEPTH_Q_LOW,
    q_high: float = HE_DEPTH_Q_HIGH,
) -> np.ndarray:
    """Percentile-normalize sanitized ``[H,W]`` depth to ``[0,1]`` (invalid stay 0)."""
    d = np.asarray(frame, dtype=np.float32)
    valid = d > 0
    if not bool(valid.any()):
        return np.zeros_like(d, dtype=np.float32)
    vals = d[valid]
    lo = float(np.percentile(vals, q_low))
    hi = float(np.percentile(vals, q_high))
    if hi <= lo + 1e-6:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo + 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    out = np.zeros_like(d, dtype=np.float32)
    out[valid] = np.clip((vals - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
