"""Detached score -> per-example weight, and backbone-only gradient routing.

Two rules from the plan document that are easy to get wrong:

1. Never multiply the whole FM loss by w. An action-dependent weight changes
   the effective conditional action distribution q(a|c) ~ p_D(a|c) w(c,a) and
   can skew the policy toward rare/exploratory modes. The weight must reach
   the shared backbone only.

2. Never backpropagate through the score. Weights are always detached.
"""
from __future__ import annotations

import torch


def robust_weight(score: torch.Tensor, z_clip: float = 2.0, exp_scale: float = 0.35,
                  w_min: float = 0.5, w_max: float = 2.0,
                  renormalize: bool = True) -> torch.Tensor:
    """Median/MAD robust-z -> bounded multiplicative weight, mean-normalised.

    Verbatim transform from the plan; examples with no fresh score get 1.0.
    """
    s = score.detach().float()
    med = s.median()
    mad = (s - med).abs().median()
    z = (s - med) / (1.4826 * mad + 1e-6)
    z = z.clamp(-z_clip, z_clip)
    w = torch.exp(exp_scale * z).clamp(w_min, w_max)
    if renormalize:
        w = w / w.mean().clamp_min(1e-6)
    return w.detach()


def grad_scale_identity(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Identity in the forward pass; scales only dL/dx by w.

    x: [B, ...]   w: [B] (broadcast over trailing dims)
    """
    if w is None:
        return x
    w = w.detach().to(x.dtype).to(x.device)
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    return x.detach() + w * (x - x.detach())


def route_context(ctx, w):
    """Apply the scaler to a context tensor, or to every tensor in a container.

    pi0/VLAFlow-style heads consume per-layer K/V rather than one context
    tensor, so every shared tensor entering the action expert must be wrapped
    or the routing silently covers only part of the path.
    """
    if w is None or ctx is None:
        return ctx
    if torch.is_tensor(ctx):
        return grad_scale_identity(ctx, w)
    if isinstance(ctx, (list, tuple)):
        return type(ctx)(route_context(c, w) for c in ctx)
    if isinstance(ctx, dict):
        return {k: route_context(v, w) for k, v in ctx.items()}
    return ctx
