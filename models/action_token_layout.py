"""Humanoid Everyday (and other multi-token) action-query layout helpers.

LIBERO / CALVIN use the legacy path in ``model_backbone``:
  * one learned ``action_token`` of shape ``[D]``
  * repeated ``latent`` times into the VLM sequence

This module is used **only** when the act head opts in via
``type == "FCContinuousDecoder"`` or an explicit ``num_action_tokens`` field.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def uses_he_action_layout(act_head_configs: Optional[dict]) -> bool:
    """True when HE / continuous-head multi-token layout should be used."""
    if not act_head_configs:
        return False
    if act_head_configs.get("type") == "FCContinuousDecoder":
        return True
    return "num_action_tokens" in act_head_configs


def resolve_action_token_layout(act_head_configs: Optional[dict]) -> Tuple[int, int, int]:
    """Return ``(num_action_tokens, token_repeat, total_tokens)`` for HE layouts.

    * ``num_action_tokens`` — distinct learned action-query embeddings.
    * ``token_repeat`` / ``latent`` — how many times each embedding is repeated.
    * ``total_tokens`` — ``num_action_tokens * token_repeat`` slots fed to the head.
    """
    if not act_head_configs:
        return 1, 1, 1
    num_tokens = int(act_head_configs.get("num_action_tokens", 1))
    token_repeat = int(
        act_head_configs.get("latent", act_head_configs.get("token_repeat", 1))
    )
    if num_tokens < 1 or token_repeat < 1:
        raise ValueError(
            f"num_action_tokens ({num_tokens}) and latent/token_repeat "
            f"({token_repeat}) must be >= 1"
        )
    return num_tokens, token_repeat, num_tokens * token_repeat


def expand_action_tokens(
    action_token: torch.Tensor,
    batch_size: int,
    token_repeat: int = 1,
) -> torch.Tensor:
    """Build ``[B, num_tokens * token_repeat, D]`` from a Parameter of shape ``[D]`` or ``[N, D]``."""
    toks = action_token
    if toks.ndim == 1:
        toks = toks.unsqueeze(0)  # [D] -> [1, D]
    if token_repeat > 1:
        toks = toks.repeat_interleave(token_repeat, dim=0)
    return toks.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
