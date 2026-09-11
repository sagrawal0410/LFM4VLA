"""Frozen V-JEPA2 target encoder for the world branch.

Produces the future-latent targets y*_{t+H}. The encoder is frozen and always
in eval mode: it supplies targets, it is never trained. Token pooling is
256 -> 64 with AvgPool-k4, matching the compression the plan specifies.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

DEFAULT_VJEPA = os.environ.get(
    "ROBOTNAV_VJEPA2",
    "facebook/vjepa2-vitl-fpc64-256")


class FrozenVJEPA2(nn.Module):
    def __init__(self, model_id: str = DEFAULT_VJEPA, pool_tokens: int = 64,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModel, AutoVideoProcessor
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.model.to(device)
        self.pool_tokens = int(pool_tokens)
        self.hidden = int(getattr(self.model.config, "hidden_size", 1024))

    @property
    def target_dim(self) -> int:
        return self.hidden

    @torch.no_grad()
    def encode(self, frames) -> torch.Tensor:
        """frames: list of B clips (each a list of PIL/ndarray) -> [B, pool, D]."""
        inp = self.processor(frames, return_tensors="pt")
        inp = {k: v.to(self.model.device) for k, v in inp.items()}
        out = self.model.get_vision_features(**inp) if hasattr(
            self.model, "get_vision_features") else self.model(**inp).last_hidden_state
        if out.ndim == 2:
            out = out.unsqueeze(1)
        return self.pool(out)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, D] -> [B, pool_tokens, D] by average pooling over tokens."""
        b, n, d = tokens.shape
        if n == self.pool_tokens:
            return tokens
        k = max(1, n // self.pool_tokens)
        usable = (n // k) * k
        t = tokens[:, :usable].reshape(b, usable // k, k, d).mean(2)
        if t.shape[1] > self.pool_tokens:
            t = t[:, : self.pool_tokens]
        elif t.shape[1] < self.pool_tokens:
            pad = t[:, -1:].expand(b, self.pool_tokens - t.shape[1], d)
            t = torch.cat([t, pad], dim=1)
        return t
