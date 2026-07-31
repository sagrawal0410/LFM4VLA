"""Depth conditioning: CNN patch encoder + small multimodal QFormer.

Design choice — QFormer over Perceiver Resampler:
  * Learnable queries with self-attention produce coherent token sets for the LLM.
  * Sequential cross-attention to depth → image → text matches the requested
    pre-LLM fusion of the three modalities.
  * Fixed ``num_queries`` keeps sequence growth small (default 16 tokens).

Depth input is a single-channel LIBERO/robosuite GT depth map ``[B, 1, H, W]``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class DepthPatchEncoder(nn.Module):
    """Small 2D CNN that maps a depth map to a sequence of patch tokens."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        out_dim: int = 1024,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        dims = (in_channels, *hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=min(8, dims[i + 1]), num_channels=dims[i + 1]),
                    nn.GELU(),
                ]
            )
        self.stem = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dims[-1], out_dim)
        self.norm = nn.LayerNorm(out_dim, eps=norm_eps)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: ``[B, 1, H, W]`` (or ``[B, H, W]``).
        Returns:
            tokens: ``[B, N, D]`` where ``N = (H/8)*(W/8)`` for 3 stride-2 layers.
        """
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4:
            raise ValueError(f"Expected depth [B,1,H,W], got {tuple(depth.shape)}")

        # Caller (DepthConditioner) disables autocast; keep compute in fp32.
        depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        feats = self.stem(depth)  # [B, C, h, w]
        tokens = feats.flatten(2).transpose(1, 2)  # [B, N, C]
        return self.norm(self.proj(tokens))


class _QFormerLayer(nn.Module):
    """Self-attn among queries, then sequential cross-attn to each modality, then FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_cross: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_cross)
            ]
        )
        self.norm_cross = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_cross)])
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm_self = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        kv_list: Tuple[torch.Tensor, ...],
        kv_masks: Tuple[Optional[torch.Tensor], ...],
    ) -> torch.Tensor:
        residual = queries
        q = self.norm_self(queries)
        attn_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = residual + attn_out

        for cross_attn, norm, kv, kv_mask in zip(
            self.cross_attns, self.norm_cross, kv_list, kv_masks
        ):
            if kv is None:
                continue
            residual = queries
            q = norm(queries)
            key_padding_mask = None
            if kv_mask is not None:
                keep = kv_mask.bool()
                # PyTorch MHA returns NaN if a row masks out *all* keys.
                if keep.any():
                    keep = keep | (~keep.any(dim=-1, keepdim=True))
                    key_padding_mask = ~keep
                else:
                    key_padding_mask = None
            attn_out, _ = cross_attn(
                q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
            )
            queries = residual + attn_out

        residual = queries
        queries = residual + self.ffn(self.norm_ffn(queries))
        return queries


class MultimodalQFormer(nn.Module):
    """Small QFormer: each layer self-attends queries, then cross-attends depth→image→text.

    Preferable to a Perceiver Resampler here: query self-attention + ordered multimodal
    cross-attention produces a compact, coherent token set for the LLM.
    """

    def __init__(
        self,
        dim: int,
        num_queries: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        fuse_text: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_queries = num_queries
        self.fuse_text = fuse_text
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        num_cross = 3 if fuse_text else 2
        self.layers = nn.ModuleList(
            [
                _QFormerLayer(dim, num_heads, num_cross, ffn_mult, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        depth_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            depth_tokens: ``[B, Nd, D]``
            image_tokens: ``[B, Ni, D]``
            text_tokens:  ``[B, Nt, D]`` (optional)
            text_mask:    ``[B, Nt]`` bool/float attention mask (1=keep)
            image_mask:   ``[B, Ni]`` bool mask for padded image tokens (1=keep)
        Returns:
            fused tokens ``[B, num_queries, D]``
        """
        bs = depth_tokens.shape[0]
        # QFormer math in fp32 avoids bf16 softmax underflow → NaN on long KV sequences.
        depth_tokens = depth_tokens.float()
        image_tokens = image_tokens.float()
        if text_tokens is not None:
            text_tokens = text_tokens.float()
        queries = self.query_tokens.expand(bs, -1, -1).to(
            dtype=torch.float32, device=depth_tokens.device
        )

        if self.fuse_text:
            kv_list = (depth_tokens, image_tokens, text_tokens)
            kv_masks = (None, image_mask, text_mask)
        else:
            kv_list = (depth_tokens, image_tokens)
            kv_masks = (None, image_mask)

        for layer in self.layers:
            queries = layer(queries, kv_list, kv_masks)

        return self.out_norm(queries)


class DepthConditioner(nn.Module):
    """Depth CNN + multimodal QFormer producing LLM-insertable tokens."""

    def __init__(self, hidden_size: int, depth_cfg: Dict[str, Any]):
        super().__init__()
        cnn_cfg = depth_cfg.get("cnn", {})
        qf_cfg = depth_cfg.get("qformer", {})

        hidden_dims = tuple(cnn_cfg.get("hidden_dims", [32, 64, 128]))
        self.encoder = DepthPatchEncoder(
            in_channels=1,
            hidden_dims=hidden_dims,
            out_dim=hidden_size,
        )
        num_heads = int(qf_cfg.get("num_heads", 8))
        if hidden_size % num_heads != 0:
            # Pick the largest divisor of hidden_size that is <= requested heads.
            for h in range(min(num_heads, hidden_size), 0, -1):
                if hidden_size % h == 0:
                    num_heads = h
                    break
        self.qformer = MultimodalQFormer(
            dim=hidden_size,
            num_queries=int(qf_cfg.get("num_queries", 16)),
            num_layers=int(qf_cfg.get("num_layers", 2)),
            num_heads=num_heads,
            ffn_mult=int(qf_cfg.get("ffn_mult", 4)),
            dropout=float(qf_cfg.get("dropout", 0.0)),
            fuse_text=bool(qf_cfg.get("fuse_text", True)),
        )
        self.num_queries = self.qformer.num_queries
        # Keep depth stack in fp32 even under Lightning bf16-mixed.
        self.encoder = self.encoder.float()
        self.qformer = self.qformer.float()

    def forward(
        self,
        depth: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            depth: ``[B, 1, H, W]``
            image_tokens: ``[B, Ni, D]`` (scattered image features, or pooled)
            text_tokens: ``[B, Nt, D]``
            text_mask: ``[B, Nt]``
            image_mask: ``[B, Ni]``
        Returns:
            ``[B, num_queries, D]``
        """
        out_dtype = image_tokens.dtype
        device_type = "cuda" if depth.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            depth_tokens = self.encoder(depth.float())
            fused = self.qformer(
                depth_tokens,
                image_tokens.float(),
                text_tokens.float() if text_tokens is not None else None,
                text_mask=text_mask,
                image_mask=image_mask,
            )
            fused = torch.nan_to_num(fused.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return fused.to(dtype=out_dtype)
