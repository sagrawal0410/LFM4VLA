"""VLA-Adapter Bridge Attention policy (paper arXiv:2509.09372).

Ported from OpenHelix-Team/VLA-Adapter for use with LFM2.5-VL. The policy
consumes per-layer Raw vision features and ActionQuery features from the VLM,
plus a proprioceptive token, and regresses a continuous action chunk with L1 loss.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from models.base_policy import BasePolicyHead


class ProprioProjector(nn.Module):
    """Projects proprio state into the LLM embedding space (policy-side only)."""

    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        x = self.fc1(proprio)
        x = self.act_fn1(x)
        return self.fc2(x)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class MLPResNetBlock(nn.Module):
    """Original Bridge Attention block (shared K/V projections)."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, h_t=None, h_a=None, p=None):
        ratio_g = torch.tanh(self.gating_factor)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)
        h = torch.cat(conditions, dim=1)

        bsz, t_len, channels = x.shape
        k_t = h.size(1)
        k_raw = h_t.size(1)

        q_1 = self.q_proj(x)
        k_tokens = self.k_proj(x)
        v_tokens = self.v_proj(x)
        k_task = self.k_proj(h)
        v_task = self.v_proj(h)
        k_adapter = self.k_proj(h_t)
        v_adapter = self.v_proj(h_t)

        def reshape_heads(t, length):
            return t.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = reshape_heads(q_1, t_len)
        k_tokens = reshape_heads(k_tokens, t_len)
        v_tokens = reshape_heads(v_tokens, t_len)
        k_task = reshape_heads(k_task, k_t)
        v_task = reshape_heads(v_task, k_t)
        k_adapter = reshape_heads(k_adapter, k_raw)
        v_adapter = reshape_heads(v_adapter, k_raw)

        attn_scores = torch.cat(
            [
                torch.matmul(q_1, k_tokens.transpose(-2, -1)),
                torch.matmul(q_1, k_task.transpose(-2, -1)),
                torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g,
            ],
            dim=-1,
        )
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(bsz, t_len, channels)
        output = self.o_proj(output)
        return self.ffn(output + x)


class MLPResNetBlockPro(nn.Module):
    """Pro Bridge Attention: separate Q/K/V channels + RoPE (recommended)."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = RotaryPositionEmbedding(self.head_dim)
        # Kept for checkpoint compatibility with upstream Pro weights; unused.
        self.film_gen = nn.Sequential(nn.Linear(dim, dim * 2))

    def forward(self, x, h_t=None, h_a=None, p=None):
        ratio_g = torch.tanh(self.gating_factor)
        h_adapter = torch.cat((h_a, p), dim=1)
        h_task = h_t
        bsz, t_len, channels = x.shape
        k_a = h_adapter.size(1)
        k_t = h_task.size(1)

        q_1 = self.q_proj(x)
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)

        def reshape_heads(t, length):
            return t.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = reshape_heads(q_1, t_len)
        k_tokens, v_tokens = reshape_heads(k_tokens, t_len), reshape_heads(v_tokens, t_len)
        k_adapter, v_adapter = reshape_heads(k_adapter, k_a), reshape_heads(v_adapter, k_a)
        k_task, v_task = reshape_heads(k_task, k_t), reshape_heads(v_task, k_t)

        cos_main, sin_main = self.rope(seq_len=t_len, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=k_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=k_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        attn_scores = torch.cat(
            [
                torch.matmul(q_1, k_tokens.transpose(-2, -1)),
                torch.matmul(q_1, k_adapter.transpose(-2, -1)),
                torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g,
            ],
            dim=-1,
        )
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        v_combined = torch.cat([v_tokens, v_adapter, v_task], dim=2)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(bsz, t_len, channels)
        output = self.o_proj(output)
        return self.ffn(output + x)


class MLPResNet(nn.Module):
    """Stack of Bridge Attention blocks matching VLM depth."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_pro_version: bool = True,
        num_heads: int = 8,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        block_cls = MLPResNetBlockPro if use_pro_version else MLPResNetBlock
        self.mlp_resnet_blocks = nn.ModuleList(
            [block_cls(dim=hidden_dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_a=None, h_t=None, p=None):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for i, block in enumerate(self.mlp_resnet_blocks):
            # Skip embedding layer (index 0); align block i with VLM layer i+1.
            x = block(x, h_t=h_t[:, i + 1], h_a=h_a[:, i + 1], p=p)
        return self.fc2(self.layer_norm2(x))


class VLAAdapterL1Head(BasePolicyHead):
    """L1 regression action head with Bridge Attention over multi-layer VLM features.

    Expected ``forward`` input ``tok_seq`` is packed multi-layer hidden states of shape
    ``[B, num_layers+1, num_task_tokens + num_action_tokens, D]`` (window flattened),
    or ``[B, T, L, N, D]`` when ``window_size > 1``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        action_dim: int = 7,
        down_sample: str = "none",
        latent: int = 64,
        fwd_pred_next_n: int = 8,
        num_blocks: Optional[int] = None,
        num_task_tokens: int = 512,
        num_action_tokens: int = 64,
        use_pro_version: bool = True,
        num_heads: int = 8,
        proprio_dim: int = 8,
        n_tokens: Optional[int] = None,
        **kwargs,
    ):
        kwargs.pop("with_history", None)
        kwargs.pop("history_type", None)
        kwargs.pop("window_size", None)
        kwargs.pop("tokenizer", None)
        kwargs.pop("action_space", None)
        kwargs.pop("fill_zero", None)
        super().__init__(hidden_size, action_dim, action_space="continuous", **kwargs)

        self.in_features = in_features
        self.fwd_pred_next_n = int(fwd_pred_next_n)
        self.down_sample = down_sample
        self.num_task_tokens = int(num_task_tokens)
        self.num_action_tokens = int(
            n_tokens if n_tokens is not None else (num_action_tokens or latent)
        )
        self.latent = self.num_action_tokens
        self.use_pro_version = bool(use_pro_version)
        self.proprio_dim = int(proprio_dim)

        if num_blocks is None:
            raise ValueError(
                "VLAAdapterL1Head requires num_blocks (= VLM num_hidden_layers)."
            )
        self.num_blocks = int(num_blocks)

        # Action latent is zeros of shape (chunk, action_dim * hidden); project to hidden.
        self.model = MLPResNet(
            num_blocks=self.num_blocks,
            input_dim=in_features * action_dim,
            hidden_dim=in_features,
            output_dim=action_dim,
            use_pro_version=self.use_pro_version,
            num_heads=num_heads,
        )
        self.proprio_projector = ProprioProjector(in_features, self.proprio_dim)

    def predict_action(
        self,
        multi_layer_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        phase: str = "Inference",
    ) -> torch.Tensor:
        """
        Args:
            multi_layer_hidden_states: ``[B, L, N_task + N_aq, D]``
            proprio: ``[B, proprio_dim]``
            phase: ``"Training"`` adds small noise to the zero action latent.
        Returns:
            ``[B, chunk, action_dim]``
        """
        if multi_layer_hidden_states.ndim != 4:
            raise ValueError(
                f"Expected multi-layer HS [B, L, N, D], got {tuple(multi_layer_hidden_states.shape)}"
            )
        batch_size = multi_layer_hidden_states.shape[0]
        device = multi_layer_hidden_states.device
        dtype = multi_layer_hidden_states.dtype
        n_total = multi_layer_hidden_states.shape[2]
        if n_total < self.num_task_tokens + self.num_action_tokens:
            raise ValueError(
                f"Packed tokens {n_total} < num_task_tokens ({self.num_task_tokens}) "
                f"+ num_action_tokens ({self.num_action_tokens})"
            )

        if proprio is None:
            proprio = multi_layer_hidden_states.new_zeros(batch_size, self.proprio_dim)
        else:
            proprio = proprio.reshape(batch_size, -1).to(dtype=dtype)
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"proprio dim {proprio.shape[-1]} != configured {self.proprio_dim}"
                )

        proprio_features = self.proprio_projector(proprio).unsqueeze(1)

        task_hs = multi_layer_hidden_states[:, :, : self.num_task_tokens, :]
        action_hs = multi_layer_hidden_states[
            :, :, self.num_task_tokens : self.num_task_tokens + self.num_action_tokens, :
        ]

        # Paper: A^0 = 0 (optionally + noise in training), shape (B, H, action_dim * D)
        action_latent = torch.zeros(
            batch_size,
            self.fwd_pred_next_n,
            self.action_dim * self.in_features,
            device=device,
            dtype=dtype,
        )
        if phase == "Training":
            action_latent = action_latent + torch.randn_like(action_latent) * 0.02

        return self.model(action_latent, h_a=action_hs, p=proprio_features, h_t=task_hs)

    def forward(self, tok_seq, proprio=None, phase: Optional[str] = None, **kwargs):
        """Adapt to RoboVLM ``forward_action_head`` calling convention."""
        if phase is None:
            phase = "Training" if self.training else "Inference"

        if tok_seq.ndim == 5:
            # [B, T, L, N, D] → flatten window
            bsz, seq_len = tok_seq.shape[:2]
            tok_seq = rearrange(tok_seq, "b t l n d -> (b t) l n d")
            if proprio is not None and proprio.ndim == 3:
                proprio = rearrange(proprio, "b t d -> (b t) d")
            actions = self.predict_action(tok_seq, proprio=proprio, phase=phase)
            return rearrange(
                actions, "(b t) h d -> b t h d", b=bsz, t=seq_len
            )
        if tok_seq.ndim == 4:
            return self.predict_action(tok_seq, proprio=proprio, phase=phase)
        raise ValueError(
            f"VLAAdapterL1Head expected 4D/5D multi-layer HS, got {tuple(tok_seq.shape)}"
        )

    def loss(self, pred_action, labels, attention_mask=None, **kwargs):
        """Full-chunk L1 over all action dims (paper Eq. 2)."""
        if labels is None:
            return {"loss": None}
        if isinstance(labels, (tuple, list)):
            arm, grip = labels
            if arm is None:
                return {"loss": None}
            if grip is None:
                target = arm
            elif grip.ndim == arm.ndim - 1:
                target = torch.cat([arm, grip.unsqueeze(-1)], dim=-1)
            else:
                target = torch.cat([arm, grip], dim=-1)
        else:
            target = labels

        if pred_action.shape != target.shape:
            # Allow [B, H, D] vs [B, 1, H, D]
            if pred_action.ndim + 1 == target.ndim and target.shape[1] == 1:
                pred_action = pred_action.unsqueeze(1)
            elif target.ndim + 1 == pred_action.ndim and pred_action.shape[1] == 1:
                target = target.unsqueeze(1)
            if pred_action.shape != target.shape:
                raise ValueError(
                    f"pred {tuple(pred_action.shape)} vs target {tuple(target.shape)}"
                )

        if attention_mask is None:
            action_l1 = torch.nn.functional.l1_loss(pred_action, target)
        else:
            per = torch.nn.functional.l1_loss(pred_action, target, reduction="none")
            keep = attention_mask.bool()
            # Broadcast mask over action dim if needed.
            while keep.ndim < per.ndim:
                keep = keep.unsqueeze(-1)
            keep = keep.expand_as(per)
            if keep.any():
                action_l1 = per[keep].mean()
            else:
                action_l1 = per.mean()
        if not torch.isfinite(action_l1):
            action_l1 = pred_action.new_zeros(())
        return {"loss_arm": action_l1}
