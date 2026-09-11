"""RobotNav action heads: MLP (Qwen-RobotNav), GR00T-N1.7-style flow-matching
DiT, and SmolVLA-style interleaved flow-matching expert.

All three follow the repo head contract (see model_backbone.forward_action_head):
``forward(tok_seq, actions=..., action_masks=..., **kwargs)`` returns a dict with
``actions``/``gripper`` keys; ``get_labels`` passes through; ``loss`` returns a
dict whose main term is ``loss_arm`` (BaseTrainer scales gripper terms away).

Labels arrive as ``actions=(chunk [B, ws, K, act_dim], None)`` with
``action_masks=[B, ws, K]`` — the RobotNav collater marks only the last window
slot real, so every loss here is mask-weighted.

Flow-matching heads compute the FM loss inside ``forward`` (they need the
sampled t / noise) and stash it for ``loss``; at inference (``actions=None``)
they integrate the learned velocity field with Euler steps and return the
denormalized-in-[-1,1] action chunk.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from models.base_policy import BasePolicyHead


# --------------------------------------------------------------------- utils
def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """DiT-style sinusoidal timestep embedding. t: [B] in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.to(t.dtype) if t.dtype.is_floating_point else emb


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype))


def _flatten_labels(actions, action_masks, window: int):
    """(chunk [B, ws, K, D], _) + mask [B, ws, K] -> ([B*ws, K, D], [B*ws, K])."""
    chunk = actions[0] if isinstance(actions, (tuple, list)) else actions
    x1 = rearrange(chunk, "b w k d -> (b w) k d")
    if action_masks is None:
        mask = torch.ones(x1.shape[:2], dtype=torch.bool, device=x1.device)
    else:
        mask = rearrange(action_masks, "b w k -> (b w) k").bool()
    return x1, mask


# ======================================================================= MLP
class _TemporalReadout(nn.Module):
    """Post-fusion for the regression heads: causal temporal encoder + query readout.

    Stage 1 — 1-2 layers of causal self-attention over the w slot summaries
    (with RELATIVE offset embeddings, since the history stride is variable),
    so slot j absorbs slots <= j into a running representation.

    Stage 2 — learned action queries cross-attend (also causally) over those
    enriched slots. This is the part a plain "take the last position" design
    misses: the query can retrieve straight from slot i if that is where the
    relevant evidence sits, instead of relying on slot j to have preserved it.

    Output keeps the per-slot shape, so the existing decoder, the [B, w, K, 3]
    contract, and per-slot supervision all stay intact.
    """

    def __init__(self, dim: int, layers: int = 2, heads: int = 8,
                 n_query: int = 1, max_window: int = 16):
        super().__init__()
        self.offset_emb = nn.Parameter(torch.zeros(max_window, dim))
        nn.init.trunc_normal_(self.offset_emb, std=0.02)
        self.enc = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                       norm_first=True, dropout=0.0)
            for _ in range(layers)])
        self.queries = nn.Parameter(torch.zeros(1, n_query, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.read = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim * n_query, dim)
        self.norm = nn.LayerNorm(dim)
        self.n_query = n_query

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, w, D] -> [b, w, D] (temporally fused, causal)."""
        b, w, d = x.shape
        idx = torch.arange(w, device=x.device)
        # position j is "now" for its own prediction -> tag by distance from j
        x = x + self.offset_emb.to(x.dtype)[(w - 1 - idx).clamp(
            0, self.offset_emb.shape[0] - 1)][None]
        causal = torch.triu(torch.ones(w, w, device=x.device, dtype=torch.bool),
                            diagonal=1)
        h = x
        for layer in self.enc:                       # stage 1: causal self-attn
            h = layer(h, src_mask=causal)

        # stage 2: per-slot queries read over slots <= j
        q = self.queries.to(x.dtype).expand(b * w, -1, -1)
        mem = h[:, None].expand(b, w, w, d).reshape(b * w, w, d)
        kpm = causal[None].expand(b, w, w).reshape(b * w, w)
        r, _ = self.read(q, mem, mem, key_padding_mask=kpm, need_weights=False)
        r = torch.nan_to_num(r)                      # slot 0 attends to nothing
        r = self.out(r.reshape(b * w, self.n_query * d)).view(b, w, d)
        return self.norm(h + r)


class RobotNavMLPHead(BasePolicyHead):
    """Qwen-RobotNav head: a simple MLP on the ActionQuery token feature that
    regresses the K normalized waypoints (x, y, yaw) in [-1, 1] with MSE.

    Use with ``action_space: continuous``, ``latent: 1`` — the backbone appends
    one learnable ActionQuery token and hands its last-layer feature here as
    ``tok_seq [B, ws, 1, D]``. Co-training with the LM loss (85/15 mixture,
    lambda=1.0) happens in RobotNavTrainer, untouched.
    """

    def __init__(self, in_features: int, hidden_size: int = 1024,
                 action_dim: int = 3, fwd_pred_next_n: int = 8,
                 latent: int = 1, **kwargs):
        kwargs.pop("down_sample", None)
        super().__init__(hidden_size, action_dim, latent=latent, **kwargs)
        self.fwd_pred_next_n = fwd_pred_next_n
        self.build_stop_head(in_features, fwd_pred_next_n)
        self.temporal = _TemporalReadout(
            in_features * latent,
            layers=int(kwargs.get("history_layers", 2)),
            n_query=int(kwargs.get("history_queries", 1)),
        ) if kwargs.get("history_fusion", False) else None
        self.net = nn.Sequential(
            nn.LayerNorm(in_features * latent),
            nn.Linear(in_features * latent, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, fwd_pred_next_n * action_dim),
            nn.Tanh(),
        )

    def forward(self, tok_seq: torch.Tensor, actions=None, action_masks=None,
                **kwargs) -> Dict[str, Optional[torch.Tensor]]:
        b, w = tok_seq.shape[:2]
        x = rearrange(tok_seq, "b w n d -> b w (n d)")
        if self.temporal is not None and w > 1:
            x = self.temporal(x)
        x = x.reshape(b * w, -1)
        pred = self.net(x).view(b, w, self.fwd_pred_next_n, self.action_dim)
        return {"actions": pred, "gripper": None}

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        return pred_actions, labels, action_masks

    def loss(self, pred_action, labels, attention_mask=None, **kwargs):
        if labels is None or labels[0] is None:
            return {"loss": None}
        pred = pred_action[0] if isinstance(pred_action, (tuple, list)) else pred_action
        err = F.mse_loss(pred, labels[0], reduction="none").mean(-1)  # [B, ws, K]
        if attention_mask is not None:
            m = attention_mask.bool()
            err = err[m].mean() if m.any() else err.sum() * 0.0
        else:
            err = err.mean()
        return {"loss_arm": err}


# ============================================================ WaypointFCDecoder
class WaypointFCDecoder(BasePolicyHead):
    """FCDecoder for waypoint tasks: the same stacked trunk-MLP -> action-head
    structure as models.continuous_policy.FCDecoder, but with a single tanh
    action head emitting the full K x action_dim waypoint chunk — no gripper
    branch. Use for lower-body / navigation commands where the VLM predicts
    the 8 (x, y, yaw) waypoints directly.
    """

    def __init__(self, in_features: int, hidden_size: int = 1024,
                 action_dim: int = 3, fwd_pred_next_n: int = 8,
                 latent: int = 1, **kwargs):
        kwargs.pop("down_sample", None)
        super().__init__(hidden_size, action_dim, latent=latent, **kwargs)
        self.fwd_pred_next_n = fwd_pred_next_n
        self.temporal = _TemporalReadout(
            in_features * latent,
            layers=int(kwargs.get("history_layers", 2)),
            n_query=int(kwargs.get("history_queries", 1)),
        ) if kwargs.get("history_fusion", False) else None
        # Stacked heads, mirroring FCDecoder: trunk MLP then MLPTanh action head.
        self.mlp = nn.Sequential(
            nn.Linear(in_features * latent, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size * latent),
        )
        self.actions = nn.Sequential(
            nn.Linear(hidden_size * latent, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, fwd_pred_next_n * action_dim),
            nn.Tanh(),
        )

    def forward(self, tok_seq: torch.Tensor, actions=None, action_masks=None,
                **kwargs) -> Dict[str, Optional[torch.Tensor]]:
        b, w = tok_seq.shape[:2]
        x = rearrange(tok_seq, "b w n d -> b w (n d)")
        if self.temporal is not None and w > 1:
            x = self.temporal(x)
        x = x.reshape(b * w, -1)
        pred = self.actions(self.mlp(x)).view(
            b, w, self.fwd_pred_next_n, self.action_dim)
        return {"actions": pred, "gripper": None}

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        return pred_actions, labels, action_masks

    def loss(self, pred_action, labels, attention_mask=None, **kwargs):
        if labels is None or labels[0] is None:
            return {"loss": None}
        pred = pred_action[0] if isinstance(pred_action, (tuple, list)) else pred_action
        err = F.huber_loss(pred, labels[0], reduction="none").mean(-1)  # [B, ws, K]
        if attention_mask is not None:
            m = attention_mask.bool()
            err = err[m].mean() if m.any() else err.sum() * 0.0
        else:
            err = err.mean()
        return {"loss_arm": err}


# ========================================================== transformer bits
class _SelfAttnBlock(nn.Module):
    """Pre-LN transformer encoder block (bidirectional, key-padding aware)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor, key_padding: Optional[torch.Tensor] = None):
        h = self.n1(x)
        q, k, v = rearrange(self.qkv(h), "b s (three h d) -> three b h s d",
                            three=3, h=self.heads)
        attn_mask = None
        if key_padding is not None:
            attn_mask = key_padding[:, None, None, :].bool()  # True = attend
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(rearrange(o, "b h s d -> b s (h d)"))
        return x + self.mlp(self.n2(x))


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _DiTCrossBlock(nn.Module):
    """AdaLN-Zero DiT block: self-attn (action tokens) -> cross-attn (VLM KV)
    -> MLP. Action/state tokens are the only queries; VLM features are never
    updated (one-directional conditioning)."""

    def __init__(self, dim: int, ctx_dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.sa_qkv = nn.Linear(dim, dim * 3)
        self.sa_proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ca_q = nn.Linear(dim, dim)
        self.ca_kv = nn.Linear(ctx_dim, dim * 2)
        self.ca_proj = nn.Linear(dim, dim)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        # 9 modulation tensors: (shift, scale, gate) x (sa, ca, mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, ctx, t_emb, ctx_padding: Optional[torch.Tensor] = None):
        (s1, g1, sc1, s2, g2, sc2, s3, g3, sc3) = self.ada(t_emb).chunk(9, dim=-1)
        h = _modulate(self.n1(x), s1, sc1)
        q, k, v = rearrange(self.sa_qkv(h), "b s (three h d) -> three b h s d",
                            three=3, h=self.heads)
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + g1.unsqueeze(1) * self.sa_proj(rearrange(o, "b h s d -> b s (h d)"))

        h = _modulate(self.n2(x), s2, sc2)
        q = rearrange(self.ca_q(h), "b s (h d) -> b h s d", h=self.heads)
        k, v = rearrange(self.ca_kv(ctx), "b s (two h d) -> two b h s d",
                         two=2, h=self.heads)
        attn_mask = None
        if ctx_padding is not None:
            attn_mask = ctx_padding[:, None, None, :].bool()
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + g2.unsqueeze(1) * self.ca_proj(rearrange(o, "b h s d -> b s (h d)"))

        h = _modulate(self.n3(x), s3, sc3)
        return x + g3.unsqueeze(1) * self.mlp(h)


# ==================================================================== GR00T
class _CausalHistoryMemory(nn.Module):
    """Cross-attention memory over preceding window slots (post-fusion).

    Each slot's encoder tokens are attention-pooled to ``n_tokens`` summaries;
    slot j then receives the summaries of slots i < j, tagged with a learned
    RELATIVE offset embedding (j - i). Ordinal position alone is not enough
    here: the history stride is variable (uniform-spread vs latest-window
    sampling), so the model needs to know how far back each summary is.

    Causal by construction, so every slot keeps a valid prediction and the
    per-slot supervision stays usable. Cost is linear in the window: the
    denoiser's action tokens are the only queries.
    """

    def __init__(self, dim: int, n_tokens: int = 32, heads: int = 8,
                 max_window: int = 16, pool: str = "mean"):
        super().__init__()
        self.n_tokens = n_tokens
        self.pool_mode = pool
        if pool == "attn":
            # Learned pooling: expressive, but ~4·dim² params. Affordable when
            # instantiated ONCE (GR00T); ruinous per-VLM-layer (SmolVLA would
            # pay 507M on a 2048-dim backbone), hence "mean" is the default.
            self.queries = nn.Parameter(torch.zeros(1, n_tokens, dim))
            nn.init.trunc_normal_(self.queries, std=0.02)
            self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.offset_emb = nn.Parameter(torch.zeros(max_window, dim))
        nn.init.trunc_normal_(self.offset_emb, std=0.02)
        self.norm = nn.LayerNorm(dim)

    def _summarize(self, x: torch.Tensor,
                   key_padding: Optional[torch.Tensor]) -> torch.Tensor:
        """[(b w), S, D] -> [(b w), k, D]; padding-aware segment means."""
        if self.pool_mode == "attn":
            q = self.queries.to(x.dtype).expand(x.shape[0], -1, -1)
            out, _ = self.pool(q, x, x, key_padding_mask=key_padding,
                               need_weights=False)
            return out
        valid = (~key_padding).to(x.dtype) if key_padding is not None \
            else x.new_ones(x.shape[:2])
        num = F.adaptive_avg_pool1d((x * valid[..., None]).transpose(1, 2),
                                    self.n_tokens).transpose(1, 2)
        den = F.adaptive_avg_pool1d(valid[:, None], self.n_tokens).transpose(1, 2)
        return num / den.clamp(min=1e-6)

    def forward(self, x: torch.Tensor, b: int, w: int,
                key_padding: Optional[torch.Tensor] = None):
        """x: [(b w), S, D] -> memory [(b w), w*k, D], pad mask [(b w), w*k]."""
        bw, _, d = x.shape
        pooled = self._summarize(x, key_padding)           # [(b w), k, D]
        k = pooled.shape[1]
        pooled = self.norm(pooled).view(b, w, k, d)

        idx = torch.arange(w, device=x.device)
        off = (idx[:, None] - idx[None, :]).clamp(0, self.offset_emb.shape[0] - 1)
        # mem[b, view j, source i, k, d]
        mem = pooled[:, None].expand(b, w, w, k, d)
        mem = mem + self.offset_emb.to(x.dtype)[off][None, :, :, None, :]
        mem = mem.reshape(b * w, w * k, d)

        valid = (idx[None, :] < idx[:, None])              # [view j, source i]
        pad = (~valid)[None, :, :, None].expand(b, w, w, k).reshape(b * w, w * k)
        return mem, pad


class GR00TFlowMatchingHead(BasePolicyHead):
    """GR00T-N1.7-style flow-matching decoder.

    Pipeline: last-layer VLM hidden states (entire sequence, pre-final-norm
    when ``use_pre_norm_features`` is set in the act_head config) -> LayerNorm
    -> 4-layer self-attention block (32 heads, d_head 64 -> width 2048) ->
    lightweight DiT (default width 1024, depth 10) where the noised action
    (+ optional state) tokens are the queries and the self-attention block
    outputs are cross-attended as K/V only. Rectified-flow objective.

    Use with ``action_space: down_sample`` so ``tok_seq`` is the full sequence
    ``[B, ws, S, D]``; the backbone also passes ``encoder_attention_mask``.
    """

    def __init__(self, in_features: int, action_dim: int = 3,
                 fwd_pred_next_n: int = 8,
                 sa_dim: int = 2048, sa_layers: int = 4, sa_heads: int = 32,
                 dit_dim: int = 1024, dit_depth: int = 10, dit_heads: int = 16,
                 mlp_ratio: int = 4, num_inference_steps: int = 10,
                 state_dim: int = 0, hidden_size: int = 1024, **kwargs):
        kwargs.pop("down_sample", None)
        super().__init__(hidden_size, action_dim, **kwargs)
        assert sa_dim == sa_heads * 64, "spec: 32 heads x d_head 64"
        self.fwd_pred_next_n = fwd_pred_next_n
        self.num_inference_steps = num_inference_steps
        self.state_dim = state_dim

        self.in_norm = nn.LayerNorm(in_features)
        self.in_proj = nn.Linear(in_features, sa_dim)
        self.sa_blocks = nn.ModuleList(
            [_SelfAttnBlock(sa_dim, sa_heads, mlp_ratio) for _ in range(sa_layers)]
        )
        # Post-fusion memory over the window (off by default: existing runs and
        # checkpoints are unaffected until a config opts in).
        self.history = _CausalHistoryMemory(
            sa_dim, n_tokens=int(kwargs.get("history_tokens", 32)),
            pool=str(kwargs.get("history_pool", "attn")),   # single instance
        ) if kwargs.get("history_fusion", False) else None
        self.t_embed = TimestepEmbedder(dit_dim)
        self.action_in = nn.Linear(action_dim, dit_dim)
        self.state_in = nn.Linear(state_dim, dit_dim) if state_dim > 0 else None
        self.pos_emb = nn.Parameter(
            torch.zeros(1, fwd_pred_next_n + (1 if state_dim > 0 else 0), dit_dim)
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [_DiTCrossBlock(dit_dim, sa_dim, dit_heads, mlp_ratio)
             for _ in range(dit_depth)]
        )
        self.out_norm = nn.LayerNorm(dit_dim, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(dit_dim, dit_dim * 2))
        nn.init.zeros_(self.out_ada[1].weight)
        nn.init.zeros_(self.out_ada[1].bias)
        self.out_proj = nn.Linear(dit_dim, action_dim)
        self._stash: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------- encoding
    def _encode_ctx(self, tok_seq: torch.Tensor,
                    encoder_attention_mask: Optional[torch.Tensor]):
        b, w = tok_seq.shape[:2]
        x = rearrange(tok_seq, "b w s d -> (b w) s d")
        x = self.in_proj(self.in_norm(x))
        pad = None
        if encoder_attention_mask is not None:
            pad = encoder_attention_mask.reshape(x.shape[0], -1).bool()
        for blk in self.sa_blocks:                    # per-slot: 9*S^2, not (9S)^2
            x = blk(x, key_padding=pad)
        if self.history is not None and w > 1:
            # Append preceding slots' pooled summaries to the DiT's K/V set.
            mem, mem_pad = self.history(x, b, w, key_padding=pad)
            x = torch.cat([x, mem], dim=1)
            zeros = mem_pad.new_zeros(x.shape[0], x.shape[1] - mem_pad.shape[1]) \
                if pad is None else pad
            pad = torch.cat([zeros.bool(), mem_pad], dim=1)
        return x, pad

    def _velocity(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor,
                  ctx_pad: Optional[torch.Tensor],
                  state: Optional[torch.Tensor]) -> torch.Tensor:
        tok = self.action_in(x_t)
        if self.state_in is not None:
            s = torch.zeros(x_t.shape[0], 1, self.state_in.in_features,
                            device=x_t.device, dtype=x_t.dtype) if state is None \
                else state[:, None, :]
            tok = torch.cat([self.state_in(s.squeeze(1))[:, None], tok], dim=1)
        tok = tok + self.pos_emb[:, : tok.shape[1]]
        t_emb = self.t_embed(t)
        for blk in self.blocks:
            tok = blk(tok, ctx, t_emb, ctx_padding=ctx_pad)
        shift, scale = self.out_ada(t_emb).chunk(2, dim=-1)
        tok = _modulate(self.out_norm(tok), shift, scale)
        if self.state_in is not None:
            tok = tok[:, 1:]
        return self.out_proj(tok)

    # -------------------------------------------------------------- forward
    def forward(self, tok_seq: torch.Tensor, actions=None, action_masks=None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                rel_state: Optional[torch.Tensor] = None, **kwargs):
        b, w = tok_seq.shape[:2]
        ctx, ctx_pad = self._encode_ctx(tok_seq, encoder_attention_mask)

        if actions is not None and actions[0] is not None:  # training
            x1, mask = _flatten_labels(actions, action_masks, w)
            x1 = x1.to(ctx.dtype)
            t = torch.rand(x1.shape[0], device=x1.device, dtype=ctx.dtype)
            x0 = torch.randn_like(x1)
            x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
            v_pred = self._velocity(x_t, t, ctx, ctx_pad, rel_state)
            err = F.mse_loss(v_pred, x1 - x0, reduction="none").mean(-1)  # [BW, K]
            fm = err[mask].mean() if mask.any() else err.sum() * 0.0
            self._stash["loss"] = fm
            pred = (x_t + (1.0 - t[:, None, None]) * v_pred)  # 1-step x1 estimate
            return {"actions": pred.view(b, w, self.fwd_pred_next_n, self.action_dim),
                    "gripper": None}

        # inference: Euler integration t 0 -> 1
        n = ctx.shape[0]
        x = torch.randn(n, self.fwd_pred_next_n, self.action_dim,
                        device=ctx.device, dtype=ctx.dtype)
        dt = 1.0 / self.num_inference_steps
        for i in range(self.num_inference_steps):
            t = torch.full((n,), i * dt, device=ctx.device, dtype=ctx.dtype)
            x = x + self._velocity(x, t, ctx, ctx_pad, rel_state) * dt
        return {"actions": x.view(b, w, self.fwd_pred_next_n, self.action_dim),
                "gripper": None}

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        return pred_actions, labels, action_masks

    def loss(self, pred_action, labels, attention_mask=None, **kwargs):
        if labels is None or labels[0] is None:
            return {"loss": None}
        return {"loss_arm": self._stash.pop("loss")}


# =================================================================== SmolVLA
class _ExpertSelfLayer(nn.Module):
    """Causal prefix self-attention: action tokens are the queries; keys/values
    are [VLM_l features ; action tokens]. Action token i attends to every VLM
    feature of layer l plus action tokens <= i. VLM features are never updated.
    """

    def __init__(self, dim: int, vlm_dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv_act = nn.Linear(dim, dim * 2)
        self.kv_vlm = nn.Linear(vlm_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, a, vlm, vlm_padding=None, vlm_kv=None):
        h = self.n1(a)
        q = rearrange(self.q(h), "b s (h d) -> b h s d", h=self.heads)
        if vlm_kv is None:
            vlm_kv = rearrange(self.kv_vlm(vlm), "b s (two h d) -> two b h s d",
                               two=2, h=self.heads)
        k_a, v_a = rearrange(self.kv_act(h), "b s (two h d) -> two b h s d",
                             two=2, h=self.heads)
        k = torch.cat([vlm_kv[0], k_a], dim=2)
        v = torch.cat([vlm_kv[1], v_a], dim=2)
        n_act, n_vlm = a.shape[1], k.shape[2] - a.shape[1]
        # mask [B, 1, n_act, n_vlm + n_act]: VLM prefix (padding-aware) + causal.
        causal = torch.tril(torch.ones(n_act, n_act, dtype=torch.bool, device=a.device))
        if vlm_padding is not None:
            prefix = vlm_padding[:, None, None, :].bool().expand(-1, 1, n_act, -1)
        else:
            prefix = torch.ones(a.shape[0], 1, n_act, n_vlm, dtype=torch.bool,
                                device=a.device)
        attn_mask = torch.cat(
            [prefix, causal[None, None].expand(a.shape[0], 1, -1, -1)], dim=-1
        )
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a + self.proj(rearrange(o, "b h s d -> b s (h d)"))
        return a + self.mlp(self.n2(a))


class _ExpertCrossLayer(nn.Module):
    """Plain cross-attention: action queries, VLM_l keys/values."""

    def __init__(self, dim: int, vlm_dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv_vlm = nn.Linear(vlm_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, a, vlm, vlm_padding=None, vlm_kv=None):
        h = self.n1(a)
        q = rearrange(self.q(h), "b s (h d) -> b h s d", h=self.heads)
        if vlm_kv is None:
            vlm_kv = rearrange(self.kv_vlm(vlm), "b s (two h d) -> two b h s d",
                               two=2, h=self.heads)
        attn_mask = None
        if vlm_padding is not None:
            attn_mask = vlm_padding[:, None, None, :].bool()
        o = F.scaled_dot_product_attention(q, vlm_kv[0], vlm_kv[1], attn_mask=attn_mask)
        a = a + self.proj(rearrange(o, "b h s d -> b s (h d)"))
        return a + self.mlp(self.n2(a))


class SmolVLAFlowMatchingHead(BasePolicyHead):
    """SmolVLA-style flow-matching action expert.

    One expert layer per VLM layer l, conditioned on that layer's hidden states
    (``per_layer_hs`` kwarg = the backbone's ``output.hidden_states`` tuple).
    Layers alternate: even l -> causal prefix self-attention over
    [VLM_l ; action tokens] (action queries only), odd l -> plain
    cross-attention (action queries, VLM_l K/V). The flow-matching timestep is
    fused into the action token embedding with an action-time MLP. At
    inference the per-layer VLM K/V are computed once and cached across all
    Euler steps.
    """

    def __init__(self, in_features: int, action_dim: int = 3,
                 fwd_pred_next_n: int = 8, num_vlm_layers: int = 16,
                 expert_dim: int = 768, expert_heads: int = 12,
                 mlp_ratio: int = 4, num_inference_steps: int = 10,
                 hidden_size: int = 768, **kwargs):
        kwargs.pop("down_sample", None)
        super().__init__(hidden_size, action_dim, **kwargs)
        self.fwd_pred_next_n = fwd_pred_next_n
        self.build_stop_head(in_features, fwd_pred_next_n)
        self.num_vlm_layers = num_vlm_layers
        self.num_inference_steps = num_inference_steps

        self.t_freq_dim = 256
        self.action_time_mlp = nn.Sequential(
            nn.Linear(action_dim + self.t_freq_dim, expert_dim),
            nn.SiLU(),
            nn.Linear(expert_dim, expert_dim),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, fwd_pred_next_n, expert_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layers = []
        for l in range(num_vlm_layers):
            cls = _ExpertSelfLayer if l % 2 == 0 else _ExpertCrossLayer
            layers.append(cls(expert_dim, in_features, expert_heads, mlp_ratio))
        self.layers = nn.ModuleList(layers)
        # One memory per VLM layer (each layer conditions on its own features).
        # One per VLM layer (each conditions on its own representation space),
        # so pooling must stay parameter-light: "mean" keeps this ~1M total
        # instead of ~500M with learned per-layer attention pooling.
        self.history = nn.ModuleList([
            _CausalHistoryMemory(in_features,
                                 n_tokens=int(kwargs.get("history_tokens", 32)),
                                 pool=str(kwargs.get("history_pool", "mean")))
            for _ in range(num_vlm_layers)
        ]) if kwargs.get("history_fusion", False) else None
        self.out_norm = nn.LayerNorm(expert_dim)
        self.out_proj = nn.Linear(expert_dim, action_dim)
        self._stash: Dict[str, torch.Tensor] = {}

    def _embed(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = _sinusoidal_embedding(t, self.t_freq_dim).to(x_t.dtype)
        t_tok = t_emb[:, None, :].expand(-1, x_t.shape[1], -1)
        return self.action_time_mlp(torch.cat([x_t, t_tok], dim=-1)) + self.pos_emb

    def _velocity(self, x_t, t, feats: Sequence[torch.Tensor], pad, kv_cache=None):
        a = self._embed(x_t, t)
        for l, layer in enumerate(self.layers):
            kv = kv_cache[l] if kv_cache is not None else None
            a = layer(a, feats[l], vlm_padding=pad, vlm_kv=kv)
        return self.out_proj(self.out_norm(a))

    def forward(self, tok_seq: torch.Tensor, actions=None, action_masks=None,
                per_layer_hs: Optional[Tuple[torch.Tensor, ...]] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None, **kwargs):
        if per_layer_hs is None:
            raise ValueError(
                "SmolVLAFlowMatchingHead needs per_layer_hs (output.hidden_states); "
                "run with the RoboLFM25VL integration patch."
            )
        if len(per_layer_hs) - 1 != self.num_vlm_layers:
            raise ValueError(
                f"num_vlm_layers={self.num_vlm_layers} but backbone produced "
                f"{len(per_layer_hs) - 1} layer outputs"
            )
        b, w = tok_seq.shape[:2]
        feats = [hs.to(tok_seq.dtype) for hs in per_layer_hs[1:]]  # layer-l outputs
        pad = None
        if encoder_attention_mask is not None:
            pad = encoder_attention_mask.reshape(feats[0].shape[0], -1).bool()
        if self.history is not None and w > 1:
            # Same post-fusion memory, applied to every layer's K/V set: the
            # action tokens are the only queries, so cost stays linear in w.
            new_feats, new_pad = [], None
            for l, f in enumerate(feats):
                mem, mem_pad = self.history[l](f, b, w, key_padding=pad)
                new_feats.append(torch.cat([f, mem], dim=1))
                if new_pad is None:
                    base = (pad if pad is not None
                            else mem_pad.new_zeros(f.shape[0], f.shape[1]).bool())
                    new_pad = torch.cat([base, mem_pad], dim=1)
            feats, pad = new_feats, new_pad

        if actions is not None and actions[0] is not None:  # training
            x1, mask = _flatten_labels(actions, action_masks, w)
            x1 = x1.to(feats[0].dtype)
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
            x0 = torch.randn_like(x1)
            x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
            v_pred = self._velocity(x_t, t, feats, pad)
            err = F.mse_loss(v_pred, x1 - x0, reduction="none").mean(-1)
            fm = err[mask].mean() if mask.any() else err.sum() * 0.0
            self._stash["loss"] = fm
            pred = (x_t + (1.0 - t[:, None, None]) * v_pred)
            return {"actions": pred.view(b, w, self.fwd_pred_next_n, self.action_dim),
                    "gripper": None}

        # inference: cache per-layer VLM K/V once, then Euler-integrate.
        kv_cache = []
        for l, layer in enumerate(self.layers):
            kv_cache.append(
                rearrange(layer.kv_vlm(feats[l]), "b s (two h d) -> two b h s d",
                          two=2, h=layer.heads)
            )
        n = feats[0].shape[0]
        x = torch.randn(n, self.fwd_pred_next_n, self.action_dim,
                        device=feats[0].device, dtype=feats[0].dtype)
        dt = 1.0 / self.num_inference_steps
        for i in range(self.num_inference_steps):
            t = torch.full((n,), i * dt, device=x.device, dtype=x.dtype)
            x = x + self._velocity(x, t, feats, pad, kv_cache=kv_cache) * dt
        return {"actions": x.view(b, w, self.fwd_pred_next_n, self.action_dim),
                "gripper": None}

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        return pred_actions, labels, action_masks

    def loss(self, pred_action, labels, attention_mask=None, **kwargs):
        if labels is None or labels[0] is None:
            return {"loss": None}
        return {"loss_arm": self._stash.pop("loss")}
