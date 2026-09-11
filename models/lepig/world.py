"""World-transition latent branch for plans B and C.

Small by design: 8 query tokens, a 2-layer bridge and a 6-block predictor, all
at width 768 regardless of backbone size (only the input/output projections
adapt). Against a 3B backbone this is a few tens of millions of parameters, so
we are not training a large adapter.

Targets come from a FROZEN V-JEPA2 encoder, pooled 256 -> 64 tokens, with the
loss taken as LayerNorm-then-L1 on the pooled latents and a stop-gradient on
the target. For Fisher/PIG only, a SEPARATE frozen PCA-whitening maps the
target space down to 128 dims; it is fitted once on training data and never
backpropagated through, so a trainable projection cannot collapse the space.

Action conditioning is optional. The plan treats it as a real hypothesis
rather than an obvious truth: V-JEPA2-AC supports it, but VLAFlow deliberately
blocks latent tokens from attending to action tokens to stop the predictor
shortcutting through the action trajectory. `action_conditioned=False` gives
the actionless arm.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(width: int, heads: int, mlp_ratio: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=width, nhead=heads, dim_feedforward=width * mlp_ratio,
        dropout=dropout, batch_first=True, norm_first=True,
        activation="gelu")


class WorldBranch(nn.Module):
    def __init__(self, in_features: int, action_dim: int = 3,
                 world_query_tokens: int = 8,
                 bridge_layers: int = 2, bridge_hidden: int = 768, bridge_heads: int = 12,
                 predictor_layers: int = 6, predictor_hidden: int = 768,
                 predictor_heads: int = 12, predictor_mlp_ratio: int = 4,
                 dropout: float = 0.0, n_horizons: int = 3,
                 target_pool_tokens: int = 64, target_dim: int = 1024,
                 action_conditioned: bool = True, max_action_steps: int = 16,
                 **_):
        super().__init__()
        self.action_conditioned = bool(action_conditioned)
        self.n_horizons = int(n_horizons)
        self.target_pool_tokens = int(target_pool_tokens)

        w = int(bridge_hidden)
        self.in_proj = nn.Linear(in_features, w)
        self.queries = nn.Parameter(torch.randn(1, int(world_query_tokens), w) * 0.02)
        self.bridge = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=w, nhead=bridge_heads,
                                        dim_feedforward=w * 4, dropout=dropout,
                                        batch_first=True, norm_first=True,
                                        activation="gelu")
             for _ in range(int(bridge_layers))])

        pw = int(predictor_hidden)
        self.bridge_to_pred = nn.Linear(w, pw) if w != pw else nn.Identity()
        self.act_embed = nn.Linear(action_dim, pw) if self.action_conditioned else None
        self.act_pos = (nn.Parameter(torch.randn(1, max_action_steps, pw) * 0.02)
                        if self.action_conditioned else None)
        # One query per (horizon, output token). Projecting per token keeps the
        # output head at pw x target_dim instead of pw x (pool * target_dim),
        # which is the difference between ~1M and ~50M parameters.
        self.horizon_q = nn.Parameter(
            torch.randn(1, self.n_horizons * int(target_pool_tokens), pw) * 0.02)
        self.predictor = nn.ModuleList(
            [_block(pw, predictor_heads, predictor_mlp_ratio, dropout)
             for _ in range(int(predictor_layers))])
        self.out_norm = nn.LayerNorm(pw)
        self.out_proj = nn.Linear(pw, target_dim)      # applied per token
        self.target_dim = int(target_dim)

    def forward(self, feats: torch.Tensor, actions: torch.Tensor | None = None,
                action_prefix_len: list[int] | None = None) -> torch.Tensor:
        """feats [B, N, D] -> predicted latents [B, n_horizons, pool, target_dim]."""
        b = feats.shape[0]
        mem = self.in_proj(feats)
        q = self.queries.expand(b, -1, -1)
        for layer in self.bridge:
            q = layer(q, mem)
        z = self.bridge_to_pred(q)                       # [B, Q, pw]

        seq = [z]
        if self.action_conditioned and actions is not None:
            a = self.act_embed(actions)                  # [B, T, pw]
            a = a + self.act_pos[:, : a.shape[1]]
            seq.append(a)
        hq = self.horizon_q.expand(b, -1, -1)
        seq.append(hq)
        x = torch.cat(seq, dim=1)

        mask = None
        if self.action_conditioned and actions is not None and action_prefix_len:
            # Each horizon query may see only the action prefix available to it;
            # without this the long-horizon query reads actions from the future
            # it is meant to predict.
            n_z, n_a = z.shape[1], actions.shape[1]
            total = n_z + n_a + self.n_horizons * self.target_pool_tokens
            mask = torch.zeros(total, total, dtype=torch.bool, device=x.device)
            P = self.target_pool_tokens
            for h, plen in enumerate(action_prefix_len[: self.n_horizons]):
                if plen < n_a:
                    rows = slice(n_z + n_a + h * P, n_z + n_a + (h + 1) * P)
                    mask[rows, n_z + plen: n_z + n_a] = True   # True = blocked
        for layer in self.predictor:
            x = layer(x, src_mask=mask)
        n_out = self.n_horizons * self.target_pool_tokens
        out = self.out_norm(x[:, -n_out:])
        return self.out_proj(out).view(b, self.n_horizons,
                                       self.target_pool_tokens, self.target_dim)

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """LayerNorm then mean L1, stop-gradient on the target."""
        t = target.detach()
        ln = lambda v: torch.nn.functional.layer_norm(v, (v.shape[-1],))
        return (ln(pred) - ln(t)).abs().mean()


class FrozenWhitening(nn.Module):
    """Fixed PCA-whitening used ONLY to build the PIG functional.

    Fitted once on training data and frozen for the whole experiment. Never in
    the loss path, never backpropagated through -- a trainable projection here
    could trivially collapse the space and make the score meaningless.
    """

    def __init__(self, dim_in: int, dim_out: int = 128):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim_in))
        self.register_buffer("W", torch.zeros(dim_out, dim_in))
        self.fitted = False

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        """X: [N, dim_in] pooled training targets."""
        X = X.float()
        self.mean.copy_(X.mean(0))
        Xc = X - self.mean
        # economy SVD; components scaled by 1/sqrt(eigenvalue) to whiten
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        k = self.W.shape[0]
        comp = Vh[:k]
        scale = (S[:k] / max(X.shape[0] - 1, 1) ** 0.5).clamp_min(1e-6)
        self.W.copy_(comp / scale.unsqueeze(1))
        self.fitted = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) @ self.W.t()
