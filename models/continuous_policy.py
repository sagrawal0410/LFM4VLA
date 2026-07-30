"""Fully-continuous action head for Humanoid Everyday (no binary gripper split).

Kept separate from ``base_policy.FCDecoder`` so LIBERO/CALVIN's two-head
(arm tanh + gripper sigmoid) architecture stays untouched.
"""

from __future__ import annotations

from einops import rearrange

import torch

from models.base_policy import BasePolicyHead, MLPTanhHead, initialize_param


class FCContinuousDecoder(BasePolicyHead):
    """Predict all continuous joint dims with one Tanh MLP (Huber loss).

    Two layers on top of the VLM action-token hidden states:
      1. ``self.mlp`` — Linear → ReLU → Linear over concatenated tokens
      2. ``self.actions`` — ``MLPTanhHead`` for all ``action_dim`` joints
         (G1 Humanoid Everyday: 28 = 14 arm + 14 Dex3 hand)

    ``latent`` / ``n_tokens`` is the number of concatenated action-token features
    (``num_action_tokens * token_repeat`` from the HE layout helpers).
    """

    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        n_tokens=None,
        **kwargs,
    ):
        kwargs.pop("with_history", None)
        kwargs.pop("history_type", None)
        kwargs.pop("window_size", None)
        kwargs.pop("tokenizer", None)
        super().__init__(hidden_size, action_dim, **kwargs)
        self.in_features = in_features
        self.fwd_pred_next_n = fwd_pred_next_n
        self.down_sample = down_sample
        self.n_tokens = int(n_tokens if n_tokens is not None else latent)
        self.latent = self.n_tokens
        self.action_dim = action_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features * self.n_tokens, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, hidden_size),
        )
        self.actions = MLPTanhHead(hidden_size, fwd_pred_next_n * action_dim)

        if self.down_sample == "pooling":
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif self.down_sample in ("resampler", "none"):
            pass
        else:
            raise NotImplementedError(self.down_sample)
        initialize_param(self)

    def forward(self, tok_seq, **kwargs):
        if len(tok_seq.shape) == 4:
            bs, seq_len, n_tok, tok_dim = tok_seq.shape
            tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
        elif tok_seq.dim() == 3:
            bs, n_tok, tok_dim = tok_seq.shape
            seq_len = None
        else:
            assert len(tok_seq.shape) == 2
            bs, tok_dim = tok_seq.shape
            seq_len = None
            n_tok = None
            tok_seq = tok_seq.unsqueeze(1)

        if self.down_sample == "pooling":
            tok_seq = self.pooling(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, "b d n -> b (n d)")
        elif self.down_sample == "none":
            if n_tok is not None and n_tok != self.n_tokens:
                raise ValueError(
                    f"FCContinuousDecoder expected {self.n_tokens} action tokens, "
                    f"got {n_tok}. Check num_action_tokens * latent."
                )
            tok_seq = rearrange(tok_seq, "b n d -> b (n d)")
        else:
            raise NotImplementedError(self.down_sample)

        tok_seq = self.mlp(tok_seq)
        actions = self.actions(tok_seq)
        if seq_len is not None:
            actions = rearrange(
                actions, "(b l) (n d) -> b l n d", b=bs, l=seq_len, n=self.fwd_pred_next_n
            )
        elif n_tok is not None:
            actions = rearrange(actions, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n)
        return actions

    def loss(self, pred_action, labels, attention_mask=None):
        """Huber over all dims. ``labels`` is ``(action_chunck, None)`` from the trainer."""
        if labels is None:
            return {"loss": None}
        target = labels[0] if isinstance(labels, (tuple, list)) else labels
        if target is None:
            return {"loss": None}

        if attention_mask is None:
            action_loss = torch.nn.functional.huber_loss(pred_action, target)
        else:
            action_loss = torch.nn.functional.huber_loss(
                pred_action, target, reduction="none"
            )
            action_loss = action_loss[attention_mask.bool()].mean()
        return {"loss_arm": action_loss}
