"""LEPIG controller: owns the scoring snapshot, posterior, and per-example weights.

One object per run, driven entirely by the `lepig` config block. It is a no-op
until warmup completes, then refreshes on a fixed step interval so the cost of
JVP scoring is amortised (scoring runs once per `refresh_steps`, not per step).

Plan variants
    a1 : posterior over the flow/action head only
    a2 : flow head + LoRA in the last 4 VLM blocks   (recommended default)
    a3 : full-network SGD-trajectory subspace
    b  : world-latent PIG, weights the WORLD loss only
    c  : world-latent PIG, additionally routes the FM gradient into the backbone

Statistical honesty: in the full-data regime a candidate is already inside the
data that produced Sigma_D, so re-adding its Fisher term is not exact Bayesian
information gain. Runs in that regime are labelled `pig_inspired_curriculum`
and must be validated against realized learning gain, not reported as
acquisition.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch

from .posterior import SubspacePosterior
from .routing import robust_weight
from .subspace import TrajectorySubspace

ACTION_PLANS = ("a1", "a2", "a3")
WORLD_PLANS = ("b", "c")


class LepigController:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.plan = str(self.cfg.get("plan", "")).lower()
        self.enabled = self.plan in ACTION_PLANS + WORLD_PLANS
        self.rank = int(self.cfg.get("posterior_rank", 12))
        self.refresh_steps = int(self.cfg.get("refresh_steps", 2000))
        self.min_warmup = int(self.cfg.get("min_warmup_steps", 5000))
        self.warmup_fraction = float(self.cfg.get("warmup_fraction", 0.20))
        self.anchor_count = int(self.cfg.get("anchor_count", 64))
        self.obs_var = float(self.cfg.get("obs_var", 1.0))
        st = self.cfg.get("score_transform", {}) or {}
        self.z_clip = float(st.get("robust_z_clip", 2.0))
        self.exp_scale = float(st.get("exp_scale", 0.35))
        self.w_min = float(st.get("weight_min", 0.5))
        self.w_max = float(st.get("weight_max", 2.0))

        self.subspace = TrajectorySubspace(
            rank=self.rank,
            n_snapshots=int(self.cfg.get("trajectory_snapshots", 13)),
            window_steps=int(self.cfg.get("trajectory_window_steps", 2000)))
        self.posterior: Optional[SubspacePosterior] = None
        self.anchors: List[torch.Tensor] = []
        self.snapshot_id = 0
        self._warmed = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def weights_the_world_loss(self) -> bool:
        return self.plan in WORLD_PLANS

    @property
    def routes_backbone_fm_grad(self) -> bool:
        """Plan C (and all action plans) scale the FM gradient into the backbone.

        Plan B deliberately does NOT: its score weights only the world loss, so
        the action supervision stays uniform.
        """
        return self.plan in ACTION_PLANS or self.plan == "c"

    def warm(self, step: int, max_steps: int) -> bool:
        thresh = max(self.min_warmup, int(self.warmup_fraction * max(max_steps, 1)))
        self._warmed = step >= thresh
        return self._warmed

    def on_step(self, step: int, params: Iterable[torch.nn.Parameter]):
        """Capture trajectory snapshots; cheap and safe to call every step."""
        if not self.enabled:
            return
        self.subspace.maybe_capture(step, params)

    def should_refresh(self, step: int) -> bool:
        return (self.enabled and self._warmed
                and step % self.refresh_steps == 0)

    def refresh(self, all_reduce=None) -> bool:
        """Rebuild the basis and reset curvature. Invalidates cached Jacobians."""
        if not self.subspace.build(all_reduce=all_reduce):
            return False
        r = self.subspace.effective_rank
        self.posterior = SubspacePosterior(
            rank=r,
            prior_precision=float(self.cfg.get("prior_precision", 1.0)),
            jitter_rel=float(self.cfg.get("jitter_rel", 1e-5)))
        self.anchors = []
        self.snapshot_id += 1        # never reuse Jacobians across snapshots
        return True

    # -- curvature / anchors ------------------------------------------------
    def add_calibration(self, G: torch.Tensor):
        if self.posterior is not None:
            self.posterior.add_fisher(
                SubspacePosterior.fisher_from_jac(G, self.obs_var))

    def add_anchor(self, G: torch.Tensor):
        if len(self.anchors) < self.anchor_count:
            self.anchors.append(G.detach().float())

    @property
    def ready(self) -> bool:
        return (self.enabled and self._warmed and self.posterior is not None
                and len(self.anchors) > 0)

    # -- scoring ------------------------------------------------------------
    def score_batch(self, G_batch: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.ready:
            return None
        return torch.stack([self.posterior.pig(G, self.anchors, self.obs_var)
                            for G in G_batch])

    def weights(self, scores: Optional[torch.Tensor], batch_size: int,
                device=None) -> torch.Tensor:
        """Detached per-example weights; all-ones when no fresh score exists."""
        if scores is None or scores.numel() != batch_size:
            return torch.ones(batch_size, device=device)
        return robust_weight(scores, self.z_clip, self.exp_scale,
                             self.w_min, self.w_max).to(device)

    # -- reporting ----------------------------------------------------------
    def regime_label(self) -> str:
        return ("clean_acquisition" if self.cfg.get("acquisition_mode", False)
                else "pig_inspired_curriculum")
