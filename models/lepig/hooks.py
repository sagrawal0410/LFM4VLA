"""Training-loop integration for LEPIG.

The machinery in this package is useless unless the training step calls it.
This module holds the call sites so the wiring is in one auditable place, and
`assert_wired()` proves at runtime that gradient actually flows -- built-but-
never-called is the failure mode that cost this project days of GPU time on
both the stop head and the first LEPIG attempt.

Scoring deviation from the plan document, stated explicitly: it specifies an
EMA scoring snapshot. Keeping an EMA copy of a 3B backbone doubles resident
memory, so the EMA is maintained over the SELECTED posterior parameters only
(which for a1/a2 is the head and the LoRA, i.e. exactly the parameters the
posterior covers). The backbone is scored from the live weights, detached.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .routing import grad_scale_identity


def selected_params(model, plan: str) -> List[torch.nn.Parameter]:
    """The parameters the posterior is defined over, per plan."""
    head = getattr(model, "act_head", None)
    if plan in ("a1",):
        return [p for p in head.parameters() if p.requires_grad] if head else []
    if plan in ("a2", "b", "c"):
        out = [p for p in head.parameters() if p.requires_grad] if head else []
        out += [p for n, p in model.named_parameters()
                if "lora" in n.lower() and p.requires_grad]
        return out
    if plan == "a3":
        return [p for p in model.parameters() if p.requires_grad]
    return []


@torch.no_grad()
def action_functional(module, batch, horizons=(1, 2, 4, 8)) -> torch.Tensor:
    """Multi-horizon prefix of the generated action chunk, whitened by 1/sqrt(H*d).

    One solve, every horizon read from it -- as the plan specifies.
    """
    pred = module._predict_waypoints(batch)          # [B, K, D]
    outs = []
    for H in horizons:
        h = min(int(H), pred.shape[1])
        outs.append(pred[:, :h].reshape(pred.shape[0], -1)
                    / (h * pred.shape[-1]) ** 0.5)
    return torch.cat(outs, dim=1)                    # [B, sum(H*D)]


def jacobian_rows(module, batch, params, subspace, n_dir: int,
                  horizons=(1, 2, 4, 8)) -> Optional[torch.Tensor]:
    """G = d g / d alpha by finite differences along each subspace direction.

    JVPs would be exact, but torch.func.jvp over a functional_call of the whole
    VLA (processor dict, stochastic head, solver) is fragile here; a central
    difference along a frozen basis direction costs 2 forwards per direction
    and is numerically adequate for a rank-12 curvature estimate.
    """
    if not params or subspace.effective_rank == 0:
        return None
    eps = 1e-3
    cols = []
    base = action_functional(module, batch, horizons)     # [B, d_g]
    for k in range(min(n_dir, subspace.effective_rank)):
        d = subspace.direction(k, params)
        with torch.no_grad():
            for p, dv in zip(params, d):
                p.add_(dv, alpha=eps)
        plus = action_functional(module, batch, horizons)
        with torch.no_grad():
            for p, dv in zip(params, d):
                p.add_(dv, alpha=-2 * eps)
        minus = action_functional(module, batch, horizons)
        with torch.no_grad():
            for p, dv in zip(params, d):
                p.add_(dv, alpha=eps)                      # restore
        cols.append(((plus - minus) / (2 * eps)))
    return torch.stack(cols, dim=-1)                       # [B, d_g, r]


def route_backbone(hidden, weights):
    """Scale ONLY the backbone's gradient by the per-example weight."""
    return grad_scale_identity(hidden, weights)


def assert_wired(module, batch, verbose=True) -> dict:
    """Prove the wiring is live: run one step and check gradient actually flows.

    Returns a dict of measured facts rather than a boolean, so a caller can
    print exactly which piece is dead.
    """
    out = {}
    lep = getattr(module, "lepig", None)
    out["controller"] = bool(lep and lep.enabled)
    out["plan"] = getattr(lep, "plan", None)
    wb = getattr(module, "world_branch", None)
    out["world_branch_built"] = wb is not None

    for p in module.parameters():
        p.grad = None
    res = module.training_step(batch, 0)
    loss = res["loss"] if isinstance(res, dict) else res
    out["loss"] = float(loss.detach())
    loss.backward()

    if wb is not None:
        g = [p.grad for p in wb.parameters() if p.grad is not None]
        out["world_grad_norm"] = (sum(float(x.norm()) ** 2 for x in g) ** 0.5
                                  if g else 0.0)
        out["world_params"] = sum(p.numel() for p in wb.parameters())
    bb = [p for n, p in module.model.named_parameters()
          if "act_head" not in n and p.grad is not None]
    out["backbone_grad_norm"] = sum(float(p.grad.norm()) ** 2 for p in bb) ** 0.5
    if verbose:
        for k, v in out.items():
            print(f"    {k:22s} {v}")
    return out
