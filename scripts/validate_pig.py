"""Does PIG actually predict realized learning gain?

This is the mechanism test the plan document requires (falsification checklist
items 7-9). It does not train a model to convergence and read off SR; it asks
the claim directly:

    if PIG says sample i is informative, does one gradient step on sample i
    reduce held-out loss more than a step on a low-PIG sample?

and it asks it against the acquisition baselines a reviewer will reach for:

    random          -- the null. Any scorer must beat this.
    loss            -- "train on what you get wrong". The cheap baseline.
    grad_norm       -- "train on what moves the weights".
    raw_epistemic   -- posterior predictive variance, WITHOUT the cross-context
                       expectation. Isolates whether the "predictive" part of
                       PIG earns its cost.
    param_ig        -- information about PARAMETERS rather than predictions.
                       High param-IG with low PIG is the pathology PIG exists
                       to avoid, so beating it is the sharpest claim.

Protocol per trial:
  1. score a candidate pool with every method
  2. for each method take its top-k as a batch
  3. clone the model, take ONE optimizer step on that batch
  4. measure loss on a FIXED held-out set, never scored and never trained on
  5. gain = held_out_loss(before) - held_out_loss(after)

Controls that make the result mean something:
  - identical batch size across methods, so gain is not a data-volume effect
  - identical starting weights and optimizer state per trial (clone, not
    continue), so methods do not inherit each other's updates
  - the SAME held-out set for every method and trial
  - a bottom-k arm for PIG: if high-PIG beats low-PIG, the ordering is real
    and not just "PIG happens to pick a different-sized gradient"
  - multiple trials with different pools, reporting mean and spread, because a
    single trial at this scale is noise
"""
from __future__ import annotations

import argparse
import copy
import json
import types

import numpy as np
import torch


def attach_stub_trainer(m):
    m._trainer = types.SimpleNamespace(
        world_size=1, global_rank=0, local_rank=0, num_devices=1,
        global_step=0, current_epoch=0, max_steps=10,
        estimated_stepping_batches=10, barebones=False, loggers=[],
        log_dir=None, state=None, sanity_checking=False)
    m.log = lambda *a, **k: None
    m.log_dict = lambda *a, **k: None
    return m


@torch.no_grad()
def heldout_loss(module, batches):
    """Mean waypoint loss over a fixed held-out set."""
    tot, n = 0.0, 0
    was = module.training
    module.eval()
    for b in batches:
        pred = module._predict_waypoints(b)
        gt = b["action_chunck"][:, -1].to(pred.device).float()
        mask = b["chunck_mask"][:, -1].to(pred.device).float()
        d = ((pred - gt) ** 2).sum(-1).sqrt()          # [B, K]
        tot += float((d * mask).sum() / mask.sum().clamp(min=1))
        n += 1
    if was:
        module.train()
    return tot / max(n, 1)


def per_sample_stats(module, samples, ds, lepig, params):
    """loss, grad-norm, PIG, raw-epistemic and param-IG for each candidate."""
    from models.lepig.hooks import jacobian_rows
    out = []
    for s in samples:
        b = ds.collater([s])
        # --- loss + grad norm (one backward, discarded) --------------------
        for p in module.parameters():
            p.grad = None
        pred = module._predict_waypoints(b)
        gt = b["action_chunck"][:, -1].to(pred.device).float()
        mask = b["chunck_mask"][:, -1].to(pred.device).float()
        loss = (((pred - gt) ** 2).sum(-1).sqrt() * mask).sum() / mask.sum().clamp(min=1)
        loss.backward()
        gn = sum(float(p.grad.norm()) ** 2
                 for p in params if p.grad is not None) ** 0.5
        for p in module.parameters():
            p.grad = None
        rec = {"loss": float(loss.detach()), "grad_norm": gn,
               "pig": 0.0, "raw_epi": 0.0, "param_ig": 0.0}
        # --- posterior-based scores ---------------------------------------
        if lepig is not None and lepig.ready:
            G = jacobian_rows(module, b, params, lepig.subspace, lepig.rank)
            if G is not None:
                g = G[0]
                try:
                    rec["pig"] = float(lepig.posterior.pig(g, lepig.anchors))
                except Exception:
                    pass
                try:
                    rec["raw_epi"] = float(lepig.posterior.raw_epistemic(g))
                except Exception:
                    pass
                try:
                    rec["param_ig"] = float(lepig.posterior.parameter_ig(g))
                except Exception:
                    pass
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", type=int, default=64, help="candidates per trial")
    ap.add_argument("--topk", type=int, default=8, help="batch size per method")
    ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cfg["train_dataset"]["mixture_mode"] = "batch"
    cfg["train_dataset"]["mixture_trajectory"] = 1.0
    cfg["batch_size"] = 1
    # engage scoring immediately -- this is an offline probe, not a training run
    if "lepig" in cfg:
        cfg["lepig"]["min_warmup_steps"] = 0
        cfg["lepig"]["warmup_fraction"] = 0.0
        cfg["lepig"]["refresh_steps"] = 1
        cfg["lepig"]["trajectory_snapshots"] = 3
        cfg["lepig"]["trajectory_window_steps"] = 3

    from train.experiment_utils import prepare_experiment
    from train.robotnav_trainer import RobotNavTrainer
    from data.build_dataset import build_dataset
    from models.lepig.hooks import selected_params, jacobian_rows

    cfg, *_ = prepare_experiment(cfg)
    base = RobotNavTrainer.from_checkpoint(args.ckpt, "torch", cfg)
    base.train(); base.float(); attach_stub_trainer(base)
    ds = build_dataset(cfg["train_dataset"], cfg, base.model)

    def take(n):
        got = []
        while len(got) < n:
            s = next(it)
            if s.get("sample_type") == "traj":
                got.append(s)
        return got

    it = iter(ds)
    held_raw = take(args.heldout)
    held = [ds.collater([s]) for s in held_raw]        # fixed, never trained on

    lepig = getattr(base, "lepig", None)
    params = selected_params(base.model, getattr(lepig, "plan", "a1")) if lepig else []
    # prime the posterior: snapshots + curvature + anchors, from data that is
    # neither the candidate pool nor the held-out set
    if lepig is not None and lepig.enabled:
        prime = take(8)
        for step in range(4):
            lepig.on_step(step, params)
            lepig.warm(step, 10)
        if lepig.should_refresh(0):
            lepig.refresh()
        for s in prime:
            G = jacobian_rows(base, ds.collater([s]), params, lepig.subspace, lepig.rank)
            if G is not None:
                lepig.add_calibration(G[0]); lepig.add_anchor(G[0])
        print(f"  posterior primed: rank={lepig.subspace.effective_rank} "
              f"anchors={len(lepig.anchors)} ready={lepig.ready}", flush=True)

    METHODS = ["pig", "pig_bottom", "loss", "grad_norm", "raw_epi",
               "param_ig", "random"]
    gains = {m: [] for m in METHODS}

    for trial in range(args.trials):
        pool = take(args.pool)
        stats = per_sample_stats(base, pool, ds, lepig, params)
        base_loss = heldout_loss(base, held)
        order = {
            "pig":       np.argsort([-s["pig"] for s in stats]),
            "pig_bottom": np.argsort([s["pig"] for s in stats]),
            "loss":      np.argsort([-s["loss"] for s in stats]),
            "grad_norm": np.argsort([-s["grad_norm"] for s in stats]),
            "raw_epi":   np.argsort([-s["raw_epi"] for s in stats]),
            "param_ig":  np.argsort([-s["param_ig"] for s in stats]),
            "random":    np.random.default_rng(trial).permutation(len(pool)),
        }
        line = []
        for m in METHODS:
            idx = list(order[m])[: args.topk]
            batch = ds.collater([pool[i] for i in idx])
            clone = copy.deepcopy(base)                 # identical start state
            attach_stub_trainer(clone)
            opt = torch.optim.AdamW(
                [p for p in clone.parameters() if p.requires_grad], lr=args.lr)
            opt.zero_grad()
            pred = clone._predict_waypoints(batch)
            gt = batch["action_chunck"][:, -1].to(pred.device).float()
            mask = batch["chunck_mask"][:, -1].to(pred.device).float()
            l = (((pred - gt) ** 2).sum(-1).sqrt() * mask).sum() / mask.sum().clamp(min=1)
            l.backward(); opt.step()
            g = base_loss - heldout_loss(clone, held)
            gains[m].append(g)
            line.append(f"{m}={g:+.5f}")
            del clone, opt
        print(f"  trial {trial}: base_loss={base_loss:.5f}  " + "  ".join(line),
              flush=True)

    print("\n  REALIZED HELD-OUT LOSS REDUCTION after one step (higher is better)")
    print("  %-12s %10s %10s %8s" % ("method", "mean gain", "std", "trials"))
    print("  " + "-" * 44)
    ranked = sorted(METHODS, key=lambda m: -float(np.mean(gains[m])))
    for m in ranked:
        v = np.array(gains[m])
        print("  %-12s %10.5f %10.5f %8d" % (m, v.mean(), v.std(), len(v)))
    top = ranked[0]
    print(f"\n  best acquisition signal: {top}")
    pig_m, rnd_m = np.mean(gains["pig"]), np.mean(gains["random"])
    lo_m, bot_m = np.mean(gains["loss"]), np.mean(gains["pig_bottom"])
    print(f"  PIG vs random      : {pig_m - rnd_m:+.5f}")
    print(f"  PIG vs loss        : {pig_m - lo_m:+.5f}")
    print(f"  PIG top vs bottom  : {pig_m - bot_m:+.5f}  "
          f"(ordering is {'REAL' if pig_m > bot_m else 'NOT supported'})")
    if top == "pig" and pig_m > rnd_m and pig_m > bot_m:
        print("  -> PIG is the best predictor of realized learning gain.")
    else:
        print("  -> PIG does NOT lead. The mechanism is not validated at this "
              "checkpoint.")


if __name__ == "__main__":
    main()
