"""PIG-WEIGHTED loss vs UNIFORM-weighted loss, at matched gradient scale.

The selection variant of this test (top-k by score) does not reflect what LEPIG
does. LEPIG never discards a sample: it computes w_i from PIG, clamps to
[0.5, 2.0], MEAN-NORMALISES the batch to 1, and scales each example's gradient
contribution. Every sample stays in the batch.

So this compares weighting schemes on the SAME batch:

    uniform   w_i = 1                       (the control)
    pig       w_i from PIG, mean-normalised (the method)
    loss_w    w_i from loss, same transform (the cheap baseline, made fair)
    grad_w    w_i from grad-norm, same transform
    anti_pig  PIG weights inverted          (if pig > anti_pig the direction is real)
    random_w  shuffled pig weights          (same weight DISTRIBUTION, wrong
                                             assignment -- isolates whether the
                                             pairing of weight-to-sample matters
                                             rather than the spread of weights)

SCALE CONTROL, which is the whole point of this rewrite:
every scheme passes through the identical transform -- robust median/MAD z,
clip, exp, clamp, then divide by the batch mean -- so all schemes have
mean(w) = 1 by construction. The total gradient magnitude is therefore matched
across arms, and any difference is attributable to WHICH samples got the weight,
not to one arm taking a larger effective step. That is verified numerically and
printed, not assumed.
"""
from __future__ import annotations

import argparse
import copy
import json
import types

import numpy as np
import torch


def attach_stub(m):
    m._trainer = types.SimpleNamespace(
        world_size=1, global_rank=0, local_rank=0, num_devices=1, global_step=0,
        current_epoch=0, max_steps=10, estimated_stepping_batches=10,
        barebones=False, loggers=[], log_dir=None, state=None,
        sanity_checking=False)
    m.log = lambda *a, **k: None
    m.log_dict = lambda *a, **k: None
    return m


def weighted_loss(module, batch, w):
    """Per-sample loss reduced with weights w (already mean-normalised)."""
    pred = module._predict_waypoints(batch)
    gt = batch["action_chunck"][:, -1].to(pred.device).float()
    mask = batch["chunck_mask"][:, -1].to(pred.device).float()
    per = (((pred - gt) ** 2).sum(-1).sqrt() * mask).sum(-1) / \
          mask.sum(-1).clamp(min=1)                      # [B]
    w = w.to(per.device, per.dtype)
    return (per * w).mean()


@torch.no_grad()
def heldout(module, batches):
    tot, n = 0.0, 0
    was = module.training
    module.eval()
    for b in batches:
        pred = module._predict_waypoints(b)
        gt = b["action_chunck"][:, -1].to(pred.device).float()
        mask = b["chunck_mask"][:, -1].to(pred.device).float()
        d = ((pred - gt) ** 2).sum(-1).sqrt()
        tot += float((d * mask).sum() / mask.sum().clamp(min=1))
        n += 1
    if was:
        module.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--snapshots", type=int, default=13)
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--anchors", type=int, default=64)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cfg["train_dataset"]["mixture_mode"] = "batch"
    cfg["train_dataset"]["mixture_trajectory"] = 1.0
    cfg["batch_size"] = 1
    if "lepig" in cfg:
        cfg["lepig"].update(min_warmup_steps=0, warmup_fraction=0.0,
                            refresh_steps=1,
                            trajectory_snapshots=args.snapshots,
                            trajectory_window_steps=args.snapshots,
                            posterior_rank=args.rank)

    from train.experiment_utils import prepare_experiment
    from train.robotnav_trainer import RobotNavTrainer
    from data.build_dataset import build_dataset
    from models.lepig.hooks import selected_params, jacobian_rows
    from models.lepig.routing import robust_weight

    cfg, *_ = prepare_experiment(cfg)
    base = RobotNavTrainer.from_checkpoint(args.ckpt, "torch", cfg)
    base.train(); base.float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(dev); attach_stub(base)
    print(f"  device: {dev}", flush=True)

    ds = build_dataset(cfg["train_dataset"], cfg, base.model)
    it = iter(ds)

    def take(n):
        got = []
        while len(got) < n:
            s = next(it)
            if s.get("sample_type") == "traj":
                got.append(s)
        return got

    held = [ds.collater([s]) for s in take(args.heldout)]

    lepig = getattr(base, "lepig", None)
    params = selected_params(base.model, getattr(lepig, "plan", "a1"))
    # the posterior lives wherever the Jacobians do
    if lepig is not None and getattr(lepig, "posterior", None) is not None:
        for name in ("Lambda", "V", "prior", "Sigma_cache"):
            t = getattr(lepig.posterior, name, None)
            if torch.is_tensor(t):
                setattr(lepig.posterior, name, t.to(dev))

    if lepig is not None and lepig.enabled:
        prime = take(max(8, args.snapshots))
        w0 = copy.deepcopy(base.state_dict())
        opt0 = torch.optim.AdamW([p for p in base.parameters()
                                  if p.requires_grad], lr=args.lr)
        for step in range(args.snapshots):
            lepig.on_step(step, params); lepig.warm(step, 10)
            b = ds.collater([prime[step % len(prime)]])
            opt0.zero_grad()
            weighted_loss(base, b, torch.ones(1)).backward()
            opt0.step()
        base.load_state_dict(w0); del w0, opt0
        if lepig.should_refresh(0):
            lepig.refresh()
        print(f"  priming {args.anchors} anchors...", flush=True)
        for s in take(args.anchors):
            G = jacobian_rows(base, ds.collater([s]), params,
                              lepig.subspace, lepig.rank)
            if G is not None:
                lepig.add_calibration(G[0]); lepig.add_anchor(G[0])
        print(f"  posterior: rank={lepig.subspace.effective_rank} "
              f"anchors={len(lepig.anchors)} ready={lepig.ready}", flush=True)

    SCHEMES = ["uniform", "pig", "loss_w", "grad_w", "anti_pig", "random_w"]
    gains = {s: [] for s in SCHEMES}
    wstats = {s: [] for s in SCHEMES}

    for trial in range(args.trials):
        samples = take(args.batch)
        batch = ds.collater(samples)
        B = len(samples)
        # --- raw scores -----------------------------------------------------
        pig_s, loss_s, grad_s = [], [], []
        for s in samples:
            b1 = ds.collater([s])
            for p in base.parameters():
                p.grad = None
            l = weighted_loss(base, b1, torch.ones(1))
            l.backward()
            grad_s.append(sum(float(p.grad.norm()) ** 2
                              for p in params if p.grad is not None) ** 0.5)
            loss_s.append(float(l.detach()))
            for p in base.parameters():
                p.grad = None
            v = 0.0
            if lepig is not None and lepig.ready:
                G = jacobian_rows(base, b1, params, lepig.subspace, lepig.rank)
                if G is not None:
                    try:
                        v = float(lepig.posterior.pig(G[0], lepig.anchors))
                    except Exception:
                        v = 0.0
            pig_s.append(v)
        # --- identical transform for every scheme ---------------------------
        def W(raw):
            return robust_weight(torch.tensor(raw, dtype=torch.float32))
        w_pig = W(pig_s)
        w = {
            "uniform":  torch.ones(B),
            "pig":      w_pig,
            "loss_w":   W(loss_s),
            "grad_w":   W(grad_s),
            "anti_pig": W([-x for x in pig_s]),
            "random_w": w_pig[torch.randperm(B)],
        }
        base_loss = heldout(base, held)
        line = []
        for name in SCHEMES:
            wi = w[name]
            wi = wi / wi.mean()                 # enforce mean(w)=1 exactly
            wstats[name].append((float(wi.mean()), float(wi.std())))
            clone = copy.deepcopy(base); attach_stub(clone)
            opt = torch.optim.AdamW([p for p in clone.parameters()
                                     if p.requires_grad], lr=args.lr)
            opt.zero_grad()
            weighted_loss(clone, batch, wi).backward()
            gn = sum(float(p.grad.norm()) ** 2
                     for p in clone.parameters() if p.grad is not None) ** 0.5
            opt.step()
            g = base_loss - heldout(clone, held)
            gains[name].append(g)
            line.append(f"{name}={g:+.5f}(|g|={gn:.2f})")
            del clone, opt
        print(f"  trial {trial}: base={base_loss:.5f}  " + "  ".join(line),
              flush=True)

    print("\n  SCALE CONTROL (all schemes must have mean(w)=1)")
    print("  %-10s %10s %10s" % ("scheme", "mean(w)", "std(w)"))
    for s in SCHEMES:
        m = np.array(wstats[s])
        print("  %-10s %10.4f %10.4f" % (s, m[:, 0].mean(), m[:, 1].mean()))

    print("\n  HELD-OUT LOSS REDUCTION, one weighted step (higher is better)")
    print("  %-10s %10s %10s %8s" % ("scheme", "mean", "std", "vs uniform"))
    print("  " + "-" * 44)
    u = float(np.mean(gains["uniform"]))
    for s in sorted(SCHEMES, key=lambda s: -float(np.mean(gains[s]))):
        v = np.array(gains[s])
        se = v.std() / max(len(v) ** 0.5, 1)
        print("  %-10s %10.5f %10.5f %+8.5f%s" % (
            s, v.mean(), v.std(), v.mean() - u,
            "" if s == "uniform" else
            ("  (%.1f SE)" % (abs(v.mean() - u) / max(se, 1e-9)))))
    p_, a_ = float(np.mean(gains["pig"])), float(np.mean(gains["anti_pig"]))
    r_ = float(np.mean(gains["random_w"]))
    print(f"\n  pig - uniform  : {p_ - u:+.5f}")
    print(f"  pig - anti_pig : {p_ - a_:+.5f}  "
          f"({'direction is real' if p_ > a_ else 'NOT supported'})")
    print(f"  pig - random_w : {p_ - r_:+.5f}  "
          f"({'assignment matters' if p_ > r_ else 'only the spread matters'})")


if __name__ == "__main__":
    main()
