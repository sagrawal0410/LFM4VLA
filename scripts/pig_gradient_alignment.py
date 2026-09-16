"""Does PIG weighting produce a better descent direction than uniform?

Measures gradient ALIGNMENT rather than realized loss after a step:

    g_ho  = gradient of held-out loss w.r.t. the selected parameters
    g_w   = weighted batch gradient under weighting scheme w
    score = cos(g_w, -g_ho)

Higher cosine means the weighted batch gradient points further downhill on data
the model has not trained on -- i.e. the weighting extracted a direction that
generalises. This is the quantity a weighting scheme can actually influence.

Why this replaces the clone-and-step design:
  - no model copies, no optimizer step -> ~10x cheaper, no memory growth
  - cosine is SCALE-INVARIANT, so the step-size confound disappears. Under the
    previous design mean(w)=1 did not imply matched |g| (observed 2.89-6.61
    across arms), so part of any difference was step size rather than direction.
  - far lower variance than a one-step loss delta, so 50+ trials is affordable
    and the comparison can actually resolve small effects.

Schemes (identical robust transform for every one, so only the ASSIGNMENT of
weight to sample differs):
    uniform    w = 1                      control
    pig        PIG                        the method
    loss_w     per-sample loss            the cheap baseline, made fair
    grad_w     per-sample grad norm
    anti_pig   -PIG                       pig > anti_pig => direction is real
    random_w   PIG's weights, shuffled    same weight DISTRIBUTION, wrong
                                          pairing => isolates whether PIG picks
                                          the right samples or merely spreads
                                          weights at all
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


def per_sample_losses(module, batch):
    pred = module._predict_waypoints(batch)
    gt = batch["action_chunck"][:, -1].to(pred.device).float()
    mask = batch["chunck_mask"][:, -1].to(pred.device).float()
    return (((pred - gt) ** 2).sum(-1).sqrt() * mask).sum(-1) / \
        mask.sum(-1).clamp(min=1)                          # [B]


def flat_grad(module, params, loss):
    """d loss / d params as one flat vector. Leaves .grad clean."""
    for p in module.parameters():
        p.grad = None
    loss.backward()
    v = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None
                    else torch.zeros(p.numel(), device=p.device))
                   for p in params])
    for p in module.parameters():
        p.grad = None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--heldout", type=int, default=24)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--snapshots", type=int, default=13)
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
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
    m = RobotNavTrainer.from_checkpoint(args.ckpt, "torch", cfg)
    m.train(); m.float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev); attach_stub(m)
    print(f"  device: {dev}", flush=True)

    ds = build_dataset(cfg["train_dataset"], cfg, m.model)
    it = iter(ds)

    def take(n):
        got = []
        while len(got) < n:
            s = next(it)
            if s.get("sample_type") == "traj":
                got.append(s)
        return got

    held = [ds.collater([s]) for s in take(args.heldout)]
    lepig = getattr(m, "lepig", None)
    params = [p for p in selected_params(m.model, getattr(lepig, "plan", "a1"))
              if p.requires_grad]
    print(f"  selected params: {len(params)} tensors, "
          f"{sum(p.numel() for p in params)/1e6:.1f}M", flush=True)

    # posterior tensors must live on the same device as the Jacobians
    if lepig is not None and getattr(lepig, "posterior", None) is not None:
        for name in dir(lepig.posterior):
            t = getattr(lepig.posterior, name, None)
            if torch.is_tensor(t):
                setattr(lepig.posterior, name, t.to(dev))

    if lepig is not None and lepig.enabled:
        prime = take(max(8, args.snapshots))
        w0 = copy.deepcopy(m.state_dict())
        opt0 = torch.optim.AdamW([p for p in m.parameters()
                                  if p.requires_grad], lr=args.lr)
        for step in range(args.snapshots):
            lepig.on_step(step, params); lepig.warm(step, 10)
            b = ds.collater([prime[step % len(prime)]])
            opt0.zero_grad(); per_sample_losses(m, b).mean().backward(); opt0.step()
        m.load_state_dict(w0); del w0, opt0
        if lepig.should_refresh(0):
            lepig.refresh()
        print(f"  priming {args.anchors} anchors...", flush=True)
        for s in take(args.anchors):
            G = jacobian_rows(m, ds.collater([s]), params, lepig.subspace,
                              lepig.rank)
            if G is not None:
                lepig.add_calibration(G[0]); lepig.add_anchor(G[0])
        print(f"  posterior: rank={lepig.subspace.effective_rank} "
              f"anchors={len(lepig.anchors)} ready={lepig.ready}", flush=True)

    SCHEMES = ["uniform", "pig", "loss_w", "grad_w", "anti_pig", "random_w"]
    cos = {s: [] for s in SCHEMES}

    for trial in range(args.trials):
        # held-out gradient: the direction that actually reduces unseen loss
        g_ho = torch.zeros(sum(p.numel() for p in params), device=dev)
        for hb in held:
            g_ho += flat_grad(m, params, per_sample_losses(m, hb).mean())
        g_ho /= len(held)
        g_ho_n = g_ho / g_ho.norm().clamp(min=1e-12)

        samples = take(args.batch)
        batch = ds.collater(samples)
        B = len(samples)

        # per-sample scores and per-sample gradients (one backward each)
        pig_s, loss_s, grad_s, gsamp = [], [], [], []
        for s in samples:
            b1 = ds.collater([s])
            l = per_sample_losses(m, b1).mean()
            gi = flat_grad(m, params, l)
            gsamp.append(gi)
            loss_s.append(float(l.detach()))
            grad_s.append(float(gi.norm()))
            v = 0.0
            if lepig is not None and lepig.ready:
                G = jacobian_rows(m, b1, params, lepig.subspace, lepig.rank)
                if G is not None:
                    try:
                        v = float(lepig.posterior.pig(G[0], lepig.anchors))
                    except Exception:
                        v = 0.0
            pig_s.append(v)
        Gs = torch.stack(gsamp)                                   # [B, P]

        def W(raw):
            w = robust_weight(torch.tensor(raw, dtype=torch.float32))
            return (w / w.mean()).to(dev)
        w_pig = W(pig_s)
        wmap = {
            "uniform":  torch.ones(B, device=dev),
            "pig":      w_pig,
            "loss_w":   W(loss_s),
            "grad_w":   W(grad_s),
            "anti_pig": W([-x for x in pig_s]),
            "random_w": w_pig[torch.randperm(B, device=dev)],
        }
        line = []
        for name in SCHEMES:
            gw = (wmap[name].unsqueeze(1) * Gs).mean(0)           # weighted grad
            c = float(torch.dot(gw / gw.norm().clamp(min=1e-12), -(-g_ho_n)))
            # cos(g_w, -(-g_ho)) == cos(g_w, g_ho): both are DESCENT directions
            # in the same convention, so higher = better alignment.
            cos[name].append(c)
            line.append(f"{name}={c:+.4f}")
        if trial % 5 == 0 or trial == args.trials - 1:
            print(f"  trial {trial}: " + "  ".join(line), flush=True)
        del Gs, gsamp

    print("\n  GRADIENT ALIGNMENT cos(g_w, g_heldout)  -- higher is better")
    print("  %-10s %9s %9s %8s %10s" % ("scheme", "mean", "std", "SE", "vs uniform"))
    print("  " + "-" * 52)
    u = np.array(cos["uniform"])
    for s in sorted(SCHEMES, key=lambda s: -float(np.mean(cos[s]))):
        v = np.array(cos[s])
        d = v - u                       # PAIRED: same batch, same held-out grad
        se = d.std() / max(len(d) ** 0.5, 1)
        print("  %-10s %9.4f %9.4f %8.4f %+10.4f%s" % (
            s, v.mean(), v.std(), v.std() / max(len(v) ** 0.5, 1), d.mean(),
            "" if s == "uniform" else "  (%.1f SE)" % (abs(d.mean()) / max(se, 1e-9))))
    def pair(a, b):
        d = np.array(cos[a]) - np.array(cos[b])
        se = d.std() / max(len(d) ** 0.5, 1)
        return d.mean(), abs(d.mean()) / max(se, 1e-9)
    for a, b in (("pig", "uniform"), ("pig", "anti_pig"),
                 ("pig", "random_w"), ("pig", "loss_w")):
        dm, sg = pair(a, b)
        print(f"  {a} - {b:<9}: {dm:+.4f}  ({sg:.1f} SE)  "
              f"{'significant' if sg > 2 else 'not significant'}")


if __name__ == "__main__":
    main()
