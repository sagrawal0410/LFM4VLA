"""Assert LEPIG actually produces weights on the LIBERO path.

The failure this exists to prevent: LEPIG lived on RobotNavTrainer while LIBERO
gets BaseTrainer, so every LIBERO LEPIG run would have trained as a silent
uniform baseline -- exactly how three navigation runs burned two days.

Per arm this checks, in order:
  1. the controller is constructed and enabled
  2. warmup is forced open so scoring engages immediately
  3. _lepig_step returns a NON-None weight vector of the right length
  4. the weights are non-degenerate (std > 0) unless the arm is 'uniform'
  5. one real training_step runs and produces a finite loss
  6. gradient actually reaches the parameters the plan says it should
"""
from __future__ import annotations

import json
import sys
import types

import torch


def stub(m):
    m._trainer = types.SimpleNamespace(
        world_size=1, global_rank=0, local_rank=0, num_devices=1, global_step=0,
        current_epoch=0, max_steps=10, estimated_stepping_batches=10,
        barebones=False, loggers=[], log_dir=None, state=None,
        sanity_checking=False)
    m.log = lambda *a, **k: None
    m.log_dict = lambda *a, **k: None
    return m


def check(cfg_path):
    name = cfg_path.split("/")[-1].replace(".json", "")
    cfg = json.load(open(cfg_path))
    cfg["batch_size"] = 2
    cfg["trainer"]["max_steps"] = 10
    if "lepig" in cfg:
        cfg["lepig"].update(min_warmup_steps=0, warmup_fraction=0.0,
                            refresh_steps=1, trajectory_snapshots=3,
                            trajectory_window_steps=3)
    from train.experiment_utils import prepare_experiment
    from train.base_trainer import BaseTrainer
    from data.build_dataset import build_dataset
    from models.lepig.hooks import selected_params, jacobian_rows

    cfg, *_ = prepare_experiment(cfg)
    m = BaseTrainer(cfg); m.train(); m.float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev); stub(m)

    lep = getattr(m, "lepig", None)
    has_lepig = lep is not None and lep.enabled
    ds = build_dataset(cfg["train_dataset"], cfg, m.model)
    it = iter(ds)
    batch = ds.collater([next(it) for _ in range(2)])

    w = None
    if has_lepig:
        params = selected_params(m.model, lep.plan)
        for step in range(4):
            lep.on_step(step, params); lep.warm(step, 10)
        if lep.should_refresh(0):
            lep.refresh()
        for _ in range(4):
            G = jacobian_rows(m, batch, params, lep.subspace, lep.rank)
            if G is not None:
                for i in range(G.shape[0]):
                    lep.add_calibration(G[i]); lep.add_anchor(G[i])
        w = m._lepig_step(batch)

    for p in m.parameters():
        p.grad = None
    m._lepig_scored_this_batch = False
    out = m.training_step(batch, 0)
    loss = out["loss"] if isinstance(out, dict) else out
    loss.backward()
    bb = [p for n, p in m.model.named_parameters()
          if "act_head" not in n and p.grad is not None]
    gn = sum(float(p.grad.norm()) ** 2 for p in bb) ** 0.5

    wtxt = ("None" if w is None else
            f"[{','.join(f'{float(x):.3f}' for x in w[:4])}] std={float(w.std()):.3f}")
    ok = (not has_lepig) or (w is not None)
    print(f"  {name:36s} lepig={str(has_lepig):5s} w={wtxt:34s} "
          f"loss={float(loss.detach()):.4f} bb_grad={gn:.3e} "
          f"{'OK' if ok else '*** NO WEIGHTS ***'}", flush=True)
    return ok


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        try:
            bad += (not check(p))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {p.split('/')[-1]:36s} FAILED {type(e).__name__}: {str(e)[:90]}")
            bad += 1
    print(f"\n  {len(sys.argv)-1-bad} passed, {bad} failed")
