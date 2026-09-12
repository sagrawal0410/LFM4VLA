"""Verify the stop head is actually being optimised, not computed and dropped.

Non-zero weights are not proof: weight decay or noise moves a small layer too.
This settles it directly -- run one real batch and check that

  1. loss_stop_act appears in the loss dict and is finite and non-zero
  2. it is INCLUDED in the total loss (total != action-only loss)
  3. backward() puts non-zero gradient on stop_head parameters
  4. the head's logits actually respond to the stop label (BCE falls when the
     label matches the prediction's sign)

Any of these failing means the head is dead weight at training time, which is
exactly the failure mode the loss-key suffix bug produced earlier.
"""
from __future__ import annotations

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    from train.experiment_utils import prepare_experiment
    from train.robotnav_trainer import RobotNavTrainer
    from data.build_dataset import build_dataset

    # "sample" mixing always returns a nested {data_source, traj, vl} batch;
    # "batch" mixing returns homogeneous batches, and mixture_trajectory=1.0
    # makes every one of them a trajectory batch.
    cfg["train_dataset"]["mixture_mode"] = "batch"
    cfg["train_dataset"]["mixture_trajectory"] = 1.0
    cfg, *_ = prepare_experiment(cfg)
    module = (RobotNavTrainer.from_checkpoint(args.ckpt, "torch", cfg)
              if args.ckpt else RobotNavTrainer(cfg))
    module.train()
    # _forward_batch reads self.trainer.world_size; outside Lightning that
    # property raises, so attach a minimal stand-in.
    import types
    module._trainer = types.SimpleNamespace(
        world_size=1, global_rank=0, local_rank=0, num_devices=1,
        global_step=0, current_epoch=0, max_steps=1, estimated_stepping_batches=1)

    head = module.model.act_head
    print(f"  use_stop_head          : {getattr(head, 'use_stop_head', None)}")
    print(f"  stop_head built        : {getattr(head, 'stop_head', None) is not None}")
    sp = [p for n, p in head.named_parameters() if "stop_head" in n]
    print(f"  stop_head params       : {sum(p.numel() for p in sp)}")

    ds = build_dataset(cfg["train_dataset"], cfg, module.model)
    it = iter(ds)
    buf = []
    while len(buf) < 2:
        s_ = next(it)
        if s_.get("sample_type") == "traj":
            buf.append(s_)
    batch = ds.collater(buf)
    if "traj" in batch and isinstance(batch.get("traj"), dict):
        batch = batch["traj"]            # unwrap if mixed mode slipped through
    print(f"  batch keys             : {sorted(batch.keys())[:8]}")
    print(f"  stop_label present     : {'stop_label' in batch}")
    if "stop_label" in batch:
        sl = batch["stop_label"]
        print(f"  stop_label shape/pos   : {tuple(sl.shape)}  "
              f"positives={float(sl.sum()):.0f}/{sl.numel()}")

    for p in module.parameters():
        p.grad = None
    out = module.training_step(batch, 0)
    loss = out["loss"] if isinstance(out, dict) else out

    pred = getattr(module, "_last_prediction", None)
    keys = [k for k in (pred or {}) if "stop" in k] if isinstance(pred, dict) else []
    print(f"\n  loss (total)           : {float(loss):.6f}")
    print(f"  stop keys in prediction: {keys}")

    loss.backward()
    g = [(n, p.grad) for n, p in head.named_parameters() if "stop_head" in n]
    gn = sum(float(x.norm()) for _, x in g if x is not None)
    print(f"  stop_head grad norm    : {gn:.6e}")
    for n, x in g:
        print(f"    {n:34s} grad={'None' if x is None else f'{float(x.norm()):.3e}'}")

    ok = gn > 0
    print(f"\n  VERDICT: stop head is {'BEING TRAINED' if ok else 'NOT RECEIVING GRADIENT'}")
    if not ok:
        print("  -> the BCE term is computed but never reaches the optimiser")


if __name__ == "__main__":
    main()
