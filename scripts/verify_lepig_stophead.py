"""Confirm a LEPIG config trains the stop head, LoRA, and world branch together."""
import json
import sys
import types

import torch

from data.build_dataset import build_dataset
from models.lepig.lora import lora_parameters
from train.experiment_utils import prepare_experiment
from train.robotnav_trainer import RobotNavTrainer


def grad_norm(params):
    g = [p.grad for p in params if p.grad is not None]
    return sum(float(x.norm()) ** 2 for x in g) ** 0.5 if g else 0.0


def main():
    plan = sys.argv[1] if len(sys.argv) > 1 else "b"
    path = f"configs/mn256x16-lfm2vl_3b-smolvla-navreason-holds-lepig{plan}.json"
    cfg = json.load(open(path))
    cfg["train_dataset"]["mixture_mode"] = "batch"
    cfg["train_dataset"]["mixture_trajectory"] = 1.0
    cfg["batch_size"] = 2
    cfg, *_ = prepare_experiment(cfg)

    m = RobotNavTrainer(cfg)
    m.train()
    m.float()
    m._trainer = types.SimpleNamespace(
        world_size=1, global_rank=0, local_rank=0, num_devices=1,
        global_step=0, current_epoch=0, max_steps=10,
        estimated_stepping_batches=10, barebones=False, loggers=[],
        log_dir=None, state=None, sanity_checking=False)
    m.log = lambda *a, **k: None
    m.log_dict = lambda *a, **k: None

    ds = build_dataset(cfg["train_dataset"], cfg, m.model)
    it = iter(ds)
    buf = []
    while len(buf) < 2:
        x = next(it)
        if x.get("sample_type") == "traj":
            buf.append(x)
    batch = ds.collater(buf)
    if isinstance(batch.get("traj"), dict):
        batch = batch["traj"]

    for p in m.parameters():
        p.grad = None
    res = m.training_step(batch, 0)
    loss = res["loss"] if isinstance(res, dict) else res
    loss.backward()

    head = m.model.act_head
    stop = [p for n, p in head.named_parameters() if "stop_head" in n]
    sg = grad_norm(stop)
    lg = grad_norm(lora_parameters(m.model))
    print(f"  plan {plan}")
    print(f"    stop_head grad : {sg:.4e}  -> "
          f"{'TRAINING' if sg > 0 else 'DEAD'}")
    print(f"    stop detached  : {getattr(head, 'stop_detach', None)}")
    print(f"    lora grad      : {lg:.4e}  -> "
          f"{'TRAINING' if lg > 0 else 'DEAD'}")
    if getattr(m, "world_branch", None) is not None:
        wg = grad_norm(list(m.world_branch.parameters()))
        print(f"    world grad     : {wg:.4e}  -> "
              f"{'TRAINING' if wg > 0 else 'DEAD'}")
    print(f"    routes_fm_grad : {m.lepig.routes_backbone_fm_grad}")
    print(f"    loss           : {float(loss.detach()):.4f}")


if __name__ == "__main__":
    main()
