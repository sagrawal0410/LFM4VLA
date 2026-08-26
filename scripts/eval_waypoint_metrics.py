"""Offline task-space evaluation: score a checkpoint on the SHARED val set.

Loads a run's checkpoint into the real trainer module, feeds an identical
(seed-777, salt-free, trajectory-only) 256-sample val draw through the head's
true inference path, and prints ADE / FDE / yaw / success@25cm (+ minADE-K).

Usage: python eval_waypoint_metrics.py --config <json> --ckpt <last.ckpt>
           --task <name> [--samples 256] [--minade-k 1] [--out results.jsonl]
"""
import argparse
import json
import math
import sys

sys.path.insert(0, "/home/shaurya.agrawal@liquid.ai/LFM4VLA")

import torch
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--minade-k", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    variant = json.load(open(args.config))
    variant["val_waypoint_metrics"] = True
    variant["val_minade_k"] = args.minade_k
    # keep dataset/init light: no wandb, no resume side-effects
    variant["trainer"]["logger"] = False
    variant.pop("resume", None)

    from train.robotnav_trainer import RobotNavTrainer
    module = RobotNavTrainer(variant)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(ck["state_dict"], strict=False)
    step = ck.get("global_step", -1)
    print(f"[eval] {args.task}: loaded step={step} "
          f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    module = module.to("cuda").eval()

    # Shared benchmark set: identical for every model.
    from data.robotnav_dataset import RobotNavMixtureDataset
    _proc = module.model.image_processor          # per-image PIL -> CHW
    def _image_fn(pils):                          # dataset feeds a LIST
        return [_proc(p) for p in pils]
    ds = RobotNavMixtureDataset(
        release_dir=variant["train_dataset"]["release_dir"],
        generated_root=variant["train_dataset"]["generated_root"],
        image_fn=_image_fn,
        window_size=variant["window_size"],
        fwd_pred_next_n=variant["fwd_pred_next_n"],
        mixture_trajectory=1.0,
        mixture_mode="batch",
        batch_size=8,
        max_samples=args.samples,
        seed=777,
        data_seed_salt="none",
    )
    loader = DataLoader(ds, batch_size=8, collate_fn=ds.collater, num_workers=0)

    sums = {}
    counts = 0
    collected = {}
    module.log = lambda name, value, **kw: collected.__setitem__(
        name, float(value))
    with torch.no_grad():
        for batch in loader:
            collected.clear()
            module._log_waypoint_metrics(batch)
            n = len(batch["family"])
            for k, v in collected.items():
                sums[k] = sums.get(k, 0.0) + v * n
            counts += n
    result = {"task": args.task, "step": step, "n": counts}
    result.update({k: round(v / counts, 4) for k, v in sums.items()})
    print(json.dumps(result), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
