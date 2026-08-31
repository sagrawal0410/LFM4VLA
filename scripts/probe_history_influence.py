"""Does the history window influence predictions at all?

Runs the shared waypoint benchmark three ways on one checkpoint:
  normal   — frames as the dataset produced them
  zerohist — the 8 history frames replaced by BLACK images (current kept real)
  duphist  — the 8 history frames replaced by copies of the current frame
Identical ADE across variants == history carries zero information.

  python scripts/probe_history_influence.py --config <cfg> --ckpt <ckpt>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args()

    variant = json.load(open(args.config))
    variant["trainer"]["logger"] = False
    variant.pop("resume", None)
    # deterministic shared benchmark, no rank/job salt
    for k in ("train_dataset", "val_dataset"):
        variant[k]["data_seed_salt"] = "none"
        variant[k].pop("obs_randomization", None)
    from train.robotnav_trainer import RobotNavTrainer
    from data.robotnav_dataset import RobotNavMixtureDataset

    module = RobotNavTrainer(variant)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module.load_state_dict(ck["state_dict"], strict=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(dev).eval()
    if dev == "cpu":
        module = module.float()

    _proc = module.model.image_processor          # per-image PIL -> CHW
    dcfg = dict(variant["val_dataset"])
    dcfg.pop("type", None)
    dcfg["image_fn"] = lambda pils: [_proc(p) for p in pils]
    dcfg["window_size"] = variant["window_size"]
    dcfg["fwd_pred_next_n"] = variant["fwd_pred_next_n"]
    dcfg["batch_size"] = 8
    dcfg["mixture_trajectory"] = 1.0
    dcfg["mixture_mode"] = "batch"
    dcfg["seed"] = 777
    ds = RobotNavMixtureDataset(**dcfg)

    batches, buf = [], []
    for s in ds:
        buf.append(s)
        if len(buf) == 8:
            batches.append(ds.collater(buf)); buf = []
        if len(batches) * 8 >= args.samples:
            break

    def ade_for(mutate) -> float:
        errs = []
        # Same noise draws for every variant, so any delta is information
        # flow from the history frames rather than FM sampling variance.
        torch.manual_seed(1234)
        for b in batches:
            b = {k: v for k, v in b.items()}
            rgb = [list(frames) for frames in b["rgb"]]
            if mutate is not None:
                rgb = [mutate(f) for f in rgb]
            b["rgb"] = rgb
            gt = b["action_chunck"][:, -1].float()
            mask = b["chunck_mask"][:, -1].bool()
            scale = b["wp_scale"].unsqueeze(1).float()
            with torch.no_grad():
                pred = module._predict_waypoints(b).cpu().float()
            d = torch.linalg.norm(pred[..., :2] * scale[..., :2]
                                  - gt[..., :2] * scale[..., :2], dim=-1)
            n = mask.sum(-1).clamp(min=1)
            errs.append(((d * mask).sum(-1) / n))
        return float(torch.cat(errs).mean())

    def zero_hist(frames):
        out = [torch.zeros_like(f) for f in frames[:-1]]
        return out + [frames[-1]]

    def dup_hist(frames):
        return [frames[-1].clone() for _ in frames[:-1]] + [frames[-1]]

    res = {
        "normal": ade_for(None),
        "zerohist": ade_for(zero_hist),
        "duphist": ade_for(dup_hist),
    }
    print(json.dumps({"ckpt": args.ckpt, "n": len(batches) * 8, **res}, indent=1))
    d0 = abs(res["zerohist"] - res["normal"])
    print(f"\n|zerohist - normal| = {d0:.6f} m  "
          f"({'HISTORY IGNORED' if d0 < 1e-6 else 'history matters'})")


if __name__ == "__main__":
    main()
