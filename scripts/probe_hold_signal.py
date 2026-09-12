"""Measure the stop signal a terminal-hold checkpoint actually emits.

The controller's stop test needs a threshold on "has the plan stopped
changing". Guessing that threshold is how you get a decoder that works on
clean plans and fails on real output. This measures it instead: run the model
over training samples whose terminal_mask marks them as holds, and over
samples that are still moving, and report the slot-to-slot delta distribution
for each.

If the two are separable, the gap between them IS the threshold, and the
report says so. If they overlap, the model has not learned a readable stop
signal and no controller change can fix that -- which is equally worth
knowing before spending GPU-days on evals.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=400)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    from train.robotnav_trainer import RobotNavTrainer
    from train.experiment_utils import prepare_experiment
    from data.build_dataset import build_dataset

    cfg, *_ = prepare_experiment(cfg)
    module = RobotNavTrainer.from_checkpoint(args.ckpt, "torch", cfg)
    module.eval()
    ds = build_dataset(cfg["train_dataset"], cfg, module.model)

    scale = None
    hold_d, move_d = [], []
    it = iter(ds)
    n = 0
    with torch.no_grad():
        while n < args.samples:
            s = next(it)
            if s.get("sample_type") != "traj":
                continue
            b = ds.collater([s])
            m = b["chunck_mask"][0, -1].numpy()        # [K] 1 = real waypoint
            pred = module._predict_waypoints(b)[0].float().cpu().numpy()  # [K,3]
            if scale is None:
                scale = b["wp_scale"][0].numpy()
            pred = pred * scale                         # -> metres / radians
            d = np.diff(pred, axis=0)
            dxy = np.linalg.norm(d[:, :2], axis=1)
            dyaw = np.abs(d[:, 2])
            # a slot is a "hold" slot when the label masks it off (padded repeat)
            for k in range(len(dxy)):
                tgt = hold_d if m[k + 1] == 0 else move_d
                tgt.append((dxy[k], dyaw[k]))
            n += 1

    hold = np.array(hold_d) if hold_d else np.zeros((0, 2))
    move = np.array(move_d) if move_d else np.zeros((0, 2))
    print(f"\n  samples={n}  hold-slots={len(hold)}  moving-slots={len(move)}")
    if not len(hold) or not len(move):
        print("  cannot separate: one class is empty"); return

    def qs(a, ps=(50, 90, 95, 99)):
        return "  ".join(f"p{p}={np.percentile(a, p):.4f}" for p in ps)

    print("\n  PREDICTED slot-to-slot deltas (metres / radians)")
    print(f"    hold  dxy : {qs(hold[:, 0])}   max={hold[:, 0].max():.4f}")
    print(f"    move  dxy : {qs(move[:, 0])}   min={move[:, 0].min():.4f}")
    print(f"    hold  dyaw: {qs(hold[:, 1])}   max={hold[:, 1].max():.4f}")
    print(f"    move  dyaw: {qs(move[:, 1])}   min={move[:, 1].min():.4f}")

    # threshold that best separates the two, by simple sweep on dxy
    cands = np.linspace(0.01, 0.30, 60)
    best, bt = -1, None
    for t in cands:
        tp = (hold[:, 0] < t).mean()          # holds correctly called static
        fp = (move[:, 0] < t).mean()          # moving wrongly called static
        j = tp - fp                            # Youden's J
        if j > best:
            best, bt = j, (t, tp, fp)
    t, tp, fp = bt
    print(f"\n  best dxy threshold = {t:.3f} m"
          f"   (holds caught {tp*100:.1f}%, moving misread {fp*100:.1f}%)")
    print(f"  separation quality J = {best:.3f}"
          f"  -> {'USABLE' if best > 0.7 else 'WEAK - the model has not learned a crisp hold'}")


if __name__ == "__main__":
    main()
