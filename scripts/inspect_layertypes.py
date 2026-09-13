"""Which of the last 4 language-tower blocks are attention vs short-conv?

LFM2 is a hybrid stack, so the plan's "7 sites per block" (which assumes a
uniform attention architecture) does not apply uniformly here.
"""
import json
import torch.nn as nn
from train.experiment_utils import prepare_experiment
from train.robotnav_trainer import RobotNavTrainer

cfg = json.load(open("configs/mn256x16-lfm2vl_3b-smolvla-navreason-holds-lepiga1.json"))
cfg.pop("lepig", None)
cfg, *_ = prepare_experiment(cfg)
m = RobotNavTrainer(cfg)
blocks = dict(m.model.named_modules())["backbone.model.language_model.layers"]
print("  language tower: %d blocks, hidden=%s" % (
    len(blocks), getattr(blocks[0], "hidden_size", "?")))
tot = 0
for i, blk in enumerate(list(blocks)[-4:], start=len(blocks) - 4):
    lin = {n: sub for n, sub in blk.named_modules() if isinstance(sub, nn.Linear)}
    kind = "attention" if any("q_proj" in n for n in lin) else "short_conv"
    leaves = sorted({n.split(".")[-1] for n in lin})
    dims = {n.split(".")[-1]: (sub.in_features, sub.out_features)
            for n, sub in lin.items()}
    sites = len(leaves)
    tot += sites
    print("  block %-3d %-11s sites=%d  %s" % (i, kind, sites, leaves))
    for k in sorted(dims):
        print("        %-10s in=%-6d out=%d" % (k, dims[k][0], dims[k][1]))
print("  TOTAL adaptable sites in last 4 blocks: %d" % tot)
