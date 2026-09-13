"""Show which transformer stacks exist and which one inject_lora selects."""
import json
import torch.nn as nn
from train.experiment_utils import prepare_experiment
from train.robotnav_trainer import RobotNavTrainer
from models.lepig.lora import _find_blocks

cfg = json.load(open("configs/mn256x16-lfm2vl_3b-smolvla-navreason-holds-lepiga2.json"))
cfg.pop("lepig", None)                 # build WITHOUT injection so we can inspect
cfg, *_ = prepare_experiment(cfg)
m = RobotNavTrainer(cfg)

WANT = {"q_proj", "k_proj", "v_proj", "out_proj", "o_proj",
        "w1", "w2", "w3", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"}
for name, mod in m.model.named_modules():
    if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) >= 4:
        leaves = {n.split(".")[-1] for n, sub in mod[0].named_modules()
                  if isinstance(sub, nn.Linear)}
        hit = leaves & WANT
        if hit:
            print("  %-54s len=%-3d linears=%s" % (name[:54], len(mod), sorted(hit)))

chosen = _find_blocks(m.model)
if chosen:
    blk = chosen[-1]
    leaves = sorted({n.split(".")[-1] for n, sub in blk.named_modules()
                     if isinstance(sub, nn.Linear)})
    print("\n  CHOSEN len=%d  last-block linears=%s" % (len(chosen), leaves))
else:
    print("\n  CHOSEN: none")
