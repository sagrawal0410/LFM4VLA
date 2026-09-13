"""Verify LoRA is injected, trainable, identity-at-init, and receives gradient."""
import json, sys, types
import torch
from train.experiment_utils import prepare_experiment
from train.robotnav_trainer import RobotNavTrainer
from data.build_dataset import build_dataset
from models.lepig.lora import LoRALinear, lora_parameters

plan = sys.argv[1] if len(sys.argv) > 1 else "a2"
cfg = json.load(open(f"configs/mn256x16-lfm2vl_3b-smolvla-navreason-holds-lepig{plan}.json"))
cfg["train_dataset"]["mixture_mode"] = "batch"
cfg["train_dataset"]["mixture_trajectory"] = 1.0
cfg["batch_size"] = 2
cfg, *_ = prepare_experiment(cfg)
m = RobotNavTrainer(cfg); m.train(); m.float()
m._trainer = types.SimpleNamespace(
    world_size=1, global_rank=0, local_rank=0, num_devices=1, global_step=0,
    current_epoch=0, max_steps=10, estimated_stepping_batches=10,
    barebones=False, loggers=[], log_dir=None, state=None, sanity_checking=False)
m.log = lambda *a, **k: None; m.log_dict = lambda *a, **k: None

sites = [n for n, mod in m.model.named_modules() if isinstance(mod, LoRALinear)]
lp = lora_parameters(m.model)
print(f"  plan                 : {plan}")
print(f"  LoRA sites           : {len(sites)}  (spec: 7 per block x 4 blocks = 28)")
print(f"  LoRA params          : {sum(p.numel() for p in lp)/1e6:.2f}M trainable")
if sites:
    print(f"  example site         : {sites[0]}")
    ex = dict(m.model.named_modules())[sites[0]]
    print(f"  B init is zero       : {bool((ex.lora_B.weight == 0).all())}  (identity at init)")
    print(f"  scaling alpha/r      : {ex.scaling}")
    base_tr = ex.base.weight.requires_grad
    print(f"  base VLM trainable   : {base_tr}  (spec: trainable)")

ds = build_dataset(cfg["train_dataset"], cfg, m.model)
it = iter(ds); buf = []
while len(buf) < 2:
    x = next(it)
    if x.get("sample_type") == "traj": buf.append(x)
b = ds.collater(buf)
if isinstance(b.get("traj"), dict): b = b["traj"]
for p in m.parameters(): p.grad = None
res = m.training_step(b, 0)
loss = res["loss"] if isinstance(res, dict) else res
loss.backward()
g = [p.grad for p in lp if p.grad is not None]
gn = sum(float(x.norm()) ** 2 for x in g) ** 0.5 if g else 0.0
print(f"  LoRA grad norm       : {gn:.4e} -> {'TRAINING' if gn > 0 else 'DEAD'}")
sel = None
try:
    from models.lepig.hooks import selected_params
    sel = selected_params(m.model, plan)
    nlora = sum(1 for p in sel for q in lp if p is q)
    print(f"  posterior covers     : {len(sel)} tensors, {nlora} of them LoRA")
except Exception as e:
    print("  posterior check failed:", e)
if getattr(m, "world_whiten", None) is not None:
    print(f"  world whitening      : built, fitted={m.world_whiten.fitted}")
