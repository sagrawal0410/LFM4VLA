"""Prove the LEPIG-A and stop-head wiring is live, not merely present.

Built-but-never-called has cost this project days twice. Each check here is a
measurement with an unambiguous failure value, not an inspection.
"""
import glob, json, os, subprocess, sys, types

import torch

CK = "/home/teams/research/robotics/checkpoints"


def code_freshness(jobid, files):
    """A running process uses the code it imported at start. Anything edited
    after that start time is NOT in the running job."""
    start = subprocess.run(["sacct", "-j", str(jobid), "-X", "-n", "--format=Start"],
                           capture_output=True, text=True).stdout.strip()
    if not start or start == "Unknown":
        return None, []
    import datetime
    st = datetime.datetime.fromisoformat(start).timestamp()
    stale = [f for f in files if os.path.getmtime(f) > st]
    return start, stale


def stop_head_delta():
    d = glob.glob(CK + "/*/lfm2vl_3b_mlp_*STOPHEAD*")
    if not d:
        return "no stophead dir"
    cks = sorted(glob.glob(d[0] + "/step-*.ckpt"), key=os.path.getmtime)
    if len(cks) < 2:
        return "only %d checkpoint(s)" % len(cks)
    def head(c):
        sd = torch.load(c, map_location="cpu", weights_only=False)["state_dict"]
        return {k: v.float() for k, v in sd.items() if "stop_head" in k}
    a, b = head(cks[-2]), head(cks[-1])
    ta = os.path.basename(cks[-2]).split("step=")[-1].replace(".ckpt", "")
    tb = os.path.basename(cks[-1]).split("step=")[-1].replace(".ckpt", "")
    tot = sum(float((b[k] - a[k]).abs().sum()) for k in a)
    return "%s->%s  |delta|=%.4e  %s" % (
        ta, tb, tot, "LEARNING" if tot > 1e-6 else "FROZEN/DEAD")


def routing_efficacy():
    """The check never run: do non-uniform weights actually change the backbone
    gradient? Uniform w=1 must give a different backbone gradient than w=[1.5,0.5]
    -- if they match, grad_scale_identity is a no-op in the live path."""
    from models.lepig.routing import grad_scale_identity
    torch.manual_seed(0)
    res = {}
    for name, w in (("uniform", torch.ones(2)), ("weighted", torch.tensor([1.5, 0.5]))):
        torch.manual_seed(0)
        backbone = torch.nn.Linear(8, 8)
        h = backbone(torch.randn(2, 4, 8))
        routed = grad_scale_identity(h, w)
        routed.pow(2).sum().backward()
        res[name] = float(backbone.weight.grad.norm())
    same = abs(res["uniform"] - res["weighted"]) < 1e-9
    return "uniform=%.6f weighted=%.6f -> %s" % (
        res["uniform"], res["weighted"],
        "NO-OP (routing dead)" if same else "ROUTING ACTIVE")


def dataloader_future_frames():
    """Plans B/C need a frame at t+H. Confirm the loader emits one."""
    from train.experiment_utils import prepare_experiment
    from train.robotnav_trainer import RobotNavTrainer
    from data.build_dataset import build_dataset
    cfg = json.load(open("configs/mn256x16-lfm2vl_3b-smolvla-navreason-holds-lepigb.json"))
    cfg["train_dataset"]["mixture_mode"] = "batch"
    cfg["train_dataset"]["mixture_trajectory"] = 1.0
    cfg["train_dataset"]["world_horizons"] = [2, 4, 8]
    cfg["batch_size"] = 2
    cfg, *_ = prepare_experiment(cfg)
    # build_dataset needs the model for its image processor
    module = RobotNavTrainer(cfg)
    ds = build_dataset(cfg["train_dataset"], cfg, module.model)
    it = iter(ds); buf = []
    while len(buf) < 2:
        s = next(it)
        if s.get("sample_type") == "traj":
            buf.append(s)
    b = ds.collater(buf)
    fut = b.get("future_rgb")
    if fut is None:
        return "future_rgb ABSENT -- B/C cannot train"
    return "future_rgb %s  dtype=%s  OK" % (tuple(fut.shape), fut.dtype)


def world_branch_gradient():
    """Plans B/C: does world_branch actually receive gradient?

    Zero here means the branch is decorative -- the exact failure that hid in
    the stop head for 25 GPU-hours and in LEPIG for three runs.
    """
    import types
    from train.experiment_utils import prepare_experiment
    from train.robotnav_trainer import RobotNavTrainer
    from data.build_dataset import build_dataset
    cfg = json.load(open("configs/mn256x16-lfm2vl_3b-smolvla-navreason-holds-lepigb.json"))
    cfg["train_dataset"]["mixture_mode"] = "batch"
    cfg["train_dataset"]["mixture_trajectory"] = 1.0
    cfg["batch_size"] = 2
    cfg, *_ = prepare_experiment(cfg)
    m = RobotNavTrainer(cfg); m.train(); m.float()
    m._trainer = types.SimpleNamespace(
        world_size=1, global_rank=0, local_rank=0, num_devices=1, global_step=0,
        current_epoch=0, max_steps=10, estimated_stepping_batches=10,
        barebones=False, loggers=[], log_dir=None, state=None,
        sanity_checking=False)
    m.log = lambda *a, **k: None; m.log_dict = lambda *a, **k: None
    if m.world_branch is None:
        return "world_branch NOT BUILT"
    if getattr(m, "vjepa", None) is None:
        return "vjepa NOT BUILT -- no targets"
    ds = build_dataset(cfg["train_dataset"], cfg, m.model)
    it = iter(ds); buf = []
    while len(buf) < 2:
        x = next(it)
        if x.get("sample_type") == "traj":
            buf.append(x)
    b = ds.collater(buf)
    if isinstance(b.get("traj"), dict):
        b = b["traj"]
    for prm in m.parameters():
        prm.grad = None
    res = m.training_step(b, 0)
    loss = res["loss"] if isinstance(res, dict) else res
    loss.backward()
    g = [prm.grad for prm in m.world_branch.parameters() if prm.grad is not None]
    gn = sum(float(x.norm()) ** 2 for x in g) ** 0.5 if g else 0.0
    npar = sum(prm.numel() for prm in m.world_branch.parameters())
    return "params=%.1fM grad_norm=%.4e loss=%.4f -> %s" % (
        npar / 1e6, gn, float(loss.detach()),
        "TRAINING" if gn > 0 else "DEAD (no gradient)")


print("=" * 68)
print("1. STOP HEAD")
print("   weight delta :", stop_head_delta())
st, stale = code_freshness(2336949, ["models/robo_lfm.py", "models/base_policy.py"])
print("   job start    :", st)
print("   stale files  :", stale if stale else "none (running job has the fix)")

print("\n2. LEPIG ROUTING")
print("   efficacy     :", routing_efficacy())

print("\n3. LEPIG-A RUNNING JOBS")
wired = ["train/robotnav_trainer.py", "train/base_trainer.py",
         "models/robo_lfm.py", "models/lepig/hooks.py"]
for jid, nm in ((2337586, "lp-a1-smolvla"), (2337587, "lp-a2-smolvla"),
                (2337588, "lp-a2-groot")):
    st, stale = code_freshness(jid, wired)
    print("   %-15s start=%s stale=%s" % (nm, st, stale if stale else "none"))

print("\n4. DATALOADER FUTURE FRAMES (plans B/C)")
try:
    print("   ", dataloader_future_frames())
except Exception as e:
    print("    FAILED:", type(e).__name__, str(e)[:150])

print("\n5. WORLD BRANCH GRADIENT (plans B/C)")
try:
    print("   ", world_branch_gradient())
except Exception as e:
    print("    FAILED:", type(e).__name__, str(e)[:200])
print("=" * 68)
