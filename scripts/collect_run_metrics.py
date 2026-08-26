import glob
import json
import os

tasks = {}
for p in glob.glob("/home/teams/research/robotics/logs/*/lfm450m_robotnav_*"
                   "/wandb/run-*/files/wandb-summary.json"):
    exp = p.split("/logs/")[1].split("/")[1]
    task = exp.split("-lfm450m-")[0]
    if "smoke" in task or "wpm" in task or "mn256" in task:
        continue
    m = os.path.getmtime(p)
    if task not in tasks or m > tasks[task][0]:
        tasks[task] = (m, p)

COLS = ["trainer/global_step", "train_loss_arm_act_step", "val_loss_arm_act",
        "val_loss_vl_cotrain", "train_loss_vl_cotrain_step"]
for task in sorted(tasks):
    try:
        d = json.load(open(tasks[task][1]))
    except Exception:
        continue
    vals = []
    for k in COLS:
        v = d.get(k)
        vals.append(str(round(v, 4)) if isinstance(v, float) else str(v if v is not None else "-"))
    print(task + "|" + "|".join(vals))
