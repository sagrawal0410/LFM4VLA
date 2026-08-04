"""Closed-loop Unitree G1 MuJoCo eval for Humanoid Everyday LFM4VLA checkpoints.

Requires:
  bash scripts/fetch_unitree_g1_mujoco.sh
  bash scripts/install_g1_mujoco_eval.sh   # mujoco, imageio, pyyaml in lfm4vla-he

Example (tok2 / lat16 run)::

  python eval/humanoid/evaluate_g1.py \\
    --ckpt /home/teams/research/robotics/checkpoints/.../he_tok2_lat16-lfm450m-...-895e10 \\
    --config configs/lfm2.5-vl-450m-humanoid-everyday.json \\
    --num_action_tokens 2 --latent 16 \\
    --data_root_dir /home/teams/research/robotics/humanoid_everyday \\
    --suite all --episodes_per_task 5 --save_video
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _configure_gl(requested: str) -> str:
    """Headless MuJoCo GL without requiring robosuite (unlike utils.mujoco_gl)."""
    import subprocess

    from utils.mujoco_gl import MUJOCO_GL_PROFILES, apply_profile, prepend_conda_lib

    probe = r"""
import mujoco
xml = '''<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.1"/></worldbody></mujoco>'''
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, height=64, width=64)
r.update_scene(d)
_ = r.render()
r.close()
print("PROBE_OK")
"""
    profiles = list(MUJOCO_GL_PROFILES)
    if requested not in ("auto", "", None):
        profiles = [p for p in profiles if p["name"] == requested or p["MUJOCO_GL"] == requested]
        if not profiles:
            profiles = list(MUJOCO_GL_PROFILES)

    failures = []
    for profile in profiles:
        print(f"[render] probing {profile['name']} ...", flush=True)
        env = os.environ.copy()
        prepend_conda_lib()
        for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBGL_ALWAYS_SOFTWARE", "EGL_PLATFORM"):
            env.pop(key, None)
        for k, v in profile.items():
            if k != "name":
                env[k] = v
        res = subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True
        )
        if res.returncode == 0 and "PROBE_OK" in (res.stdout or ""):
            apply_profile(profile)
            print(f"[render] using {profile['name']}", flush=True)
            return profile["name"]
        err = (res.stderr or res.stdout or "").strip().splitlines()
        failures.append(f"{profile['name']}: {err[-1] if err else 'fail'}")
        print(f"[render]   failed: {failures[-1]}", flush=True)
    raise RuntimeError("No MuJoCo GL backend.\n" + "\n".join(failures))


def _save_mp4(frames: List[Any], path: Path, fps: int = 20) -> None:
    import imageio.v2 as imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), [np.asarray(f) for f in frames], fps=fps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="Checkpoint file or experiment directory")
    ap.add_argument(
        "--config",
        default="configs/lfm2.5-vl-450m-humanoid-everyday.json",
        help="Base HE config (token/latent overridden below)",
    )
    ap.add_argument("--num_action_tokens", type=int, default=2)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument(
        "--data_root_dir",
        default="/home/teams/research/robotics/humanoid_everyday",
        help="HE data root containing meta/action_stats_g1_*.json",
    )
    ap.add_argument("--suite", choices=("in", "ood", "all"), default="all")
    ap.add_argument("--episodes_per_task", type=int, default=5)
    ap.add_argument("--execute_step", type=int, default=1, help="Open-loop chunk length")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mujoco_gl", default="auto")
    ap.add_argument("--control_hz", type=float, default=20.0)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument(
        "--output_dir",
        default=None,
        help="Metrics + videos (default: runs/g1_eval/<ckpt_name>)",
    )
    ap.add_argument("--tasks_yaml", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _configure_gl(args.mujoco_gl)

    from models.model_backbone import load_config
    from eval.humanoid.g1_env import G1MujocoEnv
    from eval.humanoid.model_wrapper import (
        LFMHumanoidModel,
        apply_he_layout_overrides,
        load_he_action_stats,
        resolve_ckpt,
    )
    from eval.humanoid.tasks import filter_suite, load_tasks

    ckpt_path = resolve_ckpt(args.ckpt)
    configs = apply_he_layout_overrides(
        load_config(args.config),
        num_action_tokens=args.num_action_tokens,
        latent=args.latent,
    )
    stats = load_he_action_stats(args.data_root_dir)
    print(
        f"[he-eval] ckpt={ckpt_path}\n"
        f"         tokens={configs['act_head']['num_action_tokens']} "
        f"latent={configs['act_head']['latent']}\n"
        f"         stats={stats['path']}",
        flush=True,
    )

    policy = LFMHumanoidModel(
        ckpt_path=ckpt_path,
        configs=configs,
        action_stats=stats,
        device=args.device,
    )

    tasks = filter_suite(load_tasks(args.tasks_yaml), args.suite)
    out_dir = Path(
        args.output_dir
        or (_REPO_ROOT / "runs" / "g1_eval" / ckpt_path.parent.name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"

    env = G1MujocoEnv(control_hz=args.control_hz)
    results: Dict[str, Any] = {
        "ckpt": str(ckpt_path),
        "config": args.config,
        "num_action_tokens": args.num_action_tokens,
        "latent": args.latent,
        "suite": args.suite,
        "episodes_per_task": args.episodes_per_task,
        "tasks": {},
    }

    try:
        for task in tasks:
            succ = 0
            ep_rows = []
            for ep in range(args.episodes_per_task):
                seed = args.seed + ep
                obs = env.reset(task, seed=seed)
                policy.reset()
                frames = [obs["rgb"]] if args.save_video else []
                t0 = time.time()
                done = False
                info: Dict[str, Any] = {"success": False}
                while not done:
                    action = policy.step(
                        obs["rgb"], task.instruction, execute_step=args.execute_step
                    )
                    obs, reward, done, info = env.step(action)
                    if args.save_video:
                        frames.append(obs["rgb"])
                ok = bool(info.get("success"))
                succ += int(ok)
                row = {
                    "episode": ep,
                    "success": ok,
                    "steps": obs["step"],
                    "seconds": round(time.time() - t0, 2),
                }
                ep_rows.append(row)
                print(
                    f"[{task.suite}/{task.name}] ep={ep} success={ok} steps={obs['step']}",
                    flush=True,
                )
                if args.save_video and frames:
                    _save_mp4(
                        frames,
                        video_dir / f"{task.name}_ep{ep}_{'ok' if ok else 'fail'}.mp4",
                        fps=int(args.control_hz),
                    )
            rate = succ / max(1, args.episodes_per_task)
            results["tasks"][task.name] = {
                "suite": task.suite,
                "instruction": task.instruction,
                "success_rate": rate,
                "successes": succ,
                "episodes": ep_rows,
            }
            print(f"==> {task.name}: {succ}/{args.episodes_per_task} ({rate:.0%})", flush=True)
    finally:
        env.close()

    # Aggregate
    in_rates = [
        v["success_rate"]
        for k, v in results["tasks"].items()
        if v["suite"] == "in"
    ]
    ood_rates = [
        v["success_rate"]
        for k, v in results["tasks"].items()
        if v["suite"] == "ood"
    ]
    results["summary"] = {
        "in_suite_mean": float(sum(in_rates) / len(in_rates)) if in_rates else None,
        "ood_mean": float(sum(ood_rates) / len(ood_rates)) if ood_rates else None,
        "overall_mean": float(
            sum(v["success_rate"] for v in results["tasks"].values())
            / max(1, len(results["tasks"]))
        ),
        "policy_action_stats": policy.action_stats_summary(),
    }
    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"[he-eval] wrote {out_json}", flush=True)
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
