"""Single-GPU LIBERO closed-loop evaluation + video recording for LFM4VLA checkpoints.

Runs a LIBERO task suite (e.g. ``libero_10``) in the robosuite/MuJoCo simulator, writes
a per-task / overall success-rate summary, and (optionally) an MP4 per episode so you can
watch the policy.

LIBERO is MuJoCo-based (robosuite). This needs the ``libero`` package + ``robosuite`` +
``mujoco`` installed (see scripts/install_libero_sim.sh). Best run on a workstation with a
GPU and working offscreen GL (EGL), e.g. your RTX 5090; on SLURM it auto-tries software
EGL and OSMesa when GPU EGL is unavailable.

Example (local, RTX 5090):
    MUJOCO_GL=egl python eval/libero/evaluate_libero.py \
        --config runs/logs/<date>/<exp>/<exp>-config.json \
        --ckpt   runs/checkpoints/<date>/<exp>/last.ckpt \
        --task_suite libero_10 \
        --num_trials_per_task 10 \
        --execute_step 5 \
        --save_video \
        --output_dir runs/libero_eval/<exp>
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

faulthandler.enable()

# Reduce CUDA allocator fragmentation across many small inference calls.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from eval.calvin.video_recorder import FrameRecorder
from eval.libero.model_wrapper import LFMLiberoModel, load_action_stats
from models.model_backbone import load_config

# Default max control steps per suite (matches OpenVLA's LIBERO eval budget).
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Map our --task_suite arg to the RLDS dataset name used at train time (for stats lookup).
SUITE_TO_DATASET = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


def _ensure_libero_config() -> None:
    """Write ~/.libero/config.yaml if missing (LIBERO calls input() otherwise).

    SLURM/Docker jobs have no stdin, so the first import of ``libero.libero`` would
    raise EOFError without this bootstrap step.
    """
    libero_config_path = os.environ.get(
        "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
    )
    config_file = os.path.join(libero_config_path, "config.yaml")
    if os.path.exists(config_file):
        return

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "libero package not installed. Run: bash scripts/install_libero_sim.sh"
        )
    benchmark_root = os.path.dirname(spec.origin)
    config = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "bddl_files"),
        "init_states": os.path.join(benchmark_root, "init_files"),
        "datasets": os.path.normpath(os.path.join(benchmark_root, "../datasets")),
        "assets": os.path.join(benchmark_root, "assets"),
    }
    os.makedirs(libero_config_path, exist_ok=True)
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(config, f)
    print(f"Created LIBERO config at {config_file}")


# Headless MuJoCo profiles tried in order when --mujoco_gl=auto.
# Newer mesalib (>=25.2) removed OSMesa; software EGL usually works instead.
MUJOCO_GL_PROFILES: tuple[dict[str, str], ...] = (
    {"name": "egl", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    {
        "name": "egl_software",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBGL_ALWAYS_SOFTWARE": "true",
    },
    {
        "name": "egl_headless",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBGL_ALWAYS_SOFTWARE": "true",
        "EGL_PLATFORM": "surfaceless",
    },
    {"name": "osmesa", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"},
)


def _prepend_conda_lib_to_env(env: dict[str, str]) -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib = os.path.join(conda_prefix, "lib")
    path = env.get("LD_LIBRARY_PATH", "")
    if lib not in path.split(os.pathsep):
        env["LD_LIBRARY_PATH"] = lib + (os.pathsep + path if path else "")


def _apply_mujoco_gl_profile(profile: dict[str, str]) -> None:
    _prepend_conda_lib_to_env(os.environ)
    for key, value in profile.items():
        if key != "name":
            os.environ[key] = value


def _probe_mujoco_gl(profile: dict[str, str]) -> tuple[bool, str]:
    """Return (ok, error_text) for importing mujoco under a GL profile."""
    env = os.environ.copy()
    _prepend_conda_lib_to_env(env)
    for key, value in profile.items():
        if key != "name":
            env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", "import mujoco"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, ""
    err = (result.stderr or result.stdout or "unknown error").strip()
    return False, err[-1500:]


def _profiles_for_request(requested: str) -> list[dict[str, str]]:
    if requested == "auto":
        profiles = list(MUJOCO_GL_PROFILES)
        env_pref = os.environ.get("MUJOCO_GL")
        if env_pref and env_pref not in ("auto", ""):
            preferred = [p for p in profiles if p["MUJOCO_GL"] == env_pref]
            others = [p for p in profiles if p["MUJOCO_GL"] != env_pref]
            profiles = preferred + others
        return profiles
    if requested == "egl":
        return [p for p in MUJOCO_GL_PROFILES if p["MUJOCO_GL"] == "egl"]
    if requested == "osmesa":
        return [p for p in MUJOCO_GL_PROFILES if p["MUJOCO_GL"] == "osmesa"]
    raise ValueError(f"unsupported --mujoco_gl value: {requested}")


def _configure_mujoco_gl(requested: str) -> str:
    """Pick a headless MuJoCo backend before robosuite/mujoco are imported."""
    profiles = _profiles_for_request(requested)
    failures: list[str] = []

    for profile in profiles:
        ok, err = _probe_mujoco_gl(profile)
        if ok:
            _apply_mujoco_gl_profile(profile)
            print(f"[render] using MuJoCo GL profile: {profile['name']}", flush=True)
            return profile["name"]
        summary = err.splitlines()[-1] if err else "import failed"
        failures.append(f"  {profile['name']}: {summary}")

    detail = "\n".join(failures)
    raise RuntimeError(
        "No headless MuJoCo GL backend available.\n"
        f"Probe results:\n{detail}\n\n"
        "Fix (run inside your conda env on a compute node):\n"
        "  conda install -y -c conda-forge mesalib libegl-devel glew\n"
        "  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n"
        "If osmesa still fails on newer mesalib, pin: conda install -c conda-forge 'mesalib<=25.1.0'\n"
        "Or rely on software EGL (auto mode tries egl_software / egl_headless)."
    )


def _make_env(task, resolution: int = 256):
    """Create a LIBERO OffScreenRenderEnv for a given task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def _get_agentview(obs) -> np.ndarray:
    """Extract the agentview RGB frame from a LIBERO observation dict."""
    return np.asarray(obs["agentview_image"])


def rollout(env, model, task, init_state, out_dir, task_i, ep_i, execute_step,
            max_steps, save_video, video_fps, num_wait_steps=10):
    """Run one episode; return (success, num_steps)."""
    instruction = task.language
    env.reset()
    obs = env.set_init_state(init_state)
    model.reset()

    # LIBERO physics settle: step a few no-op actions before control starts.
    dummy = np.zeros(7, dtype=np.float32)
    dummy[6] = -1.0  # keep gripper open while settling
    for _ in range(num_wait_steps):
        obs, _, _, _ = env.step(dummy)

    stem = f"task{task_i:02d}-ep{ep_i:02d}"
    recorder = FrameRecorder(out_dir, stem, fps=video_fps) if save_video else None

    success = False
    try:
        for step_i in range(max_steps):
            frame = _get_agentview(obs)
            if recorder is not None:
                # Record the human-viewable frame (rot180 to undo robosuite flip).
                recorder.add(np.ascontiguousarray(frame[::-1, ::-1]))
            action = model.step(frame, instruction, execute_step=execute_step)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                if recorder is not None:
                    recorder.add(np.ascontiguousarray(_get_agentview(obs)[::-1, ::-1]))
                return True, step_i + 1
        return False, max_steps
    finally:
        stats = model.action_stats()
        if stats:
            print(f"    action stats: {stats}", flush=True)
            if max(stats.get("arm_std_per_dim", [0])) < 1e-3:
                print("    [WARN] arm action std ~0 — policy may be frozen (not reacting "
                      "to image).", flush=True)
        if recorder is not None and recorder.count > 0:
            tag = "SUCC" if success else "FAIL"
            video_path = recorder.finalize(tag)
            if video_path is not None:
                print(f"    >>> watch rollout: {video_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="LFM4VLA LIBERO closed-loop eval (single GPU)")
    ap.add_argument("--config", required=True, help="Saved run config JSON (*-config.json)")
    ap.add_argument("--ckpt", required=True, help=".ckpt file or checkpoint dir")
    ap.add_argument("--task_suite", default="libero_10",
                    choices=list(SUITE_TO_DATASET.keys()),
                    help="LIBERO task suite to evaluate")
    ap.add_argument("--data_root_dir", default=None,
                    help="RLDS dir with dataset_statistics*.json (for action denorm). "
                         "Defaults to the train_dataset.data_root_dir in the config.")
    ap.add_argument("--num_trials_per_task", type=int, default=10,
                    help="Init-state episodes per task (LIBERO ships 50).")
    ap.add_argument("--execute_step", type=int, default=1,
                    help="Env steps replayed per model query (1 = closed-loop).")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Override per-episode step budget (default: per-suite).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gripper_open_positive", action="store_true",
                    help="Flip gripper sign if the robot closes when it should open.")
    ap.add_argument("--image_transform", default="rot180",
                    choices=["rot180", "flip_vertical", "none"],
                    help="Geometry applied to agentview before the policy (match training).")
    ap.add_argument("--save_video", action="store_true", help="Write an MP4 per episode")
    ap.add_argument("--video_fps", type=int, default=20)
    ap.add_argument("--output_dir", default="runs/libero_eval")
    ap.add_argument("--mujoco_gl", default=os.environ.get("MUJOCO_GL", "auto"),
                    choices=["auto", "egl", "osmesa"],
                    help="Headless MuJoCo renderer. 'auto' tries EGL, software-EGL, OSMesa.")
    args = ap.parse_args()

    _configure_mujoco_gl(args.mujoco_gl)
    np.random.seed(args.seed)

    configs = load_config(args.config)
    dataset_name = SUITE_TO_DATASET[args.task_suite]
    data_root_dir = args.data_root_dir or configs["train_dataset"]["data_root_dir"]
    action_stats = load_action_stats(data_root_dir, dataset_name)
    print(f"Loaded action stats from {action_stats['path']}")

    image_size = int(configs.get("train_dataset", {}).get("image_size", 224))
    max_steps = args.max_steps or SUITE_MAX_STEPS.get(args.task_suite, 520)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LFM policy from {args.ckpt}")
    model = LFMLiberoModel(
        args.ckpt,
        configs,
        action_stats=action_stats,
        device=args.device,
        image_size=image_size,
        image_transform=args.image_transform,
        gripper_open_is_negative=not args.gripper_open_positive,
        norm_min=float(configs.get("norm_min", -1.0)),
        norm_max=float(configs.get("norm_max", 1.0)),
    )

    _ensure_libero_config()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite]()
    num_tasks = suite.n_tasks
    print(f"Task suite '{args.task_suite}': {num_tasks} tasks, "
          f"{args.num_trials_per_task} episodes each, max_steps={max_steps}")

    per_task = {}
    results = []
    t0 = time.time()
    for task_i in range(num_tasks):
        task = suite.get_task(task_i)
        init_states = suite.get_task_init_states(task_i)
        env = _make_env(task, resolution=256)

        n_success = 0
        n_ep = min(args.num_trials_per_task, len(init_states))
        for ep_i in range(n_ep):
            ok, nsteps = rollout(
                env, model, task, init_states[ep_i], out_dir, task_i, ep_i,
                execute_step=args.execute_step, max_steps=max_steps,
                save_video=args.save_video, video_fps=args.video_fps,
            )
            n_success += int(ok)
            results.append(int(ok))
            print(f"[task {task_i+1}/{num_tasks} | ep {ep_i+1}/{n_ep}] "
                  f"'{task.language}' -> {'SUCCESS' if ok else 'fail'} ({nsteps} steps) | "
                  f"task SR so far: {n_success}/{ep_i+1}", flush=True)

        env.close()
        per_task[task.language] = {"success": n_success, "total": n_ep,
                                   "success_rate": n_success / max(n_ep, 1)}
        print(f"== task {task_i+1} done: {n_success}/{n_ep} "
              f"({100*n_success/max(n_ep,1):.1f}%) ==", flush=True)

    overall = float(np.mean(results)) if results else 0.0
    summary = {
        "task_suite": args.task_suite,
        "num_tasks": num_tasks,
        "num_trials_per_task": args.num_trials_per_task,
        "execute_step": args.execute_step,
        "max_steps": max_steps,
        "overall_success_rate": overall,
        "per_task": per_task,
        "ckpt": str(args.ckpt),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LIBERO results ===")
    for lang, r in per_task.items():
        print(f"  {r['success']:2d}/{r['total']:2d}  {r['success_rate']*100:5.1f}%  {lang}")
    print(f"  OVERALL: {overall*100:.1f}%  ({sum(results)}/{len(results)})")
    print(f"  Saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
