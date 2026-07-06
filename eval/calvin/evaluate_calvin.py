"""Single-GPU CALVIN closed-loop evaluation + visualization for LFM4VLA checkpoints.

Runs the standard CALVIN long-horizon protocol (chains of language subtasks) in the
PyBullet play-table environment and writes a success-rate summary plus a GIF per subtask
rollout so you can watch the policy.

CALVIN is PyBullet-based (NOT MuJoCo / IsaacSim). This must run on a machine with the
`calvin_env` + `calvin_agent` packages installed and a GPU with EGL rendering (e.g. your
RTX 4090 desktop). See the header docstring in the repo chat / README for setup.

Example:
    python eval/calvin/evaluate_calvin.py \
        --config runs/logs/<date>/<exp>/<exp>-config.json \
        --ckpt   runs/checkpoints/<date>/<exp>/last.ckpt \
        --calvin_root ~/calvin \
        --dataset_path ~/calvin/dataset/task_ABC_D \
        --num_sequences 20 \
        --execute_step 10 \
        --save_video \
        --output_dir runs/calvin_eval/<exp>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Headless GPU rendering for PyBullet/pyrender. Set before importing sim libs.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from eval.calvin.obs_utils import obs_to_uint8_rgb
from models.model_backbone import load_config
from eval.calvin.model_wrapper import LFMCalvinModel

EP_LEN = 360

_FNV1_32_INIT = 0x811C9DC5
_FNV_32_PRIME = 0x01000193


def _ensure_numpy_legacy_aliases() -> None:
    """CALVIN env code still uses NumPy 1.x aliases removed in NumPy 2.0."""
    for name, typ in (
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("complex", complex),
        ("object", object),
        ("str", str),
    ):
        if not hasattr(np, name):
            setattr(np, name, typ)


def _ensure_pyhash() -> None:
    """CALVIN imports pyhash; the PyPI wheel fails to build on Python 3.10+."""
    if "pyhash" in sys.modules:
        return
    try:
        import pyhash  # noqa: F401
        return
    except ImportError:
        pass

    class _Fnv1_32:
        def __call__(self, *parts, seed=0):
            h = seed if seed else _FNV1_32_INIT
            for part in parts:
                data = part if isinstance(part, (bytes, bytearray)) else str(part).encode()
                for byte in data:
                    h = (h * _FNV_32_PRIME) & 0xFFFFFFFF
                    h ^= byte
            return h

    class _PyhashShim:
        @staticmethod
        def fnv1_32():
            return _Fnv1_32()

    sys.modules["pyhash"] = _PyhashShim()


def _resolve_ckpt(ckpt: str) -> Path:
    path = Path(ckpt).expanduser().resolve()
    if path.is_dir():
        cand = path / "last.ckpt"
        if cand.is_file():
            return cand
        ckpts = sorted(path.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt in {path}")
        return ckpts[-1]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _rollout_stem(out_dir: Path, seq_i: int, subtask_i: int, subtask: str) -> Path:
    return out_dir / f"seq{seq_i:03d}-{subtask_i}-{subtask}"


def _is_valid_frame(frame: np.ndarray) -> bool:
    return (
        frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] >= 64
        and frame.shape[1] >= 64
        and float(frame.std()) > 2.0
    )


def rollout(env, model, task_oracle, subtask, val_annotations, out_dir, seq_i, subtask_i,
            execute_step, save_video, episode_len: int = EP_LEN, video_fps: int = 10):
    from eval.calvin.video_recorder import FrameRecorder

    obs = env.get_obs()
    lang = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    success = False
    stem = _rollout_stem(out_dir, seq_i, subtask_i, subtask)
    recorder = FrameRecorder(out_dir, stem.name, fps=video_fps) if save_video else None
    try:
        for step_i in range(episode_len):
            action = model.step(obs, lang, execute_step=execute_step)
            obs, _, _, current_info = env.step(action)
            if recorder is not None:
                frame = obs_to_uint8_rgb(obs)
                if _is_valid_frame(frame):
                    recorder.add(frame)
                    if step_i % 30 == 0:
                        print(f"  recorded frame {recorder.count}", flush=True)
            done = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(done) > 0:
                success = True
                return True
        return False
    finally:
        if recorder is not None and recorder.count > 0:
            tag = "SUCC" if success else "FAIL"
            video_path = recorder.finalize(tag)
            if video_path is not None:
                print(f"  >>> watch rollout: {video_path}", flush=True)


def evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations,
                      out_dir, seq_i, execute_step, save_video):
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    solved = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        ok = rollout(env, model, task_oracle, subtask, val_annotations, out_dir, seq_i,
                     subtask_i, execute_step, save_video)
        if ok:
            solved += 1
        else:
            break
    return solved


def count_success(results):
    """Fraction of sequences completing at least i+1 consecutive subtasks, for i in 0..4."""
    out = []
    for i in range(5):
        n = sum(1 for r in results if r > i)
        out.append(n / len(results) if results else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="LFM4VLA CALVIN closed-loop eval (single GPU)")
    ap.add_argument("--config", required=True, help="Run config JSON (the saved *-config.json)")
    ap.add_argument("--ckpt", required=True, help=".ckpt file or checkpoint dir")
    ap.add_argument("--calvin_root", required=True, help="Path to the calvin repo (for conf/)")
    ap.add_argument("--dataset_path", required=True, help="Path to task_ABC_D (contains validation/)")
    ap.add_argument("--num_sequences", type=int, default=20, help="Eval chains (full benchmark = 1000)")
    ap.add_argument("--execute_step", type=int, default=10, help="Open-loop steps per predicted chunk")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_video", action="store_true", help="Write a GIF per subtask rollout")
    ap.add_argument("--output_dir", type=str, default="runs/calvin_eval")
    args = ap.parse_args()

    from pytorch_lightning import seed_everything
    seed_everything(args.seed, workers=True)

    _ensure_numpy_legacy_aliases()
    _ensure_pyhash()
    import hydra
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_env.envs.play_table_env import get_env

    configs = load_config(args.config)
    ckpt_path = _resolve_ckpt(args.ckpt)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CALVIN task oracle + language annotations.
    conf_dir = Path(args.calvin_root) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Environment (PyBullet).
    val_folder = Path(args.dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    print(f"Loading LFM policy from {ckpt_path}")
    model = LFMCalvinModel(ckpt_path, configs, device=args.device)

    eval_sequences = get_sequences(args.num_sequences)

    results = []
    t0 = time.time()
    for seq_i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        solved = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence,
                                   val_annotations, out_dir, seq_i, args.execute_step, args.save_video)
        results.append(solved)
        sr = count_success(results)
        print(f"[{seq_i + 1}/{len(eval_sequences)}] chain={'->'.join(eval_sequence)} solved={solved}/5 | "
              + " ".join(f"{i+1}:{v*100:.1f}%" for i, v in enumerate(sr)))

    sr = count_success(results)
    avg_len = float(np.mean(results)) if results else 0.0
    summary = {
        "num_sequences": len(results),
        "success_rate_at_k": {str(i + 1): sr[i] for i in range(5)},
        "avg_seq_len": avg_len,
        "task_completion_counts": dict(Counter(results)),
        "execute_step": args.execute_step,
        "ckpt": ckpt_path.as_posix(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CALVIN results ===")
    for i in range(5):
        print(f"  {i+1} consecutive tasks: {sr[i]*100:.1f}%")
    print(f"  Avg sequence length: {avg_len:.3f} / 5")
    print(f"  Saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
