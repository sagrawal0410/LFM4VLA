"""Single-task CALVIN closed-loop eval: close_drawer only.

Runs isolated close_drawer rollouts in the PyBullet play-table environment with
drawer initially open. Ignores the standard 5-task long-horizon chains.

Example:
    python eval/calvin/evaluate_calvin_close_drawer.py \
        --config runs/logs/<date>/<exp>/<exp>-config.json \
        --ckpt runs/checkpoints/<date>/<exp>/last.ckpt \
        --calvin_root ~/calvin \
        --dataset_path /lambdafs/datasets/lfm4vla_calvin/calvin/dataset/task_ABC_D \
        --num_episodes 20 \
        --execute_step 10 \
        --save_video \
        --output_dir runs/calvin_eval/close_drawer/<exp>
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from itertools import product
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from omegaconf import OmegaConf

from eval.calvin.evaluate_calvin import (
    _ensure_numpy_legacy_aliases,
    _ensure_pyhash,
    _resolve_ckpt,
    rollout,
)
from models.model_backbone import load_config
from eval.calvin.model_wrapper import LFMCalvinModel

TASK_NAME = "close_drawer"


def get_close_drawer_initial_states(num_episodes: int, seed: int = 0) -> list[dict]:
    """Sample valid CALVIN scene configs with drawer open (required for close_drawer)."""
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }

    def _valid(block_locs):
        return (
            block_locs.count("table") in (1, 2)
            and block_locs.count("slider_right") < 2
            and block_locs.count("slider_left") < 2
        )

    value_combinations = filter(
        _valid,
        product(
            possible_conditions["led"],
            possible_conditions["lightbulb"],
            possible_conditions["slider"],
            possible_conditions["drawer"],
            possible_conditions["red_block"],
            possible_conditions["blue_block"],
            possible_conditions["pink_block"],
            possible_conditions["grasped"],
        ),
    )
    initial_states = [
        dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations
    ]
    if not initial_states:
        raise RuntimeError("No valid close_drawer initial states found")

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(initial_states),
        size=num_episodes,
        replace=num_episodes > len(initial_states),
    )
    return [initial_states[int(i)] for i in indices]


def evaluate_episode(
    env,
    model,
    task_oracle,
    initial_state,
    val_annotations,
    out_dir,
    episode_i,
    execute_step,
    save_video,
    episode_len: int,
    video_fps: int = 10,
):
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    return rollout(
        env,
        model,
        task_oracle,
        TASK_NAME,
        val_annotations,
        out_dir,
        episode_i,
        0,
        execute_step,
        save_video,
        episode_len=episode_len,
        video_fps=video_fps,
    )


def _write_summary(out_dir, results, initial_states, ckpt_path, config_path, execute_step, t0):
    success_rate = float(np.mean(results)) if results else 0.0
    summary = {
        "task": TASK_NAME,
        "num_episodes": len(results),
        "successes": int(sum(results)),
        "success_rate": success_rate,
        "execute_step": execute_step,
        "ckpt": ckpt_path.as_posix(),
        "config": str(config_path),
        "elapsed_sec": round(time.time() - t0, 1),
        "episodes": [
            {"episode": i, "success": bool(results[i]), "initial_state": initial_states[i]}
            for i in range(len(results))
        ],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="LFM4VLA CALVIN closed-loop eval (close_drawer only)"
    )
    ap.add_argument("--config", required=True, help="Run config JSON (the saved *-config.json)")
    ap.add_argument("--ckpt", required=True, help=".ckpt file or checkpoint dir")
    ap.add_argument("--calvin_root", required=True, help="Path to the calvin repo (for conf/)")
    ap.add_argument("--dataset_path", required=True, help="Path to task_ABC_D (contains validation/)")
    ap.add_argument("--num_episodes", type=int, default=20, help="Number of close_drawer rollouts")
    ap.add_argument("--episode_len", type=int, default=180,
                    help="Max env steps per rollout (CALVIN default is 360)")
    ap.add_argument("--execute_step", type=int, default=10, help="Open-loop steps per predicted chunk")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_video", action="store_true", help="Write an MP4 per rollout")
    ap.add_argument("--video_fps", type=int, default=10, help="FPS for rollout MP4/GIF")
    ap.add_argument("--output_dir", type=str, default="runs/calvin_eval/close_drawer")
    args = ap.parse_args()

    from pytorch_lightning import seed_everything

    seed_everything(args.seed, workers=True)
    _ensure_numpy_legacy_aliases()
    _ensure_pyhash()

    import hydra
    from calvin_env.envs.play_table_env import get_env

    configs = load_config(args.config)
    ckpt_path = _resolve_ckpt(args.ckpt)
    config_path = Path(args.config).expanduser().resolve()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_dir = Path(args.calvin_root) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    val_folder = Path(args.dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    print(f"Loading LFM policy from {ckpt_path}")
    model = LFMCalvinModel(ckpt_path, configs, device=args.device)

    initial_states = get_close_drawer_initial_states(args.num_episodes, seed=args.seed)

    results = []
    t0 = time.time()
    for episode_i, initial_state in enumerate(initial_states):
        try:
            ok = evaluate_episode(
                env,
                model,
                task_oracle,
                initial_state,
                val_annotations,
                out_dir,
                episode_i,
                args.execute_step,
                args.save_video,
                args.episode_len,
                args.video_fps,
            )
        except Exception:
            print(f"[episode {episode_i}] crashed:\n{traceback.format_exc()}")
            ok = False

        results.append(int(ok))
        sr = sum(results) / len(results) if results else 0.0
        print(
            f"[{episode_i + 1}/{len(initial_states)}] close_drawer "
            f"drawer={initial_state['drawer']} slider={initial_state['slider']} "
            f"{'SUCCESS' if ok else 'FAIL'} | running SR={sr * 100:.1f}%"
        )
        _write_summary(out_dir, results, initial_states[: len(results)], ckpt_path, config_path,
                       args.execute_step, t0)
        gc.collect()

    summary = _write_summary(out_dir, results, initial_states, ckpt_path, config_path,
                             args.execute_step, t0)
    success_rate = summary["success_rate"]

    print("\n=== close_drawer results ===")
    print(f"  Success rate: {success_rate * 100:.1f}% ({sum(results)}/{len(results)})")
    print(f"  Saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
