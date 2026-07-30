#!/usr/bin/env python3
"""Replay LIBERO demos and write HDF5 with agentview RGB + GT depth.

Stock ``modified_libero_rlds`` has no depth. This script replays raw LIBERO
HDF5 demos in the sim with ``camera_depths=True`` and writes a new HDF5 that
includes ``obs/agentview_depth`` (metric meters, float32) alongside RGB.

Usage:
  python scripts/regenerate_libero_hdf5_with_depth.py \\
      --libero_task_suite libero_10 \\
      --libero_raw_data_dir /path/to/LIBERO/libero/datasets/libero_10 \\
      --libero_target_dir /path/to/libero_10_depth \\
      --mujoco_gl auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root on path before local utils / late sim imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

IMAGE_RESOLUTION = 256


def is_noop(action, prev_action=None, threshold=1e-4) -> bool:
    import numpy as np

    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return (
        np.linalg.norm(action[:-1]) < threshold
        and action[-1] == prev_action[-1]
    )


def ensure_libero_config() -> None:
    """Create ~/.libero/config.yaml if missing (same as eval)."""
    import importlib.util

    import yaml

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
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    print(f"Created LIBERO config at {config_file}", flush=True)


def get_env(task, resolution: int = IMAGE_RESOLUTION):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=True,
    )
    env.seed(0)
    return env


def metric_depth(env, obs):
    import numpy as np
    from robosuite.utils.camera_utils import get_real_depth_map

    depth = np.asarray(obs["agentview_depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = get_real_depth_map(env.sim, depth)
    if isinstance(depth, np.ndarray) and depth.ndim == 3:
        depth = depth[..., 0]
    return np.asarray(depth, dtype=np.float32)


def main(args):
    # MUST configure GL before importing mujoco/robosuite/libero.
    from utils.mujoco_gl import configure_mujoco_gl

    configure_mujoco_gl(args.mujoco_gl)
    ensure_libero_config()

    import h5py
    import numpy as np
    import robosuite.utils.transform_utils as T
    import tqdm
    from libero.libero import benchmark

    os.makedirs(args.libero_target_dir, exist_ok=True)
    metainfo = {}
    meta_path = os.path.join(
        args.libero_target_dir, f"{args.libero_task_suite}_metainfo.json"
    )

    suite = benchmark.get_benchmark_dict()[args.libero_task_suite]()
    num_replays = num_success = num_noops = 0

    for task_id in tqdm.tqdm(range(suite.n_tasks)):
        task = suite.get_task(task_id)
        env = get_env(task)
        task_description = task.language

        orig_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        if not os.path.exists(orig_path):
            raise FileNotFoundError(orig_path)
        orig_f = h5py.File(orig_path, "r")
        orig = orig_f["data"]

        out_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        out_f = h5py.File(out_path, "w")
        grp = out_f.create_group("data")

        for i in range(len(orig.keys())):
            demo = orig[f"demo_{i}"]
            actions_in = demo["actions"][()]
            states_in = demo["states"][()]

            env.reset()
            env.set_init_state(states_in[0])
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

            states, actions, ee_states, gripper_states = [], [], [], []
            joint_states, robot_states = [], []
            agentview_images, eye_images, agentview_depths = [], [], []

            done = False
            for action in actions_in:
                prev = actions[-1] if actions else None
                if is_noop(action, prev):
                    num_noops += 1
                    continue

                if not states:
                    states.append(states_in[0])
                    robot_states.append(demo["robot_states"][0])
                else:
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate(
                            [
                                obs["robot0_gripper_qpos"],
                                obs["robot0_eef_pos"],
                                obs["robot0_eef_quat"],
                            ]
                        )
                    )

                actions.append(action)
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                    joint_states.append(obs["robot0_joint_pos"])
                    ee_states.append(
                        np.hstack(
                            (
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"]),
                            )
                        )
                    )
                agentview_images.append(obs["agentview_image"])
                eye_images.append(obs["robot0_eye_in_hand_image"])
                agentview_depths.append(metric_depth(env, obs))

                obs, reward, done, info = env.step(action.tolist())

            if done and actions:
                ep = grp.create_group(f"demo_{i}")
                obs_g = ep.create_group("obs")
                obs_g.create_dataset("gripper_states", data=np.stack(gripper_states))
                obs_g.create_dataset("joint_states", data=np.stack(joint_states))
                obs_g.create_dataset("ee_states", data=np.stack(ee_states))
                obs_g.create_dataset("agentview_rgb", data=np.stack(agentview_images))
                obs_g.create_dataset("eye_in_hand_rgb", data=np.stack(eye_images))
                obs_g.create_dataset(
                    "agentview_depth",
                    data=np.stack(agentview_depths).astype(np.float32),
                )
                ep.create_dataset("actions", data=np.asarray(actions))
                ep.create_dataset("states", data=np.stack(states))
                ep.create_dataset("robot_states", data=np.stack(robot_states))
                dones = np.zeros(len(actions), dtype=np.uint8)
                dones[-1] = 1
                rewards = dones.copy()
                ep.create_dataset("rewards", data=rewards)
                ep.create_dataset("dones", data=dones)
                num_success += 1

            num_replays += 1
            key = task_description.replace(" ", "_")
            metainfo.setdefault(key, {})[f"demo_{i}"] = {
                "success": bool(done),
                "initial_state": states_in[0].tolist(),
            }
            with open(meta_path, "w") as f:
                json.dump(metainfo, f, indent=2)

        orig_f.close()
        out_f.close()
        env.close()
        print(
            f"Saved {out_path}  (replays={num_replays}, success={num_success}, noops={num_noops})",
            flush=True,
        )

    print(f"Done. HDF5 with depth under {args.libero_target_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--libero_task_suite",
        required=True,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
    )
    ap.add_argument("--libero_raw_data_dir", required=True)
    ap.add_argument("--libero_target_dir", required=True)
    ap.add_argument(
        "--mujoco_gl",
        default=os.environ.get("MUJOCO_GL", "auto"),
        choices=["auto", "egl", "egl_software", "egl_headless", "osmesa"],
        help="Headless GL backend. 'auto' tries software EGL → OSMesa → GPU EGL.",
    )
    main(ap.parse_args())
