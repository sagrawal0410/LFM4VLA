"""Render policy-free scene tours: a 360 spin, then a walk along a shortest
path between random navigable points — judging raw Habitat-Sim scene quality
at the robot's camera config (640x400, hfov 105, height 1.0 m).

Run under the habitat env (robotnav-sim):
  python -m eval.robotnav_sim.scene_tour --scenes s8pcmisQ38h,17DRP5sb8fy \
      --out /path/tours [--hm3d-split none] [--seed 7] [--fps 15]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from eval.robotnav_sim import core


def yaw_quat(yaw: float):
    import quaternion  # noqa: F401
    return np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)


def heading_to(a, b) -> float:
    d = np.asarray(b) - np.asarray(a)
    return math.atan2(-float(d[0]), -float(d[2]))   # habitat fwd = -z


def tour_frames(sim, seed: int, spin_frames: int = 60, walk_step: float = 0.08,
                min_geo: float = 8.0, tries: int = 60):
    import habitat_sim
    rng = np.random.default_rng(seed)
    pf = sim.pathfinder
    frames = []

    start = pf.get_random_navigable_point()
    # 360 spin at the start point
    for i in range(spin_frames):
        core.set_agent(sim, start, yaw_quat(2 * math.pi * i / spin_frames))
        frames.append(sim.get_sensor_observations()["front"][..., :3])

    # walk a long-ish shortest path
    best = None
    for _ in range(tries):
        goal = pf.get_random_navigable_point()
        p = habitat_sim.ShortestPath()
        p.requested_start, p.requested_end = start, goal
        if pf.find_path(p) and len(p.points) > 1 and \
                p.geodesic_distance >= min_geo:
            best = list(p.points)
            break
    if best is None:
        return frames
    dense = core.densify_path(best, step_m=walk_step)
    yaw = heading_to(dense[0], dense[1])
    for i, pt in enumerate(dense[:-1]):
        tgt = heading_to(pt, dense[min(i + 6, len(dense) - 1)])
        d = (tgt - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw += np.clip(d, -0.12, 0.12)          # smooth the turns
        core.set_agent(sim, pt, yaw_quat(yaw))
        frames.append(sim.get_sensor_observations()["front"][..., :3])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True,
                    help="comma-separated MP3D scene ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as iio
    for scene in args.scenes.split(","):
        scene = scene.strip()
        try:
            sim = core.make_eval_sim(core.mp3d_glb(scene), turn_deg=15.0)
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] scene load FAILED: {e}", flush=True)
            continue
        try:
            frames = tour_frames(sim, seed=args.seed)
            f = out / f"tour_{scene}.mp4"
            iio.mimwrite(str(f), frames, fps=args.fps, quality=7,
                         macro_block_size=1)
            print(f"[{scene}] {len(frames)} frames -> {f}", flush=True)
        finally:
            sim.close()


if __name__ == "__main__":
    main()
