"""Pick ObjectNav MP3D episodes whose target is genuinely visible at step 0.

Two phases. The cheap phase filters on episode metadata (same floor, sensible
range, target inside the camera cone, unambiguous category). The sim phase
then loads each scene once and *verifies* what metadata can only suggest:

  * true geodesic distance from the pathfinder, so "clear path" means the
    walked route is barely longer than the straight line, and
  * an unoccluded ray from the camera to the target, so "visible at frame 0"
    is measured rather than assumed.

    python scripts/pick_objectnav_easy.py --top 20
    python scripts/pick_objectnav_easy.py --top 10 --emit-ep-ids
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os

import numpy as np

from eval.robotnav_sim import core

MP3D_OBJNAV = os.environ.get(
    "ROBOTNAV_OBJNAV_MP3D",
    "/home/teams/research/robotics/datasets/robotnav_sources/habitat_objectnav/mp3d")
TRAIN_SCENES = "/tmp/train_scenes.txt"

# Large, visually unambiguous furniture reads clearly at a distance; "cushion"
# or "picture" can be metres away and still be a guess, which defeats the point
# of an easy set.
CLEAR = {"bed", "sofa", "toilet", "bathtub", "tv_monitor", "fireplace",
         "table", "counter", "sink", "shower", "plant", "chest_of_drawers"}


def _rot_conj(rot_xyzw, v: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(c) for c in rot_xyzw)
    u = np.array([-x, -y, -z])
    uv = np.cross(u, v)
    return v + 2.0 * w * uv + 2.0 * np.cross(u, uv)


def load_val(split: str = "val") -> list[dict]:
    out = []
    for f in sorted(glob.glob(f"{MP3D_OBJNAV}/{split}/content/*.json.gz")):
        d = json.loads(gzip.open(f, "rt").read())
        by_cat = d.get("goals_by_category") or {}
        for e in d.get("episodes", []):
            sid = e["scene_id"]
            scene = sid.split("/")[-1].replace(".basis.glb", "").replace(".glb", "")
            cat = e.get("object_category", "object")
            inst = by_cat.get(f"{scene}.glb_{cat}")
            if not inst:
                continue
            out.append({"id": str(e["episode_id"]), "scene": scene, "cat": cat,
                        "start_pos": e["start_position"],
                        "start_rot": e["start_rotation"],
                        "goals": [g["position"] for g in inst]})
    return out


def prefilter(e: dict, G) -> dict | None:
    """Metadata-only gates; cheap enough to run over every episode."""
    sp = np.asarray(e["start_pos"], dtype=float)
    best = None
    for g in e["goals"]:
        gp = np.asarray(g, dtype=float)
        v = gp - sp
        if abs(float(v[1])) > G.max_dy:                 # different floor
            continue
        dist = math.hypot(float(v[0]), float(v[2]))
        if not (G.dist_min <= dist <= G.dist_max):
            continue
        lv = _rot_conj(e["start_rot"], v)
        bearing = abs(math.degrees(math.atan2(float(-lv[0]), float(-lv[2]))))
        if bearing > G.max_bearing:                     # outside the cone
            continue
        if best is None or dist < best["dist"]:
            best = {"goal": gp, "dist": dist, "bearing": bearing}
    if best is None:
        return None
    if G.clear_only and e["cat"] not in CLEAR:
        return None
    return {**e, **best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--per-scene", type=int, default=2)
    ap.add_argument("--dist-min", type=float, default=3.0)
    ap.add_argument("--dist-max", type=float, default=10.0)
    ap.add_argument("--max-dy", type=float, default=1.0)
    ap.add_argument("--max-bearing", type=float, default=35.0)
    ap.add_argument("--max-detour", type=float, default=1.3)
    ap.add_argument("--clear-only", action="store_true", default=True)
    ap.add_argument("--any-category", dest="clear_only", action="store_false")
    ap.add_argument("--no-sim", action="store_true",
                    help="skip geodesic + visibility verification")
    ap.add_argument("--emit-ep-ids", action="store_true")
    args = ap.parse_args()

    train = set()
    if os.path.exists(TRAIN_SCENES):
        train = {l.strip().split("-")[-1] for l in open(TRAIN_SCENES) if l.strip()}

    eps = load_val(args.split)
    cands, leaked = [], 0
    for e in eps:
        if e["scene"] in train:
            leaked += 1
            continue
        c = prefilter(e, args)
        if c:
            cands.append(c)

    verified = []
    if args.no_sim:
        for c in cands:
            verified.append({**c, "geo": c["dist"], "detour": 1.0, "visible": None})
    else:
        import habitat_sim
        by_scene: dict[str, list] = {}
        for c in cands:
            by_scene.setdefault(c["scene"], []).append(c)
        for scene, group in sorted(by_scene.items()):
            glb = core.mp3d_glb(scene)
            if not os.path.exists(glb):
                continue
            sim = core.make_eval_sim(glb, turn_deg=30.0)
            try:
                for c in group:
                    geo = core.geodesic(sim, c["start_pos"], c["goal"])
                    if not math.isfinite(geo) or geo <= 0:
                        continue
                    detour = geo / max(c["dist"], 1e-6)
                    if detour > args.max_detour:
                        continue
                    vis = ray_visible(sim, c["start_pos"], c["goal"])
                    verified.append({**c, "geo": geo, "detour": detour,
                                     "visible": vis})
            finally:
                sim.close()
        verified = [v for v in verified if v["visible"]]

    for v in verified:
        v["score"] = (v["bearing"] / max(args.max_bearing, 1e-6) * 2.0
                      + (v["detour"] - 1.0) / max(args.max_detour - 1.0, 1e-6) * 2.0
                      + abs(v["geo"] - 5.0) / 5.0)
    verified.sort(key=lambda d: d["score"])

    picked, per = [], {}
    for c in verified:
        if per.get(c["scene"], 0) >= args.per_scene:
            continue
        per[c["scene"]] = per.get(c["scene"], 0) + 1
        picked.append(c)
        if len(picked) >= args.top:
            break

    if args.emit_ep_ids:
        print(",".join(c["id"] for c in picked))
        return

    print(f"episodes={len(eps)} excluded_as_seen={leaked} "
          f"prefiltered={len(cands)} verified_visible={len(verified)} "
          f"showing={len(picked)}")
    print(f"gates: dy<={args.max_dy} | {args.dist_min}-{args.dist_max}m | "
          f"bearing<={args.max_bearing}deg | detour<={args.max_detour} | "
          f"{'clear categories only' if args.clear_only else 'any category'} | "
          f"{'ray-verified visible' if not args.no_sim else 'NOT verified'}\n")
    hdr = (f"{'#':>2} {'ep':>7} {'scene':<12} {'category':<16} {'geo':>5} "
           f"{'bear':>5} {'detour':>6} {'vis':>4} {'score':>5}")
    print(hdr); print("-" * len(hdr))
    for i, c in enumerate(picked, 1):
        print(f"{i:>2} {c['id']:>7} {c['scene']:<12} {c['cat']:<16} "
              f"{c['geo']:>5.2f} {c['bearing']:>5.1f} {c['detour']:>6.2f} "
              f"{str(c['visible']):>4} {c['score']:>5.2f}")


def ray_visible(sim, eye, target, cam_height: float = 1.0) -> bool:
    """True when nothing blocks the camera's line of sight to the target.

    Cast from the camera toward the target and compare the first hit against
    the target range: a hit meaningfully short of it means a wall or furniture
    is in the way, so the object is not actually on screen at step 0.
    """
    import habitat_sim
    o = np.asarray(eye, dtype=np.float32).copy()
    o[1] += cam_height
    t = np.asarray(target, dtype=np.float32).copy()
    t[1] += 0.3                                   # aim at the body, not the floor
    d = t - o
    rng = float(np.linalg.norm(d))
    if rng < 1e-6:
        return True
    ray = habitat_sim.geo.Ray(o, d / rng)
    hits = sim.cast_ray(ray, max_distance=rng * 1.05)
    if not hits.has_hits():
        return True                                # clear all the way through
    first = min(h.ray_distance for h in hits.hits) * rng
    return first >= rng * 0.90                     # tolerate hitting the object


if __name__ == "__main__":
    main()
