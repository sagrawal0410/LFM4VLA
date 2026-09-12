"""Closed-loop visual evaluation over Habitat-Sim with rollout videos.

Run under the HABITAT env (robotnav-sim):
  python -m eval.robotnav_sim.run --suite vlnce_r2r --split val_unseen \
      --config configs/<train>.json --ckpt <last.ckpt> --episodes 10 \
      --out results/r2r_unseen [--policy-device cuda|cpu]

Suites: vlnce_r2r, vlnce_rxr (MP3D; splits train/val_seen/val_unseen),
        objectnav_hm3d, objectnav_mp3d (splits train/val).
Stubs (data or assets not yet staged): hm3d_ovon, evt_bench, hm_eqa,
        mt_hm3d, express_bench — see suite_stubs().
Metrics per episode + aggregate: NE, SR, OS, SPL, path length. Every episode
also writes rollout.mp4 with a HUD (instruction / step / action / distance).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

from eval.robotnav_sim import core


def easy_filter(pool, max_instr=130, max_bear_deg=45.0, dmin=1.5, dmax=7.0):
    """Short-instruction episodes whose goal is close, near the start floor,
    and inside the camera's initial field of view (bearing proxy)."""
    import quaternion as qt
    out = []
    for p in pool:
        if len(p["instruction"]) > max_instr:
            continue
        v = np.asarray(p["goal"]) - np.asarray(p["start_pos"])
        if abs(float(v[1])) > 1.0:               # different floor
            continue
        if not (dmin <= math.hypot(float(v[0]), float(v[2])) <= dmax):
            continue
        r = p["start_rot"]                       # [x, y, z, w]
        q = np.quaternion(r[3], r[0], r[1], r[2])
        lv = qt.rotate_vectors(q.conjugate(), v)
        bear = math.degrees(math.atan2(float(-lv[0]), float(-lv[2])))
        if abs(bear) > max_bear_deg:
            continue
        out.append(p)
    return out


# ------------------------------------------------------------- episode IO --
def load_vlnce(which: str, split: str, limit: int, easy: bool = False,
               max_instr: int = 130):
    root = Path(core.PATHS[which])
    f = root / split / f"{split}.json.gz"
    if not f.exists():
        cands = [c for c in sorted(root.glob(f"**/{split}*.json.gz"))
                 if "_gt" not in c.name]        # gt files: different schema
        if not cands:
            raise FileNotFoundError(f"no {split} episodes under {root}")
        guide = [c for c in cands if "guide" in c.name]
        f = (guide or cands)[0]
    data = json.loads(gzip.open(f, "rt").read())
    eps = data["episodes"]
    if which == "vlnce_rxr":
        eps = [e for e in eps
               if str(e.get("instruction", {}).get("language", "en")).startswith("en")]
    pool = []
    for e in eps:
        scene = e["scene_id"].split("/")[-2] if "/" in e["scene_id"] else e["scene_id"]
        pool.append({
            "id": str(e["episode_id"]),
            "scene_glb": core.mp3d_glb(scene),
            "scene": scene,
            "start_pos": e["start_position"],
            "start_rot": e["start_rotation"],
            "goal": e["goals"][0]["position"],
            "ref_path": e.get("reference_path"),
            "instruction": e["instruction"]["instruction_text"],
            "success_dist": 3.0,
            "family": "vln_r2r" if which == "vlnce_r2r" else "vln_rxr",
            "turn_deg": 15.0 if which == "vlnce_r2r" else 30.0,
        })
    if easy:
        pool = easy_filter(pool, max_instr=max_instr)
    # round-robin across scenes so a small sample isn't one building, then
    # regroup by scene so the sim reloads once per scene, not per episode
    by_scene: dict = {}
    for p in pool:
        by_scene.setdefault(p["scene"], []).append(p)
    out = []
    while len(out) < limit and any(by_scene.values()):
        for s in list(by_scene):
            if by_scene[s]:
                out.append(by_scene[s].pop(0))
                if len(out) >= limit:
                    break
            else:
                del by_scene[s]
    out.sort(key=lambda p: p["scene"])
    return out


def load_objectnav(which: str, split: str, limit: int):
    root = Path(core.PATHS[which])
    # Per-scene content shards hold the episodes; the split-root <split>.json.gz
    # is a stub with an empty episode list (and no goals_by_category), so it
    # must not win the lookup or the suite loads zero episodes.
    cands = sorted(root.glob(f"**/{split}/content/*.json.gz")) or \
        sorted(root.glob(f"**/{split}/{split}.json.gz")) or \
        sorted(root.glob(f"**/{split}*.json.gz"))
    if not cands:
        raise FileNotFoundError(
            f"no objectnav episodes under {root} — set ROBOTNAV_OBJNAV_* env")
    out = []
    for f in cands:
        data = json.loads(gzip.open(f, "rt").read())
        # MP3D ObjectNav v1 leaves episode["goals"] empty and stores the target
        # instances once per scene under goals_by_category, keyed
        # "<scene>.glb_<category>". Without this the episodes load with no goal.
        by_cat = data.get("goals_by_category") or {}
        for e in data.get("episodes", []):
            sid = e["scene_id"]
            scene = sid.split("/")[-1].replace(".basis.glb", "").replace(".glb", "")
            if which == "objectnav_hm3d":
                scene_dir = sid.split("/")[-2]
                hsplit = "val" if "val" in split else "train"
                glb = core.hm3d_glb(scene_dir, hsplit)
            else:
                glb = core.mp3d_glb(scene)
            cat = e.get("object_category", "object")
            goals = [g["position"] for g in e.get("goals", [])] or None
            if not goals:
                key = f"{scene}.glb_{cat}"
                inst = by_cat.get(key) or by_cat.get(f"{sid.split('/')[-1]}_{cat}")
                goals = [g["position"] for g in inst] if inst else None
            if not goals:
                continue          # unresolvable target: skip rather than crash
            out.append({
                "id": str(e["episode_id"]), "scene_glb": glb, "scene": scene,
                "start_pos": e["start_position"], "start_rot": e["start_rotation"],
                "goal": goals[0] if goals else None, "goals": goals,
                "instruction": f"Find the {cat} and stop next to it.",
                "cat": cat,
                "success_dist": 1.0,
                "family": which, "turn_deg": 30.0,
            })
            if len(out) >= limit:
                return out
    return out


def select_history(n_frames: int, n_hist: int = 8, mode: str = "uniform",
                   alpha: float = 2.0, jitter: bool = False, rng=None):
    """Indices of the history frames to feed, oldest -> newest (current last).

    mode:
      uniform  — evenly spread over the whole past (matches non-rand training)
      latest   — the n_hist most recent consecutive frames (sliding window)
      recency  — power-law spacing: dense near the present, sparse far back,
                 while still anchoring the oldest slot at frame 0. alpha=1
                 degenerates to uniform; larger alpha skews harder to recent.
                 With jitter, the older slots are drawn randomly inside their
                 bucket so repeat visits don't always see identical old frames.
    """
    cur = n_frames - 1
    if n_frames <= 1:
        return [0]
    k = min(n_hist, n_frames - 1)
    if mode == "latest":
        idx = list(range(max(0, cur - k), cur))
    elif mode == "recency":
        idx, prev = [], None
        for j in range(k, 0, -1):                 # far -> near
            off = int(round(((j / k) ** alpha) * cur))
            lo = int(round((((j - 1) / k) ** alpha) * cur))
            if jitter and rng is not None and lo < off:
                off = rng.randint(lo + 1, off)
            i = max(0, min(cur - 1, cur - off))
            if prev is not None and i <= prev:     # keep strictly increasing
                i = min(cur - 1, prev + 1)
            idx.append(i)
            prev = i
    else:
        idx = list(np.unique(np.linspace(0, cur - 1, k).astype(int)))
    idx = sorted(set(int(i) for i in idx if 0 <= i < cur))
    return idx + [cur]


def suite_stubs(name: str):
    notes = {
        "hm3d_ovon": "HM3D-OVON annotations are staged under robotnav_sources/"
                     "hm3d_ovon; adapter = objectnav loader with open-vocab "
                     "categories. Wire load_objectnav at that root.",
        "evt_bench": "EVT-Bench needs the vendored habitat-lab shim from "
                     "robotnav_data tooling (moving humanoid target).",
        "hm_eqa": "HM-EQA episode data not downloaded (not part of the "
                  "training corpus). Fetch from the HM-EQA release, then "
                  "adapt load_objectnav-style loader + answer-accuracy metric.",
        "mt_hm3d": "MT-HM3D data not downloaded; same pattern as hm_eqa.",
        "express_bench": "EXPRESS-Bench data not downloaded; same pattern.",
    }
    raise SystemExit(f"[{name}] not wired yet: {notes[name]}")


# ---------------------------------------------------------------- rollout --
def rollout(sim, client, ep, args, recorder):
    import quaternion  # noqa: F401  (habitat dep, ensures registry)
    rot = ep["start_rot"]
    q = np.quaternion(rot[3], rot[0], rot[1], rot[2])
    core.set_agent(sim, ep["start_pos"], q)
    ctrl = core.WaypointController(turn_deg=ep["turn_deg"],
                                   stop_radius=args.stop_radius,
                                   stop_mode=args.stop_mode,
                                   min_steps_before_stop=args.min_steps_before_stop,
                                   stop_debounce=args.stop_debounce)
    goals = ep.get("goals") or [ep["goal"]]
    start_geo = core.geodesic_min(sim, ep["start_pos"], goals)
    ref = ep.get("ref_path")
    ref_dense = core.densify_path(ref) if ref and len(ref) > 1 else None
    import random as _random
    hist_rng = _random.Random(hash(ep["id"]) % (2 ** 31))
    plan, plan_used = None, 0
    turn_rad = math.radians(ep["turn_deg"])
    from PIL import Image
    frames_hist = []
    positions = [list(ep["start_pos"])]
    action = "start"
    for step in range(args.max_steps):
        obs = sim.get_sensor_observations()
        rgb = obs["front"][..., :3]
        pil = Image.fromarray(rgb)
        frames_hist.append(pil)
        hidx = select_history(len(frames_hist), n_hist=8,
                              mode=args.history_mode, alpha=args.history_alpha,
                              jitter=args.history_jitter, rng=hist_rng)
        sel = [frames_hist[i] for i in hidx]
        if step % 20 == 0:
            print(f"[hist] mode={args.history_mode} step={step} idx={hidx}",
                  flush=True)
        # Replan every `replan_every` actions. In between, the cached plan is
        # re-expressed in the robot's new frame after each executed action, so
        # waypoints 2..k are actually followed instead of discarded. k=1 is the
        # original behaviour (a fresh forward pass per action).
        dt = 0.0
        if plan is None or plan_used >= args.replan_every:
            t0 = time.time()
            plan = client.predict(ep["instruction"], sel, ep["family"],
                                  frame_ids=hidx)
            dt = time.time() - t0
            plan_used = 0
        wps = plan
        action = ctrl.act(wps, fresh=(plan_used == 0), step=step)
        st = sim.get_agent(0).get_state()
        d_goal = core.geodesic_min(sim, st.position, goals)
        near_goal = core.nearest_goal(sim, st.position, goals)
        # goal in the robot's ego frame (habitat local: fwd=-z, left=-x)
        gl = quaternion.rotate_vectors(
            st.rotation.conjugate(),
            np.asarray(near_goal) - np.asarray(st.position))
        import textwrap
        recorder.add(rgb, [
            f"{ep['family']} ep{ep['id']} step {step} act={action} ({dt:.1f}s)",
            *textwrap.wrap("instr: " + ep["instruction"], width=104)[:6],
            f"dist-to-goal {d_goal:.2f} m | wp8=({wps[-1][0]:+.2f},{wps[-1][1]:+.2f})",
        ], overlay={
            "wps": wps, "action": action,
            "goal_ego": (float(-gl[2]), float(-gl[0])),
            "gt_path": (core.path_to_ego(ref_dense, st)
                        if ref_dense is not None else None),
            "path_ego": core.next_path_dir(sim, st, near_goal),
            "lookahead": ctrl.lookahead, "stop_radius": ctrl.stop_radius,
        })
        if action == "stop":
            break
        sim.step(action)
        plan = core.advance_plan(plan, action, 0.25, turn_rad)
        plan_used += 1
        positions.append(list(sim.get_agent(0).get_state().position))
    m = core.episode_metrics(sim, goals, positions, start_geo,
                             ep["success_dist"])
    m.update({"episode": ep["id"], "scene": ep["scene"], "steps": len(positions),
              "stopped": action == "stop"})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=["vlnce_r2r", "vlnce_rxr", "objectnav_hm3d",
                             "objectnav_mp3d", "hm3d_ovon", "evt_bench",
                             "hm_eqa", "mt_hm3d", "express_bench"])
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-ids", default="",
                    # ObjectNav needs "<scene>:<category>:<id>" (ids restart per
                    # category within a scene); bare ids work for VLN
                    help="comma-separated episode ids to run (overrides "
                         "round-robin selection)")
    ap.add_argument("--easy", action="store_true",
                    help="only short-instruction episodes with the goal "
                         "nearby, same floor, inside the initial camera FOV")
    ap.add_argument("--history-mode", default="uniform",
                    choices=["uniform", "latest", "recency"],
                    help="how the 8 history frames are chosen each step")
    ap.add_argument("--history-alpha", type=float, default=2.0,
                    help="recency mode: >1 skews toward recent frames")
    ap.add_argument("--history-jitter", action="store_true",
                    help="recency mode: randomize the older slots")
    ap.add_argument("--replan-every", type=int, default=1,
                    help="actions executed per policy forward pass (1 = replan "
                         "every step; 5 = follow 5 waypoints then re-observe)")
    ap.add_argument("--max-instr", type=int, default=130,
                    help="easy mode: max instruction length in characters "
                         "(RxR needs ~450; R2R fits in 130)")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--out", default="results/rollouts")
    ap.add_argument("--stop-debounce", type=int, default=1,
                    help="honour a stop only after this many consecutive fresh "
                         "plans request it; 1 = current behaviour")
    ap.add_argument("--stop-mode", default="geometric",
                    choices=["geometric", "plan_static", "both"],
                    help="geometric: controller decides from the last waypoint's "
                         "distance. plan_static: the MODEL decides -- stop when "
                         "it emits a plan that stops changing, which is what "
                         "terminal-hold training teaches it to produce.")
    ap.add_argument("--min-steps-before-stop", type=int, default=7,
                    help="suppress the stop test for this many steps; 0 lets "
                         "the model's own signal speak from step 0")
    ap.add_argument("--stop-radius", type=float, default=0.50,
                    # Fires the stop test while the model still predicts this
                    # much remaining travel. Raising it stops earlier along the
                    # path, which counteracts the observed 1-2 m overshoot.
                    help="metres of predicted remaining travel below which the "
                         "plan counts as a hold")
    ap.add_argument("--policy-device", default="cuda")
    args = ap.parse_args()

    if args.suite in ("hm3d_ovon", "evt_bench", "hm_eqa", "mt_hm3d",
                      "express_bench"):
        suite_stubs(args.suite)
    if args.episode_ids:
        want = {s.strip() for s in args.episode_ids.split(",") if s.strip()}
        big = 10 ** 6                       # pull the full pool, then filter
        pool = (load_vlnce(args.suite, args.split, big)
                if args.suite.startswith("vlnce")
                else load_objectnav(args.suite, args.split, big))
        # ObjectNav ids restart per scene -- "2" exists in most of the 11 val
        # scenes, so a bare id selects hundreds of episodes. Accept the
        # scene-qualified form "<scene>:<id>" and keep bare ids for VLN, whose
        # ids are unique across the split.
        eps = [e for e in pool
               if e["id"] in want
               or f'{e["scene"]}:{e["id"]}' in want
               or f'{e["scene"]}:{e.get("cat")}:{e["id"]}' in want]
    elif args.suite.startswith("vlnce"):
        eps = load_vlnce(args.suite, args.split, args.episodes, easy=args.easy,
                         max_instr=args.max_instr)
    else:
        eps = load_objectnav(args.suite, args.split, args.episodes)
    unique_ids = len({e["id"] for e in eps}) == len(eps)
    print(f"[suite] {args.suite}/{args.split}: {len(eps)} episodes")

    client = core.PolicyClient(args.config, args.ckpt, device=args.policy_device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    sim, cur_glb = None, None
    try:
        for ep in eps:
            if ep["scene_glb"] != cur_glb:
                if sim is not None:
                    sim.close()
                    sim, cur_glb = None, None
                try:
                    sim = core.make_eval_sim(ep["scene_glb"],
                                             turn_deg=ep["turn_deg"])
                    cur_glb = ep["scene_glb"]
                except Exception as e:  # noqa: BLE001 — missing scene asset
                    print(f"[ep {ep['id']}] scene {ep['scene']} load FAILED: "
                          f"{e}", flush=True)
                    continue
            # ObjectNav ids repeat across scenes and categories, so a bare
            # "ep<id>.mp4" silently overwrites an earlier episode's video.
            # Qualify the name whenever the pool is not uniquely keyed by id.
            stem = f"ep{ep['id']}"
            if not unique_ids:
                stem = f"{ep['scene']}-{ep.get('cat', ep['family'])}-ep{ep['id']}"
            rec = core.RolloutRecorder(str(out / f"{stem}.mp4"))
            try:
                m = rollout(sim, client, ep, args, rec)
            except Exception as e:  # noqa: BLE001
                print(f"[ep {ep['id']}] FAILED: {e}", flush=True)
                continue
            video = rec.save()
            m["video"] = video
            results.append(m)
            print(json.dumps(m), flush=True)
    finally:
        client.close()
        if sim is not None:
            sim.close()

    if results:
        agg = {k: round(float(np.mean([r[k] for r in results])), 4)
               for k in ("ne_m", "sr", "os", "spl", "steps")}
        agg.update({"suite": args.suite, "split": args.split,
                    "episodes": len(results), "ckpt": args.ckpt})
        (out / "metrics.json").write_text(json.dumps(
            {"aggregate": agg, "episodes": results}, indent=2))
        print("[aggregate]", json.dumps(agg))


if __name__ == "__main__":
    main()
