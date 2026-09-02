"""Rank held-out VLN-CE episodes by how easy they are, for a clean 4-model comparison.

Easiness here means the failure modes we keep hitting are not in play: the goal
is close, on the same floor, already inside the camera's opening field of view,
reachable by a near-straight path, and described by one short unambiguous
sentence. Every candidate is also checked against the scene list actually
present in the training corpus, so a "held-out" episode is verified rather than
assumed from the split name.

    python scripts/pick_easy_episodes.py --split val_unseen --top 25
    python scripts/pick_easy_episodes.py --top 10 --emit-ep-ids
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

from eval.robotnav_sim import core

TRAIN_SCENES = "/tmp/train_scenes.txt"
# Multi-clause connectives and vague deixis are what make an instruction hard to
# ground; each occurrence costs a point.
VAGUE = re.compile(r"\b(then|after|once|until|around|past|through|toward|towards|"
                   r"continue|second|third|other side|all the way|back)\b", re.I)


def _rotate_by_conj(rot_xyzw, v: np.ndarray) -> np.ndarray:
    """Rotate v by the conjugate of quaternion [x, y, z, w].

    Inlined rather than pulling in `quaternion`, which only exists in the
    habitat env -- this way the picker runs from any interpreter.
    """
    x, y, z, w = (float(c) for c in rot_xyzw)
    u = np.array([-x, -y, -z])                     # conjugate
    uv = np.cross(u, v)
    return v + 2.0 * w * uv + 2.0 * np.cross(u, uv)


def load_pool(which: str, split: str) -> list[dict]:
    root = Path(core.PATHS[which])
    f = root / split / f"{split}.json.gz"
    if not f.exists():
        cands = [c for c in sorted(root.glob(f"**/{split}*.json.gz"))
                 if "_gt" not in c.name]
        if not cands:
            raise FileNotFoundError(f"no {split} episodes under {root}")
        guide = [c for c in cands if "guide" in c.name]
        f = (guide or cands)[0]
    eps = json.loads(gzip.open(f, "rt").read())["episodes"]
    if which == "vlnce_rxr":
        eps = [e for e in eps if str(
            e.get("instruction", {}).get("language", "en")).startswith("en")]
    pool = []
    for e in eps:
        sid = e["scene_id"]
        scene = sid.split("/")[-2] if "/" in sid else sid
        pool.append({
            "id": str(e["episode_id"]), "scene": scene,
            "start_pos": e["start_position"], "start_rot": e["start_rotation"],
            "goal": e["goals"][0]["position"], "ref_path": e.get("reference_path"),
            "instruction": e["instruction"]["instruction_text"].strip(),
        })
    return pool


def score(p: dict) -> dict | None:
    """Geometric + linguistic easiness. Returns None if it fails a hard gate."""
    v = np.asarray(p["goal"], dtype=float) - np.asarray(p["start_pos"], dtype=float)
    dy = abs(float(v[1]))
    if dy > 0.5:                                   # hard gate: same floor
        return None
    dist = math.hypot(float(v[0]), float(v[2]))
    if not (2.0 <= dist <= 6.0):                   # hard gate: close, not trivial
        return None
    lv = _rotate_by_conj(p["start_rot"], v)         # goal in the robot's frame
    bearing = abs(math.degrees(math.atan2(float(-lv[0]), float(-lv[2]))))
    if bearing > 30.0:                             # hard gate: already in view
        return None

    # Path straightness: reference path length vs straight-line distance.
    ref = p.get("ref_path") or []
    if len(ref) > 1:
        plen = sum(math.dist(ref[i][::2], ref[i + 1][::2])
                   for i in range(len(ref) - 1))
        detour = plen / max(dist, 1e-6)
    else:
        plen, detour = dist, 1.0
    if detour > 1.35:                              # hard gate: no doubling back
        return None

    instr = p["instruction"]
    n_char = len(instr)
    if n_char > 110:                               # hard gate: one short sentence
        return None
    n_clause = instr.count(",") + len(re.findall(r"\band\b", instr, re.I))
    n_vague = len(VAGUE.findall(instr))

    # Lower is easier. Weights chosen so a hard gate failure dominates any
    # single soft term, and bearing/detour (the things that actually strand the
    # controller) outweigh raw sentence length.
    s = (bearing / 30.0 * 2.0 + (detour - 1.0) / 0.35 * 2.0
         + n_char / 110.0 * 1.0 + n_clause * 0.6 + n_vague * 1.0
         + abs(dist - 3.5) / 2.5 * 0.5)
    return {**p, "dist": dist, "dy": dy, "bearing": bearing, "detour": detour,
            "path_len": plen, "n_char": n_char, "n_clause": n_clause,
            "n_vague": n_vague, "score": s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="vlnce_r2r")
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--per-scene", type=int, default=2,
                    help="cap per scene so the set is not one building")
    ap.add_argument("--emit-ep-ids", action="store_true",
                    help="print a comma-separated EP_IDS string and nothing else")
    args = ap.parse_args()

    train = set()
    if Path(TRAIN_SCENES).exists():
        train = {l.strip().split("-")[-1] for l in open(TRAIN_SCENES) if l.strip()}

    pool = load_pool(args.suite, args.split)
    scored, leaked = [], 0
    for p in pool:
        if p["scene"] in train:          # verified held-out, not assumed
            leaked += 1
            continue
        s = score(p)
        if s:
            scored.append(s)
    scored.sort(key=lambda d: d["score"])

    picked, per = [], {}
    for c in scored:
        if per.get(c["scene"], 0) >= args.per_scene:
            continue
        per[c["scene"]] = per.get(c["scene"], 0) + 1
        picked.append(c)
        if len(picked) >= args.top:
            break

    if args.emit_ep_ids:
        print(",".join(c["id"] for c in picked))
        return

    print(f"pool={len(pool)}  excluded_as_seen={leaked}  "
          f"passed_gates={len(scored)}  showing={len(picked)}")
    print("gates: same floor (dy<=0.5m) | 2-6m | bearing<=30deg | detour<=1.35 | "
          "instr<=110 chars\n")
    hdr = (f"{'#':>2} {'ep':>6} {'scene':<12} {'dist':>5} {'bear':>5} "
           f"{'detour':>6} {'chars':>5} {'cl':>2} {'vg':>2} {'score':>5}  instruction")
    print(hdr); print("-" * len(hdr))
    for i, c in enumerate(picked, 1):
        print(f"{i:>2} {c['id']:>6} {c['scene']:<12} {c['dist']:>5.2f} "
              f"{c['bearing']:>5.1f} {c['detour']:>6.2f} {c['n_char']:>5} "
              f"{c['n_clause']:>2} {c['n_vague']:>2} {c['score']:>5.2f}  "
              f"{c['instruction']}")


if __name__ == "__main__":
    main()
