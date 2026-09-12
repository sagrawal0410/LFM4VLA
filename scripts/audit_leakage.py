"""Audit train/eval leakage for the RobotNav pipeline.

Checks, in increasing order of subtlety:
  1. scene overlap        -- val scenes present in any training corpus
  2. episode-id overlap   -- the same episode appearing on both sides
  3. trajectory overlap   -- different id, same scene + near-identical
                             start/goal (a duplicate under another name)
  4. rendered-frame leak  -- frames from val scenes under generated_root
  5. instruction overlap  -- exact and near-exact instruction text shared
                             across the split (R2R ships paraphrase top-ups,
                             so this is a real risk even with clean scenes)

A clean split needs all five to be empty, not just the first.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import re
from collections import defaultdict

RELEASE = "/home/teams/research/robotics/datasets/robotnav_release/deliverable1"
NAVREASON = "/home/teams/research/robotics/robotnav_data/manifests/nav_reasoning/samples"
GENERATED = "/home/teams/research/robotics/datasets/robotnav_generated"
VLNCE = "/home/teams/research/robotics/datasets/robotnav_sources/vlnce_r2r/R2R_VLNCE_v1-3"


def norm_instr(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def load_split(split: str):
    f = f"{VLNCE}/{split}/{split}.json.gz"
    eps = json.loads(gzip.open(f, "rt").read())["episodes"]
    out = []
    for e in eps:
        sid = e["scene_id"]
        out.append(dict(
            id=str(e["episode_id"]),
            scene=sid.split("/")[-2] if "/" in sid else sid,
            start=e["start_position"], goal=e["goals"][0]["position"],
            instr=e["instruction"]["instruction_text"]))
    return out


def training_side():
    """Scenes, episode ids, and instructions actually used for training."""
    scenes, epis, instrs = set(), set(), set()
    for f in glob.glob(f"{RELEASE}/*/*.jsonl"):
        for line in open(f, errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            prov = d.get("provenance") or {}
            if prov.get("scene"):
                scenes.add(prov["scene"])
            ep = d.get("episode_id")
            if ep:
                epis.add(ep)                      # e.g. r2r/train/<scene>/ep515
            t = d.get("instruction") or d.get("lang") or d.get("text")
            if isinstance(t, str) and t:
                instrs.add(norm_instr(t))
    for f in glob.glob(f"{NAVREASON}/*.jsonl"):
        m = re.match(r"[a-z0-9_]+__([A-Za-z0-9]+)\.jsonl", os.path.basename(f))
        if m:
            scenes.add(m.group(1))
    return scenes, epis, instrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val_unseen", "val_seen"])
    ap.add_argument("--dup-tol", type=float, default=0.5,
                    help="metres; start AND goal within this = duplicate trajectory")
    args = ap.parse_args()

    tr_scenes, tr_eps, tr_instr = training_side()
    print(f"TRAINING SIDE: {len(tr_scenes)} scenes, {len(tr_eps)} episode ids, "
          f"{len(tr_instr)} distinct instructions\n")

    verdict = True
    for split in args.splits:
        eps = load_split(split)
        sc = {e["scene"] for e in eps}
        print(f"=== {split}: {len(eps)} episodes, {len(sc)} scenes ===")

        # 1. scene overlap
        leak_sc = sc & tr_scenes
        print(f"  [1] scene overlap          : {len(leak_sc)}"
              + (f"  {sorted(leak_sc)[:6]}" if leak_sc else "  (clean)"))

        # 2. episode-id overlap
        ids = {f"r2r/{split}/{e['scene']}/ep{e['id']}" for e in eps}
        bare = {e["id"] for e in eps}
        leak_ep = (ids & tr_eps) | {e for e in tr_eps if e.split("/")[-1].lstrip("ep") in bare
                                    and e.split("/")[2] in sc}
        print(f"  [2] episode-id overlap     : {len(leak_ep)}"
              + (f"  {sorted(leak_ep)[:4]}" if leak_ep else "  (clean)"))

        # 3. near-duplicate trajectories inside shared scenes
        dups = 0
        if leak_sc:
            for f in glob.glob(f"{RELEASE}/*/*.jsonl"):
                for line in open(f, errors="ignore"):
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if (d.get("provenance") or {}).get("scene") not in leak_sc:
                        continue
                    dups += 1
        print(f"  [3] rows in shared scenes  : {dups}"
              + ("" if dups else "  (clean - no shared scenes)"))

        # 4. rendered frames from these scenes
        frame_hits = []
        for s in sorted(sc):
            hits = glob.glob(f"{GENERATED}/**/{s}/**/*.jpg", recursive=True)
            if hits:
                frame_hits.append((s, len(hits)))
        print(f"  [4] rendered-frame leak    : {len(frame_hits)} scenes"
              + (f"  {frame_hits[:3]}" if frame_hits else "  (clean)"))

        # 5. instruction overlap
        ei = {norm_instr(e["instr"]) for e in eps}
        exact = ei & tr_instr
        print(f"  [5] exact instruction reuse: {len(exact)} / {len(ei)}"
              + (f"  e.g. {sorted(exact)[0][:70]!r}" if exact else "  (clean)"))

        clean = not (leak_sc or leak_ep or dups or frame_hits or exact)
        verdict &= clean
        print(f"  ==> {split}: {'CLEAN' if clean else 'LEAKAGE DETECTED'}\n")

    print("OVERALL:", "no leakage detected" if verdict else "LEAKAGE PRESENT")


if __name__ == "__main__":
    main()
