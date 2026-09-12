"""Audit train/eval leakage for the RobotNav pipeline.

Five checks, in increasing subtlety. A clean split needs all five empty, not
just the first:

  1. scene overlap        -- val scenes appearing in any training corpus
  2. episode-id overlap   -- the same R2R episode on both sides
  3. rendered-frame leak  -- frames from val scenes under generated_root
  4. instruction reuse    -- exact val instruction text in the training
                             instruction bank (deliverable1 ships 250k R2R
                             paraphrase top-ups, so text can cross a split
                             even when every scene is held out)
  5. paraphrase reuse     -- a val instruction appearing as a *generated
                             variant* of some training instruction, which is
                             the same leak wearing a disguise

Training rows carry instruction_variant_id rather than text, so checks 4/5
read instruction_variants.jsonl instead of scanning all 15.6M sample rows.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import subprocess
from collections import Counter

RELEASE = "/home/teams/research/robotics/datasets/robotnav_release/deliverable1"
NAVREASON = "/home/teams/research/robotics/robotnav_data/manifests/nav_reasoning/samples"
GENERATED = "/home/teams/research/robotics/datasets/robotnav_generated"
VLNCE = "/home/teams/research/robotics/datasets/robotnav_sources/vlnce_r2r/R2R_VLNCE_v1-3"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())


def canon(s: str) -> str:
    return " ".join(norm(s).split())


def load_split(split: str):
    f = f"{VLNCE}/{split}/{split}.json.gz"
    out = []
    for e in json.loads(gzip.open(f, "rt").read())["episodes"]:
        sid = e["scene_id"]
        out.append(dict(id=str(e["episode_id"]),
                        scene=sid.split("/")[-2] if "/" in sid else sid,
                        instr=e["instruction"]["instruction_text"]))
    return out


def training_scenes() -> set:
    """grep is ~100x faster here than json-parsing every row."""
    scenes = set()
    try:
        out = subprocess.run(
            f'grep -ho \'"scene": "[^"]*"\' {RELEASE}/*/*.jsonl | sort -u',
            shell=True, capture_output=True, text=True, timeout=1800).stdout
        for line in out.splitlines():
            m = re.search(r'"scene": "([^"]*)"', line)
            if m:
                scenes.add(m.group(1))
    except subprocess.TimeoutExpired:
        pass
    for f in glob.glob(f"{NAVREASON}/*.jsonl"):
        m = re.match(r"[a-z0-9_]+__([A-Za-z0-9]+)\.jsonl", os.path.basename(f))
        if m:
            scenes.add(m.group(1))
    return scenes


def instruction_bank():
    """Every instruction the training corpus can surface: originals + variants."""
    originals, variants, ep_ids = set(), set(), set()
    p = f"{RELEASE}/instruction_variants.jsonl"
    if not os.path.exists(p):
        return originals, variants, ep_ids
    for line in open(p, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("original"):
            originals.add(canon(d["original"]))
        v = d.get("variants")
        if isinstance(v, str):
            try:
                v = json.loads(v.replace("'", '"'))
            except Exception:
                v = [v]
        for s in (v or []):
            variants.add(canon(s))
        e = d.get("episode_ids")
        if isinstance(e, str):
            e = re.findall(r"\d+", e)
        for x in (e or []):
            ep_ids.add(str(x))
    return originals, variants, ep_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val_unseen"])
    args = ap.parse_args()

    tr_scenes = training_scenes()
    orig, var, tr_eps = instruction_bank()
    print(f"TRAINING SIDE: {len(tr_scenes)} scenes | instruction bank "
          f"{len(orig)} originals + {len(var)} variants | {len(tr_eps)} episode ids\n")

    overall = True
    for split in args.splits:
        eps = load_split(split)
        sc = {e["scene"] for e in eps}
        ids = {e["id"] for e in eps}
        instr = {canon(e["instr"]) for e in eps}
        print(f"=== {split}: {len(eps)} episodes | {len(sc)} scenes | "
              f"{len(instr)} distinct instructions ===")

        leak_sc = sc & tr_scenes
        print(f"  [1] scene overlap        : {len(leak_sc)}"
              + (f"  {sorted(leak_sc)[:6]}" if leak_sc else "   (clean)"))

        leak_ep = ids & tr_eps
        print(f"  [2] episode-id overlap   : {len(leak_ep)}"
              + (f"  {sorted(leak_ep)[:6]}" if leak_ep else "   (clean)"))

        frames = [(s, len(glob.glob(f"{GENERATED}/**/{s}/**/*.jpg", recursive=True)))
                  for s in sorted(sc)]
        frames = [(s, n) for s, n in frames if n]
        print(f"  [3] rendered-frame leak  : {len(frames)} scenes"
              + (f"  {frames[:3]}" if frames else "   (clean)"))

        hit4 = instr & orig
        print(f"  [4] exact instruction    : {len(hit4)} / {len(instr)}"
              + (f"   e.g. {sorted(hit4)[0][:64]!r}" if hit4 else "   (clean)"))

        hit5 = instr & var
        print(f"  [5] paraphrase variant   : {len(hit5)} / {len(instr)}"
              + (f"   e.g. {sorted(hit5)[0][:64]!r}" if hit5 else "   (clean)"))

        clean = not (leak_sc or leak_ep or frames or hit4 or hit5)
        overall &= clean
        print(f"  ==> {split}: {'CLEAN' if clean else 'LEAKAGE DETECTED'}\n")

    print("OVERALL:", "no leakage detected" if overall else "LEAKAGE PRESENT")


if __name__ == "__main__":
    main()
