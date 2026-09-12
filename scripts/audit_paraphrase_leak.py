"""Check 5, done properly: does any val instruction reach the training corpus?

Exact string matching is close to useless here -- a paraphrase is by
definition not an exact match. But naive fuzzy matching is worse than useless,
because R2R instructions share a tiny vocabulary ("walk down the hall and turn
left"), so *every* pair looks similar and any threshold is arbitrary.

So this does four things:

  A. PROVENANCE (decisive). instruction_variants.jsonl records the episode ids
     each entry was built from. If any entry references a val episode id, that
     val instruction is literally in the training bank. This is ground truth,
     not a similarity heuristic.

  B. EXACT collision after canonicalisation.

  C. NEAR-DUPLICATE search over all training originals + generated variants,
     using an inverted index on rare tokens so it is tractable, scored by
     token Jaccard and character 5-gram containment.

  D. NULL BASELINE. The same similarity computed between training
     instructions from *different* trajectories -- pairs that cannot be
     leakage. A val/train match only means something if it sits outside this
     distribution. Without D, C is unfalsifiable.

Verdict logic: leakage is claimed only when A is non-empty, or when C produces
matches above the null's extreme upper tail.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import re
from collections import Counter, defaultdict

RELEASE = "/home/teams/research/robotics/datasets/robotnav_release/deliverable1"
VLNCE = "/home/teams/research/robotics/datasets/robotnav_sources/vlnce_r2r/R2R_VLNCE_v1-3"

STOP = set("the a an and or of to in on at is are be go goes going walk walks "
           "then you your it its there here with into onto from for by".split())


def canon(s) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def toks(s: str) -> set:
    return {w for w in canon(s).split() if w not in STOP}


def ngrams(s: str, n: int = 5) -> set:
    c = canon(s).replace(" ", "")
    return {c[i:i + n] for i in range(max(len(c) - n + 1, 0))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set, b: set) -> float:
    """|A∩B| / min(|A|,|B|) -- catches a short instruction embedded in a long one."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def load_split(split: str):
    f = f"{VLNCE}/{split}/{split}.json.gz"
    out = []
    for e in json.loads(gzip.open(f, "rt").read())["episodes"]:
        out.append(dict(id=str(e["episode_id"]),
                        traj=str(e.get("trajectory_id", e["episode_id"])),
                        scene=e["scene_id"].split("/")[-2],
                        text=e["instruction"]["instruction_text"]))
    return out


def load_bank():
    """Every instruction string the training corpus can surface, with provenance."""
    entries = []
    p = f"{RELEASE}/instruction_variants.jsonl"
    for line in open(p, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        eids = d.get("episode_ids")
        if isinstance(eids, str):
            eids = re.findall(r"\d+", eids)
        eids = [str(x) for x in (eids or [])]
        v = d.get("variants")
        if isinstance(v, str):
            try:
                v = json.loads(v.replace("'", '"'))
            except Exception:
                v = re.findall(r"'([^']{10,})'", v) or [v]
        texts = [("original", d.get("original", ""))] + \
                [("variant", s) for s in (v or [])]
        for kind, t in texts:
            if t:
                entries.append(dict(kind=kind, text=t, key=d.get("key", ""),
                                    eids=eids))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val_unseen", "val_seen"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--null-pairs", type=int, default=20000)
    args = ap.parse_args()

    bank = load_bank()
    print(f"training instruction bank: {len(bank)} strings "
          f"({sum(1 for b in bank if b['kind']=='original')} originals, "
          f"{sum(1 for b in bank if b['kind']=='variant')} variants)")

    bank_tok = [toks(b["text"]) for b in bank]
    bank_ng = [ngrams(b["text"]) for b in bank]
    bank_canon = [canon(b["text"]) for b in bank]
    canon_index = defaultdict(list)
    for i, c in enumerate(bank_canon):
        canon_index[c].append(i)

    # inverted index on non-ubiquitous tokens keeps the search tractable
    df = Counter()
    for t in bank_tok:
        df.update(t)
    rare = {w for w, c in df.items() if c <= len(bank) * 0.02}
    inv = defaultdict(list)
    for i, t in enumerate(bank_tok):
        for w in (t & rare):
            inv[w].append(i)
    print(f"inverted index: {len(rare)} discriminative tokens\n")

    # ---- D. null baseline: training vs training, different trajectories ----
    rng = random.Random(0)
    null = []
    for _ in range(args.null_pairs):
        i, j = rng.randrange(len(bank)), rng.randrange(len(bank))
        if i == j or bank[i]["key"] == bank[j]["key"]:
            continue
        null.append(max(jaccard(bank_tok[i], bank_tok[j]),
                        containment(bank_ng[i], bank_ng[j])))
    null.sort()
    def q(p):
        return null[min(int(p * len(null)), len(null) - 1)]
    p50, p99, p999, pmax = q(.50), q(.99), q(.999), null[-1]
    print(f"NULL (train-vs-train, different trajectories, n={len(null)}):")
    print(f"  median={p50:.3f}  p99={p99:.3f}  p99.9={p999:.3f}  max={pmax:.3f}")
    print(f"  -> a val/train pair is only suspicious above ~{pmax:.3f}\n")

    overall_clean = True
    for split in args.splits:
        eps = load_split(split)
        print(f"=== {split}: {len(eps)} episodes, "
              f"{len({e['traj'] for e in eps})} trajectories ===")

        # ---- A. provenance: does the bank reference val episode ids? -------
        val_ids = {e["id"] for e in eps}
        prov_hits = [b for b in bank if val_ids & set(b["eids"])]
        print(f"  [A] bank entries citing a {split} episode id : {len(prov_hits)}")
        if prov_hits:
            for b in prov_hits[:5]:
                print(f"        key={b['key']} kind={b['kind']} eids={b['eids'][:4]}")

        # ---- B. exact collision -------------------------------------------
        exact = [e for e in eps if canon(e["text"]) in canon_index]
        print(f"  [B] exact instruction collisions            : {len(exact)}")
        for e in exact[:3]:
            print(f"        ep{e['id']} {e['scene']}: {e['text'][:70]!r}")

        # ---- C. near-duplicate search --------------------------------------
        best = []
        for e in eps:
            et, en = toks(e["text"]), ngrams(e["text"])
            cand = Counter()
            for w in (et & rare):
                for i in inv[w]:
                    cand[i] += 1
            top = [i for i, c in cand.most_common(400) if c >= 2]
            bi, bs = -1, 0.0
            for i in top:
                s = max(jaccard(et, bank_tok[i]), containment(en, bank_ng[i]))
                if s > bs:
                    bi, bs = i, s
            if bi >= 0:
                best.append((bs, e, bank[bi]))
        best.sort(key=lambda x: -x[0])
        above = [b for b in best if b[0] > pmax]
        print(f"  [C] val instructions above the null max      : {len(above)} "
              f"/ {len(eps)}")
        print(f"      top {args.top} most similar val<->train pairs:")
        for s, e, b in best[:args.top]:
            flag = "  <-- ABOVE NULL" if s > pmax else ""
            print(f"        sim={s:.3f}{flag}")
            print(f"          {split}: {e['text'][:78]!r}")
            print(f"          train({b['kind']}): {b['text'][:78]!r}")

        clean = not prov_hits and not exact and not above
        overall_clean &= clean
        print(f"  ==> {split}: "
              f"{'CLEAN' if clean else 'SUSPECT - inspect the pairs above'}\n")

    print("OVERALL:", "no paraphrase leakage detected" if overall_clean
          else "POSSIBLE PARAPHRASE LEAKAGE")


if __name__ == "__main__":
    main()
