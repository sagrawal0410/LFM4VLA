"""Audit Humanoid Everyday egocentric depth for valid vs corrupted/invalid pixels.

Checks each episode's depth volume (``.npy`` preferred, parquet fallback) and
reports how much of the data is usable under the same rules used by training
(``HumanoidEverydayDataset._prepare_depth`` / ``extract_he_depth_arrays.py``):

  valid pixel:   finite, ``0 < z < 1e5``
  invalid:       ``<= 0`` (incl. missing / RealSense holes)
  corrupted:     NaN, ±Inf, or ``>= 1e5`` (incl. float16-overflow sentinels)

Usage:
    conda activate lfm4vla-he
    python scripts/audit_he_depth.py \\
        --data_root /home/teams/research/robotics/humanoid_everyday_depth

    # faster smoke check
    python scripts/audit_he_depth.py --data_root ... --limit 50 --workers 8

    # force parquet (ignore npy)
    python scripts/audit_he_depth.py --data_root ... --source parquet

Writes ``meta/depth_audit.json`` under ``data_root`` (override with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.humanoid_everyday_dataset import (
    DEPTH_COLUMN,
    SKILL_KEYWORDS,
    depth_npy_rel,
    parquet_rel,
    task_matches_skills,
)

FLOAT16_MAX = float(np.finfo(np.float16).max)  # 65504
ABSURD_MAX = 1.0e5


def _select_episodes(root: Path, skills, robot_type) -> List[int]:
    tasks = {}
    with open(root / "meta/tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            tasks[t["task_index"]] = t
    selected = []
    with open(root / "meta/episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            if robot_type and ep.get("robot_type") != robot_type:
                continue
            if task_matches_skills(tasks[ep["tasks"][0]]["task"], skills):
                selected.append(int(ep["episode_index"]))
    return selected


def _load_depth(root: Path, ep_idx: int, source: str) -> Tuple[np.ndarray, str]:
    """Return ``[T,H,W]`` float32 depth and the source label used."""
    npy = root / depth_npy_rel(ep_idx)
    pq_path = root / parquet_rel(ep_idx)

    if source == "npy" or (source == "auto" and npy.is_file()):
        if not npy.is_file():
            raise FileNotFoundError(npy)
        arr = np.asarray(np.load(npy, mmap_mode="r"), dtype=np.float32)
        src = "npy"
    elif source in ("parquet", "auto"):
        if not pq_path.is_file():
            raise FileNotFoundError(pq_path)
        import pyarrow.parquet as pq

        table = pq.read_table(pq_path, columns=[DEPTH_COLUMN])
        frames = table.column(DEPTH_COLUMN).to_pylist()
        arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
        src = "parquet"
    else:
        raise ValueError(source)

    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"ep {ep_idx}: unexpected shape {arr.shape}")
    return arr, src


def _audit_array(arr: np.ndarray) -> Dict[str, Any]:
    """Pixel-level stats for one ``[T,H,W]`` volume (float32 view)."""
    # Work on a contiguous float64 reduce for stable counts without full copy when possible.
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    if n == 0:
        return {
            "num_pixels": 0,
            "num_frames": 0,
            "valid": 0,
            "invalid_nonpositive": 0,
            "corrupted_nan": 0,
            "corrupted_inf": 0,
            "corrupted_absurd": 0,
            "float16_overflow_risk": 0,
            "valid_frac": 0.0,
            "corrupted_frac": 0.0,
            "invalid_frac": 0.0,
            "frames_all_invalid": 0,
            "frames_high_corrupt": 0,
            "frames_ok": 0,
            "finite_min": None,
            "finite_max": None,
            "finite_mean": None,
            "dtype_in": str(arr.dtype),
            "shape": list(arr.shape),
        }

    nan_m = np.isnan(flat)
    inf_m = np.isinf(flat)
    finite = np.isfinite(flat)
    absurd_m = finite & (flat >= ABSURD_MAX)
    nonpos_m = finite & (flat <= 0.0)
    # Would become Inf if cast to float16 without clipping.
    f16_risk = finite & (flat > FLOAT16_MAX) & (flat < ABSURD_MAX)
    valid_m = finite & (flat > 0.0) & (flat < ABSURD_MAX)

    n_nan = int(nan_m.sum())
    n_inf = int(inf_m.sum())
    n_absurd = int(absurd_m.sum())
    n_nonpos = int(nonpos_m.sum())
    n_f16 = int(f16_risk.sum())
    n_valid = int(valid_m.sum())
    n_corrupt = n_nan + n_inf + n_absurd

    # Per-frame quality (same rules).
    t, h, w = arr.shape
    frame = np.asarray(arr, dtype=np.float32)
    f_nan = np.isnan(frame).reshape(t, -1)
    f_inf = np.isinf(frame).reshape(t, -1)
    f_fin = np.isfinite(frame).reshape(t, -1)
    f_val = (f_fin & (frame.reshape(t, -1) > 0.0) & (frame.reshape(t, -1) < ABSURD_MAX))
    f_corrupt = f_nan | f_inf | (f_fin & (frame.reshape(t, -1) >= ABSURD_MAX))
    valid_frac_f = f_val.mean(axis=1)
    corrupt_frac_f = f_corrupt.mean(axis=1)

    frames_all_invalid = int((valid_frac_f == 0.0).sum())
    frames_high_corrupt = int((corrupt_frac_f >= 0.01).sum())  # ≥1% corrupt pixels
    frames_ok = int(((valid_frac_f >= 0.5) & (corrupt_frac_f < 0.01)).sum())

    finite_vals = flat[valid_m]
    return {
        "num_pixels": n,
        "num_frames": int(t),
        "hw": [int(h), int(w)],
        "valid": n_valid,
        "invalid_nonpositive": n_nonpos,
        "corrupted_nan": n_nan,
        "corrupted_inf": n_inf,
        "corrupted_absurd": n_absurd,
        "float16_overflow_risk": n_f16,
        "valid_frac": float(n_valid / n),
        "corrupted_frac": float(n_corrupt / n),
        "invalid_frac": float(n_nonpos / n),
        "frames_all_invalid": frames_all_invalid,
        "frames_high_corrupt": frames_high_corrupt,
        "frames_ok": frames_ok,
        "finite_min": float(finite_vals.min()) if finite_vals.size else None,
        "finite_max": float(finite_vals.max()) if finite_vals.size else None,
        "finite_mean": float(finite_vals.mean()) if finite_vals.size else None,
        "dtype_in": str(arr.dtype),
        "shape": [int(t), int(h), int(w)],
    }


def _classify_episode(stats: Dict[str, Any]) -> str:
    """Coarse episode label for triage."""
    if stats["num_pixels"] == 0:
        return "empty"
    if stats["corrupted_frac"] >= 0.01 or stats["corrupted_nan"] or stats["corrupted_inf"]:
        return "corrupted"
    if stats["valid_frac"] < 0.1:
        return "mostly_invalid"
    if stats["valid_frac"] < 0.5:
        return "degraded"
    if stats["float16_overflow_risk"] > 0:
        return "float16_risk"
    return "ok"


def _audit_one(root_str: str, ep_idx: int, source: str) -> Dict[str, Any]:
    root = Path(root_str)
    try:
        arr, src = _load_depth(root, ep_idx, source)
        stats = _audit_array(arr)
        stats["episode_index"] = ep_idx
        stats["source"] = src
        stats["label"] = _classify_episode(stats)
        stats["error"] = None
    except Exception as e:  # noqa: BLE001
        stats = {
            "episode_index": ep_idx,
            "source": None,
            "label": "missing",
            "error": f"{type(e).__name__}: {e}",
            "num_pixels": 0,
            "valid_frac": 0.0,
            "corrupted_frac": 0.0,
            "invalid_frac": 0.0,
            "valid": 0,
            "corrupted_nan": 0,
            "corrupted_inf": 0,
            "corrupted_absurd": 0,
            "invalid_nonpositive": 0,
            "float16_overflow_risk": 0,
            "frames_ok": 0,
            "frames_all_invalid": 0,
            "frames_high_corrupt": 0,
            "num_frames": 0,
        }
    return stats


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels: Dict[str, int] = {}
    tot_pix = valid = nonpos = nan = inf = absurd = f16 = 0
    frames = frames_ok = frames_bad = frames_corrupt = 0
    sources: Dict[str, int] = {}

    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
        src = r.get("source") or "none"
        sources[src] = sources.get(src, 0) + 1
        tot_pix += int(r.get("num_pixels") or 0)
        valid += int(r.get("valid") or 0)
        nonpos += int(r.get("invalid_nonpositive") or 0)
        nan += int(r.get("corrupted_nan") or 0)
        inf += int(r.get("corrupted_inf") or 0)
        absurd += int(r.get("corrupted_absurd") or 0)
        f16 += int(r.get("float16_overflow_risk") or 0)
        frames += int(r.get("num_frames") or 0)
        frames_ok += int(r.get("frames_ok") or 0)
        frames_bad += int(r.get("frames_all_invalid") or 0)
        frames_corrupt += int(r.get("frames_high_corrupt") or 0)

    n_ep = len(rows)
    corrupt = nan + inf + absurd
    return {
        "num_episodes": n_ep,
        "episode_labels": labels,
        "sources": sources,
        "pixels": {
            "total": tot_pix,
            "valid": valid,
            "invalid_nonpositive": nonpos,
            "corrupted_nan": nan,
            "corrupted_inf": inf,
            "corrupted_absurd": absurd,
            "corrupted_total": corrupt,
            "float16_overflow_risk": f16,
            "valid_frac": (valid / tot_pix) if tot_pix else 0.0,
            "invalid_frac": (nonpos / tot_pix) if tot_pix else 0.0,
            "corrupted_frac": (corrupt / tot_pix) if tot_pix else 0.0,
        },
        "frames": {
            "total": frames,
            "ok": frames_ok,
            "all_invalid": frames_bad,
            "high_corrupt_ge_1pct": frames_corrupt,
            "ok_frac": (frames_ok / frames) if frames else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--skills", nargs="+", default=list(SKILL_KEYWORDS),
                    choices=list(SKILL_KEYWORDS))
    ap.add_argument("--robot_type", default="g1", choices=["g1", "h1", ""])
    ap.add_argument("--source", default="auto", choices=["auto", "npy", "parquet"],
                    help="auto: prefer depths/*.npy, else parquet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="JSON report path (default: <data_root>/meta/depth_audit.json)")
    ap.add_argument("--top_worst", type=int, default=20,
                    help="Print this many worst episodes by corrupted_frac then valid_frac")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    episodes = _select_episodes(root, args.skills, args.robot_type)
    if args.limit:
        episodes = episodes[: args.limit]

    print(f"Auditing {len(episodes)} episodes under {root}")
    print(f"  skills={args.skills} robot={args.robot_type or 'all'} source={args.source}")
    print(
        "  rules: valid = finite & (0, 1e5); "
        "invalid = <=0; corrupted = NaN|Inf|>=1e5; "
        f"float16_risk = ({FLOAT16_MAX}, 1e5)"
    )

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_audit_one, str(root), ep, args.source): ep for ep in episodes
        }
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 50 == 0 or i == len(episodes):
                print(f"  {i}/{len(episodes)}", flush=True)

    rows.sort(key=lambda r: r["episode_index"])
    summary = _aggregate(rows)

    # Worst episodes: most corrupted, then least valid.
    ranked = sorted(
        rows,
        key=lambda r: (-float(r.get("corrupted_frac") or 0.0), float(r.get("valid_frac") or 0.0)),
    )
    worst = [
        {
            "episode_index": r["episode_index"],
            "label": r["label"],
            "source": r.get("source"),
            "valid_frac": r.get("valid_frac"),
            "corrupted_frac": r.get("corrupted_frac"),
            "invalid_frac": r.get("invalid_frac"),
            "float16_overflow_risk": r.get("float16_overflow_risk"),
            "finite_min": r.get("finite_min"),
            "finite_max": r.get("finite_max"),
            "error": r.get("error"),
        }
        for r in ranked[: max(0, args.top_worst)]
    ]

    report = {
        "data_root": str(root),
        "skills": list(args.skills),
        "robot_type": args.robot_type,
        "source_mode": args.source,
        "rules": {
            "valid": "finite and 0 < z < 1e5",
            "invalid_nonpositive": "finite and z <= 0",
            "corrupted": "NaN or Inf or z >= 1e5",
            "float16_overflow_risk": f"finite and {FLOAT16_MAX} < z < 1e5",
        },
        "summary": summary,
        "worst_episodes": worst,
        "episodes": rows,
    }

    out = Path(args.out) if args.out else root / "meta" / "depth_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    pix = summary["pixels"]
    fr = summary["frames"]
    lab = summary["episode_labels"]
    print("\n=== HE depth audit ===")
    print(f"episodes: {summary['num_episodes']}  labels: {lab}")
    print(f"sources:  {summary['sources']}")
    print(
        f"pixels:   valid={pix['valid_frac']:.4%}  "
        f"invalid(<=0)={pix['invalid_frac']:.4%}  "
        f"corrupted={pix['corrupted_frac']:.4%}  "
        f"(nan={pix['corrupted_nan']} inf={pix['corrupted_inf']} "
        f"absurd={pix['corrupted_absurd']} f16_risk={pix['float16_overflow_risk']})"
    )
    print(
        f"frames:   ok={fr['ok']}/{fr['total']} ({fr['ok_frac']:.4%})  "
        f"all_invalid={fr['all_invalid']}  high_corrupt(≥1%)={fr['high_corrupt_ge_1pct']}"
    )
    if worst:
        print(f"\nWorst {len(worst)} episodes:")
        for w in worst:
            print(
                f"  ep={w['episode_index']:6d}  {w['label']:16s}  "
                f"valid={w['valid_frac'] or 0:.4f}  corrupt={w['corrupted_frac'] or 0:.4f}  "
                f"invalid={w['invalid_frac'] or 0:.4f}  "
                f"f16_risk={w['float16_overflow_risk']}  "
                f"range=[{w['finite_min']}, {w['finite_max']}]"
                + (f"  ERR={w['error']}" if w.get("error") else "")
            )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
