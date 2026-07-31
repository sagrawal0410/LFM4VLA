"""Extract HE egocentric depth from parquet into per-episode ``.npy`` memmaps.

Why: random-access training reads one frame per sample. Loading the full
``observation.depth.egocentric`` column from a ~100 MB parquet on every cache miss
(via ``to_pylist``) makes HE depth training crawl on shared NFS. A contiguous
float16 ``.npy`` lets the dataset ``np.load(..., mmap_mode="r")`` and slice one
frame in microseconds.

Usage (after depth download):
    python scripts/extract_he_depth_arrays.py \\
        --data_root /home/teams/research/robotics/humanoid_everyday_depth

    # or: sbatch scripts/extract_he_depth_arrays.sbatch

Writes:
    <data_root>/depths/chunk-XXX/episode_XXXXXX.npy   # [T, H, W] float16
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


def _select_episodes(root: Path, skills, robot_type):
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
                selected.append(ep["episode_index"])
    return selected


def _extract_one(root_str: str, ep_idx: int, overwrite: bool) -> str:
    import pyarrow.parquet as pq

    root = Path(root_str)
    dest = root / depth_npy_rel(ep_idx)
    if dest.is_file() and not overwrite:
        return "skip"
    src = root / parquet_rel(ep_idx)
    if not src.is_file():
        raise FileNotFoundError(src)

    table = pq.read_table(src, columns=[DEPTH_COLUMN])
    frames = table.column(DEPTH_COLUMN).to_pylist()
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"ep {ep_idx}: unexpected depth shape {arr.shape}")

    # Sanitize before float16: values > 65504 become Inf and later min-max → NaN loss.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where((arr > 0.0) & (arr < 1.0e5), arr, 0.0).astype(np.float32)
    arr = np.clip(arr, 0.0, np.finfo(np.float16).max)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npy")
    np.save(tmp, arr.astype(np.float16))
    tmp.replace(dest)
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--skills", nargs="+", default=list(SKILL_KEYWORDS),
                    choices=list(SKILL_KEYWORDS))
    ap.add_argument("--robot_type", default="g1", choices=["g1", "h1", ""])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    episodes = _select_episodes(root, args.skills, args.robot_type)
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"Extracting depth npy for {len(episodes)} episodes under {root}/depths "
          f"(workers={args.workers})")

    n_done = n_skip = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_extract_one, str(root), ep, args.overwrite): ep
            for ep in episodes
        }
        for i, fut in enumerate(as_completed(futs), 1):
            ep = futs[fut]
            try:
                status = fut.result()
                n_done += status == "done"
                n_skip += status == "skip"
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  [fail] episode {ep}: {type(e).__name__}: {e}")
            if i % 50 == 0 or i == len(episodes):
                print(f"  {i}/{len(episodes)} (new={n_done} cached={n_skip} failed={n_fail})",
                      flush=True)

    print(f"\nDone. new={n_done} cached={n_skip} failed={n_fail}")
    if n_fail:
        sys.exit(1)
    print("Restart HE depth training; HumanoidEverydayDataset will mmap these files.")


if __name__ == "__main__":
    main()
