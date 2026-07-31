"""Download the Humanoid Everyday subset needed for LFM4VLA fine-tuning.

Fetches, from the LeRobot-format HF dataset (default ``USC-GVL/humanoid-everyday``):
  * ``meta/`` (info.json, tasks.jsonl, episodes.jsonl)
  * per selected episode: a parquet (low-dim columns; optionally egocentric depth)
    + the egocentric RGB MP4

Episodes are selected by skill keywords in the task name (pick/place, push/pull,
stack/align, pour by default) and robot type (G1 by default, 28-D actions).

Preferred env (NumPy 1.26 pin; leaves ``lfm4vla`` free for NumPy 2.x work):
    bash scripts/install_humanoid_everyday_env.sh
    conda activate lfm4vla-he

RGB-only (slim, ~small parquet + MP4):
    python scripts/download_humanoid_everyday.py \\
        --output_dir /home/teams/research/robotics/humanoid_everyday

RGB + depth (separate tree; parquet keeps ``observation.depth.egocentric``):
    python scripts/download_humanoid_everyday.py \\
        --output_dir /home/teams/research/robotics/humanoid_everyday_depth \\
        --include_depth
    # Then extract memmap sidecars (required for usable training speed):
    python scripts/extract_he_depth_arrays.py \\
        --data_root /home/teams/research/robotics/humanoid_everyday_depth

Resume-safe: already-downloaded episodes are skipped (depth downloads re-fetch a
parquet that is missing the depth column).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.humanoid_everyday_dataset import DEPTH_COLUMN, SKILL_KEYWORDS, task_matches_skills

# Low-dim columns worth keeping in the slim parquet (intersected with the actual
# schema per episode). Depth is opt-in via --include_depth; lidar/tactile stay out.
SLIM_COLUMNS = [
    "action",
    "observation.arm_joints",
    "observation.hand_joints",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
]


def parquet_rel(ep_idx: int) -> str:
    return f"data/chunk-{ep_idx // 1000:03d}/episode_{ep_idx:06d}.parquet"


def video_rel(ep_idx: int) -> str:
    return f"videos/chunk-{ep_idx // 1000:03d}/egocentric/episode_{ep_idx:06d}.mp4"


def download_meta(repo_id: str, out: Path) -> None:
    from huggingface_hub import hf_hub_download

    for rel in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        if not (out / rel).is_file():
            hf_hub_download(repo_id, rel, repo_type="dataset", local_dir=str(out))
            print(f"  meta: {rel}")


def select_episodes(out: Path, skills, robot_type):
    tasks = {}
    with open(out / "meta/tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            tasks[t["task_index"]] = t
    selected = []
    with open(out / "meta/episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            if robot_type and ep.get("robot_type") != robot_type:
                continue
            if task_matches_skills(tasks[ep["tasks"][0]]["task"], skills):
                selected.append(ep["episode_index"])
    return selected


def _parquet_has_column(path: Path, column: str) -> bool:
    import pyarrow.parquet as pq

    try:
        return column in pq.read_schema(path).names
    except Exception:  # noqa: BLE001
        return False


def write_parquet(repo_id: str, ep_idx: int, out: Path, columns: list[str]) -> None:
    """Write a local parquet with the requested columns (HTTP column projection when possible)."""
    import pyarrow.parquet as pq

    dest = out / parquet_rel(ep_idx)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.parquet")

    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        remote = f"datasets/{repo_id}/{parquet_rel(ep_idx)}"
        schema_names = pq.read_schema(remote, filesystem=fs).names
        cols = [c for c in columns if c in schema_names]
        if not cols:
            raise RuntimeError(f"none of {columns} found in remote schema {schema_names[:20]}...")
        table = pq.read_table(remote, columns=cols, filesystem=fs)
    except Exception:
        # Fallback: download the full file, strip locally.
        from huggingface_hub import hf_hub_download

        full = hf_hub_download(repo_id, parquet_rel(ep_idx), repo_type="dataset")
        schema_names = pq.read_schema(full).names
        cols = [c for c in columns if c in schema_names]
        if not cols:
            raise RuntimeError(f"none of {columns} found in schema {schema_names[:20]}...")
        table = pq.read_table(full, columns=cols)

    pq.write_table(table, tmp)
    tmp.replace(dest)


def download_video(repo_id: str, ep_idx: int, out: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id, video_rel(ep_idx), repo_type="dataset", local_dir=str(out))


def fetch_episode(repo_id: str, ep_idx: int, out: Path, columns: list[str],
                  require_depth: bool) -> str:
    parquet_path = out / parquet_rel(ep_idx)
    done_video = (out / video_rel(ep_idx)).is_file()
    done_parquet = parquet_path.is_file()
    if require_depth and done_parquet and not _parquet_has_column(parquet_path, DEPTH_COLUMN):
        done_parquet = False  # force re-fetch with depth column

    if done_parquet and done_video:
        return "skip"
    if not done_parquet:
        write_parquet(repo_id, ep_idx, out, columns)
        if require_depth and not _parquet_has_column(parquet_path, DEPTH_COLUMN):
            raise RuntimeError(
                f"episode {ep_idx}: requested depth but column '{DEPTH_COLUMN}' "
                "missing from HF parquet schema"
            )
    if not done_video:
        download_video(repo_id, ep_idx, out)
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output_dir", required=True, help="Local dataset root (data_root_dir in configs)")
    ap.add_argument("--repo_id", default="USC-GVL/humanoid-everyday")
    ap.add_argument("--skills", nargs="+", default=list(SKILL_KEYWORDS),
                    choices=list(SKILL_KEYWORDS), help="Skill buckets to fetch")
    ap.add_argument("--robot_type", default="g1", choices=["g1", "h1", ""],
                    help="Robot filter ('' = all; note G1=28-D vs H1=26-D actions)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel download threads")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap episode count (e.g. small smoke-test download)")
    ap.add_argument(
        "--include_depth",
        action="store_true",
        help=(
            f"Keep '{DEPTH_COLUMN}' in each parquet (~100 MB/episode). "
            "Use a separate --output_dir from the RGB-only tree "
            "(e.g. .../humanoid_everyday_depth)."
        ),
    )
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    columns = list(SLIM_COLUMNS)
    if args.include_depth:
        columns.append(DEPTH_COLUMN)

    print(f"[1/3] meta files -> {out}/meta")
    download_meta(args.repo_id, out)

    # Record what this tree contains so train configs / humans can tell RGB vs depth.
    marker = {
        "include_depth": bool(args.include_depth),
        "depth_column": DEPTH_COLUMN if args.include_depth else None,
        "parquet_columns": columns,
        "repo_id": args.repo_id,
        "skills": list(args.skills),
        "robot_type": args.robot_type,
    }
    marker_path = out / "meta" / "download_manifest.json"
    with open(marker_path, "w") as f:
        json.dump(marker, f, indent=2)
        f.write("\n")

    episodes = select_episodes(out, args.skills, args.robot_type)
    if args.limit:
        episodes = episodes[: args.limit]
    mode = "RGB+depth" if args.include_depth else "RGB-only (slim)"
    print(f"[2/3] selected {len(episodes)} episodes "
          f"(skills={args.skills}, robot={args.robot_type or 'all'}, mode={mode})")
    if args.include_depth:
        print("  NOTE: depth parquets are large (~100 MB/episode). Prefer a dedicated "
              "output_dir (e.g. humanoid_everyday_depth).")

    print(f"[3/3] downloading parquet + egocentric MP4 with {args.workers} threads...")
    n_done = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_episode, args.repo_id, ep, out, columns, args.include_depth
            ): ep
            for ep in episodes
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ep = futures[fut]
            try:
                status = fut.result()
                n_done += status == "done"
                n_skip += status == "skip"
            except Exception as e:  # noqa: BLE001 — keep going, report at the end
                n_fail += 1
                print(f"  [fail] episode {ep}: {type(e).__name__}: {e}")
            if i % 50 == 0 or i == len(episodes):
                print(f"  {i}/{len(episodes)} (new={n_done} cached={n_skip} failed={n_fail})",
                      flush=True)

    # hf_hub_download(local_dir=...) leaves a .cache dir behind; tidy it up.
    cache_dir = out / ".cache"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)

    print(f"\nDone. new={n_done} cached={n_skip} failed={n_fail}")
    if n_fail:
        print("Re-run the same command to retry failed episodes (resume-safe).")
    print(f"\nPoint train_dataset.data_root_dir at: {out}")
    if args.include_depth:
        print("Then extract depth .npy (do this before training — parquet random access is tiny):")
        print(f"  python scripts/extract_he_depth_arrays.py --data_root {out}")
        print("  # or: sbatch scripts/extract_he_depth_arrays.sbatch")
        print("Then (depth + Q-Former token sweep):")
        print("  sbatch scripts/train_lfm_he_depth_token_sweep.sbatch")
    else:
        print("Then:")
        print("  sbatch scripts/train_lfm_he_450m.sbatch")
        print("  sbatch scripts/train_lfm_he_token_sweep.sbatch")


if __name__ == "__main__":
    main()
