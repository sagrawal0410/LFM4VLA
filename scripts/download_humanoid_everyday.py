"""Download the Humanoid Everyday subset needed for LFM4VLA fine-tuning.

Fetches, from the LeRobot-format HF dataset (default ``USC-GVL/humanoid-everyday``):
  * ``meta/`` (info.json, tasks.jsonl, episodes.jsonl)
  * per selected episode: a SLIM parquet (low-dim columns only — the full parquet
    bundles depth/LiDAR and is ~100 MB/episode) + the egocentric MP4

Episodes are selected by skill keywords in the task name (pick/place, push/pull,
stack/align, pour by default) and robot type (G1 by default, 28-D actions).

Preferred env (NumPy 1.26 pin; leaves ``lfm4vla`` free for NumPy 2.x work):
    bash scripts/install_humanoid_everyday_env.sh
    conda activate lfm4vla-he

Usage (cluster login node or workstation):
    python scripts/download_humanoid_everyday.py \
        --output_dir /home/teams/research/robotics/humanoid_everyday

Resume-safe: already-downloaded episodes are skipped.
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

from data.humanoid_everyday_dataset import SKILL_KEYWORDS, task_matches_skills

# Low-dim columns worth keeping in the slim parquet (intersected with the actual
# schema per episode; depth/lidar/tactile blobs are dropped).
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


def slim_parquet(repo_id: str, ep_idx: int, out: Path) -> None:
    """Write a slim local parquet, fetching only low-dim columns when possible."""
    import pyarrow.parquet as pq

    dest = out / parquet_rel(ep_idx)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.parquet")

    try:
        # Column projection over HTTP: only the selected columns' bytes are fetched.
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        remote = f"datasets/{repo_id}/{parquet_rel(ep_idx)}"
        schema_names = pq.read_schema(remote, filesystem=fs).names
        cols = [c for c in SLIM_COLUMNS if c in schema_names]
        table = pq.read_table(remote, columns=cols, filesystem=fs)
    except Exception:
        # Fallback: download the full file, strip locally, delete the original.
        from huggingface_hub import hf_hub_download

        full = hf_hub_download(repo_id, parquet_rel(ep_idx), repo_type="dataset")
        schema_names = pq.read_schema(full).names
        cols = [c for c in SLIM_COLUMNS if c in schema_names]
        table = pq.read_table(full, columns=cols)

    pq.write_table(table, tmp)
    tmp.replace(dest)


def download_video(repo_id: str, ep_idx: int, out: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id, video_rel(ep_idx), repo_type="dataset", local_dir=str(out))


def fetch_episode(repo_id: str, ep_idx: int, out: Path) -> str:
    done_parquet = (out / parquet_rel(ep_idx)).is_file()
    done_video = (out / video_rel(ep_idx)).is_file()
    if done_parquet and done_video:
        return "skip"
    if not done_parquet:
        slim_parquet(repo_id, ep_idx, out)
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
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] meta files -> {out}/meta")
    download_meta(args.repo_id, out)

    episodes = select_episodes(out, args.skills, args.robot_type)
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"[2/3] selected {len(episodes)} episodes "
          f"(skills={args.skills}, robot={args.robot_type or 'all'})")

    print(f"[3/3] downloading slim parquet + egocentric MP4 with {args.workers} threads...")
    n_done = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_episode, args.repo_id, ep, out): ep for ep in episodes}
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
    print("Then:")
    print("  sbatch scripts/train_lfm_he_450m.sbatch")
    print("  sbatch scripts/train_lfm_he_1.6b.sbatch")


if __name__ == "__main__":
    main()
