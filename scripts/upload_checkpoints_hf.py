"""Upload LFM4VLA checkpoints (+ their run config) to a HuggingFace model repo.

Each checkpoint lives at:
    runs/checkpoints/<date>/<run_name>/<file>.ckpt
and its config at:
    runs/logs/<date>/<run_name>/<run_name>-config.json

This script uploads each checkpoint into its own subfolder on the Hub, alongside
the matching config, so every checkpoint is self-contained.

Usage:
    huggingface-cli login            # or export HF_TOKEN=...
    python scripts/upload_checkpoints_hf.py \
        --repo-id <user>/lfm4vla-libero \
        --ckpt runs/checkpoints/2026-07-09/lfm450m_libero10-.../last.ckpt \
        --ckpt runs/checkpoints/2026-07-09/lfm1.6b_libero10-.../step-....ckpt

    # or grab every *.ckpt under a run directory:
    python scripts/upload_checkpoints_hf.py --repo-id <user>/lfm4vla-libero \
        --run-dir runs/checkpoints/2026-07-09/lfm450m_libero10-...
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from huggingface_hub import HfApi


def find_config(ckpt: Path, log_root: str) -> Path | None:
    """Locate <run_name>-config.json for a checkpoint via the parallel logs tree."""
    run_name = ckpt.parent.name
    date_dir = ckpt.parent.parent.name
    candidate = Path(log_root) / date_dir / run_name / f"{run_name}-config.json"
    if candidate.is_file():
        return candidate
    # Fallback: search anywhere under log_root for the run's config.
    matches = glob.glob(str(Path(log_root) / "**" / f"{run_name}-config.json"), recursive=True)
    return Path(matches[0]) if matches else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/lfm4vla-libero")
    ap.add_argument("--ckpt", action="append", default=[], help="path to a .ckpt (repeatable)")
    ap.add_argument("--run-dir", default=None, help="upload every *.ckpt in this run directory")
    ap.add_argument("--log-root", default="runs/logs", help="root of the logs tree with configs")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or use login)")
    args = ap.parse_args()

    ckpts = [Path(c) for c in args.ckpt]
    if args.run_dir:
        ckpts += [Path(p) for p in sorted(glob.glob(os.path.join(args.run_dir, "*.ckpt")))]
    ckpts = [c for c in dict.fromkeys(ckpts)]  # de-dupe, keep order
    if not ckpts:
        ap.error("no checkpoints given; use --ckpt and/or --run-dir")

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    for ckpt in ckpts:
        if not ckpt.is_file():
            print(f"[skip] not found: {ckpt}")
            continue
        run_name = ckpt.parent.name
        subdir = f"{run_name}/{ckpt.name}"
        print(f"[upload] {ckpt}  ->  {args.repo_id}/{subdir}")
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=subdir,
            repo_id=args.repo_id,
            repo_type="model",
        )
        cfg = find_config(ckpt, args.log_root)
        if cfg is None:
            print(f"  [warn] no config found for run {run_name} under {args.log_root}")
            continue
        print(f"  [upload] {cfg}  ->  {args.repo_id}/{run_name}/{cfg.name}")
        api.upload_file(
            path_or_fileobj=str(cfg),
            path_in_repo=f"{run_name}/{cfg.name}",
            repo_id=args.repo_id,
            repo_type="model",
        )

    print("Done:", f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
