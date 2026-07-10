"""Upload LFM4VLA checkpoints (+ their run config) to a HuggingFace model repo.

Local layout:
    runs/checkpoints/<date>/<run_name>/<file>.ckpt
    runs/logs/<date>/<run_name>/<run_name>-config.json

Hub layout (mirrors the date folder):
    <date>/<run_name>/<file>.ckpt
    <date>/<run_name>/<run_name>-config.json

Usage:
    huggingface-cli login            # or export HF_TOKEN=...

    # Upload every run + checkpoint under a date folder:
    python scripts/upload_checkpoints_hf.py \
        --repo-id <user>/lfm4vla-checkpoints \
        --date-dir runs/checkpoints/2026-07-02

    # Shorthand (same thing):
    python scripts/upload_checkpoints_hf.py \
        --repo-id <user>/lfm4vla-checkpoints \
        --date 2026-07-02

    # Single run or individual checkpoints still work:
    python scripts/upload_checkpoints_hf.py --repo-id <user>/repo \
        --run-dir runs/checkpoints/2026-07-02/lfm450m_libero10-...
    python scripts/upload_checkpoints_hf.py --repo-id <user>/repo \
        --ckpt runs/checkpoints/2026-07-02/<run>/last.ckpt
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from huggingface_hub import HfApi


def resolve_date_dir(date: str | None, date_dir: str | None, checkpoint_root: str) -> Path | None:
    if date_dir:
        path = Path(date_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    if date:
        return Path(checkpoint_root) / date
    return None


def collect_ckpts(
    *,
    ckpt_paths: list[str],
    run_dir: str | None,
    date_dir: Path | None,
) -> list[Path]:
    ckpts: list[Path] = [Path(p) for p in ckpt_paths]

    if run_dir:
        ckpts += [Path(p) for p in sorted(glob.glob(os.path.join(run_dir, "*.ckpt")))]

    if date_dir is not None:
        if not date_dir.is_dir():
            raise FileNotFoundError(f"date dir not found: {date_dir}")
        ckpts += [Path(p) for p in sorted(date_dir.glob("*/*.ckpt"))]

    # De-dupe while preserving order.
    return list(dict.fromkeys(ckpts))


def find_config(ckpt: Path, log_root: str) -> Path | None:
    """Locate <run_name>-config.json for a checkpoint via the parallel logs tree."""
    run_name = ckpt.parent.name
    date_dir = ckpt.parent.parent.name
    candidate = Path(log_root) / date_dir / run_name / f"{run_name}-config.json"
    if candidate.is_file():
        return candidate
    matches = glob.glob(str(Path(log_root) / "**" / f"{run_name}-config.json"), recursive=True)
    return Path(matches[0]) if matches else None


def hub_prefix(ckpt: Path) -> str:
    """Repo subfolder: <date>/<run_name>/"""
    return f"{ckpt.parent.parent.name}/{ckpt.parent.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/lfm4vla-checkpoints")
    ap.add_argument("--ckpt", action="append", default=[], help="path to a .ckpt (repeatable)")
    ap.add_argument("--run-dir", default=None, help="upload every *.ckpt in one run directory")
    ap.add_argument(
        "--date-dir",
        default=None,
        help="upload every run under a date folder, e.g. runs/checkpoints/2026-07-02",
    )
    ap.add_argument(
        "--date",
        default=None,
        help="shorthand for --date-dir runs/checkpoints/<date>",
    )
    ap.add_argument(
        "--checkpoint-root",
        default="runs/checkpoints",
        help="root used with --date (default: runs/checkpoints)",
    )
    ap.add_argument("--log-root", default="runs/logs", help="root of the logs tree with configs")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or use login)")
    args = ap.parse_args()

    date_dir = resolve_date_dir(args.date, args.date_dir, args.checkpoint_root)
    ckpts = collect_ckpts(ckpt_paths=args.ckpt, run_dir=args.run_dir, date_dir=date_dir)
    if not ckpts:
        ap.error("no checkpoints found; use --date/--date-dir, --run-dir, and/or --ckpt")

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    uploaded_configs: set[str] = set()
    print(f"Uploading {len(ckpts)} checkpoint(s) to {args.repo_id}")

    for ckpt in ckpts:
        if not ckpt.is_file():
            print(f"[skip] not found: {ckpt}")
            continue

        prefix = hub_prefix(ckpt)
        ckpt_dest = f"{prefix}/{ckpt.name}"
        print(f"[upload] {ckpt}  ->  {args.repo_id}/{ckpt_dest}")
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=ckpt_dest,
            repo_id=args.repo_id,
            repo_type="model",
        )

        if prefix in uploaded_configs:
            continue

        cfg = find_config(ckpt, args.log_root)
        if cfg is None:
            print(f"  [warn] no config found for {prefix} under {args.log_root}")
            continue

        cfg_dest = f"{prefix}/{cfg.name}"
        print(f"  [upload] {cfg}  ->  {args.repo_id}/{cfg_dest}")
        api.upload_file(
            path_or_fileobj=str(cfg),
            path_in_repo=cfg_dest,
            repo_id=args.repo_id,
            repo_type="model",
        )
        uploaded_configs.add(prefix)

    print("Done:", f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
