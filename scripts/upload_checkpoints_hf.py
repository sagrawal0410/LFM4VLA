"""Upload LFM4VLA checkpoints and/or the full runs/logs tree to HuggingFace.

Local layout (cluster):
    runs/checkpoints/<date>/<run_name>/<file>.ckpt
    runs/logs/<date>/<run_name>/<run_name>-config.json  (+ wandb/lightning artifacts)

Hub layout (checkpoints):
    checkpoints/<date>/<run_name>/<file>.ckpt

Hub layout (run configs only — default for --logs-date):
    logs/<date>/<run_name>/<run_name>-config.json

Use --logs-full-tree to upload everything under runs/logs/ (wandb artifacts, etc.).
SLURM output_*.out/*.err in repo root are never uploaded (not under runs/logs).

Usage:
    huggingface-cli login            # or export HF_TOKEN=...

    # Checkpoints + entire logs tree for one date:
    python scripts/upload_checkpoints_hf.py \
        --repo-id <user>/lfm4vla-checkpoints \
        --date 2026-07-02 \
        --logs-date 2026-07-02

    # Logs only (exact runs/logs structure):
    python scripts/upload_checkpoints_hf.py \
        --repo-id <user>/lfm4vla-checkpoints \
        --logs-only --logs-date 2026-07-02

    # Entire runs/logs tree:
    python scripts/upload_checkpoints_hf.py \
        --repo-id <user>/lfm4vla-checkpoints \
        --logs-only --logs-dir runs/logs
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from huggingface_hub import HfApi


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def resolve_date_dir(date: str | None, date_dir: str | None, checkpoint_root: str) -> Path | None:
    if date_dir:
        return resolve_path(date_dir)
    if date:
        return resolve_path(checkpoint_root) / date
    return None


def resolve_logs_dir(
    logs_date: str | None,
    logs_dir: str | None,
    log_root: str,
) -> Path | None:
    if logs_dir:
        return resolve_path(logs_dir)
    if logs_date:
        return resolve_path(log_root) / logs_date
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

    return list(dict.fromkeys(ckpts))


def run_key(ckpt: Path) -> tuple[str, str]:
    return ckpt.parent.parent.name, ckpt.parent.name


def find_config(ckpt: Path, log_root: str) -> Path | None:
    date_dir, run_name = run_key(ckpt)
    candidate = Path(log_root) / date_dir / run_name / f"{run_name}-config.json"
    if candidate.is_file():
        return candidate
    matches = glob.glob(str(Path(log_root) / "**" / f"{run_name}-config.json"), recursive=True)
    return Path(matches[0]) if matches else None


def hub_ckpt_path(ckpt: Path) -> str:
    date_dir, run_name = run_key(ckpt)
    return f"checkpoints/{date_dir}/{run_name}/{ckpt.name}"


def hub_config_path(ckpt: Path, cfg: Path, separate: bool) -> str:
    date_dir, run_name = run_key(ckpt)
    if separate:
        return f"configs/{date_dir}/{run_name}/{cfg.name}"
    return f"checkpoints/{date_dir}/{run_name}/{cfg.name}"


def hub_logs_prefix(local_logs: Path, log_root: Path) -> str:
    """Map local logs path to hub prefix under logs/."""
    try:
        rel = local_logs.resolve().relative_to(log_root.resolve())
    except ValueError:
        rel = local_logs.name
    return f"logs/{rel}".rstrip("/")


def upload_logs_tree(
    api: HfApi,
    repo_id: str,
    local_logs: Path,
    log_root: Path,
    *,
    configs_only: bool,
) -> None:
    if not local_logs.is_dir():
        raise FileNotFoundError(f"logs dir not found: {local_logs}")

    dest_prefix = hub_logs_prefix(local_logs, log_root)

    if configs_only:
        configs = sorted(local_logs.glob("*/*-config.json"))
        if not configs:
            print(f"[warn] no *-config.json under {local_logs}")
            return
        print(f"[upload] {len(configs)} run config(s) from {local_logs}")
        for cfg in configs:
            rel = cfg.relative_to(local_logs)
            dest = f"{dest_prefix}/{rel}"
            print(f"  {cfg}  ->  {repo_id}/{dest}")
            api.upload_file(
                path_or_fileobj=str(cfg),
                path_in_repo=dest,
                repo_id=repo_id,
                repo_type="model",
            )
        return

    print(f"[upload folder] {local_logs}  ->  {repo_id}/{dest_prefix}/")
    api.upload_folder(
        folder_path=str(local_logs),
        path_in_repo=dest_prefix,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.out", "*.err", "wandb/**", "**/__pycache__/**"],
    )


def upload_checkpoints(
    api: HfApi,
    repo_id: str,
    ckpts: list[Path],
    log_root: str,
    *,
    separate_configs: bool,
    pair_configs: bool,
) -> None:
    uploaded_configs: set[tuple[str, str]] = set()
    layout = "checkpoints/ + configs/" if separate_configs else "checkpoints/ (configs co-located)"
    print(f"Uploading {len(ckpts)} checkpoint(s) ({layout})")

    for ckpt in ckpts:
        if not ckpt.is_file():
            print(f"[skip] not found: {ckpt}")
            continue

        ckpt_dest = hub_ckpt_path(ckpt)
        print(f"[upload] {ckpt}  ->  {repo_id}/{ckpt_dest}")
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=ckpt_dest,
            repo_id=repo_id,
            repo_type="model",
        )

        if not pair_configs:
            continue

        key = run_key(ckpt)
        if key in uploaded_configs:
            continue

        cfg = find_config(ckpt, log_root)
        if cfg is None:
            print(f"  [warn] no config for {key[1]} under {log_root}/{key[0]}/")
            continue

        cfg_dest = hub_config_path(ckpt, cfg, separate=separate_configs)
        print(f"  [upload] {cfg}  ->  {repo_id}/{cfg_dest}")
        api.upload_file(
            path_or_fileobj=str(cfg),
            path_in_repo=cfg_dest,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded_configs.add(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/lfm4vla-checkpoints")
    ap.add_argument("--ckpt", action="append", default=[], help="path to a .ckpt (repeatable)")
    ap.add_argument("--run-dir", default=None, help="upload every *.ckpt in one run directory")
    ap.add_argument("--date-dir", default=None, help="e.g. runs/checkpoints/2026-07-02")
    ap.add_argument("--date", default=None, help="shorthand: runs/checkpoints/<date>")
    ap.add_argument("--checkpoint-root", default="runs/checkpoints")
    ap.add_argument("--log-root", default="runs/logs", help="local runs/logs root")

    ap.add_argument(
        "--logs-date",
        default=None,
        help="upload run *-config.json files from runs/logs/<date>/ (default: configs only)",
    )
    ap.add_argument(
        "--logs-dir",
        default=None,
        help="upload from an arbitrary runs/logs subfolder",
    )
    ap.add_argument(
        "--logs-full-tree",
        action="store_true",
        help="upload entire logs folder (not just *-config.json); still skips *.out/*.err",
    )
    ap.add_argument(
        "--logs-only",
        action="store_true",
        help="skip checkpoints; only upload the logs tree (--logs-date or --logs-dir required)",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="when pairing configs with checkpoints, put them under checkpoints/<date>/<run>/",
    )
    ap.add_argument(
        "--no-pair-configs",
        action="store_true",
        help="skip per-checkpoint config uploads (use with --logs-date to upload configs via logs tree)",
    )
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    log_root = resolve_path(args.log_root)
    logs_target = resolve_logs_dir(args.logs_date, args.logs_dir, str(log_root))
    date_dir = resolve_date_dir(args.date, args.date_dir, args.checkpoint_root)
    ckpts = collect_ckpts(ckpt_paths=args.ckpt, run_dir=args.run_dir, date_dir=date_dir)

    if args.logs_only and logs_target is None:
        ap.error("--logs-only requires --logs-date or --logs-dir")
    if not args.logs_only and not ckpts and logs_target is None:
        ap.error("nothing to upload; use --date, --logs-date, --logs-dir, etc.")

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    if not args.logs_only and ckpts:
        upload_checkpoints(
            api,
            args.repo_id,
            ckpts,
            str(log_root),
            separate_configs=not args.flat,
            pair_configs=not args.no_pair_configs,
        )

    if logs_target is not None:
        upload_logs_tree(
            api,
            args.repo_id,
            logs_target,
            log_root,
            configs_only=not args.logs_full_tree,
        )

    print("Done:", f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
