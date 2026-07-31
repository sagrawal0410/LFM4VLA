#!/usr/bin/env python3
"""Generate / list the plain LIBERO action-token sweep (no depth).

Sweeps:
  num_action_tokens ∈ {1, 2, 4, 6, 8, 16}   # distinct learned action queries
  latent            = 1                     # fixed (not swept)

→ 6 runs. Setting ``num_action_tokens`` opts into the multi-token layout used
by HE (``models.action_token_layout``) while keeping FCDecoder + LIBERO RGB
data (no depth conditioning).

Usage:
  python scripts/sweep_libero_tokens.py --list
  python scripts/sweep_libero_tokens.py --write \\
      --base_config configs/lfm2.5-vl-450m-libero.json
  sbatch scripts/train_lfm_libero_token_sweep.sbatch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NUM_ACTION_TOKENS = [1, 2, 4, 6, 8, 16]
LATENT = 1  # fixed; this sweep varies learned queries only


def grid():
    """Return list of (index, num_action_tokens, latent, total_tokens)."""
    return [
        (i, n, LATENT, n * LATENT) for i, n in enumerate(NUM_ACTION_TOKENS)
    ]


def combo_at(index: int):
    rows = grid()
    if index < 0 or index >= len(rows):
        raise IndexError(f"sweep index {index} out of range [0, {len(rows) - 1}]")
    return rows[index]


def write_configs(base_config: Path, out_dir: Path) -> None:
    with open(base_config) as f:
        base = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, n_tok, lat, total in grid():
        cfg = json.loads(json.dumps(base))  # deep copy
        cfg.pop("use_depth", None)
        cfg.pop("depth", None)
        cfg["output_root"] = "/home/teams/research/robotics/checkpoints"
        cfg["log_root"] = "/home/teams/research/robotics/logs"
        cfg["cache_root"] = "/home/teams/research/robotics/cache"
        cfg.setdefault("act_head", {})
        cfg["act_head"]["type"] = "FCDecoder"
        cfg["act_head"]["num_action_tokens"] = n_tok
        cfg["act_head"]["latent"] = lat
        cfg["task_name"] = f"{base.get('task_name', 'libero')}_tok{n_tok}"
        path = out_dir / f"{idx:02d}_tok{n_tok}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(f"wrote {path}  (tokens={n_tok} latent={lat} total={total})")
    print(f"\n{len(grid())} configs under {out_dir}")


def emit_sbatch(base_config: str, conda_env: str = "lfm4vla") -> None:
    root = "${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
    print("#!/bin/bash")
    print("set -euo pipefail")
    print(f'LFM4VLA_ROOT="{root}"')
    print(f'CONDA_ENV="{conda_env}"')
    print(f'BASE_CONFIG="{base_config}"')
    print('cd "$LFM4VLA_ROOT"')
    print('source "$(conda info --base)/etc/profile.d/conda.sh"')
    print('conda activate "$CONDA_ENV"')
    print()
    for idx, n_tok, lat, total in grid():
        task = f"libero_tok{n_tok}"
        print(
            f"sbatch --export=ALL,"
            f"CONFIG=\"$BASE_CONFIG\","
            f"NUM_ACTION_TOKENS={n_tok},"
            f"LATENT={lat},"
            f"TASK_NAME={task} "
            f"scripts/train_lfm_libero_token_sweep.sbatch  "
            f"# idx={idx} total_tokens={total}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="Print the 6-run grid")
    ap.add_argument("--write", action="store_true", help="Write per-combo JSON configs")
    ap.add_argument(
        "--emit_sbatch",
        action="store_true",
        help="Print a bash script that submits 6 sbatch jobs",
    )
    ap.add_argument(
        "--base_config",
        default="configs/lfm2.5-vl-450m-libero.json",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Where to write configs (default: configs/sweeps/libero_tokens_<stem>)",
    )
    ap.add_argument(
        "--index",
        type=int,
        default=None,
        help="Print a single combo by 0-based index (for array jobs)",
    )
    ap.add_argument("--conda_env", default="lfm4vla")
    args = ap.parse_args()

    if args.index is not None:
        idx, n_tok, lat, total = combo_at(args.index)
        print(f"INDEX={idx}")
        print(f"NUM_ACTION_TOKENS={n_tok}")
        print(f"LATENT={lat}")
        print(f"TOTAL_TOKENS={total}")
        print(f"TASK_NAME=libero_tok{n_tok}")
        return

    if args.list or not (args.write or args.emit_sbatch):
        print(f"{'idx':>3}  {'tokens':>6}  {'latent':>6}  {'total':>5}")
        for idx, n_tok, lat, total in grid():
            print(f"{idx:3d}  {n_tok:6d}  {lat:6d}  {total:5d}")
        print(f"\n{len(grid())} combinations")
        if not (args.write or args.emit_sbatch):
            return

    if args.write:
        base = Path(args.base_config)
        out = (
            Path(args.out_dir)
            if args.out_dir
            else Path(f"configs/sweeps/libero_tokens_{base.stem}")
        )
        write_configs(base, out)

    if args.emit_sbatch:
        emit_sbatch(args.base_config, args.conda_env)


if __name__ == "__main__":
    main()
