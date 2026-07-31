#!/usr/bin/env python3
"""Generate / list the LIBERO depth-conditioning latent × QFormer-query sweep.

Sweeps:
  latent              ∈ {1, 4}          # action-token repeats (FCDecoder)
  depth.qformer.num_queries ∈ {8, 16, 32, 64}   # fused tokens inserted into LLM

→ 2 × 4 = 8 runs.

Usage:
  # Print the grid
  python scripts/sweep_libero_depth.py --list

  # Write 8 JSON configs under configs/sweeps/libero_depth_<stem>/
  python scripts/sweep_libero_depth.py --write \\
      --base_config configs/lfm2.5-vl-450m-libero-depth-latent1.json

  # Emit a bash script that submits 8 separate sbatch jobs
  python scripts/sweep_libero_depth.py --emit_sbatch > /tmp/submit_libero_depth_sweep.sh

  # Preferred: SLURM array launcher
  sbatch scripts/train_lfm_libero_depth_sweep.sbatch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LATENTS = [1, 4]
NUM_QUERIES = [8, 16, 32, 64]


def grid():
    """Return list of (index, latent, num_queries)."""
    out = []
    i = 0
    for lat in LATENTS:
        for nq in NUM_QUERIES:
            out.append((i, lat, nq))
            i += 1
    return out


def combo_at(index: int):
    rows = grid()
    if index < 0 or index >= len(rows):
        raise IndexError(f"sweep index {index} out of range [0, {len(rows) - 1}]")
    return rows[index]


def write_configs(base_config: Path, out_dir: Path) -> None:
    with open(base_config) as f:
        base = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, lat, nq in grid():
        cfg = json.loads(json.dumps(base))  # deep copy
        cfg["use_depth"] = True
        cfg["output_root"] = "/home/teams/research/robotics/checkpoints"
        cfg["log_root"] = "/home/teams/research/robotics/logs"
        cfg["cache_root"] = "/home/teams/research/robotics/cache"
        cfg.setdefault("depth", {}).setdefault("qformer", {})
        cfg["depth"]["qformer"]["num_queries"] = nq
        cfg.setdefault("act_head", {})
        cfg["act_head"]["latent"] = lat
        cfg["task_name"] = f"{base.get('task_name', 'libero_depth')}_lat{lat}_q{nq}"
        path = out_dir / f"{idx:02d}_lat{lat}_q{nq}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(f"wrote {path}  (latent={lat} num_queries={nq})")
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
    for idx, lat, nq in grid():
        task = f"libero_depth_lat{lat}_q{nq}"
        print(
            f"sbatch --export=ALL,"
            f"CONFIG=\"$BASE_CONFIG\","
            f"LATENT={lat},"
            f"DEPTH_NUM_QUERIES={nq},"
            f"TASK_NAME={task} "
            f"scripts/train_lfm_libero_depth_sweep.sbatch  "
            f"# idx={idx}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="Print the 8-run grid")
    ap.add_argument("--write", action="store_true", help="Write per-combo JSON configs")
    ap.add_argument(
        "--emit_sbatch",
        action="store_true",
        help="Print a bash script that submits 8 sbatch jobs",
    )
    ap.add_argument(
        "--base_config",
        default="configs/lfm2.5-vl-450m-libero-depth-latent1.json",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Where to write configs (default: configs/sweeps/libero_depth_<stem>)",
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
        idx, lat, nq = combo_at(args.index)
        print(f"INDEX={idx}")
        print(f"LATENT={lat}")
        print(f"DEPTH_NUM_QUERIES={nq}")
        print(f"TASK_NAME=libero_depth_lat{lat}_q{nq}")
        return

    if args.list or not (args.write or args.emit_sbatch):
        print(f"{'idx':>3}  {'latent':>6}  {'queries':>7}")
        for idx, lat, nq in grid():
            print(f"{idx:3d}  {lat:6d}  {nq:7d}")
        print(f"\n{len(grid())} combinations")
        if not (args.write or args.emit_sbatch):
            return

    if args.write:
        base = Path(args.base_config)
        out = (
            Path(args.out_dir)
            if args.out_dir
            else Path(f"configs/sweeps/libero_depth_{base.stem}")
        )
        write_configs(base, out)

    if args.emit_sbatch:
        emit_sbatch(args.base_config, args.conda_env)


if __name__ == "__main__":
    main()
