#!/usr/bin/env python3
"""Generate / list the Humanoid Everyday action-token × latent sweep grid.

Sweeps:
  num_action_tokens ∈ {1, 2, 4, 8, 16, 24, 28}   # distinct learned queries
  latent            ∈ {1, 2, 4, 8, 16}           # per-token repeat in the VLM

→ 7 × 5 = 35 runs. Total action tokens inserted = num_action_tokens * latent.

Same grid is reused for the RGB-only and depth+Q-Former sweeps:
  sbatch scripts/train_lfm_he_token_sweep.sbatch            # RGB
  sbatch scripts/train_lfm_he_depth_token_sweep.sbatch      # depth, q_former=32

Usage:
  # Print the grid (index -> tokens, latent, total)
  python scripts/sweep_he_tokens.py --list

  # Write 35 JSON configs under configs/sweeps/he_tokens_<model>/
  python scripts/sweep_he_tokens.py --write --base_config configs/lfm2.5-vl-450m-humanoid-everyday.json

  # Emit a bash script that submits all 35 as separate sbatch jobs
  python scripts/sweep_he_tokens.py --emit_sbatch > /tmp/submit_he_sweep.sh

  # Or use the SLURM array launcher (preferred):
  sbatch scripts/train_lfm_he_token_sweep.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NUM_ACTION_TOKENS = [1, 2, 4, 8, 16, 24, 28]
LATENTS = [1, 2, 4, 8, 16]


def grid():
    """Return list of (index, num_action_tokens, latent, total_tokens)."""
    out = []
    i = 0
    for n in NUM_ACTION_TOKENS:
        for lat in LATENTS:
            out.append((i, n, lat, n * lat))
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
    model = base.get("model", "lfm")
    for idx, n_tok, lat, total in grid():
        cfg = json.loads(json.dumps(base))  # deep copy via JSON
        cfg["task_name"] = f"{base.get('task_name', 'he')}_tok{n_tok}_lat{lat}"
        cfg["wandb_project"] = "lfm4vla_he"
        cfg["output_root"] = "/home/teams/research/robotics/checkpoints"
        cfg["log_root"] = "/home/teams/research/robotics/logs"
        cfg["cache_root"] = "/home/teams/research/robotics/cache"
        cfg.setdefault("act_head", {})
        cfg["act_head"]["type"] = "FCContinuousDecoder"
        cfg["act_head"]["action_dim"] = 28
        cfg["act_head"]["down_sample"] = "none"
        cfg["act_head"]["num_action_tokens"] = n_tok
        cfg["act_head"]["latent"] = lat
        # Drop legacy gripper-only head knobs if present.
        cfg["act_head"].pop("gripper", None)
        path = out_dir / f"{idx:02d}_tok{n_tok}_lat{lat}_total{total}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(f"wrote {path}  (tokens={n_tok} latent={lat} total={total})")
    print(f"\n{len(grid())} configs under {out_dir} (model={model})")


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
        task = f"he_tok{n_tok}_lat{lat}"
        print(
            f"sbatch --export=ALL,"
            f"CONFIG=\"$BASE_CONFIG\","
            f"NUM_ACTION_TOKENS={n_tok},"
            f"LATENT={lat},"
            f"TASK_NAME={task} "
            f"scripts/train_lfm_he_token_sweep.sbatch  # idx={idx} total_tokens={total}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="Print the 35-run grid")
    ap.add_argument("--write", action="store_true", help="Write per-combo JSON configs")
    ap.add_argument("--emit_sbatch", action="store_true",
                    help="Print a bash script that submits 35 sbatch jobs")
    ap.add_argument("--base_config",
                    default="configs/lfm2.5-vl-450m-humanoid-everyday.json")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write configs (default: configs/sweeps/he_tokens_<stem>)")
    ap.add_argument("--index", type=int, default=None,
                    help="Print a single combo by 0-based index (for array jobs)")
    ap.add_argument("--conda_env", default="lfm4vla")
    args = ap.parse_args()

    if args.index is not None:
        idx, n_tok, lat, total = combo_at(args.index)
        # Machine-readable for the sbatch wrapper.
        print(f"INDEX={idx}")
        print(f"NUM_ACTION_TOKENS={n_tok}")
        print(f"LATENT={lat}")
        print(f"TOTAL_TOKENS={total}")
        print(f"TASK_NAME=he_tok{n_tok}_lat{lat}")
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
        out = Path(args.out_dir) if args.out_dir else Path(
            f"configs/sweeps/he_tokens_{base.stem}"
        )
        write_configs(base, out)

    if args.emit_sbatch:
        emit_sbatch(args.base_config, args.conda_env)


if __name__ == "__main__":
    main()
