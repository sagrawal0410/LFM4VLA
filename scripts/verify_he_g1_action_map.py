#!/usr/bin/env python3
"""Compare HE parquet ``action`` vs arm/hand observation columns to lock 28-D order.

Prints per-dim correlation / L2 between ``action`` and candidate concatenations of
``observation.arm_joints`` + ``observation.hand_joints`` so we can confirm the
layout used by ``eval/humanoid/g1_joint_map.py``.

Usage:
  python scripts/verify_he_g1_action_map.py \
      --data_root_dir /home/teams/research/robotics/humanoid_everyday \
      --max_episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.humanoid_everyday_dataset import load_meta, parquet_rel  # noqa: E402
from eval.humanoid.g1_joint_map import POLICY_JOINTS  # noqa: E402


def _stack_col(table, name: str) -> np.ndarray:
    import pyarrow.parquet as pq  # noqa: F401 — already have table

    return np.stack(table.column(name).to_pylist()).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root_dir", required=True)
    ap.add_argument("--max_episodes", type=int, default=5)
    ap.add_argument("--robot_type", default="g1")
    args = ap.parse_args()

    root = Path(args.data_root_dir)
    tasks, episodes = load_meta(str(root))
    import pyarrow.parquet as pq

    candidates = []
    for ep in episodes:
        if ep.get("robot_type", args.robot_type) != args.robot_type:
            continue
        path = root / parquet_rel(int(ep["episode_index"]))
        if not path.is_file():
            continue
        candidates.append(path)
        if len(candidates) >= args.max_episodes:
            break

    if not candidates:
        raise SystemExit(f"No local G1 parquet under {root}. Download HE data first.")

    print(f"Canonical POLICY_JOINTS ({len(POLICY_JOINTS)}):")
    for i, n in enumerate(POLICY_JOINTS):
        print(f"  [{i:02d}] {n}")

    layouts = {
        "arm14+hand14": lambda arm, hand: np.concatenate([arm, hand], axis=-1),
        "hand14+arm14": lambda arm, hand: np.concatenate([hand, arm], axis=-1),
    }

    for path in candidates:
        table = pq.read_table(path)
        cols = set(table.column_names)
        need = {"action", "observation.arm_joints", "observation.hand_joints"}
        if not need.issubset(cols):
            print(f"\n[skip] {path.name}: missing {need - cols}")
            continue
        action = _stack_col(table, "action")
        arm = _stack_col(table, "observation.arm_joints")
        hand = _stack_col(table, "observation.hand_joints")
        print(f"\n=== {path.name}  T={action.shape[0]}  "
              f"action={action.shape} arm={arm.shape} hand={hand.shape} ===")
        # Action vs next-step proprio is common for absolute targets; try t and t+1.
        for layout_name, fn in layouts.items():
            proprio = fn(arm, hand)
            if proprio.shape[-1] != action.shape[-1]:
                print(f"  {layout_name}: dim mismatch {proprio.shape} vs {action.shape}")
                continue
            for shift, label in ((0, "t"), (1, "t+1")):
                if shift:
                    a, p = action[:-shift], proprio[shift:]
                else:
                    a, p = action, proprio
                l2 = float(np.mean(np.linalg.norm(a - p, axis=-1)))
                # Mean per-dim corr
                corrs = [_corr(a[:, d], p[:, d]) for d in range(a.shape[-1])]
                mean_c = float(np.nanmean(corrs))
                print(f"  {layout_name} vs proprio[{label}]: "
                      f"mean_L2={l2:.4f}  mean_corr={mean_c:.4f}")

        # Per-dim report for best guess arm+hand @ t
        proprio = np.concatenate([arm, hand], axis=-1)
        print("  per-dim corr action vs arm+hand @ t:")
        for d in range(min(28, action.shape[-1])):
            c = _corr(action[:, d], proprio[:, d])
            print(f"    dim {d:02d}: corr={c:+.3f}  "
                  f"act[{action[:, d].min():+.2f},{action[:, d].max():+.2f}]  "
                  f"prop[{proprio[:, d].min():+.2f},{proprio[:, d].max():+.2f}]")

    # Print action stats path if present
    stats = sorted((root / "meta").glob("action_stats_g1_*.json"))
    if stats:
        with open(stats[0]) as f:
            s = json.load(f)
        print(f"\naction stats: {stats[0]}  q01_len={len(s.get('q01', []))}")


if __name__ == "__main__":
    main()
