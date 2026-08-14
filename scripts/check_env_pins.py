#!/usr/bin/env python
"""Preflight check for the fragile NumPy/protobuf/TensorFlow pin set.

The RLDS data pipeline dies deep inside a TF import when one of these drifts
(most often NumPy getting bumped to 2.x by an unrelated ``pip install``). Run
this before a job so it fails in seconds with a fix, instead of after Slurm has
already handed out the allocation.

    python scripts/check_env_pins.py

Exit code 0 = env is usable; 1 = something is pinned wrong (details on stderr).
"""

from __future__ import annotations

import sys

REPAIR_HINT = "bash scripts/repair_env_pins.sh"

# (import name, distribution name, predicate on version string, requirement text)
VERSION_CHECKS = [
    ("numpy", "numpy", lambda v: v.startswith("1."), "1.x (TF 2.15 / ml_dtypes ABI)"),
    ("google.protobuf", "protobuf", lambda v: v.startswith("3.20."), "3.20.x (TF 2.15)"),
]

# Imports that must succeed for the RLDS pipeline to run at all.
REQUIRED_IMPORTS = ["ml_dtypes", "tensorflow", "tensorflow_datasets", "dlimp", "torch"]


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "<not installed>"


def main() -> int:
    problems: list[str] = []

    for _mod, dist, ok, requirement in VERSION_CHECKS:
        found = _version(dist)
        if not ok(found):
            problems.append(f"{dist}=={found} but needs {requirement}")

    for mod in REQUIRED_IMPORTS:
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001 - report any import failure verbatim
            problems.append(f"import {mod} failed: {type(exc).__name__}: {exc}")

    if problems:
        print("[env-check] FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(f"\n[env-check] repair with:\n  {REPAIR_HINT}", file=sys.stderr)
        return 1

    import numpy as np
    import tensorflow as tf

    print(f"[env-check] OK (numpy {np.__version__}, tensorflow {tf.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
