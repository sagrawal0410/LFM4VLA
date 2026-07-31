#!/usr/bin/env python3
"""Ensure TF / TFDS / Beam deps for ``tfds build`` (LIBERO RLDS conversion).

TF 2.15 needs NumPy 1.x. Recent ``opencv-python`` 5.x wants NumPy 2.x and
``tfds`` CLI imports ``apache_beam`` even for non-Beam builders. This script
installs a coherent set and re-pins NumPy last.
"""

from __future__ import annotations

import importlib.metadata as m
import subprocess
import sys


def _ver(pkg: str) -> str | None:
    try:
        return m.version(pkg)
    except m.PackageNotFoundError:
        return None


def _pip(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *args]
    )


def _need_apache_beam() -> bool:
    try:
        import apache_beam  # noqa: F401

        return False
    except Exception:
        return True


def main() -> None:
    np_v = _ver("numpy") or "0"
    if int(np_v.split(".", 1)[0]) >= 2:
        print(f"[rlds-deps] NumPy {np_v} → pin 1.26.4", flush=True)
        _pip("numpy==1.26.4")

    # OpenCV 5.x requires NumPy>=2; keep 4.x for TF/dlimp.
    for pkg in ("opencv-python", "opencv-python-headless"):
        v = _ver(pkg)
        if v is None:
            continue
        major = int(v.split(".", 1)[0])
        if major >= 5:
            print(f"[rlds-deps] {pkg}={v} needs NumPy 2 → pin 4.10.0.84", flush=True)
            _pip(f"{pkg}==4.10.0.84")

    if _need_apache_beam():
        print("[rlds-deps] installing apache-beam (required by tfds CLI)", flush=True)
        # Pin a Beam that still supports py3.10; avoid pulling NumPy 2.
        _pip("apache-beam==2.54.0")

    # Re-pin after Beam / OpenCV may have upgraded things.
    _pip(
        "numpy==1.26.4",
        "protobuf==3.20.3",
        "tensorflow-metadata==1.14.0",
    )

    # Verify the imports the build actually needs.
    import numpy as np

    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise SystemExit(f"NumPy still {np.__version__} after pin")

    import apache_beam  # noqa: F401
    import h5py  # noqa: F401
    import tensorflow as tf  # noqa: F401
    import tensorflow_datasets as tfds  # noqa: F401

    print(
        f"[rlds-deps] OK  numpy={np.__version__}  tf={tf.__version__}  "
        f"tfds={tfds.__version__}  beam={_ver('apache-beam')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
