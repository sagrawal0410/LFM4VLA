#!/usr/bin/env python3
"""Build LIBERO-10 depth RLDS via TFDS Python API (avoids fragile ``tfds`` CLI).

The ``tfds`` CLI imports ``apache_beam`` unconditionally (convert_format).
This builder is a plain ``GeneratorBasedBuilder`` and does not need Beam.

Env:
  TFDS_DATA_DIR              output root (e.g. .../modified_libero_rlds_depth)
  LIBERO_DEPTH_HDF5_GLOB     input HDF5 glob
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    data_dir = os.environ.get("TFDS_DATA_DIR")
    if not data_dir:
        raise SystemExit("TFDS_DATA_DIR must be set")
    hdf5_glob = os.environ.get("LIBERO_DEPTH_HDF5_GLOB")
    if not hdf5_glob:
        raise SystemExit("LIBERO_DEPTH_HDF5_GLOB must be set")

    builder_py = (
        Path(__file__).resolve().parent
        / "rlds_builders"
        / "libero_10_no_noops"
        / "libero_10_no_noops_dataset_builder.py"
    )
    if not builder_py.is_file():
        raise FileNotFoundError(builder_py)

    # Import builder module by path (no need to cd / rely on tfds discovery).
    spec = importlib.util.spec_from_file_location(
        "libero_10_no_noops_dataset_builder", builder_py
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load builder from {builder_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    builder = mod.Libero10NoNoops(data_dir=data_dir)

    # Match ``tfds build --overwrite``.
    version_dir = Path(data_dir) / builder.name / str(builder.VERSION)
    if version_dir.exists():
        print(f"[tfds] removing existing {version_dir}", flush=True)
        shutil.rmtree(version_dir)

    print(
        f"[tfds] building {builder.name} → {data_dir}\n"
        f"[tfds] HDF5 glob: {hdf5_glob}",
        flush=True,
    )
    builder.download_and_prepare()
    print(f"[tfds] DONE  info={builder.info}", flush=True)


if __name__ == "__main__":
    main()
