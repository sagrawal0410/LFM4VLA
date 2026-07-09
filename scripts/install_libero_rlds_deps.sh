#!/usr/bin/env bash
# One-shot install for LIBERO RLDS deps inside an existing LFM4VLA conda env.
# Run on a login/head node (needs network + git for dlimp).
#
#   conda activate lfm4vla
#   bash scripts/install_libero_rlds_deps.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Remove conflicting protobuf / TFDS installs ==="
conda remove -y protobuf 2>/dev/null || true
pip uninstall -y protobuf google.protobuf tensorflow-metadata tensorflow-datasets tensorflow dlimp 2>/dev/null || true

# Wipe stale google.protobuf files that pip sometimes fails to overwrite.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  SP="${CONDA_PREFIX}/lib/python${PYVER}/site-packages"
  rm -rf "${SP}/google/protobuf" "${SP}/google/protobuf-"*.dist-info "${SP}/protobuf-"*.dist-info 2>/dev/null || true
fi

# Also remove any too-new tensorflow-metadata whose generated *_pb2.py call
# `google.protobuf.runtime_version` (only exists in protobuf 5.26+, breaks TF 2.15).
pip uninstall -y tensorflow-metadata 2>/dev/null || true

echo "=== 2. Install pinned RLDS stack (ordered to avoid resolver conflicts) ==="
# protobuf 3.20.3 + tensorflow-metadata 1.14.0 is the coherent combo for TF 2.15.
pip install --no-cache-dir "numpy==1.26.4" "protobuf==3.20.3"
pip install --no-cache-dir "tensorflow-metadata==1.14.0"
pip install --no-cache-dir --no-deps \
  "tensorflow==2.15.0" \
  "tensorflow-datasets==4.9.3"
pip install --no-cache-dir \
  "ml-dtypes>=0.2.0,<0.4.0" \
  "tensorflow-graphics==2021.12.3" \
  "absl-py" "rich" "tqdm"
pip install --no-cache-dir --no-deps --force-reinstall \
  git+https://github.com/moojink/dlimp_openvla

# Re-pin protobuf + metadata last in case a dep pulled newer ones.
pip install --no-cache-dir --force-reinstall --no-deps \
  "protobuf==3.20.3" "tensorflow-metadata==1.14.0"

echo "=== 3. Verify ==="
python - <<'PY'
import google.protobuf
print("protobuf", google.protobuf.__version__, "->", google.protobuf.__file__)
import tensorflow_metadata as tfm
print("tensorflow-metadata", tfm.__version__)
import numpy as np
print("numpy", np.__version__)
import tensorflow as tf
print("tensorflow", tf.__version__)
import tensorflow_datasets as tfds
print("tensorflow_datasets", tfds.__version__)
import dlimp
print("dlimp OK", dlimp.__file__)
print("ALL OK")
PY

echo "Done. Re-run: sbatch scripts/train_lfm_libero_450m.sbatch"