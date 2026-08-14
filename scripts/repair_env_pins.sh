#!/usr/bin/env bash
# Repair a drifted LIBERO/RLDS training env WITHOUT a full reinstall.
#
# Use when a job dies with one of:
#   ImportError: numpy.core.umath failed to import          (NumPy bumped to 2.x)
#   ImportError: cannot import name 'runtime_version'       (tensorflow-metadata too new)
#   ImportError: cannot import name 'bfloat16' (ml_dtypes)  (ml-dtypes/NumPy mismatch)
#
#   conda activate lfm4vla
#   bash scripts/repair_env_pins.sh
#
# For a from-scratch install use scripts/install_libero_rlds_deps.sh instead.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[repair] No conda env active. Run: conda activate lfm4vla" >&2
  exit 1
fi

echo "=== Repairing pins in ${CONDA_PREFIX} ==="
python -c 'import sys; print("python", sys.version.split()[0])'

# --no-deps so we only move these packages; nothing else gets re-resolved.
# ml-dtypes is reinstalled after NumPy so it binds to the 1.x ABI.
pip install --no-cache-dir --force-reinstall --no-deps \
  "numpy==1.26.4" \
  "protobuf==3.20.3" \
  "tensorflow-metadata==1.14.0"
pip install --no-cache-dir --force-reinstall --no-deps "ml-dtypes>=0.2.0,<0.4.0"

echo "=== Verifying ==="
python scripts/check_env_pins.py

echo "[repair] Done. Re-submit your job."
echo "[repair] To stop this recurring, install with the shared pin set:"
echo "           pip install -c constraints.txt <package>"
