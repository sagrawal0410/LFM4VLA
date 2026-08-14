#!/usr/bin/env bash
# Submit independent VLA-Adapter finetunes for all four LIBERO suites (450M).
# DEPTH=1 submits the depth-conditioned variants (requires the depth branch/configs).
set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SBATCH_SCRIPT="${LFM4VLA_ROOT}/scripts/train_lfm_vla_adapter_libero_450m.sbatch"
DEPTH="${DEPTH:-0}"

for SUITE in spatial object goal long; do
  echo "Submitting SUITE=${SUITE} DEPTH=${DEPTH}"
  sbatch --export=ALL,SUITE="${SUITE}",DEPTH="${DEPTH}" "${SBATCH_SCRIPT}"
done
