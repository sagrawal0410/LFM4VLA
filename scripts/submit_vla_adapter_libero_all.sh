#!/usr/bin/env bash
# Submit independent VLA-Adapter finetunes for all four LIBERO suites (450M).
# DEPTH=0 (default) submits the RGB-only setup for all four suites.
# DEPTH=1 submits the depth-conditioned setup, which only has data for LIBERO-Long.
set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SBATCH_SCRIPT="${LFM4VLA_ROOT}/scripts/train_lfm_vla_adapter_libero_450m.sbatch"
DEPTH="${DEPTH:-0}"

if [[ "$DEPTH" == "1" ]]; then
  SUITES=(long)
else
  SUITES=(spatial object goal long)
fi

for SUITE in "${SUITES[@]}"; do
  echo "Submitting SUITE=${SUITE} DEPTH=${DEPTH}"
  sbatch --export=ALL,SUITE="${SUITE}",DEPTH="${DEPTH}" "${SBATCH_SCRIPT}"
done
