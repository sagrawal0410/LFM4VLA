#!/usr/bin/env bash
# Submit the 8-run LIBERO depth sweep (latent × qformer queries), 1 GPU each.
#
# Usage:
#   bash scripts/submit_libero_depth_sweep.sh
#   CONFIG=configs/lfm2.5-vl-1.6b-libero-depth-latent1.json bash scripts/submit_libero_depth_sweep.sh
#   DEPENDENCY=12345 bash scripts/submit_libero_depth_sweep.sh   # afterok on data-build job

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero-depth-latent1.json}"
DEPENDENCY="${DEPENDENCY:-}"

cd "$LFM4VLA_ROOT"

EXTRA=()
if [[ -n "$DEPENDENCY" ]]; then
  EXTRA+=(--dependency="afterok:${DEPENDENCY}")
fi

# Array 0-7 → 8 tasks, each with --gpus-per-node=1 (separate GPU allocation).
sbatch "${EXTRA[@]}" \
  --export=ALL,CONFIG="$CONFIG",LFM4VLA_ROOT="$LFM4VLA_ROOT" \
  scripts/train_lfm_libero_depth_sweep.sbatch

echo "Submitted depth sweep with CONFIG=$CONFIG"
python scripts/sweep_libero_depth.py --list
