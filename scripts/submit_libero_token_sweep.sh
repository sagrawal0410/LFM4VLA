#!/usr/bin/env bash
# Submit the 6-run plain LIBERO action-token sweep (num_action_tokens), 1 GPU each.
#
# Usage:
#   bash scripts/submit_libero_token_sweep.sh
#   CONFIG=configs/lfm2.5-vl-1.6b-libero.json bash scripts/submit_libero_token_sweep.sh

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero.json}"

cd "$LFM4VLA_ROOT"

sbatch \
  --export=ALL,CONFIG="$CONFIG",LFM4VLA_ROOT="$LFM4VLA_ROOT" \
  scripts/train_lfm_libero_token_sweep.sbatch

echo "Submitted LIBERO token sweep with CONFIG=$CONFIG"
python scripts/sweep_libero_tokens.py --list
