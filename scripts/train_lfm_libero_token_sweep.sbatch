#!/bin/bash
#SBATCH --job-name=lfm4vla_libero_tok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --array=0-5
#SBATCH --output=output_lfm4vla_libero_tok_%A_%a.out
#SBATCH --error=output_lfm4vla_libero_tok_%A_%a.err

# 6-run plain LIBERO action-token sweep (no depth):
#   num_action_tokens ∈ {1, 2, 4, 6, 8, 16}
#   latent            = 1  (fixed)
#
# Uses stock modified_libero_rlds (RGB only) via configs/lfm2.5-vl-450m-libero.json.
#
# Usage:
#   sbatch scripts/train_lfm_libero_token_sweep.sbatch
#   CONFIG=configs/lfm2.5-vl-1.6b-libero.json sbatch scripts/train_lfm_libero_token_sweep.sbatch
#   # subset: sbatch --array=0-2 scripts/train_lfm_libero_token_sweep.sbatch

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero.json}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# LIBERO streams RLDS through TensorFlow; keep TF off the GPU (PyTorch owns it).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Resolve num_action_tokens from the SLURM array index, unless overridden.
# latent is always forced to 1 for this sweep.
if [[ -z "${NUM_ACTION_TOKENS:-}" ]]; then
  eval "$(python scripts/sweep_libero_tokens.py --index "${SLURM_ARRAY_TASK_ID:?}")"
fi
LATENT="${LATENT:-1}"
TASK_NAME="${TASK_NAME:-libero_tok${NUM_ACTION_TOKENS}}"

echo "=== LIBERO action-token sweep (plain, no depth) ==="
echo "array_id=${SLURM_ARRAY_TASK_ID:-n/a}  tokens=${NUM_ACTION_TOKENS}  latent=${LATENT}  task=${TASK_NAME}"
echo "config=${CONFIG}"

srun python main.py \
  --config "$CONFIG" \
  --num_action_tokens "$NUM_ACTION_TOKENS" \
  --latent "$LATENT" \
  --task_name "$TASK_NAME"
