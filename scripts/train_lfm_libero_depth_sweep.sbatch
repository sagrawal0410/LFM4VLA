#!/bin/bash
#SBATCH --job-name=lfm4vla_libero_depth_swp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --array=0-7
#SBATCH --output=output_lfm4vla_libero_depth_swp_%A_%a.out
#SBATCH --error=output_lfm4vla_libero_depth_swp_%A_%a.err

# 8-run LIBERO depth sweep:
#   latent                    ∈ {1, 4}
#   depth.qformer.num_queries ∈ {8, 16, 32, 64}
#
# Usage:
#   sbatch scripts/train_lfm_libero_depth_sweep.sbatch
#   CONFIG=configs/lfm2.5-vl-1.6b-libero-depth-latent1.json sbatch scripts/train_lfm_libero_depth_sweep.sbatch
#   # subset: sbatch --array=0-3 scripts/train_lfm_libero_depth_sweep.sbatch

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero-depth-latent1.json}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Resolve (latent, depth_num_queries) from the SLURM array index, unless overridden.
if [[ -z "${LATENT:-}" || -z "${DEPTH_NUM_QUERIES:-}" ]]; then
  eval "$(python scripts/sweep_libero_depth.py --index "${SLURM_ARRAY_TASK_ID:?}")"
fi
TASK_NAME="${TASK_NAME:-libero_depth_lat${LATENT}_q${DEPTH_NUM_QUERIES}}"

echo "=== LIBERO depth sweep ==="
echo "array_id=${SLURM_ARRAY_TASK_ID:-n/a}  latent=${LATENT}  num_queries=${DEPTH_NUM_QUERIES}  task=${TASK_NAME}"
echo "config=${CONFIG}"

srun python main.py \
  --config "$CONFIG" \
  --latent "$LATENT" \
  --depth_num_queries "$DEPTH_NUM_QUERIES" \
  --task_name "$TASK_NAME"
