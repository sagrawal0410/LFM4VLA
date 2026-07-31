#!/bin/bash
#SBATCH --job-name=lfm4vla_libero_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm4vla_libero_depth_%j.out
#SBATCH --error=output_lfm4vla_libero_depth_%j.err

set -euo pipefail

# Depth-conditioned LIBERO runs (CNN patch encoder + multimodal QFormer).
# Example:
#   sbatch --export=ALL,CONFIG=configs/lfm2.5-vl-450m-libero-depth-latent1.json scripts/train_lfm_libero_depth.sbatch
#   sbatch --export=ALL,CONFIG=configs/lfm2.5-vl-450m-libero-depth-latent4.json scripts/train_lfm_libero_depth.sbatch

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero-depth-latent1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$LFM4VLA_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "Using CONFIG=${CONFIG}"
echo "output_root=${OUTPUT_ROOT}  log_root=${LOG_ROOT}"
srun python main.py \
  --config "$CONFIG" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"
