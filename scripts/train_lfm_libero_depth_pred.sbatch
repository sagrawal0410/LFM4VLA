#!/bin/bash
#SBATCH --job-name=lfm4vla_libero_depth_pred
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G
#SBATCH --output=output_lfm4vla_libero_depth_pred_%j.out
#SBATCH --error=output_lfm4vla_libero_depth_pred_%j.err

set -euo pipefail

# Auxiliary depth prediction on LIBERO-10 (third alternate):
#   RGB → LLM + learned action/depth query tokens → HierarchicalFCDecoder
#   (separate action trunk → arm/gripper; separate depth trunk → depth map)
#   with aux L1 depth loss.
# No input-side depth conditioning (use_depth=false); GT depth is supervision only.
#
# Example:
#   sbatch scripts/train_lfm_libero_depth_pred.sbatch
#   sbatch --export=NONE,CONFIG=configs/lfm2.5-vl-450m-libero-depth-pred.json \
#     scripts/train_lfm_libero_depth_pred.sbatch

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero-depth-pred.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$LFM4VLA_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Keep TF off the GPU so JAX/TF don't fight PyTorch for VRAM (depth RLDS path).
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Using CONFIG=${CONFIG}"
echo "output_root=${OUTPUT_ROOT}  log_root=${LOG_ROOT}"
srun python main.py \
  --config "$CONFIG" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"
