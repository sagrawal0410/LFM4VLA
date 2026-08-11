#!/bin/bash
#SBATCH --job-name=lfm_vla_adapter_libero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm_vla_adapter_libero_%j.out
#SBATCH --error=output_lfm_vla_adapter_libero_%j.err

set -euo pipefail

# --- edit these for your cluster layout ---
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
# One of: spatial | object | goal | long
SUITE="${SUITE:-spatial}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"

case "$SUITE" in
  spatial) CONFIG="configs/lfm2.5-vl-450m-vla-adapter-libero-spatial.json" ;;
  object)  CONFIG="configs/lfm2.5-vl-450m-vla-adapter-libero-object.json" ;;
  goal)    CONFIG="configs/lfm2.5-vl-450m-vla-adapter-libero-goal.json" ;;
  long)    CONFIG="configs/lfm2.5-vl-450m-vla-adapter-libero-long.json" ;;
  all4)    CONFIG="configs/lfm2.5-vl-450m-vla-adapter-libero-all4.json" ;;
  *) echo "Unknown SUITE=$SUITE (expected spatial|object|goal|long|all4)"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$LFM4VLA_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[vla-adapter] suite=$SUITE config=$CONFIG"

srun python main.py \
  --config "$CONFIG" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"
