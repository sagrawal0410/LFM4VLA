#!/bin/bash
#SBATCH --job-name=lfm_vla_adapter_libero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --output=output_lfm_vla_adapter_libero_%j.out
#SBATCH --error=output_lfm_vla_adapter_libero_%j.err

set -euo pipefail

# --- edit these for your cluster layout ---
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
# One of: spatial | object | goal | long | all4
SUITE="${SUITE:-spatial}"
# DEPTH=1 -> depth-conditioned adapter configs; DEPTH=0 (default) -> RGB-only.
DEPTH="${DEPTH:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"

if [[ "$DEPTH" == "1" ]]; then
  PREFIX="configs/lfm2.5-vl-450m-vla-adapter-depth-libero"
else
  PREFIX="configs/lfm2.5-vl-450m-vla-adapter-libero"
fi

case "$SUITE" in
  spatial|object|goal|long|all4) CONFIG="${PREFIX}-${SUITE}.json" ;;
  *) echo "Unknown SUITE=$SUITE (expected spatial|object|goal|long|all4)"; exit 1 ;;
esac

if [[ ! -f "$LFM4VLA_ROOT/$CONFIG" ]]; then
  echo "Config not found: $CONFIG (SUITE=$SUITE DEPTH=$DEPTH)"; exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$LFM4VLA_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Guarantee online W&B logging (set WANDB_API_KEY in your shell/rc beforehand).
export WANDB_MODE="${WANDB_MODE:-online}"

echo "[vla-adapter] suite=$SUITE depth=$DEPTH config=$CONFIG"

srun python main.py \
  --config "$CONFIG" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"
