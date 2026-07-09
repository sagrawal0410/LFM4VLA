#!/bin/bash
#SBATCH --job-name=lfm4vla_libero
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm4vla_libero_%j.out
#SBATCH --error=output_lfm4vla_libero_%j.err
#SBATCH --account=liquidai

set -euo pipefail

# --- edit these for your cluster layout ---
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla_libero}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero.json}"

# LIBERO streams RLDS through TensorFlow; keep TF off the GPU (PyTorch owns it).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Optional: cache + logging (uncomment/set as needed)
# export HF_HOME=/lambdafs/$USER/hf_cache
# export WANDB_API_KEY=...
# export WANDB_MODE=online

cd "$LFM4VLA_ROOT"

# conda (recommended) — or swap for: source venv/bin/activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# LIBERO is a single-GPU streaming pipeline (RLDS stream is not rank-sharded).
srun python main.py --config "$CONFIG"
