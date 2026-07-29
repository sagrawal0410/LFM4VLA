#!/bin/bash
#SBATCH --job-name=lfm4vla_he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm4vla_he_%j.out
#SBATCH --error=output_lfm4vla_he_%j.err

set -euo pipefail

# Humanoid Everyday fine-tune (pick/place, push/pull, stack/align, pour) — G1 28-D.
# Prerequisite: download the subset once on a login node:
#   python scripts/download_humanoid_everyday.py \
#       --output_dir /home/teams/research/robotics/humanoid_everyday
#   pip install av pyarrow huggingface_hub

# --- edit these for your cluster layout ---
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-humanoid-everyday.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$LFM4VLA_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

srun python main.py --config "$CONFIG"
