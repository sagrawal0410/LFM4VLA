#!/bin/bash
#SBATCH --job-name=he_depth_npy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=output_he_depth_npy_%j.out
#SBATCH --error=output_he_depth_npy_%j.err

# One-time: parquet depth → per-episode float16 .npy (required for fast HE depth train).
#
#   sbatch scripts/extract_he_depth_arrays.sbatch
#   DATA_ROOT=/path/to/humanoid_everyday_depth sbatch scripts/extract_he_depth_arrays.sbatch

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
DATA_ROOT="${DATA_ROOT:-/home/teams/research/robotics/humanoid_everyday_depth}"
WORKERS="${WORKERS:-8}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
echo "=== Extract HE depth npy ==="
echo "data_root=${DATA_ROOT}  workers=${WORKERS}"

python scripts/extract_he_depth_arrays.py \
  --data_root "$DATA_ROOT" \
  --workers "$WORKERS"
