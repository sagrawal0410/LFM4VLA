#!/bin/bash
#SBATCH --job-name=he_depth_dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=output_he_depth_dl_%j.out
#SBATCH --error=output_he_depth_dl_%j.err

# Download G1 4-skill Humanoid Everyday subset WITH egocentric depth into a
# dedicated tree (do not overwrite the RGB-only humanoid_everyday dir).
#
#   sbatch scripts/download_humanoid_everyday_depth.sbatch
#   OUTPUT_DIR=/path/to/custom sbatch scripts/download_humanoid_everyday_depth.sbatch

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/teams/research/robotics/humanoid_everyday_depth}"
WORKERS="${WORKERS:-8}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
echo "=== HE depth download ==="
echo "output_dir=${OUTPUT_DIR}  workers=${WORKERS}  env=${CONDA_ENV}"

python scripts/download_humanoid_everyday.py \
  --output_dir "$OUTPUT_DIR" \
  --include_depth \
  --workers "$WORKERS"
