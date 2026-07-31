#!/bin/bash
#SBATCH --job-name=lfm4vla_he_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/teams/research/robotics/logs/slurm/he_depth_%j.out
#SBATCH --error=/home/teams/research/robotics/logs/slurm/he_depth_%j.err

# Single-run HE depth fine-tune (Q-Former num_queries=32).
# Env: bash scripts/install_humanoid_everyday_env.sh

set -eo pipefail

echo "[he-depth] job start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-n/a}"

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-humanoid-everyday-depth.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"

mkdir -p /home/teams/research/robotics/logs/slurm \
         "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - <<'PY'
import numpy as np, sys
if int(np.__version__.split(".", 1)[0]) >= 2:
    raise SystemExit(
        f"Env '{sys.prefix}' has NumPy {np.__version__}. "
        "Use lfm4vla-he (bash scripts/install_humanoid_everyday_env.sh)."
    )
print(f"[he-depth] env={sys.prefix} numpy={np.__version__}", flush=True)
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

_MIOPEN_BASE="${TMPDIR:-/tmp}/miopen_${USER}_${SLURM_JOB_ID:-$$}"
mkdir -p "$_MIOPEN_BASE"
export MIOPEN_USER_DB_PATH="$_MIOPEN_BASE"
export MIOPEN_CUSTOM_CACHE_DIR="$_MIOPEN_BASE"
echo "[he-depth] MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH}"

echo "config=${CONFIG}  output_root=${OUTPUT_ROOT}"

srun --cpu-bind=none python main.py \
  --config "$CONFIG" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"

echo "[he-depth] job end $(date -Is)"
