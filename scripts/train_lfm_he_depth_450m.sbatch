#!/bin/bash
#SBATCH --job-name=lfm4vla_he_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm4vla_he_depth_%j.out
#SBATCH --error=output_lfm4vla_he_depth_%j.err

# Single-run Humanoid Everyday depth fine-tune (Q-Former num_queries=32).
#
# Data:
#   python scripts/download_humanoid_everyday.py \
#       --output_dir /home/teams/research/robotics/humanoid_everyday_depth \
#       --include_depth
#
# Env: bash scripts/install_humanoid_everyday_env.sh  →  conda activate lfm4vla-he

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-humanoid-everyday-depth.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"

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
print(f"[he-depth] env={sys.prefix} numpy={np.__version__}")
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== HE depth train ==="
echo "config=${CONFIG}  env=${CONDA_ENV}"
echo "output_root=${OUTPUT_ROOT}  log_root=${LOG_ROOT}"

srun python main.py \
  --config "$CONFIG" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"
