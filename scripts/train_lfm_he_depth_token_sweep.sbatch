#!/bin/bash
#SBATCH --job-name=lfm4vla_he_depth_tok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --array=0-34
#SBATCH --output=output_lfm4vla_he_depth_tok_%A_%a.out
#SBATCH --error=output_lfm4vla_he_depth_tok_%A_%a.err

# 35-run Humanoid Everyday DEPTH sweep (450M by default):
#   num_action_tokens ∈ {1,2,4,8,16,24,28}
#   latent            ∈ {1,2,4,8,16}
#   depth Q-Former    fixed at num_queries=32 (same DepthConditioner as LIBERO)
#
# Data (separate from RGB-only tree):
#   python scripts/download_humanoid_everyday.py \
#       --output_dir /home/teams/research/robotics/humanoid_everyday_depth \
#       --include_depth
#   # REQUIRED for speed (else ~minutes/step reading parquet depth):
#   sbatch scripts/extract_he_depth_arrays.sbatch
#
# Env:
#   bash scripts/install_humanoid_everyday_env.sh
#
# Usage:
#   sbatch scripts/train_lfm_he_depth_token_sweep.sbatch
#   CONFIG=configs/lfm2.5-vl-1.6b-humanoid-everyday-depth.json \
#     sbatch scripts/train_lfm_he_depth_token_sweep.sbatch
#   # subset: sbatch --array=0-4 scripts/train_lfm_he_depth_token_sweep.sbatch

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-humanoid-everyday-depth.json}"
DEPTH_NUM_QUERIES="${DEPTH_NUM_QUERIES:-32}"
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

# Resolve (num_action_tokens, latent) from the SLURM array index, unless overridden.
if [[ -z "${NUM_ACTION_TOKENS:-}" || -z "${LATENT:-}" ]]; then
  eval "$(python scripts/sweep_he_tokens.py --index "${SLURM_ARRAY_TASK_ID:?}")"
fi
TASK_NAME="${TASK_NAME:-he_depth_q${DEPTH_NUM_QUERIES}_tok${NUM_ACTION_TOKENS}_lat${LATENT}}"

echo "=== HE depth token sweep ==="
echo "array_id=${SLURM_ARRAY_TASK_ID:-n/a}  tokens=${NUM_ACTION_TOKENS}  latent=${LATENT}"
echo "depth_num_queries=${DEPTH_NUM_QUERIES}  task=${TASK_NAME}"
echo "config=${CONFIG}  env=${CONDA_ENV}"

srun python main.py \
  --config "$CONFIG" \
  --num_action_tokens "$NUM_ACTION_TOKENS" \
  --latent "$LATENT" \
  --depth_num_queries "$DEPTH_NUM_QUERIES" \
  --task_name "$TASK_NAME" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"
