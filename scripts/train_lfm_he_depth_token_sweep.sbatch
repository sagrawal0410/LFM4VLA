#!/bin/bash
#SBATCH --job-name=lfm4vla_he_depth_tok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --array=0-34
#SBATCH --output=/home/teams/research/robotics/logs/slurm/he_depth_tok_%A_%a.out
#SBATCH --error=/home/teams/research/robotics/logs/slurm/he_depth_tok_%A_%a.err

# 35-run Humanoid Everyday DEPTH sweep (450M by default):
#   num_action_tokens ∈ {1,2,4,8,16,24,28}
#   latent            ∈ {1,2,4,8,16}
#   depth Q-Former    fixed at num_queries=32
#
# Prerequisites (once):
#   bash scripts/install_humanoid_everyday_env.sh
#   python scripts/download_humanoid_everyday.py \
#       --output_dir /home/teams/research/robotics/humanoid_everyday_depth --include_depth
#   sbatch scripts/extract_he_depth_arrays.sbatch
#
# Usage:
#   mkdir -p /home/teams/research/robotics/logs/slurm
#   sbatch scripts/train_lfm_he_depth_token_sweep.sbatch

set -eo pipefail  # not -u: conda activate breaks with nounset

echo "[he-depth] job start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-n/a} array=${SLURM_ARRAY_TASK_ID:-n/a}"

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-humanoid-everyday-depth.json}"
DEPTH_NUM_QUERIES="${DEPTH_NUM_QUERIES:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"
DATA_ROOT="${DATA_ROOT:-/home/teams/research/robotics/humanoid_everyday_depth}"

mkdir -p /home/teams/research/robotics/logs/slurm \
         "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT"

cd "$LFM4VLA_ROOT"
echo "[he-depth] cwd=$(pwd) config=${CONFIG}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "[he-depth] conda=${CONDA_DEFAULT_ENV:-?} prefix=${CONDA_PREFIX:-?}"

python - <<'PY'
import numpy as np, sys
from pathlib import Path
if int(np.__version__.split(".", 1)[0]) >= 2:
    raise SystemExit(
        f"Env '{sys.prefix}' has NumPy {np.__version__}. "
        "Use lfm4vla-he (bash scripts/install_humanoid_everyday_env.sh)."
    )
print(f"[he-depth] env={sys.prefix} numpy={np.__version__}", flush=True)
PY

# Fail fast if depth npy sidecars are missing (parquet-only = tiny / NaN-prone).
python - <<PY
from pathlib import Path
root = Path("${DATA_ROOT}")
n_npy = len(list(root.glob("depths/chunk-*/episode_*.npy")))
if n_npy < 100:
    raise SystemExit(
        f"[he-depth] only {n_npy} depth npy files under {root}/depths. "
        "Run: sbatch scripts/extract_he_depth_arrays.sbatch"
    )
print(f"[he-depth] found {n_npy} depth npy files under {root}", flush=True)
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# MIOpen: array jobs racing on ~/.config/miopen corrupt the SQLite kernel DB
# ("database disk image is malformed" → miopenStatusInternalError on Conv2d).
# Give every task its own cache under local /tmp (or SLURM TMPDIR).
_MIOPEN_BASE="${TMPDIR:-/tmp}/miopen_${USER}_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$_MIOPEN_BASE"
export MIOPEN_USER_DB_PATH="$_MIOPEN_BASE"
export MIOPEN_CUSTOM_CACHE_DIR="$_MIOPEN_BASE"
echo "[he-depth] MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH}"

if [[ -z "${NUM_ACTION_TOKENS:-}" || -z "${LATENT:-}" ]]; then
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "[he-depth] ERROR: SLURM_ARRAY_TASK_ID unset and NUM_ACTION_TOKENS/LATENT not provided" >&2
    exit 1
  fi
  eval "$(python scripts/sweep_he_tokens.py --index "${SLURM_ARRAY_TASK_ID}")"
fi
TASK_NAME="${TASK_NAME:-he_depth_q${DEPTH_NUM_QUERIES}_tok${NUM_ACTION_TOKENS}_lat${LATENT}}"

echo "=== HE depth token sweep ==="
echo "array_id=${SLURM_ARRAY_TASK_ID:-n/a}  tokens=${NUM_ACTION_TOKENS}  latent=${LATENT}"
echo "depth_num_queries=${DEPTH_NUM_QUERIES}  task=${TASK_NAME}"
echo "config=${CONFIG}  env=${CONDA_ENV}"
echo "output_root=${OUTPUT_ROOT}  log_root=${LOG_ROOT}"

srun --cpu-bind=none python main.py \
  --config "$CONFIG" \
  --num_action_tokens "$NUM_ACTION_TOKENS" \
  --latent "$LATENT" \
  --depth_num_queries "$DEPTH_NUM_QUERIES" \
  --task_name "$TASK_NAME" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"

echo "[he-depth] job end $(date -Is)"
