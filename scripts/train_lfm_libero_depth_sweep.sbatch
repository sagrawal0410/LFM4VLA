#!/bin/bash
#SBATCH --job-name=lfm4vla_libero_depth_swp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G
#SBATCH --array=0-7
#SBATCH --output=output_lfm4vla_libero_depth_swp_%A_%a.out
#SBATCH --error=output_lfm4vla_libero_depth_swp_%A_%a.err

# 8-run LIBERO depth sweep:
#   latent                    ∈ {1, 4}
#   depth.qformer.num_queries ∈ {8, 16, 32, 64}
#
# Usage:
#   sbatch scripts/train_lfm_libero_depth_sweep.sbatch
#   CONFIG=configs/lfm2.5-vl-1.6b-libero-depth-latent1.json sbatch scripts/train_lfm_libero_depth_sweep.sbatch
#   # subset: sbatch --array=0-3 scripts/train_lfm_libero_depth_sweep.sbatch
#
# Do NOT submit with --export=ALL from a login shell (breaks conda/PATH on compute).

set -euo pipefail

# Log immediately so empty .out/.err means the job never started the script.
echo "[depth-sweep] host=$(hostname)  date=$(date -Is)  pwd=$(pwd)"
echo "[depth-sweep] job=${SLURM_JOB_ID:-?}  array=${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?}"
echo "[depth-sweep] PATH=${PATH}"

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero-depth-latent1.json}"
# Checkpoints / logs land under <root>/<YYYY-MM-DD>/<exp_name>/
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}"
CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}"

cd "$LFM4VLA_ROOT"
echo "[depth-sweep] LFM4VLA_ROOT=$LFM4VLA_ROOT  CONFIG=$CONFIG"

# Prefer a concrete conda base; fall back to `conda info`.
CONDA_BASE="${CONDA_BASE:-}"
if [[ -z "$CONDA_BASE" ]]; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BASE="$HOME/miniconda3"
  elif [[ -x "$HOME/mambaforge/bin/conda" ]]; then
    CONDA_BASE="$HOME/mambaforge"
  else
    CONDA_BASE="$(conda info --base)"
  fi
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "[depth-sweep] python=$(command -v python)  ($(python -V 2>&1))"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Keep TF on CPU for RLDS and cap threads so the shuffle buffer doesn't blow host RAM.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Clear stale sweep knobs from a polluted login env unless explicitly set for this task.
# (Submitting with --export=ALL can freeze LATENT/DEPTH_NUM_QUERIES across all array indices.)
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset — submit as an array job (this script has #SBATCH --array=0-7)" >&2
  exit 1
fi

# Always resolve from array index so every task gets a distinct (latent, q) pair.
eval "$(python scripts/sweep_libero_depth.py --index "${SLURM_ARRAY_TASK_ID}")"
TASK_NAME="${TASK_NAME:-libero_depth_lat${LATENT}_q${DEPTH_NUM_QUERIES}}"

echo "=== LIBERO depth sweep ==="
echo "array_id=${SLURM_ARRAY_TASK_ID}  latent=${LATENT}  num_queries=${DEPTH_NUM_QUERIES}  task=${TASK_NAME}"
echo "config=${CONFIG}"
echo "output_root=${OUTPUT_ROOT}  log_root=${LOG_ROOT}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG (cwd=$(pwd))" >&2
  exit 1
fi

srun python main.py \
  --config "$CONFIG" \
  --latent "$LATENT" \
  --depth_num_queries "$DEPTH_NUM_QUERIES" \
  --task_name "$TASK_NAME" \
  --output_root "$OUTPUT_ROOT" \
  --log_root "$LOG_ROOT" \
  --cache_root "$CACHE_ROOT"
