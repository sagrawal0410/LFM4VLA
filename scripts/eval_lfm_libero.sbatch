#!/bin/bash
#SBATCH --job-name=lfm4vla_libero_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=output_libero_eval_%j.out
#SBATCH --error=output_libero_eval_%j.err

set -euo pipefail

# --- edit for your run ---
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"

CONFIG="${CONFIG:?Set CONFIG to your saved *-config.json}"
CKPT="${CKPT:?Set CKPT to last.ckpt or a checkpoint dir}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-10}"
# 1 = closed-loop (query every env step). >1 = open-loop chunking (faster, less reactive).
EXECUTE_STEP="${EXECUTE_STEP:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/libero_eval/$(basename "$CKPT" .ckpt)}"
# Optional: RLDS dir holding dataset_statistics*.json (defaults to config's data_root_dir).
DATA_ROOT_DIR="${DATA_ROOT_DIR:-}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# auto = try GPU EGL, fall back to CPU OSMesa (needed on many SLURM nodes).
export MUJOCO_GL="${MUJOCO_GL:-auto}"
# Optional: override ~/.libero/config.yaml location (evaluate_libero.py auto-creates it).
# export LIBERO_CONFIG_PATH="$HOME/.libero"

EXTRA_ARGS=()
if [[ -n "$DATA_ROOT_DIR" ]]; then
  EXTRA_ARGS+=(--data_root_dir "$DATA_ROOT_DIR")
fi

srun python eval/libero/evaluate_libero.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --task_suite "$TASK_SUITE" \
  --num_trials_per_task "$NUM_TRIALS" \
  --execute_step "$EXECUTE_STEP" \
  --save_video \
  --output_dir "$OUTPUT_DIR" \
  --device cuda:0 \
  "${EXTRA_ARGS[@]}"

echo "Results + MP4s: $LFM4VLA_ROOT/$OUTPUT_DIR"
