#!/bin/bash
#SBATCH --job-name=lfm4vla_calvin_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=output_calvin_eval_%j.out
#SBATCH --error=output_calvin_eval_%j.err
#SBATCH --account=liquidai

set -euo pipefail

# --- edit for your run ---
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/calvin}"
DATASET_PATH="${DATASET_PATH:-/lambdafs/datasets/lfm4vla_calvin/calvin/dataset/task_ABC_D}"

CONFIG="${CONFIG:?Set CONFIG to your saved *-config.json}"
CKPT="${CKPT:?Set CKPT to last.ckpt or checkpoint dir}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/calvin_eval/$(basename "$CKPT" .ckpt)}"
NUM_SEQUENCES="${NUM_SEQUENCES:-20}"
# 1 = truly closed-loop (query policy every env step). >1 = open-loop chunking.
EXECUTE_STEP="${EXECUTE_STEP:-1}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Headless GPU rendering for PyBullet (required on cluster nodes without a display).
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=4.1

srun python eval/calvin/evaluate_calvin.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --calvin_root "$CALVIN_ROOT" \
  --dataset_path "$DATASET_PATH" \
  --num_sequences "$NUM_SEQUENCES" \
  --execute_step "$EXECUTE_STEP" \
  --save_video \
  --output_dir "$OUTPUT_DIR" \
  --device cuda:0

echo "Results + GIFs: $LFM4VLA_ROOT/$OUTPUT_DIR"
