#!/bin/bash
#SBATCH --job-name=lfm4vla_close_drawer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=output_close_drawer_eval_%j.out
#SBATCH --error=output_close_drawer_eval_%j.err
#SBATCH --account=liquidai

set -euo pipefail
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/calvin}"
DATASET_PATH="${DATASET_PATH:-/lambdafs/datasets/lfm4vla_calvin/calvin/dataset/task_ABC_D}"

# Default to the lfm450m run from 2026-07-06 (override via env vars if needed).
CONFIG="${CONFIG:-$LFM4VLA_ROOT/runs/logs/2026-07-06/lfm450m_finetune-lfm450m-bs16x1-lr2e-05-ws1-FCDecoder-lat1-20260706-182828-fa18dc/lfm450m_finetune-lfm450m-bs16x1-lr2e-05-ws1-FCDecoder-lat1-20260706-182828-fa18dc-config.json}"
CKPT="${CKPT:-$LFM4VLA_ROOT/runs/checkpoints/2026-07-06/lfm450m_finetune-lfm450m-bs16x1-lr2e-05-ws1-FCDecoder-lat1-20260706-182828-fa18dc/last.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/calvin_eval/close_drawer_lfm450m}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_LEN="${EPISODE_LEN:-180}"
EXECUTE_STEP="${EXECUTE_STEP:-10}"
VIDEO_FPS="${VIDEO_FPS:-10}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=4.1

srun python eval/calvin/evaluate_calvin_close_drawer.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --calvin_root "$CALVIN_ROOT" \
  --dataset_path "$DATASET_PATH" \
  --num_episodes "$NUM_EPISODES" \
  --episode_len "$EPISODE_LEN" \
  --execute_step "$EXECUTE_STEP" \
  --video_fps "$VIDEO_FPS" \
  --save_video \
  --output_dir "$OUTPUT_DIR" \
  --device cuda:0

echo "Results + GIFs: $LFM4VLA_ROOT/$OUTPUT_DIR"
