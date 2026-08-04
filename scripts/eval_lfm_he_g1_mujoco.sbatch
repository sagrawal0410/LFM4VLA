#!/bin/bash
#SBATCH --job-name=lfm4vla_he_g1eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm4vla_he_g1eval_%j.out
#SBATCH --error=output_lfm4vla_he_g1eval_%j.err

# Closed-loop G1 MuJoCo eval for HE tok2/lat16 checkpoint.
#
# Prereqs (once):
#   bash scripts/install_humanoid_everyday_env.sh
#   bash scripts/install_g1_mujoco_eval.sh
#
# Usage:
#   sbatch scripts/eval_lfm_he_g1_mujoco.sbatch
#   CKPT=/path/to/he_tok2_lat16-... SUITE=in sbatch scripts/eval_lfm_he_g1_mujoco.sbatch

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
CKPT_ROOT="${CKPT_ROOT:-/home/teams/research/robotics/checkpoints}"
CKPT="${CKPT:-${CKPT_ROOT}/he_tok2_lat16-lfm450m-bs16x4-lr2e-05-ws1-FCContinuousDecoder-lat16-20260730-214623-895e10}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-humanoid-everyday.json}"
DATA_ROOT="${DATA_ROOT:-/home/teams/research/robotics/humanoid_everyday}"
SUITE="${SUITE:-all}"
EPISODES="${EPISODES:-5}"
NUM_ACTION_TOKENS="${NUM_ACTION_TOKENS:-2}"
LATENT="${LATENT:-16}"
MUJOCO_GL="${MUJOCO_GL:-auto}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Ensure robot assets exist.
if [[ ! -f third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml ]]; then
  bash scripts/fetch_unitree_g1_mujoco.sh
fi

python - <<'PY'
import numpy as np, sys
if int(np.__version__.split(".", 1)[0]) >= 2:
    raise SystemExit(
        f"Env '{sys.prefix}' has NumPy {np.__version__}. Use lfm4vla-he."
    )
import mujoco
print(f"[he-g1] env={sys.prefix} numpy={np.__version__} mujoco={mujoco.__version__}")
PY

OUT_DIR="${OUT_DIR:-runs/g1_eval/$(basename "$CKPT")}"
echo "=== HE G1 MuJoCo eval ==="
echo "ckpt=${CKPT}"
echo "tokens=${NUM_ACTION_TOKENS} latent=${LATENT} suite=${SUITE} eps=${EPISODES}"
echo "out=${OUT_DIR}"

srun python eval/humanoid/evaluate_g1.py \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --num_action_tokens "$NUM_ACTION_TOKENS" \
  --latent "$LATENT" \
  --data_root_dir "$DATA_ROOT" \
  --suite "$SUITE" \
  --episodes_per_task "$EPISODES" \
  --mujoco_gl "$MUJOCO_GL" \
  --output_dir "$OUT_DIR" \
  --save_video
