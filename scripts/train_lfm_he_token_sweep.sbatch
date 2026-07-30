#!/bin/bash
#SBATCH --job-name=lfm4vla_he_tok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --array=0-34
#SBATCH --output=output_lfm4vla_he_tok_%A_%a.out
#SBATCH --error=output_lfm4vla_he_tok_%A_%a.err

# 35-run Humanoid Everyday sweep:
#   num_action_tokens ∈ {1,2,4,8,16,24,28}
#   latent            ∈ {1,2,4,8,16}   (per-token repeat)
#
# Single continuous 28-D head (FCContinuousDecoder): mlp + MLPTanhHead, no gripper.
#
# Usage:
#   sbatch scripts/train_lfm_he_token_sweep.sbatch
#   CONFIG=configs/lfm2.5-vl-1.6b-humanoid-everyday.json sbatch scripts/train_lfm_he_token_sweep.sbatch
#   # subset: sbatch --array=0-4 scripts/train_lfm_he_token_sweep.sbatch
#
# One-time env pin (HE download can pull NumPy 2.x, which breaks older wandb):
#   pip install "numpy==1.26.4"

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-humanoid-everyday.json}"

cd "$LFM4VLA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Guard: wandb + TF stack in this env expect NumPy 1.x (np.float_ removed in 2.0).
python - <<'PY'
import numpy as np
maj = int(np.__version__.split(".", 1)[0])
if maj >= 2:
    raise SystemExit(
        f"NumPy {np.__version__} is incompatible with this env's wandb/TF pins.\n"
        "  pip install 'numpy==1.26.4'\n"
        "Then re-submit the job."
    )
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Resolve (num_action_tokens, latent) from the SLURM array index, unless overridden.
if [[ -z "${NUM_ACTION_TOKENS:-}" || -z "${LATENT:-}" ]]; then
  eval "$(python scripts/sweep_he_tokens.py --index "${SLURM_ARRAY_TASK_ID:?}")"
fi
TASK_NAME="${TASK_NAME:-he_tok${NUM_ACTION_TOKENS}_lat${LATENT}}"

echo "=== HE token sweep ==="
echo "array_id=${SLURM_ARRAY_TASK_ID:-n/a}  tokens=${NUM_ACTION_TOKENS}  latent=${LATENT}  task=${TASK_NAME}"
echo "config=${CONFIG}"

srun python main.py \
  --config "$CONFIG" \
  --num_action_tokens "$NUM_ACTION_TOKENS" \
  --latent "$LATENT" \
  --task_name "$TASK_NAME"
