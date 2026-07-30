#!/usr/bin/env bash
# Create a dedicated conda env for Humanoid Everyday fine-tuning / token sweeps.
#
# Why a separate env: HE training pins NumPy 1.26.x (wandb 0.16 / TF-adjacent pins).
# Keep ``lfm4vla`` free for LIBERO/depth/other work that may need NumPy 2.x.
#
# Usage (login node):
#   bash scripts/install_humanoid_everyday_env.sh
#
# Then download data *inside* this env:
#   conda activate lfm4vla-he
#   python scripts/download_humanoid_everyday.py \
#       --output_dir /home/teams/research/robotics/humanoid_everyday
#
# Train:
#   sbatch scripts/train_lfm_he_token_sweep.sbatch   # CONDA_ENV defaults to lfm4vla-he
set -euo pipefail

ENV_NAME="${ENV_NAME:-lfm4vla-he}"
CLONE_FROM="${CLONE_FROM:-lfm4vla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "=== Humanoid Everyday env: ${ENV_NAME} ==="

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[info] env '${ENV_NAME}' already exists — activating and re-pinning deps"
  conda activate "$ENV_NAME"
else
  if conda env list | awk '{print $1}' | grep -qx "$CLONE_FROM"; then
    echo "[1/3] Cloning '${CLONE_FROM}' -> '${ENV_NAME}' (keeps AMD/ROCm torch match)"
    conda create -y -n "$ENV_NAME" --clone "$CLONE_FROM"
  else
    echo "[1/3] '${CLONE_FROM}' not found; creating fresh Python ${PYTHON_VERSION} env"
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
  fi
  conda activate "$ENV_NAME"
fi

echo "[2/3] Installing HE runtime deps + NumPy 1.26 pin..."
# HE is map-style parquet/MP4 — no TensorFlow / RLDS required.
pip install --no-cache-dir \
  "numpy==1.26.4" \
  "wandb>=0.17.2" \
  "av" \
  "pyarrow" \
  "huggingface_hub" \
  "pillow" \
  "einops" \
  "omegaconf" \
  "safetensors" \
  "accelerate" \
  "transformers>=4.46" \
  "lightning" \
  "opencv-python-headless"

# Re-pin NumPy last in case a dep pulled 2.x.
pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.26.4"

echo "[3/3] Verifying..."
python - <<'PY'
import numpy as np
import wandb
import av  # noqa: F401
import pyarrow  # noqa: F401
import lightning  # noqa: F401
import torch
import transformers

maj = int(np.__version__.split(".", 1)[0])
assert maj < 2, f"NumPy must be 1.x, got {np.__version__}"
print("numpy", np.__version__)
print("wandb", wandb.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("lightning", lightning.__version__)
print("ALL OK")
PY

cat <<EOF

Done. Activate with:
  conda activate ${ENV_NAME}

Next:
  1) Download the 4-skill G1 subset (if not already):
       # RGB-only
       python scripts/download_humanoid_everyday.py \\
         --output_dir /home/teams/research/robotics/humanoid_everyday
       # RGB + egocentric depth (separate tree; large parquets)
       python scripts/download_humanoid_everyday.py \\
         --output_dir /home/teams/research/robotics/humanoid_everyday_depth \\
         --include_depth
       # or: sbatch scripts/download_humanoid_everyday_depth.sbatch

  2) Train / sweep (sbatch defaults CONDA_ENV=${ENV_NAME}):
       sbatch scripts/train_lfm_he_450m.sbatch
       sbatch scripts/train_lfm_he_token_sweep.sbatch
       sbatch scripts/train_lfm_he_depth_token_sweep.sbatch   # depth Q-Former q=32

Leave env '${CLONE_FROM}' alone for NumPy-2 / other experiments.
EOF
