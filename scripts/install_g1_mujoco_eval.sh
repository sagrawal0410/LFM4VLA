#!/usr/bin/env bash
# Install MuJoCo eval deps into the Humanoid Everyday conda env (lfm4vla-he).
#
# Usage:
#   bash scripts/install_g1_mujoco_eval.sh
#   ENV_NAME=lfm4vla-he bash scripts/install_g1_mujoco_eval.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-lfm4vla-he}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== Installing G1 MuJoCo eval deps into ${ENV_NAME} ==="
pip install --no-cache-dir \
  "mujoco>=3.1.0" \
  "imageio[ffmpeg]" \
  "imageio-ffmpeg" \
  "pyyaml"

# Keep NumPy 1.x pin for HE wandb/TF-adjacent stack.
pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.26.4"

python - <<'PY'
import mujoco
import numpy as np
import yaml
import imageio
print("mujoco", mujoco.__version__)
print("numpy", np.__version__)
print("yaml", yaml.__version__)
print("imageio", imageio.__version__)
assert int(np.__version__.split(".", 1)[0]) < 2
print("ALL OK")
PY

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${ROOT}/third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml" ]]; then
  echo "[info] fetching Unitree G1 assets..."
  bash "${ROOT}/scripts/fetch_unitree_g1_mujoco.sh"
fi

echo "Done. Activate with: conda activate ${ENV_NAME}"
