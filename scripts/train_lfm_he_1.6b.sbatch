#!/bin/bash
#SBATCH --job-name=lfm4vla_he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm4vla_he_%j.out
#SBATCH --error=output_lfm4vla_he_%j.err

set -euo pipefail

# Humanoid Everyday fine-tune (1.6B) — G1 28-D.
# Env: bash scripts/install_humanoid_everyday_env.sh  →  conda activate lfm4vla-he

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla-he}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-1.6b-humanoid-everyday.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
print(f"[he] env={sys.prefix} numpy={np.__version__}")
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

srun python main.py --config "$CONFIG"
