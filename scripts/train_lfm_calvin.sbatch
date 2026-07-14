#!/bin/bash
#SBATCH --job-name=lfm4vla_calvin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=output_lfm4vla_%j.out
#SBATCH --error=output_lfm4vla_%j.err

set -euo pipefail

# --- edit these for your cluster layout ---
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-1.6b-finetune.json}"

# Optional: cache + logging (uncomment/set as needed)
# export HF_HOME=/lambdafs/$USER/hf_cache
# export WANDB_API_KEY=...
# export WANDB_MODE=online

cd "$LFM4VLA_ROOT"

# conda (recommended) — or swap for: source venv/bin/activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Lightning picks up SLURM GPU layout from srun
srun python main.py --config "$CONFIG"
