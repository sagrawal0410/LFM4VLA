#!/bin/bash
#SBATCH --job-name=hist-probe-all
#SBATCH --partition=gpuworker
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH --output=/home/teams/research/robotics/logs/hist_probe_all_%A_%a.log
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lfm4vla
cd "$HOME/LFM4VLA"
export PYTHONPATH="$HOME/LFM4VLA"
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-hpa-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
LIST=/home/teams/research/robotics/logs/hist_probe_fm.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST")
TASK=$(echo "$LINE" | cut -d'|' -f1)
CFG=$(echo "$LINE" | cut -d'|' -f2)
CKPT=$(echo "$LINE" | cut -d'|' -f3)
echo "PROBE_TASK=$TASK"
python scripts/probe_history_influence.py --config "$CFG" --ckpt "$CKPT" --samples 48
