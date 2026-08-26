#!/bin/bash
#SBATCH --job-name=eval-wpm
#SBATCH --partition=gpuworker
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --qos=normal
#SBATCH --output=/home/teams/research/robotics/logs/eval_wpm_%A_%a.log
# Evaluates a list of (task|config|ckpt) triples given via EVAL_LIST file,
# one line per model; SLURM_ARRAY_TASK_ID picks a stripe (i % NTASKS == id).
set -uo pipefail
cd "$HOME/LFM4VLA"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lfm4vla
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
i=0
while IFS="|" read -r TASK CFG CKPT KMIN; do
  if [ $((i % ${NSTRIPES:-5})) -eq "${SLURM_ARRAY_TASK_ID:-0}" ]; then
    python scripts/eval_waypoint_metrics.py --task "$TASK" --config "$CFG" \
      --ckpt "$CKPT" --samples 256 --minade-k "${KMIN:-1}" \
      --out /home/teams/research/robotics/logs/eval_wpm_results.jsonl \
      || echo "[eval] FAILED $TASK"
  fi
  i=$((i+1))
done < "$EVAL_LIST"
echo EVAL_STRIPE_DONE
