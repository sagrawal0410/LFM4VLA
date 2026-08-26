#!/bin/bash
#SBATCH --job-name=lfm4vla_robotnav
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=400G
#SBATCH --time=96:00:00
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --output=output_robotnav_%j.out
#SBATCH --error=output_robotnav_%j.err
# Launch: CONFIG=configs/<robotnav config>.json sbatch scripts/train_robotnav.sbatch
# On preemption Slurm requeues the job; on restart we auto-resume from the
# newest last.ckpt written for this config's task_name (checkpoints every 1k).
set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
CONFIG="${CONFIG:?set CONFIG=configs/<file>.json}"

cd "$LFM4VLA_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-online}
# No credential -> offline logging instead of a crash (wandb sync later).
# Re-checked every (re)start: after a one-time `wandb login`, requeued
# attempts flip back to online automatically.
if [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "[train_robotnav] WARNING: no wandb credential; falling back to WANDB_MODE=offline"
  export WANDB_MODE=offline
fi
# Variable-size native-resolution image batches fragment glibc arenas over
# hours (observed RSS climb to the cgroup limit); cap arenas + trim eagerly.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=134217728

# MIOpen kernel caches must be node-local: the shared ~/.config/miopen db is
# corrupted by concurrent GPU jobs (filesystem_error / miopenStatusUnknownError).
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID:-$$}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Auto-resume: newest last.ckpt for this task (smoke runs never write one).
read -r TASK OUT_ROOT << EOF
$(python - "$CONFIG" << 'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(c["task_name"], c.get("output_root", ""))
PY
)
EOF
RESUME_ARG=()
# Semantics: every sbatch submission is a NEW experiment (step 0, randomized
# head, NEW wandb run). Auto-resume applies ONLY to Slurm requeues of this
# same job, and only from this job's OWN checkpoints (dir suffix sj<jobid>r*);
# a requeue before the first checkpoint restarts clean under a new wandb run.
# Deliberate continuation of an older lineage: RESUME=/path/to/last.ckpt sbatch ...
if [ -n "${RESUME:-}" ]; then
  echo "[train_robotnav] explicit resume from $RESUME"
  RESUME_ARG=(--resume "$RESUME")
else
  LAST=$(ls -t "$OUT_ROOT"/*/"$TASK"-*sj"${SLURM_JOB_ID}"r*/last.ckpt 2>/dev/null | head -1 || true)
  if [ -n "$LAST" ]; then
    echo "[train_robotnav] requeue — auto-resuming own checkpoint: $LAST"
    RESUME_ARG=(--resume "$LAST")
  else
    echo "[train_robotnav] fresh start (new wandb run)"
  fi
fi

srun python main.py --config "$CONFIG" "${RESUME_ARG[@]}"
