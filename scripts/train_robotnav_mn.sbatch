#!/bin/bash
#SBATCH --job-name=lfm4vla_robotnav_mn
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=0
#SBATCH --time=96:00:00
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --output=output_robotnav_%j.out
#SBATCH --error=output_robotnav_%j.err
# Multi-node launcher (FSDP/DDP): geometry must match the config's
# devices x num_nodes. Defaults 4x8; override at submit time, e.g. a smoke:
#   CONFIG=... sbatch --nodes=2 --ntasks-per-node=2 --gpus-per-node=2 \
#     scripts/train_robotnav_mn.sbatch
# Same requeue/auto-resume/wandb semantics as the single-node launcher.
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

# MIOpen caches: set PER RANK inside the srun wrapper below — 8 ranks per
# node must not share one cache dir (the concurrent-access corruption).

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

# Per-rank MIOpen dir, then exec python (env vars expand per srun task).
# Placement made explicit: without it srun can pack tasks unevenly, giving
# some ranks SLURM_LOCALID >= devices-per-node (Lightning IndexError + the
# remaining ranks hanging in rendezvous until timeout).
# NCCL watchdog: the default 1800s collective timeout has killed six multi-day
# runs mid-training (DistBackendError + "Watchdog caught", no Python traceback).
# A slow-but-recovering collective should not cost a whole run.
export TORCH_NCCL_TIMEOUT="${TORCH_NCCL_TIMEOUT:-5400}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Auto-restart: on non-zero exit, resume from this run's own last.ckpt instead
# of losing the run. Slurm --requeue does not cover this: the job exits
# non-zero rather than being preempted.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
attempt=1
while : ; do
  echo "[train_robotnav] attempt $attempt/$MAX_ATTEMPTS $(date -Is)"
  set +e
  srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    bash -c 'export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}-${SLURM_PROCID}"; \
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"; \
    exec python main.py --config '"$CONFIG"' '"${RESUME_ARG[*]:-}"''
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then echo "[train_robotnav] finished cleanly"; break; fi
  attempt=$((attempt+1))
  if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "[train_robotnav] exhausted $MAX_ATTEMPTS attempts (last rc=$rc)"
    exit "$rc"
  fi
  LAST=$(ls -t "$OUT_ROOT"/*/"$TASK"-*sj"${SLURM_JOB_ID}"r*/last.ckpt 2>/dev/null | head -1 || true)
  if [ -n "$LAST" ]; then
    echo "[train_robotnav] rc=$rc -- auto-resuming from $LAST"
    RESUME_ARG=(--resume "$LAST")
  else
    echo "[train_robotnav] rc=$rc -- no checkpoint yet, restarting clean"
  fi
  sleep 60
done
