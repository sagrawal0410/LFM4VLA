#!/bin/bash
#SBATCH --job-name=eval-rollout
#SBATCH --partition=gpuworker
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --output=/home/teams/research/robotics/logs/eval_rollout_%j.log
# Closed-loop Habitat rollouts with videos (cluster: CPU render + CPU policy).
# CONFIG=... CKPT=... SUITE=vlnce_r2r SPLIT=val_seen N_EP=3 OUT=... sbatch ...
set -uo pipefail
LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
RND=/home/teams/research/robotics/robotnav_data
cd "$LFM4VLA_ROOT"
export LFM4VLA_ROOT
export LFM4VLA_PYTHON="$HOME/miniconda3/envs/lfm4vla/bin/python"
export PYTHONUNBUFFERED=1
OUT="${OUT:-/home/teams/research/robotics/logs/rollouts_${SLURM_JOB_ID}}"
# habitat env + software-EGL render environment (from robotnav_data), then run
# inside the mount namespace that hides /dev/dri (=> policy must be CPU here).
source "$RND/scripts/render_env.sh"
export PYTHONPATH="$LFM4VLA_ROOT:${PYTHONPATH:-}"
unshare -Urm bash -c "mount -t tmpfs none /dev/dri 2>/dev/null || true; \
  python -m eval.robotnav_sim.run --suite ${SUITE:-vlnce_r2r} \
    --split ${SPLIT:-val_seen} --episodes ${N_EP:-3} \
    --config ${CONFIG:?} --ckpt ${CKPT:?} --out $OUT \
    --policy-device cpu --max-steps ${MAX_STEPS:-200} \
    ${EP_IDS:+--episode-ids $EP_IDS} ${EASY:+--easy} \
    ${MAX_INSTR:+--max-instr $MAX_INSTR} \
    ${HIST_MODE:+--history-mode $HIST_MODE} ${HIST_ALPHA:+--history-alpha $HIST_ALPHA} ${HIST_JITTER:+--history-jitter} ${REPLAN:+--replan-every $REPLAN}"
echo "EVAL_ROLLOUT_DONE out=$OUT"
