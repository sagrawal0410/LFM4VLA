#!/usr/bin/env bash
# Submit the 8-run LIBERO depth sweep (latent × qformer queries), 1 GPU each.
#
# Usage:
#   bash scripts/submit_libero_depth_sweep.sh
#   CONFIG=configs/lfm2.5-vl-1.6b-libero-depth-latent1.json bash scripts/submit_libero_depth_sweep.sh
#   DEPENDENCY=12345 bash scripts/submit_libero_depth_sweep.sh   # afterok on data-build job

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONFIG="${CONFIG:-configs/lfm2.5-vl-450m-libero-depth-latent1.json}"
CONDA_ENV="${CONDA_ENV:-lfm4vla}"
DEPENDENCY="${DEPENDENCY:-}"

cd "$LFM4VLA_ROOT"

EXTRA=()
if [[ -n "$DEPENDENCY" ]]; then
  EXTRA+=(--dependency="afterok:${DEPENDENCY}")
fi

# IMPORTANT: do NOT use --export=ALL. Login-shell PATH/CONDA_*/CUDA_* vars often
# break the job on the compute node before any useful logging.
sbatch "${EXTRA[@]}" \
  --export=NONE,LFM4VLA_ROOT="$LFM4VLA_ROOT",CONFIG="$CONFIG",CONDA_ENV="$CONDA_ENV",OUTPUT_ROOT="${OUTPUT_ROOT:-/home/teams/research/robotics/checkpoints}",LOG_ROOT="${LOG_ROOT:-/home/teams/research/robotics/logs}",CACHE_ROOT="${CACHE_ROOT:-/home/teams/research/robotics/cache}" \
  scripts/train_lfm_libero_depth_sweep.sbatch

echo "Submitted depth sweep with CONFIG=$CONFIG"
python scripts/sweep_libero_depth.py --list
echo
echo "Logs (from submit cwd): output_lfm4vla_libero_depth_swp_<JOBID>_<0-7>.{out,err}"
echo "Check:  sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,NodeList -P"
