#!/bin/bash
#SBATCH --job-name=libero_rlds_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=48:00:00
#SBATCH --output=output_libero_rlds_depth_%j.out
#SBATCH --error=output_libero_rlds_depth_%j.err

# Build LIBERO-10 RLDS with GT agentview depth at:
#   /home/teams/research/robotics/modified_libero_rlds_depth
#
# Usage:
#   sbatch scripts/build_libero_rlds_depth.sbatch
#
# Then launch the 8-run training sweep (1 GPU per array task):
#   sbatch --dependency=afterok:<JOBID> \
#     --export=ALL,CONFIG=configs/lfm2.5-vl-450m-libero-depth-latent1.json \
#     scripts/train_lfm_libero_depth_sweep.sbatch

set -euo pipefail

LFM4VLA_ROOT="${LFM4VLA_ROOT:-$HOME/LFM4VLA}"
CONDA_ENV_SIM="${CONDA_ENV_SIM:-lfm4vla}"
CONDA_ENV_RLDS="${CONDA_ENV_RLDS:-${CONDA_ENV_SIM}}"

DATA_ROOT="${DATA_ROOT:-/home/teams/research/robotics}"
RAW_ROOT="${RAW_ROOT:-$DATA_ROOT/libero_raw}"
RAW_DIR="${RAW_DIR:-$RAW_ROOT/LIBERO-datasets}"
HDF5_DEPTH_DIR="${HDF5_DEPTH_DIR:-$DATA_ROOT/libero_hdf5_depth/libero_10_no_noops}"
RLDS_OUT="${RLDS_OUT:-$DATA_ROOT/modified_libero_rlds_depth}"
SUITE="${SUITE:-libero_10}"

mkdir -p "$RAW_ROOT" "$HDF5_DEPTH_DIR" "$RLDS_OUT"
cd "$LFM4VLA_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_SIM"

export PYTHONUNBUFFERED=1
# "auto" is our probe mode (utils/mujoco_gl.py), NOT a valid MuJoCo value.
# Do not export MUJOCO_GL=auto — mujoco rejects it on import. Pass via --mujoco_gl.
MUJOCO_GL_MODE="${MUJOCO_GL:-auto}"
# Clear invalid leftover from prior jobs / interactive shells.
if [[ "${MUJOCO_GL:-}" == "auto" ]]; then
  unset MUJOCO_GL
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# Help mesa software paths find DRI drivers.
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-${CONDA_PREFIX}/lib/dri}"

echo "=== LIBERO depth RLDS build ==="
echo "job=${SLURM_JOB_ID:-local}"
echo "RAW_DIR=$RAW_DIR"
echo "HDF5_DEPTH_DIR=$HDF5_DEPTH_DIR"
echo "RLDS_OUT=$RLDS_OUT"
echo "MUJOCO_GL_MODE=$MUJOCO_GL_MODE CONDA_PREFIX=$CONDA_PREFIX"

# Headless GL libs (safe no-op if already installed).
conda install -y -c conda-forge "mesalib<=25.1.0" libegl-devel glew || true
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LIBGL_DRIVERS_PATH="${CONDA_PREFIX}/lib/dri:${LIBGL_DRIVERS_PATH:-}"

# robosuite 1.4.x breaks on mujoco>=3.1.0 (mj_fullM signature change).
# Pin to the LIBERO-community versions used by scripts/install_libero_sim.sh.
# Use importlib.metadata (not import mujoco) so we never hit MUJOCO_GL validation.
python - <<'PY'
import importlib.metadata as m
def ver(pkg):
    try:
        return m.version(pkg)
    except m.PackageNotFoundError:
        return None
print(f"mujoco version before pin: {ver('mujoco')}")
PY
pip install -q "mujoco==3.3.2" "robosuite==1.4.1"
# mujoco/pip may pull NumPy 2.x; TF / tfds / dlimp require NumPy 1.x.
python - <<'PY'
import importlib.metadata as m
import subprocess
import sys

def ver(pkg):
    try:
        return m.version(pkg)
    except m.PackageNotFoundError:
        return "0"

v = ver("numpy")
if int(v.split(".", 1)[0]) >= 2:
    print(f"[libero] NumPy {v} incompatible with TF; pinning 1.26.4", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "numpy==1.26.4"]
    )
    v = ver("numpy")
print(f"mujoco={ver('mujoco')} robosuite={ver('robosuite')} numpy={v}", flush=True)
if int(v.split(".", 1)[0]) >= 2:
    raise SystemExit("NumPy still >=2 after pin; fix the conda env manually.")
PY

# ---------------------------------------------------------------------------
# 1) Download raw LIBERO HDF5
# ---------------------------------------------------------------------------
if [[ -z "$(ls -A "$RAW_DIR/$SUITE"/*.hdf5 2>/dev/null || true)" ]]; then
  echo "[1/3] Downloading yifengzhu-hf/LIBERO-datasets/$SUITE → $RAW_DIR"
  # Newer huggingface_hub removed working `huggingface-cli`; prefer `hf` or Python API.
  pip install -q -U "huggingface_hub>=0.26"
  # Only pull the suite we need (libero_10), not the full ~100GB dump.
  if command -v hf >/dev/null 2>&1; then
    hf download yifengzhu-hf/LIBERO-datasets \
      --repo-type dataset \
      --include "${SUITE}/*" \
      --local-dir "$RAW_DIR"
  else
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    local_dir="${RAW_DIR}",
    allow_patterns=["${SUITE}/*"],
)
print("download complete")
PY
  fi
else
  echo "[1/3] Raw HDF5 already present: $RAW_DIR/$SUITE — skip download"
fi

if [[ -z "$(ls -A "$RAW_DIR/$SUITE"/*.hdf5 2>/dev/null || true)" ]]; then
  echo "ERROR: no demos at $RAW_DIR/$SUITE/*.hdf5" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2) Replay with depth → HDF5
# ---------------------------------------------------------------------------
if [[ -z "$(ls -A "$HDF5_DEPTH_DIR"/*.hdf5 2>/dev/null || true)" ]]; then
  echo "[2/3] Regenerating HDF5 with agentview_depth → $HDF5_DEPTH_DIR"
  # Do not use `env -u` — some nodes resolve a broken ~/.local/bin/env.
  unset MUJOCO_GL || true
  srun python scripts/regenerate_libero_hdf5_with_depth.py \
    --libero_task_suite "$SUITE" \
    --libero_raw_data_dir "$RAW_DIR/$SUITE" \
    --libero_target_dir "$HDF5_DEPTH_DIR" \
    --mujoco_gl "$MUJOCO_GL_MODE"
else
  echo "[2/3] Depth HDF5 already present: $HDF5_DEPTH_DIR — skip regenerate"
fi

# ---------------------------------------------------------------------------
# 3) HDF5 → RLDS
# ---------------------------------------------------------------------------
echo "[3/3] Converting to RLDS → $RLDS_OUT/libero_10_no_noops"
conda activate "$CONDA_ENV_RLDS"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TFDS_DATA_DIR="$RLDS_OUT"
export LIBERO_DEPTH_HDF5_GLOB="$HDF5_DEPTH_DIR/*.hdf5"

# Re-pin in case this env differs from CONDA_ENV_SIM or pip drifted.
python - <<'PY'
import importlib.metadata as m
import subprocess
import sys

def ver(pkg):
    try:
        return m.version(pkg)
    except m.PackageNotFoundError:
        return "0"

v = ver("numpy")
if int(v.split(".", 1)[0]) >= 2:
    print(f"[libero] NumPy {v} incompatible with TF; pinning 1.26.4", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "numpy==1.26.4"]
    )
    v = ver("numpy")
print(f"[libero] numpy={v}", flush=True)
if int(v.split(".", 1)[0]) >= 2:
    raise SystemExit("NumPy still >=2 after pin; fix the conda env manually.")
PY

python - <<'PY'
import importlib
for m in ("tensorflow", "tensorflow_datasets", "h5py", "numpy"):
    importlib.import_module(m)
print("TFDS deps OK")
PY

BUILDER_DIR="$LFM4VLA_ROOT/scripts/rlds_builders/libero_10_no_noops"
cd "$BUILDER_DIR"
# Avoid `srun env` — some nodes resolve a broken ~/.local/bin/env.
srun tfds build --overwrite

echo "=== DONE ==="
echo "RLDS root: $RLDS_OUT"
find "$RLDS_OUT" -maxdepth 3 -type d | head -40
echo
echo "Next — 8-run depth sweep (1 GPU per array index 0..7):"
echo "  sbatch --dependency=afterok:${SLURM_JOB_ID} \\"
echo "    --export=ALL,CONFIG=configs/lfm2.5-vl-450m-libero-depth-latent1.json \\"
echo "    scripts/train_lfm_libero_depth_sweep.sbatch"
