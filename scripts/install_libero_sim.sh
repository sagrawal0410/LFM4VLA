#!/usr/bin/env bash
# Install the LIBERO simulator stack for closed-loop eval (Option B: local workstation,
# e.g. an RTX 5090 desktop). This is SEPARATE from install_libero_rlds_deps.sh, which only
# sets up the TensorFlow RLDS *data* pipeline for training.
#
# Recommended: a fresh conda env so MuJoCo/robosuite pins don't collide with the TF
# training env.
#
# Usage:
#   conda create -n lfm4vla-libero-eval python=3.10 -y
#   conda activate lfm4vla-libero-eval
#   bash scripts/install_libero_sim.sh
#
# Then install a CUDA build of PyTorch that matches your GPU (Blackwell / RTX 5090 needs a
# recent CUDA 12.x wheel), e.g.:
#   pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
set -euo pipefail

echo "[1/5] Core sim deps (mujoco, robosuite)..."
# mujoco>=3.10 changed mj_fullM's signature and breaks robosuite 1.4.x controllers.
# 3.3.2 is the LIBERO-community pin (matches training-time rendering appearance).
pip install "mujoco==3.3.2" "robosuite==1.4.1" PyOpenGL PyOpenGL_accelerate

echo "[2/5] Headless rendering (EGL + software-EGL + OSMesa for SLURM nodes)..."
if command -v conda >/dev/null 2>&1; then
  conda install -y -c conda-forge mesalib libegl-devel glew || true
fi

echo "[3/5] LIBERO benchmark (clone + editable install)..."
# pip install git+https://... builds an empty ~5KB wheel (missing libero/__init__.py upstream).
# Clone, patch packaging, and editable-install so assets stay on disk.
LIBERO_SRC="${LIBERO_SRC:-${HOME}/.local/src/LIBERO}"
mkdir -p "$(dirname "$LIBERO_SRC")"
if [[ ! -d "$LIBERO_SRC/.git" ]]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_SRC"
else
  git -C "$LIBERO_SRC" pull --ff-only 2>/dev/null || true
fi
# Without this file, setuptools find_packages() skips the whole tree.
touch "$LIBERO_SRC/libero/__init__.py"
pip uninstall -y libero 2>/dev/null || true
pip install -e "$LIBERO_SRC" --no-cache-dir --config-settings editable_mode=compat 2>/dev/null \
  || pip install -e "$LIBERO_SRC" --no-cache-dir
pip install "bddl==1.0.1"

echo "[4/5] Video + model runtime deps..."
pip install imageio imageio-ffmpeg opencv-python-headless pillow einops
pip install "transformers>=4.46" accelerate safetensors

echo "[5/5] Quick import smoke test..."
python - <<'PY'
import importlib.util
import os
import yaml

# libero.libero calls input() on first import if ~/.libero/config.yaml is missing.
libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
config_file = os.path.join(libero_config_path, "config.yaml")
if not os.path.exists(config_file):
    spec = importlib.util.find_spec("libero.libero")
    benchmark_root = os.path.dirname(spec.origin)
    os.makedirs(libero_config_path, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump({
            "benchmark_root": benchmark_root,
            "bddl_files": os.path.join(benchmark_root, "bddl_files"),
            "init_states": os.path.join(benchmark_root, "init_files"),
            "datasets": os.path.normpath(os.path.join(benchmark_root, "../datasets")),
            "assets": os.path.join(benchmark_root, "assets"),
        }, f)
    print(f"Created LIBERO config at {config_file}")

import mujoco, robosuite  # noqa: F401
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
suite = d["libero_10"]()
print(f"OK: libero_10 has {suite.n_tasks} tasks; robosuite + mujoco import fine.")
PY

if ! python - <<'PY'
from libero.libero import benchmark
print("libero package:", benchmark.__file__)
PY
then
  echo "[ERROR] LIBERO import failed. Do NOT run: pip install libero"
  echo "        Re-run: bash scripts/install_libero_sim.sh"
  exit 1
fi

cat <<'EOF'

Done. Next steps on the workstation:

1. Make the trained checkpoint + its saved config available locally, e.g.:
     runs/logs/<date>/<exp>/<exp>-config.json
     runs/checkpoints/<date>/<exp>/last.ckpt

2. Base VLM path: if the saved config points at a missing cluster path, eval
   auto-falls back to the HuggingFace id from model_url (e.g. LiquidAI/LFM2.5-VL-450M)
   and loads architecture/tokenizer from the hub; finetuned weights come from the ckpt.

3. You need the RLDS dataset_statistics*.json used at training for action denormalization.
   Copy the file from the cluster, e.g.:
     <data_root_dir>/libero_10_no_noops/1.0.0/dataset_statistics_*.json
   and pass --data_root_dir pointing at its parent-of-parent, OR keep the same
   data_root_dir layout as the config.

4. Run (auto-picks EGL or OSMesa; use MUJOCO_GL=glfw if you have a display):
     python eval/libero/evaluate_libero.py \
       --config runs/logs/<date>/<exp>/<exp>-config.json \
       --ckpt   runs/checkpoints/<date>/<exp>/last.ckpt \
       --task_suite libero_10 \
       --num_trials_per_task 10 \
       --execute_step 5 \
       --save_video \
       --output_dir runs/libero_eval/<exp>

MP4s land in the output dir, one per episode, tagged -SUCC / -FAIL.
EOF
