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

echo "[1/4] Core sim deps (mujoco, robosuite)..."
pip install "mujoco>=3.1" "robosuite==1.4.1"

echo "[2/4] LIBERO benchmark (from GitHub)..."
# LIBERO is not on PyPI; install from source. Pin to a commit if you need reproducibility.
pip install "git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"

echo "[3/4] Video + model runtime deps..."
pip install imageio imageio-ffmpeg opencv-python-headless pillow einops
pip install "transformers>=4.46" accelerate safetensors

echo "[4/4] Quick import smoke test..."
python - <<'PY'
import mujoco, robosuite  # noqa: F401
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
suite = d["libero_10"]()
print(f"OK: libero_10 has {suite.n_tasks} tasks; robosuite + mujoco import fine.")
PY

cat <<'EOF'

Done. Next steps on the workstation:

1. Make the trained checkpoint + its saved config available locally, e.g.:
     runs/logs/<date>/<exp>/<exp>-config.json
     runs/checkpoints/<date>/<exp>/last.ckpt

2. You still need the LFM VLM weights the config points at (vlm.model_id /
   tokenizer.pretrained_model_name_or_path). Either copy that local checkpoint dir over,
   or edit the config to a HuggingFace id you can download.

3. You need the RLDS dataset_statistics*.json used at training for action denormalization.
   Copy the file from the cluster, e.g.:
     <data_root_dir>/libero_10_no_noops/1.0.0/dataset_statistics_*.json
   and pass --data_root_dir pointing at its parent-of-parent, OR keep the same
   data_root_dir layout as the config.

4. Run (EGL offscreen rendering; use MUJOCO_GL=glfw if you have a display and prefer it):
     MUJOCO_GL=egl python eval/libero/evaluate_libero.py \
       --config runs/logs/<date>/<exp>/<exp>-config.json \
       --ckpt   runs/checkpoints/<date>/<exp>/last.ckpt \
       --task_suite libero_10 \
       --num_trials_per_task 10 \
       --execute_step 5 \
       --save_video \
       --output_dir runs/libero_eval/<exp>

MP4s land in the output dir, one per episode, tagged -SUCC / -FAIL.
EOF
