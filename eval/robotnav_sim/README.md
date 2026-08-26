# RobotNav closed-loop visual eval suite (Habitat-Sim)

Rolls trained LFM4VLA checkpoints out **closed-loop** in Habitat-Sim, records
an **MP4 video per episode** (HUD: instruction / step / action / distance),
and reports NE / SR / OS / SPL per episode + aggregate.

Two-process design (habitat-sim needs py3.9, the models need py3.10):
`run.py` (habitat env) drives the sim and spawns `policy_server.py`
(lfm4vla env) over a stdio JSON pipe.

## Suites
| suite | scenes | splits | status |
|---|---|---|---|
| `vlnce_r2r` | MP3D | `train` (trained scenes), `val_seen`, `val_unseen` (held-out) | ready |
| `vlnce_rxr` | MP3D | same (English episodes only) | ready |
| `objectnav_hm3d` | HM3D | `train`, `val` | ready (set `ROBOTNAV_OBJNAV_HM3D`) |
| `objectnav_mp3d` | MP3D | `train`, `val` | ready (set `ROBOTNAV_OBJNAV_MP3D`) |
| `hm3d_ovon` / `evt_bench` / `hm_eqa` / `mt_hm3d` / `express_bench` | — | — | stubs; `run.py` prints wiring notes |

## Desktop setup (recommended: real GPU GL + GPU policy)
```bash
# 1) habitat env
conda create -n robotnav-sim python=3.9 -y
conda install -n robotnav-sim habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat -y
conda run -n robotnav-sim pip install imageio imageio-ffmpeg pillow numpy quaternion numpy-quaternion
# 2) model env: your existing lfm4vla env (torch + transformers + lightning)
# 3) data + paths (rsync scenes/episodes/checkpoints from the cluster):
export ROBOTNAV_MP3D=/data/scene_datasets/mp3d
export ROBOTNAV_HM3D=/data/scene_datasets/hm3d
export ROBOTNAV_VLNCE_R2R=/data/vlnce_r2r/R2R_VLNCE_v1-3
export ROBOTNAV_VLNCE_RXR=/data/vlnce_rxr/RxR_VLNCE_v0
export LFM4VLA_PYTHON=$HOME/miniconda3/envs/lfm4vla/bin/python
export LFM4VLA_ROOT=$HOME/LFM4VLA
```

## Run
```bash
cd $LFM4VLA_ROOT
conda activate robotnav-sim
python -m eval.robotnav_sim.run \
  --suite vlnce_r2r --split val_unseen --episodes 10 \
  --config configs/lfm2.5-vl-450m-robotnav-mlp.json \
  --ckpt  /path/to/that/run/last.ckpt \
  --out results/r2r_unseen_mlp --policy-device cuda
open results/r2r_unseen_mlp/ep*.mp4
```
- `--split train` rolls out in **trained scenes**; `val_unseen` is held-out.
- Turn granularity auto-matches the family (R2R 15°, RxR/ObjectNav 30°).
- Stopping: the policy emits waypoints only; STOP fires when the predicted
  final waypoint stays within 0.24 m (tune `--max-steps`, controller consts).

## Cluster (headless CPU rendering; policy on CPU)
The compute nodes render via Mesa llvmpipe inside a mount namespace that
hides `/dev/dri` — which also hides the GPU from ROCm, so on the cluster the
policy server runs on CPU (slow: ~10–20 s/step; fine for a few videos).
Use `scripts/eval_rollout.sbatch`:
```bash
CONFIG=configs/... CKPT=/path/last.ckpt SUITE=vlnce_r2r SPLIT=val_seen N_EP=3 \
  sbatch scripts/eval_rollout.sbatch
```
For full sweeps, prefer the desktop.

## Metrics
NE (geodesic, m) · SR (NE < 3 m VLN / < 1 m ObjectNav) · OS (oracle success
along the path) · SPL · path length · steps · stopped-by-policy flag.
Aggregate + per-episode JSON at `<out>/metrics.json`; videos `<out>/ep*.mp4`.
