#!/bin/bash
# Wait for the final nav-reasoning shard (max 3h), then submit the 4 runs.
cd "$HOME/LFM4VLA"
RND=/home/teams/research/robotics/robotnav_data
DEADLINE=$(( $(date +%s) + 3*3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  n=$(cd $RND && python3 scripts/reset_orphan_shards.py 2>/dev/null | grep -oP "nav_reasoning.*DONE.: \K[0-9]+")
  echo "[$(date +%H:%M:%S)] nav_reasoning DONE=$n/120"
  [ "${n:-0}" -ge 120 ] && { echo "ALL SHARDS DONE -> launching"; break; }
  sleep 300
done
[ "$(date +%s)" -ge "$DEADLINE" ] && echo "3h TIMEOUT -> launching with $(ls $RND/manifests/nav_reasoning/samples/*.jsonl | grep -vc partial) shards"
for f in mn256x16-molmo2_1_6b-mlp-navreason mn256x16-molmo2_1_6b-smolvla-navreason \
         mn256x16-ermix3-mlp-navreason mn256x16-ermix3-smolvla-navreason; do
  CONFIG=configs/$f.json sbatch --nodes=2 --ntasks-per-node=8 --gpus-per-node=8 scripts/train_robotnav_mn.sbatch
done
