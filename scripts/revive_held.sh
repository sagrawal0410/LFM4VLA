#!/bin/bash
# Revive REQUEUE_HOLD training jobs (admin-held; user release denied):
# scancel + resubmit, resuming each job's OWN checkpoint lineage when one
# exists (same wandb run via run_meta), else a clean fresh start.
set -uo pipefail
cd "$HOME/LFM4VLA"
CK=/home/teams/research/robotics/checkpoints

config_for_task() {
  case "$1" in
    *waypointfc_8515_rand_mixed*) echo configs/rand-8515-mixed.json ;;
    *waypointfc_8515_rand_sep*)   echo configs/rand-8515-sep.json ;;
    *waypointfc_8515_hlr_mixed*)  echo configs/lfm2.5-vl-450m-robotnav-8515-mixed.json ;;
    *waypointfc_8515_hlr*)        echo configs/lfm2.5-vl-450m-robotnav-8515-hlr.json ;;
    *waypointfc_8515*)            echo configs/lfm2.5-vl-450m-robotnav-8515.json ;;
    *vla_adapter_8515_rand_mixed*) echo configs/rand-vla-adapter-mixed.json ;;
    *vla_adapter_8515_rand_sep*)  echo configs/rand-vla-adapter-sep.json ;;
    *vla_adapter_8515_hlr_mixed*) echo configs/lfm2.5-vl-450m-robotnav-vla-adapter-mixed.json ;;
    *vla_adapter_8515_hlr*)       echo configs/lfm2.5-vl-450m-robotnav-vla-adapter-hlr.json ;;
    *vla_adapter_8515*)           echo configs/lfm2.5-vl-450m-robotnav-vla-adapter.json ;;
    *mlp_8515_rand_mixed*)        echo configs/rand-mlp-mixed.json ;;
    *mlp_8515_rand_sep*)          echo configs/rand-mlp-sep.json ;;
    *mlp_8515_hlr_mixed*)         echo configs/lfm2.5-vl-450m-robotnav-mlp-mixed.json ;;
    *mlp_8515_hlr*)               echo configs/lfm2.5-vl-450m-robotnav-mlp-hlr.json ;;
    *mlp_8515*)                   echo configs/lfm2.5-vl-450m-robotnav-mlp.json ;;
    *groot_fm_rand_mixed*)        echo configs/rand-groot-fm-mixed.json ;;
    *groot_fm_rand_sep*)          echo configs/rand-groot-fm-sep.json ;;
    *groot_fm_hlr_mixed*)         echo configs/lfm2.5-vl-450m-robotnav-groot-fm-mixed.json ;;
    *groot_fm_hlr*)               echo configs/lfm2.5-vl-450m-robotnav-groot-fm-hlr.json ;;
    *groot_fm*)                   echo configs/lfm2.5-vl-450m-robotnav-groot-fm.json ;;
    *smolvla_fm_rand_mixed*)      echo configs/rand-smolvla-fm-mixed.json ;;
    *smolvla_fm_rand_sep*)        echo configs/rand-smolvla-fm-sep.json ;;
    *smolvla_fm_hlr_mixed*)       echo configs/lfm2.5-vl-450m-robotnav-smolvla-fm-mixed.json ;;
    *smolvla_fm_hlr*)             echo configs/lfm2.5-vl-450m-robotnav-smolvla-fm-hlr.json ;;
    *smolvla_fm*)                 echo configs/lfm2.5-vl-450m-robotnav-smolvla-fm.json ;;
    *) echo "" ;;
  esac
}

HELD=$(squeue -u "$USER" -h -o "%i %T" | awk '$2=="REQUEUE_HOLD" {print $1}')
[ -z "$HELD" ] && { echo "no held jobs"; exit 0; }
for J in $HELD; do
  TASK=$(grep -m1 '^run: ' "$HOME/LFM4VLA/output_robotnav_$J.out" 2>/dev/null | awk '{print $2}')
  CFG=$(config_for_task "$TASK")
  # The run name IS the lineage dir name (it keeps the original run id even
  # for jobs that were themselves RESUME= relaunches); job-id glob is fallback.
  CKPT=$(ls -t "$CK"/*/"$TASK"/last.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && CKPT=$(ls -t "$CK"/*/*sj${J}r*/last.ckpt 2>/dev/null | head -1)
  if [ -z "$CFG" ]; then
    echo "$J: task='$TASK' -> NO CONFIG MATCH, skipping (manual attention)"
    continue
  fi
  scancel "$J"
  if [ -n "$CKPT" ]; then
    NEW=$(sbatch --parsable --export=ALL,CONFIG=$CFG,RESUME=$CKPT scripts/train_robotnav.sbatch 2>/dev/null)
    echo "$J ($TASK) -> $NEW resume=$CKPT"
  else
    NEW=$(sbatch --parsable --export=ALL,CONFIG=$CFG scripts/train_robotnav.sbatch 2>/dev/null)
    echo "$J ($TASK) -> $NEW fresh"
  fi
done
