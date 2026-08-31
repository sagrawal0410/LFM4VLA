#!/bin/bash
# Escalation watchdog for the 4 nav-reasoning runs.
# Frees capacity by retiring the single-GPU molmo2 fleet ONLY if the runs are
# genuinely blocked on the account GPU cap. A code FAILURE is NOT a capacity
# problem, so it never triggers the kill -- it just gets logged for diagnosis.
LOG=/home/teams/research/robotics/logs/navreason_escalate.log
STUCK_MIN=45
stuck_since=0
exec >> "$LOG" 2>&1
echo "[$(date +%F\ %H:%M:%S)] watchdog armed (kill fleet after ${STUCK_MIN}m of AssocGrpGRES)"
while true; do
  ids=$(squeue -u "$USER" -h -t PD,R -o "%i %j" | awk "\$2==\"lfm4vla_robotnav_mn\"{print \$1}" | tr "\n" ",")
  ids=${ids%,}
  [ -z "$ids" ] && { sleep 300; continue; }
  pend=$(squeue -j "$ids" -h -t PD -o "%r" 2>/dev/null | grep -c AssocGrpGRES)
  run=$(squeue -j "$ids" -h -t R 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] mn jobs: $run running, $pend pending-on-quota"
  if [ "$pend" -gt 0 ]; then
    now=$(date +%s); [ "$stuck_since" -eq 0 ] && stuck_since=$now
    mins=$(( (now - stuck_since) / 60 ))
    if [ "$mins" -ge "$STUCK_MIN" ]; then
      victims=$(squeue -u "$USER" -h -t R -o "%i %j" | awk "\$2==\"lfm4vla_robotnav\"{print \$1}" | tr "\n" " ")
      n=$(echo $victims | wc -w)
      if [ "$n" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] ESCALATING: ${mins}m blocked on quota -> cancelling $n single-GPU runs"
        echo "  ids: $victims"
        scancel $victims
        echo "[$(date +%H:%M:%S)] fleet retired; checkpoints preserved (resumable via RESUME=)"
      fi
      exit 0
    fi
  else
    stuck_since=0
  fi
  [ "$run" -ge 4 ] && { echo "[$(date +%H:%M:%S)] all 4 running - watchdog standing down"; exit 0; }
  sleep 300
done
