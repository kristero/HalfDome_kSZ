#!/bin/bash
# Background watcher: exits (so the session is notified) at the first event:
#   driver stopped | new staged results fetched | a rerun finished or its jobs ended without output | C: below 2 GB
D=$(cd "$(dirname "$0")" && pwd)
SSH="ssh.exe -o BatchMode=yes -o ConnectTimeout=60 idark"
L=sbi_reruns_20261001/SBI_analysis/tsz64k_sbi_local
n0=$(grep -c "fetched results" $D/driver.log)
conv0=${1:-0}; merlin0=${2:-0}                       # 1 = already reported
while true; do
  kill -0 "$(cat $D/driver.pid 2>/dev/null)" 2>/dev/null || { echo "EVENT driver not running"; break; }
  [ "$(grep -c "fetched results" $D/driver.log)" -gt "$n0" ] && { echo "EVENT staged results fetched"; tail -3 $D/driver.log | cut -c1-200; break; }
  tail -n 20 $D/driver.log | grep -q "stopping\|failed twice\|Traceback\|regen tasks failed" && { echo "EVENT driver problem"; tail -5 $D/driver.log; break; }
  free=$(timeout 30 powershell.exe -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1MB)" 2>/dev/null | tr -d '\r')
  [ -n "$free" ] && [ "$free" -lt 2000 ] && { echo "EVENT C: low: $free MB"; break; }
  st=$(timeout 120 $SSH "cd $L; c=0; m=0; [ -e results_final_noise/convergence.json ] && c=1; [ -e results_merlin_unbinned_asinh_p30/comparison.md ] && m=1; j=\$(qstat -u kristero10 2>/dev/null | grep -c 'cv_\|merlin_as'); echo \$c \$m \$j" < /dev/null 2>/dev/null | tr -d '\r')
  if [ -n "$st" ]; then
    set -- $st
    [ "$1" = 1 ] && [ "$conv0" = 0 ] && { echo "EVENT convergence rerun finished"; break; }
    [ "$2" = 1 ] && [ "$merlin0" = 0 ] && { echo "EVENT merlin rerun finished"; break; }
    [ "$3" = 0 ] && { [ "$1" = 0 ] || [ "$2" = 0 ]; } && { echo "EVENT rerun jobs ended without all outputs (conv $1, merlin $2)"; break; }
  fi
  sleep 300
done
date
