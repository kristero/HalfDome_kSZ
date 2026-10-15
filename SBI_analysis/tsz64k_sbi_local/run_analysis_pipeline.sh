#!/bin/bash
# After the resumed Globus transfers: merge the 64,440 finished rows, upload the small arrays to idark, run the SBI
# convergence / correlation / noise study (sbi_final.py) and its report. Each step logs to logs_final/.
set -euo pipefail
cd /home/cbllover/HalfDome/SBI_analysis/tsz64k_sbi_local
export LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib
P=/home/cbllover/miniconda3/envs/halfdome/bin/python
G=/root/globus-venv/bin/globus
echo "$(date -u +%H:%M) prepare"; $P sbi_final.py prepare > logs_final/prepare.log 2>&1
echo "$(date -u +%H:%M) train"
pids=()
for v in bins40_noisy moped_noisy bins40_clean bins40_gbase bins40_ggoal; do
  $P sbi_final.py train --variants $v --workers 2 --threads 2 > logs_final/train_$v.log 2>&1 & pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done
echo "$(date -u +%H:%M) evaluate"
pids=()
for v in bins40_noisy moped_noisy bins40_clean bins40_gbase bins40_ggoal; do
  $P sbi_final.py evaluate --variants $v --threads 4 > logs_final/evaluate_$v.log 2>&1 & pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done
echo "$(date -u +%H:%M) report"; $P sbi_final.py report > logs_final/report.log 2>&1
echo "$(date -u +%H:%M) all done"
