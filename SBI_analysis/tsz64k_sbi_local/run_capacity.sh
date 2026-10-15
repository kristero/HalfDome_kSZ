#!/bin/bash
# Capacity control for the convergence study: the 40-band SO-baseline features at N = 61,440 with a larger MAF and with
# an NSF (3 seeds each, early stopping as before). If these match the 128x5 MAF, the curve in fig_convergence is set by
# the training-set size, not by the network.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
export LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib
$PY train_ensemble.py train --out results_final/bins40_noisy --kinds N61440_maf256x8 --features-from N61440 --seeds 1 2 3 \
    --workers 3 --threads 3 --hidden 256 --transforms 8 --model maf > logs_final/capacity_maf256x8.log 2>&1 &
$PY train_ensemble.py train --out results_final/bins40_noisy --kinds N61440_nsf128x5 --features-from N61440 --seeds 1 2 3 \
    --workers 3 --threads 3 --hidden 128 --transforms 5 --model nsf --bins 10 > logs_final/capacity_nsf128x5.log 2>&1 &
wait
echo "capacity runs done $(date -u +%H:%M)"
