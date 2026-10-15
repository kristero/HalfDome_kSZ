#!/bin/bash
# nine training jobs (merlin with 8 threads, pca and moped with 4), then the evaluation once all nine succeeded
set -euo pipefail
cd /lustre/work/kristero10/merlin_unbinned_20260929
q=(mini mini_B); i=0; ids=()
for v in merlin pca moped; do
  t=4; [ $v = merlin ] && t=8
  for s in 1 2 3; do
    ids+=($(qsub -q ${q[$((i % 2))]} -l select=1:ncpus=$t:mem=16gb -o logs/ -N merlin_${v}_s$s -v VARIANT=$v,SEED=$s,THREADS=$t train.pbs))
    i=$((i + 1))
  done
done
dep=$(IFS=:; echo "${ids[*]}")
qsub -q mini -l select=1:ncpus=8:mem=24gb -o logs/ -W depend=afterok:$dep evaluate.pbs
printf '%s\n' "${ids[@]}"
