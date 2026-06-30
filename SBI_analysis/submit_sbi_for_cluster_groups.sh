#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_PBS="$SCRIPT_DIR/run_sbi_for_cluster.pbs"

: "${PREPARED_DATASET_PATH:=/home/kristero10/HalfDome_kSZ/SBI_runs/datasets/sbi_100k_uniform_prior_emulated_log10_dl.npz}"
: "${PREPARED_OBS_PATH:=/home/kristero10/HalfDome_kSZ/SBI_runs/datasets/x_obs_log10_dl.npy}"
: "${TMP_SUBMIT_DIR:=$SCRIPT_DIR/generated_group_pbs}"

if [ ! -f "$BASE_PBS" ]; then
  echo "Base PBS script not found: $BASE_PBS" >&2
  exit 2
fi
if [ ! -f "$PREPARED_DATASET_PATH" ]; then
  echo "Prepared dataset not found: $PREPARED_DATASET_PATH" >&2
  exit 2
fi
if [ ! -f "$PREPARED_OBS_PATH" ]; then
  echo "Prepared observation not found: $PREPARED_OBS_PATH" >&2
  exit 2
fi

mkdir -p "$TMP_SUBMIT_DIR"

for group_id in 1 2 3 4; do
  group_pbs="$TMP_SUBMIT_DIR/run_sbi_for_cluster_group${group_id}.pbs"
  cat > "$group_pbs" <<EOF
#PBS -N sbi_g${group_id}
#PBS -q mini
#PBS -l select=1:ncpus=26:mpiprocs=1:mem=128gb
#PBS -l walltime=04:00:00
#PBS -o /home/kristero10/logs/SBI_runs/
#PBS -e /home/kristero10/logs/SBI_runs/
#PBS -S /bin/bash

set -euo pipefail

export PREPARED_DATASET_PATH="$PREPARED_DATASET_PATH"
export PREPARED_OBS_PATH="$PREPARED_OBS_PATH"
export SBI_GROUP_ID="$group_id"

cd "\${PBS_O_WORKDIR:-$SCRIPT_DIR}"
bash "$BASE_PBS"
EOF

  echo "Submitting group $group_id with $group_pbs"
  qsub "$group_pbs"
done
