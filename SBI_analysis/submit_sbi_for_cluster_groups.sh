#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_PBS="$SCRIPT_DIR/run_sbi_for_cluster.pbs"

: "${PREPARED_DATASET_PATH:=}"
: "${PREPARED_OBS_PATH:=}"
: "${SBI_GAUSSIAN_BEAM_FWHM_ARCMIN:=0.0}"
: "${SBI_GAUSSIAN_BEAM_MODE:=off}"
: "${TMP_SUBMIT_DIR:=$SCRIPT_DIR/generated_group_pbs}"

if [ -z "$PREPARED_DATASET_PATH" ]; then
  for candidate in \
    "$SCRIPT_DIR/data_for_cluster/emulated_so_binned_beam2arcmin_sbi_run.npz" \
    "$SCRIPT_DIR/data_for_cluster/prepared_cluster_sbi_run.npz" \
    "$SCRIPT_DIR/data_for_cluster/planck_binned16_no_noise_sbi_run.npz" \
    "/lustre/work/kristero10/tSZ_data/sbi_battaglia9_full_ell/sbi_prepared_dataset.npz" \
    "$SCRIPT_DIR/data_for_cluster/_sbi_run.npz"
  do
    if [ -f "$candidate" ]; then
      PREPARED_DATASET_PATH=$candidate
      break
    fi
  done
fi
PREPARE_SBI_DATASET_CANDIDATE=/lustre/work/kristero10/tSZ_data/sbi_battaglia9_full_ell/sbi_prepared_dataset.npz
PREPARE_SBI_OBS_CANDIDATE=/lustre/work/kristero10/tSZ_data/sbi_battaglia9_full_ell/observed_log10_dl.npy
if [ -z "$PREPARED_OBS_PATH" ] && [ "$PREPARED_DATASET_PATH" = "$PREPARE_SBI_DATASET_CANDIDATE" ] && [ -f "$PREPARE_SBI_OBS_CANDIDATE" ]; then
  PREPARED_OBS_PATH=$PREPARE_SBI_OBS_CANDIDATE
fi

if [ ! -f "$BASE_PBS" ]; then
  echo "Base PBS script not found: $BASE_PBS" >&2
  exit 2
fi
if [ ! -f "$PREPARED_DATASET_PATH" ]; then
  echo "Prepared dataset not found: $PREPARED_DATASET_PATH" >&2
  exit 2
fi
if [ -n "$PREPARED_OBS_PATH" ] && [ ! -f "$PREPARED_OBS_PATH" ]; then
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
export SBI_GAUSSIAN_BEAM_FWHM_ARCMIN="$SBI_GAUSSIAN_BEAM_FWHM_ARCMIN"
export SBI_GAUSSIAN_BEAM_MODE="$SBI_GAUSSIAN_BEAM_MODE"

cd "\${PBS_O_WORKDIR:-$SCRIPT_DIR}"
bash "$BASE_PBS"
EOF

  echo "Submitting group $group_id with $group_pbs"
  qsub "$group_pbs"
done
