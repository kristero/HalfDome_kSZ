#!/bin/bash
# Five ordinary jobs: preparation -> three independent MAF jobs -> summary.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${PYTHON:=python3}"
: "${COMPRESSION_ROOT:=/lustre/work/kristero10/adrian_9param_compression_baseline_deproj0}"
DATA_DIR="${PROJECT_ROOT}/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow"
: "${PREPARED_DATASET:=${DATA_DIR}/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${CLEAN_DATASET:=${DATA_DIR}/so_masked_no_noise_ell80_7979_sbi_run.npz}"
for input in "${PREPARED_DATASET}" "${CLEAN_DATASET}"; do
  [[ -f "${input}" ]] || { echo "Missing input: ${input}" >&2; exit 2; }
done
[[ "${COMPRESSION_ROOT}" != *,* ]] || { echo "PBS output root must not contain commas" >&2; exit 2; }

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Would submit 5 ordinary mini jobs, 26 CPUs/64 GB/23:59:00 each:"
  echo "prepare -> bins40, pca, moped (at most 3 running) -> summarize"
  echo "Prepared dataset: ${PREPARED_DATASET}"
  echo "Clean dataset: ${CLEAN_DATASET}"
  echo "Output: ${COMPRESSION_ROOT}"
  exit 0
fi
[[ $# == 0 ]] || { echo "Usage: bash $0 [--dry-run]" >&2; exit 2; }
command -v qsub >/dev/null || { echo "qsub is unavailable; run on the cluster" >&2; exit 2; }
mkdir -p "${COMPRESSION_ROOT}/logs" /home/kristero10/logs/SBI_runs
CONFIG="${COMPRESSION_ROOT}/submission_$(date +%Y%m%dT%H%M%S)_$$.sh"
for variable in PROJECT_ROOT PYTHON COMPRESSION_ROOT PREPARED_DATASET CLEAN_DATASET \
  HOLDOUT_LAST_N PCA_COMPONENTS MOPED_LOCAL_N COVARIANCE_SHRINKAGE MOPED_RCOND \
  HIDDEN_FEATURES NUM_TRANSFORMS TRAINING_BATCH_SIZE STOP_AFTER_EPOCHS MAX_NUM_EPOCHS \
  VALIDATION_FRACTION POSTERIOR_SAMPLES MAX_PROPOSALS SAMPLING_SECONDS SBI_SEED; do
  if [[ -n "${!variable:-}" ]]; then
    printf '%s=%q\n' "${variable}" "${!variable}" >> "${CONFIG}"
  fi
done
PBS="${SCRIPT_DIR}/run_so_sbi_compression_comparison.pbs"
LOG="${COMPRESSION_ROOT}/submission_jobs.tsv"
prep=$(qsub -N SOcmp_prep -v "COMPRESSION_CONFIG=${CONFIG},COMPRESSION_STAGE=prepare" "${PBS}")
printf 'prepare\t%s\n' "${prep}" | tee -a "${LOG}"
dependency=""
for method in bins40 pca moped; do
  job=$(qsub -N "SOcmp_${method}" -W "depend=afterok:${prep}" \
    -v "COMPRESSION_CONFIG=${CONFIG},COMPRESSION_STAGE=run,COMPRESSION_METHOD=${method}" "${PBS}")
  printf '%s\t%s\n' "${method}" "${job}" | tee -a "${LOG}"
  dependency="${dependency}:${job}"
done
summary=$(qsub -N SOcmp_summary -W "depend=afterok${dependency}" \
  -v "COMPRESSION_CONFIG=${CONFIG},COMPRESSION_STAGE=summarize" "${PBS}")
printf 'summary\t%s\n' "${summary}" | tee -a "${LOG}"
echo "Live logs: ${COMPRESSION_ROOT}/logs"
echo "Final plots: ${COMPRESSION_ROOT}/summary"
