#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
: "${PROJECT_ROOT:=${DEFAULT_PROJECT_ROOT}}"
: "${PREPARED_DATASET:=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${TWO_PARAM_ROOT:=/lustre/work/kristero10/adrian_two_param_npe_baseline_deproj0}"
: "${PBS_QUEUE:=mini}"
: "${OVERWRITE:=0}"

PBS_SCRIPT="${PROJECT_ROOT}/SBI_analysis/run_so_two_param_npe.pbs"
for path in "${PBS_SCRIPT}" "${PREPARED_DATASET}"; do
  [[ -e "${path}" ]] || { echo "Required input not found: ${path}" >&2; exit 2; }
done
mkdir -p "${TWO_PARAM_ROOT}"

LOG="${TWO_PARAM_ROOT}/submitted_training_jobs.csv"
if [[ -f "${LOG}" ]]; then
  active=""
  while IFS=, read -r mode job_id; do
    [[ "${mode}" == "mode" || -z "${job_id}" ]] && continue
    if qstat "${job_id}" >/dev/null 2>&1; then
      active+="${job_id}"$'\n'
    fi
  done < "${LOG}"
  if [[ -n "${active}" ]]; then
    echo "Two-parameter training jobs are already active:" >&2
    printf '%s' "${active}" >&2
    exit 3
  fi
fi

for mode in none asinh; do
  if [[ -f "${TWO_PARAM_ROOT}/${mode}/density_estimator.pkl" && "${OVERWRITE}" != "1" ]]; then
    echo "Refusing to replace completed ${mode} estimator. Set OVERWRITE=1 intentionally." >&2
    exit 2
  fi
done

printf 'mode,job_id\n' > "${LOG}"
for mode in none asinh; do
  job_id=$(
    qsub \
      -q "${PBS_QUEUE}" \
      -N "SO2p_${mode}" \
      -v "PROJECT_ROOT=${PROJECT_ROOT},PREPARED_DATASET=${PREPARED_DATASET},TWO_PARAM_ROOT=${TWO_PARAM_ROOT},X_RESCALE_MODE=${mode},OVERWRITE=${OVERWRITE}" \
      "${PBS_SCRIPT}"
  )
  job_id=$(printf '%s\n' "${job_id}" | tail -n 1 | tr -d '[:space:]')
  if [[ -z "${job_id}" || "${job_id}" == *"["* || "${job_id}" == *"]"* ]]; then
    echo "Invalid non-scalar PBS job id for ${mode}: ${job_id}" >&2
    [[ -z "${job_id}" ]] || qdel "${job_id}" >/dev/null 2>&1 || true
    exit 2
  fi
  printf '%s,%s\n' "${mode}" "${job_id}" >> "${LOG}"
  echo "Submitted independent ${mode} training job: ${job_id}"
done

echo "Job map: ${LOG}"
echo "After both jobs finish successfully, run:"
echo "  qsub SBI_analysis/run_so_two_param_evaluation.pbs"
