#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
: "${PROJECT_ROOT:=${DEFAULT_PROJECT_ROOT}}"
: "${PYTHON:=python3}"
: "${PREPARED_DATASET:=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${CONVERGENCE_ROOT:=/lustre/work/kristero10/adrian_two_param_maf_convergence_baseline_deproj0}"
: "${DATASET_SIZES:=256,512,1024,2048,4096,8192,16384,32768,65536,98304,131072,196608,262144,327680,393216,458752,523288}"
: "${CORNER_SIZES:=256,32768,523288}"
: "${PBS_QUEUE:=mini}"
: "${FULL_DATASET_QUEUE:=large}"
: "${SUMMARY_QUEUE:=mini}"
: "${PBS_WALLTIME:=23:59:00}"
: "${PBS_NCPUS:=26}"
: "${PBS_MEM:=64gb}"
: "${FULL_DATASET_N:=523288}"
: "${MAX_CONCURRENT:=5}"
: "${HIDDEN_FEATURES:=64}"
: "${NUM_TRANSFORMS:=6}"
: "${STOP_AFTER_EPOCHS:=20}"
: "${TRAINING_BATCH_SIZE:=1024}"
: "${VALIDATION_FRACTION:=0.1}"
: "${MAX_NUM_EPOCHS:=200}"
: "${NUM_POSTERIOR_SAMPLES:=2000}"
: "${BATTAGLIA_SIZES:=256,32768,523288}"
: "${NUM_BATTAGLIA_SAMPLES:=20000}"
: "${OVERWRITE:=0}"
: "${REFRESH_BATTAGLIA:=0}"
: "${SUBMIT_SUMMARY:=1}"

if (( MAX_CONCURRENT < 1 || MAX_CONCURRENT > 5 )); then
  echo "MAX_CONCURRENT must be between 1 and 5; got ${MAX_CONCURRENT}." >&2
  exit 2
fi

WORKER_PBS="${PROJECT_ROOT}/SBI_analysis/run_so_two_param_maf_convergence.pbs"
SUMMARY_PBS="${PROJECT_ROOT}/SBI_analysis/run_so_two_param_maf_convergence_summary.pbs"
RUNTIME_CHECK="${PROJECT_ROOT}/SBI_analysis/check_sbi_cluster_runtime.py"
EVAL_SCRIPT="${PROJECT_ROOT}/SBI_analysis/evaluate_so_two_param_nsf_convergence_run.py"
SUMMARY_SCRIPT="${PROJECT_ROOT}/SBI_analysis/summarize_so_two_param_nsf_convergence.py"
for path in \
  "${WORKER_PBS}" \
  "${SUMMARY_PBS}" \
  "${RUNTIME_CHECK}" \
  "${EVAL_SCRIPT}" \
  "${SUMMARY_SCRIPT}" \
  "${PREPARED_DATASET}"
do
  [[ -e "${path}" ]] || {
    echo "Required input not found: ${path}" >&2
    exit 2
  }
done

for path in "${EVAL_SCRIPT}" "${SUMMARY_SCRIPT}"; do
  if ! grep -q -- "--expected-density-estimator" "${path}"; then
    echo "Incompatible stale script: ${path}" >&2
    echo "Update the cluster checkout before submitting MAF convergence jobs." >&2
    exit 2
  fi
done

echo "Checking the selected Python before submission: ${PYTHON}"
if ! "${PYTHON}" "${RUNTIME_CHECK}"; then
  echo "No jobs were submitted. Set PYTHON to a consistent SBI environment." >&2
  exit 2
fi

mkdir -p /home/kristero10/logs/SBI_runs "${CONVERGENCE_ROOT}"
CONFIG="${CONVERGENCE_ROOT}/maf_convergence_config.env"
{
  printf 'PREPARED_DATASET=%q\n' "${PREPARED_DATASET}"
  printf 'CONVERGENCE_ROOT=%q\n' "${CONVERGENCE_ROOT}"
  printf 'DATASET_SIZES=%q\n' "${DATASET_SIZES}"
  printf 'CORNER_SIZES=%q\n' "${CORNER_SIZES}"
  printf 'HIDDEN_FEATURES=%q\n' "${HIDDEN_FEATURES}"
  printf 'NUM_TRANSFORMS=%q\n' "${NUM_TRANSFORMS}"
  printf 'STOP_AFTER_EPOCHS=%q\n' "${STOP_AFTER_EPOCHS}"
  printf 'TRAINING_BATCH_SIZE=%q\n' "${TRAINING_BATCH_SIZE}"
  printf 'VALIDATION_FRACTION=%q\n' "${VALIDATION_FRACTION}"
  printf 'MAX_NUM_EPOCHS=%q\n' "${MAX_NUM_EPOCHS}"
  printf 'NUM_POSTERIOR_SAMPLES=%q\n' "${NUM_POSTERIOR_SAMPLES}"
  printf 'BATTAGLIA_SIZES=%q\n' "${BATTAGLIA_SIZES}"
  printf 'NUM_BATTAGLIA_SAMPLES=%q\n' "${NUM_BATTAGLIA_SAMPLES}"
  printf 'OVERWRITE=%q\n' "${OVERWRITE}"
  printf 'REFRESH_BATTAGLIA=%q\n' "${REFRESH_BATTAGLIA}"
} > "${CONFIG}"

IFS=', ' read -r -a RAW_SIZES <<< "${DATASET_SIZES}"
SIZES=()
for value in "${RAW_SIZES[@]}"; do
  [[ -z "${value}" ]] && continue
  value=${value//_/}
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "Invalid dataset size: ${value}" >&2
    exit 2
  }
  (( value > 0 && value <= 523288 )) || {
    echo "Dataset size ${value} is outside 1..523288." >&2
    exit 2
  }
  SIZES+=("${value}")
done
(( ${#SIZES[@]} > 0 )) || {
  echo "No dataset sizes supplied." >&2
  exit 2
}

LOG="${CONVERGENCE_ROOT}/submitted_convergence_jobs.csv"
if [[ -f "${LOG}" ]]; then
  ACTIVE_IDS=()
  while IFS=, read -r kind mode n_train job_id depends_on; do
    [[ "${kind}" == "kind" || -z "${job_id}" ]] && continue
    if qstat "${job_id}" >/dev/null 2>&1; then
      ACTIVE_IDS+=("${job_id}")
    fi
  done < "${LOG}"
  if [[ -n "${ACTIVE_IDS[0]-}" ]]; then
    echo "This MAF submission still has active jobs:" >&2
    printf '  %s\n' "${ACTIVE_IDS[@]}" >&2
    exit 3
  fi
fi

printf 'kind,mode,n_train,job_id,depends_on\n' > "${LOG}"
LANE_LAST=()
for ((lane=0; lane<MAX_CONCURRENT; lane++)); do
  LANE_LAST[lane]=""
done

task_index=0
submitted=0
skipped=0
for n_train in "${SIZES[@]}"; do
  completion="${CONVERGENCE_ROOT}/asinh/N${n_train}/evaluation/evaluation_complete.json"
  if [[ -f "${completion}" && "${OVERWRITE}" != "1" ]]; then
    echo "Completed, skipping: N=${n_train}"
    ((skipped += 1))
    continue
  fi

  lane=$((task_index % MAX_CONCURRENT))
  dependency="${LANE_LAST[lane]}"
  worker_queue="${PBS_QUEUE}"
  if (( n_train == FULL_DATASET_N )); then
    worker_queue="${FULL_DATASET_QUEUE}"
  fi

  QSUB_ARGS=(
    -q "${worker_queue}"
    -l "select=1:ncpus=${PBS_NCPUS}:mpiprocs=1:mem=${PBS_MEM}"
    -l "walltime=${PBS_WALLTIME}"
    -N "S2m${n_train}"
  )
  if [[ -n "${dependency}" ]]; then
    QSUB_ARGS+=( -W "depend=afterany:${dependency}" )
  fi
  QSUB_ARGS+=(
    -v "PROJECT_ROOT=${PROJECT_ROOT},PYTHON=${PYTHON},CONVERGENCE_CONFIG=${CONFIG},N_TRAIN=${n_train}"
    "${WORKER_PBS}"
  )

  job_id=$(qsub "${QSUB_ARGS[@]}")
  job_id=$(printf '%s\n' "${job_id}" | tail -n 1 | tr -d '[:space:]')
  if [[ -z "${job_id}" || "${job_id}" == *"["* || "${job_id}" == *"]"* ]]; then
    echo "Invalid non-scalar PBS job ID for N=${n_train}: ${job_id}" >&2
    [[ -z "${job_id}" ]] || qdel "${job_id}" >/dev/null 2>&1 || true
    exit 2
  fi
  LANE_LAST[lane]="${job_id}"
  printf 'worker,asinh,%s,%s,%s\n' \
    "${n_train}" "${job_id}" "${dependency}" >> "${LOG}"
  echo "Submitted lane $((lane+1))/${MAX_CONCURRENT}: queue=${worker_queue}, N=${n_train}, job=${job_id}, after=${dependency:-none}"
  ((task_index += 1))
  ((submitted += 1))
done

if [[ "${SUBMIT_SUMMARY}" == "1" ]]; then
  TAILS=()
  for ((lane=0; lane<MAX_CONCURRENT; lane++)); do
    [[ -n "${LANE_LAST[lane]}" ]] && TAILS+=("${LANE_LAST[lane]}")
  done
  SUMMARY_ARGS=(
    -q "${SUMMARY_QUEUE}"
    -N SO2pMAFsum
    -v "PROJECT_ROOT=${PROJECT_ROOT},PYTHON=${PYTHON},CONVERGENCE_CONFIG=${CONFIG}"
  )
  summary_dependency=""
  if [[ -n "${TAILS[0]-}" ]]; then
    summary_dependency=$(IFS=:; echo "${TAILS[*]}")
    SUMMARY_ARGS+=( -W "depend=afterany:${summary_dependency}" )
  fi
  SUMMARY_ARGS+=("${SUMMARY_PBS}")
  summary_id=$(qsub "${SUMMARY_ARGS[@]}")
  summary_id=$(printf '%s\n' "${summary_id}" | tail -n 1 | tr -d '[:space:]')
  if [[ -z "${summary_id}" || "${summary_id}" == *"["* || "${summary_id}" == *"]"* ]]; then
    echo "Invalid summary job ID: ${summary_id}" >&2
    exit 2
  fi
  printf 'summary,,,%s,%s\n' \
    "${summary_id}" "${summary_dependency}" >> "${LOG}"
  echo "Submitted summary job: ${summary_id}, after=${summary_dependency:-immediate}"
fi

echo "Submitted workers: ${submitted}; already complete: ${skipped}"
echo "At most ${MAX_CONCURRENT} independent jobs can run concurrently."
echo "N=${FULL_DATASET_N} uses ${FULL_DATASET_QUEUE}; smaller runs use ${PBS_QUEUE}."
echo "MAF: hidden_features=${HIDDEN_FEATURES}, num_transforms=${NUM_TRANSFORMS}, asinh transform."
echo "Outputs: ${CONVERGENCE_ROOT}"
echo "Job map: ${LOG}"
