#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
: "${PROJECT_ROOT:=${DEFAULT_PROJECT_ROOT}}"
: "${PREPARED_DATASET:=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${CONVERGENCE_ROOT:=/lustre/work/kristero10/adrian_two_param_nsf_convergence_baseline_deproj0}"
: "${DATASET_SIZES:=256,512,1024,2048,4096,8192,16384,32768,65536,98304,131072,196608,262144,327680,393216,458752,523288}"
: "${PBS_QUEUE:=mini2}"
: "${SUMMARY_QUEUE:=mini2}"
: "${PBS_WALLTIME:=48:00:00}"
: "${MAX_CONCURRENT:=5}"
: "${OVERWRITE:=0}"
: "${SUBMIT_SUMMARY:=1}"

if (( MAX_CONCURRENT < 1 || MAX_CONCURRENT > 5 )); then
  echo "MAX_CONCURRENT must be between 1 and 5; got ${MAX_CONCURRENT}." >&2
  exit 2
fi

WORKER_PBS="${PROJECT_ROOT}/SBI_analysis/run_so_two_param_nsf_convergence.pbs"
SUMMARY_PBS="${PROJECT_ROOT}/SBI_analysis/run_so_two_param_nsf_convergence_summary.pbs"
for path in "${WORKER_PBS}" "${SUMMARY_PBS}" "${PREPARED_DATASET}"; do
  [[ -e "${path}" ]] || { echo "Required input not found: ${path}" >&2; exit 2; }
done
mkdir -p /home/kristero10/logs/SBI_runs "${CONVERGENCE_ROOT}"

IFS=', ' read -r -a RAW_SIZES <<< "${DATASET_SIZES}"
SIZES=()
for value in "${RAW_SIZES[@]}"; do
  [[ -z "${value}" ]] && continue
  value=${value//_/}
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid dataset size: ${value}" >&2; exit 2; }
  (( value > 0 && value <= 523288 )) || {
    echo "Dataset size ${value} is outside 1..523288." >&2
    exit 2
  }
  SIZES+=("${value}")
done
(( ${#SIZES[@]} > 0 )) || { echo "No dataset sizes supplied." >&2; exit 2; }

LOG="${CONVERGENCE_ROOT}/submitted_convergence_jobs.csv"
if [[ -f "${LOG}" ]]; then
  ACTIVE_IDS=()
  while IFS=, read -r kind mode n_train job_id depends_on; do
    [[ "${kind}" == "kind" || -z "${job_id}" ]] && continue
    if qstat "${job_id}" >/dev/null 2>&1; then
      ACTIVE_IDS+=("${job_id}")
    fi
  done < "${LOG}"
  if (( ${#ACTIVE_IDS[@]} > 0 )); then
    echo "This convergence submission still has active or queued jobs:" >&2
    printf '  %s\n' "${ACTIVE_IDS[@]}" >&2
    echo "Wait for them to finish before resubmitting missing runs." >&2
    exit 3
  fi
fi

printf 'kind,mode,n_train,job_id,depends_on\n' > "${LOG}"
LANE_LAST=()
for ((lane = 0; lane < MAX_CONCURRENT; lane++)); do
  LANE_LAST[lane]=""
done

task_index=0
submitted=0
skipped=0
for n_train in "${SIZES[@]}"; do
  for mode in asinh; do
    completion="${CONVERGENCE_ROOT}/${mode}/N${n_train}/evaluation/evaluation_complete.json"
    if [[ -f "${completion}" && "${OVERWRITE}" != "1" ]]; then
      echo "Completed, skipping: mode=${mode}, N=${n_train}"
      ((skipped += 1))
      continue
    fi

    lane=$((task_index % MAX_CONCURRENT))
    dependency="${LANE_LAST[lane]}"
    QSUB_ARGS=(
      -q "${PBS_QUEUE}"
      -l "walltime=${PBS_WALLTIME}"
      -N "S2n${mode:0:1}${n_train}"
    )
    if [[ -n "${dependency}" ]]; then
      QSUB_ARGS+=( -W "depend=afterany:${dependency}" )
    fi
    QSUB_ARGS+=(
      -v "PROJECT_ROOT=${PROJECT_ROOT},PREPARED_DATASET=${PREPARED_DATASET},CONVERGENCE_ROOT=${CONVERGENCE_ROOT},N_TRAIN=${n_train},X_RESCALE_MODE=${mode},OVERWRITE=${OVERWRITE}"
      "${WORKER_PBS}"
    )

    job_id=$(qsub "${QSUB_ARGS[@]}")
    job_id=$(printf '%s\n' "${job_id}" | tail -n 1 | tr -d '[:space:]')
    if [[ -z "${job_id}" || "${job_id}" == *"["* || "${job_id}" == *"]"* ]]; then
      echo "Invalid non-scalar PBS job ID for ${mode}, N=${n_train}: ${job_id}" >&2
      [[ -z "${job_id}" ]] || qdel "${job_id}" >/dev/null 2>&1 || true
      exit 2
    fi
    LANE_LAST[lane]="${job_id}"
    printf 'worker,%s,%s,%s,%s\n' "${mode}" "${n_train}" "${job_id}" "${dependency}" >> "${LOG}"
    echo "Submitted lane $((lane + 1))/${MAX_CONCURRENT}: mode=${mode}, N=${n_train}, job=${job_id}, after=${dependency:-none}"
    ((task_index += 1))
    ((submitted += 1))
  done
done

if [[ "${SUBMIT_SUMMARY}" == "1" ]]; then
  TAILS=()
  for ((lane = 0; lane < MAX_CONCURRENT; lane++)); do
    [[ -n "${LANE_LAST[lane]}" ]] && TAILS+=("${LANE_LAST[lane]}")
  done
  SUMMARY_ARGS=(
    -q "${SUMMARY_QUEUE}"
    -N SO2pNSFsum
    -v "PROJECT_ROOT=${PROJECT_ROOT},CONVERGENCE_ROOT=${CONVERGENCE_ROOT}"
  )
  summary_dependency=""
  if (( ${#TAILS[@]} > 0 )); then
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
  printf 'summary,,,%s,%s\n' "${summary_id}" "${summary_dependency}" >> "${LOG}"
  echo "Submitted summary job: ${summary_id}, after=${summary_dependency:-immediate}"
fi

echo "Submitted workers: ${submitted}; already complete: ${skipped}"
echo "At most ${MAX_CONCURRENT} independent worker jobs can run concurrently."
echo "Job map: ${LOG}"
echo "Outputs: ${CONVERGENCE_ROOT}"
