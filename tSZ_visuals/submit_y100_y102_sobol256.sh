#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${QSUB:=qsub}"
: "${PBS_SCRIPT:=${SCRIPT_DIR}/run_full_map_sobol_parallel.pbs}"
: "${PBS_SELECT:=}"
: "${PBS_WALLTIME:=}"
: "${SOBOL_BASE_DIR:=/home/kristero10/tSZ_data}"
: "${SOBOL_BASENAME:=battaglia_sobol_512}"
: "${SOBOL_FULL_CSV:=${SOBOL_BASE_DIR}/${SOBOL_BASENAME}.csv}"
: "${SOBOL_SPLIT_DIR:=${SOBOL_BASE_DIR}}"
: "${SOBOL_SPLIT_ROWS:=128}"
: "${SOBOL_SPLIT_COUNT:=4}"
: "${CREATE_SOBOL_SPLITS:=true}"
: "${OVERWRITE_SOBOL_SPLITS:=true}"
: "${HALFDOME_BASE_DIR:=/lustre/work/Globus-lt/halfdome/full_res/halos}"
: "${OUTPUT_BASE_DIR:=/lustre/work/kristero10/tSZ_data}"
: "${CACHE_DIR:=/lustre/work/kristero10/tSZ_data/cache}"
: "${LOG_DIR:=/home/kristero10/logs/tSZ_baryon_run}"
: "${NSIDE:=4096}"
: "${THREADS_PER_TASK:=26}"
: "${MAX_PARALLEL:=1}"
: "${INTERPOLATOR_PAD:=256}"
: "${INTERPOLATOR_LOGM_MAX:=15.7}"
: "${INTERPOLATOR_TIMEOUT_SECONDS:=700}"
: "${CL_LMAX:=4096}"
: "${CL_NITER:=0}"
: "${REUSE_EXISTING_CACHE:=false}"
: "${CACHE_WAIT_SECONDS:=0}"
: "${CACHE_POLL_SECONDS:=30}"
: "${SKIP_EXISTING_OUTPUTS:=true}"
: "${SKIP_EXISTING_ANY_RUN_INSTANCE:=true}"
: "${RUN_INSTANCE_TAG:=}"
: "${ENFORCE_BATTAGLIA_GUARDRAILS:=true}"
: "${SKIP_INVALID_BATTAGLIA_ROWS:=true}"
: "${CONTINUE_ON_ROW_ERROR:=true}"
: "${PRINT_RUNTIME_ENVIRONMENT:=false}"
: "${Y100_MODEL_EXISTS:=false}"
: "${Y102_MODEL_EXISTS:=true}"
: "${DEPEND_Y102_ON_Y100:=true}"
: "${CHECK_INPUTS:=true}"
: "${DRY_RUN:=false}"

is_true() {
  case "${1:-}" in
    true|TRUE|True|1|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

sobol_split_csv_path() {
  local split="$1"
  printf '%s/%s_%s.csv' "${SOBOL_SPLIT_DIR}" "${SOBOL_BASENAME}" "${split}"
}

split_sobol_csv() {
  is_true "${CREATE_SOBOL_SPLITS}" || return 0

  if [[ ! -f "${SOBOL_FULL_CSV}" ]]; then
    echo "Missing full Sobol CSV: ${SOBOL_FULL_CSV}" >&2
    exit 1
  fi

  mkdir -p "${SOBOL_SPLIT_DIR}"

  local total_rows expected_rows
  total_rows="$(awk 'END {print NR-1}' "${SOBOL_FULL_CSV}")"
  expected_rows=$(( SOBOL_SPLIT_ROWS * SOBOL_SPLIT_COUNT ))
  if [[ "${total_rows}" -ne "${expected_rows}" ]]; then
    echo "Expected ${expected_rows} data rows in ${SOBOL_FULL_CSV}, got ${total_rows}" >&2
    exit 1
  fi

  local split start stop output_csv
  for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
    output_csv="$(sobol_split_csv_path "${split}")"
    if [[ -f "${output_csv}" ]] && ! is_true "${OVERWRITE_SOBOL_SPLITS}"; then
      continue
    fi

    start=$(( (split - 1) * SOBOL_SPLIT_ROWS + 1 ))
    stop=$(( split * SOBOL_SPLIT_ROWS ))
    awk -v start="${start}" -v stop="${stop}" '
      NR == 1 { print; next }
      NR - 1 >= start && NR - 1 <= stop { print }
    ' "${SOBOL_FULL_CSV}" > "${output_csv}"
  done
}

check_inputs() {
  local missing=0
  local split

  if [[ ! -f "${PBS_SCRIPT}" ]]; then
    echo "Missing PBS script: ${PBS_SCRIPT}" >&2
    missing=1
  fi

  if [[ "${DRY_RUN}" != "true" ]] && ! command -v "${QSUB}" >/dev/null 2>&1; then
    echo "Could not find qsub command: ${QSUB}" >&2
    missing=1
  fi

  for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
    local sobol_csv
    sobol_csv="$(sobol_split_csv_path "${split}")"
    if [[ ! -f "${sobol_csv}" ]]; then
      echo "Missing Sobol split CSV: ${sobol_csv}" >&2
      missing=1
    else
      local row_count
      row_count="$(awk 'END {print NR-1}' "${sobol_csv}")"
      if [[ "${row_count}" -ne "${SOBOL_SPLIT_ROWS}" ]]; then
        echo "Expected ${SOBOL_SPLIT_ROWS} data rows in ${sobol_csv}, got ${row_count}" >&2
        missing=1
      fi
    fi
  done

  for lightcone_id in 100 102; do
    local halfdome_path="${HALFDOME_BASE_DIR}/lightcone_${lightcone_id}.hdf5"
    if [[ ! -f "${halfdome_path}" ]]; then
      echo "Missing HalfDome lightcone: ${halfdome_path}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    exit 1
  fi
}

row_bound_value() {
  local lightcone_id="$1"
  local split="$2"
  local suffix="$3"
  local var_name="Y${lightcone_id}_SPLIT${split}_ROW_${suffix}"
  printf '%s' "${!var_name:-}"
}

submit_flag_value() {
  local lightcone_id="$1"
  local split="$2"
  local var_name="SUBMIT_Y${lightcone_id}_SPLIT${split}"
  printf '%s' "${!var_name:-true}"
}

qsub_var_list() {
  local lightcone_id="$1"
  local split="$2"
  local model_exists="$3"
  local row_start="${4:-}"
  local row_stop="${5:-}"

  local sobol_csv
  sobol_csv="$(sobol_split_csv_path "${split}")"
  local halfdome_path="${HALFDOME_BASE_DIR}/lightcone_${lightcone_id}.hdf5"
  local output_dir="${OUTPUT_BASE_DIR}/y${lightcone_id}"
  local simulation_name="halfdome_lightcone_${lightcone_id}"

  local vars
  vars="$(printf '%s' \
    "LIGHTCONE_ID=${lightcone_id}" \
    ",SOBOL_SPLIT=${split}" \
    ",SOBOL_CSV=${sobol_csv}" \
    ",HALFDOME_PATH=${halfdome_path}" \
    ",OUTPUT_DIR=${output_dir}" \
    ",CACHE_DIR=${CACHE_DIR}" \
    ",LOG_DIR=${LOG_DIR}" \
    ",SIMULATION_NAME=${simulation_name}" \
    ",NSIDE=${NSIDE}" \
    ",THREADS_PER_TASK=${THREADS_PER_TASK}" \
    ",MAX_PARALLEL=${MAX_PARALLEL}" \
    ",MODEL_EXISTS=${model_exists}" \
    ",REUSE_EXISTING_CACHE=${REUSE_EXISTING_CACHE}" \
    ",CACHE_WAIT_SECONDS=${CACHE_WAIT_SECONDS}" \
    ",CACHE_POLL_SECONDS=${CACHE_POLL_SECONDS}" \
    ",INTERPOLATOR_PAD=${INTERPOLATOR_PAD}" \
    ",INTERPOLATOR_LOGM_MAX=${INTERPOLATOR_LOGM_MAX}" \
    ",INTERPOLATOR_TIMEOUT_SECONDS=${INTERPOLATOR_TIMEOUT_SECONDS}" \
    ",CL_LMAX=${CL_LMAX}" \
    ",CL_NITER=${CL_NITER}" \
    ",SKIP_EXISTING_OUTPUTS=${SKIP_EXISTING_OUTPUTS}" \
    ",SKIP_EXISTING_ANY_RUN_INSTANCE=${SKIP_EXISTING_ANY_RUN_INSTANCE}" \
    ",RUN_INSTANCE_TAG=${RUN_INSTANCE_TAG}" \
    ",ENFORCE_BATTAGLIA_GUARDRAILS=${ENFORCE_BATTAGLIA_GUARDRAILS}" \
    ",SKIP_INVALID_BATTAGLIA_ROWS=${SKIP_INVALID_BATTAGLIA_ROWS}" \
    ",CONTINUE_ON_ROW_ERROR=${CONTINUE_ON_ROW_ERROR}" \
    ",PRINT_RUNTIME_ENVIRONMENT=${PRINT_RUNTIME_ENVIRONMENT}")"

  if [[ -n "${row_start}" ]]; then
    vars+=",SOBOL_ROW_START=${row_start}"
  fi
  if [[ -n "${row_stop}" ]]; then
    vars+=",SOBOL_ROW_STOP=${row_stop}"
  fi

  printf '%s' "${vars}"
}

submit_job() {
  local lightcone_id="$1"
  local split="$2"
  local model_exists="$3"
  local dependency="${4:-}"
  local row_start="${5:-}"
  local row_stop="${6:-}"

  local job_name="tSZ_y${lightcone_id}_s512_${split}"
  local vars
  vars="$(qsub_var_list "${lightcone_id}" "${split}" "${model_exists}" "${row_start}" "${row_stop}")"

  local cmd=("${QSUB}" -N "${job_name}")
  if [[ -n "${PBS_SELECT}" ]]; then
    cmd+=(-l "select=${PBS_SELECT}")
  fi
  if [[ -n "${PBS_WALLTIME}" ]]; then
    cmd+=(-l "walltime=${PBS_WALLTIME}")
  fi
  cmd+=(-v "${vars}")
  if [[ -n "${dependency}" ]]; then
    cmd+=(-W "depend=afterok:${dependency}")
  fi
  cmd+=("${PBS_SCRIPT}")

  echo "Submitting ${job_name}" >&2
  printf '  ' >&2
  printf '%q ' "${cmd[@]}" >&2
  printf '\n' >&2

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'DRYRUN_%s' "${job_name}"
    return 0
  fi

  local job_id
  job_id="$("${cmd[@]}")"
  job_id="${job_id//$'\n'/}"
  job_id="${job_id//$'\r'/}"
  echo "Submitted ${job_name}: ${job_id}" >&2
  printf '%s' "${job_id}"
}

maybe_submit_job() {
  local should_submit="$1"
  shift

  if ! is_true "${should_submit}"; then
    echo "Skipping tSZ_y${1}_s512_${2} because its SUBMIT flag is ${should_submit}" >&2
    return 0
  fi

  submit_job "$@"
}

split_sobol_csv

if [[ "${CHECK_INPUTS}" == "true" ]]; then
  check_inputs
fi

mkdir -p "${LOG_DIR}"

declare -a y100_jobs
declare -a y102_jobs

for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
  row_start="$(row_bound_value 100 "${split}" START)"
  row_stop="$(row_bound_value 100 "${split}" STOP)"
  y100_jobs[$split]="$(maybe_submit_job "$(submit_flag_value 100 "${split}")" 100 "${split}" "${Y100_MODEL_EXISTS}" "" "${row_start}" "${row_stop}")"
done

for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
  row_start="$(row_bound_value 102 "${split}" START)"
  row_stop="$(row_bound_value 102 "${split}" STOP)"
  dependency=""
  if is_true "${DEPEND_Y102_ON_Y100}"; then
    dependency="${y100_jobs[$split]:-}"
  fi
  y102_jobs[$split]="$(maybe_submit_job "$(submit_flag_value 102 "${split}")" 102 "${split}" "${Y102_MODEL_EXISTS}" "${dependency}" "${row_start}" "${row_stop}")"
done

echo "Submitted Sobol 512 jobs:"
for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
  echo "  y100 split ${split}: ${y100_jobs[$split]:-}"
done
for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
  echo "  y102 split ${split}: ${y102_jobs[$split]:-}"
done
