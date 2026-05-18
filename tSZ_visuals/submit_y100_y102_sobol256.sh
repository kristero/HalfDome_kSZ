#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${QSUB:=qsub}"
: "${PBS_SCRIPT:=${SCRIPT_DIR}/run_full_map_sobol_parallel.pbs}"
: "${PBS_SELECT:=}"
: "${PBS_WALLTIME:=}"
: "${SOBOL_BASE_DIR:=/home/kristero10/tSZ_data}"
: "${HALFDOME_BASE_DIR:=/lustre/work/Globus-lt/halfdome/full_res/halos}"
: "${OUTPUT_BASE_DIR:=/lustre/work/kristero10/tSZ_data}"
: "${CACHE_DIR:=/lustre/work/kristero10/tSZ_data/cache}"
: "${LOG_DIR:=/home/kristero10/logs/tSZ_baryon_run}"
: "${NSIDE:=4096}"
: "${THREADS_PER_TASK:=26}"
: "${MAX_PARALLEL:=1}"
: "${INTERPOLATOR_PAD:=256}"
: "${INTERPOLATOR_LOGM_MAX:=15.7}"
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
: "${Y100_MODEL_EXISTS:=false}"
: "${Y102_MODEL_EXISTS:=true}"
: "${DEPEND_Y102_ON_Y100:=true}"
: "${SUBMIT_Y100_SPLIT1:=true}"
: "${SUBMIT_Y100_SPLIT2:=true}"
: "${SUBMIT_Y102_SPLIT1:=true}"
: "${SUBMIT_Y102_SPLIT2:=true}"
: "${Y100_SPLIT1_ROW_START:=}"
: "${Y100_SPLIT1_ROW_STOP:=}"
: "${Y100_SPLIT2_ROW_START:=}"
: "${Y100_SPLIT2_ROW_STOP:=}"
: "${Y102_SPLIT1_ROW_START:=}"
: "${Y102_SPLIT1_ROW_STOP:=}"
: "${Y102_SPLIT2_ROW_START:=}"
: "${Y102_SPLIT2_ROW_STOP:=}"
: "${CHECK_INPUTS:=true}"
: "${DRY_RUN:=false}"

check_inputs() {
  local missing=0
  local expected_split_rows=128

  if [[ ! -f "${PBS_SCRIPT}" ]]; then
    echo "Missing PBS script: ${PBS_SCRIPT}" >&2
    missing=1
  fi

  if [[ "${DRY_RUN}" != "true" ]] && ! command -v "${QSUB}" >/dev/null 2>&1; then
    echo "Could not find qsub command: ${QSUB}" >&2
    missing=1
  fi

  for split in 1 2; do
    local sobol_csv="${SOBOL_BASE_DIR}/battaglia_sobol_256_${split}.csv"
    if [[ ! -f "${sobol_csv}" ]]; then
      echo "Missing Sobol split CSV: ${sobol_csv}" >&2
      missing=1
    else
      local row_count
      row_count="$(awk 'END {print NR-1}' "${sobol_csv}")"
      if [[ "${row_count}" -ne "${expected_split_rows}" ]]; then
        echo "Expected ${expected_split_rows} data rows in ${sobol_csv}, got ${row_count}" >&2
        missing=1
      fi
    fi
  done

  local split1_csv="${SOBOL_BASE_DIR}/battaglia_sobol_256_1.csv"
  local split2_csv="${SOBOL_BASE_DIR}/battaglia_sobol_256_2.csv"
  if [[ -f "${split1_csv}" && -f "${split2_csv}" ]] && cmp -s "${split1_csv}" "${split2_csv}"; then
    echo "Sobol split CSVs are identical; regenerate the 1/2 and 2/2 split files before submitting." >&2
    missing=1
  fi

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

qsub_var_list() {
  local lightcone_id="$1"
  local split="$2"
  local model_exists="$3"
  local row_start="${4:-}"
  local row_stop="${5:-}"

  local sobol_csv="${SOBOL_BASE_DIR}/battaglia_sobol_256_${split}.csv"
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
    ",CL_LMAX=${CL_LMAX}" \
    ",CL_NITER=${CL_NITER}" \
    ",SKIP_EXISTING_OUTPUTS=${SKIP_EXISTING_OUTPUTS}" \
    ",SKIP_EXISTING_ANY_RUN_INSTANCE=${SKIP_EXISTING_ANY_RUN_INSTANCE}" \
    ",RUN_INSTANCE_TAG=${RUN_INSTANCE_TAG}" \
    ",ENFORCE_BATTAGLIA_GUARDRAILS=${ENFORCE_BATTAGLIA_GUARDRAILS}" \
    ",SKIP_INVALID_BATTAGLIA_ROWS=${SKIP_INVALID_BATTAGLIA_ROWS}")"

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

  local job_name="tSZ_y${lightcone_id}_s256_${split}"
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

  if [[ "${should_submit}" != "true" ]]; then
    echo "Skipping tSZ_y${1}_s256_${2} because its SUBMIT flag is ${should_submit}" >&2
    return 0
  fi

  submit_job "$@"
}

if [[ "${CHECK_INPUTS}" == "true" ]]; then
  check_inputs
fi

mkdir -p "${LOG_DIR}"

y100_split1="$(maybe_submit_job "${SUBMIT_Y100_SPLIT1}" 100 1 "${Y100_MODEL_EXISTS}" "" "${Y100_SPLIT1_ROW_START}" "${Y100_SPLIT1_ROW_STOP}")"
y100_split2="$(maybe_submit_job "${SUBMIT_Y100_SPLIT2}" 100 2 "${Y100_MODEL_EXISTS}" "" "${Y100_SPLIT2_ROW_START}" "${Y100_SPLIT2_ROW_STOP}")"

y102_split1_dependency=""
y102_split2_dependency=""
if [[ "${DEPEND_Y102_ON_Y100}" == "true" ]]; then
  y102_split1_dependency="${y100_split1}"
  y102_split2_dependency="${y100_split2}"
fi

y102_split1="$(maybe_submit_job "${SUBMIT_Y102_SPLIT1}" 102 1 "${Y102_MODEL_EXISTS}" "${y102_split1_dependency}" "${Y102_SPLIT1_ROW_START}" "${Y102_SPLIT1_ROW_STOP}")"
y102_split2="$(maybe_submit_job "${SUBMIT_Y102_SPLIT2}" 102 2 "${Y102_MODEL_EXISTS}" "${y102_split2_dependency}" "${Y102_SPLIT2_ROW_START}" "${Y102_SPLIT2_ROW_STOP}")"

cat <<EOF
Submitted Sobol 256 jobs:
  y100 split 1: ${y100_split1}
  y100 split 2: ${y100_split2}
  y102 split 1: ${y102_split1}
  y102 split 2: ${y102_split2}
EOF
