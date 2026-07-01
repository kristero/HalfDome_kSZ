#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =========================
# USER EDITS / target-cluster paths
# =========================
: "${SBATCH:=sbatch}"
: "${JULIA:=julia}"
: "${ENV_SETUP:=}"  # e.g. 'module load julia' or 'module load julia; export JULIA_PROJECT=/path/to/env'
: "${SLURM_SCRIPT:=${SCRIPT_DIR}/run_sobol32768_full_maps_slurm.sbatch}"
: "${HALFDOME_BASE_DIR:=/lustre/work/Globus-lt/halfdome/full_res/halos}"
: "${OUTPUT_BASE_DIR:=/lustre/work/${USER}/tSZ_data/so_noise_s32768}"
: "${CACHE_DIR:=/lustre/work/${USER}/tSZ_data/cache}"
: "${LOG_DIR:=${HOME}/logs/tSZ_so_noise}"
: "${SO_NOISE_DIR:=${PROJECT_DIR}/other_sims/SO}"
: "${BASELINE_NOISE_PATH:=${SO_NOISE_DIR}/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt}"
: "${GOAL_NOISE_PATH:=${SO_NOISE_DIR}/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt}"

# =========================
# Dataset/splitting
# =========================
: "${SOBOL_BASE_DIR:=${PROJECT_DIR}/Sobol_tSZ}"
: "${SOBOL_BASENAME:=battaglia_sobol_32768}"
: "${SOBOL_FULL_CSV:=${SOBOL_BASE_DIR}/${SOBOL_BASENAME}.csv}"
: "${SOBOL_SPLIT_DIR:=${SOBOL_BASE_DIR}/splits_${SOBOL_BASENAME}}"
: "${SOBOL_SPLIT_ROWS:=128}"
: "${SOBOL_SPLIT_COUNT:=}"
: "${CREATE_SOBOL_SPLITS:=true}"
: "${OVERWRITE_SOBOL_SPLITS:=true}"
: "${CHECK_INPUTS:=true}"

# =========================
# SLURM resources
# =========================
: "${CPUS_PER_TASK:=25}"
: "${MEM:=128G}"
: "${TIME:=23:59:00}"
: "${SLURM_PARTITION:=}"
: "${SLURM_ACCOUNT:=}"
: "${SLURM_QOS:=}"
: "${ARRAY_CONCURRENCY:=4}"

# =========================
# SO-noise science options
# =========================
: "${NSIDE:=4096}"
: "${ELL_MIN:=80}"
: "${ELL_MAX:=7979}"
: "${SO_NOISE_DEPROJECTIONS:=0,2}"
: "${SO_NOISE_IS_DL:=false}"
: "${MASK_FSKY:=0.4}"
: "${MASK_APODIZATION_ARCMIN:=60.0}"
: "${SEED:=12345}"

# Profiles are saved as .npy only. These six outputs are expected per row for
# two deprojections: no-noise, baseline cross, goal cross for each deprojection.
: "${SAVE_NO_NOISE_CL:=true}"
: "${SAVE_BASELINE_NOISE_CROSS_CL:=true}"
: "${SAVE_GOAL_NOISE_CROSS_CL:=true}"
: "${SAVE_UNMASKED_NO_NOISE_CL:=false}"

# Debug map outputs are FITS and large. Keep false for production.
: "${SAVE_NOISE_MAPS:=false}"
: "${SAVE_NOISY_MAPS:=false}"
: "${SAVE_MASK_MAP:=false}"
: "${SAVE_SIGNAL_MAP:=false}"
: "${SAVE_MASKED_SIGNAL_MAP:=false}"

# Runtime/cache behavior.
: "${MAX_PARALLEL:=1}"
: "${INTERPOLATOR_PAD:=256}"
: "${INTERPOLATOR_LOGM_MAX:=15.7}"
: "${CL_NITER:=0}"
: "${REUSE_EXISTING_CACHE:=false}"
: "${CACHE_WAIT_SECONDS:=0}"
: "${CACHE_POLL_SECONDS:=30}"
: "${ENFORCE_BATTAGLIA_GUARDRAILS:=true}"
: "${SKIP_INVALID_BATTAGLIA_ROWS:=true}"
: "${CONTINUE_ON_ROW_ERROR:=true}"
: "${PRINT_RUNTIME_ENVIRONMENT:=false}"

# Lightcones. Keep RUN_Y102=false if they only need one HalfDome lightcone.
: "${RUN_Y100:=true}"
: "${RUN_Y102:=false}"
: "${DEPEND_Y102_ON_Y100:=true}"
: "${Y100_MODEL_EXISTS:=false}"
: "${Y102_MODEL_EXISTS:=true}"
: "${JOB_SET_TAG:=so_s32768}"
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

configure_sobol_split_count() {
  if [[ ! -f "${SOBOL_FULL_CSV}" ]]; then
    echo "Missing full Sobol CSV: ${SOBOL_FULL_CSV}" >&2
    exit 1
  fi
  if [[ ! "${SOBOL_SPLIT_ROWS}" =~ ^[0-9]+$ || "${SOBOL_SPLIT_ROWS}" -lt 1 ]]; then
    echo "SOBOL_SPLIT_ROWS must be a positive integer, got ${SOBOL_SPLIT_ROWS}" >&2
    exit 1
  fi

  local total_rows
  total_rows="$(awk 'END {print NR-1}' "${SOBOL_FULL_CSV}")"
  if [[ "${total_rows}" -lt 1 ]]; then
    echo "Sobol CSV has no data rows: ${SOBOL_FULL_CSV}" >&2
    exit 1
  fi
  if [[ $(( total_rows % SOBOL_SPLIT_ROWS )) -ne 0 ]]; then
    echo "Cannot split ${total_rows} rows evenly into chunks of ${SOBOL_SPLIT_ROWS} rows." >&2
    exit 1
  fi

  local computed_split_count
  computed_split_count=$(( total_rows / SOBOL_SPLIT_ROWS ))
  if [[ -n "${SOBOL_SPLIT_COUNT}" && "${SOBOL_SPLIT_COUNT}" -ne "${computed_split_count}" ]]; then
    echo "SOBOL_SPLIT_COUNT=${SOBOL_SPLIT_COUNT} does not match ${total_rows}/${SOBOL_SPLIT_ROWS}=${computed_split_count}." >&2
    exit 1
  fi
  SOBOL_SPLIT_COUNT="${computed_split_count}"
}

split_sobol_csv() {
  is_true "${CREATE_SOBOL_SPLITS}" || return 0

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

validate_sobol_splits() {
  local tmp_dir
  tmp_dir="$(mktemp -d)"

  local full_data="${tmp_dir}/full_rows.csv"
  local split_data="${tmp_dir}/split_rows.csv"
  local duplicate_rows="${tmp_dir}/duplicate_rows.csv"
  : > "${split_data}"

  local split sobol_csv row_count expected_total_rows
  expected_total_rows=$(( SOBOL_SPLIT_ROWS * SOBOL_SPLIT_COUNT ))

  for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
    sobol_csv="$(sobol_split_csv_path "${split}")"
    row_count="$(awk 'END {print NR-1}' "${sobol_csv}")"
    if [[ "${row_count}" -ne "${SOBOL_SPLIT_ROWS}" ]]; then
      echo "Expected ${SOBOL_SPLIT_ROWS} data rows in ${sobol_csv}, got ${row_count}" >&2
      rm -rf "${tmp_dir}"
      return 1
    fi
    awk 'NR > 1 { print }' "${sobol_csv}" >> "${split_data}"
  done

  local split_total_rows
  split_total_rows="$(wc -l < "${split_data}")"
  if [[ "${split_total_rows}" -ne "${expected_total_rows}" ]]; then
    echo "Expected ${expected_total_rows} total split rows, got ${split_total_rows}" >&2
    rm -rf "${tmp_dir}"
    return 1
  fi

  LC_ALL=C sort "${split_data}" | uniq -d > "${duplicate_rows}"
  if [[ -s "${duplicate_rows}" ]]; then
    echo "Duplicate Sobol parameter rows found across split CSVs; refusing to submit." >&2
    head -n 5 "${duplicate_rows}" >&2
    rm -rf "${tmp_dir}"
    return 1
  fi

  awk 'NR > 1 { print }' "${SOBOL_FULL_CSV}" > "${full_data}"
  if ! cmp -s "${full_data}" "${split_data}"; then
    echo "Sobol split CSVs do not exactly cover ${SOBOL_FULL_CSV} in split order; regenerate splits before submitting." >&2
    rm -rf "${tmp_dir}"
    return 1
  fi

  rm -rf "${tmp_dir}"
  echo "Sobol split validation passed: ${SOBOL_SPLIT_COUNT} splits, ${SOBOL_SPLIT_ROWS} rows each."
}

check_inputs() {
  local missing=0
  local split

  [[ -f "${SLURM_SCRIPT}" ]] || { echo "Missing SLURM worker script: ${SLURM_SCRIPT}" >&2; missing=1; }
  [[ -f "${SOBOL_FULL_CSV}" ]] || { echo "Missing Sobol CSV: ${SOBOL_FULL_CSV}" >&2; missing=1; }
  [[ -f "${BASELINE_NOISE_PATH}" ]] || { echo "Missing SO baseline noise file: ${BASELINE_NOISE_PATH}" >&2; missing=1; }
  [[ -f "${GOAL_NOISE_PATH}" ]] || { echo "Missing SO goal noise file: ${GOAL_NOISE_PATH}" >&2; missing=1; }

  for split in $(seq 1 "${SOBOL_SPLIT_COUNT}"); do
    local sobol_csv
    sobol_csv="$(sobol_split_csv_path "${split}")"
    [[ -f "${sobol_csv}" ]] || { echo "Missing Sobol split CSV: ${sobol_csv}" >&2; missing=1; }
  done

  if is_true "${RUN_Y100}"; then
    [[ -f "${HALFDOME_BASE_DIR}/lightcone_100.hdf5" ]] || { echo "Missing HalfDome lightcone: ${HALFDOME_BASE_DIR}/lightcone_100.hdf5" >&2; missing=1; }
  fi
  if is_true "${RUN_Y102}"; then
    [[ -f "${HALFDOME_BASE_DIR}/lightcone_102.hdf5" ]] || { echo "Missing HalfDome lightcone: ${HALFDOME_BASE_DIR}/lightcone_102.hdf5" >&2; missing=1; }
  fi

  if [[ "${DRY_RUN}" != "true" ]] && ! command -v "${SBATCH}" >/dev/null 2>&1; then
    echo "Could not find sbatch command: ${SBATCH}" >&2
    missing=1
  fi

  [[ "${missing}" -eq 0 ]] || exit 1
  validate_sobol_splits || exit 1
}

submit_lightcone_array() {
  local lightcone_id="$1"
  local model_exists="$2"
  local dependency="${3:-}"
  local job_name="tSZ_SO_y${lightcone_id}_${JOB_SET_TAG}"

  export PROJECT_DIR JULIA ENV_SETUP
  export SOBOL_BASENAME SOBOL_SPLIT_DIR SOBOL_SPLIT_ROWS
  export CACHE_DIR LOG_DIR
  export BASELINE_NOISE_PATH GOAL_NOISE_PATH
  export NSIDE ELL_MIN ELL_MAX SO_NOISE_DEPROJECTIONS SO_NOISE_IS_DL
  export MASK_FSKY MASK_APODIZATION_ARCMIN SEED
  export SAVE_NO_NOISE_CL SAVE_BASELINE_NOISE_CROSS_CL SAVE_GOAL_NOISE_CROSS_CL SAVE_UNMASKED_NO_NOISE_CL
  export SAVE_NOISE_MAPS SAVE_NOISY_MAPS SAVE_MASK_MAP SAVE_SIGNAL_MAP SAVE_MASKED_SIGNAL_MAP
  export MAX_PARALLEL INTERPOLATOR_PAD INTERPOLATOR_LOGM_MAX CL_NITER REUSE_EXISTING_CACHE
  export CACHE_WAIT_SECONDS CACHE_POLL_SECONDS ENFORCE_BATTAGLIA_GUARDRAILS SKIP_INVALID_BATTAGLIA_ROWS
  export CONTINUE_ON_ROW_ERROR PRINT_RUNTIME_ENVIRONMENT JOB_SET_TAG
  export LIGHTCONE_ID="${lightcone_id}"
  export MODEL_EXISTS="${model_exists}"
  export HALFDOME_PATH="${HALFDOME_BASE_DIR}/lightcone_${lightcone_id}.hdf5"
  export OUTPUT_DIR="${OUTPUT_BASE_DIR}/y${lightcone_id}"
  export SIMULATION_NAME="halfdome_lightcone_${lightcone_id}"

  # Avoid inherited values from interactive debugging or previous jobs
  # defeating the SLURM array mapping.
  unset SOBOL_SPLIT SOBOL_CSV SOBOL_ROW_START SOBOL_ROW_STOP SOBOL_ROW_LIST THREADS_PER_TASK REDO_LOG

  local cmd=(
    "${SBATCH}"
    --export=ALL
    --job-name="${job_name}"
    --array="1-${SOBOL_SPLIT_COUNT}%${ARRAY_CONCURRENCY}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --time="${TIME}"
    --output="${LOG_DIR}/%x_%A_%a.out"
    --error="${LOG_DIR}/%x_%A_%a.err"
  )

  if [[ -n "${SLURM_PARTITION}" ]]; then
    cmd+=(--partition="${SLURM_PARTITION}")
  fi
  if [[ -n "${SLURM_ACCOUNT}" ]]; then
    cmd+=(--account="${SLURM_ACCOUNT}")
  fi
  if [[ -n "${SLURM_QOS}" ]]; then
    cmd+=(--qos="${SLURM_QOS}")
  fi
  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency="afterok:${dependency}")
  fi

  cmd+=("${SLURM_SCRIPT}")

  printf 'Submitting %s: ' "${job_name}" >&2
  printf '%q ' "${cmd[@]}" >&2
  printf '\n' >&2

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'DRYRUN_%s' "${job_name}"
    return 0
  fi

  local submit_output job_id
  submit_output="$("${cmd[@]}")"
  echo "${submit_output}" >&2
  job_id="$(awk '{print $NF}' <<< "${submit_output}")"
  printf '%s' "${job_id}"
}

configure_sobol_split_count
split_sobol_csv

if [[ "${CHECK_INPUTS}" == "true" ]]; then
  check_inputs
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_BASE_DIR}" "${CACHE_DIR}"

echo "Submit configuration:"
echo "  pipeline=SO noise split-map cross-spectrum"
echo "  project_dir=${PROJECT_DIR}"
echo "  sobol_full_csv=${SOBOL_FULL_CSV}"
echo "  split_dir=${SOBOL_SPLIT_DIR}"
echo "  split_count=${SOBOL_SPLIT_COUNT}, split_rows=${SOBOL_SPLIT_ROWS}"
echo "  SO noise baseline=${BASELINE_NOISE_PATH}"
echo "  SO noise goal=${GOAL_NOISE_PATH}"
echo "  deprojections=${SO_NOISE_DEPROJECTIONS}, ell=${ELL_MIN}:${ELL_MAX}"
echo "  mask fsky=${MASK_FSKY}, apodization_arcmin=${MASK_APODIZATION_ARCMIN}"
echo "  outputs=${OUTPUT_BASE_DIR}, cache=${CACHE_DIR}, logs=${LOG_DIR}"
echo "  run_y100=${RUN_Y100}, run_y102=${RUN_Y102}, depend_y102_on_y100=${DEPEND_Y102_ON_Y100}"
echo "  array_concurrency=${ARRAY_CONCURRENCY}, cpus_per_task=${CPUS_PER_TASK}, mem=${MEM}, time=${TIME}"
echo "  debug maps: noise=${SAVE_NOISE_MAPS}, noisy=${SAVE_NOISY_MAPS}, mask=${SAVE_MASK_MAP}, signal=${SAVE_SIGNAL_MAP}, masked_signal=${SAVE_MASKED_SIGNAL_MAP}"

y100_job=""
y102_dependency=""
if is_true "${RUN_Y100}"; then
  y100_job="$(submit_lightcone_array 100 "${Y100_MODEL_EXISTS}")"
  echo "Submitted y100 SO-noise array: ${y100_job}"
fi

if is_true "${RUN_Y102}"; then
  if is_true "${DEPEND_Y102_ON_Y100}" && [[ -n "${y100_job}" ]]; then
    y102_dependency="${y100_job}"
  fi
  y102_job="$(submit_lightcone_array 102 "${Y102_MODEL_EXISTS}" "${y102_dependency}")"
  echo "Submitted y102 SO-noise array: ${y102_job}"
fi

echo "SO-noise submission finished."
