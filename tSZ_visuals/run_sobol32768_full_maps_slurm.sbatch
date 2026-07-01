#!/usr/bin/env bash
#SBATCH --job-name=tSZ_SO_s32768
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

is_true() {
  case "${1:-}" in
    true|TRUE|True|1|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

# =========================
# USER EDITS / cluster-specific paths
# These can be edited here, but the submit_sobol32768_slurm.sh helper normally
# exports them. Prefer editing the submitter unless running this worker directly.
# =========================
: "${PROJECT_DIR:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${JULIA:=julia}"
: "${ENV_SETUP:=}"
: "${LIGHTCONE_ID:=100}"
: "${HALFDOME_BASE_DIR:=/lustre/work/Globus-lt/halfdome/full_res/halos}"
: "${HALFDOME_PATH:=${HALFDOME_BASE_DIR}/lightcone_${LIGHTCONE_ID}.hdf5}"
: "${OUTPUT_BASE_DIR:=/lustre/work/${USER}/tSZ_data/so_noise_s32768}"
: "${OUTPUT_DIR:=${OUTPUT_BASE_DIR}/y${LIGHTCONE_ID}}"
: "${CACHE_DIR:=/lustre/work/${USER}/tSZ_data/cache}"
: "${LOG_DIR:=${HOME}/logs/tSZ_so_noise}"
: "${SO_NOISE_DIR:=${PROJECT_DIR}/other_sims/SO}"
: "${BASELINE_NOISE_PATH:=${SO_NOISE_DIR}/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt}"
: "${GOAL_NOISE_PATH:=${SO_NOISE_DIR}/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt}"

# =========================
# Dataset/split configuration
# =========================
: "${SOBOL_SPLIT:=${SLURM_ARRAY_TASK_ID:-1}}"
: "${SOBOL_BASE_DIR:=${PROJECT_DIR}/Sobol_tSZ}"
: "${SOBOL_BASENAME:=battaglia_sobol_32768}"
: "${SOBOL_SPLIT_DIR:=${SOBOL_BASE_DIR}/splits_${SOBOL_BASENAME}}"
: "${SOBOL_CSV:=${SOBOL_SPLIT_DIR}/${SOBOL_BASENAME}_${SOBOL_SPLIT}.csv}"
: "${SOBOL_SPLIT_ROWS:=128}"

# =========================
# SO-noise science configuration
# =========================
: "${SIMULATION_NAME:=halfdome_lightcone_${LIGHTCONE_ID}}"
: "${NSIDE:=4096}"
: "${ELL_MIN:=80}"
: "${ELL_MAX:=7979}"
: "${SO_NOISE_DEPROJECTIONS:=0,3}"
: "${SO_NOISE_IS_DL:=false}"
: "${MASK_FSKY:=0.4}"
: "${MASK_APODIZATION_ARCMIN:=60.0}"
: "${SEED:=12345}"

# C_l profile save flags. Profiles are saved as .npy only by the Julia code.
: "${SAVE_NO_NOISE_CL:=true}"
: "${SAVE_BASELINE_NOISE_CROSS_CL:=true}"
: "${SAVE_GOAL_NOISE_CROSS_CL:=true}"
: "${SAVE_UNMASKED_NO_NOISE_CL:=false}"

# Debug map save flags. Keep all false for production; FITS maps are large.
: "${SAVE_NOISE_MAPS:=false}"
: "${SAVE_NOISY_MAPS:=false}"
: "${SAVE_MASK_MAP:=false}"
: "${SAVE_SIGNAL_MAP:=false}"
: "${SAVE_MASKED_SIGNAL_MAP:=false}"

# =========================
# Runtime/cache configuration
# =========================
: "${THREADS_PER_TASK:=${SLURM_CPUS_PER_TASK:-25}}"
: "${MAX_PARALLEL:=1}"
: "${MODEL_EXISTS:=false}"
: "${REUSE_EXISTING_CACHE:=false}"
: "${CACHE_WAIT_SECONDS:=0}"
: "${CACHE_POLL_SECONDS:=30}"
: "${INTERPOLATOR_PAD:=256}"
: "${INTERPOLATOR_LOGM_MAX:=15.7}"
: "${CL_NITER:=0}"
: "${ENFORCE_BATTAGLIA_GUARDRAILS:=true}"
: "${SKIP_INVALID_BATTAGLIA_ROWS:=true}"
: "${CONTINUE_ON_ROW_ERROR:=true}"
: "${PRINT_RUNTIME_ENVIRONMENT:=false}"
: "${JOB_SET_TAG:=so_s32768}"
: "${REDO_LOG:=${LOG_DIR}/tSZ_SO_sobol32768_y${LIGHTCONE_ID}_split${SOBOL_SPLIT}_${SLURM_JOB_ID:-manual}.redo.log}"
: "${OMP_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"

cd "${PROJECT_DIR}"

if [[ -n "${ENV_SETUP}" ]]; then
  echo "Running environment setup command."
  eval "${ENV_SETUP}"
fi

SCRIPT_CANDIDATE_1="${PROJECT_DIR}/run_halfdome_fullsky_so_noise.jl"
SCRIPT_CANDIDATE_2="${PROJECT_DIR}/tSZ_visuals/run_halfdome_fullsky_so_noise.jl"
if [[ -f "${SCRIPT_CANDIDATE_1}" ]]; then
  SCRIPT="${SCRIPT_CANDIDATE_1}"
elif [[ -f "${SCRIPT_CANDIDATE_2}" ]]; then
  SCRIPT="${SCRIPT_CANDIDATE_2}"
else
  echo "Could not locate run_halfdome_fullsky_so_noise.jl from PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 1
fi

for required_path in "${SCRIPT}" "${SOBOL_CSV}" "${HALFDOME_PATH}" "${BASELINE_NOISE_PATH}" "${GOAL_NOISE_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${CACHE_DIR}"
: > "${REDO_LOG}"

export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export ENFORCE_BATTAGLIA_GUARDRAILS SKIP_INVALID_BATTAGLIA_ROWS

ROW_COUNT="$(awk 'END {print NR-1}' "${SOBOL_CSV}")"
if [[ "${ROW_COUNT}" -lt 1 ]]; then
  echo "Sobol split CSV has no data rows: ${SOBOL_CSV}" >&2
  exit 1
fi

rows_to_run=()
if [[ -n "${SOBOL_ROW_LIST:-}" ]]; then
  row_list_normalized="${SOBOL_ROW_LIST//,/ }"
  row_list_normalized="${row_list_normalized//:/ }"
  row_list_normalized="${row_list_normalized//;/ }"
  for row_item in ${row_list_normalized}; do
    if [[ ! "${row_item}" =~ ^[0-9]+$ ]]; then
      echo "SOBOL_ROW_LIST contains a non-integer row value: ${row_item}" >&2
      exit 1
    fi
    row_value=$((10#${row_item}))
    if [[ "${row_value}" -lt 1 || "${row_value}" -gt "${ROW_COUNT}" ]]; then
      echo "SOBOL_ROW_LIST row ${row_value} is outside 1:${ROW_COUNT} for ${SOBOL_CSV}" >&2
      exit 1
    fi
    rows_to_run+=("${row_value}")
  done
  row_selection_display="list(${#rows_to_run[@]})=${SOBOL_ROW_LIST}"
else
  : "${SOBOL_ROW_START:=1}"
  : "${SOBOL_ROW_STOP:=${ROW_COUNT}}"
  if [[ "${SOBOL_ROW_START}" -lt 1 || "${SOBOL_ROW_STOP}" -lt "${SOBOL_ROW_START}" || "${SOBOL_ROW_STOP}" -gt "${ROW_COUNT}" ]]; then
    echo "Invalid row range ${SOBOL_ROW_START}:${SOBOL_ROW_STOP} for ${ROW_COUNT} rows in ${SOBOL_CSV}" >&2
    exit 1
  fi
  row_item="${SOBOL_ROW_START}"
  while [[ "${row_item}" -le "${SOBOL_ROW_STOP}" ]]; do
    rows_to_run+=("${row_item}")
    row_item=$((row_item + 1))
  done
  row_selection_display="${SOBOL_ROW_START}:${SOBOL_ROW_STOP}/${ROW_COUNT}"
fi

REQUESTED_ROW_COUNT="${#rows_to_run[@]}"
if [[ "${REQUESTED_ROW_COUNT}" -lt 1 ]]; then
  echo "No rows selected." >&2
  exit 1
fi

if [[ "${REQUESTED_ROW_COUNT}" -lt "${MAX_PARALLEL}" ]]; then
  CONCURRENT_JOBS="${REQUESTED_ROW_COUNT}"
else
  CONCURRENT_JOBS="${MAX_PARALLEL}"
fi

if [[ "${CONCURRENT_JOBS}" -gt 1 ]]; then
  THREADS_PER_TASK=$(( THREADS_PER_TASK / CONCURRENT_JOBS ))
  if [[ "${THREADS_PER_TASK}" -lt 1 ]]; then
    THREADS_PER_TASK=1
  fi
fi

echo "======================================="
echo "SLURM job id: ${SLURM_JOB_ID:-NA}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Project dir: ${PROJECT_DIR}"
echo "Julia script: ${SCRIPT}"
echo "Julia executable: ${JULIA}"
"${JULIA}" --version
echo "Lightcone id: ${LIGHTCONE_ID}"
echo "Simulation name: ${SIMULATION_NAME}"
echo "Sobol split: ${SOBOL_SPLIT}"
echo "Sobol CSV: ${SOBOL_CSV}"
echo "Sobol rows selected: ${row_selection_display}"
echo "HalfDome path: ${HALFDOME_PATH}"
echo "Baseline SO noise: ${BASELINE_NOISE_PATH}"
echo "Goal SO noise: ${GOAL_NOISE_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "NSIDE: ${NSIDE}"
echo "SO native ell_min: ${ELL_MIN}"
echo "C_l/noise ell_max: ${ELL_MAX}"
echo "SO deprojections: ${SO_NOISE_DEPROJECTIONS}"
echo "Mask fsky: ${MASK_FSKY}"
echo "Mask apodization arcmin: ${MASK_APODIZATION_ARCMIN}"
echo "Seed: ${SEED}"
echo "Model exists: ${MODEL_EXISTS}"
echo "Reuse cache: ${REUSE_EXISTING_CACHE}"
echo "Concurrent Julia tasks: ${CONCURRENT_JOBS}"
echo "Threads per Julia task: ${THREADS_PER_TASK}"
echo "Debug maps: noise=${SAVE_NOISE_MAPS}, noisy=${SAVE_NOISY_MAPS}, mask=${SAVE_MASK_MAP}, signal=${SAVE_SIGNAL_MAP}, masked_signal=${SAVE_MASKED_SIGNAL_MAP}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "======================================="

log_redo_required() {
  local sobol_row="$1"
  local reason="$2"
  local detail="${3:-}"
  local message
  message="REDO_REQUIRED lightcone=${LIGHTCONE_ID} split=${SOBOL_SPLIT} sobol_row=${sobol_row} reason=${reason}"
  if [[ -n "${detail}" ]]; then
    message+=" ${detail}"
  fi
  echo "${message}"
  echo "${message}" >> "${REDO_LOG}"
}

run_so_row() {
  local sobol_row="$1"

  "${JULIA}" --threads="${THREADS_PER_TASK}" "${SCRIPT}" \
    simulation_name="${SIMULATION_NAME}" \
    halfdome_path="${HALFDOME_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    cache_dir="${CACHE_DIR}" \
    sobol_csv_path="${SOBOL_CSV}" \
    sobol_row="${sobol_row}" \
    baseline_noise_path="${BASELINE_NOISE_PATH}" \
    goal_noise_path="${GOAL_NOISE_PATH}" \
    nside="${NSIDE}" \
    cl_lmax="${ELL_MAX}" \
    so_noise_lmax="${ELL_MAX}" \
    cl_niter="${CL_NITER}" \
    so_noise_deprojections="${SO_NOISE_DEPROJECTIONS}" \
    so_noise_is_dl="${SO_NOISE_IS_DL}" \
    mask_fsky="${MASK_FSKY}" \
    mask_apodization_arcmin="${MASK_APODIZATION_ARCMIN}" \
    seed="${SEED}" \
    model_exists="${MODEL_EXISTS}" \
    reuse_existing_cache="${REUSE_EXISTING_CACHE}" \
    cache_wait_seconds="${CACHE_WAIT_SECONDS}" \
    cache_poll_seconds="${CACHE_POLL_SECONDS}" \
    interpolator_pad="${INTERPOLATOR_PAD}" \
    interpolator_logM_max="${INTERPOLATOR_LOGM_MAX}" \
    enforce_battaglia_guardrails="${ENFORCE_BATTAGLIA_GUARDRAILS}" \
    skip_invalid_battaglia_rows="${SKIP_INVALID_BATTAGLIA_ROWS}" \
    apply_gaussian_beam=false \
    gaussian_beam_fwhm_arcmin=0.0 \
    save_no_noise_cl="${SAVE_NO_NOISE_CL}" \
    save_baseline_noise_cross_cl="${SAVE_BASELINE_NOISE_CROSS_CL}" \
    save_goal_noise_cross_cl="${SAVE_GOAL_NOISE_CROSS_CL}" \
    save_unmasked_no_noise_cl="${SAVE_UNMASKED_NO_NOISE_CL}" \
    save_noise_maps="${SAVE_NOISE_MAPS}" \
    save_noisy_maps="${SAVE_NOISY_MAPS}" \
    save_mask_map="${SAVE_MASK_MAP}" \
    save_signal_map="${SAVE_SIGNAL_MAP}" \
    save_masked_signal_map="${SAVE_MASKED_SIGNAL_MAP}" \
    print_runtime_environment="${PRINT_RUNTIME_ENVIRONMENT}"
}

run_one_row() {
  local sobol_row="$1"
  local run_status=0
  local row_wall_start
  local global_sobol_row=$(( (SOBOL_SPLIT - 1) * SOBOL_SPLIT_ROWS + sobol_row ))

  set +e
  row_wall_start="$(date +%s)"
  export JULIA_NUM_THREADS="${THREADS_PER_TASK}"
  echo "SO row start: lightcone=${LIGHTCONE_ID}, split=${SOBOL_SPLIT}, sobol_row=${sobol_row}, global_sobol_row=${global_sobol_row}, threads=${THREADS_PER_TASK}."

  run_so_row "${sobol_row}"
  run_status=$?
  if [[ "${run_status}" -ne 0 ]]; then
    log_redo_required "${sobol_row}" "so_noise_failed" "exit_status=${run_status}"
    if is_true "${CONTINUE_ON_ROW_ERROR}"; then
      echo "Row wall elapsed time: $(( $(date +%s) - row_wall_start )) s"
      return 0
    fi
    echo "Row wall elapsed time: $(( $(date +%s) - row_wall_start )) s"
    return "${run_status}"
  fi

  echo "SO row finished: sobol_row=${sobol_row}."
  echo "Row wall elapsed time: $(( $(date +%s) - row_wall_start )) s"
  return 0
}

row_index=0
while [[ "${row_index}" -lt "${REQUESTED_ROW_COUNT}" ]]; do
  pids=()
  batch_rows=()

  for ((slot = 0; slot < CONCURRENT_JOBS && row_index < REQUESTED_ROW_COUNT; slot++)); do
    row="${rows_to_run[$row_index]}"
    row_tag="$(printf '%04d' "${row}")"
    log_path="${LOG_DIR}/tSZ_SO_y${LIGHTCONE_ID}_${JOB_SET_TAG}_split${SOBOL_SPLIT}_row${row_tag}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}.log"

    echo "Launching SO sobol_row=${row} with ${THREADS_PER_TASK} threads; log=${log_path}"
    run_one_row "${row}" > "${log_path}" 2>&1 &

    pids+=("$!")
    batch_rows+=("${row}")
    row_index=$((row_index + 1))
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      failed_row="${batch_rows[$idx]}"
      failed_tag="$(printf '%04d' "${failed_row}")"
      echo "SO Sobol row ${failed_row} failed. Check ${LOG_DIR}/tSZ_SO_y${LIGHTCONE_ID}_${JOB_SET_TAG}_split${SOBOL_SPLIT}_row${failed_tag}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}.log" >&2
      exit 1
    fi
  done
done

if [[ -s "${REDO_LOG}" ]]; then
  redo_count="$(wc -l < "${REDO_LOG}")"
  echo "Rows requiring redo: ${redo_count}. See ${REDO_LOG}"
fi

echo "SO-noise job finished at $(date -Is)."
