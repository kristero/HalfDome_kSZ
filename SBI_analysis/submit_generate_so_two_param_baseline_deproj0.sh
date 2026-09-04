#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# User-editable design and cluster configuration.
: "${PROJECT_DIR:=${DEFAULT_PROJECT_DIR}}"
: "${PYTHON:=python3}"
: "${QSUB:=qsub}"
: "${RUN_ROOT:=/lustre/work/kristero10/adrian_two_param_so_baseline_deproj0}"
: "${N_SAMPLES:=32768}"
: "${P0_LOW:=1.832524}"
: "${P0_HIGH:=34.341221}"
: "${BETA_LOW:=3.480627}"
: "${BETA_HIGH:=5.216611}"
: "${SOBOL_SEED:=12345}"
: "${SOBOL_SCRAMBLE:=1}"
: "${SOBOL_SEQUENCE_OFFSET:=0}"
: "${REGENERATE_DESIGN:=0}"

: "${MASK_SEED:=12345}"
: "${NOISE_SEED_BASE:=1000000}"
: "${CHUNK_ROWS:=256}"
: "${MAX_ACTIVE_JOBS:=5}"
: "${TEST_LAST_N:=1000}"
: "${DRY_RUN:=0}"

: "${LIGHTCONE_ID:=100}"
: "${HALFDOME_PATH:=/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_${LIGHTCONE_ID}.hdf5}"
: "${BASELINE_NOISE_PATH:=/home/kristero10/tSZ_data/SO_noise/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt}"
: "${GOAL_NOISE_PATH:=/home/kristero10/tSZ_data/SO_noise/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt}"
: "${CACHE_DIR:=/lustre/work/kristero10/tSZ_data/cache}"

DESIGN_DIR="${RUN_ROOT}/design"
SOBOL_CSV="${DESIGN_DIR}/battaglia_sobol_P0_beta_${N_SAMPLES}.csv"
RAW_OUTPUT_DIR="${RUN_ROOT}/raw"
RUN_META_DIR="${RUN_ROOT}/run_manifests"
LOG_DIR="/home/kristero10/logs/tSZ_two_param"
WORKER="${PROJECT_DIR}/SBI_analysis/run_generate_so_two_param_baseline_deproj0.pbs"
COMBINER_PBS="${PROJECT_DIR}/SBI_analysis/run_combine_so_two_param_baseline_deproj0.pbs"
GENERATOR="${PROJECT_DIR}/SBI_analysis/generate_so_two_param_sobol.py"

is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}
for value in "${N_SAMPLES}" "${SOBOL_SEED}" "${SOBOL_SEQUENCE_OFFSET}" \
             "${MASK_SEED}" "${NOISE_SEED_BASE}" "${CHUNK_ROWS}" \
             "${MAX_ACTIVE_JOBS}"; do
  is_uint "${value}" || {
    echo "Expected a non-negative integer, got: ${value}" >&2
    exit 2
  }
done
(( N_SAMPLES >= 2 )) || { echo "N_SAMPLES must be at least 2" >&2; exit 2; }
(( CHUNK_ROWS >= 1 )) || { echo "CHUNK_ROWS must be positive" >&2; exit 2; }
(( MAX_ACTIVE_JOBS >= 1 )) || { echo "MAX_ACTIVE_JOBS must be positive" >&2; exit 2; }

for path in "${GENERATOR}" "${WORKER}" "${COMBINER_PBS}" \
            "${HALFDOME_PATH}" "${BASELINE_NOISE_PATH}" "${GOAL_NOISE_PATH}"; do
  [[ -e "${path}" ]] || {
    echo "Required path does not exist: ${path}" >&2
    exit 2
  }
done
if [[ "${DRY_RUN}" != "1" ]]; then
  command -v "${QSUB}" >/dev/null 2>&1 || {
    echo "Could not find qsub command: ${QSUB}" >&2
    exit 2
  }
fi

mkdir -p "${DESIGN_DIR}" "${RAW_OUTPUT_DIR}" "${RUN_META_DIR}" "${LOG_DIR}" "${CACHE_DIR}"

generator_args=(
  "${GENERATOR}"
  --output-csv "${SOBOL_CSV}"
  --n-samples "${N_SAMPLES}"
  --sobol-seed "${SOBOL_SEED}"
  --sequence-offset "${SOBOL_SEQUENCE_OFFSET}"
  --p0-low "${P0_LOW}"
  --p0-high "${P0_HIGH}"
  --beta-low "${BETA_LOW}"
  --beta-high "${BETA_HIGH}"
  --noise-seed-base "${NOISE_SEED_BASE}"
)
if [[ "${SOBOL_SCRAMBLE}" == "1" ]]; then
  generator_args+=(--scramble)
else
  generator_args+=(--no-scramble)
fi
if [[ "${REGENERATE_DESIGN}" == "1" ]]; then
  if compgen -G "${RUN_META_DIR}/chunk_*.csv" >/dev/null; then
    echo "Refusing to regenerate a design in a run root with simulation manifests." >&2
    echo "Set RUN_ROOT to a new directory when changing priors or the Sobol sequence." >&2
    exit 2
  fi
  generator_args+=(--force)
fi

if [[ ! -f "${SOBOL_CSV}" || "${REGENERATE_DESIGN}" == "1" ]]; then
  echo "Generating two-parameter Sobol design..."
  "${PYTHON}" "${generator_args[@]}"
else
  echo "Reusing existing Sobol design: ${SOBOL_CSV}"
  "${PYTHON}" "${generator_args[@]}" --validate-existing
fi

chunk_count=$(( (N_SAMPLES + CHUNK_ROWS - 1) / CHUNK_ROWS ))
echo "Submitting ${chunk_count} separate generation jobs in ${MAX_ACTIVE_JOBS} dependency lanes."
echo "At most ${MAX_ACTIVE_JOBS} generation jobs can be runnable simultaneously."
echo "Rows per job: ${CHUNK_ROWS}"
echo "Run root: ${RUN_ROOT}"

declare -a lane_last
for (( lane=0; lane<MAX_ACTIVE_JOBS; lane++ )); do
  lane_last[lane]=""
done

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

for (( chunk=0; chunk<chunk_count; chunk++ )); do
  row_start=$(( chunk * CHUNK_ROWS + 1 ))
  row_stop=$(( row_start + CHUNK_ROWS - 1 ))
  (( row_stop > N_SAMPLES )) && row_stop="${N_SAMPLES}"
  lane=$(( chunk % MAX_ACTIVE_JOBS ))
  job_name="$(printf 'S2p%03d' "${chunk}")"

  variables="PROJECT_DIR=${PROJECT_DIR},LIGHTCONE_ID=${LIGHTCONE_ID},HALFDOME_PATH=${HALFDOME_PATH},SOBOL_CSV=${SOBOL_CSV},BASELINE_NOISE_PATH=${BASELINE_NOISE_PATH},GOAL_NOISE_PATH=${GOAL_NOISE_PATH},RAW_OUTPUT_DIR=${RAW_OUTPUT_DIR},RUN_META_DIR=${RUN_META_DIR},CACHE_DIR=${CACHE_DIR},ROW_START=${row_start},ROW_STOP=${row_stop},MASK_SEED=${MASK_SEED},NOISE_SEED_BASE=${NOISE_SEED_BASE}"

  qsub_args=(-N "${job_name}" -v "${variables}")
  if [[ -n "${lane_last[lane]}" ]]; then
    qsub_args+=(-W "depend=afterok:${lane_last[lane]}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo -n "DRY RUN:"
    print_command "${QSUB}" "${qsub_args[@]}" "${WORKER}"
    job_id="dry_lane${lane}_chunk${chunk}"
  else
    job_id="$("${QSUB}" "${qsub_args[@]}" "${WORKER}")"
    echo "Submitted chunk ${chunk}: rows ${row_start}-${row_stop}, lane ${lane}, job ${job_id}"
  fi
  lane_last[lane]="${job_id}"
done

dependency_ids=()
for (( lane=0; lane<MAX_ACTIVE_JOBS; lane++ )); do
  if [[ -n "${lane_last[lane]}" ]]; then
    dependency_ids+=("${lane_last[lane]}")
  fi
done
dependency="$(IFS=:; echo "${dependency_ids[*]}")"
combine_vars="PROJECT_DIR=${PROJECT_DIR},PYTHON=${PYTHON},RUN_ROOT=${RUN_ROOT},SOBOL_CSV=${SOBOL_CSV},MANIFEST_DIR=${RUN_META_DIR},OUTPUT_DIR=${RUN_ROOT}/prepared,MASK_SEED=${MASK_SEED},NOISE_SEED_BASE=${NOISE_SEED_BASE},TEST_LAST_N=${TEST_LAST_N}"
combine_args=(-N S2pCombine -W "depend=afterok:${dependency}" -v "${combine_vars}")

if [[ "${DRY_RUN}" == "1" ]]; then
  echo -n "DRY RUN:"
  print_command "${QSUB}" "${combine_args[@]}" "${COMBINER_PBS}"
else
  combine_job="$("${QSUB}" "${combine_args[@]}" "${COMBINER_PBS}")"
  echo "Submitted dependent combine job: ${combine_job}"
fi

echo "Design: ${SOBOL_CSV}"
echo "Raw spectra: ${RAW_OUTPUT_DIR}"
echo "Prepared dataset will be written under: ${RUN_ROOT}/prepared"
