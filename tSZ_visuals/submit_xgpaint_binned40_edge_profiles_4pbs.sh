#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =========================
# USER EDITS / target cluster
# =========================
: "${QSUB:=qsub}"
: "${PROJECT_DIR:=${PROJECT_DIR_DEFAULT}}"
: "${JULIA:=/usr/local/bin/julia}"
: "${ENV_SETUP:=}"  # e.g. 'module load julia' or 'source /path/to/julia_env.sh'
: "${CLUSTER_USER:=${USER:-kristero10}}"
: "${LIGHTCONE_ID:=100}"

# Required HalfDome input.
: "${HALFDOME_PATH:=/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_${LIGHTCONE_ID}.hdf5}"

# All four jobs write profiles into this same directory.
: "${OUTPUT_DIR:=/lustre/work/${CLUSTER_USER}/tSZ_data/xgpaint_binned40_edge_profiles/y${LIGHTCONE_ID}}"
: "${CACHE_DIR:=/lustre/work/${CLUSTER_USER}/tSZ_data/cache}"
: "${LOG_DIR:=/home/${CLUSTER_USER}/logs/tSZ_xgpaint_binned40}"

# PBS resources.
: "${PBS_QUEUE:=mini}"
: "${NCPUS:=26}"
: "${MEM:=128gb}"
: "${WALLTIME:=23:59:00}"

# XGPaint/full-map settings.
: "${SIMULATION_NAME:=halfdome_lightcone_${LIGHTCONE_ID}}"
: "${NSIDE:=4096}"
: "${CL_LMAX:=7979}"
: "${CL_NITER:=0}"
: "${CHUNKN:=2000000}"
: "${MASS_MIN:=1.0e12}"
: "${APPLY_MASS_CUT:=true}"

# Beam setting. 2 arcsec = 2/60 arcmin = 0.03333333333333333.
# For no beam, set APPLY_GAUSSIAN_BEAM=false and GAUSSIAN_BEAM_FWHM_ARCMIN=0.
# For 2 arcmin, set GAUSSIAN_BEAM_FWHM_ARCMIN=2.0.
: "${APPLY_GAUSSIAN_BEAM:=true}"
: "${GAUSSIAN_BEAM_FWHM_ARCMIN:=0.03333333333333333}"

# Cache/runtime behavior.
: "${THREADS_PER_TASK:=${NCPUS}}"
: "${MODEL_EXISTS:=false}"
: "${REUSE_EXISTING_CACHE:=true}"
: "${CACHE_WAIT_SECONDS:=0}"
: "${CACHE_POLL_SECONDS:=30}"
: "${INTERPOLATOR_PAD:=128}"
: "${INTERPOLATOR_LOGM_MAX:=15.7}"
: "${ENFORCE_BATTAGLIA_GUARDRAILS:=false}"
: "${SKIP_INVALID_BATTAGLIA_ROWS:=false}"
: "${OVERWRITE_EDGE_PROFILE:=true}"
: "${SAVE_EDGE_METADATA:=false}"
: "${PRINT_RUNTIME_ENVIRONMENT:=false}"
: "${OMP_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"

# Four normal PBS jobs. Each job runs its list sequentially.
PROFILE_GROUPS=(
  "1 2 3 4 5"
  "6 7 8 9 10"
  "11 12 13 14 15"
  "16 17 18 19"
)

: "${GENERATED_PBS_DIR:=${PROJECT_DIR}/tSZ_visuals/generated_pbs/xgpaint_binned40_edge_profiles}"
: "${DRY_RUN:=false}"
: "${OVERWRITE_GENERATED_PBS:=true}"

is_true() {
  case "${1:-}" in
    true|TRUE|True|1|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

shell_quote() {
  printf '%q' "$1"
}

write_export() {
  local key="$1"
  local value="$2"
  printf 'export %s=%s\n' "${key}" "$(shell_quote "${value}")"
}

check_inputs() {
  local worker="${PROJECT_DIR}/tSZ_visuals/run_xgpaint_binned40_edge_profiles.pbs"
  local julia_script="${PROJECT_DIR}/tSZ_visuals/run_xgpaint_binned40_edge_profiles.jl"

  [[ -f "${worker}" ]] || { echo "Missing PBS worker: ${worker}" >&2; exit 2; }
  [[ -f "${julia_script}" ]] || { echo "Missing Julia script: ${julia_script}" >&2; exit 2; }
  [[ -e "${HALFDOME_PATH}" ]] || { echo "Missing HalfDome input: ${HALFDOME_PATH}" >&2; exit 2; }
  if ! is_true "${DRY_RUN}" && ! command -v "${QSUB}" >/dev/null 2>&1; then
    echo "Could not find qsub command: ${QSUB}" >&2
    exit 2
  fi
}

write_group_pbs() {
  local group_index="$1"
  local profile_list="$2"
  local job_name="xgp_b40_g${group_index}"
  local job_file="${GENERATED_PBS_DIR}/${job_name}.pbs"
  local pbs_stdout="${LOG_DIR}/${job_name}.pbs.out"
  local pbs_stderr="${LOG_DIR}/${job_name}.pbs.err"

  if [[ -f "${job_file}" ]] && ! is_true "${OVERWRITE_GENERATED_PBS}"; then
    echo "Refusing to overwrite existing generated PBS file: ${job_file}" >&2
    exit 2
  fi

  {
    echo '#!/bin/bash'
    echo "#PBS -N ${job_name}"
    echo "#PBS -q ${PBS_QUEUE}"
    echo "#PBS -l select=1:ncpus=${NCPUS}:mpiprocs=1:mem=${MEM}"
    echo "#PBS -l walltime=${WALLTIME}"
    echo "#PBS -o ${pbs_stdout}"
    echo "#PBS -e ${pbs_stderr}"
    echo '#PBS -S /bin/bash'
    echo
    echo 'set -euo pipefail'
    echo
    write_export PROJECT_DIR "${PROJECT_DIR}"
    write_export JULIA "${JULIA}"
    write_export ENV_SETUP "${ENV_SETUP}"
    write_export CLUSTER_USER "${CLUSTER_USER}"
    write_export LIGHTCONE_ID "${LIGHTCONE_ID}"
    write_export HALFDOME_PATH "${HALFDOME_PATH}"
    write_export OUTPUT_DIR "${OUTPUT_DIR}"
    write_export CACHE_DIR "${CACHE_DIR}"
    write_export LOG_DIR "${LOG_DIR}"
    write_export SIMULATION_NAME "${SIMULATION_NAME}"
    write_export NSIDE "${NSIDE}"
    write_export CL_LMAX "${CL_LMAX}"
    write_export CL_NITER "${CL_NITER}"
    write_export CHUNKN "${CHUNKN}"
    write_export MASS_MIN "${MASS_MIN}"
    write_export APPLY_MASS_CUT "${APPLY_MASS_CUT}"
    write_export APPLY_GAUSSIAN_BEAM "${APPLY_GAUSSIAN_BEAM}"
    write_export GAUSSIAN_BEAM_FWHM_ARCMIN "${GAUSSIAN_BEAM_FWHM_ARCMIN}"
    write_export THREADS_PER_TASK "${THREADS_PER_TASK}"
    write_export MODEL_EXISTS "${MODEL_EXISTS}"
    write_export REUSE_EXISTING_CACHE "${REUSE_EXISTING_CACHE}"
    write_export CACHE_WAIT_SECONDS "${CACHE_WAIT_SECONDS}"
    write_export CACHE_POLL_SECONDS "${CACHE_POLL_SECONDS}"
    write_export INTERPOLATOR_PAD "${INTERPOLATOR_PAD}"
    write_export INTERPOLATOR_LOGM_MAX "${INTERPOLATOR_LOGM_MAX}"
    write_export ENFORCE_BATTAGLIA_GUARDRAILS "${ENFORCE_BATTAGLIA_GUARDRAILS}"
    write_export SKIP_INVALID_BATTAGLIA_ROWS "${SKIP_INVALID_BATTAGLIA_ROWS}"
    write_export OVERWRITE_EDGE_PROFILE "${OVERWRITE_EDGE_PROFILE}"
    write_export SAVE_EDGE_METADATA "${SAVE_EDGE_METADATA}"
    write_export PRINT_RUNTIME_ENVIRONMENT "${PRINT_RUNTIME_ENVIRONMENT}"
    write_export OMP_NUM_THREADS "${OMP_NUM_THREADS}"
    write_export OPENBLAS_NUM_THREADS "${OPENBLAS_NUM_THREADS}"
    write_export MKL_NUM_THREADS "${MKL_NUM_THREADS}"
    write_export EDGE_PROFILE_LIST "${profile_list}"
    echo
    echo 'bash "${PROJECT_DIR}/tSZ_visuals/run_xgpaint_binned40_edge_profiles.pbs"'
  } > "${job_file}"

  chmod +x "${job_file}"
  printf '%s\n' "${job_file}"
}

check_inputs
mkdir -p "${GENERATED_PBS_DIR}" "${OUTPUT_DIR}" "${CACHE_DIR}" "${LOG_DIR}"

echo "Submitting four independent PBS jobs."
echo "All profile .npy outputs will be written to: ${OUTPUT_DIR}"
echo "Generated PBS directory: ${GENERATED_PBS_DIR}"

submitted_jobs=()
for group_idx in "${!PROFILE_GROUPS[@]}"; do
  group_number=$((group_idx + 1))
  profile_list="${PROFILE_GROUPS[group_idx]}"
  job_file="$(write_group_pbs "${group_number}" "${profile_list}")"

  echo "Group ${group_number}: profiles ${profile_list}"
  echo "  PBS: ${job_file}"
  if is_true "${DRY_RUN}"; then
    echo "  DRY_RUN=true, not submitting."
  else
    job_id="$("${QSUB}" "${job_file}")"
    submitted_jobs+=("${job_id}")
    echo "  Submitted: ${job_id}"
  fi
done

if ! is_true "${DRY_RUN}"; then
  echo "Submitted job ids:"
  printf '  %s\n' "${submitted_jobs[@]}"
fi
