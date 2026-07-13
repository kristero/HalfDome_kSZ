#!/bin/bash
set -euo pipefail

# Simple cluster runner for observed FRB LOS-DM maps.
#
# Default run list:
#   alpha_star=1.0 : stellar-mass-weighted FRB hosts
#   alpha_star=0.0 : comparable random host halos from the same redshift shell
#
# Edit the block below for your cluster paths and science settings.

# -------------------------
# Paths
# -------------------------
: "${PROJECT_DIR:=${PBS_O_WORKDIR:-$PWD}}"
: "${JULIA:=julia}"
: "${ENV_SETUP:=}"  # example: 'module load julia' or 'source /path/to/env.sh'
: "${LIGHTCONE_ID:=100}"
: "${HALFDOME_PATH:=${PROJECT_DIR}/lightcone_100.hdf5}"
: "${OUTPUT_BASE:=${PROJECT_DIR}/frb_map_generation/outputs/stellar_weighted_frb_los_cluster}"
: "${LOG_DIR:=${OUTPUT_BASE}/logs}"

# -------------------------
# Science settings
# -------------------------
: "${NFRB:=10000}"
: "${ZSOURCE:=1.0}"
: "${DZ:=0.02}"
: "${SOURCE_SELECTION_MODE:=shell}"
: "${SOURCE_Z_MIN:=0.0}"
: "${SOURCE_Z_MAX:=Inf}"
: "${SOURCE_HALO_MASS_MIN:=0.0}"
: "${SOURCE_HALO_MASS_MAX:=Inf}"
: "${ALPHA_STAR_LIST:=1.0 0.0}"
: "${SEED:=42}"
: "${NSIDE:=4096}"

# Host stellar mass.
# 'computed' uses the analytic Mstar(Mh,z) relation because HalfDome has no Mstar field.
: "${STELLAR_MASS_FIELD:=computed}"
: "${STELLAR_MASS_DIVIDE_BY_H:=false}"

# Foreground LOS DM settings.
: "${Z_MIN_FOREGROUND:=0.0}"
case "${SOURCE_SELECTION_MODE}" in
  all|all_redshifts|allredshifts|full) DEFAULT_Z_MAX_FOREGROUND="Inf" ;;
  *) DEFAULT_Z_MAX_FOREGROUND="${ZSOURCE}" ;;
esac
: "${Z_MAX_FOREGROUND:=${DEFAULT_Z_MAX_FOREGROUND}}"
: "${HALO_MASS_MIN:=0.0}"
: "${HALO_MASS_MAX:=Inf}"

# Runtime settings.
: "${THREADS_PER_TASK:=26}"
: "${CHUNKN:=1000000}"
: "${DM_HIST_BINS:=60}"
: "${FRB_OVERLAP_MODE:=mean}"
: "${DM_CACHE_OVERWRITE:=false}"
: "${DM_CLEANUP_NONPOSITIVE:=true}"
: "${SAVE_SHELL_PROBABILITIES:=false}"
: "${SAVE_FOREGROUND_MAP:=true}"
: "${FOREGROUND_PROGRESS_EVERY:=5}"
: "${SAVE_POWER_SPECTRUM:=true}"
: "${CL_LMAX:=$((3 * NSIDE - 1))}"
: "${CL_NITER:=0}"
: "${SUBTRACT_CL_MEAN:=true}"
: "${SAVE_FRB_CORRECTED_ESTIMATOR:=true}"
: "${FRB_CORRECTED_LMAX:=${CL_LMAX}}"
: "${FRB_CORRECTED_SHOT_NOISE:=shuffle}"
: "${FRB_CORRECTED_N_SHUFFLE:=5}"
: "${FRB_CORRECTED_SUBTRACT_SAMPLE_MEAN:=true}"
: "${FRB_CORRECTED_SEED:=${SEED}}"

# Keep BLAS libraries from oversubscribing CPU threads.
: "${OMP_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"

cd "${PROJECT_DIR}"

if [[ -n "${ENV_SETUP}" ]]; then
  echo "Running environment setup: ${ENV_SETUP}"
  eval "${ENV_SETUP}"
fi

SCRIPT="${PROJECT_DIR}/frb_map_generation/make_stellar_weighted_frb_los_dm_map.jl"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Could not find Julia script: ${SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${HALFDOME_PATH}" ]]; then
  echo "Could not find HalfDome catalog: ${HALFDOME_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export JULIA_NUM_THREADS="${THREADS_PER_TASK}"

JOB_TAG="${PBS_JOBID:-manual}"
JOB_TAG="${JOB_TAG//[/_}"
JOB_TAG="${JOB_TAG//]/_}"

echo "======================================="
echo "Stellar-weighted FRB LOS-DM cluster run"
echo "Date: $(date -Is)"
echo "Host: $(hostname)"
echo "PBS job id: ${PBS_JOBID:-NA}"
echo "Project dir: ${PROJECT_DIR}"
echo "Julia: ${JULIA}"
"${JULIA}" --version
echo "Julia threads: ${THREADS_PER_TASK}"
echo "Script: ${SCRIPT}"
echo "HalfDome path: ${HALFDOME_PATH}"
echo "Output base: ${OUTPUT_BASE}"
echo "Log dir: ${LOG_DIR}"
echo "NFRB: ${NFRB}"
echo "ZSOURCE: ${ZSOURCE}"
echo "DZ: ${DZ}"
echo "SOURCE_SELECTION_MODE: ${SOURCE_SELECTION_MODE}"
echo "SOURCE_Z_MIN/MAX: [${SOURCE_Z_MIN}, ${SOURCE_Z_MAX})"
echo "SOURCE_HALO_MASS_MIN/MAX: [${SOURCE_HALO_MASS_MIN}, ${SOURCE_HALO_MASS_MAX})"
echo "ALPHA_STAR_LIST: ${ALPHA_STAR_LIST}"
echo "NSIDE: ${NSIDE}"
echo "CHUNKN: ${CHUNKN}"
echo "STELLAR_MASS_FIELD: ${STELLAR_MASS_FIELD}"
echo "Foreground z range: [${Z_MIN_FOREGROUND}, ${Z_MAX_FOREGROUND})"
echo "Foreground mass range: [${HALO_MASS_MIN}, ${HALO_MASS_MAX})"
echo "Save continuous foreground map: ${SAVE_FOREGROUND_MAP}"
echo "Power spectrum: save=${SAVE_POWER_SPECTRUM}, lmax=${CL_LMAX}, niter=${CL_NITER}, subtract_mean=${SUBTRACT_CL_MEAN}"
echo "Corrected FRB estimator: save=${SAVE_FRB_CORRECTED_ESTIMATOR}, lmax=${FRB_CORRECTED_LMAX}, shot_noise=${FRB_CORRECTED_SHOT_NOISE}, n_shuffle=${FRB_CORRECTED_N_SHUFFLE}"
echo "======================================="

alpha_list_normalized="${ALPHA_STAR_LIST//,/ }"
alpha_list_normalized="${alpha_list_normalized//;/ }"

for alpha_star in ${alpha_list_normalized}; do
  alpha_tag="${alpha_star//./p}"
  alpha_tag="${alpha_tag//-/m}"
  if [[ "${alpha_star}" == "0" || "${alpha_star}" == "0.0" || "${alpha_star}" == "0.00" ]]; then
    case_tag="random_hosts_alpha${alpha_tag}"
  else
    case_tag="stellar_weighted_alpha${alpha_tag}"
  fi

  output_dir="${OUTPUT_BASE}/${case_tag}"
  cache_file="${OUTPUT_BASE}/cache/stellar_weighted_frb_los_dm_profile_cache.jld2"
  stdout_path="${LOG_DIR}/${case_tag}_${JOB_TAG}.out"
  stderr_path="${LOG_DIR}/${case_tag}_${JOB_TAG}.err"
  mkdir -p "${output_dir}" "$(dirname "${cache_file}")"

  echo "---------------------------------------"
  echo "Running case: ${case_tag}"
  echo "alpha_star=${alpha_star}"
  echo "output_dir=${output_dir}"
  echo "stdout=${stdout_path}"
  echo "stderr=${stderr_path}"
  echo "---------------------------------------"

  (
    set -x
    "${JULIA}" --threads="${THREADS_PER_TASK}" "${SCRIPT}" \
      halfdome_path="${HALFDOME_PATH}" \
      output_dir="${output_dir}" \
      N="${NFRB}" \
      z_source="${ZSOURCE}" \
      dz="${DZ}" \
      source_selection_mode="${SOURCE_SELECTION_MODE}" \
      source_z_min="${SOURCE_Z_MIN}" \
      source_z_max="${SOURCE_Z_MAX}" \
      source_halo_mass_min="${SOURCE_HALO_MASS_MIN}" \
      source_halo_mass_max="${SOURCE_HALO_MASS_MAX}" \
      alpha_star="${alpha_star}" \
      seed="${SEED}" \
      nside="${NSIDE}" \
      chunkN="${CHUNKN}" \
      stellar_mass_field="${STELLAR_MASS_FIELD}" \
      stellar_mass_divide_by_h="${STELLAR_MASS_DIVIDE_BY_H}" \
      z_min_foreground="${Z_MIN_FOREGROUND}" \
      z_max_foreground="${Z_MAX_FOREGROUND}" \
      halo_mass_min="${HALO_MASS_MIN}" \
      halo_mass_max="${HALO_MASS_MAX}" \
      frb_overlap_mode="${FRB_OVERLAP_MODE}" \
      dm_hist_bins="${DM_HIST_BINS}" \
      dm_cache_file="${cache_file}" \
      dm_cache_overwrite="${DM_CACHE_OVERWRITE}" \
      dm_cleanup_nonpositive="${DM_CLEANUP_NONPOSITIVE}" \
      save_shell_probabilities="${SAVE_SHELL_PROBABILITIES}" \
      save_foreground_map="${SAVE_FOREGROUND_MAP}" \
      foreground_progress_every="${FOREGROUND_PROGRESS_EVERY}" \
      save_power_spectrum="${SAVE_POWER_SPECTRUM}" \
      cl_lmax="${CL_LMAX}" \
      cl_niter="${CL_NITER}" \
      subtract_cl_mean="${SUBTRACT_CL_MEAN}" \
      save_frb_corrected_estimator="${SAVE_FRB_CORRECTED_ESTIMATOR}" \
      frb_corrected_lmax="${FRB_CORRECTED_LMAX}" \
      frb_corrected_shot_noise="${FRB_CORRECTED_SHOT_NOISE}" \
      frb_corrected_n_shuffle="${FRB_CORRECTED_N_SHUFFLE}" \
      frb_corrected_subtract_sample_mean="${FRB_CORRECTED_SUBTRACT_SAMPLE_MEAN}" \
      frb_corrected_seed="${FRB_CORRECTED_SEED}"
  ) >"${stdout_path}" 2>"${stderr_path}"

  echo "Finished case: ${case_tag}"
done

echo "All requested FRB LOS-DM cases finished."
