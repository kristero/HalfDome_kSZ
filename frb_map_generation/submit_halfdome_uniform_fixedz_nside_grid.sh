#!/bin/bash

set -euo pipefail

# Submit complete-catalogue runs for every NSIDE/source-redshift combination.
# Each job draws NFRB unique uniform HEALPix rays on one exact source plane.
#
# The optional catalogue-cap test processes the first CATALOG_CAP HalfDome
# rows. It is a runtime/convergence test, not a random halo subsample.

: "${PROJECT_DIR:=/home/${USER}/HalfDome_kSZ}"
: "${PBS_FILE:=${PROJECT_DIR}/frb_map_generation/run_halfdome_z1_mass_histograms_120k.pbs}"
: "${DM_CACHE:=${PROJECT_DIR}/frb_map_generation/outputs/shared_xgpaint_dm_cache.jld2}"
: "${NSIDES:=2048 8192}"
: "${SOURCE_REDSHIFTS:=1.0 2.0 3.0 4.0}"
: "${NFRB:=120000}"
: "${SEED:=42}"
: "${SUBMIT_CATALOG_CAP_TEST:=true}"
: "${CATALOG_CAP:=120000}"
: "${CAP_TEST_NSIDE:=2048}"
: "${CAP_TEST_ZSOURCE:=1.0}"
: "${DRY_RUN:=false}"

[[ -f "${PBS_FILE}" ]] || {
  echo "PBS file not found: ${PBS_FILE}" >&2
  exit 2
}
[[ -f "${DM_CACHE}" ]] || {
  echo "DM cache not found: ${DM_CACHE}" >&2
  exit 2
}
if [[ "${DRY_RUN}" != "true" ]]; then
  command -v qsub >/dev/null || {
    echo "qsub is not available." >&2
    exit 2
  }
fi

submit_one() {
  local nside="$1"
  local source_redshift="$2"
  local catalog_cap="$3"
  local job_name="$4"
  local variables
  variables="PROJECT_DIR=${PROJECT_DIR},DM_CACHE=${DM_CACHE},NSIDE=${nside},ZSOURCE=${source_redshift},NFRB=${NFRB},SEED=${SEED},MAX_CATALOG_HALOS=${catalog_cap}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "qsub -N ${job_name} -v ${variables} ${PBS_FILE}"
  else
    local job_id
    job_id="$(qsub -N "${job_name}" -v "${variables}" "${PBS_FILE}")"
    echo "Submitted ${job_id}: NSIDE=${nside}, z_source=${source_redshift}, rays=${NFRB}, catalog_cap=${catalog_cap}"
  fi
}

for nside in ${NSIDES}; do
  for source_redshift in ${SOURCE_REDSHIFTS}; do
    redshift_tag="${source_redshift//./p}"
    submit_one "${nside}" "${source_redshift}" 0 "u${nside}z${redshift_tag}"
  done
done

if [[ "${SUBMIT_CATALOG_CAP_TEST}" == "true" ]]; then
  cap_redshift_tag="${CAP_TEST_ZSOURCE//./p}"
  submit_one \
    "${CAP_TEST_NSIDE}" \
    "${CAP_TEST_ZSOURCE}" \
    "${CATALOG_CAP}" \
    "c120kz${cap_redshift_tag}"
fi
