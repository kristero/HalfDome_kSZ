#!/bin/bash
# Local Battaglia16 1R200c products at several NSIDE values (same 120k-ray count, seed,
# catalogue, cache and M200c windows as the NSIDE=4096 product) to test whether the
# HEALPix ray placement changes the low-DM tail. Usage: bash <script> 2048 8192
set -euo pipefail
export PROJECT_DIR=/home/cbllover/HalfDome
export JULIA=/home/kn18001/.juliaup/bin/julia
export JULIA_DEPOT_PATH=/home/kn18001/.julia
export PYTHON=/home/cbllover/miniconda3/envs/halfdome/bin/python
export HALFDOME_PATH=${PROJECT_DIR}/lightcone_100.hdf5
export THREADS_PER_TASK=${THREADS_PER_TASK:-20}
export CLUSTER_USER=local OVERWRITE=${OVERWRITE:-false}
export NFRB=120000 ZSOURCE=1.0 SEED=42 MAX_CATALOG_HALOS=0
export HALO_EXTENSION_R200_MULTIPLIER=${HALO_EXTENSION_R200_MULTIPLIER:-1.0}
export XGPAINT_PROFILE_MASS_DEFINITION=m200c DM_PROFILE=battaglia16
export MASS_WINDOWS="m1e10_to_1e12:1e10:1e12,m1e10_to_1e13:1e10:1e13,m1e10_to_1e14:1e10:1e14,m1e10_to_1e15:1e10:1e15,m1e10_to_1e16:1e10:1e16,m1e11_to_1e14:1e11:1e14,m1e12_to_1e14:1e12:1e14,m1e13_to_1e14:1e13:1e14,m1e12_to_1e16:1e12:1e16,m1e13_to_1e16:1e13:1e16,m1e14_to_1e16:1e14:1e16,m1e15_to_1e16:1e15:1e16"
export HDF5_USE_FILE_LOCKING=FALSE
export OUTPUT_BASE=${PROJECT_DIR}/frb_map_generation/outputs
export DM_CACHE=${PROJECT_DIR}/frb_map_generation/outputs/shared_xgpaint_dm_cache.jld2 DM_CACHE_OVERWRITE=false
tag="$(printf '%.1f' "${HALO_EXTENSION_R200_MULTIPLIER}")"; tag="${tag//./p}"
for nside in "$@"; do
  export NSIDE=${nside}
  export RUN_TAG="zsrc1p0_nside${nside}_nrays120000_allhalos_r200cx${tag}_m200cprofile_seed42"
  echo "############ NSIDE ${nside} -> ${OUTPUT_BASE}/${RUN_TAG} ############"
  /bin/bash "${PROJECT_DIR}/frb_map_generation/run_halfdome_z1_mass_histograms_120k.pbs"
done
echo "NSIDE series complete: $*"
