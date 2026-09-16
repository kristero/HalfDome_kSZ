#!/bin/bash
# Local (WSL, 20 threads) Lee22 companion runs to the Battaglia16 1R200c product:
# identical 120k rays, complete lightcone, M200c windows and bins; projected 1R200c
# aperture; profile-owned Lee22 LOS. Four convention variants are produced so the
# effect of the implementation check (LEE22_IMPLEMENTATION_CHECK_20260916.md) is visible:
#   noconc_corrected : Table A2/8, baryon-fraction normalization, common M_cut pivot, shape clip
#   pref_corrected   : Table 3, baryon-fraction normalization, TNG-mean concentration, shape clip
#   noconc_literal   : Table A2/8 exactly as implemented before (literal eq. 9, 1e14 pivot)
#   pref_literal     : Table 3 with Duffy08 c, literal eq. 9; shape clip is required for a finite LOS
#   noconc_corrected_zhyp / pref_corrected_zhyp : corrected variants with the additional
#                      (1+z)^3/E^2(z) comoving-bookkeeping hypothesis (see the implementation check)
# Usage: bash run_lee2022_z1_nside4096_1r200c_variants_local.sh [variant ...]
set -euo pipefail

export PROJECT_DIR=/home/cbllover/HalfDome
export JULIA=/home/kn18001/.juliaup/bin/julia
export JULIA_DEPOT_PATH=/home/kn18001/.julia
export PYTHON=/home/cbllover/miniconda3/envs/halfdome/bin/python
export HALFDOME_PATH=${PROJECT_DIR}/lightcone_100.hdf5
export THREADS_PER_TASK=${THREADS_PER_TASK:-20}
export CLUSTER_USER=local
export OVERWRITE=${OVERWRITE:-false}
export NFRB=120000 ZSOURCE=1.0 SEED=42 NSIDE=4096 MAX_CATALOG_HALOS=0
export HALO_EXTENSION_R200_MULTIPLIER=1.0
export XGPAINT_PROFILE_MASS_DEFINITION=m200c
export DM_PROFILE=lee2022
export MASS_WINDOWS="m1e10_to_1e12:1e10:1e12,m1e10_to_1e13:1e10:1e13,m1e10_to_1e14:1e10:1e14,m1e10_to_1e15:1e10:1e15,m1e10_to_1e16:1e10:1e16,m1e11_to_1e14:1e11:1e14,m1e12_to_1e14:1e12:1e14,m1e13_to_1e14:1e13:1e14,m1e12_to_1e16:1e12:1e16,m1e13_to_1e16:1e13:1e16,m1e14_to_1e16:1e14:1e16,m1e15_to_1e16:1e15:1e16"
export HDF5_USE_FILE_LOCKING=FALSE
export OUTPUT_BASE=${PROJECT_DIR}/frb_map_generation/outputs
CACHE_DIR=${PROJECT_DIR}/frb_map_generation/outputs/halfdome_frb_inputs
GENERIC=${PROJECT_DIR}/frb_map_generation/run_halfdome_z1_mass_histograms_120k.pbs

variants=("$@")
[[ ${#variants[@]} -gt 0 ]] || variants=(noconc_corrected pref_corrected noconc_literal pref_literal)

for variant in "${variants[@]}"; do
  export LEE2022_REDSHIFT_SCALING=physical
  case "${variant}" in
    noconc_corrected_zhyp)
      export LEE2022_CONCENTRATION_MODE=none LEE2022_NORMALIZATION=baryon_fraction LEE2022_N0_PIVOT=mcut \
             LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=fit LEE2022_REDSHIFT_SCALING=comoving_hypothesis
      cache=lee2022_noconc_fbnorm_mcutpivot_shapeclip_zhyp_dm_cache.jld2 ;;
    pref_corrected_zhyp)
      export LEE2022_CONCENTRATION_MODE=duffy2008 LEE2022_NORMALIZATION=baryon_fraction LEE2022_N0_PIVOT=mcut \
             LEE2022_CONCENTRATION_SOURCE=tng_mean LEE2022_SHAPE_MASS_CLIP=fit LEE2022_REDSHIFT_SCALING=comoving_hypothesis
      cache=lee2022_pref_tngc_fbnorm_shapeclip_zhyp_dm_cache.jld2 ;;
    noconc_corrected)
      export LEE2022_CONCENTRATION_MODE=none LEE2022_NORMALIZATION=baryon_fraction LEE2022_N0_PIVOT=mcut \
             LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=fit
      cache=lee2022_noconc_fbnorm_mcutpivot_shapeclip_dm_cache.jld2 ;;
    pref_corrected)
      export LEE2022_CONCENTRATION_MODE=duffy2008 LEE2022_NORMALIZATION=baryon_fraction LEE2022_N0_PIVOT=mcut \
             LEE2022_CONCENTRATION_SOURCE=tng_mean LEE2022_SHAPE_MASS_CLIP=fit
      cache=lee2022_pref_tngc_fbnorm_shapeclip_dm_cache.jld2 ;;
    noconc_literal)
      export LEE2022_CONCENTRATION_MODE=none LEE2022_NORMALIZATION=literal LEE2022_N0_PIVOT=legacy_1e14 \
             LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=none
      cache=lee2022_tablea2_noconcentration_m200c_profile_owned_los_v2_dm_cache.jld2 ;;
    pref_literal)
      export LEE2022_CONCENTRATION_MODE=duffy2008 LEE2022_NORMALIZATION=literal LEE2022_N0_PIVOT=mcut \
             LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=fit
      cache=lee2022_pref_duffy_literal_shapeclip_dm_cache.jld2 ;;
    *) echo "Unknown variant ${variant}" >&2; exit 2 ;;
  esac
  export RUN_TAG="zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_${variant}_seed42"
  export DM_CACHE="${CACHE_DIR}/${cache}"
  if [[ -f "${DM_CACHE}" && -f "${DM_CACHE}.profile_signature.txt" ]]; then
    export DM_CACHE_OVERWRITE=false
  else
    export DM_CACHE_OVERWRITE=true
  fi
  echo "############ ${variant} -> ${OUTPUT_BASE}/${RUN_TAG} (cache ${cache}, build=${DM_CACHE_OVERWRITE}) ############"
  /bin/bash "${GENERIC}"
done
echo "Lee22 1R200c variants complete: ${variants[*]}"
