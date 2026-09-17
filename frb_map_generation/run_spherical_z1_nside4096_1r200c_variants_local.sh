#!/bin/bash
# Like-for-like companions to TNG's within-R200 halo DM: gas counted only inside the
# R200c sphere (chord-limited LOS), same 120k rays, catalogue, M200c windows and bins as
# the projected 1R200c products. Variants: b16, lee22_noconc, lee22_pref (corrected conventions);
# sensitivity: lee22_noconc_literalnorm, lee22_pref_literalnorm (same conventions, but the literal eq. 9
# normalization without the Omega_b/Omega_m factor, to isolate that factor in the spherical geometry).
# electron-count fix: lee22_noconc_efix, lee22_pref_efix (eq. 9 with (1+X_H)/(2 m_p) instead of 1/(X_H m_p)).
set -euo pipefail
export PROJECT_DIR=/home/cbllover/HalfDome
export JULIA=/home/kn18001/.juliaup/bin/julia
export JULIA_DEPOT_PATH=/home/kn18001/.julia
export PYTHON=/home/cbllover/miniconda3/envs/halfdome/bin/python
export HALFDOME_PATH=${PROJECT_DIR}/lightcone_100.hdf5
export THREADS_PER_TASK=${THREADS_PER_TASK:-20}
export CLUSTER_USER=local OVERWRITE=${OVERWRITE:-false}
export NFRB=120000 ZSOURCE=1.0 SEED=42 NSIDE=4096 MAX_CATALOG_HALOS=0
export HALO_EXTENSION_R200_MULTIPLIER=${HALO_EXTENSION_R200_MULTIPLIER:-1.0}
export HALO_BOUNDARY=spherical
export XGPAINT_PROFILE_MASS_DEFINITION=m200c
export MASS_WINDOWS="m1e10_to_1e12:1e10:1e12,m1e10_to_1e13:1e10:1e13,m1e10_to_1e14:1e10:1e14,m1e10_to_1e15:1e10:1e15,m1e10_to_1e16:1e10:1e16,m1e11_to_1e14:1e11:1e14,m1e12_to_1e14:1e12:1e14,m1e13_to_1e14:1e13:1e14,m1e12_to_1e16:1e12:1e16,m1e13_to_1e16:1e13:1e16,m1e14_to_1e16:1e14:1e16,m1e15_to_1e16:1e15:1e16"
export HDF5_USE_FILE_LOCKING=FALSE
export OUTPUT_BASE=${PROJECT_DIR}/frb_map_generation/outputs
CACHE_DIR=${PROJECT_DIR}/frb_map_generation/outputs/halfdome_frb_inputs
tag="$(printf '%.1f' "${HALO_EXTENSION_R200_MULTIPLIER}")"; tag="${tag//./p}"
variants=("$@"); [[ ${#variants[@]} -gt 0 ]] || variants=(b16 lee22_noconc lee22_pref)
for variant in "${variants[@]}"; do
  export LEE2022_REDSHIFT_SCALING=physical
  case "${variant}" in
    b16)          export DM_PROFILE=battaglia16 LEE2022_CONCENTRATION_MODE=none
                  cache=battaglia16_profile_owned_sphere${tag}_chordmean_dm_cache.jld2 ;;
    lee22_noconc) export DM_PROFILE=lee2022 LEE2022_CONCENTRATION_MODE=none LEE2022_NORMALIZATION=baryon_fraction \
                         LEE2022_N0_PIVOT=mcut LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=fit
                  cache=lee2022_noconc_fbnorm_mcutpivot_shapeclip_sphere${tag}_chordmean_dm_cache.jld2 ;;
    lee22_pref)   export DM_PROFILE=lee2022 LEE2022_CONCENTRATION_MODE=duffy2008 LEE2022_NORMALIZATION=baryon_fraction \
                         LEE2022_N0_PIVOT=mcut LEE2022_CONCENTRATION_SOURCE=tng_mean LEE2022_SHAPE_MASS_CLIP=fit
                  cache=lee2022_pref_tngc_fbnorm_shapeclip_sphere${tag}_chordmean_dm_cache.jld2 ;;
    lee22_noconc_literalnorm) export DM_PROFILE=lee2022 LEE2022_CONCENTRATION_MODE=none LEE2022_NORMALIZATION=literal \
                         LEE2022_N0_PIVOT=mcut LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=fit
                  cache=lee2022_noconc_literalnorm_mcutpivot_shapeclip_sphere${tag}_chordmean_dm_cache.jld2 ;;
    lee22_pref_literalnorm) export DM_PROFILE=lee2022 LEE2022_CONCENTRATION_MODE=duffy2008 LEE2022_NORMALIZATION=literal \
                         LEE2022_N0_PIVOT=mcut LEE2022_CONCENTRATION_SOURCE=tng_mean LEE2022_SHAPE_MASS_CLIP=fit
                  cache=lee2022_pref_tngc_literalnorm_shapeclip_sphere${tag}_chordmean_dm_cache.jld2 ;;
    lee22_noconc_efix) export DM_PROFILE=lee2022 LEE2022_CONCENTRATION_MODE=none LEE2022_NORMALIZATION=electron_count \
                         LEE2022_N0_PIVOT=mcut LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=fit
                  cache=lee2022_noconc_necount_mcutpivot_shapeclip_sphere${tag}_chordmean_dm_cache.jld2 ;;
    lee22_pref_efix) export DM_PROFILE=lee2022 LEE2022_CONCENTRATION_MODE=duffy2008 LEE2022_NORMALIZATION=electron_count \
                         LEE2022_N0_PIVOT=mcut LEE2022_CONCENTRATION_SOURCE=tng_mean LEE2022_SHAPE_MASS_CLIP=fit
                  cache=lee2022_pref_tngc_necount_shapeclip_sphere${tag}_chordmean_dm_cache.jld2 ;;
    *) echo "Unknown variant ${variant}" >&2; exit 2 ;;
  esac
  export RUN_TAG="zsrc1p0_nside4096_nrays120000_allhalos_sphere${tag}_m200c_${variant}_seed42"
  export DM_CACHE="${CACHE_DIR}/${cache}"
  if [[ -f "${DM_CACHE}" && -f "${DM_CACHE}.profile_signature.txt" ]]; then export DM_CACHE_OVERWRITE=false; else export DM_CACHE_OVERWRITE=true; fi
  echo "############ ${variant} (spherical ${HALO_EXTENSION_R200_MULTIPLIER} R200c) -> ${OUTPUT_BASE}/${RUN_TAG} (cache build=${DM_CACHE_OVERWRITE}) ############"
  /bin/bash "${PROJECT_DIR}/frb_map_generation/run_halfdome_z1_mass_histograms_120k.pbs"
done
echo "Spherical variants complete: ${variants[*]}"
