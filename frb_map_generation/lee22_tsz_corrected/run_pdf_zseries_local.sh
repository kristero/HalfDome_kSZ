#!/bin/bash
# Local (WSL, 20 threads) 120k-ray R200c-sphere halo-DM histograms at the extra source redshifts of the PDF redshift
# series (2026-10-01): Battaglia16, Lee22 x0.6016 (XGPaint-native ne2d, previous production) and Lee22 x [h(1+z)]^3 X_H^2
# (derived, final). Settings identical to STAGE=rays of run_lee22_tngcal_cluster.pbs / run_lee22_corrected_cluster.pbs
# (seed 42, NSIDE 4096, complete lightcone, spherical 1 R200c boundary, same mass windows). z_s = 0.1, 0.5, 1 exist
# from idark; VALIDATE=1 also reruns Lee22-physical at z_s = 1 locally to check the local runs reproduce idark.
# Usage: bash frb_map_generation/lee22_tsz_corrected/run_pdf_zseries_local.sh   (from the repository root)
# Overrides: OUT (output root), MASS_WINDOWS, VARIANTS, ZSOURCES, VALIDATE=0|1.
set -euo pipefail

export PROJECT_DIR=/home/cbllover/HalfDome
export JULIA=/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia
export JULIA_DEPOT_PATH=/home/kn18001/.julia
export PYTHON=/home/cbllover/miniconda3/envs/halfdome/bin/python
export HALFDOME_PATH=${PROJECT_DIR}/lightcone_100.hdf5
export THREADS_PER_TASK=${THREADS_PER_TASK:-20} CLUSTER_USER=local OVERWRITE=false HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export NFRB=120000 SEED=42 NSIDE=4096 MAX_CATALOG_HALOS=0 HALO_EXTENSION_R200_MULTIPLIER=1.0 HALO_BOUNDARY=spherical
export XGPAINT_PROFILE_MASS_DEFINITION=m200c
export MASS_WINDOWS="${MASS_WINDOWS:-m1e10_to_1e12:1e10:1e12,m1e10_to_1e13:1e10:1e13,m1e10_to_1e14:1e10:1e14,m1e10_to_1e15:1e10:1e15,m1e10_to_1e16:1e10:1e16,m1e11_to_1e14:1e11:1e14,m1e12_to_1e14:1e12:1e14,m1e13_to_1e14:1e13:1e14,m1e12_to_1e16:1e12:1e16,m1e13_to_1e16:1e13:1e16,m1e14_to_1e16:1e14:1e16,m1e15_to_1e16:1e15:1e16}"
export LEE2022_CONCENTRATION_MODE=none LEE2022_N0_PIVOT=mcut LEE2022_CONCENTRATION_SOURCE=duffy2008 LEE2022_SHAPE_MASS_CLIP=fit
OUT=${OUT:-${PROJECT_DIR}/frb_map_generation/outputs/lee22_pdf_zseries_20261001}
INPUTS=${PROJECT_DIR}/frb_map_generation/outputs/halfdome_frb_inputs
cd "$PROJECT_DIR"

run() {  # variant zsource output_base
    local variant=$1 cache
    export ZSOURCE=$2 OUTPUT_BASE=$3
    case "$variant" in
        b16)             export DM_PROFILE=battaglia16 LEE2022_NORMALIZATION=literal LEE2022_REDSHIFT_SCALING=physical
                         cache=battaglia16_profile_owned_sphere1p0_chordmean_dm_cache.jld2 ;;
        lee22_xgpnative) export DM_PROFILE=lee2022_xgpaint LEE2022_NORMALIZATION=xgpaint_ne2d LEE2022_REDSHIFT_SCALING=physical
                         cache=lee2022_noconc_xgpaintne2d_mcutpivot_shapeclip_xgpaintpipeline_sphere1p0_chordmean_dm_cache.jld2 ;;
        lee22_phys)      export DM_PROFILE=lee2022_xgpaint LEE2022_NORMALIZATION=literal LEE2022_REDSHIFT_SCALING=bookkeeping_physical
                         cache=lee2022_noconc_literal_phys_mcutpivot_shapeclip_xgpaintpipeline_sphere1p0_chordmean_dm_cache.jld2 ;;
        *) echo "unknown variant $variant" >&2; return 2 ;;
    esac
    export DM_CACHE="$INPUTS/$cache"
    export RUN_TAG="zsrc${ZSOURCE//./p}_nside4096_nrays120000_allhalos_sphere1p0_m200c_${variant}_seed42"
    if [[ -f "$DM_CACHE" && -f "$DM_CACHE.profile_signature.txt" ]]; then export DM_CACHE_OVERWRITE=false; else export DM_CACHE_OVERWRITE=true; fi
    if [[ -f "$OUTPUT_BASE/$RUN_TAG/halfdome_uniform_fixedz_foreground_mass_histograms.h5" ]]; then
        echo "exists, skipping: $RUN_TAG"; return 0
    fi
    echo "######## $(date -Is) $variant z_s=$ZSOURCE -> $OUTPUT_BASE/$RUN_TAG (cache build=$DM_CACHE_OVERWRITE) ########"
    /bin/bash frb_map_generation/run_halfdome_z1_mass_histograms_120k.pbs
}

mkdir -p "$OUT/rays" "$OUT/validation" "$OUT/logs"
sha256sum frb_map_generation/*.jl frb_map_generation/run_halfdome_z1_mass_histograms_120k.pbs "$0" > "$OUT/logs/source_sha256.txt"
[[ "${VALIDATE:-1}" == 1 ]] && run lee22_phys 1.0 "$OUT/validation"
for variant in ${VARIANTS:-lee22_phys b16 lee22_xgpnative}; do
    for z in ${ZSOURCES:-1.5 2.0 3.0}; do
        run "$variant" "$z" "$OUT/rays"
    done
done
echo "######## $(date -Is) all done ########"
