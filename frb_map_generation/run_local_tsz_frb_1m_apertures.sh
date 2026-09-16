#!/usr/bin/env bash
set -euo pipefail

# Local, restartable Battaglia12-y x FRB-DM workflow.
# The Battaglia12 Compton-y map is painted here from the complete HalfDome
# lightcone; it is not inherited from an unrelated FITS product.

PROJECT_DIR="${PROJECT_DIR:-/home/cbllover/HalfDome}"
JULIA="${JULIA:-/home/kn18001/.juliaup/bin/julia}"
PYTHON="${PYTHON:-/home/cbllover/miniconda3/bin/python}"
CLASS_SZ_PYTHON="${CLASS_SZ_PYTHON:-/home/kn18001/.conda/envs/class_sz/bin/python}"
CLASS_SZ_SITE_PACKAGES="${CLASS_SZ_SITE_PACKAGES:-/home/kn18001/.local/lib/python3.11/site-packages}"
CLASS_SZ_DATA="${CLASS_SZ_DATA:-/home/kn18001/class_sz_data_directory}"
HALFDOME_CATALOG="${HALFDOME_CATALOG:-$PROJECT_DIR/lightcone_100.hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/frb_map_generation/outputs/tsz_frb_1m_z1_r200c_apertures}"
TSZ_CACHE="${TSZ_CACHE:-$PROJECT_DIR/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2}"
TSZ_APERTURE_R200C="${TSZ_APERTURE_R200C:-4}"
OVERWRITE_TSZ_MAP="${OVERWRITE_TSZ_MAP:-false}"
OVERWRITE_CLASS_SZ_REFERENCE="${OVERWRITE_CLASS_SZ_REFERENCE:-$OVERWRITE_TSZ_MAP}"
BATTAGLIA16_CACHE="${BATTAGLIA16_CACHE:-$PROJECT_DIR/frb_map_generation/outputs/shared_xgpaint_dm_cache.jld2}"
LEE22_CACHE="${LEE22_CACHE:-$PROJECT_DIR/frb_map_generation/outputs/halfdome_frb_inputs/lee2022_tablea2_noconcentration_m200c_profile_owned_los_v2_dm_cache.jld2}"
JULIA_THREADS="${JULIA_THREADS:-20}"
JULIA_PROJECT="${JULIA_PROJECT:-$OUTPUT_ROOT/local_julia_env}"
LOCAL_JULIA_DEPOT="${LOCAL_JULIA_DEPOT:-$OUTPUT_ROOT/.julia_depot}"
JULIA_SOURCE_DEPOT="${JULIA_SOURCE_DEPOT:-/home/kn18001/.julia}"
XGPAINT_LOCAL_PATH="${XGPAINT_LOCAL_PATH:-/home/kn18001/.julia/dev/XGPaint}"

GENERATOR="$PROJECT_DIR/frb_map_generation/generate_halfdome_z1_dm_mass_windows.jl"
TSZ_PAINTER="$PROJECT_DIR/frb_map_generation/paint_halfdome_battaglia12_tsz_map.jl"
ANALYZER="$PROJECT_DIR/frb_map_generation/compute_local_tsz_frb_1m_aperture_spectra.py"
CLASS_SZ_GENERATOR="$PROJECT_DIR/frb_map_generation/compute_class_sz_battaglia12_reference.py"
SETUP_JULIA="$PROJECT_DIR/frb_map_generation/setup_local_frb_julia_env.jl"
RAY_DIR="$OUTPUT_ROOT/ray_catalogues"
MAP_DIR="$OUTPUT_ROOT/maps"
SPECTRA_DIR="$OUTPUT_ROOT/spectra"
LOG_DIR="$OUTPUT_ROOT/logs"
TSZ_APERTURE_TAG="${TSZ_APERTURE_R200C//./p}"
TSZ_MAP="$MAP_DIR/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx${TSZ_APERTURE_TAG}.fits"
TSZ_PROVENANCE="${TSZ_MAP%.fits}_provenance.txt"
CLASS_SZ_REFERENCE="$SPECTRA_DIR/class_sz_battaglia12_halfdome_bounds_reference.npz"
mkdir -p "$RAY_DIR" "$MAP_DIR" "$SPECTRA_DIR" "$LOG_DIR" "$LOCAL_JULIA_DEPOT"

for required in "$JULIA" "$PYTHON" "$CLASS_SZ_PYTHON" "$CLASS_SZ_SITE_PACKAGES" "$CLASS_SZ_DATA" "$HALFDOME_CATALOG" "$GENERATOR" "$TSZ_PAINTER" "$ANALYZER" "$CLASS_SZ_GENERATOR" "$SETUP_JULIA" "$BATTAGLIA16_CACHE" "$LEE22_CACHE" "$XGPAINT_LOCAL_PATH"; do
    [[ -e "$required" ]] || { echo "Required input does not exist: $required" >&2; exit 1; }
done

export JULIA_DEPOT_PATH="$LOCAL_JULIA_DEPOT:$JULIA_SOURCE_DEPOT"
if ! "$JULIA" --project="$JULIA_PROJECT" -e 'using HDF5, Healpix, Interpolations, XGPaint' >/dev/null 2>&1; then
    echo "Preparing isolated local Julia environment; log: $LOG_DIR/julia_environment_setup.log"
    "$JULIA" "$SETUP_JULIA" "$JULIA_PROJECT" "$XGPAINT_LOCAL_PATH" \
        2>&1 | tee "$LOG_DIR/julia_environment_setup.log"
fi

if [[ "$OVERWRITE_TSZ_MAP" != "true" && -s "$TSZ_MAP" && -s "$TSZ_PROVENANCE" ]]; then
    echo "Reusing internally generated tSZ map: $TSZ_MAP"
elif [[ "$OVERWRITE_TSZ_MAP" != "true" && ( -e "$TSZ_MAP" || -e "$TSZ_PROVENANCE" ) ]]; then
    echo "Incomplete tSZ product pair. Both map and provenance must exist, or neither:" >&2
    echo "  map: $TSZ_MAP" >&2
    echo "  provenance: $TSZ_PROVENANCE" >&2
    exit 1
else
    echo "Painting complete-lightcone Battaglia12 tSZ map; log: $LOG_DIR/tsz_generation.log"
    JULIA_NUM_THREADS="$JULIA_THREADS" "$JULIA" --project="$JULIA_PROJECT" "$TSZ_PAINTER" \
        --halfdome-path="$HALFDOME_CATALOG" \
        --nside=4096 --maximum-halo-redshift=Inf \
        --halo-extension-r200-multiplier="$TSZ_APERTURE_R200C" \
        --chunk-size=1000000 --max-catalog-halos=0 --progress-every=5 \
        --tsz-cache="$TSZ_CACHE" --tsz-cache-overwrite=false \
        --interpolator-logmass-max=15.7 --tsz-value-sanity-max=1.0 \
        --output-map="$TSZ_MAP" --provenance="$TSZ_PROVENANCE" \
        --overwrite="$OVERWRITE_TSZ_MAP" \
        2>&1 | tee "$LOG_DIR/tsz_generation.log"
fi

generate_class_sz_reference() {
    local overwrite_flag="$1"
    echo "Computing CLASS-SZ Battaglia12 reference; log: $LOG_DIR/class_sz_reference.log"
    if [[ "$overwrite_flag" == "true" ]]; then
        PATH_TO_CLASS_SZ_DATA="$CLASS_SZ_DATA" \
        PYTHONPATH="$CLASS_SZ_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" \
            "$CLASS_SZ_PYTHON" "$CLASS_SZ_GENERATOR" \
            --tsz-provenance "$TSZ_PROVENANCE" --output "$CLASS_SZ_REFERENCE" \
            --ell-max 8192 --x-out-sz 4 --overwrite \
            2>&1 | tee "$LOG_DIR/class_sz_reference.log"
    else
        PATH_TO_CLASS_SZ_DATA="$CLASS_SZ_DATA" \
        PYTHONPATH="$CLASS_SZ_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" \
            "$CLASS_SZ_PYTHON" "$CLASS_SZ_GENERATOR" \
            --tsz-provenance "$TSZ_PROVENANCE" --output "$CLASS_SZ_REFERENCE" \
            --ell-max 8192 --x-out-sz 4 \
            2>&1 | tee "$LOG_DIR/class_sz_reference.log"
    fi
}

if [[ "$OVERWRITE_CLASS_SZ_REFERENCE" != "true" && -s "$CLASS_SZ_REFERENCE" ]]; then
    echo "Reusing CLASS-SZ Battaglia12 reference: $CLASS_SZ_REFERENCE"
else
    generate_class_sz_reference "$OVERWRITE_CLASS_SZ_REFERENCE"
fi

run_frb_case() {
    local profile="$1"
    local aperture="$2"
    local cache="$3"
    local label="${profile}_r${aperture}"
    local output="$RAY_DIR/${label}_zsrc1p0_nside4096_nfrb1000000_seed42.h5"
    local log="$LOG_DIR/${label}_generation.log"
    if [[ -s "$output" ]]; then
        echo "Reusing completed ray file: $output"
        return
    fi
    echo "Generating $label; log: $log"
    JULIA_NUM_THREADS="$JULIA_THREADS" "$JULIA" --project="$JULIA_PROJECT" "$GENERATOR" \
        --catalog="$HALFDOME_CATALOG" \
        --sightline-mode=uniform --unique-pixels=true \
        --source-redshift=1.0 --nfrb=1000000 --seed=42 --nside=4096 \
        --chunk-size=1000000 --max-catalog-halos=0 --progress-every-batches=5 \
        --mass-windows=all:0:inf --apply-catalog-mass-floor=false \
        --catalog-masses-are-msun-h=true --xgpaint-profile-mass-definition=m200c \
        --dm-profile="$profile" --lee2022-concentration-mode=none \
        --dm-cache="$cache" --dm-cache-overwrite=false \
        --halo-extension-r200-multiplier="$aperture" \
        --save-ray-dm=true --output="$output" --overwrite=false \
        2>&1 | tee "$log"
}

# Identical N, NSIDE, and seed force exactly the same random rays in all cases.
run_frb_case battaglia16 3 "$BATTAGLIA16_CACHE"
run_frb_case battaglia16 5 "$BATTAGLIA16_CACHE"
run_frb_case lee2022 3 "$LEE22_CACHE"
run_frb_case lee2022 5 "$LEE22_CACHE"

echo "Computing spectra; log: $LOG_DIR/spectra.log"
OMP_NUM_THREADS="$JULIA_THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$ANALYZER" \
    --tsz-map "$TSZ_MAP" \
    --tsz-provenance "$TSZ_PROVENANCE" \
    --class-sz-spectrum "$CLASS_SZ_REFERENCE" \
    --frb-input battaglia16_r3 battaglia16 3 "$RAY_DIR/battaglia16_r3_zsrc1p0_nside4096_nfrb1000000_seed42.h5" \
    --frb-input battaglia16_r5 battaglia16 5 "$RAY_DIR/battaglia16_r5_zsrc1p0_nside4096_nfrb1000000_seed42.h5" \
    --frb-input lee22_r3 lee2022 3 "$RAY_DIR/lee2022_r3_zsrc1p0_nside4096_nfrb1000000_seed42.h5" \
    --frb-input lee22_r5 lee2022 5 "$RAY_DIR/lee2022_r5_zsrc1p0_nside4096_nfrb1000000_seed42.h5" \
    --output-dir "$SPECTRA_DIR" --nside 4096 --lmax 8192 --niter 0 --log-bins 55 \
    "$@" 2>&1 | tee "$LOG_DIR/spectra.log"

echo "Finished. Analysis products: $SPECTRA_DIR"
