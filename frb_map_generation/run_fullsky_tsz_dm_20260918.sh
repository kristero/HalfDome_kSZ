#!/bin/bash
# Local production chain for the full-sky tSZ x FRB-DM comparison (updated implementations):
#   1. Lee22 no-c pressure Compton-y map with only the halos inside the Lee22 calibration ranges
#      (1e13-10^14.8 h^-1 Msun, z <= 2); the all-halo Lee22 and Battaglia12 maps already exist
#   2. six kernel-weighted full-sky DM maps (3 density models x Planck/ACT observed-redshift kernels)
#   3. alm / auto- and cross-spectra of the 3 y and 6 DM maps
#   4. map-mean check against the 100k individual sightlines
#   5. annular y samples of the calibrated-range y map at the 100k positions (finite-source realizations)
set -euo pipefail
PROJECT=/home/cbllover/HalfDome
OUT=$PROJECT/frb_map_generation/outputs/tsz_dm_fullsky_20260918
JULIA=/home/kn18001/.juliaup/bin/julia
PYTHON=/home/cbllover/miniconda3/envs/halfdome/bin/python
export JULIA_DEPOT_PATH=/home/kn18001/.julia JULIA_NUM_THREADS=20 HDF5_USE_FILE_LOCKING=FALSE GKSwstype=100 MPLBACKEND=Agg
H=0.68
MASS_MIN=$(python3 -c "print(1e13/$H)")
MASS_MAX=$(python3 -c "print(10**14.8/$H)")
YCAL=$OUT/maps/lee22_noconc_pressure_compton_y_calib_zmax2p0_nside4096_m200c_r200cx4.fits
cd "$PROJECT"
mkdir -p "$OUT/logs"
echo "chain start $(date -Is)"

if [[ ! -f "$YCAL" ]]; then
  echo "[1] Lee22 pressure y map, calibrated range: $(date -Is)"
  "$JULIA" --threads=20 frb_map_generation/paint_halfdome_battaglia12_tsz_map.jl \
    --halfdome-path="$PROJECT/lightcone_100.hdf5" --tsz-profile=lee2022_noconc \
    --tsz-cache="$PROJECT/frb_map_generation/outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2" \
    --output-map="$YCAL" --nside=4096 --maximum-halo-redshift=2.0 \
    --minimum-halo-mass-msun="$MASS_MIN" --maximum-halo-mass-msun="$MASS_MAX" \
    --halo-extension-r200-multiplier=4 --max-catalog-halos=0 --chunk-size=100000 --progress-every=100 --overwrite=false \
    > "$OUT/logs/paint_lee22_pressure_calib_ymap.log" 2>&1
  sha256sum "$YCAL" > "${YCAL%.fits}.sha256"
else
  echo "[1] reuse $YCAL"
fi

if [[ ! -f "$OUT/maps/lee22_noconc_sphere1_calib_act.fits" ]]; then
  echo "[2] kernel-weighted DM maps: $(date -Is)"
  "$JULIA" --threads=20 frb_map_generation/paint_halfdome_kernel_weighted_dm_maps.jl \
    --output-dir="$OUT" --halfdome-path="$PROJECT/lightcone_100.hdf5" --nside=4096 --chunk-size=100000 \
    > "$OUT/logs/paint_kernel_weighted_dm_maps.log" 2>&1
else
  echo "[2] reuse DM maps"
fi

if [[ ! -f "$OUT/spectra/fullsky_spectra.npz" ]]; then
  echo "[3] spectra: $(date -Is)"
  "$PYTHON" frb_map_generation/fullsky_tsz_dm_comparison.py spectra --lmax 8192 --iter 3 > "$OUT/logs/spectra.log" 2>&1
fi
echo "[4] map-mean check: $(date -Is)"
"$PYTHON" frb_map_generation/fullsky_tsz_dm_comparison.py check > "$OUT/logs/check.log" 2>&1
cat "$OUT/logs/check.log"

if [[ ! -f "$PROJECT/frb_map_generation/outputs/tsz_dm_cross_updated_20260917/rays/annular_y_samples_lee22_pressure_calib.npz" ]]; then
  echo "[5] annular y samples of the calibrated-range map: $(date -Is)"
  "$PYTHON" frb_map_generation/cross_finite_source_realizations.py sample-y --ykey lee22p_calib --ymap "$YCAL" \
    > "$OUT/logs/sample_y_lee22p_calib.log" 2>&1
fi
echo "chain done $(date -Is)"
