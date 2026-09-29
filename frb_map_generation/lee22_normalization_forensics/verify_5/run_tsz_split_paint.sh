#!/bin/bash
# verify_5 (adversarial check of claim C6): repaint the tSZ pieces needed for the
# "Lee22 pressure below 10^14.8/h, Battaglia12 above" hybrid map, with the unmodified
# repository painter (paint_halfdome_battaglia12_tsz_map.jl) and its mass-selection options.
#   lee22p_high : Lee22 no-c pressure, M200c >= 10^14.8/0.68 Msun   (existing Lee22 cache, copied)
#   b12_high    : Battaglia12 fiducial, M200c >= cut                 (new B12 cache in verify_5/caches)
#   b12_low     : Battaglia12 fiducial, M200c <  cut
# Lee22p_low = existing full Lee22p map - lee22p_high (exact: same painter, same cache).
set -euo pipefail
PROJECT=/home/cbllover/HalfDome
OUT=$PROJECT/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_5
JULIA=/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia
export JULIA_DEPOT_PATH=/home/kn18001/.julia HDF5_USE_FILE_LOCKING=FALSE
CUT=9.278784477649915e14
PAINTER=$PROJECT/frb_map_generation/paint_halfdome_battaglia12_tsz_map.jl
COMMON="--halfdome-path=$PROJECT/lightcone_100.hdf5 --nside=4096 --halo-extension-r200-multiplier=4 --max-catalog-halos=0 --chunk-size=100000 --progress-every=100 --overwrite=false"
cd "$PROJECT"
[[ -f $OUT/caches/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2 ]] || \
  cp -p $PROJECT/frb_map_generation/outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2 $OUT/caches/
echo "start $(date -Is)"
if [[ ! -f $OUT/maps/lee22p_high.fits ]]; then
  "$JULIA" --threads=20 $PAINTER $COMMON --tsz-profile=lee2022_noconc \
    --tsz-cache=$OUT/caches/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2 \
    --output-map=$OUT/maps/lee22p_high.fits --minimum-halo-mass-msun=$CUT > $OUT/logs/paint_lee22p_high.log 2>&1
  echo "lee22p_high done $(date -Is)"
fi
if [[ ! -f $OUT/maps/b12_high.fits ]]; then
  "$JULIA" --threads=20 $PAINTER $COMMON --tsz-profile=battaglia12 \
    --tsz-cache=$OUT/caches/b12_fiducial_h068_m200c_logm15p7_tsz_interpolator.jld2 \
    --output-map=$OUT/maps/b12_high.fits --minimum-halo-mass-msun=$CUT > $OUT/logs/paint_b12_high.log 2>&1
  echo "b12_high done $(date -Is)"
fi
if [[ ! -f $OUT/maps/b12_low.fits ]]; then
  "$JULIA" --threads=20 $PAINTER $COMMON --tsz-profile=battaglia12 \
    --tsz-cache=$OUT/caches/b12_fiducial_h068_m200c_logm15p7_tsz_interpolator.jld2 \
    --output-map=$OUT/maps/b12_low.fits --maximum-halo-mass-msun=$CUT > $OUT/logs/paint_b12_low.log 2>&1
  echo "b12_low done $(date -Is)"
fi
echo "all done $(date -Is)"
