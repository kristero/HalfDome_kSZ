#!/bin/bash
# verify_5 chain: after the tSZ split paints finish, paint the kSZ literal z-split / C_TNG maps and compute
# the tSZ split spectra in parallel, then the kSZ spectra.
set -euo pipefail
PROJECT=/home/cbllover/HalfDome
CODE=$PROJECT/frb_map_generation/lee22_normalization_forensics/verify_5
OUT=$PROJECT/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_5
JULIA=/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
export JULIA_DEPOT_PATH=/home/kn18001/.julia HDF5_USE_FILE_LOCKING=FALSE
until grep -q "all done" $OUT/logs/run_tsz_split_paint.log; do sleep 20; done
echo "tsz paints done $(date -Is)"
cd $PROJECT
/usr/bin/time -v "$JULIA" --threads=20 $CODE/paint_ksz_literal_zsplit_ctng.jl $OUT --nside=4096 > $OUT/logs/paint_ksz.log 2>&1 &
PJ=$!
OMP_NUM_THREADS=8 $PY $CODE/tsz_split_spectra.py > $OUT/logs/tsz_split_spectra.log 2>&1 || echo "tsz spectra FAILED"
echo "tsz spectra done $(date -Is)"
wait $PJ
echo "ksz paint done $(date -Is)"
OMP_NUM_THREADS=20 $PY $CODE/ksz_split_spectra.py > $OUT/logs/ksz_split_spectra.log 2>&1 || echo "ksz spectra FAILED"
echo "chain done $(date -Is)"
