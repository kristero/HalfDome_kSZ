#!/bin/bash
# Particle-level wrap / no-wrap test on CAMELS CV_0 raw snapshots (downloaded to the session scratchpad, not kept).
# Usage: run_particle_tests.sh RAWDIR SNAP [SNAP ...]
set -u
RAW=$1; shift
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
export LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib
HERE=/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/boundary_check
OUT=/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007
STORED=/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/direct
for s in "$@"; do
  $PY $HERE/camels_wrap_particle_test.py $RAW/snapshot_$s.hdf5 $RAW/groups_$s.hdf5 $STORED/direct_CV_0_$s.npz \
      $OUT/particle_test_CV_0_$s.npz --minlogm 12.0 > $OUT/particle_test_CV_0_$s.log 2>&1
  echo "snap $s exit $?"
done
