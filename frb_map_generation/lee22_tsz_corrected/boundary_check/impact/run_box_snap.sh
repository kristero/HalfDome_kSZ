#!/usr/bin/env bash
# Full chain for one CAMELS box/snapshot: replica (all translations in SHIFTS) + C++ group_particles runs
# (translation 0: buggy/plain/perfix/naive; translations 1..NSHIFT_CPP: buggy/plain) + C++-vs-replica comparison.
#   ./run_box_snap.sh BOX SNAP NSHIFT_CPP
set -euo pipefail
B=$1; S=$2; NC=$3
HERE=/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/boundary_check/impact
SP=/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/impact
OUT=/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007/impact
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
PYLIB=/home/cbllover/miniconda3/envs/halfdome/lib
SH=$OUT/shifts_L25000.txt
RAW=$SP/raw/CV_$B
TAG=CV_${B}_$S
mkdir -p $OUT/replica $OUT/cpp_vs_replica $OUT/logs
LD_LIBRARY_PATH=$PYLIB $PY $HERE/camels_bug_replica.py $RAW/snapshot_$S.hdf5 $RAW/groups_$S.hdf5 $SH $OUT/replica/replica_$TAG.npz \
    > $OUT/logs/replica_$TAG.log 2>&1
for i in $(seq 0 $NC); do
    read sx sy sz < <(sed -n "$((i+1))p" $SH)
    P=$SP/proxy/${TAG}_s$i; R=$SP/runs/${TAG}_s$i
    LD_LIBRARY_PATH=$PYLIB $PY $HERE/make_shift_proxy.py $RAW/snapshot_$S.hdf5 $RAW/groups_$S.hdf5 $P $sx $sy $sz > /dev/null
    V="buggy plain"; [ $i = 0 ] && V="buggy plain perfix naive"
    BUILD_DIR=$SP/build $HERE/run_gp_camels.sh $R $P/snapshot_proxy.hdf5 $P/groups_proxy.hdf5 $S "$V" > $OUT/logs/cpp_${TAG}_s$i.log 2>&1
    LD_LIBRARY_PATH=$PYLIB $PY $HERE/compare_cpp_replica.py $OUT/replica/replica_$TAG.npz $i $R $S $OUT/cpp_vs_replica/${TAG}_s$i.json > /dev/null
    # keep the C++ binary outputs (small), drop the proxy files
    mkdir -p $OUT/cpp_runs/${TAG}_s$i
    for v in $V; do mkdir -p $OUT/cpp_runs/${TAG}_s$i/$v; cp $R/$v/output/*.bin $R/$v/ne_prof_results_Jun22/*.bin $OUT/cpp_runs/${TAG}_s$i/$v/; done
    rm -rf $P
done
echo "$TAG done $(date)"
