#!/usr/bin/env bash
# Run the four group_particles builds (buggy / plain / perfix / naive) of y_prof and ne_prof on one CAMELS snapshot.
#   BUILD_DIR=... ./run_gp_camels.sh RUNDIR SNAPFILE GROUPFILE SNAPNUM [variants]
# RUNDIR/input/{groups_S/fof_subhalo_tab_S.0.hdf5, snapdir_S/snap_S.0.hdf5} (and the same as _099 for ne_prof,
# whose file names are hard-coded to snapshot 099) are symlinks to SNAPFILE / GROUPFILE.
set -euo pipefail
BUILD_DIR=${BUILD_DIR:?}
RUN=$1; SNAP=$(readlink -f "$2"); GRP=$(readlink -f "$3"); S=$4
VARIANTS=${5:-"buggy plain perfix naive"}
unset LD_LIBRARY_PATH
mkdir -p "$RUN/input"
for s in $S 099; do
    mkdir -p "$RUN/input/groups_$s" "$RUN/input/snapdir_$s"
    ln -sfn "$GRP" "$RUN/input/groups_$s/fof_subhalo_tab_$s.0.hdf5"
    ln -sfn "$SNAP" "$RUN/input/snapdir_$s/snap_$s.0.hdf5"
done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
for v in $VARIANTS; do
    W="$RUN/$v"; rm -rf "$W"; mkdir -p "$W/output" "$W/ne_prof_results_Jun22"
    ln -sfn ../input "$W/input"
    ( cd "$W" && /usr/bin/time -f "%e s wall, %M kB" "$BUILD_DIR/y_prof_$v" "$S" > y_prof.log 2>&1 ) || { echo "y_prof_$v FAILED"; tail "$W/y_prof.log"; exit 1; }
    ( cd "$W" && /usr/bin/time -f "%e s wall, %M kB" "$BUILD_DIR/ne_prof_$v" > ne_prof.log 2>&1 ) || { echo "ne_prof_$v FAILED"; tail "$W/ne_prof.log"; exit 1; }
    echo "$v: y_prof $(tail -1 "$W/y_prof.log"), ne_prof $(tail -1 "$W/ne_prof.log"), groups $(( $(stat -c %s "$W/output/grp_M200c_$S.bin") / 8 ))"
done
