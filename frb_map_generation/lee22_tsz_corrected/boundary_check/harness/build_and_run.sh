#!/usr/bin/env bash
# Periodic-boundary check of group_particles (Lee et al. 2022 profile extraction).
#
#   GP_DIR=<git clone of github.com/leanderthiele/group_particles> \
#   BUILD_DIR=<scratch dir> OUT_DIR=<results dir> ./build_and_run.sh [export|unit|e2e|all]
#
# export : git-archive the Lee22-era tree (c3fe9b8, 2021-06-25) and HEAD (6c6d486), plus a copy
#          of c3fe9b8 whose only change is a signed comparison inside the PER() hint macro
# unit   : build unit_harness.cpp against each tree with the compile.sh flags and run it
# e2e    : build the unmodified example y_prof.cpp (only its I/O paths are rewritten) against
#          c3fe9b8, c3fe9b8 -DNAIVE and HEAD, run it on synthetic TNG300-geometry HDF5 files and
#          compare with a brute-force minimum-image calculation (e2e/compare_e2e.py)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
GP_DIR=${GP_DIR:?set GP_DIR to a git clone of group_particles}
BUILD_DIR=${BUILD_DIR:?set BUILD_DIR to a scratch directory}
OUT_DIR=${OUT_DIR:-/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007}
PY=${PY:-/home/cbllover/miniconda3/envs/halfdome/bin/python}
H5INC=${H5INC:-/usr/include/hdf5/serial}
H5LIB=${H5LIB:-/usr/lib/x86_64-linux-gnu/hdf5/serial}
STAGE=${1:-all}
# the system HDF5/curl must not be shadowed by the conda env at link/run time; python gets its own lib dir
PYLIB=${PYLIB:-$(dirname "$(dirname "$PY")")/lib}
unset LD_LIBRARY_PATH
LEE22_REV=c3fe9b8
HEAD_REV=6c6d486
# verbatim from compile.sh (both revisions)
CXXFLAGS="-std=c++17 -O3 -ffast-math -funroll-loops -g3 -Wall -Wextra -Wno-unused-parameter -Wno-reorder"
mkdir -p "$BUILD_DIR" "$OUT_DIR/unit" "$OUT_DIR/e2e" "$OUT_DIR/logs"

do_export () {
    for rev in $LEE22_REV $HEAD_REV; do
        rm -rf "$BUILD_DIR/gp_$rev"; mkdir -p "$BUILD_DIR/gp_$rev"
        git -C "$GP_DIR" archive "$rev" detail include examples compile.sh | tar -x -C "$BUILD_DIR/gp_$rev"
        git -C "$GP_DIR" show -s --format="%H %ad %an %s" "$rev" > "$BUILD_DIR/gp_$rev/REVISION"
    done
    rm -rf "$BUILD_DIR/gp_${LEE22_REV}_perfix"
    cp -r "$BUILD_DIR/gp_$LEE22_REV" "$BUILD_DIR/gp_${LEE22_REV}_perfix"
    sed -i 's/#define PER(x) ((x>=Ncells_side) ? 1 : (x<0) ? -1 : 0)/#define PER(x) ((x>=(int)Ncells_side) ? 1 : (x<0) ? -1 : 0)/' \
        "$BUILD_DIR/gp_${LEE22_REV}_perfix/detail/workspace_sorting.hpp"
    diff -u "$BUILD_DIR/gp_$LEE22_REV/detail/workspace_sorting.hpp" \
            "$BUILD_DIR/gp_${LEE22_REV}_perfix/detail/workspace_sorting.hpp" > "$OUT_DIR/unit/perfix_patch.diff" || true
    test -s "$OUT_DIR/unit/perfix_patch.diff" || { echo "PER patch did not apply"; exit 1; }
    # provenance: geometry headers identical between the Lee22 tree and HEAD?
    { for f in detail/workspace_sorting.hpp detail/geom_utils.hpp; do
          if cmp -s "$BUILD_DIR/gp_$LEE22_REV/$f" "$BUILD_DIR/gp_$HEAD_REV/$f"; then echo "identical $f"; else echo "DIFFERENT $f"; fi
      done
      diff -u "$BUILD_DIR/gp_$LEE22_REV/detail/prt_loop.hpp" "$BUILD_DIR/gp_$HEAD_REV/detail/prt_loop.hpp" | grep -E '^[-+].*periodic_hypotsq' || true
    } > "$OUT_DIR/logs/tree_provenance.txt"
    for t in gp_$LEE22_REV gp_$HEAD_REV; do echo "$t: $(cat "$BUILD_DIR/$t/REVISION")"; done >> "$OUT_DIR/logs/tree_provenance.txt"; echo "gp_${LEE22_REV}_perfix: gp_$LEE22_REV + perfix_patch.diff" >> "$OUT_DIR/logs/tree_provenance.txt"
}

build_unit () {   # $1 = tree name, $2 = binary name
    local T="$BUILD_DIR/$1"
    g++ $CXXFLAGS -I"$T/include" -I"$T/include/callback_utils" -I"$T/detail" -I"$H5INC" \
        -o "$BUILD_DIR/$2" "$HERE/unit_harness.cpp" -L"$H5LIB" -lhdf5 -lhdf5_cpp -fopenmp \
        > "$OUT_DIR/logs/compile_$2.log" 2>&1
}

do_unit () {
    build_unit gp_$LEE22_REV unit_$LEE22_REV
    build_unit gp_${LEE22_REV}_perfix unit_${LEE22_REV}_perfix
    build_unit gp_$HEAD_REV unit_$HEAD_REV
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
    for b in $LEE22_REV ${LEE22_REV}_perfix $HEAD_REV; do
        /usr/bin/time -v "$BUILD_DIR/unit_$b" mode=targeted label=$b out="$OUT_DIR/unit" \
            > "$OUT_DIR/logs/unit_${b}_targeted.log" 2>&1
    done
    for b in $LEE22_REV ${LEE22_REV}_perfix; do
        /usr/bin/time -v "$BUILD_DIR/unit_$b" mode=population label=$b out="$OUT_DIR/unit" \
            > "$OUT_DIR/logs/unit_${b}_population.log" 2>&1
    done
}

do_e2e () {
    local D="$BUILD_DIR/e2e_data" S=099
    rm -rf "$D"; mkdir -p "$D"
    LD_LIBRARY_PATH="$PYLIB" "$PY" "$HERE/e2e/make_synthetic_tng.py" --outdir "$D" --snap $S > "$OUT_DIR/logs/e2e_make_data.log" 2>&1
    local NG NP
    NG=$(cat "$D/n_grp_chunks"); NP=$(cat "$D/n_prt_chunks")
    for variant in ${LEE22_REV} ${LEE22_REV}_NAIVE ${HEAD_REV}; do
        local tree=gp_${variant%_NAIVE} extra=""
        [[ $variant == *_NAIVE ]] && extra="-DNAIVE"
        local src="$BUILD_DIR/y_prof_$variant.cpp" odir="$BUILD_DIR/e2e_out_$variant"
        rm -rf "$odir"; mkdir -p "$odir"
        # only the four I/O lines of main() are rewritten; checked by the diff below
        sed -e "s|#define ROOT \"/tigress/lthiele/Illustris_300-1_TNG/output/\"|#define ROOT \"$D/\"|" \
            -e "s|grp_max_idx = 599;|grp_max_idx = $((NG-1));|" \
            -e "s|prt_max_idx = 599;|prt_max_idx = $((NP-1));|" \
            -e "s|#define ROOT \"DEFINE YOUR OUTPUT PATH HERE\"|#define ROOT \"$odir\"|" \
            "$BUILD_DIR/$tree/examples/y_prof.cpp" > "$src"
        diff "$BUILD_DIR/$tree/examples/y_prof.cpp" "$src" > "$OUT_DIR/logs/y_prof_${variant}_io_edits.diff" || true
        local T="$BUILD_DIR/$tree"
        g++ $CXXFLAGS $extra -I"$T/include" -I"$T/include/callback_utils" -I"$T/detail" -I"$H5INC" \
            -o "$BUILD_DIR/y_prof_$variant" "$src" -L"$H5LIB" -lhdf5 -lhdf5_cpp -fopenmp \
            > "$OUT_DIR/logs/compile_y_prof_$variant.log" 2>&1
        OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} /usr/bin/time -v "$BUILD_DIR/y_prof_$variant" $S \
            > "$OUT_DIR/logs/run_y_prof_$variant.log" 2>&1
    done
    LD_LIBRARY_PATH="$PYLIB" "$PY" "$HERE/e2e/compare_e2e.py" --data "$D" --snap $S \
        --run c3fe9b8="$BUILD_DIR/e2e_out_${LEE22_REV}" \
        --run c3fe9b8_NAIVE="$BUILD_DIR/e2e_out_${LEE22_REV}_NAIVE" \
        --run HEAD_6c6d486="$BUILD_DIR/e2e_out_${HEAD_REV}" \
        --out "$OUT_DIR/e2e" > "$OUT_DIR/logs/e2e_compare.log" 2>&1
}

case $STAGE in
    export) do_export ;;
    unit)   do_unit ;;
    e2e)    do_e2e ;;
    all)    do_export; do_unit; do_e2e ;;
    *) echo "unknown stage $STAGE"; exit 1 ;;
esac
