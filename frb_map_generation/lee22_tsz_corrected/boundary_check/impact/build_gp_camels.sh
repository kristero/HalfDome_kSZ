#!/usr/bin/env bash
# Build the Lee22-era group_particles example programs (y_prof.cpp = electron pressure, ne_prof.cpp = electron
# density; tree c3fe9b8, 2021-06-25) for runs on single-file CAMELS IllustrisTNG snapshots.
#
#   GP_DIR=<clone of github.com/leanderthiele/group_particles> BUILD_DIR=<scratch> ./build_gp_camels.sh
#
# Trees (all exported with git archive from GP_DIR):
#   c3fe9b8        : Lee22-era tree, default build = periodicity-hint distance (the buggy path)
#   c3fe9b8_plain  : c3fe9b8 with ONLY the one-line hotfix 5791098 (2024-04-08) back-ported: plain minimum-image
#                    distance periodic_hypotsq(rgrp, rprt, Bsize); the cell search is unchanged ("fixed")
#   c3fe9b8_perfix : c3fe9b8 with ONLY the PER() macro made a signed comparison (alternative fix of the hint)
#   c3fe9b8 -DNAIVE: brute force over all particles (no cell search), plain minimum-image distance (reference)
# Program edits (logged as diffs): input ROOT -> "./input/", chunk max index 599 -> 0 (CAMELS files are single
# chunks, symlinked as chunk .0), output ROOT -> "./output" (y_prof) / unchanged relative "ne_prof_results_Jun22"
# (ne_prof), and the group mass cut Mmin -> 1e2 (1e12 Msun/h; y_prof had 1e3, ne_prof 1e4) so that more halos are
# profiled; the Lee22 mass range is selected afterwards (a group's profile does not depend on which other groups
# are selected).  Compiler flags verbatim from compile.sh.
set -euo pipefail
GP_DIR=${GP_DIR:?set GP_DIR}
BUILD_DIR=${BUILD_DIR:?set BUILD_DIR}
OUT_DIR=${OUT_DIR:-/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007/impact}
H5INC=${H5INC:-/usr/include/hdf5/serial}
H5LIB=${H5LIB:-/usr/lib/x86_64-linux-gnu/hdf5/serial}
unset LD_LIBRARY_PATH
REV=c3fe9b8
CXXFLAGS="-std=c++17 -O3 -ffast-math -funroll-loops -g3 -Wall -Wextra -Wno-unused-parameter -Wno-reorder"
mkdir -p "$BUILD_DIR" "$OUT_DIR/logs"
LOG="$OUT_DIR/logs"

# ---- trees
rm -rf "$BUILD_DIR/gp_$REV"; mkdir -p "$BUILD_DIR/gp_$REV"
git -C "$GP_DIR" archive "$REV" detail include examples compile.sh | tar -x -C "$BUILD_DIR/gp_$REV"
git -C "$GP_DIR" show -s --format="%H %ad %an %s" "$REV" > "$BUILD_DIR/gp_$REV/REVISION"
grep -q -- "-std=c++17 -O3 -ffast-math -funroll-loops" "$BUILD_DIR/gp_$REV/compile.sh" && \
    grep -q -- "-g3 -Wall -Wextra -Wno-unused-parameter -Wno-reorder" "$BUILD_DIR/gp_$REV/compile.sh" || { echo "compile.sh flags changed"; exit 1; }

rm -rf "$BUILD_DIR/gp_${REV}_plain"; cp -r "$BUILD_DIR/gp_$REV" "$BUILD_DIR/gp_${REV}_plain"
sed -i 's|^    coord_t Rsq = GeomUtils::periodic_hypotsq(rgrp, rprt, Bsize, periodic_to_add);|    coord_t Rsq = GeomUtils::periodic_hypotsq(rgrp, rprt, Bsize); // hotfix 5791098 back-ported|' \
    "$BUILD_DIR/gp_${REV}_plain/detail/prt_loop.hpp"
diff -u "$BUILD_DIR/gp_$REV/detail/prt_loop.hpp" "$BUILD_DIR/gp_${REV}_plain/detail/prt_loop.hpp" > "$LOG/patch_plain.diff" || true
[ "$(grep -c '^+ ' "$LOG/patch_plain.diff")" = 1 ] || { echo "plain patch did not apply exactly once"; exit 1; }

rm -rf "$BUILD_DIR/gp_${REV}_perfix"; cp -r "$BUILD_DIR/gp_$REV" "$BUILD_DIR/gp_${REV}_perfix"
sed -i 's/#define PER(x) ((x>=Ncells_side) ? 1 : (x<0) ? -1 : 0)/#define PER(x) ((x>=(int)Ncells_side) ? 1 : (x<0) ? -1 : 0)/' \
    "$BUILD_DIR/gp_${REV}_perfix/detail/workspace_sorting.hpp"
diff -u "$BUILD_DIR/gp_$REV/detail/workspace_sorting.hpp" "$BUILD_DIR/gp_${REV}_perfix/detail/workspace_sorting.hpp" > "$LOG/patch_perfix.diff" || true
[ "$(grep -c '^+ .*#define PER' "$LOG/patch_perfix.diff")" = 1 ] || { echo "perfix patch did not apply exactly once"; exit 1; }

# ---- program sources (I/O + mass-cut edits only)
SRC="$BUILD_DIR/src"; mkdir -p "$SRC"
sed -e 's|#define ROOT "/tigress/lthiele/Illustris_300-1_TNG/output/"|#define ROOT "./input/"|' \
    -e 's|grp_max_idx = 599;|grp_max_idx = 0;|' \
    -e 's|prt_max_idx = 599;|prt_max_idx = 0;|' \
    -e 's|#define ROOT "DEFINE YOUR OUTPUT PATH HERE"|#define ROOT "./output"|' \
    -e 's|Mmin = 1e3F;|Mmin = 1e2F;|' \
    "$BUILD_DIR/gp_$REV/examples/y_prof.cpp" > "$SRC/y_prof.cpp"
diff "$BUILD_DIR/gp_$REV/examples/y_prof.cpp" "$SRC/y_prof.cpp" > "$LOG/y_prof_edits.diff" || true
[ "$(grep -c '^>' "$LOG/y_prof_edits.diff")" = 5 ] || { echo "y_prof edits != 5 lines"; exit 1; }
sed -e 's|#define ROOT "/tigress/lthiele/Illustris_300-1_TNG/output/"|#define ROOT "./input/"|' \
    -e 's|grp_max_idx = 599;|grp_max_idx = 0;|' \
    -e 's|prt_max_idx = 599;|prt_max_idx = 0;|' \
    -e 's|Mmin = 1e4F;|Mmin = 1e2F;|' \
    "$BUILD_DIR/gp_$REV/examples/ne_prof.cpp" > "$SRC/ne_prof.cpp"
diff "$BUILD_DIR/gp_$REV/examples/ne_prof.cpp" "$SRC/ne_prof.cpp" > "$LOG/ne_prof_edits.diff" || true
[ "$(grep -c '^>' "$LOG/ne_prof_edits.diff")" = 4 ] || { echo "ne_prof edits != 4 lines"; exit 1; }

# ---- compile
build () {   # $1 tree, $2 program, $3 binary name, $4 extra flags
    local T="$BUILD_DIR/$1"
    g++ $CXXFLAGS $4 -I"$T/include" -I"$T/include/callback_utils" -I"$T/detail" -I"$H5INC" \
        -o "$BUILD_DIR/$3" "$SRC/$2.cpp" -L"$H5LIB" -lhdf5 -lhdf5_cpp -fopenmp > "$LOG/compile_$3.log" 2>&1
    echo "built $3 ($(grep -c 'warning' "$LOG/compile_$3.log") warning lines; Wsign-compare at PER: $(grep -c 'Wsign-compare' "$LOG/compile_$3.log"))"
}
for prog in y_prof ne_prof; do
    build gp_$REV $prog ${prog}_buggy ""
    build gp_${REV}_plain $prog ${prog}_plain ""
    build gp_${REV}_perfix $prog ${prog}_perfix ""
    build gp_$REV $prog ${prog}_naive "-DNAIVE"
done
for t in gp_$REV gp_${REV}_plain gp_${REV}_perfix; do echo "$t: $(cat "$BUILD_DIR/gp_$REV/REVISION")"; done > "$LOG/build_provenance.txt"
for prog in y_prof ne_prof; do for v in buggy plain perfix naive; do md5sum "$BUILD_DIR/${prog}_$v"; done; done >> "$LOG/build_provenance.txt"
