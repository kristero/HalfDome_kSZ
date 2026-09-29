#!/usr/bin/env bash
# Workstream A (XGPaint / painted-profile audit), 2026-09-29. Local, light (< 5 min total).
# Writes only to frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit/.
set -euo pipefail
A=/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/A_xgpaint_audit
O=/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit
SCRATCH=${SCRATCH:-/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad}
J2=/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia
J7=/root/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia      # only to read its own precompile headers
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
export LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib
# scratch depot first: any new compile cache goes there, not into shared depots
export JULIA_DEPOT_PATH=$SCRATCH/julia_depot:/home/kn18001/.julia
mkdir -p "$O" "$SCRATCH/julia_depot"
cd "$SCRATCH"
# goal 1: provenance (source hashes, git trees)
rm -f "$O/xgpaint_precompile_cache_hashes.txt"
$J2 --startup-file=no $A/git_tree_crc32c.jl "$O/xgpaint_git_tree_crc32c.txt"
for J in $J2 $J7; do for D in /home/kn18001/.julia /home/cbllover/HalfDome/frb_map_generation/outputs/tsz_frb_1m_z1_r200c_apertures/.julia_depot; do
  $J --startup-file=no $A/ji_source_hashes.jl "$D" "$O/xgpaint_precompile_cache_hashes.txt"; done; done
git -C /home/kn18001/.julia/dev/XGPaint diff e67a8a1 HEAD -- src > "$O/xgpaint_diff_e67a8a1_to_HEAD_src.patch"
# goal 2: production interpolators + cache vintage test (Julia), independent LOS (Python)
JULIA_NUM_THREADS=8 $J2 --startup-file=no $A/eval_production_interpolators.jl
$PY $A/compare_painted_profiles.py
# map level: repaint three discs of each production y map with today's code (~2 min, ~5 GB RAM)
JULIA_NUM_THREADS=4 $J2 --startup-file=no $A/map_patch_repaint_test.jl
# goal 3: production Julia density at table points, then the tables (+ cross-check)
$J2 --startup-file=no $A/julia_density_crosscheck.jl
$PY $A/make_profile_tables.py
$PY $A/pressure_ratio_summary.py
# goal 4: y_prof.cpp bookkeeping (sources fetched into $O/group_particles_fetch/)
$PY $A/yprof_bookkeeping_check.py
$PY $A/plot_audit_figures.py
