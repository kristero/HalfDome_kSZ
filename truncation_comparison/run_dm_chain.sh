#!/bin/bash
set -euo pipefail
cd /home/cbllover/HalfDome/truncation_comparison
export JULIA_DEPOT_PATH=/home/kn18001/.julia
J=/home/kn18001/.juliaup/bin/julia
$J -t 20 build_dm_caches.jl 2>&1 | grep -v "^\[interpolator\] START"
echo "=== test paint"; $J -t 20 paint_dm_truncation_maps.jl --nside=1024 --max-halos=300000 --chunk-size=100000 --output-dir=maps_test --tag=_test 2>&1 | grep -v "^\[interpolator\]\|Found cached"
echo "=== full paint $(date)"; $J -t 20 paint_dm_truncation_maps.jl --nside=4096 --max-halos=0 --chunk-size=1000000 --output-dir=maps 2>&1 | grep -v "^\[interpolator\]\|Found cached"
echo "dm chain done $(date)"
