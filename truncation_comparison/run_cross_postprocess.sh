#!/bin/bash
set -euo pipefail
cd /home/cbllover/HalfDome/truncation_comparison
export JULIA_DEPOT_PATH=/home/kn18001/.julia
until grep -q "dm chain done" logs/dm_chain.log; do sleep 30; done
echo "DM paint finished $(date)"; grep -v "^\[interpolator\]" logs/dm_chain.log | tail -7
/home/kn18001/.juliaup/bin/julia -t 20 compute_catalogue_poisson_cross_terms.jl maps/halo_mass_redshift_histogram.h5 spectra/catalogue_poisson_cross_terms.h5 2>&1 | grep -v "^\[interpolator\]\|Found cached"
/home/cbllover/miniconda3/envs/halfdome/bin/python compute_cross_spectra_and_plots.py --maps-dir maps --out-dir spectra --poisson spectra/catalogue_poisson_cross_terms.h5
echo "cross postprocess done $(date)"
