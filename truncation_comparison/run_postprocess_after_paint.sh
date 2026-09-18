#!/bin/bash
# Waits for the full painter to finish, then computes the catalogue Poisson terms and the spectra/figures.
set -euo pipefail
cd /home/cbllover/HalfDome/truncation_comparison
export JULIA_DEPOT_PATH=/home/kn18001/.julia
while kill -0 528035 2>/dev/null; do sleep 30; done
echo "painter finished at $(date)"; tail -6 logs/paint_nside4096.log
ls -la maps/
/home/kn18001/.juliaup/bin/julia -t 20 compute_catalogue_poisson_terms.jl maps/halo_mass_redshift_histogram.h5 spectra/catalogue_poisson_terms.h5 2>&1 | grep -v "^\[interpolator\]\|Found cached"
/home/cbllover/miniconda3/envs/halfdome/bin/python compute_spectra_and_plots.py --maps-dir maps --out-dir spectra --poisson spectra/catalogue_poisson_terms.h5
echo "postprocess done at $(date)"
