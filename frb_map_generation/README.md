# FRB Map Generation Code

This folder collects the code needed to generate the FRB-related maps from the
HalfDome workspace. It intentionally excludes large inputs and outputs such as
HDF5 lightcones, FITS maps, JLD2 interpolator caches, and plot products.

## Entry Points

- `make_random_frb_dm_map_z1.jl`
  - Standalone fixed-redshift random FRB DM map generator.
  - Draws random HEALPix pixels, accumulates foreground halo DM up to
    `frb_redshift`, and writes a sparse FRB DM FITS map.

- `tSZ_cross_FRB_halo_test.jl`
  - Full host-sampled workflow.
  - Samples FRB host halos, builds an FRB overdensity map, computes foreground
    DM or DM residual maps, optionally computes tSZ x FRB/DM spectra, and writes
    FITS/HDF5/diagnostic products.

- `compute_frb_dm_xgpaint_backend.jl`
  - Julia backend used by the Python analysis script.
  - Reads an FRB catalog HDF5 file, optionally writes the FRB overdensity map,
    and writes a CSV table of XGPaint DM values.

- `analyze_frb_dm_mass_relation.py`
  - Analysis wrapper around `compute_frb_dm_xgpaint_backend.jl`.
  - Useful for catalog-derived FRB map checks and DM PDF plots.

## Local Helpers

- `utils.jl`
  - Shared coordinate and catalog utility code used by `tSZ_cross_FRB_halo_test.jl`.

- `SOConvertNFW.jl`
  - Local mass-conversion helper included by `tSZ_cross_FRB_halo_test.jl`.

- `profiles.jl`
  - Local XGPaint profile/interpolator helper code retained for reproducibility
    of the FRB DM profile path.

## External Inputs

The scripts still expect the original external inputs to exist or be passed via
arguments/environment variables:

- HalfDome catalog, usually `lightcone_100.hdf5`.
- Optional WebSky catalog path, usually `other_sims/sims/halos.pksc`.
- XGPaint/Julia package environment with `XGPaint`, `Healpix`, `HDF5`,
  `Interpolations`, and plotting dependencies installed.
- Optional interpolator caches such as `cached_FRB_true_DM_Websky_cosmo.jld2`.

Example fixed-redshift run:

```bash
julia make_random_frb_dm_map_z1.jl output_dir=batched_data/frb_random frb_redshift=1.0 frb_count=10000
```

Example host-sampled workflow run:

```bash
julia tSZ_cross_FRB_halo_test.jl frb_count=10000 frb_selection_mode=redshift dm_los_mode=direct
```
