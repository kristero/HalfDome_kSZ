# FRB Map Generation Code

This folder collects the code needed to generate the FRB-related maps from the
HalfDome workspace. It intentionally excludes large inputs and outputs such as
HDF5 lightcones, FITS maps, JLD2 interpolator caches, and plot products.

## Entry Points

- `make_random_frb_dm_map_z1.jl`
  - Standalone fixed-redshift random FRB DM map generator.
  - Draws `N` random HEALPix source pixels, puts every FRB at the same
    `z_source`, accumulates foreground halo DM only at those source pixels, and
    writes a sparse FRB DM FITS map.
  - With `save_foreground_map=true`, also paints and saves the continuous
    foreground DM map. With `save_power_spectrum=true`, saves the foreground
    power spectrum table and log-log plot from that continuous foreground map.

- `paint_halfdome_full_foreground_dm_map.jl`
  - Fully paints the HalfDome foreground halo DM field.
  - Default `z_source=1`, foreground cut `0 <= z_halo <= 1`, and `N=10000`
    random FRB source pixels sampled from the finished full map.
  - Writes both the full foreground DM FITS map and a sparse sampled-source
    FITS map/CSV.

- `make_stellar_weighted_frb_los_dm_map.jl`
  - Selects FRB host halos in a fixed redshift shell with probability
    proportional to `Mstar^alpha_star`.
  - With `source_selection_mode=all`, selects FRB host halos across the full
    chosen HalfDome source redshift range, still with probability proportional
    to `Mstar^alpha_star`. In this mode each FRB uses its own host redshift for
    the LOS cut, so only halos with `z_halo < z_FRB` contribute to that FRB.
  - If the catalog has no stellar-mass dataset, `stellar_mass_field=auto`
    computes `Mstar(Mh,z)` from the analytic stellar-mass-halo-mass relation.
  - Computes foreground DM only along the selected host sightlines. Shell mode
    uses `z_halo < z_source`; all-redshift mode uses each FRB's own
    `z_halo < z_FRB` cut.
  - Writes the observed sparse LOS-DM FITS map, FRB count map, host catalog,
    summary, DM log-log PDF histogram, host stellar-mass histogram, continuous
    foreground DM FITS map, foreground power spectrum CSV table, and foreground
    log-log power spectrum plot.
  - With `save_frb_corrected_estimator=true`, also saves an inverse-density
    weighted sparse FRB estimator map and shot-noise-corrected FRB estimator
    power spectrum. The default correction is `frb_corrected_shot_noise=shuffle`
    with `frb_corrected_n_shuffle=5`.
  - Use the sparse LOS-DM map/PDF for the mock FRB observations. Use the
    continuous foreground DM map and `*_foreground_dm_power_spectrum*` files for
    the full-sky foreground `C_ell`.

- `stellar_mass_weighted_hosts.py`
  - NumPy/Matplotlib helper for selecting FRB host/source positions with
    probability proportional to `Mstar^alpha_star`.
  - Does not modify foreground gas painting or DM integration.
  - Saves the all-shell vs selected-host stellar-mass diagnostic histogram.

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
julia make_random_frb_dm_map_z1.jl N=10000 z_source=1.0 output_dir=frb_map_generation/outputs/random_frb_z1
```

Example full foreground DM map to z=1, plus 10000 random FRB samples:

```bash
julia paint_halfdome_full_foreground_dm_map.jl N=10000 z_source=1.0 output_dir=frb_map_generation/outputs/halfdome_full_foreground_dm_z1
```

Example stellar-mass-weighted observed FRB LOS-DM map:

```bash
julia make_stellar_weighted_frb_los_dm_map.jl N=10000 z_source=1.0 dz=0.02 alpha_star=1.0 stellar_mass_field=auto output_dir=frb_map_generation/outputs/stellar_weighted_frb_los_dm_z1
```

Example PBS cluster run for both stellar-weighted and random-host control:

```bash
qsub frb_map_generation/run_stellar_weighted_frb_los_cluster.pbs
```

The PBS defaults run `alpha_star=1.0` and `alpha_star=0.0` with the same
redshift shell and foreground LOS-DM integration.

Example host-sampled workflow run:

```bash
julia tSZ_cross_FRB_halo_test.jl frb_count=10000 frb_selection_mode=redshift dm_los_mode=direct
```
