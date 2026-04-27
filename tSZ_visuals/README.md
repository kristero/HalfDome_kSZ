# tSZ Visual FITS Pipeline

This folder now separates the visual FITS pipeline into small pieces:

- `HalfDome_tSZ.jl`: compatibility entry point; calls the structured pipeline.
- `run_tSZ_visuals.jl`: main runner that loads config, model, catalogs, painting, and output code.
- `run_full_map.jl`: run one full accumulated Healpix map and its `C_l`, with no saved mass/redshift/chunk split outputs.
- `run_sobol_full_maps.jl`: run the full-map pipeline for a range of Sobol CSV rows.
- `run_single_profile.jl`: paint one synthetic halo with the fiducial Battaglia profile and save the y map, mass map, and tSZ `C_l`.
- `run_by_redshift.jl`: run cumulative FITS outputs in redshift bins.
- `run_by_mass.jl`: run cumulative FITS outputs in mass bins.
- `run_initial_batches.jl`: run cumulative FITS outputs in the original catalog chunk order.
- `catalog_halfdome.jl`: HalfDome HDF5 loading and batching.
- `catalog_websky.jl`: streaming WebSky PKSC loading and batching.
- `painting.jl`: tSZ painting and radial halo mass-map painting.
- `output.jl`: FITS writing for the cumulative y and mass maps.
- `config.jl`: command-line and environment configuration.

Example calls:

```bash
julia tSZ_visuals/run_by_redshift.jl catalog_source=halfdome redshift_bin_width=1.0
julia tSZ_visuals/run_by_redshift.jl catalog_source=websky redshift_binning_mode=log1p log_redshift_bin_width=0.2
julia tSZ_visuals/run_by_mass.jl catalog_source=halfdome mass_bin_width_dex=0.5
julia tSZ_visuals/run_by_mass.jl catalog_source=halfdome mass_bin_width_dex=0.5 cumulative_bin_maps=false
julia tSZ_visuals/run_initial_batches.jl catalog_source=websky chunkN=2000000
julia tSZ_visuals/run_full_map.jl catalog_source=halfdome nside=1024 save_mass_map=false
julia tSZ_visuals/run_full_map.jl catalog_source=halfdome nside=1024 sobol_csv_path=Sobol_tSZ/battaglia_sobol_256.csv sobol_row=1
julia tSZ_visuals/run_sobol_full_maps.jl catalog_source=halfdome nside=1024 sobol_csv_path=Sobol_tSZ/battaglia_sobol_256.csv sobol_row_start=1 sobol_row_stop=8
julia tSZ_visuals/run_single_profile.jl mass_msun=1e14 redshift=0.5 radius_comoving_mpc=1.0 ra_deg=0 dec_deg=0 nside=1024
```

The batch FITS outputs are cumulative: each saved y and mass map includes all halos painted so far, not only the halos inside that one bin.
For mass binning, the cumulative order now runs from the smallest selected halos to the largest selected halos.
Set `cumulative_bin_maps=false` to save only the map of each individual bin instead of the running cumulative sum.

The default run writes only y and mass FITS maps. Add `save_cl=true` if you also want the tSZ angular power spectrum FITS output.
The single-profile runner always writes the y map, the mass map, and the tSZ angular power spectrum FITS output.
Use `batching_mode=full save_bin_maps=false save_cl=true` when you want only the final accumulated full-simulation Healpix map plus the corresponding `C_l`.
Add `save_mass_map=false` if you do not want any mass FITS output.

Sobol CSV support:

- `sobol_csv_path=...` points to the Sobol table.
- `sobol_row=N` loads one Battaglia model from row `N` of that CSV.
- The Sobol loader expects columns for `P0`, `xc`, `beta`, `alpha_m_P0`, `alpha_m_xc`, `alpha_m_beta`, `alpha_z_P0`, `alpha_z_xc`, and `alpha_z_beta`.
- `alpha`, `gamma`, and their exponents stay at the fiducial defaults unless you override them directly in the CLI.
