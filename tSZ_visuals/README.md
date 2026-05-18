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
julia tSZ_visuals/run_full_map.jl catalog_source=halfdome nside=1024 sobol_csv_path=Sobol_tSZ/battaglia_sobol_512.csv sobol_row=1
julia tSZ_visuals/run_sobol_full_maps.jl catalog_source=halfdome nside=1024 sobol_csv_path=Sobol_tSZ/battaglia_sobol_512.csv sobol_row_start=1 sobol_row_stop=8
julia tSZ_visuals/run_single_profile.jl mass_msun=1e14 redshift=0.5 radius_comoving_mpc=1.0 ra_deg=0 dec_deg=0 nside=1024
```

The batch FITS outputs are cumulative: each saved y and mass map includes all halos painted so far, not only the halos inside that one bin.
For mass binning, the cumulative order now runs from the smallest selected halos to the largest selected halos.
Set `cumulative_bin_maps=false` to save only the map of each individual bin instead of the running cumulative sum.

The default run writes only y and mass FITS maps. Add `save_cl=true` if you also want the tSZ angular power spectrum FITS output.
The single-profile runner always writes the y map, the mass map, and the tSZ angular power spectrum FITS output.
Use `batching_mode=full save_bin_maps=false save_cl=true` when you want only the final accumulated full-simulation Healpix map plus the corresponding `C_l`.
Add `save_mass_map=false` if you do not want any mass FITS output.
At high resolution, cap the harmonic transform with `cl_lmax=4096` unless the job requests enough memory for the Healpix default `lmax=3*nside-1`. The production PBS sets `CL_LMAX=4096` by default to avoid `anafast` out-of-memory failures at `NSIDE=4096` with 10 GB jobs.

Interpolator cache behavior:

- `model_exists=true` now requires an existing `.jld2` interpolator cache. The code checks `cache_dir` first and then the older repo-root cache location.
- Use `cache_dir=...` to point at a directory that already contains the cache, or copy the cache there before the run.
- Use `model_exists=false` only when you intentionally want to rebuild the Battaglia interpolator cache. This is expensive on the cluster; request enough walltime/memory and avoid launching several cache builds in parallel.
- Use `reuse_existing_cache=true` with `model_exists=false` for resume runs: an existing row cache is loaded, while missing caches are still built.
- Use `cache_wait_seconds=N` with `model_exists=true` when a dependent lightcone should start before every cache exists; it waits for each missing cache instead of failing immediately.
- Use `skip_existing_outputs=true skip_existing_any_run_instance=true` for resume runs so completed products are skipped even when the previous PBS job used a different `run_instance_tag`.
- Cache-build controls are exposed as `interpolator_pad=256` and `interpolator_logM_max=15.7`. The PBS Sobol runner also uses `INTERPOLATOR_TIMEOUT_SECONDS=700` by default for cache builds.
- New caches are keyed by a stable hash of the physical Battaglia parameters plus cosmology, not by the Sobol CSV filename/row. This lets y100/y102 reuse the same interpolator even when the same parameter point is referenced through different split CSV files. Older row-tagged cache names are still checked as fallbacks.
- To submit the full production set, place the 512-row Sobol table at `/home/kristero10/tSZ_data/battaglia_sobol_512.csv` and run `bash submit_y100_y102_sobol512.sh` from this directory on the cluster login node. The wrapper creates four 128-row split CSVs, submits y100 splits 1-4 with `model_exists=false`, then submits the matching y102 jobs with `model_exists=true` and `afterok` dependencies so they reuse the caches built by y100.

Sobol CSV support:

- `sobol_csv_path=...` points to the Sobol table.
- `sobol_row=N` loads one Battaglia model from row `N` of that CSV.
- The Sobol loader expects columns for `P0`, `xc`, `beta`, `alpha_m_P0`, `alpha_m_xc`, `alpha_m_beta`, `alpha_z_P0`, `alpha_z_xc`, and `alpha_z_beta`.
- `alpha`, `gamma`, and their exponents stay at the fiducial defaults unless you override them directly in the CLI.
- Battaglia guardrails are enabled by default with `enforce_battaglia_guardrails=true skip_invalid_battaglia_rows=true`. The guarded priors match the current 512-point Sobol ranges; derived checks reject only non-finite, non-positive, or line-of-sight divergent profiles. Slow but physical interpolators are skipped by the PBS timeout and listed in the redo log.

Emulator test run:

- `run_battaglia_emulator_optuna.pbs` now defaults to a test emulator using `/lustre/work/kristero10/tSZ_data/y100` and `/lustre/work/kristero10/tSZ_data/y102`.
- Its default Sobol CSVs are `battaglia_sobol_512_1.csv` through `battaglia_sobol_512_4.csv`; it intentionally excludes older 256-point products.
- The default `PROFILE_INCLUDE_REGEX=sobol_battaglia_sobol_512_[1-4]_row[0-9]+` and `PROFILE_EXCLUDE_REGEX=sobol_battaglia_sobol_512_row[0-9]+|sobol_battaglia_sobol_256` keep older products out of the training set.
- The test run writes to `/lustre/work/kristero10/tSZ_data/emulator_battaglia_y100_y102_test`, uses 20 Optuna trials, and requires 512 matched y100/y102 parameter points by default.
