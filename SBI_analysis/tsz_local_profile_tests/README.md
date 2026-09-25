# Local single-profile tests of the tSZ dataset engine

`tsz_single_profile_local.ipynb` paints **one** Compton-y sky with the frozen code of
`tSZ_64k_flat_prior_SO_baseline_deproj0/` (the bundle the collaborator runs) and shows the
intermediate maps the dataset never stores: raw painted map, map after the 2 arcmin beam,
apodized f_sky = 0.4 mask, one SO noise split, masked signal + noise, and the three spectra
the dataset keeps (full-sky clean, masked clean, masked noisy cross) plus the 40-bin version.
Nothing is written to disk; maps stay in memory and are released as the notebook proceeds.

## Setup (once)

1. Julia environment: `julia_env/` is the bundle's validated `Project.toml`/`Manifest.toml`
   (Julia 1.12.2, XGPaint tag `validated-tsz256-20260922`, Healpix 4.2.4, ...) plus IJulia and
   Plots, added with every pinned version preserved. `setup/verify_env.jl` of the bundle passes
   on it ("12 package versions and 17 XGPaint files match the validated 256-row test").
   It is already instantiated in the shared depot `/home/kn18001/.julia`. To rebuild elsewhere:

   ```bash
   JULIA_DEPOT_PATH=/home/kn18001/.julia /home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia \
     --project=SBI_analysis/tsz_local_profile_tests/julia_env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
   ```

2. Jupyter kernel: `./install_kernel.sh` copies `kernelspec/julia-1.12.2-tsz64k/kernel.json` to
   `~/.local/share/jupyter/kernels/`. The kernel starts the local Julia 1.12.2 binary (not the
   `juliaup` launcher, which resolves to 1.12.7 for root) with `--project=julia_env` and
   `--threads=20`. Pick "Julia 1.12.2 tSZ64k (20 threads)" in JupyterLab.

3. The HalfDome catalogue must be at `/home/cbllover/HalfDome/lightcone_100.hdf5` (or set
   `HALFDOME_CATALOGUE`); the repository root can be changed with `HALFDOME_REPO`.

## Using the notebook

The first cell chooses the fidelity preset, the profile parameters and the noise seeds:

| setting | values |
|---|---|
| `PRESET` | `:production` (raw NSIDE 8192 -> 4096 output, lmax 7979; exactly the dataset engine, 13-31 min observed, about 14 GiB), `:preview` (4096 -> 2048, 5-16 min observed, about 6 GiB), `:smoke` (2048 -> 1024, halos above 1e13 Msun only; about 1 min; not a valid sky) |
| `THETA_SOURCE` | `:random` (uniform in the flat prior, from `RANDOM_SEED`), `:design_row` (row `DESIGN_ROW` of the 64k design; rows 0-255 are the 256-row test), `:battaglia12` |
| `NOISE_SEEDS` | `:test256` (the 256-row test's seeds for that row), `:design64k` (the collaborator's dataset seeds), `:random` |

The same four choices can be given as environment variables `HALFDOME_NB_PRESET`,
`HALFDOME_NB_THETA`, `HALFDOME_NB_ROW`, `HALFDOME_NB_SEEDS` for unattended runs, e.g.

```bash
cd SBI_analysis/tsz_local_profile_tests
HALFDOME_NB_PRESET=production HALFDOME_NB_THETA=design_row HALFDOME_NB_ROW=0 HALFDOME_NB_SEEDS=test256 \
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=julia-1.12.2-tsz64k --output executed.ipynb tsz_single_profile_local.ipynb
```

With `:production`, a design row 0-255 and `:test256` seeds the last section is a local
repro-check against the idark test (metric of `manage.py spectrum_metrics`, tolerance 1e-6).

Measured on this machine (i5-13600HX laptop CPU, 20 threads, 25 September 2026), `:production`,
design row 0, `:test256` seeds, two runs: 13 and 31 min wall, 13.9 GiB peak RSS. The painting
step took 477 s in one run and 1550 s in the other for identical work (the CPU throttles);
map2alm 35 s, alm2map 19 s, catalogue reads 7 s with a warm page cache and about 220 s cold.
85,224,251 halos; the mask is the dataset mask (SHA-256 a6c3d64d...); full-sky clean and masked
clean spectra agree with the idark test to 4.4e-15 and 4.2e-15, the noisy cross to 7.6e-10
(noise maps differ in the last bit, as `repro-check` allows for other CPUs).
`executed_examples/` holds that run and a `:preview` run of the Battaglia+12 profile, with
their outputs (PNG figures only).

## What is reused and what is not

- Included unchanged, in the order of `engine/engine.jl`: `engine/benchmark.jl` (which loads
  `fullsky_test.jl`, `halfdome_sources/code/process_maps.jl`, the HalfDome tSZ operators and
  `spherical_truncation_profiles.jl`), `smooth_exterior.jl`, `balanced_painter.jl`.
- Copied into cells because `engine.jl` runs a production task on load: `probe_cache`
  (identical) and `make_signal` (identical for `:production`/`:preview`; it takes `lmax` as an
  argument so the `:smoke` preset stays consistent).
- The test's command line is read from `reference/test256/base_command.json` and the
  environment is scrubbed as `pipeline/common.py` does (local paths substituted; for
  `:production` every value equals the test's).
- The dataset paints four rows per catalogue pass; here `paint_shared!` runs with one
  profile. The pressure arithmetic, cache, mask, noise and spectra are the same functions.
