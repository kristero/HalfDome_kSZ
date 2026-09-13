# Mock-generation audit: what changed and what the checks establish

The active folder is `SBI_analysis/so_32k_independent_noise`. The default still
generates two 32768-row datasets, baseline deproj0 only. No default prior,
noise normalization, cosmology or number of simulations changed in this update.

## 2. Threaded halo painting

The Compton-y map is a sum of halo contributions. For a given pixel p,

    y[p] = sum_h y_h[p].

The original XGPaint painter distributed halos between threads but let those
threads add to the same map array. Halo footprints overlap. A floating-point
`map[p] += contribution` is a read/add/write operation, not an atomic addition.
If the starting value is 10, thread A adding 2 and thread B adding 3 can both
read 10 and write 12 and 13. The final value can be 13 instead of the correct 15.
Sorting halos by declination does not guarantee disjoint footprints.

`safe_paint.jl` adds a HEALPix-specific `XGPaint.paint!` method. It creates a lock
for each ring and holds that ring's lock while adding a halo contribution to
the ring's pixels. It releases that lock before moving to another ring.
Non-overlapping rings can still be painted concurrently. It does not allocate
one NSIDE4096 map per thread and does not change the pressure formula, radial
cutoff, masses, beam, mask or noise generation.

`simulate.jl` includes this method before running the simulator. The dispatch
therefore replaces the unsafe accumulation in this handoff without editing the
original vendored source snapshot. The worker records/checks `Painter: ring_locked`.

The saved regression test uses 1000 deliberately overlapping toy profiles and
compares five threaded runs with a serial reference. The old painter differed
by up to 0.529% in relative L1 map error; the ring-locked result matched the
serial map in this test. See `validation_plots/05_threaded_painter_regression.png`.

This percentage is NOT a measured bias of the actual HalfDome catalogue or its
power spectrum. It proves a race exists, not how large its production impact is.
Parallel summation order can still change the last floating-point bits. The fix
prevents lost increments, not every possible numerical error in halo modeling.
There is no justified universal rescaling that repairs previously affected maps.

## 3. Beam setting and preprocessing consistency

The included SO driver sets fallback environment values for the beam, including
`GAUSSIAN_BEAM_FWHM_ARCMIN=2.0`. The downstream argument helper gives environment
variables precedence over CLI arguments. Consequently, a request for 3 arcmin
could still run with the fallback 2 arcmin value.

The wrapper now copies the explicit beam CLI setting into the corresponding
environment value after including the driver, before loading the configuration.
It prints the actual FWHM used, which Python compares with `config.json` before
writing a row-completion marker. `check_simulator_cli.py` tested both 2 and
3 arcmin through the real Julia configuration path, for both parameter designs.
Those checks do not paint maps.

For a Gaussian beam, before masking,

    B_ell = exp[-ell(ell+1) sigma_beam^2 / 2],
    sigma_beam = FWHM_radians / sqrt(8 ln 2),
    C_ell(signal, smoothed) = B_ell^2 C_ell(signal).

An incorrect width therefore changes high-ell signal power much more strongly
than low-ell power. It is not just a constant amplitude error. After masking,
mode coupling must also be included to relate the masked and full-sky spectra.

The default was and remains 2 arcmin. This override issue is NOT evidence that
an earlier run explicitly intended to use 2 arcmin had the wrong beam. The
patch makes changed widths effective and auditable. Noise is not convolved with
an additional beam; the existing table convention is retained. SBI training,
observations and Fisher predictions must use that same observable definition.

## 4. Checks before a row or dataset is accepted

### Inputs and identity

- Parameter CSV headers must match the nine canonical names in their exact order.
- There must be exactly the configured number of rows, all finite and inside
  the explicitly saved generation prior. Bounds are not inferred from sample extrema.
- In the two-parameter design, the other seven columns must equal Battaglia12.
- CSV hashes, row counts and offsets must match the design metadata. A row is
  identified by its local CSV row and global offset, not filesystem scan order.
- Noise tables must have every integer ell from 80 to 7979 exactly once, in
  order, with finite nonnegative power in the selected columns. Julia column 2
  is deproj0 and column 4 is deproj2; Python uses indices 1 and 3 respectively.
- Seed ranges are disjoint between rows, splits, products and the two designs.
  The mask seed stays fixed. The check is arithmetic, so it scales to 524288
  rows without storing millions of seed values in every worker.
- Source/design/noise hashes, configuration and catalogue path/size/mtime form
  an experiment identity. Changed inputs require a new output root.

The catalogue identity currently uses path/size/mtime, not a full content hash.
The HDF5 runtime check verifies array lengths and required keys; it cannot prove
the catalogue's physical units or reconstruct missing external provenance.

### Physical and runtime checks

The wrapper explicitly calls the pressure validator with derived checks on:
finite parameters, positive amplitudes/core scales, and line-of-sight convergence
of the inner and outer slopes. It checks representative masses and redshifts
using the bundled guardrail routine, not every possible halo in parameter space.
The old hard-coded prior is disabled because Python already validates the chosen
prior. Thus `enforce_battaglia_guardrails=false` in the generated CLI does NOT
mean that the wrapper skips its separate physical validation.

After the simulator finishes, Python checks the actual nine pressure parameters,
mask seed, both noise seeds for every selected product, beam width and painter
against the request. It requires all requested spectra, finite values and the
expected ell-vector length. Clean references written for different deprojections
must match: deprojection should not alter the clean signal map.

### Outputs and restart safety

- A row is complete only after validation and an atomic completion-marker write.
- SHA256 hashes detect changed/truncated output files during resume/combination.
- Row locks prevent simultaneous duplicate work. Submission and combination
  locks prevent concurrent operations from overwriting each other's bookkeeping.
- Failed smoke runs return a nonzero exit status. Missing rows are reported;
  the combiner does not silently drop them and shift subsequent theta labels.
- Recombination removes its previous completion marker first, writes temporary
  arrays, and writes a new marker only after all final outputs are available.
- Theta remains float64; it is not rounded across a saved prior boundary.
- Signed C_ell and D_ell are preserved. Binning first forms D_ell, then averages
  it with 2ell+1 weights. There is no floor, logarithm or asinh at generation.

These are consistency checks, not a proof that the simulator is a complete
physical model of SO. Instrumental noise is independent, but the halo lightcone
and mask are fixed. Different noise products are independent mock configurations,
not a joint correlated multifrequency-ILC simulation. The confirmed N_ell-per-split
normalization remains unchanged. A full-resolution cluster smoke run is still needed.

## Extending to 524288 rows and other noise products

No simulator-code edit is needed for baseline/goal and deproj0/2. In a separate
extracted handoff folder, activate the supplied example:

```bash
cp config_524288_all_noise.example.json config.json
python3 make_designs.py --output designs_524288 \
  --source-two-param /path/to/battaglia_sobol_P0_beta_524288.csv \
  --source-nine-param /path/to/battaglia_sobol_1048576.csv
```

The example enables baseline and goal for both deprojections. For baseline0
only, set `noise_cases` to `["baseline"]`, `deprojections` to `[0]`, keeping
`default_product` as `"baseline_deproj0"`. For goal2 only, use `["goal"]`, `[2]`
and `"goal_deproj2"`. Each row paints the signal once and reuses it for the
selected products. The clean reference is saved automatically.

The original full CSVs preserve the exact 32k prefixes; the source hashes in
`designs/provenance.json` identify them. They are not included in the compact
handoff archive. Omitting the source options generates a NEW scrambled Sobol
design; it must not be combined with a different sequence's existing prefix.

Edit `cluster.env` to use a new `OUTPUT_ROOT`, then run:

```bash
source cluster.env
"$PYTHON" generate.py check
"$JULIA" --startup-file=no --project=vendor/XGPaint check_runtime.jl designs_524288
bash submit.sh smoke
# After the smoke completes and its logs/outputs pass inspection:
bash submit.sh wave
```

Changing N does not change a row's assigned seed. Changing worker count only
changes scheduling. With the same number of products, 524288 rows involve
16 times as many simulations as 32768 per design; enabling more products also
increases per-row cost. Twenty workers are not a promise of completion in 24 hours.
For an arbitrary new noise-curve format, correlated splits, or different sky
statistics, a loader/forward-model change and renewed physical validation are
needed; those changes should not be represented merely by renaming a case.

## Reading the new Sobol plots

`sobol_plots/` contains PNG/PDF plots at N=32768 and N=524288, a parameter-statistics
CSV, and a JSON report with source hashes and prefix checks. These use the actual
existing parameter files. No new sky simulations were generated for these plots.

- The nine marginal panels have physical parameter axes and show both designs.
  All varied marginals are uniform in the stated linear prior. The orange
  vertical lines in seven panels mean FIXED parameters, not narrow posteriors.
- In 32 equal-width bins, the varied parameters have exactly 1024 rows per bin
  at N=32768 and 16384 at N=524288. The flat histograms are expected, not a failure.
- The nine-parameter corner plots use all rows on the diagonal but only the first
  1024 points in joint scatter panels for legibility. These joint panels match
  between the two sizes because the larger design contains the same prefix.
- The P0-beta coverage figure shows the first 256 and 2048 points, then full
  32768/524288-row histograms on a 64x64 grid. At those grid sizes the measured
  cell counts are uniform, which explains the solid-color density panels.
- No duplicated parameter rows or out-of-prior values were found. The two designs
  share prior bounds but are not matched row-by-row to each other's P0/beta pairs.
- Sobol is a low-discrepancy design, not an iid random sample. The report gives
  a descriptive CDF-versus-uniform distance, not a KS-test p-value. Uniform
  low-dimensional projections do not imply exhaustive coverage of a 9D box.

To regenerate plots for the currently configured tables:

```bash
python3 plot_sobol_designs.py
```

To include the existing larger sequence while retaining the default 32k config:

```bash
python3 plot_sobol_designs.py --sizes 32768 524288 \
  --two-param-csv /path/to/battaglia_sobol_P0_beta_524288.csv \
  --nine-param-csv /path/to/battaglia_sobol_1048576.csv
```
