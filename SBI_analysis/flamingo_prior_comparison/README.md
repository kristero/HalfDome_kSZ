# Prior comparison with the new FLAMINGO fits

This is a separate plotting analysis. It reads the frozen 8,192-row production
design, the verified original 40-bin SBI prior, and the selected cosmology-corrected
FLAMINGO fits. It does not change or resubmit the production run.

## Figures and data

The cluster-rendered figures are copied to `cluster_results/plots/`:

- `prior_all_parameters.png` and `.pdf`: all nine one-dimensional marginals,
  with the old prior boundaries and the new FLAMINGO fit values.
- `joint_prior_all_parameters.png` and `.pdf`: all 36 parameter pairs and all
  nine marginals. Use the large-format PDF to inspect the correlations.
- `joint_prior.png` and `.pdf`: updated versions of the original three projections.
- `prior_and_fits.csv`: exact old bounds, extended rectangular bounds, accepted
  sample extrema, Battaglia12, and all three new parameter vectors.
- `plot_provenance.json`: input/output SHA-256 hashes, support checks, parameter
  ordering, and the original optimizer status for each selected fit.

Grey points and histograms show the actual accepted production design. They
approximate the extended prior after all joint restrictions. They do not show
FLAMINGO posterior samples. Histogram heights are probability mass per bin;
the P0 and xc bins are logarithmically spaced and the other bins are linearly
spaced. Marker heights in the marginal panels are offset for readability and
have no statistical meaning. Dashed purple lines/rectangles show the original
trained SBI model's prior boundaries. Panel ranges cover the extended base
rectangle, including parts removed by the joint cuts.

The old bounds are taken from `verified_bundle.prior_low` / `prior_high` in
the original FLAMINGO comparison metadata and checked against that comparison's
`original_prior_low` / `original_prior_high`. These are the actual stored support
bounds of the verified nine-parameter SBI model, not rounded nominal limits or
extrema of the new design. The numeric CSV retains their full precision.

## Physical interpretation

The parameter order is
`P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta, alpha_z_P0, alpha_z_xc, alpha_z_beta`.
For A = P0, xc, or beta, the evolution is

```
A(M,z) = A_0 (M200c / 1e14 Msun)^alpha_m (1+z)^alpha_z
P_th / P200 = P0 q^(-0.3) (1+q)^(-beta),  q = r / (xc R200c)
P_e = 0.5176 P_th
```

Masses are physical M200c; the amplitude pivot is 1e14 Msun at z=0. The beta
parameter is the raw Battaglia exponent, with asymptotic outer slope
`-(beta + 0.3)`. P0 sets the pressure normalization, xc the radial scale, and
beta the radial decline. Their mass and redshift exponents alter which halo
populations contribute most strongly to the tSZ power spectrum.

The base prior uses log-uniform P0 and xc and uniform remaining parameters,
conditioned on the frozen support restrictions. These include beta(M,z) in
[2.8, 50] across the interpolation domain; a relative xc/beta range [0.4, 8];
Y200/Battaglia12 in [0.003, 30] on the prescribed 9-by-9 catalogue-domain grid;
and the existing central-column tail check. Therefore rectangular parameter
limits alone do not specify the prior. The accepted samples illustrate its
induced correlations; their extrema are not exact boundaries of the support.

The new coloured markers come from
`flamingo_cosmology_refit_20260915/results/comparison_summary.json`, using
`corrected_parameters` for L1_m9, fgas-8sigma, and Mstar-1sigma. These are
selected effective fits to clean tSZ spectra including the approximate
halo-model correction for cosmological abundance, growth and geometry. They
are not direct halo-pressure measurements or posterior confidence constraints;
no confidence intervals are inferred from the spread between feedback variants.
Noise was not used to determine these points.

All three selected fits pass the current joint prior. The low-gas and low-stellar
fits reach beta(M,z) very close to the engineering ceiling of 50. This is a
joint, evolved-profile restriction, not a bound of 50 on the pivot beta_0.
The fiducial and low-stellar optimization candidates reached their evaluation
limit; only the selected low-gas optimizer reported convergence. The plotted
values remain the previously validated best checked candidates, not new refits.

## Reproduce or adjust

Dependencies: Python, NumPy, SciPy and Matplotlib. No simulation maps, SBI model
loading, new sampling, or additional FLAMINGO downloads are required.

From the repository root, using the local verified archives:

```bash
python SBI_analysis/flamingo_prior_comparison/plot_prior_comparison.py
```

This writes to the separate local `flamingo_prior_comparison/plots/` directory.
For different archived results, use `--production-root`, `--old-metadata`,
`--old-comparison`, `--fit-summary`, and `--output`. Use `--bins` and `--dpi`
to adjust the display. Labels and colours are constants at the top of the
script. A later larger design can be plotted without changing the parameter
definitions; the sample count is read from its manifest.

The cluster run used:

```bash
/home/anaconda3/bin/python3 \
  /lustre/work/kristero10/flamingo_prior_comparison_20260915/code/plot_prior_comparison.py \
  --production-root /lustre/work/kristero10/halfdome_flamingo_extended_8192_20260915 \
  --old-metadata /lustre/work/kristero10/flamingo_tsz_comparison_20260914/preflight/metadata_manifest.json \
  --old-comparison /lustre/work/kristero10/flamingo_tsz_comparison_20260914/comparison/comparison_summary.json \
  --fit-summary /lustre/work/kristero10/flamingo_cosmology_refit_20260915/results/comparison_summary.json \
  --output /lustre/work/kristero10/flamingo_prior_comparison_20260915/plots
```

The script verifies frozen production code/design checksums, checks all accepted
rows and the three fit vectors against the unchanged support implementation,
checks the old bounds against two saved records, verifies histogram mass
normalization, and hashes all input files again after rendering. It prevents
outputs inside its source-data roots and suppresses bytecode writes when
importing the frozen support implementation.
