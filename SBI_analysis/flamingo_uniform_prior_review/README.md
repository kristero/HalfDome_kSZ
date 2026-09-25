# Why the accepted prior is not uniform

The user selected **uniform in physical values for all nine parameters** for a
separate revised prior. This directory diagnoses the existing design; it does
not change the running 8k campaign or install a replacement prior.

## What caused the shapes

The current campaign starts with log-uniform P0 and xc and linear-uniform
remaining parameters, then rejects vectors that fail coupled restrictions.
Consequently its density is proportional to

```
p(theta) = constant * 1 / (P0 * xc) * indicator(all joint restrictions pass).
```

For any one parameter, its marginal density is its original density multiplied
by the probability that the other eight parameters pass the restrictions at
that value. Extreme evolution exponents have fewer compatible combinations.
This produces peaked, skewed marginals without using a Gaussian distribution.

The physics enters through A(M,z) = A0 (M/1e14 Msun)^alpha_m (1+z)^alpha_z,
for A = P0, xc, beta. Limits on evolved beta, xc/beta, and integrated pressure
couple the amplitudes to their mass and redshift exponents. These restrictions
are additional model/engineering choices, not measured FLAMINGO posteriors or
proof that every rejected model would fail numerically.

## Measured coverage

The cluster diagnostic used 262,144 scrambled Sobol probes in each of the old
linear-uniform box and the extended linear-uniform box (seed 91015).

- Only 64.3169% of old-box probes pass the new joint cuts. Sequential exclusions
  are 11.4986% by the beta lower limit, another 23.3868% by the relative-size
  bounds, and another 0.7977% by the finite-Y200 bounds. Fractions are relative
  to the original probe count; changing the cut order changes the attribution.
- No current 8k design row lies inside the entire old nine-dimensional box.
  Many rows lie within individual old parameter ranges; these statements are
  different. The exact per-parameter counts are saved in `uniformity_audit.json`.
- With all nine proposal coordinates linear-uniform over the extended rectangle,
  only 2.1755% pass the existing cuts. Their accepted marginals remain nonuniform.
- Even without rejection, the old box occupies only 2.2094e-6 of the volume of
  the full extended linear rectangle. A random 8,192-row draw would have an
  expected 0.0181 rows inside that old box. A globally uniform extended prior
  therefore cannot also guarantee dense old-box sampling at this dataset size.

The diagnostic plot uses linear axes and equal-width bins for every parameter.
It compares the actual current design, linear-uniform proposals after the same
cuts, and the desired flat marginal. All curves are probability mass per bin;
none are Gaussian fits or posterior constraints.

## What a uniform replacement would require

An independent uniform prior has density `product_i Uniform(theta_i; low_i, high_i)`.
Choose and validate the rectangular bounds first, then sample by an affine
transform of Sobol coordinates without conditioning away parts of the box.
Simply changing the proposal measure and retaining the existing rejection step
does not produce this prior.

The complete exploratory rectangle cannot be used unchanged with the current
projector: its exact extreme beta(M,z) values are 0.1812 and 1125.50 across the
interpolation domain. Some combinations violate the projector's beta > 0.7
assertion. For P ~ r^(-beta-0.3), beta > 0.7 also ensures convergence of the
infinite line-of-sight column. The retained finite endpoint is 1e5 R200c.

Keeping every present gate while making every full-range marginal uniform is
also impossible. For example the beta-range gate implies

```
abs(alpha_m_beta) <= log(50 / 2.8) / (3.7 * log(10)) = 0.33832756,
```

whereas the exploratory upper limit is 0.4. No balancing or larger sample can
fill that forbidden interval without changing the support.

The smallest coordinate-wise box containing the entire old prior and all three
new FLAMINGO fits has exact beta extrema 2.1987 and 65.9652. It therefore also
violates the current [2.8, 50] gate, even without adding any margin. This box is
recorded only as a diagnostic in the JSON; it is **not** a validated replacement
or a statement of uncertainty on the fitted parameters.

Before defining the revised bounds, the pending scientific choice is whether to
retain the entire old prior plus the FLAMINGO fits and reassess the added cuts,
or preserve all current cuts and accept correlated, only approximately uniform
marginals on a feasible range. Preserving old-box coverage and an independent
uniform extension requires the former. Map/interpolator checks are then needed
on the new extremes; the diagnostic calculations here are not full-map validation.

## Files and reproduction

Added source: `audit_uniformity.py`. Cluster-generated outputs, copied locally
under `cluster_results/`, are `uniformity_diagnosis.png`,
`uniformity_diagnosis.pdf`, and `uniformity_audit.json`.

The script verifies the frozen design/support-code hashes, checks that its cut
accounting agrees exactly with the production `contains()` function, and verifies
that all five input files are unchanged after the audit. No simulation rows are
generated or modified.

```bash
/home/anaconda3/bin/python3 \
  /lustre/work/kristero10/flamingo_uniform_prior_review_20260915/code/audit_uniformity.py \
  --production-root /lustre/work/kristero10/halfdome_flamingo_extended_8192_20260915 \
  --old-metadata /lustre/work/kristero10/flamingo_tsz_comparison_20260914/preflight/metadata_manifest.json \
  --fit-summary /lustre/work/kristero10/flamingo_cosmology_refit_20260915/results/comparison_summary.json \
  --output /lustre/work/kristero10/flamingo_uniform_prior_review_20260915/results
```

Use the corresponding local archived paths for a local rerun. `--proposal-power`
controls the number of diagnostic probes (2**power); it does not control or
submit production simulations.
