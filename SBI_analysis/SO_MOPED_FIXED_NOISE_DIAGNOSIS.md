# MOPED zero-acceptance diagnosis

## Completed local experiment

The original observation used noise seed 3000001. A separate full-sky
Battaglia12 simulation was run using noise seed 12345. Both used mask seed
12345, the same baseline-deproj0 N_ell curve, nside 4096, 2 arcmin beam,
catalogue, cosmology and nine pressure parameters. The new Julia simulation
completed in 313.2 seconds with 16 threads.

The original observation, trained estimator and compression bundle were not
overwritten. Both observations use the same signed 40-bin D_ell construction
and the saved training-only asinh, standardization and MOPED transformation.
The comparison checks the simulation source hashes and physical settings.

| Diagnostic | Independent seed 3000001 | Fixed seed 12345 |
| --- | ---: | ---: |
| Prior acceptance, 20,000 proposals | 0 / 20,000 | 19,891 / 20,000 (99.455%) |
| Largest absolute standardized MOPED coordinate | 181.034 | 0.617 |
| Largest residual offset / training residual scatter | 106.06 | 0.361 |

These residual offsets are descriptive. The reference scatter spans changing
pressure parameters; it is not a repeated-noise covariance and these values
must not be interpreted as Gaussian detection significances.

The fixed-noise observation yielded 10,000 prior-restricted posterior samples
from 10,240 proposals (99.512% raw acceptance). The recovered posterior means
are P0 = 18.557 +/- 1.057 and beta = 4.312 +/- 0.224 (posterior standard
deviations), compared with truth 18.1 and 4.35. All nine mean-minus-truth
offsets are below 0.44 posterior standard deviations. This is one conditional
spot check, not evidence of independent-noise calibration.

## What caused the failure

The legacy tSZ_visuals/run_sobol32768_full_maps_slurm.sbatch and
tSZ_visuals/run_halfdome_fullsky_so_noise_cluster.pbs pass the same seed to
each parameter row. In run_halfdome_fullsky_so_noise.jl, the noise seed
defaults to that seed. Each split uses a deterministic offset for product,
deprojection and split index, but not for the parameter row. Consequently
the noise maps are reused.

For the two masked noisy maps, the cross spectrum is

    C_cross = C_ss + C_s,n2 + C_n1,s + C_n1,n2 .

With fixed noise maps, the noise-noise term is repeated and the signal-noise
terms change coherently with the pressure-dependent signal. Varying theta
does not provide independent noise realizations. In 10,000 aligned
training-reference rows, noisy-minus-clean residuals show a strong shared
pattern. The new fixed-seed residual reproduces that pattern.

The existing MOPED transform was fitted using local residual scatter after
removing a fitted parameter-dependent residual mean. Directions with little
remaining training variation receive large weights. Independent noise excites
those directions, producing extreme compressed coordinates even when the
uncompressed profiles appear similar.

The separate 64-seed Battaglia12 ensemble also produces extreme compressed
coordinates. Its high-ell scatter is up to 71 times the varying-theta
training residual scatter. Thus the original seed was not merely one unlucky
draw. Those 64 profiles are useful for a stochastic noise model; they do not
make the already-trained fixed-noise NPE noise-marginalized.

## What was ruled out for this experiment

- Saved observation and context match direct recomputation exactly.
- Paired clean/noisy dataset theta, Sobol row labels and binning align.
- The local density estimator reproduces 32 saved cluster log probabilities.
- Both observations use the same simulation source hashes and physical
  settings, except noise seed. Thread counts and runtime paths differ.
- The fixed-seed test succeeds with the original prior, transform and model.

These checks strongly identify a noise-realization distribution mismatch as
the cause of this zero-acceptance failure. They do not certify every historical
run or every observation. Historical mask maps have not been compared pixel
by pixel; matching seeds alone is not an exact cross-version RNG guarantee.

## Interpretation and next steps

The earlier notebook assumption that independently selecting a noise seed was
a supported validation of this model was incorrect. The notebook now warns
about this limitation and stops early on zero accepted pilot proposals.

Use the fixed-noise result only to test the trained conditional mapping.
Its finite contour widths also reflect learned approximation, training
resolution and parameter degeneracies; they are not established SO
independent-noise measurement uncertainties.

Do not compare these widths as equivalent to a Fisher covariance that
marginalizes independent noise realizations. A future physical comparison
needs stochastic noise augmentation or new simulations across theta,
training-only refitting of compression, NPE retraining, independent-noise
held-out tests and a Fisher calculation using the same observational noise
model. Merely changing the observation seed, rescaling the context, clipping
samples, widening the prior or switching to MCMC does not accomplish this.

The newer two-parameter generator explicitly varies the noise seed by row;
this diagnosis should not be applied to it without checking its own data.

## Files and reproduction

Open moped_fixed_noise_diagnostic.ipynb with the HalfDome (clean) kernel.
Its default cells only read completed diagnostics, samples and figures.
Optional switches reproduce the fixed-noise simulation and sampling.

Results are under:

    SBI_analysis/outputs/moped_local_profiles/0669be4fb61ca0d5/fixed_noise_diagnostic/

raw/ holds clean and noisy spectra; observation.npz holds binned data
and compressed context; posterior_samples.npy holds the 10,000 draws.
diagnostics/ contains the comparison PNG/PDF, conditional corner PNG/PDF,
parameter metrics CSV, pilot diagnostics, provenance and sampling summary.

To recompute the diagnostics and conditional posterior from saved spectra,
run from the repo root in a compatible HalfDome environment:

~~~bash
python SBI_analysis/diagnose_so_moped_observation.py \
  --observation-dir SBI_analysis/outputs/moped_local_profiles/0669be4fb61ca0d5/fixed_noise_diagnostic \
  --compare-observation-dir SBI_analysis/outputs/moped_local_profiles/0669be4fb61ca0d5 \
  --fixed-noise-posterior
~~~

This does not regenerate maps or retrain. Raw proposals outside the prior are
discarded, never clipped or presented as posterior samples.

