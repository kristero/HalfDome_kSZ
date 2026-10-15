# Truth versus posterior mean: original Merlin, band50, and the 580 cut

Status: **inference and figure production completed on idark, 2026-10-06.**
Job 600740 evaluated the frozen models; job 600741 regenerated corner typography
from the saved draws. Both ran on ansys1 in mini2 with 26 CPUs and exited 0.
Earlier models and scientific outputs remain read-only inputs. The new 44-page
PDF combines their seed/mask evidence with the corrected truth-mean analysis;
the previous PDF is preserved. See `VERIFIED.json` for final delivery checks.

## Completed results

Primary unbounded Pearson correlations across the same 1,500 reserved cases:

| Parameter | Original | Band50 | Band50 + ell>=580 |
| --- | ---: | ---: | ---: |
| P0 | 0.7327 | 0.5373 | 0.5075 |
| xc | 0.7896 | 0.6942 | 0.6448 |
| beta | 0.6964 | 0.4780 | 0.3897 |

All nine correlations decline with band50 in both bounded and unbounded
coordinates. The full-prior geometric mean width increase is 45.64% for
band50 unbounded, 46.41% for band50 bounded, and 56.51% for the 580 unbounded
control. These differ from the earlier 18-29% fixed-Battaglia seed/mask costs:
the observations and pressure parameters over which widths are averaged differ.

Full-range band50 remains preferable to the 580 cut in this comparison, but
the goal of retaining original conditional precision while improving sky
agreement is not achieved. These empirical costs include changed preprocessing
and model retraining; they do not isolate irreducible averaging loss.

Band50 unbounded marginal 68% coverage spans 63.2-66.8%; bounded spans 68.3-73.0%.
Paired Pearson changes, 95% coverage, slopes, signed biases, width costs and
Monte Carlo errors are in `data/truth_mean_metrics.csv` and
`data/paired_pearson_changes.csv`. Every banded test feature reconstructs
exactly. The four previously evaluated truth-density averages agree within
7.5e-8, providing a separate check on rows, models and physical Jacobians.

## What band50 means

For integer multipoles ell=80,...,7979, each contiguous band contains 50 values:
80-129, 130-179, ..., 7930-7979. The saved linear operator computes

    band[b] = sum((2*ell+1)*D[ell]) / sum(2*ell+1)

within each band. The 2*ell+1 factor counts harmonic modes on an ideal full sky;
under the actual mask it is the chosen averaging prescription, not a proof of
inverse-covariance optimality. The saved covariance is transformed consistently
as C_band = A C A.T. Mask-induced covariance remains in this transformation.

Band50 reduces 7,900 values to 158. It keeps the complete ell range but loses
within-band information. It is not Gaussian smoothing with sigma_ell=50, and
is not an average of 50 simulations. New MOPED and Merlin recipes shared this
input for tractable matched comparisons. The original Merlin uses its original
7,900-multipole noise whitening/PCA pipeline, not band50. No bandwidth sweep
was done: width 50 is a practical choice, not an optimized physical scale.

Widths 25, 50 and 100 give 316, 158 and 79 bandpowers, respectively, for this
ell range. The illustration applies these operators to an actual saved noisy
seed100/mask12345 spectrum. It does not claim inference performance for 25 or
100. Comparing their constraints requires fitting the corresponding transforms
and retraining/validating the density estimators. Existing saved training
spectra can be reused for that bandwidth comparison; no repainting is needed
solely because the band width changes. Different input dimensions cannot be
passed directly to the frozen original Merlin model.

## Physical interpretation and simulation seeds

tSZ measures the integral of electron pressure along a line of sight:

    y = sigma_T / (m_e*c**2) * integral P_e dl_physical

Here sigma_T is the Thomson cross-section, m_e is the electron mass, c is the
speed of light, P_e is physical electron pressure, and dl_physical is physical
path length. P0, xc and beta describe pressure normalization, radial scale and
outer-profile behavior. Six alpha parameters describe their mass and redshift
scalings. All nine parameter definitions, priors and order are preserved.

Changing halo phases changes the realized angular spectrum even at identical
pressure parameters. Changing the mask selects a different portion of that
sky and couples harmonic modes. A model trained on one fixed catalogue/mask
can learn fine spectrum patterns specific to that sky. Banding discards some
fine patterns and can reduce sensitivity to these nuisance changes. It also
discards potentially useful pressure information. This is an empirical
robustness tradeoff, not a general cure for misspecification.

The same fixed band operator can be applied at every parameter draw and to
every catalogue. A fresh simulation seed is not required for each parameter
iteration. To train a posterior that marginalizes sky/mask variation, those
nuisance variables must instead be varied independently of theta (for example,
draw from the existing catalogue/mask pool across training rows). Reusing a
finite pool is possible; the relationship must be checked on entire held-out
catalogues/masks and over different pressure parameters. The current prior
test remains conditional on halo100/mask12345. The 10x10 sky test is at a single
Battaglia truth, so it does not establish sky-marginalized recovery over the
whole pressure prior.

## The lower cut is also banded

The saved Merlin ell>=580 model retains bands 580-629 through 7930-7979:
148 of 158 bands, or 7,400 of 7,900 original multipoles. It removes 6.33% of
the raw multipole entries; this is not a 6.33% Fisher-information claim. This
removes the largest angular scales, not the small angular scales. Roughly,
angular features scale as pi/ell, so ell=580 corresponds to about 19 arcmin.

The old unbounded held-out sky results are:

| Arm | Band50 width increase | Band50+580 width increase | Band50 / +580 truth RMS | Band50 / +580 spread RMS |
| --- | ---: | ---: | ---: | ---: |
| Simulation | 24.67% | 28.74% | 1.169 / 1.148 | 0.04996 / 0.04575 |
| Mask | 17.93% | 22.47% | 1.162 / 1.167 | 0.09794 / 0.09048 |
| Combined | 22.59% | 26.96% | 1.162 / 1.166 | 0.09735 / 0.09906 |

Width increases are geometric means over parameters and held-out sky cases,
relative to the original model on those same cases. The original seed100
baseline half-width normalizes both RMS diagnostics, so smaller is better.
Simulation, mask and combined subsets contain 4, 5 and 20 cases. The cut gives
a modest simulation benefit but no demonstrated combined improvement over
band50 alone. The arbitrary 25% threshold should not veto it. Its worst
individual width increase is 68.1%, 60.0%, and 80.7% across these three arms;
the mean width cost does not bound each parameter's cost. These are saved
screening results for the 580 control, not newly rerun high-sampling results.

## New reserved-test experiment

The intended correlation is r(theta_true, E[theta|x]) across distinct test
parameter points, separately for each of nine physical parameters. This
differs from correlations among posterior parameters in the earlier PDF.
The fixed-Battaglia grid has constant truth and cannot measure this Pearson r.

All methods evaluate the same 1,500 odd held-row positions. The earlier method
screen used the even positions. This is a diagnostic follow-up; it does not
re-rank models using the reserved test cases.

The fifteen consumed frozen network files and previously verified original
features/loader inputs have their earlier SHA-256 values recorded in
`evidence/frozen_input_hashes.json`. The job checks them before and after
inference, then records additional newly consumed input hashes. A changed
known frozen input aborts the run rather than silently changing the comparison.
Every tested band50/580 feature is also reconstructed from its saved noisy
bandpowers and frozen preprocessing. Original-feature provenance retains the
earlier direct-spectrum reconstruction audit; the job verifies that audit's
saved design and bandpower hashes before sampling.

Evaluate frozen ensembles, with 9,000 physical posterior draws per case:

1. Original Merlin, unbounded (primary).
2. Band50 Merlin, unbounded (primary).
3. Original Merlin, bounded.
4. Band50 Merlin, bounded.
5. Band50+ell>=580 Merlin, unbounded (additional diagnostic).

The 580 bounded companion was not trained in the earlier experiment, so this
evaluation does not invent one. No network training is needed for this job.
Linear/logit evaluation uses fresh interpreters because frozen parameter
transforms are patched at import. No unbounded samples are clipped.

Report Pearson r with paired case-resampling intervals, prior-normalized RMSE
and signed bias, regression slope/intercept, marginal 68/95% coverage, widths,
truth log density and out-of-prior probability. Sampling standard errors and
split-sample Pearson differences quantify finite-draw precision. Large r can
coexist with systematic bias; posterior means can legitimately shrink toward
a prior for poorly constrained directions. Coverage and density are needed
alongside regression metrics. Test-case bootstrap intervals describe this
parameter/noise cohort, not cosmic variance or training uncertainty.

Select three corner cases transparently: two seeded-random reserved cases
(RNG seed 20261006), excluding the stress case; plus the case with the largest
original-unbounded prior-normalized mean error. Each gets 45,000 draws from
each original/band50 bounded/unbounded model. Export six full nine-parameter
corners and three compact four-parameter overlays. Contours enclose 68% and
95% of all draws; displayed axes cover the union of central 99.6% ranges,
padded to include truth. One histogram-bin smoothing is for visualization
only and does not modify observations or inference.

## Run when idark is reachable

Copy this directory to:

    /lustre/work/kristero10/tsz_truth_mean_validation_20261006

Then, on the login node:

    /home/kristero10/envs/sbi27/bin/python submit_idark.py

The scheduler helper submits at most one job per task, selecting an eligible free
26-CPU/16-GB node across mini, mini2 and mini_B groups. It records a ledger
and will not submit that task again on repeated invocation. It refuses if 20 of the
user's jobs are active. Wall time is two hours; inference/plotting occur on
the compute node. It does not pin the run to the mini queue. The separate
`--task publication` job only reformats stored corners; it never resamples
posteriors or changes inference. Two new jobs were submitted in this follow-up.

After completion, require INFERENCE_COMPLETE.json, PLOTS_COMPLETE.json, the
resource-log exit status and all recorded output hashes to agree. Download
the small metrics, plots and corner draws for final PDF authoring/review.
Full 1,500-row draw cubes are deliberately not saved: per-case moments,
quantiles, errors and the three useful corner draw sets are enough.

On the local WSL science environment, after downloading completed results:

    /home/cbllover/miniconda3/bin/python build_supplement.py

The PDF builder creates one combined report, retaining the previous contract,
seed/mask methods, offset diagnosis, future-MOPED discussion and references,
and replacing the earlier posterior-correlation section with truth-mean tests.
The builder refuses to run without verified completed plots. Render all
PDF pages with pdftoppm and visually inspect before delivery. The PDF skill's
artifact-operation marker must be run immediately before first PDF authoring.

Existing-only figures can be regenerated independently:

    python plot_truth_mean.py --existing-only

Numerical checks can be run without models:

    python diagnostic_metrics.py

These checks test formulas and unit invariance. Actual model execution,
source/held-row parity and scientific outputs have also been verified on
idark. `source_snapshot/` preserves the code version used by inference;
publication formatting and document authoring have their own recorded hashes.
