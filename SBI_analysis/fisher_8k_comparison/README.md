# Battaglia12 Fisher forecast versus the independent-noise 8k SBI

This is the matched comparison requested on 20 September 2026. It uses the
completed `halfdome_flamingo_linear_8192_20260915` dataset and the saved MAF
estimators in `sbi_linear_8k_20260918/analysis`, at N=7349. The other 843 rows
were held out. The earlier 524k fixed-noise estimators are not used.

**Completed and verified on 21 September 2026:** all 41 calibration tasks and
the dependent analysis finished. All 41 final artifacts, task checksums and
frozen analysis-source hashes passed verification after download. The figures
were inspected. Read the [results report](../outputs/battaglia12_fisher_8k_20260920/review_20260921/RESULTS.md)
for the nine-parameter constraints and [forecast corner](../outputs/battaglia12_fisher_8k_20260920/comparison/battaglia12_forecast_all.pdf).

The saved MOPED transform retains 71.4–96.6% of the local Fisher information in
the six-mode generalized-eigenvalue comparison; its Fisher-plus-prior marginal
widths differ from the full-bin result by at most 2.33%. PCA retains much less
information in several combinations. MOPED SBI widths are 1.90 times the full
Fisher width for xc and 1.65 times for beta. Other marginal widths can be
smaller; this is not evidence of beating a Fisher information bound.

There is also a displacement, not merely a width difference. At the exact
Battaglia12 mean spectrum, the MOPED SBI central 95% intervals are
xc=[0.5231, 2.9166] and beta=[4.8449, 14.3633], excluding the fiducial values
0.497 and 4.35. The shift appears across the SBI methods. Nonlinear degeneracy
and prior-volume effects, parameter-dependent noise, and finite-training
errors can all contribute. These plots do not establish which dominates or
validate SBI coverage. Compression loss alone does not explain the difference.

The exact-joint-prior 16-versus-32 covariance check changes forecast widths by
at most 2.61%. OAS correlation shrinkage is 0.9534, so this nested pilot check
does not establish covariance convergence. The derivative step check is
0.02614% at worst. The separate noisy observation has chi-square 39.95 in the
40-bin frozen covariance, without fitting parameters.

All nine computed Fisher eigenvalues are positive, from 3.58e8 to 0.00673 in
prior-width coordinates (condition number about 5.32e10). The saved
`fisher_rank=6` field counts eigenvalues above `1e-8 * lambda_max`; it is a
chosen numerical cutoff, not an assertion of three exactly zero eigenvalues.
All nine directions were included in the posterior integration.

The new cluster root is
`/lustre/work/kristero10/battaglia12_fisher_8k_20260920`.
The local delivery root is `SBI_analysis/outputs/battaglia12_fisher_8k_20260920/`.
Read `run_status.json` there for the latest saved state. A submission receipt
or running job does not mean that the scientific constraints are complete.

## Physical and prior matching

The 8k dataset has independent noise in every row and split: all 16384 seeds
and noise-map hashes were verified distinct in the original run. New
calibration seeds are checked against every training seed and the old
observation seeds. The parameter order is
`P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta, alpha_z_P0, alpha_z_xc,
alpha_z_beta`, with Battaglia12 values
`[18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]`.

Each pressure coefficient evolves as
`q(M,z)=A_q (M200c/1e14 Msun)^alpha_m,q (1+z)^alpha_z,q`. The pressure is
`Pth/P200=P0 (r/(xc R200c))^-0.3 [1+r/(xc R200c)]^-beta`, and the frozen
electron-pressure conversion is retained. This run preserves the **historical
projected-aperture/long-LOS painter used by the 8k dataset**. It does not use the
separate spherical-truncation update or change the HEALPix sampling resolution.

The base parameter ranges are `[1,60]`, `[0.1,4]`, `[2.8,16]`, `[-0.2,1.5]`,
`[-0.6,0.4]`, `[-0.2,0.4]`, `[-4.5,0.5]`, `[-1.5,2]`, `[-0.5,1.5]`.
The prior is physical-uniform on the **joint accepted region**, including all
frozen evolved-beta, size, finite-Y200 and LOS-tail cuts. It is not a box-uniform
prior. Both Fisher and SBI samples use the exact `JointPrior.contains` from the
frozen dataset code. Prior marginals are shown as grey dotted lines on corners.

The beam is 2 arcmin, NSIDE is 4096, and the same fsky=0.4 apodized mask and
SO baseline deproj0 noise table are used. The table is N_ell per split, with no
extra factor two or noise beam. The observable is the same signed, mode-count
weighted masked cross D_ell in 40 bins, starting at ell=80–279 and ending at
7880–7979. No clipping, logarithm, extra beam or fsky division is introduced.
Only binned training spectra were retained in this dataset; the 524k unbinned
MOPED estimator belongs to a different noise model and prior.

## Why fresh calibration is required

The original Fisher fiducial differs from the 8k Battaglia12 clean spectrum by
up to **10.6376%** in a bin. That derivative bundle is not reused. The separate
historical-grid, stable-LOS Battaglia12 control agrees with the 8k observation's
clean spectrum to 8.9e-16 relative, confirming which frozen painter is needed.
The old 64-noise archive also has shared adjacent split seeds; its label alone
does not establish 64 independent covariance realizations.

`calibrate_maps.jl` calls the frozen 8k painting function and stable-LOS override.
It validates the runtime/source, mask hash and noise-table hash against the
archived operator. Each covariance batch paints one signal and reuses it for
eight independent split pairs. Thus each cross spectrum includes signal-noise
as well as noise-noise fluctuations. Noise-only spectra would be insufficient.

The small campaign has 32 covariance pairs plus one separate observation pair.
There are 36 parameter variations: positive/negative steps at h and 2h for
each of nine parameters. Here h is 0.002 times the **broad 8k prior range**;
the steps are recorded explicitly. Together with the fiducial, these are the
37 unique mean spectra for Richardson derivatives `(4 D_h-D_2h)/3`.
Repeated fiducial maps from the five noise/observation batches must agree.
Their binned spectrum must also reproduce the archived 8k clean Battaglia12
observation to 1e-7 relative, allowing for floating-point thread-order changes.
The derivative check fails if the smaller-step discrepancy exceeds 1% in the
noise-weighted norm. No noisy spectrum is differentiated.

Five worker lanes were scheduled. The first 26-CPU mini request could not fit;
the pilot was moved to an available two-CPU mini2 allocation (job 598440).
The remaining workers use the default mini/26 CPUs (598441–598444).
Analysis job 598449 depends on successful completion of all five. Its held
predecessor 598445 was replaced to include the final source-verification and
resume guards. The earlier
queued requests 598437 and 598438 were replaced before they ran. Other users'
jobs and other campaigns under this account were left untouched.

## Fisher calculation and SBI overlays

The covariance C is estimated once at Battaglia12 using OAS correlation
shrinkage while preserving empirical bin variances. For any linearized
compression A, use `C_y=A.T C A`, `J_y=A.T J`, and
`F_y=J_y.T solve(C_y,J_y)`. Do not fit a new OAS estimator after compression.
PCA's across-prior training covariance chooses axes; it is not C.

The saved 8k PCA and MOPED transforms contain asinh preprocessing. Their Fisher
comparison uses the derivative of the **actual saved transform** at the
fiducial, including all normalizations. This covariance propagation is a local
delta-method approximation; its error is measured on the new noise ensemble.
The older MOPED weights came from a local regression covariance and need not
preserve the new fiducial Fisher information exactly. Their information loss
is measured, not assumed zero. A freshly optimal nine-coordinate MOPED basis
is also saved as an algebra check; it is not substituted for the trained model.

Fisher constraints integrate the local Gaussian likelihood times the actual
joint prior, keeping all nine coordinates and weak modes. A Gaussian importance
proposal is only a proposal: its penalty is analytically cancelled. Two
independent integrations must have ESS>5000 and agree in means/widths to the
recorded tolerance. No eigenvalue floor manufactures finite marginal errors.
The data-only Fisher matrix and its resolved modes are saved separately.

Two comparisons were rendered:

1. **Mean-spectrum forecast:** Fisher and SBI condition on the same noiseless
   Battaglia12 mean spectrum while retaining the noisy likelihood/training.
   This is an expected-data (Asimov) forecast, not an average over posteriors
   from many noise realizations.
2. **Observed constraints:** both condition on the same newly generated,
   independent Battaglia12 noisy observation, excluded from covariance fitting.

Each SBI method supplies 10000 accepted draws under the exact joint prior.
The code also generates PCA-versus-its-projected-Fisher and
MOPED-versus-its-projected-Fisher forecast corners, so compression loss can be
separated from the full-data forecast comparison.

Completed outputs in `comparison/`: `battaglia12_forecast_all.{png,pdf}`,
`battaglia12_observed_all.{png,pdf}`, the two method-specific forecast corners,
forecast/observed interval plots, `constraints.csv`, `constraint_summary.json`,
`calibration.npz`, per-method likelihood matrices and posterior samples,
`compression_audit.json`, and a final checksummed `complete.json`.
Thick/thin interval bars represent 68%/95% marginal intervals.

## Interpretation limits

This matches the dataset, physical operator, bins, prior and conditioning data.
Fisher still approximates the mean linearly and freezes covariance at
Battaglia12. It omits `0.5 Tr(C^-1 C_,i C^-1 C_,j)`, changing-sky variance and
model discrepancy. A broad-prior nonlinear SBI posterior need not match the
local Fisher contour even when these inputs agree. The exact joint prior can
dominate weak directions. Saved 8k models are reused, not retrained or newly
coverage-calibrated here.

Thirty-two covariance realizations are a pilot. Inspect the shrinkage and
16-versus-32 sensitivity before claiming covariance convergence. The prepared
figures are forecast comparisons within the frozen 8k model, not a certification
of its numerical painter accuracy or a complete SO observing forecast.

## Files and verification

All new source files are in this folder; no old dataset or trained model is
modified: `calibration.py`, `calibrate_maps.jl`, `analyze.py`, `test_analysis.py`,
`submit.py`, `run_calibration.pbs`, `run_analysis.pbs`, `watch_results.py`,
`review_results.py`, and this README. The completion turn added
`review_results.py`, updated this README, and added a single-process lock plus
completion timestamps to `watch_results.py`. The frozen cluster analysis and
completed scientific exports were preserved.

Three numerical tests cover central-derivative assembly and observation
exclusion, an analytic correlated-support prior with unconstrained modes,
and Fisher invariance under the invertible full-bin asinh tangent. The saved
MAF models also loaded and produced correctly shaped samples locally.
These software checks do not substitute for completion of the full-map run.

The first real cluster task (004, the held-out observation) finished in
1276.3 seconds on two CPUs. Its clean spectrum agrees with the archived 8k
Battaglia12 spectrum to 6.66e-16 maximum relative error. A 4096-proposal check
at each context gave 57.2–58.5% joint-prior acceptance for the mean-spectrum
forecast and 50.0–54.2% for the noisy observation across the three saved MAFs.
These checks are recorded in the local delivery root's `pilot_validation.json`;
they validate input compatibility, not Fisher constraints or SBI coverage.
The three numerical tests passed locally and on the cluster. Full plotting
and checksummed export also ran locally with clearly labelled synthetic samples.
The completed calibration and scientific comparison have now also passed
artifact verification. `review_results.py` independently checks the exported
matrices and all posterior samples against the joint prior, compares 16/32
noise ensembles with two exact-prior importance integrations, and saves a
compact information/width plot and a full numerical report under
`review_20260921/`. This folder has its own completion checksum manifest.

Re-run tests with the HalfDome scientific Python environment:

```bash
python -m unittest discover -s SBI_analysis/fisher_8k_comparison -p test_analysis.py -v
```

The local watcher checks this campaign's receipt once per minute, records PBS
status, and downloads a checksummed archive after the final completion marker.
It verifies every delivered artifact. PBS job history is disabled on this
cluster, so finished workers are verified from their task completion records.
It stops on missing jobs without completed outputs or after 48 hours; it never
cancels, resubmits or changes jobs. Restart it if the queue outlasts that window:

```bash
python SBI_analysis/fisher_8k_comparison/watch_results.py \
  --root /lustre/work/kristero10/battaglia12_fisher_8k_20260920 \
  --output SBI_analysis/outputs/battaglia12_fisher_8k_20260920
```
