# Small Battaglia12 Fisher run from existing spectra

This pilot uses 16 noise realizations, all nine gNFW parameters, and the
collaborator package's 40 bins. It requires no new noise maps or halo painting:
both noisy cross-spectra and controlled clean finite differences already exist
on `idark`. It does not perform the pending independent-noise NPE training.

**Completed:** PBS job `597160.idark`, `mini`, host `ansys08`, four CPUs and
four application threads. The wrapper recorded exit code 0, starting at
2026-09-13 17:16:36 UTC and finishing at 17:17:13 UTC. The analysis itself took
33.2 seconds. The earlier plotting attempt is preserved separately; no prior
science products were overwritten.

A separate [524k MOPED overlay](SO_NINE_FISHER_MOPED524K_OVERLAY.md) now adds
the verified fixed-noise Battaglia12 posterior to a copy of this corner plot.
The different observation and noise model are labelled explicitly; the
original Fisher figure and numerical artifacts remain unchanged.

## Results of the 16-realization pilot

The 40-bin and nine-coordinate MOPED Fisher matrices agree to relative error
`1.9451e-16`. Their observed likelihood score vectors agree to `1.6322e-15`.
There are six eigenmodes above the declared relative threshold `1e-8` in
prior-width coordinates. All nine computed eigenvalues are positive; the
three below this threshold are weak modes, not three parameters fixed by hand.
The hard-prior likelihood calculation retains all nine parameters and modes.

The following sigma values are marginalized posterior standard deviations
under the **local Gaussian likelihood times the hard uniform prior**. They
are not unconstrained square roots of the diagonal of an inverted Fisher
matrix. The fiducial forecast uses zero residual; the observed posterior uses
the reserved independent root 20001.

| Parameter | Battaglia12 | Fiducial forecast sigma | Observed mean | Observed sigma |
|---|---:|---:|---:|---:|
| P0 | 18.1 | 5.02 | 15.02 | 4.92 |
| xc | 0.497 | 0.118 | 0.560 | 0.113 |
| beta | 4.35 | 0.490 | 4.56 | 0.464 |
| alpha_m_P0 | 0.154 | 0.0766 | 0.127 | 0.0749 |
| alpha_m_xc | -0.00865 | 0.0515 | 0.00916 | 0.0511 |
| alpha_m_beta | 0.0393 | 0.0333 | 0.0439 | 0.0331 |
| alpha_z_P0 | -0.758 | 0.314 | -0.750 | 0.311 |
| alpha_z_xc | 0.731 | 0.290 | 0.793 | 0.283 |
| alpha_z_beta | 0.415 | 0.177 | 0.456 | 0.174 |

The beta and several evolution-parameter marginals remain close to their
uniform-prior widths. Strong degeneracies therefore coexist with tight
constraints on some linear combinations of parameters.

- OAS shrinkage is **0.96949**, so the estimated correlation matrix is close
  to diagonal. This is strong regularization, not a measured absence of
  inter-bin correlations.
- Relative to the Richardson derivative, the largest noise-weighted derivative
  change is **0.0227%** for the small steps and **0.0907%** for the large steps.
  This checks the saved spectra's finite differences, not the new painter.
- Independent one-million-proposal integrations agree in widths within
  **0.675%** and in means within **0.0071 posterior sigma**; effective sample
  sizes exceed 130,000 in all four integrations.
- Going from 16 to 8 covariance rows changes the Gaussian prior-moment width
  diagnostic by at most **1.06%**, but individual bin standard-deviation ratios
  span **0.631--1.393**. The prior and degeneracies can hide substantial
  covariance uncertainty; this is not a covariance convergence result.
- The bootstrap panel shows resampling diagnostics, not calibrated confidence
  intervals on the true errors. Small-sample bootstrap variance bias and
  repeated shrinkage also affect its ratios.

The [nine-parameter contours](outputs/so9_fisher_noise16_20260913/results/battaglia12_fisher_9param_pilot.png),
[covariance diagnostics](outputs/so9_fisher_noise16_20260913/results/small_ensemble_covariance_diagnostics.png),
and [full interval table](outputs/so9_fisher_noise16_20260913/results/parameter_constraints.csv)
are available locally, with PDF copies of both figures. The downloaded
`complete.json` verifies 12 artifact hashes. All 200,000 stored samples were
checked for finite values and prior containment; split independence, the
observed MOPED score, source-code identity and both plots were also checked.
The local receipt is `outputs/so9_fisher_noise16_20260913/verification.json`.

## Inputs and independence

The source root is `/lustre/work/kristero10/adrian_fisher_baseline_deproj0`.
The run rebins all 37 clean spectra (fiducial plus 36 variations) and the saved
noisy spectra directly from Float64 unbinned C_ell. It does not reuse the old
40-bin arrays: their first bin was ell=80--279, whereas the collaborator's
first bin is ell=80--199, followed by 200--399, etc. Each bin averages
D_ell=ell(ell+1)C_ell/(2 pi), weighted by 2ell+1.

**The historical 64 outputs are not 64 independent noise realizations.**
The saved simulator's baseline/deproj0 split seeds are
`(noise_seed + 10101, noise_seed + 10102)`. Roots 20001 through 20064 therefore
contain only 65 unique split seeds. Adjacent roots reuse a split.

Root 20001 is the observation. The covariance uses roots 20003, 20005, ...,
20033: 16 pairs whose 32 splits are mutually disjoint and exclude both
observation splits. There are at most 31 covariance pairs under this selection
after reserving the observation. The script rejects an oversized request.

## Physics and numerical calculation

The parameter order and Battaglia12 fiducials are
`[P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta,
alpha_z_P0, alpha_z_xc, alpha_z_beta]` =
`[18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]`.
The first three specify the pressure normalization, scale, and outer-shape
parameter; the remaining six describe their mass and redshift evolution.
The radial exponents alpha=1 and gamma=-0.3 are fixed. Thermal SZ measures
integrated electron pressure, so these changes affect the angular y spectrum.

The two observed splits are s+n1 and s+n2. Their cross-spectrum contains
ss, s*n2, n1*s, and n1*n2. The saved cross-spectra retain both signal-noise
and noise-noise fluctuations; pure noise-only cross-spectra would miss the
signal-noise contribution. The beam is 2 arcmin, the mask has fsky=0.4 and
60 arcmin apodization, and SO baseline/deproj0 N_ell is used per split.
No additional beam or fsky correction is applied.

The mean derivative J uses central differences at the source manifest's 1%
and 2% steps and Richardson extrapolation. These steps use the old campaign's
prior widths; the posterior bounds instead come from the new collaborator
config. Step sizes are differentiation settings, not posterior bounds.

With only 16 realizations, the raw 40-bin covariance has rank at most 15.
OAS shrinks its correlation matrix toward the identity while preserving
unbiased measured bin variances. The script evaluates F=J^T C^-1 J and
samples the linear Gaussian likelihood times the exact hard uniform prior.
It never converts unresolved modes into finite data-only errors by flooring
eigenvalues. Two independent importance integrations must agree.

Nine optimal linear MOPED coordinates are constructed by whitening J and
using its left singular vectors. Their local Fisher matrix must agree with
the full 40-bin matrix. This identity assumes fixed C and a local mean model;
it is not a comparison with the previously trained fixed-noise MOPED model.

Outputs include all-nine-parameter contours, parameter intervals, J and C,
Fisher eigenmodes, MOPED weights, derivative-step checks, and covariance checks
using 8 versus 16 rows and 200 bootstrap resamples. The covariance sensitivity
plot uses a Gaussian prior-moment diagnostic (precision 12 in prior-width
coordinates); reported intervals use the actual hard-prior samples instead.

These remain preliminary constraints. Sixteen rows do not establish covariance
convergence, the likelihood is linearized, and the saved earlier painter has
not been demonstrated to match the collaborator's corrected painter. The sky,
halo catalogue, cosmology and mask are fixed; there is no changing-sky variance
or covariance-derivative Fisher term. The new 32k dataset and matched calibration
are still needed for the final Fisher/40-bin NPE/MOPED NPE comparison.

## Reproduction

The script requires NumPy, SciPy, scikit-learn, matplotlib and GetDist. Supply
the new package's `config.json`; use a fresh output directory.

```bash
python3 run_so_nine_fisher_small_ensemble.py \
  --fisher-root /lustre/work/kristero10/adrian_fisher_baseline_deproj0 \
  --config /path/to/so_32k_independent_noise/config.json \
  --output /path/to/fresh_results --n-noise 16
```

The PBS wrapper defaults to `mini`, 26 CPUs, 8 GB and 30 minutes. For this
small spectra-only run the queued 26-CPU job was replaced with a 4-CPU job
because no 26-CPU GroupA slot was available; all application threads were
also set to four. The submission receipt preserves both job requests.

The cluster run root is
`/lustre/work/kristero10/so9_fisher_noise16_20260913`.
It retains the staged code/config, submission receipt, logs and results.
`results/complete.json` is written only after calculation and plotting finish,
and records artifact checksums. Queue disappearance alone is not completion.
PBS history is disabled on this server, so `results.job_status.json` records
the actual wrapper exit code, job ID, hostname, thread count and UTC times.

Run the split-independence regression checks with:

```bash
python3 -m unittest test_so_nine_fisher_small_ensemble -v
```

See [the full matched-comparison runbook](SO_NINE_INDEPENDENT_FISHER.md) for
the pipeline that will use the collaborator's 32k simulations.
