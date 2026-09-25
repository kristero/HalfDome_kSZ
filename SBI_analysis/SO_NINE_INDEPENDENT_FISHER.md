# Matched nine-parameter Battaglia12 Fisher, 40-bin NPE and MOPED NPE

The code is prepared for the collaborator's **32,768 independent-noise
nine-parameter simulations**. No new production simulation or training job was
submitted. Numerical constraints using that new dataset remain pending.
A separate [small Fisher pilot](SO_NINE_FISHER_SMALL_ENSEMBLE.md) uses existing
cluster spectra and disjoint noise splits, with no new maps. Its earlier
painter and limited covariance are explicitly separate from this matched run.

## Inputs to retain from the collaborator

Keep these files together, preferably preserving their directory structure:

- `nine_param/prepared/dataset.npz` and its `complete.json`.
- The generation root's `experiment.json`.
- The exact `so_32k_independent_noise/` simulator package used to generate them,
  including its config, parameter designs, noise curves, Julia source and lockfile.
- Access to the same original HalfDome `lightcone_100.hdf5` for calibration maps.

The loader checks the completion checksum, all 32,768 rows, all nine parameter
columns, noise-split uniqueness, mask, beam, binning, source manifest, and prior
bounds. It rejects the earlier fixed-noise data. It does not infer priors from
sample extrema or substitute the slightly narrower bounds from old SBI runs.
The priors in the collaborator's **new saved dataset** are authoritative.

The new package's first bin is ell=80--199, then 200--399, etc. The old
Fisher/SBI files started with 80--279. Both have 40 bins, so matching dimension
alone is insufficient. Calibration and training use the exact new bin edges.

Catalogue identity must be preserved by the person supplying its path. The
available generation manifest records catalogue path, size and timestamp, not
a full catalogue checksum. Calibration checks the size and simulator hashes;
equal file sizes alone do not prove two catalogues are identical.

## What the comparison means physically

For x=r/R200c, the Battaglia pressure shape is

    P/P200 = P0 (x/xc)^gamma [1 + (x/xc)^alpha]^(-beta)
    q(M,z) = A_q [M200c/(10^14 Msun)]^(alpha_m_q) (1+z)^(alpha_z_q)

for q in {P0, xc, beta}. Alpha=1 and gamma=-0.3 remain fixed. The nine free
parameters and Battaglia12 fiducials, in the exact simulator order, are:

| Parameter | Fiducial |
|---|---:|
| P0 | 18.1 |
| xc | 0.497 |
| beta | 4.35 |
| alpha_m_P0 | 0.154 |
| alpha_m_xc | -0.00865 |
| alpha_m_beta | 0.0393 |
| alpha_z_P0 | -0.758 |
| alpha_z_xc | 0.731 |
| alpha_z_beta | 0.415 |

Thermal SZ y measures integrated electron pressure along the line of sight.
Changing pressure amplitude, scale and slope, or their mass/redshift evolution,
changes the angular y power spectrum. Integrating over halo mass and redshift
mixes these effects, so the nine parameters can be strongly degenerate.
See [Battaglia et al. (2012)](https://arxiv.org/abs/1109.3711).

The observable is signed, masked pseudo-D_ell, with the signal's existing
2 arcmin beam and 2ell+1 weighted bins. No second beam, fsky deconvolution,
logarithm of signed power or clipping is applied. The lightcone, cosmology and
apodized mask remain fixed. Independent SO noise splits change between rows.
The confirmed convention is the supplied N_ell **per split**, without a factor
of two. This is a conditional-on-sky forecast, not a changing-sky cosmological
variance forecast.

The linear Fisher likelihood is

    m(theta) = m_B12 + J (theta - theta_B12)
    F = J^T C^(-1) J

J uses central differences at 1% and 2% of each saved prior width and
Richardson extrapolation, J=(4 J_small-J_large)/3. C comes from independent
noise realizations at fixed Battaglia12 signal and mask. Their scatter includes
signal-noise and noise-noise split-cross terms and the simulated mask effects.
OAS regularizes the correlation matrix while retaining unbiased measured bin
variances; no raw-covariance Hartlap factor is applied to this estimator.

Fisher freezes C at Battaglia12. It omits covariance-derivative information and
changing-sky/halo variance. The NPE sees the actual varying-theta independent
noise simulator, so local Gaussian approximation errors can still distinguish
Fisher and NPE even when the physical inputs match.

All nine parameters vary jointly. The posterior multiplies the local Gaussian
likelihood by the **original hard uniform prior**. A Gaussian importance
proposal is divided out exactly. Independent importance runs must agree in
means and widths; rank is measured in prior-width coordinates. Weak modes are
not given finite data-only uncertainties through eigenvalue flooring.

## Additional fiducial calibration

The 32k data alone do not contain controlled central differences or a repeated
noise ensemble at exactly Battaglia12. The supplied calibration helper prepares
**294 additional maps by default**:

- 1 clean fiducial reference and 36 finite-difference variations.
- 256 independent-noise realizations at Battaglia12 for C.
- 1 separate Battaglia12 observation, excluded from C and from training.

`--noise-realizations 64` reduces this to 102 maps, with greater covariance
uncertainty. It is an explicit numerical setting, not a change in the noise
physics. Rows are resumable, checked against actual theta/beam/split seeds and
the ring-locked painter, and individually checksummed. A reserved seed namespace
is disjoint from every training and covariance/observation split. The simulator
source must match the collaborator's generation; old derivative arrays are not
silently reused across the painter and binning changes.

The helper runs the supplied full-map simulator for each calibration row. It
does not optimize repeated-noise rows by caching a new signal map, and it does
not introduce a Gaussian bandpower surrogate. Julia **1.12.2** and the supplied
dependencies are required by that simulator package.

## Training, compression and comparison

Defaults reserve the last 1,000 rows as untouched test observations. A saved
shuffle of the other 31,768 rows gives nested training+validation sizes
4,096, 8,192, 16,384 and 31,768. Each size uses the same 90/10 optimization/
validation split for all three methods.

- **40-bin NPE:** train-only asinh scaling and per-bin standardization retain
  all 40 coordinates.
- **PCA NPE:** nine principal components of train-standardized raw D_ell.
  This is a linear transform, distinct from the legacy asinh-PCA estimator.
  Its axes use only optimization rows. The training covariance includes
  changing parameter values and is used only to choose these axes.
- **MOPED NPE:** nine raw-linear D_ell combinations use the matched J and C.
  If C=L L^T and L^-1 J=U S V^T, the weights are B=L^-T U. Their covariance
  is identity and the compressed local Fisher matrix equals the full one.
  The code verifies this equality to relative error <1e-8, then standardizes
  the compressed values using optimization rows only.

The independent fiducial calibration is shared design information, not held-out
accuracy data. It is never fitted using the chosen observation. This new MOPED
definition is deliberately fitted to the new physical covariance; it does not
reuse the legacy fixed-noise regression transform. The information identity is
local and assumes frozen covariance, not globally lossless inference over the
whole prior. See [Heavens, Jimenez & Lahav](https://arxiv.org/abs/astro-ph/9911102).

All three methods use MAFs with 50 hidden features, 5 transforms, batch size 50,
patience 60, and maximum 2,000 epochs. The existing checked trainer preserves
the best validation weights explicitly, including when the epoch cap is hit.
These defaults are editable in the prepare CLI; training completion records
state whether early stopping occurred.

For each training size, `covariance_audit.json` and
`{bins40,pca,moped}_likelihood.npz` save an explicit shared-covariance check.
For a row-vector transform `x @ A`, the code uses `A.T @ C @ A` and
`A.T @ J`. Output standardization is included in A. It does not estimate a
second covariance after compression or use PCA's training scatter as noise.
The full-bin Fisher calculation stays in raw D_ell coordinates; the NPE's
asinh transformation is invertible. Both Fisher information and the observed
likelihood score must agree for full bins and MOPED. PCA has its own projected
Fisher posterior, with the same observation and hard prior, and may lose
information.

This matches the simulator, conditional covariance at Battaglia12, bins,
priors and observation. It does **not** make the local Gaussian approximation
identical to the full simulator likelihood. With cross spectra of
`s+n1` and `s+n2`, signal-noise terms make the covariance depend on the signal
and therefore the parameters. The implemented Fisher freezes C at Battaglia12;
it omits the Gaussian covariance-derivative term
`0.5 Tr(C^-1 C_,i C^-1 C_,j)`. Covariance derivatives and non-Gaussian likelihood
information require additional calibration. No completed output should be
described as an exact full-likelihood Fisher bound.

Evaluation draws 2,000 accepted samples per held-out profile and 20,000 at the
same independent Battaglia12 observation. Finite proposal/time budgets expose
sampling failures. No failed profile is silently omitted, no draw is clipped,
and no MCMC fallback hides a conditioning problem. Test metrics include
RMSE/prior width, RMS pulls, Pearson correlation, and 68/95% interval coverage.

## Run after the simulations arrive

Use a fresh output root and a Python environment containing NumPy, SciPy,
scikit-learn, pandas, matplotlib, GetDist, Torch and compatible SBI. The existing
SBI 0.22 training path is reused. The lightweight numerical/data code does not
upgrade or modify the environment.

Example paths below must be replaced with the collaborator's actual locations:

```bash
cd /path/to/HalfDome
DATA=/scratch/so32k/nine_param/prepared/dataset.npz
GEN=/scratch/so32k/experiment.json
BUNDLE=/path/to/so_32k_independent_noise
CATALOGUE=/path/to/lightcone_100.hdf5
CAL=/scratch/so9_battaglia12_calibration
RUN=/scratch/so9_matched_comparison
JULIA=/path/to/julia-1.12.2/bin/julia

python3 SBI_analysis/prepare_so_nine_fisher_calibration.py init \
  --dataset "$DATA" --generation-manifest "$GEN" --bundle "$BUNDLE" \
  --catalogue "$CATALOGUE" --root "$CAL"

# Four compute workers and a dependent combine job; remove --submit for dry run.
python3 SBI_analysis/submit_so_nine_independent_comparison.py calibration \
  --root "$RUN" --calibration-root "$CAL" --julia "$JULIA" --submit

# Wait for calibration_complete.json, not merely an empty PBS queue.
python3 SBI_analysis/submit_so_nine_independent_comparison.py prepare \
  --root "$RUN" --calibration-root "$CAL" --dataset "$DATA" \
  --generation-manifest "$GEN" --submit

# After prepare_complete.json: three training lanes plus dependent final summary.
python3 SBI_analysis/submit_so_nine_independent_comparison.py train \
  --root "$RUN" --calibration-root "$CAL" --submit
```

PBS jobs use `mini`, 26 CPUs/application threads, 128 GB and `23:59:00`.
No simulations run on a login node. Calibration workers stop at a walltime
margin; if rows remain, resubmit the calibration stage after its jobs finish.
The combiner rejects missing/corrupt rows. Submission receipts are written next
to RUN under `<RUN>_pbs/`. The helper refuses overlapping active stages.
Queue exit is not proof of completion.

For a customized training setup, run `run_so_nine_independent_comparison.py
prepare --help`, then invoke prepare directly on a suitable compute node. Once
prepared, the submission helper reads the saved sizes and settings. Changing
priors, sizes, source code or inputs requires a fresh experiment rather than
overwriting old results. An interrupted summary can be rerun until its final
completion marker exists; a completed summary is preserved.

## Output and verification

`RUN/summary/` contains the nine-parameter Fisher/40-bin/PCA/MOPED triangle, a separate
fiducial Fisher triangle, convergence figures, Fisher eigenvalue/covariance
plots (PNG/PDF), parameter interval tables, eigenmodes, derivative and covariance
sensitivity tables, and sampling diagnostics. `RUN/calibration.npz` stores the
Fisher matrices, derivatives, covariance, eigenvectors and MOPED weights.
`summary_complete.json` is written only after every requested run and test
observation has completed and the plots have been saved.

Software verification performed locally:

- Nine tests cover analytic truncated-normal moments, uniform null modes,
  correlated-noise MOPED identity, unbiased covariance diagonals, calibration
  roles and combination, train/test separation, noise-seed reuse, and observation leakage.
- A deliberately tiny **synthetic** workflow completed preparation, both MAF
  trainings with restored best weights, all held-out/Battaglia12 sampling,
  importance integration and all-nine-parameter plots. Its two-epoch cap is a
  software smoke test, not training convergence or scientific validation.
- CLI help and PBS shell syntax were checked. Submission is dry-run by default.

Re-run tests:

```bash
python3 -m unittest discover -s SBI_analysis -p test_so_nine_independent_comparison.py -v
bash -n SBI_analysis/run_so_nine_independent_stage.pbs
```

Full-resolution calibration, real 32k training, independent-noise posterior
calibration and the final scientific Fisher comparison have **not** been run.

September 20 update: PCA and exact covariance projection are implemented.
The suite now has 12 tests, including observed likelihood-ratio preservation,
output-scaling invariance, and loss of parameter information under PCA.
The cluster connection timed out, so availability of the collaborator's new
simulations has not been verified. See `BATTAGLIA12_COVARIANCE_AUDIT.md` for
the current audit and the separately labelled archived SBI diagnostic.
