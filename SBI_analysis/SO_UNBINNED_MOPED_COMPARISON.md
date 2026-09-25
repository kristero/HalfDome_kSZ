# Nine-parameter unbinned MOPED comparison

The requested comparison uses every available integer multipole from **80 to
7979 inclusive (7,900 inputs)**. The new estimator receives the nine MOPED
summaries fitted directly to those multipoles. It does not receive 7,900 neural
network inputs. The reference pipelines are the existing 40-bin MOPED estimator
and the estimator trained directly on 40 bins.

## Completed results (14 September 2026)

All three methods completed the same 495 held-out rows, with 2,000 posterior
samples per row, and the same fixed-noise Battaglia12 observation, with 10,000
samples per method. The final summary completed successfully as job
`597331.idark`. No held-out rows were omitted.

| Pipeline | RMSE / prior width | Mean posterior std / prior width | 68% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| Direct 40-bin NPE | 0.23945 | 0.19287 | 54.86% | 88.55% |
| 40 bins -> MOPED -> NPE | 0.12479 | 0.09049 | 58.99% | 90.26% |
| 7,900 multipoles -> MOPED -> NPE | 0.12298 | 0.08669 | 58.16% | 90.98% |

Removing the initial bin averaging reduced aggregate RMSE by only **1.45%**
relative to 40-bin MOPED, and reduced average posterior standard deviation by
**4.20%**. A paired bootstrap over whole held-out rows (4,000 resamples, seed
20260914) gives a 95% interval of **-1.28% to +4.13%** for the RMSE reduction.
Thus this run does not establish a clear overall accuracy gain over 40-bin
MOPED. This bootstrap measures held-out row-selection sensitivity, excluding
training-seed variation and independent-noise uncertainty. The corresponding
width-reduction interval is 2.94% to 5.46%.

Against the direct 40-bin NPE, unbinned MOPED reduces RMSE by **48.64%**. The large
gain is associated with using MOPED in the inference pipeline; removing the
40-bin average adds comparatively little. Both MOPED methods under-cover on
the held-out set, so their narrower posteriors should not be treated as fully
calibrated uncertainty estimates.

At Battaglia12 specifically, unbinned MOPED is not uniformly tighter. Relative
to 40-bin MOPED, its posterior standard deviations are **38.8%, 43.2% and 35.8%
larger for P0, xc and beta**. They are **17.7% and 13.4% smaller for
alpha_z_P0 and alpha_z_beta**. These point-specific changes differ from the
average over the prior. All three methods use the identical observation.

The two MOPED estimators both hit the configured training cap, with the best
validation snapshot restored. Results therefore compare these finite-budget
trained estimators; they do not prove that 40-bin averaging is exactly lossless.
The historical fixed-noise limitation discussed below also remains.

Figures and numerical tables are in
`outputs/unbinned_moped_524k_20260914/summary/`. The paired comparison is in
`outputs/unbinned_moped_524k_20260914/paired_comparison.json`.

## Data and matching

All three methods use the historical 524,288-row masked SO baseline-noise cross
spectrum dataset, deprojection 0, and all nine gNFW parameters, in this order:

`P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta, alpha_z_P0, alpha_z_xc, alpha_z_beta`.

The Battaglia12 values are
`[18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]`.
The first three parameters control the pressure normalization, characteristic
radius, and outer-profile shape. The other six describe their mass and redshift
dependence. Cosmology, beam, mask and other profile settings retain the historical
dataset convention.

The saved inference bounds are authoritative. The script verifies the prepared
input digest against the reference experiment, then checks parameter values,
Sobol row identities, multipoles and product labels in both raw metadata files.
It re-bins **every raw noisy and clean spectrum** and checks that it agrees with
the corresponding reference input to float32 summation accuracy. It does not
assume that dataset order equals Sobol CSV order.

The shared split is copied and independently reconstructed from the original
seed and bounds: 461,737 optimization rows, 51,305 validation rows, and 495
in-prior rows among the original final 500 test rows. The 10,751 rows outside
the saved bounds are excluded. Validation and test rows never fit a transform.

## Compression and its physical interpretation

For each multipole, convert the signed cross spectrum using

\[
D_\ell=\frac{\ell(\ell+1)}{2\pi}C_\ell.
\]

The raw spectrum keeps its sign. On optimization rows alone, fit the same
feature-wise preprocessing as the reference pipeline: lower median absolute
value as the scale, signed `asinh`, then mean and standard deviation. This
invertible preprocessing handles the large dynamic range without taking a
logarithm of a negative cross spectrum. In the reference, preprocessing acts on
the already averaged 40 bins; here it acts on individual multipoles. Therefore
this is a comparison of complete preprocessing/compression pipelines.

Select the same 20,000 optimization rows nearest Battaglia12 in prior-width
units. Fit the 55-term quadratic design (constant, nine linear terms and 45
quadratic terms) to the transformed clean spectra and to paired noisy-minus-clean
residuals. The derivative of their fitted sum gives the 7,900 by 9 matrix `J` at
Battaglia12, with respect to prior-normalized parameters. Fitting the residual
mean accounts for the nonlinear `asinh` transformation.

Estimate the residual covariance after subtracting its fitted conditional mean,
correct its regression degrees of freedom, and use the reference 5% diagonal
shrinkage:

\[
C=0.95S+0.05\operatorname{diag}(S).
\]

For `C = L L^T`, form the thin SVD `L^{-1} J = U s V^T` and retain modes with
`s > 10^{-6} s_max`. The compression weights are `B = L^{-T} U`. The summaries
are `B^T (x - mu_fid)`, standardized again using optimization rows alone. The
code verifies

\[
B^T C B=I,\qquad
F_{\rm compressed}=(B^T J)^T(B^T J)\simeq J^T C^{-1}J.
\]

These identities concern the fitted local, fixed-covariance mean-sensitivity
model. They do not prove exact information preservation for a nonlinear
posterior, or account for parameter derivatives of the covariance. The
half-neighborhood derivative change and clean quadratic fit residuals are saved
for judging the local approximation.

The historical dataset uses fixed noise maps across parameter rows. Consequently,
the residual scatter above is not an ensemble covariance over independent sky
noise realizations. Additional multipoles can expose more parameter information
conditional on that fixed realization. Narrower contours alone would not show
better calibrated uncertainties for new noise realizations. This comparison
cannot replace the separate planned independent-noise training campaign.

## Training and comparison outputs

The reference 40-bin checkpoints are copied into an isolated output directory
and verified as best-validation snapshots. The new model uses their training
recipe: MAF, 64 hidden features, six transforms, batch size 1,024, patience 20,
maximum 200 epochs, seed 42, and the identical optimization/validation split.
The best-validation snapshot is explicitly restored even if the epoch cap is
reached. Numerical validation likelihoods from models conditioned on different
summaries should not be treated alone as information-loss estimates.

Each method draws 2,000 prior-supported posterior samples for each of the same
495 held-out rows. The script records RMSE divided by prior width, posterior
standard deviation divided by prior width, RMS standardized error, marginal
68%/95% coverage, and true-versus-posterior-mean correlation. A failed row
prevents a completed comparison; it is not silently omitted. Samples are never
clipped to the prior or replaced with a different inference method.

Three reference MOPED rows require more than the historical 200,000 proposals:
indices 523791, 523946 and 524162 accepted 1,491, 1,929 and 542 samples at that
limit. Their nonzero acceptance rates distinguish this from a zero-support
observation failure. The current PBS wrapper permits up to 5,000,000 proposals
and 600 seconds per row, preserving the same posterior target and 2,000 accepted
samples. `evaluation_runtime.json` records effective budgets separately from the
immutable training/preparation configuration. Completed row checkpoints are
reused, and no test rows are dropped. These limits can be set with
`EVAL_MAX_PROPOSALS` and `EVAL_SAMPLING_SECONDS` or the corresponding evaluation
CLI options. A regression test checks configuration preservation and budget
reuse across restarts.

All three methods also use the same previously painted Battaglia12 observation
with noise and mask seed 12345, verified by its receipt and SHA-256. Each
fiducial posterior uses 10,000 accepted samples after a 20,000-draw support
preflight. The summary saves PNG and PDF versions of:

- `battaglia12_unbinned_vs_40bin_corner`: all three methods at Battaglia12;
- `heldout_unbinned_vs_40bin_corner`: all methods for the shared held-out row
  nearest Battaglia12, explicitly labelled as a different parameter point;
- `unbinned_vs_40bin_metrics`: the six parameter-wise diagnostics.

CSV tables contain aggregate metrics, individual-parameter metrics and
Battaglia12 credible intervals. JSON markers record provenance, training
convergence, sampling failures and completion. Read those before interpreting
the figures.

## Cluster execution

Campaign directory:
`/lustre/work/kristero10/so_unbinned_moped_524k_20260914`.
The `code/` directory is a staged snapshot, `logs/` contains live logs and exit
receipts, and `analysis/` contains transforms, models, samples and `summary/`.
Raw inputs remain in `/lustre/work/kristero10/adrian_dataset`.

Stage the runner, PBS wrapper, submission script and their imported helper
modules in `code/` of a fresh campaign. Then run:

```bash
python3 code/submit_so_unbinned_moped_comparison.py --campaign "$PWD"
python3 code/submit_so_unbinned_moped_comparison.py --campaign "$PWD" --submit
```

The first command previews the five jobs. The second submits preparation,
new training plus evaluation, two reference evaluations, and a summary that
depends on all evaluations succeeding. The default queue is `mini`, with 26
CPUs and application threads, 23:59:00 walltime, 64 GB for preparation and
32 GB for subsequent stages. Memory mapping and blocks avoid materializing
all 524,288 by 7,900 transformed spectra simultaneously. The dense local
covariance still uses 7,900 by 7,900 entries.

Preparation requires a fresh `analysis/` directory. Training and evaluation
can reuse verified completed checkpoints. A stage can be rerun after correcting
an execution problem, retaining the same campaign and matching configuration:

```bash
qsub -v CAMPAIGN=/path/to/campaign,STAGE=evaluate,METHOD=unbinned_moped \
  /path/to/campaign/code/run_so_unbinned_moped_stage.pbs
```

Do not change numerical settings or the source data while reusing an experiment.
Create a new campaign for a changed scientific configuration.

## Validation and execution record

The numerical tests passed both locally and under cluster Python 3.8 before
submission. They check exact block statistics against a dense calculation,
independence from held-out rows, local Fisher preservation, streamed projection
and observation agreement, and rejection of permuted row identities or altered
spectra. These are software checks, separate from the real-data analysis.

Preparation completed as job `597272.idark` in 970 seconds. Both full-dataset
re-binning checks were bitwise exact on the cluster. The independent saved-array
audit found nine MOPED components, finite contexts for every row, no local-fit
test leakage, a compressed-covariance identity error of `1.93e-14`, and a relative
Fisher error of `2.19e-15`. Recomputing 200 projected optimization/test rows gave
zero float32 differences. Audit: `analysis/compression_audit.json`.

Halving the local neighborhood changed the nine derivatives by 1.61% to 4.80%.
The worst clean quadratic fit error was 314 times the corresponding fitted
residual standard deviation. The latter is a warning against interpreting this
fixed-noise local approximation as a precision Fisher forecast; the held-out
neural posterior diagnostics remain essential.

The initial inference attempts (`597273`--`597275`) stopped before inference
because `PYTHONNOUSERSITE=1` hid the cluster's user-installed Torch and nflows.
The wrapper now enables those packages and records a runtime preflight. No
scientific Python code or prepared arrays changed. Failed-attempt logs are
retained under `logs/attempt1/`, and `submission_runtime_restart.json` records
the amended wrapper checksum. The cluster uses Torch 2.4.1 and SBI 0.22.0.

Training plus unbinned evaluation is job `597283.idark`. Direct 40-bin evaluation
completed as `597284.idark`. Reference MOPED evaluation `597285.idark` reproduced
the three historical proposal-limit failures; its resume is `597288.idark`.
The final summary is now `597289.idark`, depending on training/evaluation and
the reference resume. The changed inference-budget code and original failures
are retained under `revisions/before_sampling_extension/`, with new checksums in
`submission_sampling_extension.json`. These dependency chains were superseded
by the final sampling resume below. The reference resume completed
successfully in 70 seconds: all 495 shared test rows now have samples for both
40-bin methods, and both methods have their 10,000-sample Battaglia12 posterior.

Unbinned training completed in 4,097 seconds. SBI reported 201 epochs at the
configured 200-epoch cap, without early-stopping convergence. Its best validation
performance was 26.2681, and the saved best-validation weights differed from the
final epoch weights. The unbinned Battaglia12 posterior accepted 98.93% of draws.
One unbinned held-out row, 523946, accepted 1,648 samples from two million
proposals. Its final resume uses the five-million/600-second limits above as
job `597330.idark`; the summary now depends on that resume as `597331.idark`.
The earlier failures and effective sampling limits remain recorded in the
campaign's revisions and submission receipts. Training was not repeated. The
final resume succeeded and the summary completed at 11:24:52 UTC on
14 September 2026. Full prepared arrays and all per-row posterior samples remain
on the cluster. The compact export includes plots, tables, per-row metrics,
fiducial samples, all three models/transforms, and a small `inference_contract.npz`
with bounds and held-out identities; it omits the full 524k-row `shared.npz`.

## Files added

- `so_unbinned_moped.py`: checked raw readers, block normalization, local MOPED
  fitting, streamed projections and single-spectrum inference preprocessing.
- `run_so_unbinned_moped_comparison.py`: matched preparation, model reuse and
  training, resumable bounded posterior sampling, metrics and comparison plots.
- `run_so_unbinned_moped_stage.pbs`: cluster resource/runtime configuration and
  stage exit receipts.
- `submit_so_unbinned_moped_comparison.py`: dry-run/submission of dependent jobs.
- `test_so_unbinned_moped.py`: four numerical, row-integrity and resume tests.
- `SO_UNBINNED_MOPED_COMPARISON.md`: this physics, execution and results record.

Run-specific utilities under `outputs/unbinned_moped_524k_20260914/` are
`verify_preparation.py`, `verify_and_compare_results.py`, `export_results.py`,
`restart_runtime.py`, `extend_sampling_budget.py`, `finish_unbinned_sampling.py`,
`cluster_ssh.sh`, and `software_smoke/render_corner.py`. The SSH helper uses the
existing credentials through WSL and removes its temporary key copy on exit.
The software-smoke image is labelled synthetic and separate from science plots.
