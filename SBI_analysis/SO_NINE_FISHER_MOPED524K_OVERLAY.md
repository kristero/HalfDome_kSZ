# Nine-parameter Fisher and historical 524k MOPED overlay

This figure adds the existing fixed-noise Battaglia12 MOPED posterior to a
copy of the Fisher pilot's corner plot. All nine gNFW parameters vary jointly;
the original two-curve Fisher plot is preserved.

Completed on `mini` as PBS job **597166.idark**, host `ansys02`, with four
CPUs/application threads and exit code 0. The wrapper ran from 18:04:32 to
18:05:52 UTC on 2026-09-13; the analysis and plotting took 63.9 seconds.

Open the [comparison PNG](outputs/so9_fisher_moped524k_20260913/fixed_noise_overlay/battaglia12_fisher_moped524k_corner.png)
or [vector PDF](outputs/so9_fisher_moped524k_20260913/fixed_noise_overlay/battaglia12_fisher_moped524k_corner.pdf).
All six output checksums passed local verification after download. The 10,000
MOPED samples are byte-identical to the verified input posterior and inside the
saved prior; all 12 original Fisher artifacts remain unchanged. The final
figure was visually inspected for all nine parameter labels and three curves.

The plotted curves are:

- Blue dashed: the fiducial Fisher forecast, with the hard uniform prior.
- Orange filled: the Fisher posterior for independent noise root 20001,
  using the 16-realization covariance.
- Purple: the historical MOPED density estimator's conditional posterior for
  fixed noise root 12345, with its original hard prior.

Black dotted markers show Battaglia12. The figure explicitly states that the
observations and noise models differ. The user selected this diagnostic overlay
after the same-observation sampling check failed.

## Estimator and observations

The MOPED checkpoint comes from
`adrian_9param_compression_baseline_deproj0_bestval_20260909`, experiment
`35fbe0a27078c4b34d8aac9aee2b95fb4e2e4c977dc720ac516dd91176c302d6`.
Its model SHA256 is
`223494532fd812a5b8eeeaab202dc3cc243151da8ed6f0b6c41e8011205fab5e`.

The original dataset contains 524,288 rows; after prior filtering and the
held-out split, 461,737 rows were used for optimization and 51,305 for
validation. The saved MAF has nine MOPED inputs, 64 hidden features and six
transforms. It uses the best validation snapshot; the training cap was reached
without establishing early-stopping convergence. No retraining is performed.

The 10,000 fixed-noise posterior samples were previously drawn from 10,240
proposals, with 99.512% raw prior acceptance. The original sample checksum is
`632d74deb409fd636f5de47e17ca7bdb6b83509021bbcc8eb51a127b9168e998`.
The overlay verifies this checksum, the observation spectrum, parameter order,
all nine fiducial values, the experiment ID and the original preprocessing.
The model is independently checked against 32 saved reference log probabilities.
Existing samples are reused, without clipping, shifting or rescaling.

An initial cluster probe (`597163.idark`) applied this estimator to the same
raw observation as the Fisher posterior. It rebinned the raw spectrum using
the estimator's own bins and saved transform. **Zero of 20,000 raw proposals
fell within the nine-dimensional prior**, while a training-reference control
accepted 53.61%. This is a finite sampling result, not proof of mathematically
zero probability. The failed attempt is retained under the cluster run's
`results/`, with `incomplete.json`, the support diagnostic and exit receipt.
No rejected draws are displayed as posterior samples.

## Physical interpretation

Thermal SZ measures integrated electron pressure. The nine Battaglia parameters
describe pressure amplitude, radial scale and shape and their mass/redshift
evolution. The angular spectrum mixes these contributions, producing correlated
parameter constraints.

For noisy map splits s+n1 and s+n2, the cross-spectrum includes signal-signal,
signal-noise and noise-noise terms. The Fisher covariance varies the noise at
fixed signal and mask. The old NPE instead learned a conditional mapping with
the same noise maps reused across training parameters. Its narrow contours
therefore do not demonstrate better independent-noise measurement precision.
They also depend on the learned density approximation and training resolution.
For example, the fixed-noise MOPED standard deviations are 1.057 for P0 and
0.224 for beta, compared with 5.017 and 0.490 in the fiducial Fisher forecast.
These width differences compare different conditional inference problems;
they do not measure an information gain over a matched Fisher bound.

Each method keeps its own preprocessing and prior. The MOPED model uses the
old 40 bins starting at ell=80--279, signed-asinh standardization and the saved
regression-based MOPED transform. The Fisher pilot uses bins starting at
ell=80--199 and its local raw-linear spectral derivatives. Their prior bounds
also differ slightly. Nothing is reinterpreted as sharing a common likelihood.

The [Fisher pilot report](SO_NINE_FISHER_SMALL_ENSEMBLE.md) describes its strong
covariance regularization, weak modes and local Gaussian likelihood. The
physically matched independent-noise Fisher/NPE comparison still requires the
new simulations and retraining described in
[the matched-comparison runbook](SO_NINE_INDEPENDENT_FISHER.md).

## Code and reproduction

`plot_so_nine_fisher_moped524k.py` validates the saved inputs, constructs the
comparison and writes PNG/PDF, posterior samples, observation context, request
metadata, summary and an artifact checksum manifest. It requires a fresh output
directory. `run_so_nine_moped524k_overlay.pbs` runs it on `mini` and records the
actual job exit status; the small plotting job uses four CPUs/application threads.

```bash
python3 plot_so_nine_fisher_moped524k.py \
  --fisher /path/to/so9_fisher_noise16_20260913/results \
  --bundle /path/to/exported_nine_parameter_model/moped \
  --fixed-noise-posterior /path/to/fixed_noise_observation \
  --output /path/to/fresh_overlay --threads 4
```

The fixed-noise directory must retain `posterior_samples.npy`, `observation.npz`,
`simulation_complete.json`, `raw/<original-spectrum-name>.npy` and
`diagnostics/fixed_noise_sampling_summary.json`. Omitting the explicit
`--fixed-noise-posterior` argument probes the same independent-noise observation;
it does not silently switch observations when sampling fails.

The cluster run root is
`/lustre/work/kristero10/so9_fisher_moped524k_20260913`.
The requested figure belongs in `fixed_noise_overlay/`, separate from the
failed same-observation probe and the unchanged Fisher results.
