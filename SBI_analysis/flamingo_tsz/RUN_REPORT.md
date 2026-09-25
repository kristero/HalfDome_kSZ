# Completed FLAMINGO tSZ comparison

The approved campaign completed on `idark`, job **597329.idark**, with exit
code **0**, on 2026-09-14 at 11:32:16 UTC. It used the selected verified
**40-bin, nine-parameter MOPED/SBI bundle**. All three approved FLAMINGO maps
were downloaded, processed and passed through the saved compression and
neural density estimator.

**The trained model produced a HalfDome posterior, but each FLAMINGO
observation failed its original-prior sampling diagnostic: 0 of 20,000 raw
network proposals were inside the nine-dimensional prior.** FLAMINGO spectra,
compressed observations and rejection diagnostics are complete. There are
no FLAMINGO posterior contours to interpret from this run.

The full maps remain on the cluster at
`/lustre/work/kristero10/flamingo_tsz_comparison_20260914/inputs`.
Small results, plots and logs are also available locally in `cluster_results/`.

## Measured comparison

The following are percentage differences of the **clean, beamed, masked**
power, before noise is added. They use the original weighted bin centres
nearest the displayed multipoles, without interpolation or f_sky correction.

| Map | Versus HalfDome, ell=2980.619 | Versus HalfDome, ell=4980.169 | Versus FLAMINGO fiducial, ell=2980.619 | Versus FLAMINGO fiducial, ell=4980.169 |
| --- | ---: | ---: | ---: | ---: |
| FLAMINGO fiducial L1_m9 | +29.00% | +16.26% | 0.00% | 0.00% |
| fgas-8sigma | -8.59% | -29.32% | -29.13% | -39.21% |
| Mstar-1sigma | +30.93% | +16.02% | +1.50% | -0.21% |

The fgas variation increasingly suppresses small-angular-scale power relative
to FLAMINGO fiducial: approximately 15.28%, 29.13%, 39.21% and 46.81% in the
bins nearest ell=1000, 3000, 5000 and 7000. The Mstar variation changes this
statistic much less: +3.21%, +1.50%, -0.21% and -2.13%, respectively. These
are results for the selected lightcone and sky mask; no ensemble scatter or
statistical significance of the differences was estimated.

![Clean and noisy spectra](cluster_results/comparison/spectra_comparison.png)

All 40 fresh HalfDome noisy bins match those of the saved historical
observation exactly at the model's float32 input precision. Applying the
same saved transform to the fresh and historical spectra in the same
evaluation process also gives **identical nine-component contexts**:
relative binned L2 difference = 0; maximum context difference = 0. This checks
the complete painting, beam, noise, mask, spectrum, binning and compression
chain, in addition to the small operator regression.

## What happened in SBI

| Observation | Inside original prior in pilot | Pilot acceptance | Saved posterior samples |
| --- | ---: | ---: | ---: |
| Fresh HalfDome control | 19,887 / 20,000 | 99.435% | 10,000 |
| FLAMINGO fiducial L1_m9 | 0 / 20,000 | 0% observed | 0 |
| fgas-8sigma | 0 / 20,000 | 0% observed | 0 |
| Mstar-1sigma | 0 / 20,000 | 0% observed | 0 |

The fresh HalfDome production sampling acceptance was 99.4434%; its nine
marginals agree visually with the separately saved historical posterior.
The marginal means, standard deviations and 16th/50th/84th percentiles are
in `cluster_results/comparison/posterior_summary.csv`. FLAMINGO entries are
explicitly marked `posterior_unavailable`, with empty parameter estimates.

The FLAMINGO failures are informative. In every variant, all 20,000 raw
proposals place `xc` and `beta` above their saved upper bounds, and
`alpha_z_P0` below its lower bound. Other parameters also violate the prior.
A separate saved training-reference context accepts 2,268/4,096 proposals
(55.37%), confirming that the same loaded estimator can sample in support.

The diagnostic figure below shows **unconstrained flow proposals**, not
posterior estimates. The shaded region is the original prior; markers and
bars show proposal medians and 5th-95th percentiles in units of prior width.

![Original-prior rejection diagnostic](cluster_results/report/inference_support.png)

Zero accepted samples does not prove mathematically zero posterior support,
nor does it prove that every physical Battaglia model is excluded. For
20,000 independent proposals with zero successes, the one-sided 95%
binomial upper bound on the learned flow's in-prior probability is about
0.01498%. This is a sampling diagnostic, not a physical confidence limit.
No forward-model goodness-of-fit search was performed.

The largest absolute standardized MOPED coordinates are 6.069 (fiducial),
4.370 (fgas-8sigma), and 5.994 (Mstar-1sigma), compared with 0.617 for the
fresh HalfDome control. All nine fgas-8sigma coordinates lie individually
inside the ranges of the 512 saved training-reference contexts, yet its
proposals still fail the joint prior check. Per-coordinate ranges are
therefore insufficient to establish compatibility with the learned joint
distribution. Priors and contexts were not widened, clipped or rescaled.

The bundle is experiment
`35fbe0a27078c4b34d8aac9aee2b95fb4e2e4c977dc720ac516dd91176c302d6`,
using density-estimator SHA256
`223494532fd812a5b8eeeaab202dc3cc243151da8ed6f0b6c41e8011205fab5e`.
It uses the saved best-validation snapshot after 201 epochs; the recorded
training run did **not** meet its early-stopping convergence condition.
This campaign used that requested checkpoint without retraining.

## Matched operations and their physics

Thermal SZ measures integrated electron pressure:

\[
y(\hat n)=\frac{\sigma_T}{m_e c^2}\int P_e\,d\ell_{\rm phys}.
\]

The supplied FLAMINGO maps contain dimensionless Compton y. No conversion
to a frequency-dependent temperature or microkelvin power was applied.
The original files are HEALPix RING maps with Nside=4096, 201,326,592
Float64 pixels. All pixels are finite. Negative values in the source maps
were retained rather than clipped.

The existing HalfDome implementation applies a 2-arcmin Gaussian beam to
the signal, with response

\[
B_\ell=\exp[-\ell(\ell+1)\sigma_b^2/2],\qquad
\sigma_b=\frac{\mathrm{FWHM}}{\sqrt{8\ln2}}.
\]

The same smoothing function and its default harmonic limit were retained.
Native SO baseline/deprojection-0 y-noise was then added without another
beam smoothing. The noise root seed and mask seed are both 12345; the
original deterministic split seeds are **22446 and 22447**. A single cached
mask and pair of full noise arrays were reused for every map, with their
pixel and file SHA256 hashes verified at each stage.

The mask is the original randomly oriented cap with support f_sky=0.4 and
60-arcmin cosine apodization. Its measured support is 0.400000155, mean
weight 0.395729447, and mean squared weight 0.394662606. These distinctions
matter because a weighted masked spectrum is not a full-sky spectrum
multiplied by exactly 0.4 at every multipole.

The estimator is the raw pseudo cross-spectrum of
`W(By+n1)` and `W(By+n2)`, using ell_max=7979 and niter=0. There is no added
beam deconvolution, f_sky division, mode-coupling correction or noise-bias
subtraction. With masked signal s and masked noise splits n1,n2, it contains

\[
\widetilde C_\ell^{12}=
\widetilde C_\ell^{ss}+
\widetilde C_\ell^{s n_2}+
\widetilde C_\ell^{n_1 s}+
\widetilde C_\ell^{n_1 n_2}.
\]

Independent split noise removes an additive noise mean only in an ensemble
expectation; the actual fixed-realization cross terms remain. Reusing the
same noise maps holds the last term fixed. Changing the sky signal changes
its correlations with those noise maps. Clean, noisy and noise-only spectra
were retained so these terms can be inspected separately.

The original 40 bins span ell=80-7979 and average signed linear
`D_ell = ell(ell+1) C_ell/(2*pi)` with `(2ell+1)` weights. The saved asinh
scale, input centering and scaling, MOPED matrix, projection centre, and
output standardization then produce the nine model inputs. MOPED weights
were not recomputed for FLAMINGO. They are optimized for derivatives and
covariance of the trained HalfDome family, so modest changes in spectra
need not correspond to valid observations under the learned joint model.

The inferred parameters are `P0`, `xc`, `beta` and their six mass/redshift
exponents. They describe the HalfDome Battaglia pressure family. Even if
sampling had succeeded, those parameters would be effective interpretations
of FLAMINGO, not recovery of FLAMINGO's feedback-calibration inputs.

## Simulation differences that remain

The provided FLAMINGO maps are lensed, use the documented shell rotations,
and for the selected L1 runs integrate to **z=3**. Recently AGN-heated gas
is excluded in the source Compton-y shells. These are conventions of the
published products; this campaign did not recreate or alter them.
[FLAMINGO map documentation](https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/integrated_lightcones.html#integrated-thermal-sz-maps),
[Yang et al. methodology](https://academic.oup.com/mnras/article/548/2/stag625/8571439).

The freshly inspected HalfDome catalogue contributes **85,224,251 haloes**,
with M200c >= 10^12 Msun, spanning **z=0.002141595 to 3.855427742**. Its
painting uses h=0.68, Omega_b=0.049 and Omega_c=0.261. It paints a halo
pressure model rather than the same hydrodynamic gas realization. The
HalfDome/FLAMINGO comparison therefore includes redshift coverage, cosmology,
gas modelling, lensing and realization differences alongside feedback.
No empirical redshift or amplitude correction was applied.

Within the three FLAMINGO variants, common cosmology and product conventions
make the relative comparison more useful for isolating feedback effects.
Feedback changes gas density, temperature and radial pressure distribution;
the increasing suppression in fgas-8sigma is consistent with less pressure
power on small angular scales. The measured ratios alone do not uniquely
identify which gas process causes that suppression. The fgas/Mstar labels
are shifts in observational calibration targets, not changes to one
Battaglia parameter by those numbers of standard deviations.
[FLAMINGO feedback models](https://dataweb.cosma.dur.ac.uk:8443/flamingo/simulations/hydro.html).

Ready integrated maps were available for these three variants. The checked
paths for fgas+2sigma, fgas-2sigma, fgas-4sigma,
Mstar-1sigma_fgas-4sigma, Jet and Jet_fgas-4sigma returned HTTP 404.
No unapproved shell-download campaign or substitute map was used.

The selected SBI model is conditional on a fixed noise realization and the
HalfDome signal family. These results do not provide SO constraints with
noise or sky realizations marginalized. Obtaining calibrated FLAMINGO
posteriors would require a separately specified validation/training study
covering the relevant simulation differences; rejection sampling tricks
would not establish calibration of this checkpoint on those observations.

## Resources, implementation and validation

The approved three downloads total **4,556,709,696 bytes = 4.244 GiB**.
The measured campaign workspace occupies approximately **11.4 GiB**,
including inputs, shared mask/noise cache, matching Julia runtime and its
archive. This is below the approved 20-GiB budget.

| Production stage | Elapsed seconds | Peak resident memory, GiB |
| --- | ---: | ---: |
| Small operator regression | 20.3 | 0.80 |
| Shared mask/noise cache | 107.4 | 7.57 |
| Fresh HalfDome painting and spectra | 292.4 | 10.97 |
| FLAMINGO fiducial | 135.9 | 9.04 |
| fgas-8sigma | 140.3 | 9.08 |
| Mstar-1sigma | 137.7 | 9.08 |
| MOPED/SBI and comparison plots | 11.4 | 0.40 |

Production used queue `mini`, 26 CPUs, 64 GB requested RAM and a
23:59:00 walltime limit. Stage timings exclude runtime installation,
downloads and queue waits. RAM values are measured GNU time maximum RSS,
not forecasts or scheduler allocation sizes.

Matching integer seeds was not sufficient with the cluster's older Julia
runtimes. An isolated **Julia 1.12.2** environment with the reference
dependency closure and unchanged XGPaint source was installed. The small
mask/noise regression then matched the reference arrays, and the final
fresh HalfDome spectrum reproduced the historical model inputs exactly.
The standalone checks did not replace this full production validation.

The download validator preserves either endian representation of Float64:
the remote metadata advertised `<f8`, while the actual HDF5 files use `>f8`.
Four focused tests cover valid signed big-endian data and rejection of wrong
shape, Float32 and non-finite inputs; they passed locally and on the cluster.
Every downloaded map was scanned for finite values and checksummed. The
selected bundle and 32 reference model log probabilities passed the existing
load-time checks. All seven production stage receipts have exit code 0.
All 37 primary output checksums were verified again locally, and all four
comparison/diagnostic figures were visually inspected.

New readable pipeline files in `SBI_analysis/flamingo_tsz/`:

- `download_maps.py` and `run_download.pbs`: approved resumable downloads,
  size/dtype/value checks and checksums.
- `process_maps.jl`: unchanged HalfDome operators with FLAMINGO signal
  ingestion, exact shared noise/mask caching, fresh HalfDome control and
  spectrum/provenance output.
- `stage_runtime.py`, `runtime_env/Project.toml`,
  `runtime_env/Manifest.toml` and `setup_runtime.pbs`: isolated matching Julia
  environment and source staging.
- `run_campaign.py` and `run_campaign.pbs`: sequential cluster execution,
  resource receipts and success markers.
- `compare_inference.py`: original binning/MOPED/network helpers, bounded
  posterior sampling, failure diagnostics, tables and three comparison plots.
- `summarize_results.py`: independent artifact/input checks, selected-scale
  comparisons and a clearly labelled network-proposal diagnostic figure.
- `fetch_results.py`: transfer and verification of small outputs, with full
  input/cache maps left on the cluster.
- `test_pipeline.py`: the four download-data validation tests.

`README.md` and `job_history.json` were updated. `approval.json`,
`RUN_REPORT.md`, source manifests/archives, runtime probes and
`cluster_results/` were added. `preflight.py` and its frozen
`metadata_manifest.json` came from the preceding approval preflight and were
retained. Existing HalfDome source, trained models, priors and prior figures
were not edited. See `changed_files.txt` for the full local file inventory.

## Reproduce the small result checks

On the cluster, after the successful campaign:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/anaconda3/bin/python3 \
  /lustre/work/kristero10/flamingo_tsz_comparison_20260914/code/summarize_results.py \
  --results /lustre/work/kristero10/flamingo_tsz_comparison_20260914 \
  --output /lustre/work/kristero10/flamingo_tsz_comparison_20260914/report
```

To retrieve the small outputs again from the local repository:

```bash
python SBI_analysis/flamingo_tsz/fetch_results.py \
  --remote-root /lustre/work/kristero10/flamingo_tsz_comparison_20260914 \
  --output SBI_analysis/flamingo_tsz/cluster_results
```

For a new full campaign, stage a new directory and use the three PBS scripts
with its `CAMPAIGN` variable. The completed directory is the provenance
record for this run; retain its original science outputs and manifests.
