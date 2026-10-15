# HalfDome simulation and mask variation

This experiment runs on **idark**. It uses the existing beam-corrected, noise-only-whitened Merlin ensemble without retraining. The unbounded model is primary; the bounded model is a companion.

## Observations

- Simulation arm: Gaussian lightcones 100, 102, ..., 118, all with mask seed 12345.
- Mask arm: lightcone 100 with mask seeds 12345, ..., 12354. Seed 12345 is the training mask.
- The shared baseline is lightcone 100, mask 12345: 19 distinct observations in total.
- One noise prescription: SO baseline deproj0. Its independent split-map draws use seeds 39001001 and 39001002, fixed across all observations as in the saved B12 observation.
- All catalogue redshifts, catalogue M200c divided by h=0.68, physical mass above 1e12 solar masses, the frozen training cosmology and B12 parameters.

Each halo contributes the electron-pressure integral

\[
y(b)=\frac{\sigma_T}{m_e c^2}\int_{-\sqrt{(4R_{200c})^2-b^2}}^{+\sqrt{(4R_{200c})^2-b^2}}
P_e\!\left(\sqrt{b^2+l^2}\right)\,dl.
\]

Here b is transverse distance from the halo centre, l is line-of-sight distance, P_e is electron pressure, sigma_T is the Thomson cross-section, and m_e c^2 is the electron rest energy. The integral is zero outside the finite 4R200c sphere. R200c encloses 200 times the critical density. The inherited spherical projector and its angular geometry are unchanged from training.

The nine B12 parameters control the amplitudes of pressure normalization P0, radius xc and outer slope beta, followed by their mass and redshift exponents. Their order and fiducial values are in `plan.json`; alpha=1 and gamma=-0.3 remain fixed. Halos differ between lightcones, while this pressure law remains fixed. The seed comparison therefore measures changes in the sky power spectrum, rather than changes to an individual halo's analytic pressure law.

## Observation and inference contract

The frozen painter evaluates a point-sampled NSIDE 8192 sky, applies a 2-arcmin Gaussian beam in harmonic space, and synthesizes NSIDE 4096. Each observing mask is a randomly oriented 40%-sky cap with 60-arcmin cosine apodization. We retain signed split-cross pseudo-spectra at ell=80,...,7979. There is no additional fsky correction, pixel-window division, noise-bias subtraction or beam deconvolution.

Let Css be the masked beamed signal spectrum, Cprod the cross of two masked signal-plus-unbeamed-noise maps, and Cnn their noise-only cross. The model was trained using

\[
C_\ell^{\mathrm{input}}=C_\ell^{ss}+B_\ell(C_\ell^{\mathrm{prod}}-C_\ell^{ss}-C_\ell^{nn})+B_\ell^2 C_\ell^{nn}.
\]

B is the 2-arcmin Gaussian harmonic beam. This gives the signal-noise and noise-noise terms the same beam convention as the signal. It is approximate after masking, since smoothing and a mask do not commute. Each observation also stores an exact map-level beamed-noise cross as a diagnostic. Only the training-matched correction enters the primary inference.

Each split has the full-depth noise power used in training; no sqrt(2) rescaling is introduced. Independent splits remove additive noise power in expectation but leave noise scatter and signal-noise terms in each realization. Holding the same pair fixed isolates sensitivity to simulation and sky-window choices.

The saved noise-only covariance Cholesky factor L and PCA basis V give s=(L^-1 D)V, where D=ell(ell+1)C/(2pi). The saved asinh transformation maps s into the 83 network inputs. All three trained network seeds are mixed equally. Each case receives 90,000 posterior samples with a common posterior random seed. The bounded and unbounded networks load in separate processes. The unbounded model's mass outside the training box is explicitly reported as extrapolation.

## Execution and validation

Cluster root: `/lustre/work/kristero10/tsz_seed_mask_variation_20261002`.

1. Stage these source files in `code/`, then run `prepare_idark.py` there. This freezes code, model weights, preprocessing and reference spectra; it only reads catalogue metadata.
2. `analyze.py --root <root> --preflight` verifies frozen SHA256 hashes and reconstructs the saved baseline features.
3. `submit.py --root <root>` submits ten painting jobs and one afterok analysis job. The submission helper distributes painting across mini and mini_B (GroupA) and mini2 (GroupB), preferring GroupB first if it has a free 26-CPU slot. Jobs request 26 CPUs and 23:59:00. Painting requests 64 GB; analysis requests 24 GB. No other jobs are submitted by this experiment. The PBS files retain mini as a fallback, overridden by the helper's explicit queue choice.
4. Seed 100 paints once and reuses its signal and noise maps for ten masks. Other seeds paint once with the baseline mask. Fresh seed-100 spectra must match the saved B12 baseline before inference is accepted.
5. Final analysis verifies output hashes, fixed noise-map hashes and the beam algebra, then produces posteriors, tables and publication-sized PDF/PNG figures. `analysis/COMPLETE.json` is written only after all cases and both models finish.

Maps stay in memory and are released after use; no multi-GB disposable maps are saved. Expected peak memory is 25-40 GiB per painter; new disk use is estimated below 2 GiB including frozen models and posterior samples. Each input catalogue is about 14 GiB; no catalogue is copied. Manifest SHA512 values are recorded as supplied provenance, not described as freshly verified full-file hashes.

The initial submission was moved between queues at the user's request: seeds 102/104 to mini2, 114/116/118 to mini_B, and the rest remain in mini. `queue_changes.json` records those operations; job IDs and dependencies did not change. The original `plan.json` resource field records the initial mini submission, not the later live queue assignment.

## Reading the results

`parameter_comparison.csv` contains medians, q16/q84, widths, shifts from the shared baseline, width ratios, Monte Carlo precision diagnostics and extrapolated mass. A normalized shift uses half the baseline 68% width; it is not automatically a Gaussian significance. `variation_summary.json` compares scatter in recovered medians and variation in widths. Figures show full-sky and masked spectra, parameter intervals, shifts and width ratios. Forty plotting bins improve readability; inference uses all 7,900 multipoles.

Ten masks on the same sky overlap. Their scatter describes sensitivity to observed sky regions, not ten independent cosmological realizations. This is a robustness test conditional on fixed noise and fiducial B12 physics. It does not calibrate coverage, marginalize cosmic variance in the posterior, or establish how retraining on multiple skies would improve SBI.
