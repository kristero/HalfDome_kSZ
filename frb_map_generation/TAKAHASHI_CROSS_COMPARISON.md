# HalfDome versus Takahashi: halo-only angular cross-correlation

## Scope and inputs

Reference: Takahashi et al., [arXiv:2511.02155v2](https://arxiv.org/abs/2511.02155v2),
Figure 13. The measurements and asymmetric error-bar extents were approximately
digitized from the published figure; they are not author-supplied data vectors.
The original digitization and catalogue are preserved under
`outputs/observational_comparison_inputs_20260913`.

The published footprint flags select 71 Planck sources after removing the three
cluster-host FRBs, and 31 ACT sources. Their mean redshifts are 0.274607 and
0.323455, respectively; both samples reach z=2.148. The 131-source DM-redshift
fit sample is NOT the cross-correlation sample. No replacement z=1 or mean-z
source plane is used here.

- tSZ: the regenerated Battaglia12 pressure map of all 85,224,251 HalfDome
  catalogue halos, up to z=3.85543. Its original projected 4R200c footprint is
  retained; the pressure model is not replaced with an isothermal model.
- DM: all resolved foreground halos up to each observed source redshift,
  physical M200c = catalogue `halo_mass_m200c / 0.68`. No additional science
  mass cut, host placement, or halo subsampling.
- DM boundary: finite **spherical 3R200c**, in both the LOS integral and painted
  footprint, using the validated radius-coordinate cache. This is distinct
  from an untruncated LOS with only a projected 3R200c footprint.
- Models: Battaglia16 and the existing no-concentration Lee22 implementation.
  Preferred Lee22 + Duffy08 is shown in a separate diagnostic plot: its
  high-mass extrapolation fails the gas-equivalent-mass consistency check.
- DM is a **halo-only partial prediction**. No diffuse/IGM or host-galaxy DM
  component is added, and no normalization is fitted to the observations.

## What the code computes

For source weights w_j and observed redshifts z_j, define

`K_s(z_h) = sum_j [w_j * I(z_j >= z_h)] / sum_j w_j`.

The generator paints `D_s(n) = sum_h K_s(z_h) * DM_h(n)` for each survey.
Because DM and the cross-correlation are linear, crossing this map with y is
exactly the weighted average of the individual-source-plane full-sky cross
signals. It is the expected signal for uniform, independent source directions,
not a particular realization of 71 or 31 sparse FRBs. Source counts enter the
redshift kernel; no sparse-sampling covariance is manufactured.

The current baseline uses **equal source weights**. Takahashi's Eq. 28 also
uses Galactic-DM and host-variance terms; those individual Galactic DM values
are not available in the extracted table. The preparation stage accepts a
future `--source-weights` CSV with columns `frb,weight`. A change in these
weights requires repainting the kernel maps, not relabelling the old curves.
No host DM is added merely because host uncertainty would enter an estimator
weight.

Each halo column already contains observer-frame `1/(1+z_h)` dilution. There
is no second source-redshift factor. Subtracting the map monopoles is equivalent
to averaging the per-source-redshift, halo-only mean-subtracted fields. The
observed total-DM mean is not subtracted from a halo-only map.

Full-sky HEALPix transforms at NSIDE4096, lmax8192, with three iterative
corrections yield unbinned **C_l(y,DM)**, including the halo clustering present
in the lightcone. No white-noise or FRB shot-noise subtraction is appropriate
for these complete-map cross spectra.

The angular correlation is

`w(theta) = sum_{l>=1} (2l+1)/(4*pi) * C_l(y,DM) * B_l(y) * P_l(cos(theta))`.

Only y receives a Gaussian beam: 10 arcmin FWHM for Planck and 1.6 arcmin for
ACT, as in the paper. The beam is applied **once**, not squared; localized FRB
DMs have no matching 10-arcmin beam. Native map pixelization is retained without
an ad hoc pixel-window deconvolution.

Model points are averaged exactly over solid angle in the 12 published
logarithmic annuli from 1 to 1000 arcmin (Delta log10 theta=0.25), using Legendre
antiderivatives. Digitized, horizontally offset marker positions are used only
to display the observed points, not as physical bin edges.

## Interpretive limits

1. This is a redshift-distribution-matched **mean halo-only prediction**, not a
   full reproduction of the masked/noisy observational estimator. No Planck
   or ACT instrumental-noise realizations, survey-window random catalogues, or
   survey-specific integral-constraint correction have been applied.
2. Equal source weights are an approximation to the published inverse-variance
   weights. HalfDome cosmology and density normalizations are left unchanged.
3. The plotted observational error bars are correlated between bins. No
   chi-square, detection significance, best-fit amplitude, or formal goodness
   of fit is inferred from their diagonal sizes. The numerical comparison CSV
   reports absolute model-minus-data differences, not percentages divided by
   noisy near-zero observed correlations.
4. The lmax diagnostic compares 6144 with 8192. It cannot establish convergence
   beyond the available map resolution, particularly for ACT's smallest angles.
   The paper's excluded scales (<10 arcmin Planck; first bin for ACT) are shaded.
5. Preferred Lee22 + Duffy remains a normalization/extrapolation diagnostic.
   Finite 3R200c integration fixes an outer-column divergence, but does not fix
   the excessive gas-equivalent mass inside R200c. No arbitrary renormalization
   is introduced to make this curve agree with data.
6. Figures 14 and 15 vary survey masks and component separation. This calculation
   does not claim to reproduce those map-level systematics. Medlock/Nagai's
   separate CHIME DM-cut sample likewise needs its own selection kernel.

## Reproduction and locations

Main notebook: `../FRB_tSZ_Takahashi_comparison.ipynb`.

Local output root:
`/home/cbllover/HalfDome/frb_map_generation/outputs/takahashi_cross_comparison_20260913`.

Cluster output root:
`/lustre/work/kristero10/frb_data/takahashi_cross_20260913_v1`.

Uploaded source snapshot:
`/home/kristero10/HalfDome_kSZ/frb_map_generation/cluster_tests/takahashi_cross_20260913_v1`.

DM map jobs (mini, 26 CPUs, 48 GB, 23:59):

- 597070.idark: Battaglia16.
- 597071.idark: Lee22 legacy, no concentration.
- 597072.idark: preferred Lee22 + Duffy08, diagnostic.

The maps completed, but these initial jobs correctly rejected an incomplete tSZ
transfer at the checksum check. No spectra from that incomplete file were used.
The original local tSZ map and previous scientific outputs remain untouched.

The final tSZ input was instead regenerated on the cluster with the same
pressure parameters, physical mass convention and projected 4R200c aperture:

- 597074.idark: tSZ repaint; moved to mini2 when mini lacked 26 free CPUs;
  completed at 2026-09-13 13:54:07 UTC, exit 0.
- 597075.idark: Battaglia16 cross spectra; completed 13:56:56 UTC, exit 0.
- 597076.idark: Lee22 legacy cross spectra; completed 13:57:04 UTC, exit 0.
- 597077.idark: Lee22 preferred cross spectra; completed 13:56:06 UTC, exit 0.

The final three spectra jobs ran in mini and used the same verified tSZ FITS,
SHA256 `9ea83f559df8ad9100a2c11d772e0a91abbce7a2645caf5a414eaee67957f311`.
Its mean y is 1.041576278625117e-6, compared with 1.0415762787005558e-6
in the previous local map (relative difference about 7.2e-11). This verifies the
mean, not every pixel; all final model comparisons share the new map exactly.

All six DM maps scanned all 85,224,251 catalogue rows and included all
74,907,259 halos with 0 < z <= 2.148. Their physical mass range is
6.822642518e12 to 3.785236256e15 Msun. Each halo is weighted separately by the
source fraction behind it. Maps and caches remain on the cluster; the small
spectra, logs, kernel files and provenance were downloaded into `cluster_results/`.
The downloaded archive checksum was verified:
`2e553d983066e2cbefaafdab971cfbb592fcc754bd9f97f09d0677db3e01a66e`.

Generate source kernels (do not change these beneath an existing map run):

```bash
python frb_map_generation/compare_halfdome_takahashi.py prepare
```

The completed small `spectra/` products are now downloaded into the local
output root. Regenerate plots, or run all cells of the notebook:

```bash
/home/cbllover/miniconda3/bin/python \
  frb_map_generation/compare_halfdome_takahashi.py plot
```

The generated annular CSV and `plots/comparison_summary.json` contain the actual
comparison and numerical sensitivity values.

## Completed comparison

These are descriptive comparisons, not fits. Values below are in units of
1e-5 pc cm^-3, averaged over the displayed annulus. Observations are approximate
digitizations; their correlated errors are shown in the plots.

| Survey | Annulus [arcmin] | Observed | B16 | Lee22 legacy | Lee22 preferred diagnostic |
| --- | --- | ---: | ---: | ---: | ---: |
| Planck | 10-17.78 | 4.28 | 1.70 | 6.63 | 22.65 |
| Planck | 17.78-31.62 | 2.48 | 1.00 | 3.88 | 16.64 |
| Planck | 56.23-100 | 1.21 | 0.247 | 0.859 | 6.05 |
| ACT | 10-17.78 | 5.37 | 1.71 | 6.62 | 23.23 |
| ACT | 17.78-31.62 | 2.57 | 1.01 | 3.86 | 16.84 |
| ACT | 56.23-100 | 2.58 | 0.250 | 0.863 | 6.02 |

Battaglia16 is below the Planck central measurements over approximately
10-100 arcmin. The current no-concentration Lee22 signal is about 3-4 times
higher than B16 over this range; it is closer in some intermediate-angle bins,
but overshoots smaller-angle measurements. ACT is noisier, so visual proximity
must not be described as a statistical model preference. Preferred+Duffy is
much higher and retains its independent high-mass normalization warning.

The two density models do not use a shared fitted pressure-density relation:
pressure is deliberately held at B12 while the electron density changes.
Consequently, larger weighted electron columns in the pressure-bearing halos
can increase the cross signal even if a different model has a denser innermost
core in selected halos. The correlation also includes outer gas, halo mass and
redshift weighting, and halo clustering. This cross comparison is not itself
a test of a single central density value.

Direct-column versus radius-cache maximum relative discrepancies on 600
extended-redshift test points per model are 0.0337% (B16), 0.0672% (Lee22 legacy),
and 0.1296% (preferred). All pass the 1% cache threshold. The 6144-versus-8192
lmax comparison changes ACT annular means by at most 0.2024%, 0.0758%, and
0.0449%, respectively (including the first, beam-excluded bin). Planck changes
are below 1e-8%. These particular numerical errors cannot explain factors of
3-4 between models, but the checks do not validate unresolved modes or the
preferred model's astrophysical extrapolation.

Saved plots:

- `plots/halfdome_vs_takahashi_fig13.png`: B16 and Lee22 legacy versus Planck/ACT.
- `plots/halfdome_vs_takahashi_preferred_diagnostic.png`: all three density models.
- `plots/observed_source_redshifts_and_kernels.png`: source counts and weights.
- `plots/angular_lmax_sensitivity.png`: numerical truncation check.

## Files added for this comparison

- `paint_halfdome_observed_source_dm_maps.jl`: observed-redshift kernel painting;
  reuses the existing density/cosmology/radius implementation without modifying it.
- `compare_halfdome_takahashi.py`: kernel preparation, cross spectra, beam and
  angular-bin transforms, plots, machine-readable differences, self-tests.
- `run_halfdome_takahashi_cross.pbs`: per-model cluster workflow.
- `run_takahashi_tsz_input.pbs`: full-lightcone tSZ regeneration using the existing
  painter and the same physical pressure settings.
- `../FRB_tSZ_Takahashi_comparison.ipynb`: readable analysis entry point.
- This report. Existing notebooks, XGPaint, profiles and previous results are
  unchanged by this comparison.
- Temporary helpers: `../tmp/takahashi_transfer_chunks.py` for the abandoned
  slow-transfer attempt and `../tmp/test_takahashi_plot_pipeline.py` for a clearly
  labelled synthetic renderer test. Synthetic spectra are not used in the
  scientific outputs. Partial transfers are isolated under the cluster run's
  `inputs/` directory, not mixed with old data.

Checks already run: Python angular normalization/annulus tests locally and on
the cluster; Julia source-screen averaging and boundary tests on Julia1.12 and
cluster Julia1.6; Python compilation, PBS shell syntax, notebook schema.
The production jobs additionally validate the extended-z radius caches,
catalogue completeness, map provenance and cross-spectrum Cauchy-Schwarz bound.
The final notebook was executed locally after downloading the real spectra:
all three code cells completed, zero error outputs, and all four scientific
PNGs were inspected. New source/notebook files are owned by `kn18001` so they
can be edited and saved from the local Jupyter session.
