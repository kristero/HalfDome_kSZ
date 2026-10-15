# Completed PDF and 3R200c kSZ comparison

Completed locally on 2026-09-22. No cluster job was submitted.

## Selections and products

- FRB PDFs: **1R200c**, physical 1e13 <= M200c/Msun < 1e14, sources at z=1,
  120,000 random sightlines, seed 42. Same retained rays and Ralf arrays as the
  completed prior comparison; each normalized-model PDF is histogrammed afresh.
- Restricted kSZ: **3R200c**, the same physical mass bin, foreground z<=1,
  20,866,344 halos. Full kSZ: all **85,224,251** HalfDome catalogue halos,
  6.55711024486e12 <= M200c/Msun <= 3.78523625595e15,
  0.00214159489 <= z <= 3.855427742.
- Finite three-dimensional spheres, including the line-of-sight limit, are
  used for both density models and both CLASS-SZ mass/redshift selections.
- Figures: `dm_pdf_ralf_lee22_normalizations.pdf`,
  `ksz_3r200c_restricted_and_full.pdf`, and
  `ksz_3r200c_full_websky_reference.pdf`. PNG and SVG counterparts are retained.
- Exact rays, maps, spectra, unbounded percentages, histogram statistics,
  configuration, source hashes, and the extracted Websky reference are saved.

## Normalization physics

Lee22 defines the fitted electron-density reference as

```text
n_e,paper(r) = n0 g(r/R200c) * 200 f_b rho_crit / (X_H m_p),  X_H=0.76.
```

The previous PDF-matching XGPaint wrapper interprets 200 n0 g as gas density
and converts it to electrons with 0.9/m_per_e. The previous/paper amplitude
ratio is 0.6016316602183194. The paper normalization is therefore
**1.662146569276 times** the previous density,
DM, optical depth, and kSZ map amplitude. Its kSZ power is
**2.762731217758 times** the previous power.
Changing aperture does not change this constant normalization ratio.

The preceding Figure 5 audit established the paper normalization independently.
Good previous PDF agreement with Ralf did not establish that this density
normalization matched Lee22. Both versions are preserved explicitly here.

## PDF results

Mean halo DM over all rays [pc cm^-3]:

| Sample | Previous Lee22 | Paper Lee22 |
|---|---:|---:|
| HalfDome | 73.594 | 122.324 |
| TNG300-1-Dark | 70.919 | 117.878 |
| Ralf IllustrisTNG300 | 72.908 | 72.908 |

Mean absolute percentage difference relative to Ralf's positive-DM PDF:

| Sample | Previous MAE | Paper MAE | Previous weighted MAE | Paper weighted MAE |
|---|---:|---:|---:|---:|
| HalfDome | 10.30% | 77.05% | 5.57% | 36.81% |
| TNG300-1-Dark | 10.75% | 78.03% | 5.69% | 35.69% |

MAE uses bins with at least 10 counts in both compared histograms. The valid
bin sets differ (HalfDome: 157 previous / 153 paper; TNG-Dark: 158 / 149).
Weighted MAE weights these bins by Ralf's probability mass. These are
descriptive differences, not covariance-aware fit probabilities or mean-DM
uncertainties. PDFs condition on DM>0; zero-hit fractions are saved in JSON.
Signed residuals display +/-20% with no arrows; CSV values are not clipped.

## kSZ results

The following D_ell values are in microkelvin squared and averaged over
2950 <= ell < 3050 with weights 2ell+1:

| Model | Restricted | Full HalfDome |
|---|---:|---:|
| Lee22 previous | 0.42402 | 0.88855 |
| Lee22 paper | 1.17145 | 2.45482 |
| Battaglia16 painted | 0.26830 | 0.63886 |
| CLASS-SZ Battaglia16 | 0.27427 | 0.68505 |

The published **Websky halo-only** band is **0.50700 microkelvin squared**.
It is a reference with a different selection/compensation, not a matched error
denominator. At ell~3000, the matched B16 painted/CLASS signed differences are
-2.18% restricted and
-6.74% full. At ell~5000 they are
1.76% and
-1.76%, respectively.

CLASS-SZ here is **Battaglia16 AGN 1-halo + 2-halo**, not an independent
implementation of Lee22. Different gas models can have different power.
The full CLASS calculation was used with the established version 0.1.70.post40.

At M200c=1e13 Msun and z=0.5, Ne(<3R200c)/Ne(<R200c) is 4.2591 for Lee22 and
2.5504 for B16. At 1e14 Msun these ratios are 1.7336 and 2.2311. The additional
outer gas in low-mass Lee22 halos helps explain the stronger large-angular-scale
power. These are per-halo electron-count ratios, not power-spectrum ratios.

## Scientific comparison limits

1. The new painting and CLASS calculations share h=0.68, Omega_m=0.31,
   Omega_b=0.049, sigma8=0.8159, n_s=0.9667 to retain the FRB contract.
   The underlying HalfDome simulation uses h=0.6774, Omega_m=0.3089,
   Omega_b=0.0486; this small difference remains.
2. Lee22's restricted selection includes low-mass extrapolation. Full HalfDome
   additionally extrapolates in mass and above z=2. The previous high-mass
   shape freeze at 10^14.8/h Msun is retained, while amplitude retains its mass
   dependence. **3R200c exceeds the fitted radial range ending near 1.34R200c.**
3. Stein et al. (2020) select M200c>1e13 Msun and theta200c>0.5 arcmin for the
   halo kSZ component, out to z<4.6. They subtract a uniform halo counterterm
   of overdensity 3. Our finite-sphere halo maps are uncompensated. The paper's
   orange halo-only curve was extracted from vector markers, not from its
   total or field curve. No amplitude was fitted to match it.
4. Full-sky NSIDE=4096 maps use pixel-centre sampling, no beam, and no pixel
   window deconvolution. Spectra use lmax=6000 and three anafast iterations.
   No additional higher-resolution spectrum convergence run was performed.
5. CLASS is an ensemble halo-model calculation; HalfDome uses one halo and
   bulk-velocity realization. Gas scatter, nonsphericity, internal gas motions,
   diffuse electrons, and reionization are outside this halo-only calculation.
   The remaining B16 residual is not uniquely attributable to profile errors.

## Verification and resources

- Independent physical chord comparison: maximum relative difference
  4.25e-14. Direct all-pixel versus threaded painter checks cover polar halos,
  the longitude seam, overlap, and restricted routing; maximum relative L2
  difference is 9.76e-7.
- 800 direct quadrature/cache checks per model: maximum Lee22 interpolation
  difference 0.4780%; B16 0.003721%. The finite edge is handled analytically.
- All four maps are finite; halo counts, shapes, cutoff, units, CLASS bounds,
  profile checksum, positive spectra, and map-variance consistency passed.
- All three final PDFs were rendered and visually inspected. Source checksums
  and a SHA256SUMS manifest accompany this report.
- Painter: 10:39.83, peak 9.306 GiB.
  CLASS: 1:09.10, peak 4.399 GiB.
  Four spectrum transforms: 2:27.47, peak
  5.107 GiB. These are per-process peaks.
- No cluster work. Temporary installers/render checks were removed; the four
  production maps (~6 GiB) and chord cache are retained for reproducibility.

## Source files created

All new analysis code is in `frb_map_generation/lee22_pdf_ksz_comparison/`:
`compare_dm.py`, `prepare_ksz.py`, `paint_ksz.jl`, `compute_ksz.py`,
`extract_websky.py`, `plot_ksz.py`, `aperture_diagnostics.jl`,
`finalize_results.py`, and `README.md`. The production Lee22 profile and the
previous result directories were not modified by this comparison.

References: [Stein et al. 2020](https://arxiv.org/abs/2001.08787),
[CLASS-SZ documentation](https://class-sz.readthedocs.io/en/latest/), and the
preceding local `lee22_fig5_audit_20260922/REPORT.md`.
