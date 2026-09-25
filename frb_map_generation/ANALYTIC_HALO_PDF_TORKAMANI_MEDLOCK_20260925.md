# Analytical DM PDFs (Torkamani+26, Medlock+24) against the TNG and HalfDome PDFs

Date: 2026-09-25. Motivated by Torkamani et al. 2026 (Baryonification III, arXiv:2601.18784) and Medlock et al.
2024 (CAMELS, arXiv:2403.02313). Code: `frb_map_generation/analytic_pdf_torkamani2026/`; outputs (not git-tracked):
`frb_map_generation/outputs/analytic_halo_pdf_20260925/{tables,analysis,plots}`.

## 1. What "the analytical PDF" is in each paper, and what it can be compared with

- **Torkamani+26** derive the one-point DM PDF from the halo model: halos from the mass function, each with its
  projected gas profile, combined as a compound Poisson process (their eqs. 29-31) plus a halo-clustering term from
  the peak-background split and Limber (eq. 46). This is a prediction for any gas profile and any halo selection, so
  it can be evaluated for exactly what the previous TNG and HalfDome PDFs measure: **halo-only DM, gas inside each
  halo's R200c sphere, rays to z_s = 1, M200c windows, 299 log bins over 0.1-3e4 pc cm^-3, normalized over DM > 0**.
- **Medlock+24** use the closed-form Macquart+20 PDF p(Delta) = A Delta^-3 exp[-(Delta^-3 - C0)^2 / (18 sigma^2)],
  Delta = DM/<DM>, sigma = F z^-1/2 (their eqs. 7-8), with F from CAMELS (IllustrisTNG fiducial 0.182, SIMBA 0.119,
  Astrid 0.229, at z = 0.5). It describes the **total** cosmic DM (IGM + halos) and has no halo-only form, so it is
  compared with TNG300's total LSS DM (Konietzka's `DMall_IGM_halo`), together with the log-normal approximation
  that Torkamani+26 Sec. 4.4 tests and their published BFC curve (Fig. 6, z = 0.7). HalfDome has no IGM term and
  cannot enter this comparison.

## 2. Implementation of the Torkamani+26 PDF

- `tabulate_halfdome_sphere_profiles.jl` (Julia 1.12.2, XGPaint dev checkout): the per-halo DM of the HalfDome sphere
  products, DM(x) = 2 R200c/(1+z) int_0^sqrt(1-x^2) n_e dl, x = b/R200c, for Battaglia16 (XGPaint parameters and
  normalization) and Lee22 no-c XGPaint-native (the published-figure variant: `normalization=xgpaint_ne2d`,
  `n0_pivot=mcut`, shape clip 10^14.8/h), evaluated with `halo_electron_density_m3` of `lee2022_frb_dm_profile.jl`
  on 700 masses (log M = 9.005-15.995, step 0.01) x 100 redshifts (0.005-0.995, step 0.01) x 200 radii. It
  reproduces the production `chord_dm_pc_cm3` to 4e-16.
- `halo_model_dm_pdf.py`: Tinker08 M200c mass function (agrees with the `hmf` package to 1-2 % over 1e10-1e15 Msun,
  z = 0-1), Tinker10 bias, CAMB linear P(k) normalized to sigma8, linear growth, and the BFC hot-gas profile with
  Torkamani's TNG-calibrated Table 1 parameters (Schneider+25 profile: truncated NFW with eps = 4 + 0.5 nu and
  Dutton-Maccio14 concentration for M_tot, rho_hga normalized to f_hga M_tot; fully ionized electrons
  n_e = 0.88 rho/m_p, X_H = 0.76). Instead of evaluating Y(M, z, lambda) on a lambda grid, each halo's disc is
  deposited on a linear DM lattice (h = 0.002 pc cm^-3, 2^24 cells, to 33554 pc cm^-3), the area between
  consecutive impact-parameter samples spread uniformly over the DM interval it spans; the compound-Poisson PDF is
  one FFT, the clustering term one FFT per Delta z = 0.05 group. The DM = 0 delta is removed and the PDF normalized
  over 0.1-3e4 like the simulation histograms.
- Engine checks: a 2-million-ray Monte Carlo of a toy compound Poisson process agrees within its noise and the
  zero-hit fraction to 1e-5; in every run the mean DM from the FFT lattice equals the direct sum over halos to
  <= 3e-4; BFC chord DMs agree with direct `scipy.quad` of the 3-D profile to 0.2 %.
- Runs (`run_analytic_halo_pdfs.py`): (a) validation, HalfDome cosmology, halos above the catalogue floor
  7.327e12 Msun, abundance from the HalfDome lightcone itself (counts on the same (z, M) cells / 4 pi) or Tinker08;
  (b) TNG, Planck15 cosmology, Tinker08 down to 1e9 Msun (TNG's catalogue reaches 1e9: its 1e9-1e14 window already
  has 100 % hits), B16 and Lee22 extrapolated below their fitted masses, and BFC.

## 3. Validation: the analytical PDF with HalfDome's own halos reproduces the HalfDome rays

Same halos (catalogue counts), same profiles, same geometry; the rays carry the catalogue's real clustering. HalfDome
per-hit-ray means and medians are taken from the 299-bin histograms (bin centres / interpolated CDF), like the
analytic ones; means over all rays are the exact ray means.

| window | product | hit % | mean DM, all rays | mean, hit rays | median, hit rays |
|---|---|---:|---:|---:|---:|
| all (>= 7.3e12) | HalfDome rays, B16 | 64.9 | 75.7 | 116.7 | 72.6 |
| | analytic B16, clustered (unclustered) | 64.3 (65.5) | 75.70 | 117.8 (115.6) | 72.9 |
| | HalfDome rays, Lee22 | 64.9 | 125.3 | 193.1 | 98.6 |
| | analytic Lee22, clustered (unclustered) | 64.3 (65.5) | 126.7 | 197.2 (193.4) | 99.2 |
| 1e13-1e14 | HalfDome rays, B16 | 53.7 | 49.9 | 93.0 | 60.7 |
| | analytic B16, clustered (unclustered) | 53.6 (54.3) | 49.92 | 93.2 (91.9) | 60.5 |
| | HalfDome rays, Lee22 | 53.7 | 74.5 | 138.6 | 83.5 |
| | analytic Lee22, clustered (unclustered) | 53.6 (54.3) | 74.92 | 139.9 (138.0) | 84.0 |

The mean over all rays does not depend on clustering and matches to 0.0-1.1 % (the 120k-ray sampling error of the
Lee22 mean is ~0.7 %). Tinker08 instead of the catalogue counts changes the numbers by < 1.5 % (the HalfDome masses
were resampled to Tinker08). Bin by bin, the expected counts of the analytical PDF against the HalfDome histograms
give a Poisson chi^2 per bin of 0.82 / 1.01 (all halos, B16 / Lee22) and 0.93 / 1.07 (1e13-1e14), over 145-179 bins
with >= 10 expected rays, clustered or not: the analytical PDF reproduces the ray-traced HalfDome PDFs to within the
sampling noise of 120,000 rays (median bin offset <= 3 %, rms 3-4 % at 10-100 pc cm^-3). The clustering term
changes hit fractions and means by 1-2 %, and the HalfDome rays lie between the clustered and unclustered values
(Lee22 all-halo mean: 0.2 % below the unclustered one).
**So the Torkamani+26 formalism, fed the same halos and profiles, is equivalent to the HalfDome ray tracing; any
difference from TNG below is the gas model and the halo population, not the method.**

## 4. The analytical PDFs against TNG (Planck15, Tinker08 down to 1e9 Msun, clustered)

Per-ray statistics of rays with DM > 0 (`analysis/analytic_halo_pdf_summary.csv`):

| window | TNG hit % / mean / median | B16 | Lee22 no-c | BFC (TNG-calibrated) |
|---|---|---|---|---|
| all | 100 / 177.9 / 129.5 | 100 / 166.5 / 129.6 | 100 / 155.7 / 71.4 | 100 / 94.5 / 50.4 |
| 1e13-1e14 | 52.6 / 138.6 / 80.8 | 53.9 / 93.5 / 60.7 | 53.9 / 139.9 / 84.0 | 53.8 / 81.6 / 53.5 |
| 1e12-1e14 | 90.6 / 128.8 / 77.3 | 92.6 / 102.6 / 74.0 | 92.6 / 110.5 / 60.4 | 92.6 / 64.2 / 37.5 |
| 1e10-1e13 | 99.99 / 147.1 / 114.4 | 100 / 87.6 / 76.7 | 100 / 33.5 / 24.6 | 100 / 23.3 / 19.1 |
| 1e10-1e12 | 99.98 / 99.8 / 80.7 | 99.99 / 42.9 / 39.3 | 99.99 / 6.5 / 5.5 | 99.99 / 7.7 / 7.3 |

Percent difference 100 (p_analytic - p_TNG)/p_TNG, median (rms) over bins with >= 10 TNG rays:

| window | model | DM 1-10 | 10-100 | 100-1000 |
|---|---|---:|---:|---:|
| 1e13-1e14 | B16 | +22 (30) | +25 (28) | -35 (52) |
| | Lee22 no-c | -12 (23) | -1.4 (3.9) | +3.4 (8.2) |
| | BFC | +32 (38) | +39 (39) | -55 (61) |
| 1e12-1e14 | B16 | -29 (37) | +7.8 (9.9) | -26 (48) |
| | Lee22 no-c | +44 (50) | +14 (23) | -16 (16) |
| | BFC | +127 (138) | +39 (57) | -67 (69) |
| all | B16 | -95 (93) | -26 (48) | -4 (23) |
| | Lee22 no-c | +1450 | +123 | -35 (39) |
| | BFC | +1415 | +212 | -59 (57) |

- **Geometry and abundance are right**: the analytic hit fractions (53.9 % and 92.6 %) track TNG's (52.6 % and 90.6 %)
  in the two windows where both hold the same halos, so the remaining differences are gas-model differences.
- **1e13-1e14 (the like-for-like window)**: Lee22 no-c (TNG n_e fits, XGPaint-native normalization) matches TNG to
  a few per cent from 3 to 1000 pc cm^-3 analytically, as it does in the HalfDome rays. Battaglia16 and the
  TNG-calibrated BFC profile both put too little gas in the cores (-35 % and -55 % above 100 pc cm^-3; mean per hit
  ray 0.67 and 0.59 of TNG).
- **All halos**: the analytical model can include TNG's 1e9-7.3e12 Msun halos that HalfDome's catalogue lacks
  (every ray now hits a halo, as in TNG). With Battaglia16 extrapolated to 1e9 Msun it reproduces TNG's median
  (129.6 vs 129.5) and mean to -6 %, but not the shape (too few rays below 30 pc cm^-3); Lee22 and BFC give low-mass
  halos almost no gas inside R200c and pile the rays up at 1-30 pc cm^-3.
- **Below 1e12 Msun** TNG's within-R200 DM is large: 1e10-1e12 halos alone give 99.8 pc cm^-3 per ray (~10 per
  crossing; tail to ~2000 pc cm^-3). No smooth hot-gas profile does this (B16 extrapolated 42.9, BFC 7.7, Lee22 6.5;
  BFC puts 0.04-0.07 f_b of hot gas inside R200c there). A plausible source is the dense, star-forming and cold gas
  of the galaxies themselves, which the hot-gas profiles exclude; this was not checked against the TNG cells.

## 5. The BFC profile: implementation check and a TNG-internal tension

My Table 1 implementation was compared with the vector curves of Torkamani+26 Fig. 5 (TNG gas profiles and the BFC
fit at z = 0.5, extracted from the arXiv source `figures/profile05.pdf`). The curves read as physical densities (read as
comoving they would imply more than f_b of gas inside R200c). My profile matches the plotted BFC curve to 5 % in the
lowest bin and to 5-30 % at R200c in the others (depending on which mass inside the 0.5-dex bin the curve stands for),
but is up to ~1.6-2x higher at 0.1 R200c in the two highest bins; the enclosed hot gas inside R200c is 1.1-1.7x that
of the plotted curve. Table 1 and Fig. 5 may not be exactly the same parameter set; with the plotted curves the
BFC-TNG deficit below would only be larger.

Integrating the plotted **TNG** gas profiles inside R200c gives M_gas(<R200c) = 0.21, 0.30, 0.45, 0.57, 0.57 f_b in
the bins log(M h) = 12.5-13 ... 14.5-15, while Konietzka's within-R200 DM corresponds to roughly 0.85-0.95 f_b of free
electrons for 1e13-1e14 halos (`LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md` Sec. 5.3). So the two published fits of
TNG gas profiles bracket TNG's within-R200 DM: the BFC fit to the gas density lands at 0.59x, the Lee22 fit to the
volume-weighted mean n_e (eq. 9 as printed) at 1.66x. Stacking conventions (median vs volume-weighted mean, hot vs
all gas, satellites and clumps) matter at this level; the TNG DM catalogue is the only one of these three built from
the actual electron column.

## 6. Closed-form total-LSS PDFs (Medlock+24 / Macquart+20, log-normal) against TNG300

`analytic_functions_total_lss.py`, TNG300 total LSS DM (120,000 rays) at z = 0.5, 0.7, 1.0; <DM> is TNG300's own, so
only shapes are tested.

- TNG300's fluctuation parameter F = sigma_DM sqrt(z) (Medlock's recipe, sigma_DM = std/mean) is 0.245, 0.234, 0.223
  at z = 0.5, 0.7, 1.0, against Medlock+24's CAMELS-TNG 0.182 at z = 0.5: TNG300 fluctuates ~35 % more, in line with
  Medlock+24's own caveat that the 25 Mpc/h CAMELS boxes miss large-scale structure.
- The Macquart form with alpha = beta = 3 does not have TNG300's shape: its deviation within +-1 sigma of the mean is
  46-61 % with F = 0.182, 72-84 % with TNG300's own F, and 57-86 % even with sigma fitted to TNG300 by maximum
  likelihood (too peaked, ~40 % too few rays just above the mean, at about 1.1-1.5 <DM>, and too heavy a high tail at z = 1).
- The log-normal with TNG300's mean and variance deviates by 36-45 % within +-1 sigma and grossly in the low tail
  (it puts rays far below TNG300's minimum DM).
- The published BFC curve (Torkamani+26 Fig. 6, z = 0.7) matches TNG300 to 5 % within +-1 sigma (20-40 % at the ends
  of the digitized range), as the paper states.

## 7. Figures and files

- `plots/analytic_halo_pdf_vs_tng_halfdome.{png,pdf,svg}`: the previous publication panels (all halos, 1e13-1e14) with
  the analytical PDFs added. Rows: PDFs; percent difference to TNG (analytic thick dashed, HalfDome rays thin solid);
  analytic with HalfDome's own halos against the HalfDome rays (clustered dashed, unclustered dotted).
- `plots/analytic_halo_pdf_vs_tng_low_mass_windows.{png,pdf,svg}`: 1e10-1e12, 1e10-1e13, 1e12-1e14 (analytic vs TNG;
  HalfDome rays in 1e12-1e14).
- `plots/analytic_functions_total_lss_vs_tng.{png,pdf,svg}`: Sec. 6.
- `analysis/analytic_halo_pdf_bins.csv` (every bin, both comparisons, clustered and unclustered model),
  `analysis/analytic_halo_pdf_summary.csv`, `analysis/analytic_summary.json`, per-run `analysis/*.npz` (binned PDFs
  and the CDF on a fine log grid), `analysis/analytic_functions_total_lss_{bins,summary}.csv`.
- `tables/halfdome_sphere_profiles.h5` (Julia tabulation), `tables/bfc_tng_fiducial_sphere_profiles_planck15.npz`,
  `tables/halfdome_catalog_counts_m200c_z.npz` (lightcone counts, z < 1, M >= 7.327e12 Msun).

Caveats: the TNG runs use the HalfDome-cosmology profile tables (rho_crit and D_A differ from Planck15 by < 1 %);
Tinker08 is extrapolated below ~1e11 Msun and B16/Lee22 far below their fitted masses; TNG's masses are taken as
physical M200c; the Limber clustering term is Torkamani's pencil-beam form; Lee22 cusps of the rarest >= 1e15.5 Msun
cells above the 33554 pc cm^-3 lattice were dropped (<= 7e-8 crossings per ray, recorded per run).
