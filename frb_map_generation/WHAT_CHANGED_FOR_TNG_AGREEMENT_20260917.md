# What was changed to bring the HalfDome halo DM onto TNG: XGPaint side and Lee22 side

Date: 2026-09-17. Branch `cluster`; commits 346895d .. 7b2692c plus the publication figure commit.
Publication figure: `frb_map_generation/outputs/publication_comparison_20260917_final/plots/halo_pdf_publication_b16_lee22_tng.{pdf,png,svg}`.

> **Scientific qualification, 2026-09-22:** The successful `xgpaint_ne2d`
> normalization below is an empirical variant, lower than the paper-defined
> Lee22 electron density by a factor 0.60163166. The claim that the unusual
> reference `n200` proves an equation-9 error is withdrawn. The independently
> extracted Figure 5 agrees with the paper-normalized no-c implementation to
> 2-3%, while this PDF-matching variant is about 38% low. See the
> [Figure 5 audit](outputs/lee22_fig5_audit_20260922/REPORT.md). Historical values,
> production code, and all previous PDF outputs remain unchanged.

## 1. The result being explained

Mean halo DM of rays with DM > 0, rays to z_s = 1, M200c window 10^13 - 10^14 Msun (the only window in which the
TNG within-R200 catalogue and the HalfDome catalogue contain the same halos; hit fractions 52.6% and 53.7%):

| Product (all: 120k rays, NSIDE 4096, seed 42) | mean DM [pc cm^-3] | ratio to TNG |
|---|---|---|
| TNG within R200 (Konietzka) | 138.6 | 1 |
| Battaglia16, previous: projected 1 R200c disc, XGPaint line of sight to 1e5 R200c | 126.9 | 0.92 (with a hard floor at 25 pc cm^-3 and no low tail) |
| Battaglia16, gas inside the R200c sphere | 92.9 | 0.67 |
| Lee22 no-c, previous: eq. 9 as printed, n0 pivot 1e14 Msun, projected 1 R200c | 232.4 | 1.68 |
| Lee22 no-c, eq. 9 as printed, M_cut pivot, inside the sphere | 230.4 | 1.66 |
| Lee22 no-c, times Omega_b/Omega_m (the retracted "correction"), inside the sphere | 36.4 | 0.26 |
| Lee22 no-c, times X_H(1+X_H)/2 (ionized electron count), inside the sphere | 154.1 | 1.11 |
| Lee22 no-c, times X_H^2 (n200 with X_H in the numerator), inside the sphere | 133.1 | 0.96 |
| **Lee22 no-c, XGPaint-native (P0 = 200 n0, XGPaint's ne2d), inside the sphere** | **138.6** | **1.00** |

The Lee22 chain factorizes: 232.4 (previous) x 1.42 (n0 pivot) x 0.70 (sphere instead of projected) x 0.602 (XGPaint-native
electron conversion instead of eq. 9 as printed) = 139. The medians are 80.8 (TNG) against 83.5, and the fractions of hit
rays below 3 / 10 / 30 pc cm^-3 are 0.57 / 4.5 / 20.1 % against 0.39 / 4.0 / 19.5 %.

## 2. Changes on the XGPaint side

No line of XGPaint (`/home/kn18001/.julia/dev/XGPaint`, v0.4.0) was modified. What changed is how its profile is used.

### 2.1 Truncation surface: sphere instead of disc-plus-infinite-line-of-sight

XGPaint's projected profile integrates the 3-D density along the whole ray (`_nfw_profile_los_quadrature`,
`zmax = 1e5` R200c, `src/profiles_y.jl:100`) and painting is cut only on the sky (`compute_θmax`, 4 R200c by default,
`src/profiles.jl:278`; the HalfDome generator used its own 3 R200c and later 1 R200c disc). A ray at the disc edge
therefore received the entire column at that impact parameter, 22 pc cm^-3 for a floor-mass halo at 1 R200c, and rays
just outside received nothing. TNG's within-R200 catalogue sums gas cells inside the R200 sphere, so its DM goes to zero
continuously for grazing rays.

Change (repository code, `lee2022_frb_dm_profile.jl` and `generate_halfdome_z1_dm_mass_windows.jl`, flag
`--halo-boundary=spherical`): the line of sight is limited to the chord inside the R200c sphere,
`DM = (2 R200c/(1+z)) int_0^L n_e(sqrt(x^2 + l^2)) dl`, `L = sqrt(1 - x^2)`, zero outside. The 3-D density comes from
XGPaint's own `get_params` and `generalized_nfw` with XGPaint's `f_b rho_crit(z)` normalization and `ne2d` electron
count (reproduced in `Battaglia16DensityDMProfile`, verified against XGPaint's projected `HaloDMProfile` to 1e-5). To keep
the interpolation cache smooth, the cached quantity is the chord-mean column `g = DM/(2L)` and the exact chord factor is
applied per ray. Effect: the low-DM floor disappears (first populated bin 0.2 instead of 25 pc cm^-3), the tail fractions
match TNG, and the mean per hit ray drops by a factor 0.70-0.73 for both profiles because 27% of the projected column at
1 R200c is gas outside the sphere.

### 2.2 Lee22 as an XGPaint profile

`Lee2022XGPaintDMProfile <: XGPaint.AbstractFRBProfile` supplies only `XGPaint.get_params` (and `object_size`). XGPaint's
`rho_2d`, `ne2d`, `compute_DM`, interpolator and painter run unchanged on it (`--dm-profile=lee2022_xgpaint`).
`get_params` returns the Lee22 parameters in XGPaint's gNFW convention, `beta = alpha beta' - gamma` (XGPaint's exponent
is `-(beta+gamma)/alpha`, Lee22's is `-beta'`), and `P0 = 200 n0`. The full mass and redshift dependence, including the
broken power laws below, is evaluated in the Lee22 code, so the mapping is exact; a `PowerLawParam` representation could
not be. Self-test: XGPaint's `compute_DM` on the wrapper matches the Lee22 code's projected column to 2e-6 and the 3-D
density to 1e-9; the spherical products are identical bin for bin.

### 2.3 What was deliberately not changed

XGPaint's `f_b = Omega_b/Omega_m` normalization of the Battaglia16 gas density, its `0.9` factor, its mass per free
electron of ionized H+He at `X_H = 0.76` (`src/profiles_tau.jl:88-106`), the `1/(1+z)` in `compute_DM`, and, for map
painting, the line of sight to 1e5 R200c and the 4 R200c disc. Battaglia16 itself was not re-normalized: inside the
sphere it stays at 0.67 of TNG.

## 3. Changes on the Lee22 side

### 3.1 The parameters used (Lee et al. 2022, arXiv:2205.01710; M200c in physical Msun, h = 0.68, x = r/R200c)

Profile form (eq. 10): `n_e / n200 = n0 (x/x_c)^gamma [1 + (x/x_c)^alpha]^(-beta')`, `alpha = 1`, `gamma = -0.3`.
Broken mass power law (eq. 12): `B(M; a, b) = (M/M_cut)^a` for `M < M_cut`, `(M/M_cut)^b` for `M >= M_cut`.

No-concentration fit (Table A2), `M_cut = 10^13.61 h^-1 Msun = 5.99e13 Msun`:

| parameter | expression used |
|---|---|
| n0 | `6.8 (M / M_cut)^0.68 (1+z)^-2.11` (pivot option `mcut`; the previous implementation used `M / 1e14 Msun`, a factor 1.42 lower) |
| x_c | `7.9 (1+z)^-0.67 B(M; 0.47, -0.45)` |
| beta' | `19.5 (1+z)^-0.31 B(M; 0.70, -0.18)` |

Best fit with concentration (Table 3), `M_cut = 10^13.75 h^-1 Msun = 8.27e13 Msun`, `c10 = c/10`:

| parameter | expression used |
|---|---|
| n0 | `15.7 (M / M_cut)^0.87 (1+z)^-2.09 c10^0.63` |
| x_c | `2.2 (1+z)^-0.74 c10^-1.37 B(M; -0.06, -1.45)` |
| beta' | `7.5 (1+z)^-0.39 c10^-1.11 B(M; 0.24, -1.10)` |
| c | TNG-mean proxy `5.65 (M h / 10^13.1)^log10(4.53/5.65) (1+z)^-0.47` (option `tng_mean`; Duffy08 available) |

In both fits `x_c` and `beta'` use `min(M, 10^14.8 h^-1 Msun = 9.28e14 Msun)` (shape frozen above the fitted mass range;
affects the 868 most massive foreground halos only); `n0` always uses the true mass. In XGPaint terms:
`xc = x_c`, `alpha = 1`, `gamma = -0.3`, `beta = beta' + 0.3`, `P0 = 200 n0`.

### 3.2 The individual changes and their size

1. **Exponent convention.** Lee22's `-beta'` versus XGPaint's `-(beta+gamma)/alpha`; implemented as written in the Lee22
   code and as `beta = alpha beta' - gamma` in the XGPaint wrapper. No numerical effect; it is what makes the wrapper exact.
2. **n0 pivot of the no-concentration fit.** Eq. 12 shares `M_cut` among all parameters; the first implementation used
   1e14 Msun for `n0`. Changing to `M_cut` multiplies `n0` by `(1e14 / 5.99e13)^0.68 = 1.42`.
3. **Normalization reading of eq. 9.** Eq. 9 prints `n200 = 200 rho_cr Omega_b/(X_H m_p Omega_m)`, which exceeds the nucleon
   density of the gas it describes (see `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md`). Readings tested inside the
   sphere, as factors on the printed form and as ratios to TNG in the 1e13-1e14 window: as printed 1 (1.66);
   times `Omega_b/Omega_m` 0.158 (0.26; the earlier "correction", retracted); ionized electron count `X_H(1+X_H)/2` 0.669
   (1.11); `X_H` in the numerator `X_H^2` 0.578 (0.96); **XGPaint-native** 0.602 (1.00). The XGPaint-native reading applies
   no hydrogen factor on the Lee22 side: `n0 f(x)` is taken as the gas density in units of `200 f_b rho_cr`, exactly as
   XGPaint takes Battaglia16's `rho_fit`, and XGPaint's `ne2d` converts to electrons. It is the reading in which both
   profiles receive identical treatment. This is the largest single change (factor 0.602).
4. **Shape clip above the fitted mass range.** Above `10^14.8 h^-1 Msun` the best fit's outer slope falls below 1 and the
   projected column diverges; `x_c` and `beta'` are frozen at the limit. Irrelevant for the 1e13-1e14 window.
5. **Concentration source (best fit only).** TNG-mean proxy from the paper's own quoted means instead of Duffy08.
6. **Redshift scaling.** Left as printed (`rho_cr(z)` at each snapshot). Caveat: the fitted `n0 propto (1+z)^-2.1` makes the
   implied gas fraction inside R200c fall by 3.4x from z = 0 to 1 (1.8 f_b at z = 0, 0.6 f_b at z = 0.5 for the
   XGPaint-native reading at 3e13 Msun), so agreement at z_s = 1 rests on halos at z = 0.3-0.8. Both this and the eq. 9
   form should be confirmed with the authors.

## 4. Changes in how the comparison is made

- Same rays (120,000 distinct HEALPix NSIDE 4096 pixel centres, seed 42), same catalogue, same exact `theta <= theta200c`
  membership for every model; only the truncation surface and the density model differ between products.
- Windows are compared where both catalogues contain the same halos; the hit fraction (rays with DM > 0) is the
  geometry check: 52.6% TNG versus 53.7% HalfDome in the 1e13-1e14 window. In the all-halo window TNG's hit fraction is
  100% because it resolves halos to 1e10 Msun (HalfDome starts at 7.3e12), and the 1e10-1e13 halos alone carry 147 of
  TNG's 178 pc cm^-3 mean, so that panel compares mass floors rather than gas models.
- Percent panels: `100 (HD - TNG)/TNG` per bin where both samples hold >= 10 rays, adjacent bins merged until they do
  elsewhere; linear axis clipped to +-100%.

## 5. Files

- Profiles: `lee2022_frb_dm_profile.jl` (Lee22 fits with options; `Battaglia16DensityDMProfile`; `SphericalChordDMProfile`;
  `Lee2022XGPaintDMProfile`; self-tests).
- Generator: `generate_halfdome_z1_dm_mass_windows.jl` (`--halo-boundary`, `--dm-profile=lee2022_xgpaint`,
  `--lee2022-normalization=...`, `--lee2022-n0-pivot`, `--lee2022-shape-mass-clip`, `--lee2022-concentration-source`).
- Runners: `run_spherical_z1_nside4096_1r200c_variants_local.sh` (all spherical variants), PBS
  `run_halfdome_z1_mass_histograms_120k.pbs`.
- Figures: `make_publication_comparisons.py` (`--publication-panels`, `--sphere-1r200c-only`, `--sphere-b16-only`).
- Products used in the publication figure: `outputs/zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200cprofile_seed42`
  (previous Battaglia16), `..._sphere1p0_m200c_b16_seed42`, `..._sphere1p0_m200c_lee22_noconc_xgpnative_seed42`.
- Background: `LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md`, `LEE22_IMPLEMENTATION_CHECK_20260916.md`,
  `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md`.
