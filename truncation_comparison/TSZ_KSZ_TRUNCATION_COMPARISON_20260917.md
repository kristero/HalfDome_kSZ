# tSZ and kSZ: the FRB truncation fix applied to the Compton-y and tau painters

Date: 2026-09-17. Companion to `frb_map_generation/LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md`.

## 1. Was the FRB problem also present in the tSZ and kSZ code? Yes.

The FRB halo-DM products before 2026-09-16 selected rays inside a projected disc of radius
`X R200c` but integrated the electron density along the whole line of sight, so a ray at
impact parameter `b < X R200c` also collected gas at 3-D radii `r > X R200c`. The tSZ and
kSZ painters share exactly this construction, because all three go through the same XGPaint
routine:

- `XGPaint/src/profiles_y.jl`, `_nfw_profile_los_quadrature(x, ...)`: `2 * quadgk(l -> gnfw(sqrt(l^2 + x^2)), 0, zmax)` with `zmax = 1e5` (R200c units), i.e. an effectively infinite line of sight.
- tSZ: `dimensionless_P_profile_los` (same file) calls it for `Battaglia16ThermalSZProfile`; `compton_y` is `P_e_factor * P_e_los`.
- kSZ: `XGPaint/src/profiles_tau.jl`, `rho_2d` calls the same quadrature for `BattagliaTauProfile`; `ne2d`/`compute_tau` turn it into tau. (`HaloDMProfile`, the old FRB profile, inherits `rho_2d` from the tau profile, which is why the FRB code had the problem in the first place.)
- The painter (`XGPaint/src/profiles.jl`, `profile_paint_generic!`) adds `model(theta, M, z)` to every pixel with `theta < theta_max`, where `compute_θmax` returns `4 * angular_size(R200c)` (`mult=4`). The repository's own painters do the same: `halfdome_halo_tSZ.jl` (`paint!`), `halfdome_halo_kSZ.jl` (`compute_θmax`, `model_interp(theta, ...)`), and `frb_map_generation/paint_halfdome_battaglia12_tsz_map.jl` (external `X R200c` aperture, default 3 or 4).

So the tSZ and kSZ maps truncate in 2-D (a disc on the sky) and not at all along the line of
sight, exactly as the pre-fix FRB products. CLASS-SZ, by contrast, Fourier-transforms the 3-D
profile truncated at a sphere (`x_outSZ` for the pressure profile, `x_out_truncated_density_profile_electrons`
for the gas density), so the earlier map-vs-CLASS-SZ comparisons were not geometrically like-for-like.

Size of the effect at the profile level (`X = 4`, ratio of the chord-limited to the infinite LOS
integral; `test_spherical_truncation.jl`):

| profile | x = b/R200c = 1 | x = 3 | x = 3.9 | disc-integrated signal, sphere / projected |
|---|---|---|---|---|
| Battaglia12 pressure (y), 1e13-1e15 Msun | 0.993-0.997 | 0.81-0.84 | 0.30-0.33 | 0.972 (1e13) to 0.991 (1e15) |
| Battaglia16 AGN density (tau), 1e13-1e15 Msun | 0.966-0.983 | 0.68-0.74 | 0.23-0.26 | 0.875 (1e13) to 0.935 (1e15) |

The density profile is shallower (`beta = 3.83` against an effective 4.65 for the pressure), so the
kSZ maps are affected several times more than the tSZ maps.

## 2. Implementation

`spherical_truncation_profiles.jl` provides `ChordMeanProfile(inner, X)`, an XGPaint `AbstractGNFW`
wrapper for either profile. As in the FRB code it caches the chord-mean `g = value_sphere / (2 L)`,
`L = sqrt(X^2 - x^2)`, which is positive and continuous, so XGPaint's `build_interpolator` and log
interpolation are used unchanged; the painter multiplies the interpolated `g` by the exact chord factor
`2 X sqrt(1 - (theta/theta_max)^2)` per pixel, giving zero at the disc edge. The amplitude is taken
from XGPaint's own evaluation (`inner(theta) / infinite quadrature`, evaluated once per (M, z)), so no
normalization is re-derived. Self-test (`test_spherical_truncation.jl`): chord-mean times chord factor
reproduces the direct chord quadrature to 1e-8, spherical <= projected everywhere, monotonic, continuous
at the edge; the cached grids match direct evaluation to 1e-7.

`paint_truncation_comparison_maps.jl` paints all four maps in one pass over the complete HalfDome
lightcone (85,224,251 halos, M200c from `halo_mass_m200c / h`, `h = 0.6774`, `Omega_b = 0.0486`,
`Omega_c = 0.2603`, NSIDE 4096): Compton-y and `-tau v_los/c` with the projected (previous) and the
spherical truncation, same `theta_max = 4 R200c` disc, same pixels, same catalogue velocities
(km/s, projected on the position unit vector). It also writes the (log M, z) histogram of painted halos
with the summed `(v_los/c)^2`, from which `compute_catalogue_poisson_terms.jl` computes the exact
one-halo (Poisson) term of each map, `C_ell = (1/4pi) sum_i w_i |a_ell,i|^2`, for the normalization checks.

CLASS-SZ references (`compute_class_sz_references.py`, classy_sz 0.1.70): tSZ `B12`, `x_outSZ = 4`;
kSZ `gas_profile = B16`, `gas_profile_mode = agn`, electrons truncated at `4 R200c`, `f_free = 0.9`
(XGPaint's `ne2d` factor; class_sz's default is 1.0). Tinker08 M200c mass function, Duffy08
concentration, mass and redshift limits equal to the catalogue's (`4.46e12 <= M200c/(Msun/h) <= 2.57e15`,
`0.0021 <= z <= 3.86`), HalfDome cosmology with massless neutrinos. class_sz's B16 gas density at
`M = 1e14 Msun/h, z = 0.5` agrees with XGPaint's `P0 gNFW f_b rho_crit` to 0.4% at all tested radii;
class_sz's fixed `mu_e = 1.14` against XGPaint's 1.137 is a 0.3% difference in tau (0.6% in power).
CLASS-SZ velocities are the linear-theory `v_rms^2(z)` (3-D `(v/c)^2 = 1.06e-6` at `z = 0`, i.e. 308 km/s),
while the maps use the N-body halo velocities, so the kSZ amplitude comparison is not expected to close
to better than the velocity difference.

## 3. Results

All four maps: NSIDE 4096, 85,224,251 halos, 4.59e9 pixel updates per map, 48 min on 20 threads
(`maps/provenance.txt`). Spectra: `anafast` (iter=0) to `ell = 8192`, pixel-window corrected, averaged in
20 log bins per decade. Figures: `spectra/tsz_truncation_comparison.png`, `spectra/ksz_truncation_comparison.png`.

### 3.1 Size of the fix (sphere / projected, same halos and pixels)

| ell | 32 | 81 | 129 | 207 | 330 | 526 | 840 | 1341 | 2141 | 3417 | 5454 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tSZ y power | 0.987 | 0.991 | 0.994 | 0.995 | 0.996 | 0.998 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| kSZ power | 0.868 | 0.877 | 0.876 | 0.885 | 0.895 | 0.906 | 0.921 | 0.939 | 0.960 | 0.983 | 1.003 |

- tSZ: the previous code overestimated the Compton-y power by at most 1.2% (`ell < 50`), by 0.4-0.6% at
  `ell = 200-500`, and by nothing above `ell ~ 2000`. For the tSZ analysis the fix is immaterial.
- kSZ: the previous code overestimated the halo kSZ power by 11-13% for `ell < 400`, 9% at `ell ~ 500`,
  8% at `ell ~ 800`, 6% at `ell ~ 1300`, 4% at `ell ~ 2100`, 1.7% at `ell ~ 3400`; above `ell ~ 5000` the two
  agree (the sphere is marginally higher, +0.3%, because removing the outskirts sharpens the profile edge).
  The catalogue one-halo term predicts the same ratio to within 1-3% for `ell > 300` (dashed green curve),
  so the effect is the geometry of the profile, not a painting artefact. Gas outside the 4 R200c sphere but
  inside the projected disc carried 6-12% of the tau signal per halo; the tSZ pressure profile is steep
  enough that the same gas carries only 1-3% of y.

### 3.2 Comparison with CLASS-SZ (1h+2h, same 3-D truncation)

| ell | 81 | 207 | 330 | 526 | 840 | 1341 | 2141 | 3417 | 5454 |
|---|---|---|---|---|---|---|---|---|---|
| tSZ previous / CLASS-SZ | 0.91 | 1.01 | 0.95 | 0.92 | 0.89 | 0.90 | 0.92 | 0.98 | 1.13 |
| tSZ fix / CLASS-SZ | 0.90 | 1.01 | 0.95 | 0.92 | 0.89 | 0.90 | 0.92 | 0.98 | 1.13 |
| kSZ previous / CLASS-SZ | 0.51 | 0.70 | 0.80 | 0.87 | 0.92 | 0.95 | 0.97 | 1.02 | 1.17 |
| kSZ fix / CLASS-SZ | 0.45 | 0.62 | 0.71 | 0.79 | 0.85 | 0.89 | 0.93 | 1.00 | 1.17 |
| CLASS-SZ 2h / 1h, tSZ | 0.16 | 0.08 | 0.06 | 0.04 | 0.02 | 0.01 | 0.01 | 0.00 | 0.00 |
| CLASS-SZ 2h / 1h, kSZ | 1.09 | 0.64 | 0.42 | 0.26 | 0.15 | 0.08 | 0.04 | 0.02 | 0.01 |
| catalogue 1h / CLASS-SZ 1h, tSZ (projected) | - | 1.13 | 1.01 | 0.94 | 0.91 | 0.90 | 0.90 | 0.90 | 0.90 |
| catalogue 1h / CLASS-SZ 1h, kSZ (projected) | - | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.97 | 0.96 | 0.94 |
| map / catalogue 1h (2h + pixel effects), tSZ | - | 0.97 | 1.00 | 1.01 | 1.00 | 1.01 | 1.03 | 1.09 | 1.25 |
| map / catalogue 1h (2h + pixel effects), kSZ | - | 1.14 | 1.15 | 1.10 | 1.06 | 1.04 | 1.03 | 1.08 | 1.25 |

- **tSZ.** With a like-for-like 3-D truncation the HalfDome y map agrees with CLASS-SZ to 10% over
  `ell = 100-3500` (0.89-1.01), and the fix does not change that. The remaining 10% deficit at
  `ell ~ 800-2000` is the one-halo term: the catalogue Poisson term is 0.90 of CLASS-SZ's 1h at the same
  multipoles, i.e. HalfDome's abundance of the halos that dominate the y power (a few 1e14 Msun) times their
  y is 10% below the Tinker08 prediction at fixed M200c. Above `ell ~ 4000` both maps rise above every
  reference (1.13 at 5454, 1.5 at 8000) and above their own Poisson term; this is the painting of
  unresolved halos at pixel centres plus the pixel-window correction, unrelated to the truncation, and was
  present in the earlier repository comparisons.
- **kSZ.** In the one-halo dominated range `ell ~ 1300-3500` the corrected map is 0.89-1.00 of CLASS-SZ
  (previous code 0.95-1.02, i.e. the earlier near-agreement at `ell ~ 2000-3500` was partly the extra
  line-of-sight gas). The catalogue Poisson term equals CLASS-SZ's 1h to 1% at `ell = 200-1300`. Below
  `ell ~ 1000` the map falls increasingly short of CLASS-SZ because CLASS-SZ's `kSZ_kSZ_2h` term is large
  (2h/1h = 0.64 at `ell ~ 200`, 1.1 at `ell ~ 80`) while the maps, which carry the actual N-body velocity
  correlations, contain only a 10-15% excess over their Poisson term at `ell = 200-500`. A kSZ two-halo term
  built from the linear momentum field should be small at these multipoles because the longitudinal modes
  cancel along the line of sight; CLASS-SZ's `kSZ_kSZ_2h` does not appear to include that cancellation, so
  the meaningful CLASS-SZ comparison for the halo kSZ map is the one-halo regime.
- **Velocity caveat.** CLASS-SZ's `get_vrms2_at_z` gives a 3-D linear rms of 308 km/s at z = 0 (1-D: 178
  km/s). The HalfDome halos have a line-of-sight rms of 300 km/s at z < 0.1, 304 at z = 0.5, 281 at z = 1,
  245 at z = 2 (from the painter's histogram), i.e. 1.7 times the linear 1-D rms and numerically equal to
  the linear 3-D rms at every redshift. Since the catalogue Poisson term and CLASS-SZ's 1h agree to 1% while
  the tSZ-weighted abundance agrees to 10%, the velocity factor that enters CLASS-SZ's kSZ 1h must be the
  3-D `v_rms^2`, not `v_rms^2/3`; with the 1-D factor the CLASS-SZ curve would be three times lower. The
  1% agreement is therefore partly a coincidence between the halo velocity dispersion of the simulation and
  the linear 3-D rms, and the kSZ normalization comparison should be read with that in mind. (The class_sz
  kSZ output was taken as dimensionless `ell(ell+1)C_ell/2pi` in `(Delta T/T)^2`; the Poisson-term match
  confirms that reading.)

### 3.3 Bottom line

The truncation problem found in the FRB code was also present in the tSZ and kSZ painters. Fixing it
changes the Compton-y power spectrum by less than 1.2% anywhere and by less than 0.5% for `ell > 300`, so
the tSZ results stand. It lowers the halo kSZ power spectrum by 11-13% at `ell < 400`, 8% at `ell ~ 800`,
4% at `ell ~ 2000` and under 2% above `ell ~ 3400`, and brings the map to 0.89-1.00 of CLASS-SZ in the
one-halo regime with the same 3-D truncation. Any kSZ product that used `BattagliaTauProfile` with XGPaint's
default painting inherits the 4-13% excess at `ell < 2000`.

## 4. Files

- `spherical_truncation_profiles.jl`, `test_spherical_truncation.jl`: the wrapper and its self-test.
- `build_truncation_caches.jl`: the four interpolator caches (`caches/`).
- `paint_truncation_comparison_maps.jl`: the four maps, histogram and provenance (`maps/`).
- `compute_catalogue_poisson_terms.jl`: one-halo terms from the catalogue histogram (`spectra/catalogue_poisson_terms.h5`).
- `compute_class_sz_references.py`: CLASS-SZ tSZ and kSZ references (`spectra/class_sz_{tsz,ksz}_x4_reference.npz`).
- `compute_spectra_and_plots.py`: spectra (`spectra/raw_cl.npz`, `spectra/binned_spectra.csv`, `spectra/summary.json`) and the figures `spectra/tsz_truncation_comparison.{png,pdf}`, `spectra/ksz_truncation_comparison.{png,pdf}`.
- `dm_profiles_common.jl`, `test_dm_profiles.jl`, `build_dm_caches.jl`, `paint_dm_truncation_maps.jl`, `compute_catalogue_poisson_cross_terms.jl`, `compute_cross_spectra_and_plots.py`: the tSZ x DM addendum (Section 5).
- `logs/`: build, paint, CLASS-SZ and post-processing logs.

## 5. Addendum (2026-09-18): tSZ x halo-DM cross-spectra, Lee22 no-c vs Battaglia16

Requested follow-up: the Battaglia12 Compton-y map crossed with halo-DM maps of the same halos,
for the Lee22 no-concentration fit (XGPaint-native normalization, `Lee2022XGPaintDMProfile`, M_cut
n0 pivot, shape clip at 10^14.8/h Msun) and for Battaglia16 (`HaloDMProfile`), each with the previous
projected truncation and with the spherical 4 R200c truncation. Both DM profiles go through the same
`ChordMeanProfile` wrapper (`dm_profiles_common.jl`); `test_dm_profiles.jl` checks it against the FRB
code's own `SphericalChordDMProfile` to 1e-6 for both. Maps: `paint_dm_truncation_maps.jl` (four DM maps,
NSIDE 4096, h = 0.6774 for the mass conversion to match the y maps; the FRB products used 0.68). Spectra
and figures: `compute_cross_spectra_and_plots.py`, `spectra/tsz_x_dm_lee22_vs_b16.{png,pdf}`,
`spectra/dm_auto_lee22_vs_b16.{png,pdf}`, `spectra/binned_cross_spectra.csv`, `spectra/summary_cross.json`;
catalogue one-halo cross terms in `spectra/catalogue_poisson_cross_terms.h5`.

| ell | 32-100 | 293 | 945 | 3040 | 6130 |
|---|---|---|---|---|---|
| y x DM(Lee22 no-c): sphere / projected | 0.985-0.995 | 0.996 | 0.998 | 1.000 | 1.000 |
| y x DM(Battaglia16): sphere / projected | 0.94-0.95 | 0.966 | 0.985 | 0.998 | 1.002 |
| DM(Lee22 no-c) auto: sphere / projected | 0.98-0.99 | 0.989 | 0.994 | 1.000 | 1.001 |
| DM(Battaglia16) auto: sphere / projected | 0.89-0.90 | 0.910 | 0.947 | 0.990 | 1.005 |
| y x DM: Lee22 no-c / Battaglia16 (sphere) | 2.5-3.0 | 2.81 | 2.10 | 1.33 | 0.84 |
| correlation coefficient r_ell, Lee22 / B16 (sphere) | 0.80-0.85 / 0.78-0.85 | 0.79 / 0.76 | 0.74 / 0.75 | 0.74 / 0.75 | 0.75 / 0.77 |

- The truncation fix matters for Battaglia16 DM (auto spectrum -10 to -11% at ell < 200, -5% at ell ~ 900,
  cross with y -5% at ell < 100, -3.4% at ell ~ 300) but hardly for Lee22 no-c (auto -1 to -2% at
  ell < 200, cross under 1%): the Lee22 fit is much steeper beyond R200c for massive halos (sphere /
  projected = 0.80 at 3.5 R200c for 1e14 Msun against 0.54 for Battaglia16), so it has little gas outside
  the 4 R200c sphere to remove.
- Lee22 no-c gives 2-3 times more y x DM power than Battaglia16 at ell < 1000 and 33% more at ell ~ 3000,
  crossing below Battaglia16 above ell ~ 5000: its low-mass halos carry far more extended electron column
  (DM at R200c of a 7e12 Msun halo: 71 against 23 pc cm^-3) while its massive halos are more compact.
  The y x DM correlation coefficient is 0.74-0.85 for both profiles and is not changed by the fix.
- Map / catalogue one-halo term for the cross: Lee22 1.00 at ell = 300-950, Battaglia16 1.09 at ell ~ 300
  and 1.03 at ell ~ 950, i.e. a small two-halo contribution; above ell ~ 4000 the unresolved-halo pixel
  sampling excess seen in the auto spectra appears here as well (1.27-1.33 at ell ~ 6000).
