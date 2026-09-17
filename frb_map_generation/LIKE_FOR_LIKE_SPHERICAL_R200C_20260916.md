# TNG versus HalfDome halo DM: what differs, and a like-for-like spherical-R200c comparison

Date: 2026-09-16. Companion to `MINIMAL_HALO_DM_AND_RESOLUTION_20260916.md`.

## 1. Both are column integrals; they differ in *which gas* the column contains

In both cases the halo DM of a ray is `DM = int n_e dl / (1+z)`. The difference is the domain of the integral.

**TNG (Konietzka's within-R200 catalogue, as far as the arrays tell us).** The simulation has discrete gas cells with known positions. A ray-tracing code sums `n_e * dl` over the cells the ray crosses. The "halo within R200" component keeps only cells that lie inside the R200 sphere of some halo (we assume the spherical membership implied by the file names; the exact recipe is not documented in the files we hold and should be confirmed with Ralf). Consequences:

- A ray with impact parameter `b` relative to a halo centre integrates only over the chord of length `2 sqrt(R200^2 - b^2)` inside the sphere. As `b -> R200` the chord and the DM go to zero continuously. Near the edge `DM proportional to sqrt(1 - b/R200)`, so with rays uniform in area `p(DM) proportional to DM` at the lowest DMs of each halo, which builds the smooth low tail down to 0.1 pc cm^-3 seen in every TNG window.
- Gas outside R200 along the same line of sight (the halo's own outskirts, the two-halo term, filaments) is not in this component; it is in the "IGM" part of TNG's split.
- The density is the actual, clumpy cell density, including satellites and cold gas inside R200; nothing is spherically averaged.

**HalfDome (this repository).** There are no gas cells, only halo centres, M200c and z. Each halo is dressed with an analytic, spherically symmetric electron-density profile (Battaglia16 or Lee22). The projected products integrate that profile along the whole line of sight: XGPaint's `_nfw_profile_los_quadrature` and the Lee22 code both integrate to `1e5 R200c`, effectively to infinity. The aperture (`1 R200c`, `3 R200c`, ...) is only a *selection* of which rays receive the halo: a ray is credited with the halo if its angular separation from the centre is below `theta_max = angular_size(X R200c, z)`, and it then receives the full projected column at that impact parameter, including all the gas far outside the sphere along the ray. Consequences:

- A ray exactly at the edge still receives the full column at `b = X R200c`, which for a floor-mass halo is 22 pc cm^-3 at `X = 1` and 3 at `X = 3`. This is the hard floor of every HalfDome PDF.
- The projected column at `b = R200c` is dominated by gas at `r > R200c` (for Battaglia16, 26% of the mean DM per hit ray at 1R200c comes from gas outside the sphere, measured as 1 - 116.6/158.4 between the spherical and projected products; for a grazing ray it is 100%).

So the statement "TNG is also a column integration" is right; the difference is the truncation surface. TNG truncates in 3-D (a sphere); the projected HalfDome products truncate in 2-D (a disc on the sky) and not at all along the line of sight.

## 2. What theta200c is, and what "resolution" means here

`theta200c = atan(R200c / D_A(z))` is the angle subtended by the halo's R200c at its angular-diameter distance. For a floor-mass halo (7.3e12 Msun) it is 7 arcmin at z = 0.05, 1.7 arcmin at z = 0.2, 0.95 arcmin at z = 0.5 and 0.7 arcmin at z = 1. The ray's impact parameter in units of R200c is `b/R200c = theta/theta200c`, where `theta` is the exact angle between the ray direction and the halo centre. The aperture edge is `theta_max = X theta200c`.

Nothing about `theta200c` is pixelised. The generator computes `theta` from the two unit vectors with `acos` of their dot product, so the grazing geometry is exact for every ray, at any NSIDE. What NSIDE controls is only *which* 120,000 directions exist (pixel centres). "Better resolution" of the edge therefore means more rays per halo disc, not smaller pixels:

- 120k rays over the sky is one ray per 0.34 deg^2 (mean spacing 35 arcmin). A floor-mass halo at z = 0.5 covers `pi theta200c^2 = 2.8 arcmin^2`, so only about 1 in 440 such halos is hit at all, and the outer 1% of the disc area (where a spherical cut gives DM below about 3 pc cm^-3) is sampled by 2% of the hits. The 120k-ray histogram resolves the tail only as far as the counts allow; more rays (the generator's `--nfrb`) sharpen it, finer pixels do not.
- The histogram itself has 300 log bins over 0.1-30000 pc cm^-3 and counts only in-range rays; rays with DM = 0 are reported as the zero fraction.

## 3. How the rays are defined

`draw_random_frb_pixels` draws 120,000 *distinct* HEALPix pixel indices uniformly at random (MersenneTwister seed 42) from the `12 NSIDE^2` pixels; the ray direction is the pixel centre (`pix2ang`). Each ray is a single infinitely thin line of sight to the exact source plane z = 1. No map is painted and no pixel is averaged: for every (ray, halo) pair with `theta <= theta_max` the exact profile value at `theta` is added to that ray. Pixelisation enters only through the discrete set of available directions, and through the disc search (`queryDiscRing` over pixels with a margin) that finds candidate rays; the final `theta <= theta_max` test is exact.

## 4. The like-for-like implementation (spherical R200c boundary)

New generator option `--halo-boundary=spherical` (environment `HALO_BOUNDARY=spherical`), implemented in `lee2022_frb_dm_profile.jl` and `generate_halfdome_z1_dm_mass_windows.jl`:

1. **Profile-owned densities.** `Battaglia16DensityDMProfile` evaluates the 3-D electron density from XGPaint's own Battaglia16 parameters and normalization (`rho_gas = P0 gNFW f_b rho_crit(z)`, electrons per mass via XGPaint's `ne2d` composition). Integrated along the long LOS it reproduces XGPaint's `HaloDMProfile` to better than 1e-5 at all tested masses, redshifts and impact parameters (self-test). The Lee22 densities are the corrected fits from `LEE22_IMPLEMENTATION_CHECK_20260916.md`.
2. **Chord-limited line of sight.** For a ray at `x = b/R200c` inside the sphere of radius `X R200c`, `DM = (2 R200c/(1+z)) int_0^{L} n_e(sqrt(x^2 + l^2)) dl` with `L = sqrt(X^2 - x^2)`; outside, DM = 0. This is the analytic version of "sum the cells inside the sphere".
3. **Cache that stays smooth.** The generator interpolates a cached profile in `(log theta, z, log M)`. Caching `DM_sphere` directly would put a zero (log of zero) at a mass- and redshift-dependent edge. Instead `SphericalChordDMProfile` caches `g = DM_sphere / (2L)`, the chord-mean column, which is positive and continuous (it tends to the edge density at the boundary and is held there outside). Per ray the generator multiplies the interpolated `g` by the exact chord factor `2 X sqrt(1 - (theta/theta_max)^2)`, so DM goes to zero exactly at `theta_max`. The self-test checks that `g times chord` reproduces the direct spherical quadrature to 1e-9, that `DM_sphere <= DM_projected`, that DM decreases outward and that `g` is continuous at the edge.
4. **Everything else unchanged.** Same 120k rays, seed, catalogue, M200c windows, bins, and the same `theta <= theta_max` selection as the projected 1R200c products, so the only difference between the dotted and solid HalfDome curves in the figure is the truncation surface.

Products: `frb_map_generation/outputs/zsrc1p0_nside4096_nrays120000_allhalos_sphere1p0_m200c_{b16,lee22_noconc,lee22_pref}_seed42/` (runner `run_spherical_z1_nside4096_1r200c_variants_local.sh`; provenance records `halo_boundary=spherical`, the sphere radius and the chord-mean cache).

What is still *not* like-for-like: HalfDome densities are smooth spherical means (no substructure, no satellites, no cold clumps), the HalfDome floor is 7.3e12 Msun while TNG resolves 1e10, TNG's masses are its own M200 while HalfDome's are M200c from the lightcone, and the exact TNG membership rule is assumed.

## 5. Results

All numbers below are for z_s = 1, 120,000 rays, NSIDE 4096, M200c windows; "hit" means DM > 0 for that window. TNG is Konietzka's within-R200 catalogue at the same source redshift. HalfDome percentages of hit rays below a threshold are taken from the 300-bin histograms, TNG's from the raw values, so they are comparable to about the bin width (2.5% in DM).

### 5.1 Geometry check: the fraction of rays that hit a halo

| Window (Msun) | TNG hit % | HalfDome hit % (sphere; all three models) | Comment |
|---|---|---|---|
| Total | 100.0 | 64.9 | TNG resolves halos down to 1e10 and every ray passes inside the R200 of some halo; HalfDome's catalogue starts at 7.3e12 |
| 1e10-1e13 | 100.0 | 14.0 | HalfDome only has 7.3e12-1e13 here |
| 1e12-1e14 | 90.6 | 60.2 | same mass floor |
| 1e13-1e14 | 52.6 | 53.7 | the only window where both catalogues contain the same halos: agreement to 2% |

The hit fraction depends only on the halo abundance, R200c and the membership rule (`theta <= theta200c` is the projection of the sphere), so the 1e13-1e14 agreement shows that TNG's "within R200" component is the sphere we assumed and that the two halo populations match in that range. Every other window is dominated by halos HalfDome does not contain; in TNG the 1e10-1e13 halos alone give a mean within-R200 DM of 147 pc cm^-3 out of 178 for all masses, so the total-window comparison mostly measures HalfDome's mass floor, not the gas model. The genuine model test is the 1e13-1e14 window (and, for the high tail, 1e14-1e16).

### 5.2 The low tail is a truncation effect, not a pixelisation effect

In the 1e13-1e14 window the fraction of hit rays with DM below 1 / 3 / 10 / 30 pc cm^-3 is 0.09 / 0.57 / 4.5 / 20.1 % in TNG. The projected 1R200c Battaglia16 product has 0 / 0 / 0 / 1.35 % (hard floor at 25 pc cm^-3); the same profile inside the sphere gives 0.06 / 0.53 / 5.4 / 26.4 %. Changing the truncation surface alone therefore reproduces TNG's tail to within the sampling noise of a few hundred rays, in every window (compare also the "first populated bin": 0.2 pc cm^-3 against TNG's 0.02-0.4). This closes the question raised in `MINIMAL_HALO_DM_AND_RESOLUTION_20260916.md`: NSIDE 2048/4096/8192 had already shown that pixelisation does not touch the tail; the sphere shows what does.

### 5.3 Amplitude: what remains after the geometry is matched

Mean DM of hit rays, 1e13-1e14 window: TNG 138.6; Battaglia16 sphere 92.9 (0.67 of TNG); Battaglia16 projected 126.9 (0.92, but 27% of it is gas outside the sphere along the line of sight, so the near-agreement of the old product was partly accidental); Lee22 corrected fits inside the sphere 36-37 (0.27). Medians of hit rays: TNG 81, Battaglia16 sphere 60, Lee22 corrected sphere 22.

An approximate translation into enclosed gas: the implementation check gives Battaglia16 0.57-0.63 f_b of the baryons inside R200c at 3e13-1e14 Msun. Scaling by the DM ratios, TNG's within-R200 sum corresponds to roughly 0.85-0.95 f_b of free electrons inside R200c for these halos, the corrected Lee22 fits to about 0.2 f_b and the literal fits to about 1.4 f_b (the literal fits' enclosed fraction falls steeply with redshift, from 2-5 f_b at z = 0 to 0.5-0.7 f_b at z = 1, and rays to z_s = 1 mostly sample halos at z = 0.3-0.8). These are order-of-magnitude figures that ignore the mass and redshift weighting of the rays.

**Electron-count reading of eq. 9 (added 2026-09-17).** Eq. 9 converts the reference gas density to a number density with 1/(X_H m_p). The free electrons per unit mass of fully ionized H+He are (1+X_H)/(2 m_p) = 0.88/m_p (TNG's own electron abundance for ionized primordial gas gives the same 0.88), while 1/(X_H m_p) = 1.32/m_p. Replacing one by the other multiplies the literal reading by X_H(1+X_H)/2 = 0.669 (option `--lee2022-normalization=electron_count`, no Omega_b/Omega_m factor, otherwise the corrected conventions). Inside the sphere, in the 1e13-1e14 window, this gives a mean DM per hit ray of 154-157 pc cm^-3 against TNG's 138.6 (1.11-1.13), a median of 95 against 81, and tail fractions 0.33 / 3.3 / 17 % below 3 / 10 / 30 pc cm^-3 against TNG's 0.57 / 4.5 / 20 %. That is closer to TNG than the literal reading (1.66), the f_b-corrected reading (0.27) and Battaglia16 (0.67). Whether the fit's n200 was really the electron count or the printed 1/(X_H m_p) is a question about the paper's bookkeeping (n0 is defined relative to whatever n200 they used), so this reading is the recommended default only until that is confirmed.

- Battaglia16 inside R200c has the right tail shape but about 30% less electron column per hit than TNG's within-R200 gas. Both are physically normalized with f_b rho_crit; the difference is the gas fraction and profile shape of the underlying simulations (Battaglia's AGN-feedback runs versus TNG300), plus everything TNG's cell sum contains that a smooth profile does not (satellites, clumps, gas of neighbouring halos inside the sphere).
- The corrected Lee22 fits fall a factor 3.7 below TNG in the very mass range they were fitted to (1e13-10^14.8 h^-1 Msun). Geometry can no longer explain this. The sensitivity runs with the literal eq. 9 normalization (no extra Omega_b/Omega_m factor; everything else, including the sphere, the M_cut pivot, the TNG-mean concentration and the shape clip, identical) give 230-234 in the same window, a factor 1.66 *above* TNG, and their low-tail fractions (0.02 / 0.14 / 1.6 / 10 % below 1 / 3 / 10 / 30 pc cm^-3) are closer to TNG's than the corrected fits' (0.64 / 4.9 / 26 / 61 %), though not as close as Battaglia16's. Neither reading of the normalization reproduces TNG: the truth lies between them, at about 0.6 of the literal or 3.8 times the f_b-corrected value. The (1+z)^3/E^2 redshift-scaling hypothesis of the implementation check (factor 1.9 at z = 0.5) would close about half of the gap on top of the corrected reading. The Omega_b/Omega_m factor of the implementation check was inferred from the paper's Figure 5; the like-for-like numbers say that inference over-corrects, so the factor must be settled from the paper's definition of n200 and of the fitted quantity rather than from the figure. The rest most plausibly comes from what was fitted: if the paper's radial profiles are medians over halos (or otherwise clump-suppressed), they underpredict the *mean* electron column that DM measures, especially near R200c where satellites dominate the mean density. This, the exact definition of n200 in eq. 9 and the gas cells included in Konietzka's within-R200 sum are the three things to confirm with the authors before using Lee22 for absolute predictions.

### 5.4 Full table

**Total (all masses)**

| Product | hit % | mean DM, all rays | mean DM, hit rays | median DM, hit rays | lowest DM seen | % of hit rays < 1 | < 3 | < 10 | < 30 pc cm^-3 |
|---|---|---|---|---|---|---|---|---|---|
| TNG within R200 (Konietzka) | 100.0 | 177.9 | 177.9 | 129.5 | 0.401 | 0.01 | 0.04 | 0.44 | 4.89 |
| Battaglia16, projected 1R200c (previous) | 64.7 | 102.6 | 158.4 | 112.2 | 23.1 | 0.00 | 0.00 | 0.00 | 2.53 |
| Battaglia16, inside R200c sphere | 64.9 | 75.7 | 116.6 | 73.6 | 0.214 | 0.04 | 0.48 | 4.72 | 22.68 |
| Lee22 no-c corrected, projected 1R200c | 64.9 | 44.6 | 68.8 | 42.5 | 4.27 | 0.00 | 0.00 | 6.62 | 36.17 |
| Lee22 no-c corrected, inside sphere | 64.9 | 32.9 | 50.7 | 25.6 | 0.104 | 0.66 | 4.69 | 23.18 | 54.38 |
| Lee22 best corrected, projected 1R200c | 64.9 | 50.6 | 77.9 | 44.4 | 3.61 | 0.00 | 0.00 | 7.11 | 35.99 |
| Lee22 best corrected, inside sphere | 64.9 | 35.0 | 54.0 | 26.7 | 0.104 | 0.66 | 4.62 | 22.70 | 53.49 |
| Lee22 no-c, literal norm., inside sphere (sensitivity) | 64.9 | 208.2 | 321.0 | 164.0 | 0.42 | 0.01 | 0.13 | 1.63 | 9.43 |
| Lee22 best, literal norm., inside sphere (sensitivity) | 64.9 | 221.7 | 341.8 | 171.1 | 0.42 | 0.01 | 0.13 | 1.62 | 9.26 |
| Lee22 no-c, eq. 9 with (1+X_H)/2m_p electrons, inside sphere | 64.9 | 139.3 | 214.7 | 107.6 | 0.287 | 0.03 | 0.33 | 3.29 | 15.80 |
| Lee22 best, eq. 9 with (1+X_H)/2m_p electrons, inside sphere | 64.9 | 148.3 | 228.6 | 112.2 | 0.287 | 0.03 | 0.34 | 3.25 | 15.47 |

**1e13-1e14 Msun (fit range, same halos in both)**

| Product | hit % | mean DM, all rays | mean DM, hit rays | median DM, hit rays | lowest DM seen | % of hit rays < 1 | < 3 | < 10 | < 30 pc cm^-3 |
|---|---|---|---|---|---|---|---|---|---|
| TNG within R200 (Konietzka) | 52.6 | 72.9 | 138.6 | 80.8 | 0.0182 | 0.09 | 0.57 | 4.54 | 20.05 |
| Battaglia16, projected 1R200c (previous) | 53.8 | 68.3 | 126.9 | 90.9 | 25.1 | 0.00 | 0.00 | 0.00 | 1.35 |
| Battaglia16, inside R200c sphere | 53.7 | 49.9 | 92.9 | 59.6 | 0.214 | 0.06 | 0.53 | 5.39 | 26.35 |
| Lee22 no-c corrected, projected 1R200c | 53.7 | 28.0 | 52.0 | 35.9 | 4.64 | 0.00 | 0.00 | 6.64 | 41.39 |
| Lee22 no-c corrected, inside sphere | 53.7 | 19.6 | 36.4 | 21.7 | 0.104 | 0.64 | 4.85 | 25.56 | 60.51 |
| Lee22 best corrected, projected 1R200c | 53.7 | 28.2 | 52.4 | 35.9 | 4.09 | 0.00 | 0.00 | 6.88 | 41.36 |
| Lee22 best corrected, inside sphere | 53.7 | 19.9 | 37.0 | 22.6 | 0.1 | 0.63 | 4.74 | 25.11 | 60.04 |
| Lee22 no-c, literal norm., inside sphere (sensitivity) | 53.7 | 123.8 | 230.4 | 138.5 | 0.42 | 0.02 | 0.14 | 1.60 | 10.15 |
| Lee22 best, literal norm., inside sphere (sensitivity) | 53.7 | 125.8 | 234.0 | 138.5 | 0.42 | 0.02 | 0.13 | 1.57 | 9.94 |
| Lee22 no-c, eq. 9 with (1+X_H)/2m_p electrons, inside sphere | 53.7 | 82.8 | 154.1 | 94.8 | 0.287 | 0.04 | 0.33 | 3.33 | 17.25 |
| Lee22 best, eq. 9 with (1+X_H)/2m_p electrons, inside sphere | 53.7 | 84.1 | 156.5 | 94.8 | 0.287 | 0.04 | 0.32 | 3.27 | 16.89 |

**1e10-1e14 Msun**

| Product | hit % | mean DM, all rays | mean DM, hit rays | median DM, hit rays | lowest DM seen | % of hit rays < 1 | < 3 | < 10 | < 30 pc cm^-3 |
|---|---|---|---|---|---|---|---|---|---|
| TNG within R200 (Konietzka) | 100.0 | 172.7 | 172.7 | 125.3 | 0.0477 | 0.02 | 0.11 | 0.79 | 6.10 |
| Battaglia16, projected 1R200c (previous) | 60.1 | 76.3 | 126.9 | 94.8 | 23.1 | 0.00 | 0.00 | 0.00 | 3.11 |
| Battaglia16, inside R200c sphere | 60.2 | 55.7 | 92.6 | 59.6 | 0.214 | 0.06 | 0.59 | 5.63 | 26.59 |
| Lee22 no-c corrected, projected 1R200c | 60.2 | 30.5 | 50.8 | 35.9 | 4.27 | 0.00 | 0.00 | 8.01 | 42.75 |
| Lee22 no-c corrected, inside sphere | 60.2 | 21.0 | 34.8 | 20.8 | 0.104 | 0.79 | 5.59 | 27.45 | 62.32 |
| Lee22 best corrected, projected 1R200c | 60.2 | 30.6 | 50.8 | 34.4 | 3.61 | 0.00 | 0.00 | 8.63 | 43.06 |
| Lee22 best corrected, inside sphere | 60.2 | 21.3 | 35.4 | 20.8 | 0.1 | 0.79 | 5.53 | 27.07 | 61.87 |
| Lee22 no-c, literal norm., inside sphere (sensitivity) | 60.2 | 132.6 | 220.3 | 132.8 | 0.42 | 0.02 | 0.17 | 1.94 | 11.29 |
| Lee22 best, literal norm., inside sphere (sensitivity) | 60.2 | 134.6 | 223.7 | 132.8 | 0.42 | 0.02 | 0.16 | 1.93 | 11.15 |
| Lee22 no-c, eq. 9 with (1+X_H)/2m_p electrons, inside sphere | 60.2 | 88.7 | 147.4 | 87.1 | 0.287 | 0.04 | 0.40 | 3.90 | 18.83 |
| Lee22 best, eq. 9 with (1+X_H)/2m_p electrons, inside sphere | 60.2 | 90.0 | 149.6 | 87.1 | 0.287 | 0.04 | 0.41 | 3.88 | 18.56 |

Notes. "Lowest DM seen" is the raw minimum for TNG and the lower edge of the first populated histogram bin for HalfDome (the histogram starts at 0.1 pc cm^-3). A hit ray crosses on average 1.4 halos in the 1e13-1e14 window (hit fraction 52.6% corresponds to a Poisson mean of 0.75 crossings per ray), identically for TNG and HalfDome, so ratios between rows are unaffected by multiple crossings. The literal/corrected ratio is exactly 1/f_b = 6.33 for both Lee22 fits, as it must be. Sensitivity figures: `..._sensitivity_{upper_mass_limits,to_1e14}` add the two literal-normalization rows to the core figure.

## Files

- Code: `lee2022_frb_dm_profile.jl` (density models, chord wrapper, self-test), `generate_halfdome_z1_dm_mass_windows.jl` (`--halo-boundary`), `run_halfdome_z1_mass_histograms_120k.pbs` (`HALO_BOUNDARY`), `run_spherical_z1_nside4096_1r200c_variants_local.sh`, `make_publication_comparisons.py` (`--sphere-1r200c-only`).
- Figures: `frb_map_generation/outputs/publication_comparison_20260916_sphere_1r200c/plots/halo_pdf_sphere_1r200c_b16_lee22_tng_{upper_mass_limits,to_1e14}.{png,svg}` (core: TNG, projected Battaglia16, and the three spherical products) and `..._sensitivity_{upper_mass_limits,to_1e14}` (adds the literal-normalization Lee22 fits inside the sphere); PDF `output/pdf/halfdome_sphere_1r200c_b16_lee22_vs_tng_20260916.pdf`; tables under `analysis/`.
- Sensitivity products: `..._sphere1p0_m200c_lee22_{noconc,pref}_literalnorm_seed42/` (runner variants `lee22_noconc_literalnorm`, `lee22_pref_literalnorm`).
- Electron-count products: `..._sphere1p0_m200c_lee22_{noconc,pref}_efix_seed42/` (runner variants `lee22_noconc_efix`, `lee22_pref_efix`); figures `..._efix_{upper_mass_limits,to_1e14}`.
