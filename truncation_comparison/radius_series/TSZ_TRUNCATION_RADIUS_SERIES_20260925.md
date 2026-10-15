# tSZ truncation-radius series: gas inside N R200c (N = 1..5) vs CLASS-SZ and WebSky

Date: 2026-09-25. Companion to `../TSZ_KSZ_TRUNCATION_COMPARISON_20260917.md` (which introduced the
spherical / chord-limited truncation and compared it with the previous projected-disc convention at 4 R200c).

## 1. Question

With the spherical truncation now available, which painting radius N R200c should the HalfDome Compton-y
maps use? The old XGPaint / WebSky convention was a projected disc of 4 R200c with an infinite line of sight.
Here every halo of the complete HalfDome lightcone (85,224,251 halos, M200c from `halo_mass_m200c / h`,
h = 0.6774, Omega_b = 0.0486, Omega_c = 0.2603, Battaglia12 AGN pressure fit) is painted five times, keeping
only the pressure inside the N R200c sphere (chord-mean interpolator times the exact per-pixel chord factor),
plus once with the old projected 4 R200c convention. All six NSIDE 4096 maps come from one pass over the
catalogue (`paint_radius_series.jl`), so they share halos, pixels and profile exactly. Spectra: `anafast`
(niter = 0) to ell = 8192, pixel window removed, averaged in 20 logarithmic bins per decade.

References:

- CLASS-SZ (classy_sz 0.1.70): Battaglia12 `B12` pressure with the 3-D truncation `x_outSZ = N` for
  N = 1..5, Tinker08 M200c mass function, Duffy08 concentration, HalfDome cosmology (massless neutrinos),
  mass and redshift limits equal to the catalogue's (4.46e12 <= M200c/(Msun/h) <= 2.57e15,
  0.0021 <= z <= 3.86): `compute_class_sz_radius_series.py`. Like-for-like with the spherical maps.
- WebSky public Compton-y map (`tsz.fits`, NSIDE 4096; Stein et al. 2020): spectrum with the same
  estimator and pixel-window correction (`websky_tsz_cl.npz`). WebSky is a different simulation
  (Omega_m = 0.31, h = 0.68, sigma8 = 0.81, halos above ~1e12 Msun/h to z = 4.6) and integrates the
  pressure profile to 4 R_Delta (Stein+2020 Section 4.4.1), so it is an external reference, not a
  like-for-like one. Two extra CLASS-SZ curves with the WebSky cosmology and halo range
  (`compute_class_sz_websky_setup.py`) separate the catalogue/cosmology difference from the truncation.

## 2. Where it ran

The job was prepared for idark (`run_radius_series.pbs`, code staged in
`~/tsz_truncation_radius_series_20260925/code`), but on 2026-09-25 the cluster's `/lustre` filesystem was
unmounted and the PBS server refused connections, so nothing could run there. A watcher on the login node
(`autosubmit.sh`) submits the job automatically when `/lustre/work/kristero10`, the catalogue and `qstat`
are back. The results below come from the identical code run locally (20 threads, exact Julia 1.12.2 and
the same XGPaint source); the earlier full-sky pipeline showed the local and idark runs of this painter to
be identical, and the cluster output will be checked against `output_local/raw_cl_radius_series.csv` when
it exists.

## 3. Results

Local run: 85,224,251 halos painted in 7512 s on 20 threads (`output_local/provenance.txt`); figures
`figures/tsz_truncation_radius_series.{png,pdf}` (spectra and three ratio panels) and
`figures/tsz_truncation_radius_context.png` (CLASS-SZ radius dependence; 1h/2h and WebSky-setup references);
tables `figures/radius_series_table.csv`, `figures/radius_series_summary.json`.

### 3.1 Convergence of the map with the truncation radius

Ratio of the binned D_ell of the N R200c sphere map to the 5 R200c map (last column: old projected-disc
4 R200c convention over the 4 R200c sphere):

| ell | N = 1 | N = 2 | N = 3 | N = 4 | projected 4 / sphere 4 |
|---|---|---|---|---|---|
| 32 | 0.50 | 0.84 | 0.95 | 0.986 | 1.013 |
| 50 | 0.57 | 0.90 | 0.975 | 0.995 | 1.008 |
| 100 | 0.71 | 0.94 | 0.978 | 0.993 | 1.008 |
| 300 | 0.79 | 0.96 | 0.988 | 0.997 | 1.004 |
| 1000 | 0.84 | 0.98 | 0.997 | 0.9994 | 1.001 |
| 3000 | 0.95 | 0.999 | 1.000 | 1.000 | 1.000 |

- The power converges quickly with N: going from 4 to 5 R200c changes the spectrum by 0.7% at ell = 100,
  0.3% at ell = 300 and < 0.1% above ell = 1000 (1.4% at ell = 32). 3 R200c is within 2.2% at ell >= 100 and
  0.3% at ell >= 1000. 2 R200c loses 4-6% at ell = 100-300 and 2% at ell = 1000; 1 R200c is not usable
  (30% low at ell = 100, 16% at ell = 1000, 5% even at ell = 3000).
- Mean Compton y of the maps: 6.50e-7, 8.97e-7, 9.79e-7, 1.014e-6, 1.032e-6 for N = 1..5 (projected 4 R200c:
  1.031e-6). The mean converges more slowly than the power (N = 4 is 1.7% below N = 5) because the outer
  gas is smooth and only adds power at ell < 100.
- The projected-disc convention at 4 R200c lies 0.8% above the 4 R200c sphere at ell = 100 and < 0.1% above
  ell = 2000, as found on 2026-09-17: it counts the gas beyond 4 R200c along the line of sight but only inside
  the disc, which is numerically close to the sphere.
- Painting cost scales with the disc area: pixel updates per map 0.29, 1.15, 2.58, 4.59, 7.17 x 1e9 for
  N = 1..5, i.e. N = 3 costs 56% of N = 4, and N = 5 costs 156% of N = 4.

### 3.2 Against CLASS-SZ with the same truncation (x_outSZ = N)

Ratio map(N) / CLASS-SZ(1h+2h, x_out = N), HalfDome setup:

| ell | N = 1 | N = 2 | N = 3 | N = 4 | N = 5 |
|---|---|---|---|---|---|
| 100 | 0.99 | 1.01 | 0.99 | 0.99 | 0.99 |
| 300 | 1.00 | 0.98 | 0.97 | 0.97 | 0.97 |
| 1000 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| 3000 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |

- CLASS-SZ reproduces the radius dependence of the maps almost exactly (context figure, left panel): its own
  x_out = N / x_out = 5 ratios at ell = 100 are 0.71, 0.93, 0.98, 0.99 against 0.71, 0.94, 0.98, 0.99 for
  the maps, and both saturate at x_out = 3-4. The remaining map / CLASS-SZ offset (0.90-1.00 over ell
  100-3500, the same for every N) is therefore the known mass-function / catalogue difference between
  HalfDome and Tinker08 documented on 2026-09-17, not a truncation effect.
- Above ell ~ 3500 the ratio rises (1.07 at ell = 4850, 1.21 at ell = 6100) identically for every N: the
  CLASS-SZ curves turn over at high ell while the NSIDE 4096 maps, after the pixel-window correction, keep a
  rising ell^2 tail from the many barely resolved low-mass halos. This is independent of the radius and was
  already excluded from the quoted ranges in the 2026-09-17 comparison.

### 3.3 Against the WebSky public y map

| ell | sphere 4 / WebSky | sphere 1 / WebSky | CLASS-SZ (WebSky cosmology, x_out = 4, M > 1e12 Msun/h) / WebSky |
|---|---|---|---|
| 100 | 1.85 | 1.32 | 1.87 |
| 300 | 1.58 | 1.25 | 1.64 |
| 1000 | 1.29 | 1.08 | 1.43 |
| 3000 | 1.33 | 1.26 | 1.39 |

- WebSky has 25-45% less y power than the HalfDome maps at every N, and also 30-45% less than CLASS-SZ run in
  WebSky's own cosmology and halo range (the HalfDome mass floor changes that CLASS-SZ spectrum by only 0.2%).
  Even the 1 R200c HalfDome map is above WebSky at all ell. WebSky's own truncation at 4 R_Delta is not the
  cause: changing 4 to 5 R200c moves the power by < 1%. The deficit sits in WebSky's halo population and/or
  its profile implementation (its quoted growth of the per-halo y with the cutoff radius, 1.9 / 2.4 / 2.7 for
  R_max = 2 / 3 / 4 at 1e14 Msun, z = 0, is not reproduced by sphere, disc or cylinder integrals of the
  Battaglia12 fit, which give at most 1.4 / 1.6 / 1.7). WebSky's mean y is 1.07e-6, close to the HalfDome
  4-5 R200c maps (1.01-1.03e-6), so the mismatch is in the clustering / one-halo shape rather than the
  total signal. WebSky is therefore not a useful yardstick for the radius choice.

### 3.4 Recommendation

Keep N = 4 R200c (sphere) as the production painting radius. It is within 0.7% of the 5 R200c result at
ell = 100 and within 0.1% above ell = 1000, matches the CLASS-SZ default `x_outSZ = 4` convention and WebSky's
choice, and costs 36% less than N = 5. If painting cost matters (large SBI training sets), N = 3 is acceptable
with a known < 2.2% (ell >= 100) / < 0.4% (ell >= 1000) low bias in the power. N = 2 and below are not
converged. The radius does not help with the WebSky offset, which is not a truncation effect.

## 4. Files

- `paint_radius_series.jl`, `run_radius_series.pbs`, `autosubmit.sh` (cluster side), `compute_class_sz_radius_series.py`,
  `compute_class_sz_websky_setup.py`, `plot_radius_series.py`.
- `class_sz/class_sz_tsz_xout{1..5}.npz`, `class_sz/class_sz_tsz_websky_setup_*.npz`, `websky_tsz_cl.npz`.
- `output_local/raw_cl_radius_series.csv`, `output_local/provenance.txt`, maps `output_local/halfdome_y_*_nside4096.fits`
  (not committed); figures and tables in `figures/`.
