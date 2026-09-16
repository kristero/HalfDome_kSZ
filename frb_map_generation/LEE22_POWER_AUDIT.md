# Lee22 versus Battaglia16: local power-spectrum audit

Date: 2026-09-09. This is a diagnostic report, not a corrected production run.

## Conclusion

The sharp terminal upturn in the sparse-ray Lee22 spectrum is not reproduced
by the complete-sky DM map. The current estimator has a finite-population
shot-noise bias, its highest-multipole bands have large sampling fluctuations,
and the plotting code hides negative estimates. Separately, the angular-profile
cache is inaccurate for very nearby halos. The larger broad-band Lee22 signal
already exists in the maps; its physical normalization still needs validation.

No production source, XGPaint source, original cache, map, notebook, or spectrum
was modified. No cluster job was submitted. Only diagnostic scripts and this
report were added; generated diagnostics are in:

`frb_map_generation/outputs/lee22_power_audit_20260909/`

## Inputs and tests actually run

- Existing NSIDE=4096 complete-sky Lee22 and Battaglia16 maps, source z=1,
  projected aperture 3R200c, complete resolved M200c mass range. Their provenance
  records 29,130,459 foreground halos from a complete 85,224,251-row scan.
- Existing one-million-unique-pixel ray catalogues, seed 42, and the completed
  `tsz_frb_1m_z1_r200c_aperture_spectra.h5` product.
- Saved complete-map spectra through ell=8192. Their source FITS sizes and
  nanosecond modification times were checked against the spectrum-cache metadata.
  These spectra were reused, not falsely represented as freshly repainted maps.
- Eight independent one-million-pixel samples from each complete map, using
  the same test seeds in both models: 16 new harmonic transforms in total.
  All halo contributions remain in these maps. Sampling rays is not subsampling halos.
- An exact finite-population identity test: all 792 five-pixel subsets of a
  12-pixel toy sky, at ell=1,2,3. Passed at 1e-12 tolerance.
- 2,112 production-cache/direct-profile evaluations: two profiles, 8 masses,
  12 redshifts, 11 projected radii. Masses cover log10(M/Msun)=12.5 to 15.5,
  including the observed catalogue minimum. Test z includes 0.0022 through 4.
- A separate full catalogue scan selected all 774 halos with z<0.02 plus 60
  mass-ordered representatives at 0.02<=z<0.1. These 834 numerical test halos
  were evaluated at 0.04, 1 and 3R200c in both models: 5,004 evaluations.
- 28 independently segmented Lee22 LOS integrations, plus enclosed-electron
  budgets and direct 3D density profiles. The segmented/default LOS ratio
  differs from one by at most 7.2e-12 on this grid.

The fresh sparse-ray/full-map numerical checks below are for 3R200c, not a new
5R200c map validation. No beam, instrumental noise, or pixel-window correction
was introduced. The tSZ sky was not repainted for this DM-specific diagnostic.

## 1. Confirmed estimator and display bugs

In `compute_local_tsz_frb_1m_aperture_spectra.py`, `sparse_dm_map` uses

    q_i = DM_i - sample_mean(DM)
    map[p_i] = (Npix/Nray) q_i; unsampled pixels are zero
    N_Poisson = 4*pi*mean(q_i**2)/Nray

The subtraction is performed at line 747. But the inputs enforce **unique**
HEALPix pixels: sampling is without replacement, not an independent Poisson
point process. At NSIDE=4096, P=201,326,592 and n=1,000,000, so n/P=0.00496705.

For known full-map-mean-subtracted values on this fixed pixel population,
the spherical-harmonic addition theorem and inclusion probabilities give

    E[C_sparse] = a*C_full + N_finite
    a = P*(n-1)/(n*(P-1))
    N_finite = (4*pi/n) * ((P-n)/(P-1)) * Var_full(DM)

The ordinary Poisson subtraction over-subtracts by about 0.497% of a very
large noise term. This is small relative to the noise, but large relative to
the actual small-scale DM signal. The diagnostic contains both an exact
known-mean test and a sample-only approximate correction; these are labelled
separately. Estimating the sky mean introduces additional finite-sample terms,
so the sample-only curve is not presented as an exact production replacement.

At the last band, effective ell=7631.97, D_ell values in (pc cm^-3)^2 are:

| Model | Complete-map reference | Saved sparse subtraction | Sample finite-population diagnostic |
|---|---:|---:|---:|
| Battaglia16 | 3,562.5 | -9,083.0 | 4,299.4 |
| Lee22 | 3,956.0 | -55,184.3 | 19,045.7 |

All five final bands in the saved 3R200c auto-spectra are negative. Lines
411-412 select only `dl > 0` before drawing a logarithmic plot. This conceals
the negative bands and makes Lee22's last positive fluctuation near ell=3582
look like a terminal physical upturn. The diagnostic plot retains negative
values using a signed/symlog axis. Taking an absolute value or clipping at zero
would not fix this problem.

After the finite-population correction, eight independent ray samples give
last-band means and sample standard deviations of approximately:

- Battaglia16: 2,576 +/- 1,839, versus complete-map 3,563.
- Lee22: -10,819 +/- 27,580, versus complete-map 3,956.

These are sampling spreads, not precision cosmological confidence intervals.
Eight draws are insufficient to establish detailed coverage or convergence.
In particular, the Lee22 high-ell estimate remains noise dominated after the
bias correction. The large spread explains why the original seed still has
an upturn after correcting the expected bias. Fixing that bias alone does not
recover a precise one-million-ray high-ell measurement.

The y-DM cross-spectrum has no additive DM auto shot-noise subtraction, so
this specific bias is not an instruction to subtract noise from the cross.
However, any derived r=y-DM/sqrt(C_yy*C_DM) using an unstable or negative
corrected DM denominator is unreliable. Values above one must not be
interpreted as physical correlation coefficients or repaired by clipping.

Plot: `sparse_vs_dense_power_audit.png`.

## 2. The ray values themselves reproduce the painted sky

Sampling the existing FITS maps at the exact production ray pixels gives:

| Model | RMS difference [pc cm^-3] | Largest absolute difference [pc cm^-3] |
|---|---:|---:|
| Battaglia16 | 2.080e-6 | 6.440e-4 |
| Lee22 | 1.064e-6 | 3.904e-4 |

This checks indexing, RING ordering, map/ray consistency and the actual ray
values. It does not independently validate profile physics because both paths
share the production profiles/caches.

The dense spectra have no corresponding sharp high-ell feature. Their Lee22/B16
ratios are about 11.13 at ell=94.7, 6.08 at ell=1068, 3.05 at ell=3079, and
1.11 at ell=7632. Thus the large *broad-band* enhancement is a different issue
from the anomalous final sparse-ray point.

## 3. Confirmed low-redshift cache interpolation problem

The generator interpolates log(DM) on log(theta), linear redshift and log(mass).
Lee22 uses a bounded linear spline, preventing the earlier catastrophic cubic
overshoots. That does not guarantee accuracy between widely separated nodes.
At fixed angle the physical impact radius changes rapidly near z=0, where
a linearly spaced redshift grid is a poor representation.

For an actual catalogue halo, zero-based row 85,224,238:

    M200c = 1.00210723e14 physical Msun
    z = 0.00443673
    R_perp = 3R200c
    Lee22 cached DM = 28.8697 pc cm^-3
    Lee22 direct DM = 5.79179 pc cm^-3
    cached/direct = 4.98459

This is a numerical discrepancy, not new physical feedback. Battaglia16 also
has nearby-cache errors: up to a factor 4.32 at 3R200c among the tested actual
nearby halos. Those maxima concern low-z outer columns, not central values.
At 0.04R200c the corresponding tested actual-halo cache discrepancies are
below 0.7%. On the regular Lee22 grid at z>=0.1 and radii<=3R200c, errors
are below 0.33%.

The catalogue contains 84 halos below z=.01, 690 at .01-.02, 9,104 at .02-.05,
and 69,424 at .05-.1. Low counts alone do not establish a negligible power
contribution: nearby halos have large angular footprints. This audit did not
repaint those contributions, so their exact change to C_ell remains unmeasured.
The complete-map reference above therefore diagnoses sampling artefacts, not
a fully cache-corrected physical spectrum.

A suitable numerical repair is a local radius-scaled table in R_perp/R200c
or accurate direct low-z evaluations, with multi-point validation in the
actual selected halo domain. Neither requires editing XGPaint. This should
produce newly tagged caches/maps rather than overwrite the legacy products.

Data: `nearby_halo_cache_vs_direct.csv`, `production_cache_vs_direct_profiles.csv`.

## 4. Why the present Lee22 model can have more power

Source z=1 is not halo z=1. Both maps integrate the foreground population
throughout 0<z_halo<=1. For M200c=1e14 physical Msun, at 0.04R200c,
the current direct projected observed columns are:

| Halo z | Battaglia16 DM | Lee22 DM | Lee22/B16 |
|---|---:|---:|---:|
| 0.1 | 846.02 | 2185.06 | 2.58 |
| 0.5 | 778.04 | 972.50 | 1.25 |
| 1.0 | 804.59 | 517.48 | 0.64 |

These are values of the **current implementation**, not proof of the intended
paper normalization. The new 3D/column figure checks physical density as well
as its LOS projection, without conflating n_e, electron column and observer DM.

Halo power depends on the squared Fourier-transformed projected profile,
integrated over mass and redshift, plus correlations between halos. A central
density comparison at a single mass/redshift cannot fix its overall ordering.
The maps also have different means: 373.73 for Lee22 and 203.88 pc cm^-3 for
B16. The requested spectra are absolute-DM spectra, not spectra of fractional
contrast DM/<DM>-1. Dividing by the respective mean squared would change the
power ratio by a factor 3.36, but would change the observable and was not done.

The previously saved mass/redshift profile-grid output is not a reproducible
reference for the current environment: it gives Lee22/B16=0.366 at M=1e14,
z=1, R_perp=.04R200c, versus about .643 now. Its provenance does not record the
XGPaint source hash needed to identify the historical cause. Current direct
B16 and its production cache agree very closely away from the low-z problem;
there is no demonstrated global current B16 cache/direct mismatch.

Plots: `physical_density_and_column_check.png` and
`current_production_profile_comparison.png`.

## 5. Additional normalization warning: not yet a verified physical model

The archived Lee22 preprint was visually checked at its equations (9)-(12)
and no-concentration fit table (Table 8 in that version, Table A2 in the
published version). The listed coefficients, direct GNFW exponent, and the
printed 1/X_H normalization are present in the local implementation.
Reference: https://arxiv.org/abs/2205.01710 and DOI 10.1093/mnras/stac2602.
This is not an independent validation against the authors' numerical code.

Integrating the current Lee22 n_e inside R200c and converting to ionized-gas-
equivalent mass with primordial fully ionized H/He gives **3.27 times f_b*M200c**
for M200c=1e14 Msun, z=0. Restricting to .04<r/R200c<1 still gives 3.267, so
the central extrapolation is not the explanation. Current B16 gives about .631.
At z=.5 and 1, Lee22 gives 1.026 and .447 respectively.

This is a serious normalization/fit-interpretation flag, not a strict theorem
that every halo must have exactly the cosmic baryon fraction. Do not interpret
the broad Lee22 enhancement as a confirmed feedback prediction yet. Resolve
the dimensional normalization, mass-unit/pivot conventions and the authors'
reference evaluation before choosing a different amplitude. In particular,
the local implementation assigns the unbroken n0 the Eq. (11) 1e14 Msun pivot,
while xc and beta use the fitted mass break; the paper also discusses a shared
Mcut in the extended family. This merits checking against author code rather
than silently switching conventions or forcing agreement with Battaglia16.

The requested 3R200c aperture is also beyond the radial fitting interval
.04-1.34R200c. Here it remains a projected painting aperture with the existing
LOS integral, not a newly imposed spherical truncation. No concentration,
mass cut, amplitude rescaling, minimum DM, or clipping was added.

Data: `ionized_gas_budget_and_los_checks.csv`, `three_dimensional_density_profiles.csv`.

## New files and how to reproduce

- `audit_lee22_sparse_power.py`: input checks, exact toy-sky regression,
  independent one-million-ray tests, finite-population diagnostics, signed
  spectrum plot and machine-readable statistics.
- `audit_lee22_profile_cache.jl`: reuse the production loader read-only, compare
  caches to direct quadrature, test actual nearby halos, integrate electron
  budgets and export physical 3D densities.
- `summarize_lee22_profile_audit.py`: scan the catalogue for numerical test
  points and generate readable profile comparisons.
- `LEE22_POWER_AUDIT.md`: this report.

Run as the normal WSL account (`kn18001` here), not root. The following uses the
existing local environment and does not rebuild or replace production data:

```bash
cd /home/cbllover/HalfDome
FRB_RUN="$PWD/frb_map_generation/outputs/tsz_frb_1m_z1_r200c_apertures"
export JULIA_DEPOT_PATH="$FRB_RUN/.julia_depot:/home/kn18001/.julia"
JULIA_BIN=/home/kn18001/.juliaup/bin/julia
PYTHON_BIN=/home/cbllover/miniconda3/bin/python

"$JULIA_BIN" --project="$FRB_RUN/local_julia_env" frb_map_generation/audit_lee22_profile_cache.jl
"$PYTHON_BIN" frb_map_generation/summarize_lee22_profile_audit.py --select-nearby
# Run again after selecting actual nearby halos, on a first-time audit.
"$JULIA_BIN" --project="$FRB_RUN/local_julia_env" frb_map_generation/audit_lee22_profile_cache.jl
"$PYTHON_BIN" frb_map_generation/summarize_lee22_profile_audit.py

OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" \
  frb_map_generation/audit_lee22_sparse_power.py --mc-seeds=8
```

The three main PNGs were visually inspected. Production corrections are still
pending: fix estimator/display logic, repair/validate low-z interpolation, and
resolve the normalization question before interpreting a regenerated model.
