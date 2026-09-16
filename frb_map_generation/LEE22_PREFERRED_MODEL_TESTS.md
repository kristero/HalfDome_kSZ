# Lee22 preferred fit, radius-cache repaints, and observational references

## Scope and decisions (13 September 2026)

- DM: all resolved foreground halos, `0 < z_halo <= 1`, source plane exactly
  `z_source=1`, NSIDE4096, physical M200c and R200c. No science mass cut or
  10,000-halo subsample. There are 29,130,459 selected halos from 85,224,251
  catalogue rows, with M200c between 7.327085e12 and 3.785236e15 Msun.
- Test three density prescriptions: installed XGPaint Battaglia16, the existing
  Lee22 no-concentration implementation, and the new preferred Lee22
  concentration-dependent fit evaluated with Duffy2008 median concentrations.
- The newly approved physical boundary is **spherical 3R200c**, for all three
  density models. No 5R200c run was submitted. Two additional corrected
  projected-aperture controls preserve the old LOS definitions for B16 and
  legacy Lee22; these isolate interpolation changes from boundary changes.
- No instrumental beam or noise in these numerical DM tests. Each complete-map
  spectrum is compared with 16 matched samples of one million distinct
  HEALPix pixel centers, not with independently generated cosmological skies.
- Observational work remains **halo-only, a partial prediction**. The tSZ
  pressure prescription stays **Battaglia12**, using the full tSZ lightcone.
  The published constant-temperature curves are references, not a replacement
  for Battaglia12 or new isothermal HalfDome maps.
- Observational FRB mocks will use **uniform sightlines carrying the measured
  source redshifts**, as subsequently selected by the user. No simulated host
  halo weighting or source-environment correlation is added. This does not
  change the fixed-z=1 numerical benchmarks above.
- Physically, `y = sigma_T/(m_e c^2) integral Pe dl`, whereas halo DM is
  `integral ne dl/(1+z_halo)`. Holding pressure fixed isolates the effect of
  the electron-density model in the cross statistic. This deliberately mixes
  separately calibrated pressure/density fits; it is not a jointly fitted
  hydrodynamic gas solution or a fixed-temperature constraint.
- XGPaint package sources, old maps, old caches and old results were not edited
  or overwritten by this experiment. All changed model/painting code is local
  to `frb_map_generation`.

## Files changed

Modified existing scripts:

1. `lee2022_frb_dm_profile.jl`: common Lee22 interface, preferred density fit,
   Duffy08 concentration, explicit divergent-tail rejection and provenance.
2. `generate_halfdome_z1_dm_mass_windows.jl`: shared runtime selection supports
   `lee2022_concentration_mode=duffy2008`; existing default remains `none`.
3. `paint_halfdome_matched_profile_dm_map.jl`: selectable external radius cache,
   spherical boundary, per-halo precomputation, validation before painting and
   provenance. Its generic hook remains compatible with the existing tSZ painter.

New scripts and documentation:

4. `radius_scaled_dm_cache.jl`: dimensionless projected-shape interpolation,
   exact per-halo physical normalization, finite-chord edge treatment, signature
   checks and independent direct-quadrature validation.
5. `test_lee22_preferred_profile.jl`: coefficient/unit/regression checks and
   13,230 density/column/extrapolation diagnostic rows.
6. `analyze_radius_dm_tests.py`: complete-map spectra and 16-realization
   finite-population sampling tests; profile diagnostic plots.
7. `run_lee22_radius_repaint_tests.pbs`: two test branches; default resources
   remain mini, 26 CPUs, 48 GB, walltime 23:59:00.
8. `compare_radius_repaint_results.py`: old versus corrected spectra and
   spherical versus projected controls, with linear percentage panels and
   metadata checks; independent unit-free gas-budget validation and 2-D
   isolated-halo views of the direct profile grid.
9. `prepare_takahashi_observational_inputs.py`: published 133-source table
   extraction and the survey/mock input inventory; no invented DMs or cuts.
10. `digitize_frb_observational_figures.py`: approximate published data vectors,
    error bars, temperature-reference curves, extraction QA and provenance.
11. This document, `LEE22_PREFERRED_MODEL_TESTS.md`.

The previously written `audit_lee22_sparse_power.py` is used unchanged.
Unrelated working-tree changes and notebooks were left alone in this turn.

## Density model and physical boundary

The new preferred fit follows equations (9), (10), (12) and Table 3 of the
[Lee22 preprint](https://arxiv.org/html/2205.01710), with

```text
x = r / R200c
ne = n200 n0 (x/xc)^gamma [1 + (x/xc)^alpha]^(-beta_prime)
n200 = 200 rho_critical (Omega_b/Omega_m) / (X_H m_p)
alpha = 1; gamma = -0.3; X_H = 0.76
Mcut = 10^13.75 Msun/h
```

Each parameter uses the common Mcut in the printed extended equation (12)
and concentration factor `(c/10)^alpha_c`:

| Parameter | Amplitude | Mass slope below / above Mcut | Redshift slope | Concentration slope |
| --- | ---: | ---: | ---: | ---: |
| n0 | 15.7 | 0.87 / 0.87 | -2.09 | 0.63 |
| xc | 2.2 | -0.06 / -1.45 | -0.74 | -1.37 |
| beta_prime | 7.5 | 0.24 / -1.10 | -0.39 | -1.11 |

The [Duffy08](https://arxiv.org/abs/0804.2486) full-sample NFW 200-critical
relation is `c=5.71 (M_physical*h/2e12)^(-0.084) (1+z)^(-0.47)`.
This is a WMAP5 median fit, calibrated over z=0..2. Lee22 used measured Klypin
concentrations; substituting Duffy is an explicit approximation and introduces
neither the scatter nor its covariance with gas structure. Tests at z=3 and 4
are extrapolation diagnostics, not calibrated predictions.

**Important legacy convention:** the old no-concentration model retains its
existing n0 pivot of 1e14 physical Msun, while its xc/beta use Mcut. The new fit
uses the common Mcut literally printed in Eq. (12). These are not identical
normalization conventions. No normalization repair was silently mixed into
the interpolation control. The legacy curve should not be called an
independently verified exact reconstruction of the published A2 fit.

For a spherical boundary, the observer-frame column is

```text
DM(b) = 2/(1+z_halo) integral_0^sqrt[(3R200c)^2-b^2]
                         ne(sqrt[b^2+l^2]) dl,       b < 3R200c
DM(b) = 0,                                        b >= 3R200c
```

The integration length is in physical units and the final DM is pc cm^-3.
The factor `1/(1+z_halo)` is applied once. A projected-footprint-only run
instead keeps the original long/infinite LOS integral and only restricts the
sky footprint. Those operations are physically different.

No package painting cutoff is applied a second time: the external painter
chooses the footprint and evaluates the column. For spherical tests the
screen-plane radius is `b/R200c=tan(theta)/tan(theta200c)`, consistent with
the `atan(R/DA)` footprint. Projected controls retain `theta/theta200c` to
avoid silently changing their old geometry.

## Numerical safeguards are not physical truncations

- The cache stores the logarithm of a positive **dimensionless shape** on
  axes `(log(R/R200c), z, logM)`. Dimensional amplitude and angular radius are
  evaluated per halo. Mass-break knots are included explicitly.
- The finite-sphere cache stores the mean density along the chord. Its exact
  shrinking chord length is multiplied back outside interpolation. This
  gives a true zero at the boundary without interpolating log(0), smearing
  a zero floor into the interior, or creating an artificial bright rim.
- The innermost grid radius is 1e-7 R200c; the unresolved central column uses
  that finite limit. This is **not** a minimum DM. Zero and arbitrarily small
  positive columns near the boundary remain possible. An additional local
  direct-quadrature check at b=0 versus b=1e-7 R200c (both Lee22 models,
  masses 1e13, 1e14 and 10^15.5 Msun, z=0 and 1) found a maximum difference
  of 0.00239%.
- Plain HDF5 arrays avoid Julia/JLD2 serialized-type incompatibilities. Source,
  runtime and installed-XGPaint signatures reject mismatched cache reuse.
- Every new cache is checked at 600 off-grid points, including near-edge
  radii, against independently evaluated physical LOS quadrature. Failure
  above 1% aborts painting; values are not clipped or renormalized.
- The 1e8 pc cm^-3 guard remains a corruption alarm, not a physical upper
  limit, selection function or PDF cutoff. No value is reduced to that guard.
- Ring locks prevent simultaneous halos from losing contributions in shared
  pixels. Floating-point summation order can vary at roundoff level.

## Sampling test

Complete maps are mean-subtracted and transformed with `lmax=8192`, `iter=0`,
no beam and no pixel-window deconvolution. Bandpowers are `(2ell+1)` weighted
averages of C_ell, displayed as `ell_eff(ell_eff+1) C_band/(2pi)`.

For P pixels and N distinct uniformly selected pixel centers, the inverse-
sampling-weight map has

```text
E[C_raw] = A C_full + B C_Poisson
A = P(N-1) / [N(P-1)]
B = (P-N) / (P-1)
C_corrected = (C_raw - B C_Poisson) / A
C_Poisson = 4pi/N times sample_mean[(DM - known_full_map_mean)^2]
```

An exhaustive 12-pixel test verifies this identity. The million-ray numerical
test deliberately uses the **known full-map mean**. It is not an estimator
available from a 71-source observed catalogue. Negative estimates are retained.
This correction removes the without-replacement mean bias; it does not remove
sampling variance. The 16 seeds quantify sampling on one sky, not cosmic
variance or the host/IGM/measurement covariance of real FRBs.

## Cluster execution and local products

Submitted snapshot (do not edit while the job runs):

```text
/home/kristero10/HalfDome_kSZ/frb_map_generation/cluster_tests/lee22_preferred_20260913_v1
```

Cluster output root:

```text
/lustre/work/kristero10/frb_data/lee22_preferred_radius_tests_20260913_v1
```

- `597062.idark`, mini2: projected3 controls **complete, exit 0**.
- `597061.idark`, mini2: spherical3 **complete, exit 0**, finished
  2026-09-13 12:38:42 UTC. All three maps and all 16-realization tests completed.
- Each job requests 26 CPUs, 48 GB and 23:59:00; the checked-in default queue
  remains mini. mini2 was supplied on submission, per the user's authorization.
- `logs/<kind>_<job>.log`, `<kind>_stage.txt`, `<kind>_exit_status.txt` show the
  actual stage and exit. A disappeared qstat entry alone is not completion proof.
- Complete FITS maps stay in cluster `maps/`; caches in `cache/`. Small
  `analysis/` products, validation CSVs and map provenance are downloaded
  and checked against SHA256. No need to download five approximately 1.6 GB maps.
- All 40 files in `final_results_sha256.txt` verified after the final download.
  The local combined and boundary figures have been regenerated with all
  five completed maps and visually inspected; these are no longer partial plots.

Local results:

```text
/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_preferred_20260913/cluster_results
```

From the repo root, regenerate comparisons after downloading completed products:

```bash
python frb_map_generation/compare_radius_repaint_results.py \
  --output-dir frb_map_generation/outputs/lee22_preferred_20260913/cluster_results/analysis \
  --baseline-dir frb_map_generation/outputs/cluster_lee22_numerical_tests_20260910_v1 \
  --profile-images
```

The combined figures explicitly show missing branches as pending when run
before completion; they must be regenerated after new results arrive.
`spherical3_projected_halo_images_z0.png` through `...z4.png` show seven
mass rows (log10 M=12.5..15.5) and three model columns at each halo redshift,
with one common logarithmic projected-DM colorbar per figure. These are
radially symmetric display interpolations of the direct-quadrature grid,
not new full-sky maps or additional independent convergence evidence.

For a future isolated rerun, copy the relevant scripts listed above plus the
unchanged helper to a **new** snapshot and submit the PBS file with
`TEST_CODE_DIR`, `TEST_OUTPUT` and `TEST_KIND=projected3` or `spherical3`.
Do not resubmit into this completed output root or overwrite the baseline.
The cluster uses Julia1.6.2 and its existing SZ Python3.6 environment; no local
Julia1.8 requirement is imposed. Local smoke tests used Julia1.12.2.
The existing Battaglia12 painter also imports successfully after the shared
hook change; at M=1e14 Msun,z=0.5,theta=1e-4 rad, the hook and unchanged
direct model both return y=4.692765152e-6. This is a scalar compatibility
check, not a new full-lightcone tSZ run.
The repo HEAD at submission was `f221de0e379694870b0f88746d9ef406f3eee820`;
these changes were not committed. The uploaded source hashes, not HEAD alone,
identify the experiment. Local postprocessing/documentation were added later;
the painter also received an indentation-only cleanup after submission. The
running snapshot was left immutable.

## Completed results

| Map | Mean DM [pc cm^-3] | Max cache error | Highest-band one-catalogue scatter / full-map signal |
| --- | ---: | ---: | ---: |
| Battaglia16 projected3 | 201.4164 | 0.0300% | 0.418 |
| Battaglia16 spherical3 | 175.6346 | 0.0266% | 0.409 |
| Lee22 legacy projected3 | 369.2481 | 0.1615% | 6.221 |
| Lee22 legacy spherical3 | 340.7766 | 0.0762% | 6.217 |
| Lee22 preferred + Duffy, spherical3 | 582.4911 | 0.1180% | 38.468 |

Highest band has effective ell about 7632. Relative to the previous complete
maps, correcting the angular cache changes that band's power by only -0.0245%
(B16) and -0.0115% (legacy Lee22). Both mean DMs fall by about 1.2%.
Low multipoles can change much more; inspect the full percentage panels.
Thus the old cache was inaccurate for some nearby halos, but it is **not**
the explanation for the highest-multipole sparse-ray peak.

The complete legacy Lee22/B16 power ratio falls from about 11.16 at ell~95 to
6.08 at ell~1068 and 1.11 at ell~7632. Neither complete map develops the
sparse-estimator terminal upturn. Even the 16-catalogue Lee22 mean differs
from truth by roughly one SEM in the highest bin, because its scatter is huge.

For B16, switching to the finite sphere lowers mean DM by 12.80%, while
highest-band power changes by +0.306%. This is a boundary/geometry experiment,
not a claim that every angular scale is unchanged.
For legacy Lee22, the sphere lowers mean DM by 7.71%; highest-band power
changes by -0.121%.

The preferred+Duffy spherical spectrum is about 164 times B16 at ell~95,
13.78 times at ell~1068 and 2.11 times at ell~7632. It does not develop the
sparse-catalogue terminal upturn in its complete-map spectrum either. Its
highest-band single-catalogue scatter is **38.5 times** the complete-map
signal, and even the 16-realization mean has SEM about 9.6 times the signal.
A sparse one-million-ray high-ell peak is therefore especially unreliable
for this model. Matching the Monte Carlo mean within its very large SEM is
not evidence of a precise recovered small-scale spectrum.

### Does 3R200c make the preferred Lee22 fit physical?

It makes the LOS integral finite, but **does not by itself validate the model**.
With these printed coefficients and the Duffy proxy, the outer density exponent
`s=beta_prime-gamma` falls below 1 for all seven tested redshifts at
log10(M/Msun)=15.5. An untruncated LOS then diverges. At M=10^15.5 Msun,z=0,
`s=0.7865`; increasing the finite LOS bound from 1e4 to 1e5 R200c increases
the central column by a factor 1.678. A spherical boundary removes that arbitrary
long-LOS dependence.

However, integrating the literal preferred fit inside **1R200c** gives a
gas-equivalent mass of 4.80 times `fb M200c` at 1e14 Msun,z=0, and 35.3 times
at 10^15.5 Msun,z=0. The legacy convention gives 3.27 and 3.98 respectively.
The cosmic allotment `fb M` is a reference, not a strict upper bound on every
individual halo's baryon fraction. The extreme preferred case is stronger:
35.3 times fb is about **5.58 times the entire stated halo mass** inside
R200c, which cannot be a consistent gas component of that same M200c halo.
This diagnostic assumes fully ionized primordial gas with
`mu_e = m_p / [(1+X_H)/2]`; it is not an imposed rescaling.
An independent SciPy check cancels all dimensional quantities analytically:
`Mgas/(fb M200c)=3*n0*integral_0^1[f(x)*x^2 dx]/(X_H*0.88)`.
Across 98 mass/redshift/model cases it agrees with the Julia unit-bearing
calculation to a maximum relative difference of 2.2e-11. The large gas budget
is therefore not caused by a Julia pc/cm/Msun conversion in this diagnostic.
At 1e14 Msun,z=0,b=0.001R200c, spherical columns are approximately
1062 (B16), 3198 (legacy Lee22), and 4781 (preferred+Duffy) pc cm^-3.

Consequently the expectation that the currently implemented Lee22 density
must always be below B16 is not satisfied by direct quadrature either.
The finite-sphere test cannot repair a normalization/pivot/concentration
convention issue inside R200c. The calibration interval was only
0.04..1.34 R200c and 1e13..10^14.8 Msun/h. The supplied extrapolation and
concentration proxy must be resolved/checked against an author reference
before calling the preferred map a physically validated observation prediction.
No arbitrary gas renormalization has been applied to force agreement.

## Observational data prepared, not an observation-matched mock yet

Local directory:

```text
/home/cbllover/HalfDome/frb_map_generation/outputs/observational_comparison_inputs_20260913
```

`takahashi2025_v2_table8_133_frbs.csv` is extracted from the published LaTeX
table, including sky positions, observed DM, redshift and survey flags.
The table contains 133 sources. It gives 74 Planck-flagged sources, with the
three explicitly named cluster-host exclusions leaving **71**; ACT has **31**.
The **131**-source DM-z sample excludes two negative extragalactic DMs after
NE2001+YT20 subtraction. Those two have not been guessed or dropped here.
The paper's rounded coordinates are not exact author positions at mask edges.

`observational_input_inventory.json` records the beam, mask, weighting,
selection, random-position subtraction and covariance requirements. Public
Planck PR2 maps/noise products, ACT DR6 maps/masks/beams/simulation locations
were located; multi-GB maps have not yet been downloaded. The PR4 location
redirects to Zenodo record18405044, but its exact filenames remain unverified.

The requested approximate data are in `digitized/`:

- `takahashi_fig13_approximate.csv`: 24 plotted Planck/ACT points.
- `takahashi_fig14_approximate.csv`: 48 points across four Galactic masks.
- `takahashi_fig15_approximate.csv`: 36 points comparing PR2/PR4/deprojection.
- `medlock_nagai_fig5_approximate.csv`: 42 points across CHIME DM cuts and
  Takahashi ACT/MILCA/NILC.
- `takahashi_fig13_temperature_reference_curves.csv`: eight published curves,
  for Planck/ACT, HMx/TNG and Te=1e7/3e7 K; references only.
- `observational_approximate_datapoints.png`: readable six-panel reference plot.
- Four `*_digitization_QA.png` overlays and `digitization_provenance.json`.

Takahashi's vector symbols and error endpoints were read from its original
figure PDFs, then calibrated against labelled ticks. Medlock Figure5 uses
manually checked coordinates in its 1704x676 source image. Nominal extraction
resolution is 0.10 PDF point or approximately 3 image pixels; occluded grey
symbols are less certain. These are extraction resolutions, not statistical
confidence limits. Values are saved in pc cm^-3, including negative points;
the plotted `1e-5` axis multiplier is accounted for.

Plotted x positions can contain display offsets. The Takahashi tables retain
both displayed theta and the paper's logarithmic bin edges for model averaging.
Repeated MILCA vectors across Figures13/14/15 agree in extracted w to within
2.3e-8 pc cm^-3, below the stated extraction resolution. Reused observations
are **not independent data**. No author covariance or formal chi-square,
likelihood or detection-significance estimate is supplied.

To reproduce the extraction using the source figure assets:

```bash
python frb_map_generation/digitize_frb_observational_figures.py \
  --source-dir frb_map_generation/outputs/observational_comparison_inputs_20260913/source_figures \
  --pdf-tools tmp/pdfs/lee22_takahashi_20260913/pdf_tools \
  --download-sources \
  --output-dir frb_map_generation/outputs/observational_comparison_inputs_20260913/digitized
```

PyMuPDF was installed only into the isolated `pdf_tools` directory; the main
Python environment was not altered. Source archive URLs and figure hashes
are saved in the extraction provenance. Copies of the four original figure
assets are kept in `source_figures/`; `--download-sources` retrieves missing
assets from pinned public arXiv source archives without arbitrary extraction.

### What remains before an observational overlay

The requested papers measure configuration-space **w_yDM(theta)**, not the
noiseless C_ell comparison. Match each actual source redshift (some exceed z=2),
survey beam and footprint, weights and random-subtraction estimator. The
chosen synthetic source placement is uniform within the relevant survey
selection; the source redshifts are taken from the observed samples, not all
set to z=1 or z=2. Source-host clustering is deliberately absent. The
71/31-source cross samples differ from the 131-source DM-z sample. Medlock
Figure5's CHIME panel uses a separate 3455-source parent catalogue and observed
DM cuts; applying those cuts to halo-only DMs would be physically wrong.

Keep Battaglia12 electron pressure for y, and vary only halo electron-density
models for DM. Apply survey beam/noise/masks in a separate observational step,
not to the noiseless numerical controls. Use a halo-only ensemble mean for
model DM residuals; do not subtract a total observed-DM relation from halo-only
DM. Missing diffuse/host/MW scatter means these remain **partial predictions**,
not a complete reproduction of the observed covariance or total extragalactic
DM. No survey mock generation, fitted observation overlay or observational
goodness-of-fit has been claimed as complete.

Primary observational sources:
[Takahashi et al., arXiv2511.02155v2](https://arxiv.org/abs/2511.02155v2),
[Medlock & Nagai, arXiv2608.06455v1](https://arxiv.org/abs/2608.06455v1).
