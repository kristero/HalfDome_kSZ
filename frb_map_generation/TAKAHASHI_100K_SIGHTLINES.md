# Explicit 100,000-sightline HalfDome comparison

Completed 14 September 2026. This is an individual-source calculation, not a
sample of the previous redshift-averaged DM map. Previous results are preserved.

## Results and notebook

Open `FRB_tSZ_Takahashi_comparison.ipynb`, section **New test: 100,000 individual
sightlines**. The earlier full-map calculation remains above it.

Local output root:

```
/home/cbllover/HalfDome/frb_map_generation/outputs/takahashi_100k_20260914
```

Windows Explorer:

```
\\wsl.localhost\myrootfs\home\cbllover\HalfDome\frb_map_generation\outputs\takahashi_100k_20260914
```

Plots in `plots/`:

- `sightlines100k_vs_previous_planck.png`: each density model, new estimate,
  previous mean, digitized Takahashi points, and linear percentage residuals.
- `sightlines100k_vs_previous_act.png`: corresponding ACT-redshift test.
- `source_redshift_and_sky_diagnostics.png`: source histograms, sky distribution,
  and cumulative redshift distributions.
- `halfdome_vs_medlock_fig5.png`: unbeamed HalfDome and digitized BP best-fit
  reference curves. **The BP source kernel is different.**

Actual source-level data are in `rays/`:

- `source_positions.h5`: 100,000 common random positions, zero-based RING
  pixel indices, longitude/latitude, and separate Planck/ACT redshifts,
  observed-source indices, stratum counts and weights.
- `individual_dm.h5`: per-source `battaglia16`, `lee22_legacy`,
  `lee22_preferred` DM vectors and foreground-halo hit counts, separately under
  the `planck` and `act` groups. DM is observer-frame pc cm^-3.
- `annular_y_samples.npz`: annular y averages at those same source positions
  for Planck, ACT and no-beam filters. These are NOT averaged FRB DMs.

`analysis/sightline_comparison.csv` contains estimates, sampling errors, old
means and percentage differences. `sampling_covariances.npz` stores joint
model/bin jackknife covariances; flattening order is model then angular bin.

## What changed physically

Each source gets its own foreground screen:

\[
D_i=\sum_{h:0<z_h\le z_i}
  \mathrm{DM}_{h}(\hat n_i;M_{200c,h},z_h).
\]

Only halos whose projected spherical 3R200c footprint intersects the source
direction contribute. The profile integrates the finite spherical chord, not
an infinite LOS with a projected footprint alone. Observer dilution
`1/(1+z_halo)` is already included in the validated profile helper.

There are **100,000 rays per survey distribution**, not 71/31 rays and not
100,000 halos. Sources are not placed in simulated host halos. All 85,224,251
catalogue rows were scanned; 74,907,259 halos lie within `0 < z <= 2.148` and
were considered before applying each individual source's redshift and angle.
No halo fraction or extra science mass cut is used. Native `halo_mass_m200c`
is Msun/h, converted to physical Msun by division by h=0.68.

The source redshift distribution is the empirical observed distribution, not
a fitted smooth function or a uniform distribution. Every Planck catalogue
redshift has 1,408 or 1,409 mock sources; every ACT redshift has 3,225 or 3,226.
The count rounding is at most one ray per observed source. The correlation
weights each observed-redshift stratum equally, exactly recovering the old
kernel despite this rounding. There is no random jitter in redshift and no
extrapolation beyond the observed range. Each ray has a random direction.

Positions use seed 20260914 and iid uniform NSIDE=4096 pixel-centre draws,
consistent with the old map geometry. The 100,000 draws occupy 99,969 distinct
pixels. Repeated pixels are allowed: different sources can fall on the same
pixel. All profiles share the positions and redshifts; the two survey tests
share positions but have their own redshift assignments. This pairing reduces
Monte Carlo scatter in model differences, so the tests are not independent.

tSZ reuses the verified Battaglia12 full-lightcone repaint: all 85,224,251
halos, all catalogue redshifts (up to 3.8554), existing projected 4R200c tSZ
extent, no noise. The **DM** boundary is spherical 3R200c; the pressure model
and its extent were not changed. Main plots smooth y alone by 10 arcmin for
Planck or 1.6 arcmin for ACT. NSIDE=4096 and lmax=8192 match the old baseline.

## Correlation and sampling errors

For angular annulus b, filter the full y map with the solid-angle-averaged
Legendre window and the appropriate Gaussian beam, and evaluate that filtered
field at each source position: Y_ib. This equals the annular average around
that source at the retained harmonic bandlimit. For observed-redshift stratum g:

\[
\hat w_{gb}=\frac{1}{n_g-1}\sum_{i\in g}
 (D_i-\overline D_g)(Y_{ib}-\overline Y_{gb}),\qquad
\hat w_b=\frac{1}{G}\sum_g\hat w_{gb}.
\]

The n_g-1 denominator corrects the mean-fitting bias for iid random positions.
This subtracts the **halo-only mean at the source redshift**, not an observed
total-DM fit containing missing components. Zero-DM rays are included; their
residuals can be negative. Approximately 22.873% of Planck-distribution and
20.372% of ACT-distribution rays have zero resolved-halo hits in this model.

The code calculates exact delete-one-source jackknife pseudovalues within each
stratum and combines their covariances with squared stratum weights. These
error bars measure finite-direction sampling on this fixed HalfDome sky. They
do not include cosmological ensemble variance, map noise, observational masks,
host/IGM fluctuations, or uncertainties in the observed redshift distribution.
Angular bins are correlated; plotted error bars must not be treated as
independent measurements for a chi-squared fit.

## Comparison with the previous result

In the 10-17.78 arcmin bin, percentage changes relative to the previous mean are:

| Source distribution | Battaglia16 | Lee22 no concentration | Preferred Lee22 diagnostic |
|---|---:|---:|---:|
| Planck | +4.84% | +7.08% | +5.14% |
| ACT | +4.00% | +6.82% | +7.99% |

The corresponding sampling errors are about 9.5-12.9% of the old mean. These
differences are below 0.7 individual-bin sampling sigma. For B16 and legacy
Lee22, the 10-100 arcmin bins agree with the previous mean within the sampling
errors. This single realization supports the previous mean calculation; it
does not remove the physical difference between density models or reproduce
the noise of the 71/31-source observations. The preferred Lee22 model retains
its known high-mass extrapolation/gas-budget problem and remains diagnostic.

## Isabel Medlock & Nagai reference

`references/medlock_fig5_best_fit_digitized.csv` contains 310 points traced from
the original Figure 5 asset in arXiv:2608.06455v1. These are the **black BP
maximum-likelihood angular cross-correlation curves** for ACT, Planck MILCA,
and Planck NILC, not electron-density profiles and not observational points.
The existing Sharma/CHIME sample has a different selection and is not presented
as a matched HalfDome test here.

The paper places all FRBs at z=2 and shows unbeamed model curves. Our separate
reference plot therefore uses unbeamed HalfDome too, but retains the requested
observed redshift distribution. Thus it is a qualitative model-reference
comparison, not a matched-kernel test or a new BP implementation. A plotted
curve alone cannot be reliably reweighted to another source redshift kernel.
The BP model also has its own pressure, cosmology and outer-boundary assumptions.

Axis calibration is log(theta), linear w. Readout uncertainty is 1.5 pixels;
near-zero tails below image resolution are flagged. No extrapolation or fit
confidence interval is invented. The trace is visually checked in
`references/medlock_fig5_best_fit_trace_QA.png`, with source SHA256 and axis
calibrations in `references/digitization_provenance.json`.

## Validation and cluster provenance

- Sparse disc lookup equals brute-force angular selection, including poles,
  longitude wrap and repeated pixels (Julia 1.6.2 cluster and 1.12.2 local).
- Exact annulus filters agree with independent quadrature; sample covariance
  and analytical jackknife agree with explicit delete-one calculations.
- The previous radius-cache signatures are checked, then 100 direct-quadrature
  points are checked for each model before streaming the halo catalogue.
  Maximum errors are 0.0271% (B16), 0.0452% (legacy Lee22), and 0.1296%
  (preferred Lee22).
- An independent full-map pixel-space versus harmonic-space mean check, across
  three models and three angular/survey cases, differs by at most 4.63e-9
  fractionally. The new estimator and old baseline have consistent normalization.
- The initial analysis job stopped because older h5py returned a checksum as
  bytes. The checksum itself was identical. Explicit UTF-8 decoding fixed this;
  the DM and y calculations did not need rerunning.
- All three downloaded ray products match their cluster SHA256 checksums.
  Independent local and cluster analyses give identical cross estimates and
  sampling errors to floating-point precision. The old-mean transforms differ
  by at most 7.6e-11 fractionally across the two NumPy versions.
- The updated notebook was executed end to end: six code cells, zero errors.

Cluster output:

```
/lustre/work/kristero10/frb_data/takahashi_100k_20260914_v1
```

Source snapshot:

```
/home/kristero10/HalfDome_kSZ/frb_map_generation/cluster_tests/takahashi_100k_20260914_v1
```

DM job 597277 and annular-y job 597278 completed with exit 0. Analysis retry
597282 completed with exit 0 (original 597279 retained its failure log).
Independent normalization check 597280 completed. Jobs used mini, 26 CPUs,
48 GB, and a 23:59:00 walltime request. Local `cluster_results/` contains
downloaded analysis, logs, source hashes, and product SHA256 checksums.

To regenerate plots locally (does not rerun the cluster calculation):

```bash
cd /home/cbllover/HalfDome
python frb_map_generation/plot_takahashi_sightlines.py
```

To repeat with another seed, use a **new** output root, run
`compare_takahashi_sightlines.py prepare --seed NEW_SEED --output NEW_ROOT`,
upload its `rays/source_positions.h5`, and submit
`run_takahashi_100k_sightlines.pbs` with `STAGE=dm` and `STAGE=y`. After both
succeed submit `STAGE=analyze`. Supply `COMPARISON_CODE`, `COMPARISON_OUT`, and
`PREVIOUS_OUT` as in the run paths above. Do not overwrite the old mean maps.

## Files added or edited

- Edited: `FRB_tSZ_Takahashi_comparison.ipynb` (new sections; old analysis retained).
- Added: `sample_halfdome_observed_sightlines.jl`, `compare_takahashi_sightlines.py`,
  `plot_takahashi_sightlines.py`, `digitize_medlock_best_fit.py`,
  `check_takahashi_sightline_estimator.py`, `run_takahashi_100k_sightlines.pbs`,
  and this report, all under `frb_map_generation/`.
- No XGPaint or density-profile implementation was modified for this test.
