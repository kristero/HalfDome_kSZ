# FLAMINGO L1_m9 to HalfDome: z=0 transfer comparison

This study compares the measured FLAMINGO gas field, FLAMINGO halo-painted
fields, and HalfDome halo-painted fields. HalfDome has no hydrodynamic gas
truth. Differences between independent simulations are statistical; their
pixels and individual halos are not matched.

The user approved recovering and validating the released HalfDome
RFoF-to-M200c mass calibration, equal physical comparison volumes/depths,
and a whole-box overview. Previous products are read-only inputs.

## Resource budget before execution

- Original budget: at most two idark jobs. The user subsequently authorized
  five simultaneous idark nodes. Fifteen job instances were submitted,
  including a cached SD refit, a cache-lock retry, two plotting retries and
  figure refinement and PDF compatibility/rendering retries. All request mini,
  26 CPUs and 32 GB. No request exceeds
  eight hours. Scientific particle extraction and painting were not repeated
  for plotting failures. The user's twenty-job limit is respected.
- Expected idark execution: 4-8 hours plus 1-2 hours for delivery; queue time
  is unknown. This is an estimate, not measured usage.
- Expected new idark storage: 5-15 GB. Temporary arrays are processed
  sequentially and removed only after their merged products are verified.
- STRW: one bounded four-worker pass, 24 GB process-tree memory cap,
  expected 1-3 hours. No raw snapshot transfer to idark.
- Live snapshot metadata: 64 files, 5,271,821,058 gas particles. The six
  necessary fields are 295,221,979,248 decoded bytes and 124,518,258,193
  stored bytes (dataset storage sizes, not measured network traffic).
- Idark Lustre had approximately 151 TB available, 95% occupied at preflight.
  This is filesystem space, not a personal quota or reservation.

## Physics contract

Use physical M200c in solar masses, physical Mpc at z=0, and physical
peculiar velocities in km/s. The HalfDome raw Length mass is RFoF mass,
not M200c. Reject raw halos with fewer than 320 particles. Validate the
reconstructed mass mapping on held-out released lightcone objects and
report the additional z=0 extrapolation sensitivity.

FLAMINGO's header Omega_m=0.304611 is baryons plus CDM; its neutrino
contribution is 0.0013890794. The total nonrelativistic matter density is
therefore approximately 0.3060000794, consistent with the baryon fraction
used in the existing extraction. HalfDome has Omega_m=0.3089 and h=0.6774;
FLAMINGO has h=0.681 and Omega_b=0.0486.

At fixed physical halo mass and dimensionless shape x=r/R200c:

- R200c scales as H^(-2/3).
- P200 scales as f_b H^(8/3).
- The electron-density normalization scales as f_b H^2.
- Projected y scales as f_b H^2 at fixed x; tau and DM as f_b H^(4/3).
- Integrated physical Y scales as f_b H^(2/3).

These are self-similar normalization transfers, not corrections to the
initial conditions, halo abundance, feedback history or velocity field.
Native measured FLAMINGO maps retain their physical values. Target-cosmology
painting and source-cosmology controls quantify the normalization effect.

Native ElectronNumberDensities already contains composition and ionization;
never multiply it by a hydrogen fraction again. Native ComptonYParameters
is an integrated particle contribution in physical Mpc^2, not local y.
Use native gas velocities for the measured FLAMINGO kSZ; halo bulk velocities
for painted kSZ, and keep that approximation visible.

For a physical slab at z=0, y = sum(Y_i)/pixel_area;
tau = sigma_T integral(n_e dl);
DM = integral(n_e dl), in pc cm^-3;
DeltaT = -T_CMB sigma_T integral(n_e v_los/c dl).

Fit and truncation radius are independent, each 1,2,3,4 R200c. All-gas
fits are the physical-observable reference; excluding other-halo-bound gas
is an overlap sensitivity control. The model retains the existing 0.5176
B12 convention because the fitted amplitude was obtained with that same
electron-pressure shape. It is not applied a second time to measured data.

The preferred Pe fit2/cut4 and ne fit3/cut3 combination is an observable-
specific halo template, not a complete joint thermodynamic gas field: in
the 3-4R shell it has pressure but no halo electrons. A derived common-
support control uses Pe fit2/cut4 and ne fit3/cut4. A physical joint model
can instead supply those outer electrons through a consistent diffuse
component. Neither procedure validates an uncalibrated redshift evolution.

The streamed FLAMINGO merge hit the separate 3.9 GB STRW home quota.
All four completed worker products were preserved and SHA-256 verified,
and the study was moved to /data/nagainis/HalfDome/flamingo_pressure_calibration.
The old home path is a symlink. The incomplete merge was discarded;
no particle pass was repeated. Worker outputs are deleted only after a
successful merged-field conservation check. The original raw snapshot
and earlier studies are untouched.

Probability distributions use all map pixels or all cells of the specified
grid, with zero atoms, underflow and overflow retained. Signed kSZ is never
log-transformed. The delivered pressure/density figures use uniform-grid
volume PDFs. Particle-volume-weighted histograms are a different estimator
and must not be substituted for these grid statistics.
Spatial region variation is not independent-mock covariance. A single
z=0 calibration cannot constrain redshift exponents.

## Local memory and delivery

`run_local_report.ps1` checks Windows and WSL available memory before local
PDF/preview generation and monitors the process. It stops this study's local
process below 5 GiB, warns at the critical 2 GiB threshold, and removes only
regenerable PDF intermediates if that critical threshold is crossed. It does
not terminate unrelated user processes or running cluster calculations.

An interrupted deployment left an empty local plotting file. The intact
cluster source was recovered as `aggregate_recovered.py`, repaired with
`repair_plotting_source.py`, hash-verified on idark, and tested against its
Matplotlib 3.3.4 before submission. `refine_figures.py` adjusts only visible
residual axis limits and symmetric-log colourbar formatting. The fit data
and model maps remain unchanged.

The final entry point is `index.html`. It links the two PDF reports, all
figures, parameter distributions and case-statistic tables. Full model map
HDF5 files remain on idark; native cached 3D gas grids remain on STRW. The
delivery bundle intentionally excludes those large caches.

Final measured application CPU time is 1.451 CPU-hours on idark across the
fifteen explicitly matched GNU time logs, plus 3.024 CPU-hours for STRW gas
workers. This excludes untimed stages. The idark allocation lower bound is
6.687 core-hours; final PBS accounting fields were unavailable and are null.
Hardware/free-space values in the detailed report are captured observations,
not reservations. See `resource_accounting_summary.json` for definitions.

The final local PDF layout uses the already verified compact results.
`report_qa.json` records rendered page counts and visual-review status.
`delivery_manifest.json` records final hashes. The earlier live cluster
verification remains dated 2026-10-14 14:30 UTC; idark was unreachable during
the final local delivery pass. No new particle reads or refits were run.
