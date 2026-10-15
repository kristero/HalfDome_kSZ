Actual HalfDome lightcone map and matched FRB sensitivity tests
===========================================================

Final audit and results (supersedes the initial implementation notes below)
-------------------------------------------------------------------------
The full render completed on all 85,224,251 catalogue halos. Numerical flux
closure is better than 1.4e-10; the fine catalogue histogram matches exactly.
The run took 733.58 s wall, 3210.46 CPU s and 12.293 GiB peak RSS. No PBS jobs,
raw gas particles or downloads were used. The first pass requested 3.068 GB
of logical halo metadata. Useful compressed baseline maps occupy 1.514 GB.

audit_dm_statistics.py adds an independent DM-only foreground pass, without
new y/kSZ allocations or harmonics. It reproduced every one of the 15 pixel
histograms and means exactly, taking 257.54 s wall, 989.73 CPU s and 4.895 GiB.
frb_DM_exact_statistics.json explicitly records zero atoms, p99.9, p99.99,
strict probabilities above 500/1000/2000 pc cm^-3, and paired high-mass controls.

The audit corrected point-ray interpolation leakage in logM=13.28--13.3 so
all alternatives share the published fallback below actual logM=13.3, just
as the maps do. Baselines, high-mass pools and cuts were unaffected. The
largest alternative mean correction was 0.1083 pc cm^-3; the largest histogram
CDF correction was 0.0004. The reversible correction and original hash are
retained in frb_boundary_ray_correction.npz/json. Canonical ray, PDF and
statistics files contain the correction. verification.json retains the
original map execution, and dm_audit_verification.json records the update.
The scientific driver was not edited during execution; its source hash and
mtime are in executed_source_receipt.json.

The projected observables obey
  y = sigma_T/(m_e*c^2) integral Pe dl,
  DeltaT/T = -sigma_T/c integral ne*v_los dl,
  DM_observed = integral ne dl/(1+z), converted to pc cm^-3.
Pe is physical electron pressure, ne free-electron number density, dl proper
distance, sigma_T Thomson cross section, m_e electron mass and c light speed.
The native profile quantities and the declared mean mass per free electron
are applied once; no additional hydrogen fraction is multiplied into them.
The half-chord table average is multiplied by 2*Rcut*sqrt(1-u^2), where
u=b/(Rcut*R200c), before physical normalizations. Only halo bulk v_los is used.

At ell=100, 2R/3R relative to 4R y power is 0.9274/0.9702 and bulk-kSZ power
is 0.5273/0.7642. At ell=1000 these are 0.9696/0.9891 and 0.6704/0.8660.
Removing logM>=15.3 changes y power by -24.25%/-5.75% at ell=100/1000;
removing logM>=15.1 changes it by -31.73%/-15.32%. Corresponding bulk-kSZ
changes are -2.12%/-0.27% and -4.05%/-0.38%. Same-realization high/low mass
auto/cross terms close to 1.3e-14 relative; this mass split is not a 1h/2h split.

Pixel mean halo-only observed DM for 2/3/4R is 138.97/190.64/233.37 pc cm^-3;
matched-ray means are 138.68/190.35/233.15. Ray p99.9 is 1096.72/1172.57/1233.22.
Pixel zero fractions are 0.0180319/0.000208855/7.35124e-7. Every primary and
ray histogram keeps all samples with zero overflow. High-mass cuts are
stochastically dominated at every pixel: their exact marginal W1 distances
equal the mean removed DM, 0.16338 and 0.61197 pc cm^-3. Saved histogram CDF
distances are lower bounds; their coarse-bin upper bounds are not tight.

Independent NSIDE1024/2048 absolute powers differ by about 6% at ell=1000
and 14--16% at 1500 after the usual pixel-window correction. Radius power
ratios are more stable: 2R/4R changes by less than 0.5% at ell=1000 and about
1--1.5% at 1500. Absolute high-ell power and the all-pixel DM PDF have not
demonstrated continuum convergence. Exact flux closure does not replace
this angular-shape convergence test.

plot_lightcone.py writes eight large-font publication PNG/SVG pairs. Its
final pass took 14.76 s wall, 14.74 CPU s and 1.702 GiB. Positive-DM PDF curves
start at 1 pc cm^-3; the zero atom stays in their fixed normalizations and
is tabulated separately. The Poisson position control changes the FRB PDF;
it is not an additive two-halo PDF. Intrinsic random profile scatter is not
realized in these 15 mean-profile maps. Separate compact/analytic products
report the conditional scatter impact.

Minimal data access:
  import numpy as np
  data = np.load('fullsky_spectra.npz')
  case = list(data['task_names']).index('baseline_r4')
  ell = data['ell']
  yy = data['cl_y_ksz'][case, 0]
  D_ell = ell*(ell+1)*yy/(2*np.pi)

  import h5py
  with h5py.File('baseline_maps_nside2048.h5') as f:
      sampled_DM_pc_cm3 = f['baseline_r4'][2, ::1024]

final_verification.json audits final normalizations and canonical hashes;
visual_qa.json records inspected figures and manifest.json records all files.
The original driver refuses to overwrite a completed run. Reproduction
requires a fresh output directory and --gates before --run.

Original implementation contract
--------------------------------

paint_lightcone.py streams every row of lightcone_100.hdf5 once. The catalogue
contains85,224,251halos. HaloM200c is divided byh=0.68; Position supplies only a
unit direction; z determines DA. Velocity supplies bulk peculiar vlos inkm/s.
The lightcone target cosmology is h=.68,Om=.31,Ob=.049, distinct from FLAMINGO.
Dimensionless calibrated z0shapes inherit the declared published redshift ratio.

Run --gates before --run. Eight workers own disjoint HEALPix rings. Every finite
sphere conserves its integrated flat-screen flux even if no pixel centre falls
inside it. Resolved profiles are centre sampled then renormalized perhalo; tiny
objects deposit finite flux in the containing pixel. This avoids lost small
halos but does not establish continuum profile/DMtail convergence. The primary
NSIDE2048maps and separately painted NSIDE1024baselines quantify that limitation.
Finite chords use b=DA sin(angle) and sharp2,3,4R200c spheres. Angular total flux
is normalized to the usual small-screen integrated-flux relation. Very nearby
objects with4R/DA>=0.5are rejected instead of silently applying that approximation.

Fifteen mean-profile tasks are encoded in the parent profile_column_tables.npz.
Baseline2/3/4R plus pooled high-mass, actual-pivot, adaptive, smooth-mass, weighted,
massuniform, high-masscut, ownership-control, unseen-tail and controlled-outer
binning alternatives. BelowlogM13.3, alternatives share the published fallback.
Alternative maps are accumulated as differences from baseline4R to avoid
repeating identical low-mass contributions. Float32difference cancellation can
leave tiny negative DMroundoff; its minimum is saved and clipped only for PDFs.

The exact historical120,000seed42source directions are reused from their saved
CSV, validated against their HEALPix4096pixel centres. Columns at those fixed
directions are evaluated independently of the new NSIDE2048map. They are
infinitesimal-ray columns, not new2048pixel averages. "Allpossible" means all
50,331,648equal-area NSIDE2048pixels, with an explicit finite-resolution limit.
FRBforeground is z<=1; observed DM includes1/(1+z), without anotherXH factor.
No diffuseIGM/host/MilkyWay/source-selection term is added. kSZ uses only halo
bulk velocity; thermal/internal gas velocities and missing diffuse gas are absent.

Outputs
-------
baseline_maps_nside2048.h5: useful baseliney,DeltaT/T,DMmaps retained compressed.
fullsky_spectra.npz: actual totalCelly andbulk-kSZ; high-massremoved auto and
cross-with-retained terms computed from the samealms. No approximate analytic
self term is promoted to an accurate map two-halo decomposition.
frb_matched_ray_dm.npz: all15tasks at the exact120kold sightlines.
frb_DM_PDF.npz: common-bin allpixel andmatched-ray histograms, overflow tracked.
fine_velocity_histogram.npz: exact finecount grid andsumvlos2 for analyticself.
nside_convergence.npz: independently painted1024baseline spectra andDMhistograms.
sky_overviews_nside64.npz: compact full-sky overview, from angular averages.
progress.json,verification.json,complete.json: status, resources andnumericalgates.
plot_lightcone.py: seven publication-sizedPNG/SVGfigures andobservable_metrics.json.

Spectra are full-sky map totals. Gas-profile overlap is not uniquely partitioned.
High-masslow/high cross terms are a mass split, not a1halo/2halo split. Conditional
amplitude-error bounds require an explicit chosen bound on high-mass amplitudes.
RMSchecks are empirical sensitivity metrics, not a calibrated likelihood error.
These maps vary mean profiles; no stochastic intrinsic-scatter field is painted.
The separate compact and analytic projects document scatter andbootstrap scope.
