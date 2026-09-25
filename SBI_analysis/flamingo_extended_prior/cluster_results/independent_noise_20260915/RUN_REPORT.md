# Extended HalfDome production run

The 8192-row production dataset was submitted after the independent-noise
preflight passed. Four production workers started: 597627.idark through
597630.idark. There are 64 chunks of 128 immutable rows in four afterok chains.
The terminal jobs are 597687--597690; collector 597691 exports the complete
ordered dataset after all four chains succeed. This is a started run, not a
completed 8192-map dataset. See production_status.json for its timestamped state.

Cluster root: `/lustre/work/kristero10/halfdome_flamingo_extended_8192_20260915`.
Every worker verifies the frozen manifest, source dependencies, design and gate.
The original HalfDome and FLAMINGO campaigns have not been edited. Earlier fixed
noise preflights and the failed old-Bash wrapper attempt are retained in attempts/
and logs/. The current PBS wrapper passes its production-branch check.

## Noise implementation and physics

Training now uses **independent SO noise per row and per split**. Master seed
12345 is a reproducibility key; it is not reused directly as the random seed of
every map. SHA256 of `halfdome-so-v1|12345|ROW_ID|SPLIT`, first eight bytes in
big-endian order shifted right one bit, defines the actual Julia MersenneTwister
seed. The complete two-column seed array is saved. Row IDs and seeds survive
restarts, chunk changes and expansion to 524288 rows. Preflight uses separate
IDs 524288--524293. All 1048588 seeds in that combined range are distinct and
exclude the observation split seeds 22446/22447.

The new process-local loader reads and verifies the historical mask, then uses
the original SO table reader, Gaussian harmonic generator and map synthesis
functions to produce fresh noise arrays. The archived signal painting and
cross-spectrum code still compute W(By+n1) cross W(By+n2). B is the 2 arcmin
Gaussian signal beam; W is the same cap mask, support f_sky=0.4 and 60 arcmin
apodization. SO baseline, deprojection 0, lmax=7979 and each split's original
N_ell normalization are unchanged. No extra noise beam or f_sky division is
applied. FLAMINGO observations retain their fixed original noise realization.

Independent zero-mean splits give zero *ensemble* noise cross power. Each
realization still contains signal-noise and noise-noise cross terms, including
possibly negative bins. This samples the instrumental-noise distribution during
SBI training. It does not sample cosmic variance: the halo catalogue, cosmology,
sky realization and mask stay fixed. Posterior coverage still needs validation.

## Working joint prior

The nine-parameter pressure model is
`Pth/P200 = P0 (x/xc)^(-0.3) (1+x/xc)^(-beta)`, x=r/R200c.
Each of P0, xc and beta evolves as `A0 (M/1e14 Msun)^alpha_m (1+z)^alpha_z`.
M is physical M200c, obtained by dividing catalogue Msun/h by h=0.68.
Electron pressure is Pe=0.5176 Pth and Compton y integrates Pe along the LOS.
The outer pressure slope is beta+0.3, so beta>2.7 gives finite infinite-volume
thermal content. The historical painter keeps its finite LOS and projected
4 R200c painting radius, catalogue/redshift range and pixel sampling.

The exploratory ranges remain broad: P0 1--60, xc 0.1--4, beta 2.8--16;
mass exponents (-0.2--1.5, -0.6--0.4, -0.2--0.4); redshift exponents
(-4.5--0.5, -1.5--2, -0.5--1.5), ordered P0, xc, beta. P0 and xc are
log-uniform; the remaining coordinates are uniform. The final prior conditions
this base distribution on **joint** bounds:

- raw beta(M,z) between 2.8 and 50 on the full interpolation domain;
- relative size proxy (xc/beta)/(xc_B12/beta_B12) between 0.4 and 8;
- finite spherical Y200/Y200_B12 between 0.003 and 30;
- estimated missing central LOS tail at most 1%.

See code/prior.json and code/README.md for the exact mass/redshift domains,
corner extrema and grid definitions. The finite Y200 integral is proportional
to `P0 xc^3 B_[1/(1+xc)](2.7,beta-2.7)`. Coupling these restrictions removes
pathological combinations that individual rectangular limits permit. Only
about 1.70% of the base product distribution survives; none of its 512 corners
passes. All three effective FLAMINGO fits and Battaglia12 remain inside support.
The 8192 accepted points also satisfy the Y200 bounds on a denser 33x33 grid.

These are broad simulation-domain restrictions, not measured confidence limits
or a guarantee of physical realism. Lee22 motivates pressure and mass/redshift
flexibility, but its concentration dependence and broken mass law cannot be
represented exactly by this family. The Lee22 curve in pressure_prior is a
fixed-concentration illustration within its calibration interval; extrapolating
its single branch across the entire catalogue can fail this prior's size cut.
The prior is informed by these FLAMINGO maps. Effective fits can absorb different
cosmology, redshift support and gas absent from the halo catalogue; they should
not be interpreted as an isolated feedback measurement or independent cosmology
prior. FLAMINGO maps end at z=3 while the retained HalfDome selection reaches
about 3.855. Pixel aliasing, low-z interpolation and cutoff limitations remain.

## Numerical implementation and checks

stable_los.jl evaluates the same finite LOS using l=x sinh(u) and a scaled
integrand, with a bounded quadrature budget. It preserves the original cache
floor where representable and otherwise uses the smallest positive Float64.
This prevents underflow to zero from contaminating logarithmic interpolation.
The steep boundary required 19756 such cache replacements. Every full map
checks 108 painted-region columns against independent native quadrature.

All six new NSIDE4096 maps passed with 12 distinct noise maps and the same mask.
The first noise map was regenerated at full resolution with exactly the same
pixel hash. Across 64 independent harmonic pairs, the two normalized auto powers
were 1.002002 and 0.999897; mean normalized cross power was -0.00009852, all
consistent with their analytic sampling errors. Exact seeds, hashes and checks
are recorded in preflight/quality_gate.json and noise_seed_check.json.

| Full-map case | Seconds | Peak GiB |
|---|---:|---:|
| Battaglia12 | 262.4 | 13.02 |
| L1_m9 | 244.1 | 11.47 |
| fgas-8sigma | 267.3 | 11.48 |
| Mstar-1sigma | 261.4 | 11.54 |
| compact_boundary | 278.2 | 11.48 |
| steep_boundary | 270.8 | 11.53 |

The first case includes an additional full-resolution noise replay. Maximum
column relative error across 648 checks: 2.89e-15. Maximum clean reference regression
error relative to peak binned power: 4.87e-16. The new noisy spectra are different
random draws and are not expected to reproduce the old noisy spectra. Restart,
row ordering, corruption rejection, seed collisions and prior support checks
passed separately. These checks cover six physical maps, not every future row.
Failed production rows stop without replacement; successful rows remain resumable.

## Scaling, inference handoff and plots

At the measured median 267.3 seconds per ordinary preflight map, 8192 rows require
about 608 worker-hours (6.3 ideal days with four workers); 524288 require about
38925 worker-hours (405 ideal days with four workers). These extrapolations exclude
queueing and retries and are sensitive to profile and hardware. Production
requests mini, 26 CPUs/application threads, 64 GB and 23:59:00 per worker.

Compact HDF5 shards are estimated at 0.105 GiB for 8192 and 6.72 GiB for 524288,
plus roughly 8.2 MiB / 524 MiB of NumPy exports. Existing input maps, catalogue
and mask cache are reused. Each active worker uses about 128 MiB of temporary
interpolation cache; full pixel maps are not accumulated on disk. Optional
retention of all unbinned spectra adds about 1.46 / 93.5 GiB. See
storage_estimate.json for the measured scope.

The code is configurable with `prepare.py --count 524288 --root NEW_ROOT`.
The accepted parameter sequence and validation split are prefix stable, and a
524288-point design was generated as a scalability check; no 524k map run has
been launched. Follow code/README.md for auditing and full-map preflight of a
new root before submission.

The outputs retain clean and noisy spectra separately in 40 mode-count-weighted
linear D_ell bins over ell 80--7979. Refit/validate MOPED on the broader prior and
retrain SBI before interpreting new-prior FLAMINGO posteriors. The old flow is
outside its training support here. joint_sbi_prior.pkl supplies the conditional
prior, including its P0/xc density Jacobian and estimated normalization; do not
replace it with a BoxUniform rectangle. No new posterior has been trained by
this dataset-generation run.

Publication figures are available as PNG and vector PDF:

- plots/tsz_preflight: Battaglia12, three FLAMINGO spectra, their HalfDome fits,
  and accepted compact/steep extremes, all clean beam-smoothed masked signals.
- plots/flamingo_noise: separate comparison with fixed observation noise.
- plots/joint_prior: accepted support and retained reference points.
- plots/pressure_prior: pressure and finite thermal-content variation, with
  the illustrative Lee22 electron-pressure curve.

## Files created or edited

All source changes are isolated to the following 20 files. Generated reports,
plots, seed/design arrays, manifests and validation products are separate
artifacts; the complete file hashes are in artifact_manifest.json.

- `SBI_analysis/flamingo_extended_prior/README.md`
- `SBI_analysis/flamingo_extended_prior/audit.py`
- `SBI_analysis/flamingo_extended_prior/check_noise_seeds.py`
- `SBI_analysis/flamingo_extended_prior/check_workflow.py`
- `SBI_analysis/flamingo_extended_prior/collect.py`
- `SBI_analysis/flamingo_extended_prior/independent_noise.jl`
- `SBI_analysis/flamingo_extended_prior/noise_seeds.py`
- `SBI_analysis/flamingo_extended_prior/paint_row.jl`
- `SBI_analysis/flamingo_extended_prior/plot_preflight.py`
- `SBI_analysis/flamingo_extended_prior/prepare.py`
- `SBI_analysis/flamingo_extended_prior/prior.json`
- `SBI_analysis/flamingo_extended_prior/prior.py`
- `SBI_analysis/flamingo_extended_prior/run_collect.pbs`
- `SBI_analysis/flamingo_extended_prior/run_config.json`
- `SBI_analysis/flamingo_extended_prior/run_worker.pbs`
- `SBI_analysis/flamingo_extended_prior/sbi_prior.py`
- `SBI_analysis/flamingo_extended_prior/stable_los.jl`
- `SBI_analysis/flamingo_extended_prior/submit.py`
- `SBI_analysis/flamingo_extended_prior/validate.py`
- `SBI_analysis/flamingo_extended_prior/worker.py`
