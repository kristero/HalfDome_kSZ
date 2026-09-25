# Extended HalfDome prior for FLAMINGO observations

This campaign conditions the exploratory rectangle on joint profile checks. It
preserves the historical HalfDome catalogue, cosmology, pixel sampling, angular
interpolator and 4 R200c projected painting radius. Only the numerical evaluation
of the same finite line-of-sight integral changes. The original campaign and
trained model files are untouched.

## Prior and physics

The physical parameter order is
`P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta, alpha_z_P0, alpha_z_xc, alpha_z_beta`.
For each of P0, xc and beta, A(M,z)=A0 (M/1e14 Msun)^alpha_m (1+z)^alpha_z.
M is physical M200c: HalfDome's catalogue masses in Msun/h are divided by h=0.68.
The pressure is

`Pth/P200 = P0 (x/xc)^(-0.3) (1+x/xc)^(-beta)`, with `x=r/R200c`.

The electron pressure is Pe=0.5176 Pth; y is its LOS integral times
sigma_T/(m_e c^2). The raw beta has outer pressure slope beta+0.3. Thus beta>0.7
gives a convergent infinite central column, and beta>2.7 gives finite infinite
thermal content. The painter retains the existing LOS endpoint at 1e5 R200c.

| Parameter | Base range | Base density |
|---|---:|---|
| P0 | 1–60 | log-uniform |
| xc | 0.1–4 | log-uniform |
| beta | 2.8–16 | uniform |
| alpha_m_P0 | -0.2–1.5 | uniform |
| alpha_m_xc | -0.6–0.4 | uniform |
| alpha_m_beta | -0.2–0.4 | uniform |
| alpha_z_P0 | -4.5–0.5 | uniform |
| alpha_z_xc | -1.5–2 | uniform |
| alpha_z_beta | -0.5–1.5 | uniform |

Every accepted parameter vector must also satisfy these editable restrictions:

1. **2.8 <= beta(M,z) <= 50** throughout the interpolation rectangle
   M=1e12–10^15.7 Msun, z=0.001–5. Exact power-law extrema occur at its corners.
   The lower limit gives a margin above infinite-energy divergence. The upper
   limit is an engineering cap informed by the previous expensive steep trials;
   it is not a physical upper bound or a quadrature theorem.
2. **0.4 <= (xc/beta)/(xc_B12/beta_B12) <= 8** over an enclosing rectangle for
   the actual catalogue. This excludes excessively compact profiles and limits
   extended profiles, while retaining all three effective FLAMINGO fits. This
   proxy controls relative size, not an absolute HEALPix resolution criterion.
3. **0.003 <= Y200/Y200_B12 <= 30**, on a 9x9 grid uniform in log M and
   log(1+z) over that catalogue rectangle. Y200 is the *finite spherical*
   pressure integral to R200, proportional to
   `P0 xc^3 B_[1/(1+xc)](2.7,beta-2.7)`. Ratios compare the same M,z, so physical
   conversion factors cancel. A separate 33x33 audit measures grid sensitivity;
   the finite grid is not a proof of a bound at every intervening M,z.
4. At most **1% missing central column** beyond the finite LOS endpoint, using
   the conservative combination of minimum beta and maximum xc on the full grid.

The catalogue rectangle uses log10 M=[12.8167,15.5781], z=[0.0021,3.8555]. It
encloses the measured lightcone support, including combinations absent from the
actual catalogue. The amplitude and size restrictions are deliberately broad
simulation-domain choices, not observational exclusions. They prevent the tiny
thermal content and huge pressure combinations found in the unrestricted pilot.
They do not guarantee that every permitted profile is physically realistic.

The four retained reference points are Battaglia12 and the pilot's best clean
fits for L1_m9, fgas-8sigma and Mstar-1sigma. Controls are kept separate from
the training design; no fitted points are injected into the prior sample.

[Lee et al. 2022](https://arxiv.org/html/2205.01710v1) motivates broad pressure
and mass/redshift flexibility, but its concentration dependence and broken mass
law cannot be represented exactly by these nine parameters. The plot shows the
published electron-pressure formula at fixed concentration 4.5 within its
calibrated radial/mass interval. All amplitudes use the paper's M_cut pivot;
no extra 0.5176 factor is applied to Lee22 electron pressure. The curve is an
illustration, not a nine-parameter observational likelihood.

## Numerical change and limitations

`stable_los.jl` substitutes l=x sinh(u) in the *same* finite LOS integral, then
normalizes the integrand at its peak. Integration therefore does not chase
relative precision in subnormal tails at unpainted interpolation nodes. It has
a bounded evaluation budget and fails explicitly when it does not converge.
The native nonpositive-cache floor is retained whenever representable; if its
`minimum_positive*1e-6` underflows, the smallest positive floating value is used
instead. This prevents log(0) from contaminating the cubic interpolator. The
steep boundary preflight exposed this second numerical failure. Each full map checks 108 columns
against independent native quadrature in the painted region.

The preflight requires six NSIDE4096 maps: four references and accepted compact
and steep boundary cases. All four clean reference spectra must agree with historical
maps to 1e-6 relative to peak binned power. `quality_gate.json` records the actual
results. Each case has two distinct SO noise draws. The first repeats a full
NSIDE4096 noise map to check exact reproducibility and checks 64 harmonic noise
pairs against the input SO auto power and zero ensemble cross power. Seed checks
cover all 524288 production rows plus six separate preflight rows.
Passing this gate validates those cases, not every future simulation.
Runtime failures stop the chunk and preserve a failure log and the original row.

Preserving the map definition also preserves its known low-redshift interpolation
and unresolved-halo pixel-sampling limitations. The relative-size cut reduces
the worst compact extremes; it does not remove pixel aliasing or establish
4 R200 cutoff convergence. The broad FLAMINGO fits partly absorb differences in
cosmology, redshift support, gas outside catalogued halos, lensing and phases.
FLAMINGO maps extend to z=3; the historical HalfDome selection extends to z~3.855.
Parameter recovery is an effective-model comparison, not an isolated measurement
of feedback. Using FLAMINGO to inform this prior is data-informed model design;
do not describe it as an independent prior for cosmological inference.

## Prepare and run on idark

All commands below run on the cluster with `/home/anaconda3/bin/python3`.
The frozen 8k run lives at
`/lustre/work/kristero10/halfdome_flamingo_extended_8192_20260915`.
The reusable source lives inside that directory's `code/`.

```bash
RUN=/lustre/work/kristero10/halfdome_flamingo_extended_8192_20260915
PY=/home/anaconda3/bin/python3

# Preflight only (already run during preparation; completed chunks are skipped).
$PY "$RUN/code/submit.py" --root "$RUN" --preflight --max-jobs 3 --submit
$PY "$RUN/code/check_noise_seeds.py" --root "$RUN"
$PY "$RUN/code/validate.py" --root "$RUN"

# Print the complete 8192-row PBS plan for review.
$PY "$RUN/code/submit.py" --root "$RUN" --all --max-jobs 4
# Submit it when ready. Each job processes 128 rows.
$PY "$RUN/code/submit.py" --root "$RUN" --all --max-jobs 4 --submit

# Alternatively use bounded waves; repeat after completion.
$PY "$RUN/code/submit.py" --root "$RUN" --max-jobs 4 --submit

# Once every chunk is complete, export ordered NumPy arrays.
$PY "$RUN/code/collect.py" --root "$RUN"
```

The default is mini, 26 CPUs, 64 GB and 23:59:00, with 26 application threads.
Change the queue explicitly with `--queue mini2` if needed. The preflight and
production commands are separate. A production submission requires a passing
gate tied to the frozen manifest. No automatic production launch follows the
preflight. Dry-run is the default for submission commands.

For **524288**, change only the count and use a new root:

```bash
LARGE=/lustre/work/kristero10/halfdome_flamingo_extended_524288
$PY "$RUN/code/prepare.py" --root "$LARGE" --count 524288
$PY "$LARGE/code/audit.py" --root "$LARGE"
$PY "$LARGE/code/check_noise_seeds.py" --root "$LARGE"
$PY "$LARGE/code/submit.py" --root "$LARGE" --preflight --max-jobs 3 --submit
# After preflight completion:
$PY "$LARGE/code/validate.py" --root "$LARGE"
$PY "$LARGE/code/submit.py" --root "$LARGE" --all --max-jobs 4 --submit
```

The accepted scrambled Sobol sequence is prefix stable: the same seed and prior
give the same first 8192 rows at 524288. Rejection removes digital-net balance;
this is a conditional space-filling design. Proposal indices and a stable split
are saved. The validation split hashes the proposal index before selecting 10%:
directly using index modulo 10 would correlate with Sobol structure. Do not mix datasets generated with different prior versions without
accounting for their sampling distributions.

Every chunk has an exclusive lock. Per-row HDF5 files are atomically published,
then packed into one HDF5 file per chunk. Each contains exact parameters, three
40-bin spectra, timing, operator hashes, quadrature checks and compressed logs.
At most one interpolation cache per active worker is retained in scratch, and
it is deleted after that row. Successful chunks are checked and skipped on
resubmission; failed rows are retried with the same parameters, never redrawn.
A node crash may leave a scratch directory; inspect and remove only that stale
directory after confirming its job is no longer running. Do not delete shared
campaign noise caches or old pilot caches.

The cluster's PBS 19.1 does not accept array concurrency syntax. `--all` instead
uses four serial `afterok` dependency chains: at most four jobs from this run can
execute concurrently. A failed job blocks its later chain; after fixing the
failure, resubmit missing chunks with the same design. `run_collect.pbs` can be
submitted with `afterok` dependencies on the four terminal jobs to export the
dataset automatically. Submission records preserve every job ID immediately.

Workers stop before the walltime budget; resubmitting unfinished chunks resumes
them. The collector refuses incomplete datasets and orders rows by immutable ID.
Full unbinned Cl retention is optional via `retain_full_cl` *before preparation*.
The default keeps sufficient 40-bin statistics for the selected analysis.

## Observables, MOPED and SBI handoff

The observable is the split cross spectrum of W(By+n1) and W(By+n2): 2 arcmin
Gaussian beam on the signal, the same cached cap mask with support fsky=0.4 and
60 arcmin apodization, and SO baseline/deprojection 0. Training uses independent
Gaussian SO noise in both splits of every row. Its master seed is 12345, but the
actual two MersenneTwister seeds vary with immutable row ID and split. The seed
is the first eight big-endian SHA256 bytes of the ASCII string
`halfdome-so-v1|12345|ROW_ID|SPLIT`, shifted right one bit. Split is 1 or 2;
training row IDs start at zero, and preflight uses IDs 524288 through 524293.
The complete seed array is saved in the design and exported with the dataset.
Retries and the first 8192 rows of a larger design reproduce exactly those seeds.

FLAMINGO observations keep their original fixed noise splits 22446/22447 and
the same mask. `independent_noise.jl` loads the verified cached mask and uses
the original SO table reader, harmonic draw and synthesis functions. Each split
uses the original table N_ell without a factor of two. There is no additional
noise beam, fsky division or deconvolution.
Bins are mode-count weighted linear D_ell in 40 bins over ell=80..7979, final
bin width100. Clean spectra and noisy cross spectra are stored separately.
Noisy bins may be negative; do not log-transform or clip them.

`collect.py --observations-only` exports existing FLAMINGO and HalfDome reference
spectra through exactly the same binning. It downloads no new simulation maps.
The final dataset exports physical `theta.npy`, `masked_clean_dl40.npy`,
`masked_noisy_cross_dl40.npy` and `unmasked_clean_dl40.npy`.

**Retrain SBI for this support.** The old flow and emulator are not valid outside
their training domain. Refit/validate MOPED on the new design, checking retained
information across this broader family; nine coefficients are not automatically
sufficient over an extended prior. The 40-bin outputs preserve that option.
Independent training noise samples the instrumental-noise distribution across
the dataset. The noise cross term has zero ensemble mean, but each realized
cross spectrum includes signal-noise and noise-noise fluctuations. This does
not introduce independent cosmological skies: the catalogue and mask remain
fixed. Noise-marginalized posterior accuracy still needs held-out calibration.

`sbi_prior.ExtendedPrior` supplies IID rejection samples and joint-support
`log_prob` for subsequent NPE training. Its normalizing mass Z is numerically
estimated in `audit.json`; the uncertainty is recorded. Never substitute a
BoxUniform over the bounding rectangle. The log-uniform P0/xc density includes
the 1/(P0 xc) Jacobian. An old prior pickle must not be reused. This preparation
does not train a new posterior or establish FLAMINGO posterior coverage.

## Resources

The earlier full-map pilot measured median294 s and peak11.3 GiB; these are
planning measurements, not guaranteed timing for this design. At that rate:
8192 maps cost ~670 worker-hours (~17411 core-hours); 524288 cost ~42858
worker-hours (~1.11 million core-hours). With four workers the ideal walltimes
are ~7 and ~446 days, excluding queueing, I/O and retries. Six-map preflight
measurements are recorded separately before committing to production. These
historical timing estimates exclude the new cost of synthesizing two noise
maps per row; use the independent-noise preflight and production timing to
revise the estimate. The first preflight includes an extra replay diagnostic.

Three Float64 40-bin arrays plus Float64 parameters need ~8.1 MiB for 8192 and
~516 MiB for 524288. HDF5 metadata, per-row provenance and compressed logs add
overhead; measure it from preflight. Retaining all three unbinned Float64 Cl
arrays adds ~1.46 GiB / ~93.5 GiB respectively. Existing maps, shared mask/noise
and the 14.4 GiB catalogue are reused. Per-worker temporary cache is ~128 MiB.
No collection of 8192 or 524288 full pixel maps is written.

## Files

All new code is isolated here: `prior.json`, `run_config.json`, `prior.py`,
`prepare.py`, `audit.py`, `sbi_prior.py`, `stable_los.jl`, `paint_row.jl`,
`worker.py`, `run_worker.pbs`, `run_collect.pbs`, `submit.py`, `validate.py`,
`collect.py`, `check_workflow.py`, `plot_preflight.py`, `noise_seeds.py`,
`independent_noise.jl`, `check_noise_seeds.py`, and this
README. Generated designs, audit tables, gate results, figures and run manifests
are artifacts, not edits to the original HalfDome or FLAMINGO workflows.
