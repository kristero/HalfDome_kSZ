# `analysis/` - shared library of the overlap/covariance study (plan step 0.3)

Reusable code for every later test (A overlap, B covariance, D literature recipes) of
`overlap_covariance_L1_m9_0077_20261012`. Built and tested on the radii234 extraction
(4,375 centrals, 36 shells to 4 R200c, four membership components). Written grid-agnostic for
the coming 69-shell x200 / 40-shell x500 extraction.

```python
import sys
sys.path.insert(0, "/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012")
import analysis
from analysis import io, masks, stacking, covariance, stability, fitting, pipeline
```

Run with `/home/cbllover/miniconda3/envs/flamingo/bin/python -B` (numpy 2.2.6, scipy 1.17.1, matplotlib 3.11.2).
Set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` **before** importing numpy when running
many worker processes (otherwise OpenBLAS threads oversubscribe the machine).

## Read-only legacy code (`_legacy.py`)

Earlier studies are never put on `sys.path` and never written to. `_legacy.load(name, path, provides)`
executes a file from its exact path under a private module name; bare imports inside it
(`import pilot_physics as P`) are satisfied by temporarily registering the dependency, and byte-code
writing is switched off during the load.

| Loader | File (radii234 unless stated) | Used for |
|---|---|---|
| `pilot_physics()` | `pilot_physics.py` (36-shell `EDGES`, `SUM_NAMES`, `p200`, `B12`, `B16`) | constants, normalisation |
| `models()` | `models.py` (`shape`, `shell_grid`, `shell`, `BOUNDS`) | the B12/B16 shell averages |
| `measure()`, `fit_study()` | `measure.py`, `fit_study.py` | parity tests only |
| `analysis_common()` | `sensitivity_projects_*/analysis_common.py` | parity test of `weighted_stack` |
| `frozen_source_check()` | `executed_sources.json` | hashes of the loaded sources |

`scatter_r4_bins02_*/pilot_physics.py` has a different `EDGES`; never use it here.

## Data contract (`io.py`)

A plain dict; every future extraction must provide it:

| Key | Content |
|---|---|
| `sums` | float64 `[halo, *middle, shell]` linear physical sums (shell axis last) |
| `axes`, `coords` | axis names (first `halo`, last `shell`, one middle axis `quantity`) and labels of every middle axis |
| `edges` | `[shell+1]` edges in units of the normalising radius |
| `model_edges` | edges for model shell averages (radii234: local `pilot_physics.EDGES`, see below) |
| `radius_mpc`, `mass_msun`, `overdensity` | normalising R and M (R200c/M200c or R500c/M500c), 200 or 500 |
| `baryon_fraction`, `softening_mpc` | 0.158823..., 0.0057 Mpc for L1_m9 |
| `hsml_median_x` or `hs_hist` + `hs_edges` | upper-median smoothing length / R per halo-shell (or its histograms) |
| `halos` | catalogue arrays incl. `mass_bin`, `inclusion_weight` (= population / sample of the stratum), `logmass` |

radii234 layout: `sums[halo, component(4), quantity(11), shell(36)]`, components
`central_bound, own_satellites_bound, other_halos_bound, unbound`, quantities `pilot_physics.SUM_NAMES`.
The coming pass (owner x phase x quantity x shell) only needs new `axes`/`coords`.

| Function | Meaning |
|---|---|
| `load_radii234()` | radii234 `measurements.npz` + `selected_halos.npz` + `metadata_plan.json` |
| `profile(data, field, selection='all', rows=None, **select)` | per-halo normalised profile `[halo, shell]` |
| `component_profiles(data, field, rows)` | normalised `[halo, *middle, shell]` |
| `linear_sum(data, quantity, selection, rows, **select)` | un-normalised sums `[halo, shell]` |
| `integrated(values, edges, cuts)` | dimensionless volume integrals sum F_s (x_hi^3 - x_lo^3) inside each cut |
| `rebin(data, new_edges)`, `rebin_sums(sums, edges, new_edges)` | exact `np.add.reduceat` rebinning onto any sub-grid |
| `nearest_edge`, `shell_centres`, `bin_rows`, `lee22_bins` | helpers |

Fields: `pressure_noagn` (y_noagn, default for tSZ), `pressure` (y, all gas, control), `electrons`
(mu_XGP n_e V), `gas` (mass). Selections (component axis): `all [0,1,2,3]`, `no_other [0,1,3]`,
`no_satellites_or_other [0,3]`, `host_bound [0,1]`; other middle axes are chosen by keyword
(`phase=['xray','other']`).

Normalisation (exactly radii234 `extract_study.merge_products`, same floating-point order): the
all-gas halo normalisation and the full geometric shell volume for every component;
pressure `sum(y) m_e c^2/sigma_T/(V Mpc)/P_Delta` with `P_Delta = G M rho_Delta f_b/(2R)`,
electrons `sum(n_e V) mu_XGP/(V rho_c f_b)`, gas `sum(m)/(V rho_c f_b)`,
`rho_c = M/(4pi/3 R^3)/Delta`. Selected components are normalised, then summed in index order (as
`fit_study`).

Edges: `measurements.npz` stores the edges computed on maas; the local recomputation of
`pilot_physics.EDGES` differs by one ULP at edge 15 (0.2335). Volumes use the stored edges; model
shell averages use `model_edges` (= local `EDGES`, which radii234 `fit_study` used). The rebuilt
profiles agree with the stored normalised arrays to <= 1.2e-15 (ULP level; the stored arrays were
computed on maas).

Rebinning: linear sums are exact. Smoothing lengths: exact upper medians from `hs_hist`, otherwise
the conservative maximum of the fine-shell medians.

## Resolution masks (`masks.py`)

`resolution_masks(data, rows, **options)` generalises `fit_study.resolution_masks` (60-70) to any grid:
a halo-shell is resolved if its inner edge >= max(0.04, 4 softening/R), it holds >= 20 particles
(all components), and inner edge >= 2 x upper-median smoothing length. `strict`: every halo of the
bin resolved; `diagnostic`: geometric criterion for every halo, >= 200 particles in total and
particles in >= 50% of haloes; `supported`: >= 6 strict shells with outer edge <= 2; `base` = strict
if supported else diagnostic. `fit_mask(base, edges, cut, mean)` adds outer edge <= cut + 1e-12
and mean > 0. All options (0.04, 4, 20, 2, 2.0, 6, 200, 0.5) are keyword arguments.

## Stacks and paired bootstraps (`stacking.py`)

`mean_stack`, `median_stack`, `log_mean_stack` (geometric mean), `weighted_stack` (verbatim
`analysis_common.weighted_stack`: mean, reliability-weighted SD, N_eff), `inclusion_weights`.
`bootstrap_rows(n, draws, seed)` = `default_rng(seed).integers(0, n, size=(draws, n))`, the radii234
generator call; with seed `20261008 + bin` the first 128 rows equal radii234's draws for any
`draws >= 128`. `stratified_bootstrap_rows` resamples within strata. `bootstrap_means` uses
multiplicity matrices (memory-light). **Pairing:** one `rows` matrix per halo sample, reused for
every variant (field, selection, AGN estimator, cut-off, objective).

## Covariances (`covariance.py`)

Two different objects:
* intrinsic scatter (Lee22 Fig. 1): `halo_log_covariance(values, mask, weights, logmass, residualise)`,
  covariance/correlation of per-halo ln F (haloes with F <= 0 on the mask are dropped and counted);
* sampling covariance of ln(mean F) for the fit: `bootstrap_log_mean_covariance(values, mask, rows)`
  (>= 1000 draws), delta-method limit `delta_log_mean_covariance` = Cov(F)/(N m m^T), whose
  correlation is that of the *linear* per-halo F (bootstrap and delta method agree to 0.89-1.05 in
  SD and <= 0.11 in correlation for the supported radii234 bins, pressure and electrons, N >= 3p).

Shrinkage: `shrunk_covariance(C, target_corr, X | var_r, force_pooled)` returns
`diag(sd) [lam T + (1 - lam) R] diag(sd)` with `lam = sum Var(r_ij) / sum (r_ij - t_ij)^2`
(Schaefer-Strimmer form, off-diagonal sums). **Var(r_ij) is estimated from halo resamples**
(`correlation_bootstrap_variance`, default 50 draws): the analytic Schaefer-Strimmer variance
(`schafer_strimmer_lambda`) ignores the self-normalisation of r and overestimates lam ~2.5x at N = 300
for the heavy-tailed (lognormal) linear profiles, which makes shrinkage *worse* than no shrinkage
(tests: shrinkage). `force_pooled=True` (lam = 1) gives the pooled correlation x the bin's own
variances, used when N < 3p (p = number of fitted shells).
`pooled_correlation(corrs, masks, ns, n_shells, lnx, projection='clip')`: (N-3)-weighted Fisher-z
average on a common grid, uncovered shells filled from the nearest covered shell, projected to a
unit-diagonal PSD matrix (`nearest_correlation`, `method='clip'` = eigenvalue clipping + diagonal
rescaling, used by the step-0.3 products, or `'higham'` = Higham 2002 alternating projections,
15-25% smaller Frobenius change); the projection change is reported. Pooling all supported bins is
acceptable for pressure (projection changes entries by 0.02-0.065, median 0.04, over the 55 pressure
targets of q4) but not for electron/gas densities, whose correlation structure drifts with mass
(changes 0.03-0.13, median 0.095; raw pooled minimum eigenvalues -0.35 to -0.65): pool neighbouring
bins only in later tests.
`factor_analysis(S, k)`: EM maximum-likelihood factor model L L^T + diag(psi) (works for N < p).
`corr_model`/`fit_corr_model`: rho_ij = a + (1-a) exp(-|ln x_i - ln x_j|/L), a profiled in closed form.
`hartlap(n, p)` = (N - p - 2)/(N - 1): reported only.

## Stability (`stability.py`)

Residualised ln F (`residualise`: bin mean and linear log10 M trend removed, unit variance per shell)
on the shells strictly resolved in both bins. `permutation_test(Za, Zb, lnx, n_perm, seed)`: pool,
reassign labels at the original N_a, N_b, recompute; metrics `frobenius_rms` (off-diagonal RMS of
R_a - R_b), `delta_a`, `delta_lnL` (2-parameter model), `max_angle_deg` (largest principal angle
between top-2 eigen-subspaces); p = (1 + #{T_perm >= T_obs})/(1 + n_perm); min-p combination
calibrated on the same permutations. `bootstrap_model` gives (a, L) bootstrap distributions
(residualisation redone per draw); `decorrelation_radius` is descriptive.

## Fits (`fitting.py`)

Model: `models.shape` averaged with `models.shell_grid` (32-point Gauss-Legendre, volume weights);
parameters fitted in ln theta with `models.BOUNDS`, x0 = clip(ln start, lo + 1e-7, hi - 1e-7),
`least_squares(max_nfev=500, ftol=xtol=gtol=1e-9)`. Default start: published B12/B16.

| Objective | Weighting of ln(mean) residuals |
|---|---|
| `objective_a(family, mean, std, mask)` | (a) diagonal sigma = max(SD/mean, 5%) - radii234 `fit_one` |
| `objective_b(family, mean, sigma_log, mask)` | (b) diagonal error on ln mean (no floor) |
| `objective_c(family, mean, cov_stat, mask, sigma_d='solve')` | (c) GLS, Cholesky-whitened; optional discrepancy sigma_d^2 I iterated to chi2/nu = 1 |

`jac='2-point'` reproduces `models.fit_profile` bit for bit (diagonal mode); `jac='analytic'` uses the
shell-averaged derivative d ln S/d ln theta = <F d ln F/d ln theta>_shell / S (B12 and B16 forms as
`extend_analytic.coefficient_derivatives`), checked against central differences to 1e-10.
Note (a) and (b) are identical up to the 5% floor and bootstrap-vs-SD/sqrt(N) differences: a diagonal
fit is invariant to a global rescaling of sigma.
sigma_d: for fixed theta, chi2(s) = sum_k u_k^2/(lambda_k + s) (C_stat = U diag(lambda) U^T) is solved
for s with brentq, theta refitted, repeated to a relative change < 1e-5 (0 if chi2/nu <= 1 at s = 0).
Acceptance (`accepted`): success, bound distance > 1e-5, Jacobian condition < 1e6 (fit_study).
`influence(J, C)`: rows H = (J^T C^-1 J)^-1 J^T C^-1 (d ln theta / d ln mean_s) and
N_eff = (sum|H_k|)^2/sum H_k^2 per parameter; `fit_influence(fit)` uses the fit's own weighting.
`cross_fit(...)`: covariance from one random half, mean from the other, swap, average ln theta.

## Flexible families and observable-driven objectives (`families.py`, `fitting.py` extension)

Added for the observable-driven calibration test. The legacy objectives (a)-(c) and their B12/B16
code paths are unchanged (byte-identical source; the q4 results were produced before the extension
and reproduce bit for bit with the final library, see `quicklook/verification.json`).

Families (`families.get(name)` or a `Family` instance; theta = ln p for positive parameters,
p for signed ones; same `models.shell_grid` shell averages, bounds in natural units, same acceptance):

| Family | Parameters | Form |
|---|---|---|
| `b12`, `b16` | P0, xc, beta / rho0, alpha, beta | radii234 `models.shape` |
| `gnfw5` | P0, xc, alpha, beta, gamma | P0 q^-gamma (1+q^alpha)^-((beta-gamma)/alpha), q = x/xc; B12(P0, xc, beta) = gnfw5(0.5176 P0, xc, 1, beta+0.3, 0.3) |
| `b12env` (`b12_env_family(pivot=2)`) | P0, xc, beta, A_env, s_env | B12 + A (x/2)^-s; s signed in [-2, 10] (s = 0: constant floor) |
| `b12_spline_family(knots_x, prior_sigma)` | P0, xc, beta, c_1..c_K | B12 [1 + s(ln x)], natural cubic spline through K = 3-4 knot values, constant beyond the end knots, ridge penalty residuals c_k/prior_sigma; the B12 parameters and the correction are partly degenerate (profile recovered, decomposition not unique) |

`families.shell_model`, `shell_model_jacobian` (analytic d ln S/d theta for every family, checked
against central differences to ~1e-9), `shell_model_truncated(family, p, edges, cut)` (model x
Theta(x < cut) averaged over the full shell volumes; a shell split by the cut is integrated exactly).

General fitter `fitting.fit_general(family, mean, mask, sigma= | cov= | weights= | metric=,
extra_residuals=[...], start, edges, jac)`: residual vector = whitened (ln model - ln mean) on the fit
shells + family penalty residuals + appended extra residuals; analytic Jacobian for the profile and
penalty rows, finite differences only for the extra rows. Objectives:

| Objective | chi^2 | Function |
|---|---|---|
| (d) importance-diagonal | sum_s w_s r_s^2 for any weight vector | `objective_d(family, mean, w, mask)` |
| (e) quadratic metric | r^T M r, M PSD (W = l^1/2 U^T on its positive eigen-space; may be rank-deficient) | `objective_e(family, mean, M, mask)` |
| (f) hybrid | r^T C^-1 r + lambda r^T K r; K trace-normalised so that tr(lambda K) = lambda tr(C^-1) (`normalise=None` to switch off) | `objective_f(...)`, `hybrid_scan(..., lams=(0.1, 1, 10, 100))` |
| (g) extra residuals | any of the above + sum_j e_j(p)^2 (direct observable-space terms) | `objective_g(family, mean, mask, extras, <metric kw>)` |
| (a), (c) for any family | as legacy | `objective_a_general`, `objective_c_general` (same sigma_d iteration) |

`observable_metric(dO_dF [n_obs, shell], O, lambda_O, F)` builds K = sum_O lambda_O g_O g_O^T / O^2
with g_O,s = F_s dO/dF_s = dO/d ln F_s, the conversion needed because every residual is in ln F.

Residual diagnostics: `shell_residuals(family, p, mean, edges, cut=None)` (relative S/m - 1 and
ln S - ln m on every shell; shells beyond a cut have relative residual -1),
`weighted_rms(values, weights, mask)`, `region_rms(relative, edges, mask)` (inside 1R, 1-2R, 2-4R),
and `fit_cut_grid(family, mean, base, edges, fit_radii, cut_radii, fitter, importance)`: fits to every
R_fit with the caller's `fitter(mask)`, truncates at every R_cut, evaluates on all resolved shells to
4R and returns importance-weighted and per-region RMS tables plus the minimising (R_fit, R_cut).
Demonstration: `quicklook/q7_importance_demo.py`.

## New particle-pass products (`newdata.py`, `variants.py`)

Loaders and variant constructors for the Step 1.2 products, so Tests A, B, D and F run as soon as the
real files arrive (same names, in the study root). Developed and validated on the synthetic rehearsal
fixture `tests/rehearsal_work/study/`, which has the exact schema; `tests/test_newdata.py` (14 checks)
writes `tests/test_newdata_results.json`.

```python
from analysis import newdata, variants as V
d = newdata.load()                    # study root (real data); newdata.load_fixture() for the rehearsal
p = V.profile(d, "pressure", estimator="geometric", agn="removed", gas_selection="all", grid="x200")
p.values        # [halo, shell] normalised (P/P200 on X200 grids, P/P_e,500 (B24) on X500 grids)
```

**`newdata.NewData`** (lazy, cached arrays; axis names from `schema.json`, checked against
`measure_overlap`): `array(key)` for every block (`owner200`, `owner500`, `aux200/500`, `hs200/500`,
`excision`, `partition`, `legacy_sums`, `apertures`, `overlapper`, ...); `grid('x200'|'x500')` (stored
edges, checked against `grids.py`); `subgrid_index(grid)`, `edges(grid)`, `rebin(values, idx)` for
`'x200'` (67), `'old'` (36), `'lee22'` (27), `'x500'` (40), `'b24'` (30) or `('x200'|'x500', edges)`
(indices are computed on `grids.py` and applied to the stored grid, so ULP differences between machines
cannot break rebinning); per-halo scales `p200()` (radii234), `rho_c()`, `p500('B24'|'A10'|
'self_similar_500')`, `kT500()`, `ne500()` (B24: f_B = 0.159, mu = 0.59, mu_e = 1.14);
`inclusion_weights()` (= 1 / stratum inclusion probability); census classes `census_class(mu_min,
kind)` (0 clean, 1 cont_reach1R, 2 cont_annulus_only) for mu >= 0.03 / 0.1 / 0.3 and kind 'all' /
'own_satellites' / 'neighbours' / 'neighbour_centrals' from the overlapper lists, cross-checked with
`census_class_from_max_mu`; `overlapper_table()`, `overlapper_select(...)`; `legacy_dataset()`
(measurements_legacy.npz as an `analysis.io` dataset).

**`variants.profile(data, field, estimator, agn, gas_selection, grid, excision=None, template=None,
normalisation='auto', rows=None)`** -> `Profile` (`values`, `edges`, `retained_fraction`, linear
`sums`, `template`, `info`). Linear sums are rebinned before any ratio is formed.

| Option | Values |
|---|---|
| field / estimator | pressure: `geometric` (sum y / V_shell; tSZ/DM-correct), `Q_V` (electron_kT / volume), `Q_V_y` (sum y m_e c^2/sigma_T / sum V), `Q_M` (mass_pe / mass), `Q_X` (lx_pe / lx_rosat, X-ray phase), `nT_X` (geometric n_e x k T_X); electrons: `geometric`, `Q_V`; gas: `geometric`, `Q_V`, `Q_M` (clumping-biased) |
| diagnostics | `diagnostic(data, 'T_M'|'T_V'|'T_X'|'clumping_density'|'clumping_pressure', ...)` |
| agn | `removed` (phases xray_selected + other = radii234 y_noagn; default), `kept` (all phases); excision/partition: AGN axis |
| gas_selection | `all`, `no_other`, `no_satellites_or_other`, `host_bound`, `central_bound`; `selection('remove_overlappers', mu_min=0.03/0.1/0.3, reach='any'/'reach1R'/'annulus', flavour='remove_all'/'keep_own_satellites')`; A2-i `remove_bound_to_other_painted` (owner 14 + per-halo neighbour-overlapper codes whose overlappers are painted); A2-ii `partition_owned`, `partition_owned_not_other_painted` |
| excision | `{'variant': 'all'/'neighbours_only'/'placebo_iso'/'placebo_nbr', 'kappa': 0.5/1/2, 'members': 'all'/'no_overlapper_bound'/'central_bound', 'min_retained': 0.5}`: retained sum / (V_shell (1 - f_exc)); NaN below 50% retained; `excision_common_set(...)` / `common_halo_set(...)` give the fixed halo set per shell |
| template | `{'model': F(M, x), 'mu_min', 'reach', 'kind': 'all'/'neighbour_centrals'/'neighbours'/'own_satellites', 'truncation': 1.0 (R_j), 'amplitude': 'model'/'measured_satellites'/'measured_all'}`: subtracts sum_k of the exact shell average of each overlapper's model profile at its offset (`geometry.offset_shell_average`), with its own R_j and P200(M_j, R_j); measured amplitudes match the overlapper's bound y / electrons / mass (`apertures.npz` overlapper block, whole sphere) |
| grid | as above; excision and partition exist on X200 sub-grids only |
| normalisation | `auto` (P200 on X200 grids, B24 on X500 grids), `P200`, `B24`, `A10`, `self_similar_500` |

Owner codes (17): 0 central-bound; 1-12 overlapper-bound = 1 + (mu class x 2 + [annulus only]) x 2 +
[neighbour] with mu classes [0.03, 0.1), [0.1, 0.3), >= 0.3; 13 other own satellites; 14 other bound,
painted host; 15 other bound, not painted; 16 unbound. radii234 components: central = 0, own satellites =
odd 1-11 + 13, other haloes = even 2-12 + 14 + 15, unbound = 16.

Also: `stack(values, weights, mask)` (NaN-aware, inclusion-weighted per-shell mean, n, n_eff),
`owner_sum`, `excision_sum`, `partition_sum`, `excision_fraction`, `template_profiles`,
`template_amplitude_scale`, `model_integral`, `owner_masks`, `painted_overlappers`, `a2i_ambiguity`,
`compare_with_legacy(data, legacy, rows)` and `radii234_reference_dataset(data, path)`.

Validation on the fixture (`tests/test_newdata_results.json`, all 14 passed):
- (i) all gas, AGN kept, geometric, rebinned to the old 36 shells equals the measurements_legacy.npz
  profile to 3.9e-16, every field and radii234 selection to 1.7e-15 (float sums reordered; counts exact);
- (ii) AGN removed equals legacy y_noagn (4.3e-16);
- (iii) excision zones, members and AGN axes and the partition totals sum to the owner totals (2.4e-15),
  aux sums equal the legacy H/He/ionised-electron sums;
- (iv) zero-sphere excision returns the unexcised profile (1.1e-15, retained fraction exactly 1), excised
  fractions grow with kappa, NaN exactly below 50% retained;
- (v) template subtraction recovers a Monte Carlo-injected truth to 0.89% of the largest injected value
  (exact injection 5.8e-16; measured-amplitude scaling exact);
- (vi) constant inclusion weights reproduce the equal-weight stack (3e-15).

Further checks: census classes (main definition equal; 0.03/0.3 consistent with census_max_mu); grid
identities (X200 -> old / Lee22 / custom, X500 -> B24); all estimators on a synthetic volume-filling
block (8.9e-16); B24/A10 normalisations; owner algebra of the selections; the radii234 reference
(rows 0-69 of the fixture, 1.7e-15). The fixture's particle values are synthetic, so estimator
magnitudes on it are not physical; formulas are validated on the synthetic block instead.
On the real data run `python -B analysis/tests/test_newdata.py --data <study root> --radii234
../radii234_contamination_L1_m9_0077_20261008/measurements.npz` (it also compares radii234's stored
normalised arrays, rescaled by the f_b ratio).

Known limitations: excision and partition blocks carry only count/mass/volume/y/electrons and an AGN
axis (no phases, so no Q_X/T_X there), and exist on X200 only; an owner code can mix painted and
unpainted neighbour overlappers, which A2-i resolves by bound-gas-weighted majority (`a2i_ambiguity`
counts such cells; none in the fixture); template subtraction loops over haloes and overlappers in
Python (~5 ms per overlapper).

## Per-bin estimators (`pipeline.py`)

`run_variant(family, values, base, edges, cutoffs, rows, inner, pooled_corr, objectives)`: for the
full sample and every replicate row it stacks, re-estimates Cov[ln mean] from `inner` resamples of
that sample (common random numbers), and fits (a), (b), (c) at every cut-off; replicates warm-start
from the full-sample solution (as radii234 did); only accepted fits are kept. Returns
`[draw, cut, objective, parameter]` arrays (draw 0 = full sample). `influence_table`, `summarise`,
`paired_difference` post-process them. Objective (c) here: bootstrap covariance, shrinkage towards a
pooled correlation (lam from 50 resamples), lam = 1 if N < 3p, sigma_d solved.

## Plot style (`plotting.py`)

Categorical order `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948`; diverging
`#2a78d6 - #f0efec - #e34948` (`DIVERGING`, bad = `#d9d7d2`); text `#0b0b0b`/`#52514e`; thin
lines, recessive grid; `save(fig, path)` writes PNG (150 dpi) and SVG.

## Tests

`python -B analysis/tests/run_tests.py` (add `--quick` for a smaller mock run) writes
`tests/test_results.json`: legacy parity (frozen sources, normalisation, masks, bit-exact objective
(a), weighted stack, legacy bootstrap rows), rebinning (synthetic 69-shell x200 and 39-shell x500 grids,
exact counts and hs-histogram medians), analytic vs numerical Jacobian, Hartlap, factor model,
shrinkage (small N), sigma_d recovery, the extension (family Jacobians, gNFW/B12 identity,
truncation exactness, general objectives (d)-(g) vs the legacy (a)/(c) and their lambda limits,
noise-free recovery for gnfw5/b12env/b12spline), permutation-null false-positive rate and power,
and GLS parameter recovery/coverage (analytic and bootstrap 68% intervals, cross-fitting). See
`verification.json` for the latest run.
