"""Library tests: legacy parity on radii234 data and Gaussian mock tests.

Run from anywhere:  <flamingo python> -B analysis/tests/run_tests.py [--quick]
Writes analysis/tests/test_results.json.  Every 'passed' flag is computed here.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import analysis  # noqa: E402
from analysis import _legacy  # noqa: E402
from analysis import covariance as C  # noqa: E402
from analysis import families as FAM  # noqa: E402
from analysis import fitting as F  # noqa: E402
from analysis import io as aio  # noqa: E402
from analysis import masks as msk  # noqa: E402
from analysis import stability as ST  # noqa: E402
from analysis import stacking as S  # noqa: E402

QUICK = "--quick" in sys.argv
WORKERS = 8
RESULTS: dict = {}
DATA = None
THETA_TRUE = np.array([15.8, 0.762, 5.66])


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def record(name, passed, started, **details):
    RESULTS[name] = {"passed": bool(passed), "seconds": round(time.perf_counter() - started, 3), **details}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  ({RESULTS[name]['seconds']} s)", flush=True)


# ============================================================================ legacy parity
def test_frozen_sources():
    t = time.perf_counter()
    check = _legacy.frozen_source_check()
    passed = check["pilot_physics.py"]["matches_executed_sources"] and check["models.py"]["matches_executed_sources"]
    record("legacy_sources_frozen", passed, t, detail=check,
           note="pilot_physics.py and models.py must equal radii234 executed_sources.json; "
                "fit_study.py was edited locally after the extraction (informational).")


def test_normalisation():
    t = time.perf_counter()
    out = {}
    worst = 0.0
    for field in ("pressure", "electrons", "gas"):
        mine = aio.component_profiles(DATA, field)
        stored = DATA["stored"][aio.FIELDS[field]["stored"]]
        rel = np.abs(mine - stored) / np.where(stored != 0, np.abs(stored), 1)
        out[field] = {"max_relative_deviation": float(rel.max()), "bit_identical_fraction": float(np.mean(mine == stored))}
        worst = max(worst, rel.max())
    for field, key in (("pressure_noagn", "pressure_noagn_ratio"), ("pressure", "pressure_ratio")):
        mine = aio.profile(DATA, field, "all")
        stored = DATA["stored"][key]
        rel = np.abs(mine - stored) / np.where(stored != 0, np.abs(stored), 1)
        out[f"{field}_total_vs_{key}"] = float(rel.max())
        worst = max(worst, rel.max())
    p200 = aio.pressure_scale(DATA)
    out["p200_max_relative_deviation"] = float(np.max(np.abs(p200 / DATA["stored"]["p200_pa"] - 1)))
    out["rho_c_max_relative_deviation"] = float(np.max(np.abs(aio.critical_density(DATA) / DATA["stored"]["critical_density_msun_mpc3"] - 1)))
    record("normalisation_reproduces_stored_arrays", worst < 1e-14, t, worst_relative_deviation=float(worst), detail=out,
           note="ULP-level differences: stored arrays were computed on maas (different numpy/CPU); "
                "stored measurements.npz edges differ from local pilot_physics.EDGES by 1 ULP at edge 15.")


def test_masks_parity():
    t = time.perf_counter()
    fs = _legacy.fit_study()
    halos = DATA["halos"]
    legacy_data = {"sums": DATA["sums"], "hsml_median_upper_x": DATA["hsml_median_x"]}
    mismatches = []
    for b in range(15):
        rows = aio.bin_rows(DATA, b)
        mine = msk.resolution_masks(DATA, rows)
        base, resolved, strict, supported = fs.resolution_masks(halos, legacy_data, rows, DATA["softening_mpc"])
        saved = np.load(_legacy.RADII234 / "products" / f"{_saved_profile_name(b)}_profiles.npz")
        ok = (np.array_equal(mine["base"], base) and np.array_equal(mine["resolved"], resolved)
              and np.array_equal(mine["strict"], strict) and mine["supported"] == supported
              and np.array_equal(mine["base"], saved["fit_base_mask"]) and np.array_equal(mine["strict"], saved["strict_mask"]))
        if not ok:
            mismatches.append(b)
    record("resolution_masks_match_fit_study", not mismatches, t, mismatched_bins=mismatches, bins=15)


def _saved_profile_name(b):
    lo, hi = DATA["mass_edges"][b], DATA["mass_edges"][b + 1]
    return f"bin_{lo:.1f}_{hi:.1f}".replace(".", "p")


def test_objective_a_bit_exact():
    t = time.perf_counter()
    worst = 0.0
    count = 0
    nonzero = 0
    for b in range(15):
        rows = aio.bin_rows(DATA, b)
        case = json.loads((_legacy.RADII234 / "products" / f"{_saved_profile_name(b)}_fits.json").read_text())
        base = np.asarray(case["fit_base_mask"], bool)
        for selection, comps in aio.SELECTIONS.items():
            for field, key in (("pressure", "pressure_components"), ("electrons", "electron_components"),
                               ("gas", "gas_components")):
                values = DATA["stored"][key][rows][:, list(comps)].sum(axis=1)
                mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
                family = aio.FIELDS[field]["family"]
                for entry in case["selections"][selection][field]:
                    if entry["fit"] is None:
                        continue
                    mask = base & (DATA["model_edges"][1:] <= entry["radius_R200c"] + 1e-12) & (mean > 0)
                    fit = F.objective_a(family, mean, std, mask, edges=DATA["model_edges"], jac="2-point")
                    diff = np.abs(np.array(fit["parameters"]) - np.array(entry["fit"]["parameters"]))
                    worst = max(worst, float(diff.max()))
                    nonzero += int(np.any(diff > 0))
                    count += 1
    record("objective_a_bit_exact_on_stored_inputs", worst == 0.0, t, fits=count,
           max_absolute_parameter_difference=worst, fits_not_bit_identical=nonzero)


def test_weighted_stack_parity():
    t = time.perf_counter()
    ac = _legacy.analysis_common()
    rng = np.random.default_rng(5)
    values = rng.lognormal(size=(400, 36))
    weights = rng.choice([1.0, 54.6, 206.5], size=400)
    a = S.weighted_stack(values, weights)
    b = ac.weighted_stack(values, weights.copy())
    same = all(np.array_equal(x, y) for x, y in zip(a[:2], b[:2])) and a[2] == b[2]
    w = S.inclusion_weights(DATA["halos"]["mass_bin"], DATA["population_per_bin"])
    same_weights = np.array_equal(w, DATA["halos"]["inclusion_weight"])
    record("weighted_stack_and_inclusion_weights_parity", same and same_weights, t,
           weighted_stack_identical=same, inclusion_weights_identical=same_weights)


def test_bootstrap_rows():
    t = time.perf_counter()
    ok = all(np.array_equal(S.bootstrap_rows(n, 1000, 20261008 + b)[:128], S.bootstrap_rows(n, 128, 20261008 + b))
             for b, n in [(0, 300), (10, 895), (14, 6)])
    rows = S.bootstrap_rows(300, 64, 3)
    vals = np.random.default_rng(1).normal(size=(300, 7))
    direct = np.array([vals[r].mean(axis=0) for r in rows])
    via_counts = S.bootstrap_means(vals, rows)
    strata = np.repeat([0, 1, 2], [50, 30, 20])
    srows = S.stratified_bootstrap_rows(strata, 20, 9)
    strat_ok = all(np.array_equal(np.sort(strata[r]), np.sort(strata)) for r in srows)
    record("bootstrap_rows_legacy_prefix_and_counts", ok and np.allclose(direct, via_counts, rtol=1e-13) and strat_ok, t,
           legacy_prefix_identical=ok, counts_vs_direct_max_rel=float(np.max(np.abs(via_counts / direct - 1))),
           stratified_sizes_preserved=strat_ok)


# ============================================================================ grids / rebinning
def _future_grids():
    old = np.asarray(DATA["edges"], float)
    lee22 = np.geomspace(0.04, 1.34, 28)
    x200 = np.unique(np.r_[old, lee22, 1.5, 1.75, 2.5, 1.01, 1.76])
    b24 = np.geomspace(0.01, 3.0, 31)
    x500 = np.unique(np.r_[0.0, b24, 0.03, 0.15, 1.0, 4.0, 5.0, 6.0, 0.005, 0.02])
    return x200, x500


def test_rebinning():
    t = time.perf_counter()
    rng = np.random.default_rng(11)
    x200, x500 = _future_grids()
    details = {"x200_shells": len(x200) - 1, "x500_shells": len(x500) - 1}
    ok = True
    for name, fine, coarse in (("x200->old36", x200, np.asarray(DATA["edges"], float)),
                               ("x200->2R sub-range", x200, np.asarray(DATA["edges"], float)[9:31]),
                               ("x500->B24-like", x500, np.r_[0.0, np.geomspace(0.01, 3.0, 31)[::2], 3.0])):
        coarse = np.unique(coarse)
        r = np.cbrt(rng.uniform(0, fine[-1]**3, 200000))
        w = rng.lognormal(size=r.size)
        fine_counts = np.stack([np.histogram(r, fine)[0], np.histogram(r, fine, weights=w)[0]])
        direct = np.stack([np.histogram(r, coarse)[0], np.histogram(r, coarse, weights=w)[0]])
        reb = aio.rebin_sums(fine_counts.astype(float), fine, coarse)
        counts_exact = np.array_equal(reb[0], direct[0])
        rel = float(np.max(np.abs(reb[1] - direct[1]) / np.maximum(np.abs(direct[1]), 1e-300)))
        details[name] = {"counts_exact": counts_exact, "weights_max_relative": rel}
        ok &= counts_exact and rel < 1e-12
    # Rebinning the real data: normalised profile on a coarse shell = volume-weighted mean.
    coarse = np.asarray(DATA["edges"], float)[[0, 9, 15, 20, 25, 27, 30, 33, 36]]
    small = {k: v for k, v in DATA.items()}
    rows = aio.bin_rows(DATA, 10)
    reb = aio.rebin(small, coarse)
    fine_p = aio.profile(DATA, "pressure_noagn", "all", rows)
    coarse_p = aio.profile(reb, "pressure_noagn", "all", rows)
    idx = aio.edge_indices(DATA["edges"], coarse)
    vol = np.diff(np.asarray(DATA["edges"], float)**3)
    expect = np.stack([(fine_p[:, a:b] * vol[a:b]).sum(axis=1) / vol[a:b].sum() for a, b in zip(idx[:-1], idx[1:])], axis=1)
    rel = float(np.max(np.abs(coarse_p / expect - 1)))
    details["real_data_volume_weighted_identity_max_relative"] = rel
    ok &= rel < 1e-12
    # hs histogram medians after rebinning equal medians of pooled samples.
    hs_edges = np.geomspace(1e-3, 10, 401)
    fine_edges = np.linspace(0, 4, 9)
    samples = [[rng.lognormal(-2 + 0.1 * s, 0.5, size=rng.integers(5, 200)) for s in range(8)] for _ in range(3)]
    hist = np.stack([np.stack([np.histogram(x, hs_edges)[0] for x in halo]) for halo in samples])
    fake = {"sums": np.zeros((3, 1, 8)), "axes": ("halo", "quantity", "shell"), "coords": {"quantity": ("count",)},
            "edges": fine_edges, "hs_hist": hist, "hs_edges": hs_edges}
    out = aio.rebin(fake, fine_edges[[0, 2, 5, 8]])
    expected = np.empty((3, 3))
    for h in range(3):
        for k, (a, b) in enumerate([(0, 2), (2, 5), (5, 8)]):
            pooled = np.sort(np.concatenate(samples[h][a:b]))
            value = pooled[len(pooled) // 2]
            expected[h, k] = hs_edges[np.searchsorted(hs_edges, value, side="right")]
    hs_ok = np.array_equal(out["hsml_median_x"], expected)
    details["hs_hist_upper_median_exact"] = bool(hs_ok)
    ok &= hs_ok
    try:
        aio.rebin_sums(np.zeros((2, 36)), DATA["edges"], [0.0, 0.5, 1.0])
        details["non_subgrid_rejected"] = False
        ok = False
    except ValueError:
        details["non_subgrid_rejected"] = True
    record("rebinning_exact_and_grid_agnostic", ok, t, **details)


# ============================================================================ Jacobian
def test_jacobian():
    t = time.perf_counter()
    rng = np.random.default_rng(3)
    edges = DATA["model_edges"]
    worst = {}
    for family, centre in (("b12", THETA_TRUE), ("b16", np.array([2000.0, 1.25, 3.88]))):
        errs = []
        for _ in range(40):
            p = centre * np.exp(rng.normal(0, 0.3, 3))
            Sv, J = F.shell_model_jacobian(family, p, edges)
            Jn = F.numerical_jacobian(family, p, edges, step=1e-5)
            scale = np.maximum(np.max(np.abs(J), axis=0), 1e-12)
            errs.append(np.max(np.abs(J - Jn) / scale))
            assert np.allclose(Sv, F.shell_model(family, p, edges), rtol=0, atol=0)
        worst[family] = float(np.max(errs))
    record("analytic_vs_numerical_jacobian", max(worst.values()) < 1e-6, t,
           max_relative_error_per_family=worst, tolerance=1e-6, step_ln_theta=1e-5)


# ============================================================================ mocks
def _mock_setup():
    """Shells and per-halo ln-scatter covariance from the 14.5-14.7 noAGN stack (2R)."""
    rows = aio.bin_rows(DATA, 10)
    base = msk.resolution_masks(DATA, rows)["base"]
    edges = DATA["model_edges"]
    mask = msk.fit_mask(base, edges, 2.0)
    values = aio.profile(DATA, "pressure_noagn", "all", rows)
    logs = np.log(values[:, mask])
    sigma = np.cov(logs, rowvar=False)
    model = F.shell_model("b12", THETA_TRUE, edges)
    return {"mask": mask, "edges": edges, "sigma": sigma, "model": model, "n_shells": int(mask.sum())}


def _draw(setup, n, rng):
    eps = rng.multivariate_normal(np.zeros(setup["n_shells"]), setup["sigma"], size=n)
    values = np.zeros((n, len(setup["model"])))
    values[:, setup["mask"]] = setup["model"][setup["mask"]] * np.exp(eps - np.diag(setup["sigma"]) / 2)
    return values


def _estimate_c(values, mask, edges, target, inner, start=None, shrink=True):
    n = len(values)
    positions = np.arange(n)
    cb = C.bootstrap_log_mean_covariance(values, mask, positions[inner])["cov"]
    if shrink:
        var_r = C.correlation_bootstrap_variance(values[:, mask], positions[inner[:50]])
        Cs, lam = C.shrunk_covariance(cb, target[np.ix_(mask, mask)], var_r=var_r)
    else:
        Cs, lam = cb, 0.0
    fit = F.objective_c("b12", values.mean(axis=0), Cs, mask, start=start, edges=edges, sigma_d="solve")
    return fit, lam


def _coverage_worker(args):
    seed, setup, target, n, n_boot, n_inner = args
    rng = np.random.default_rng(seed)
    values = _draw(setup, n, rng)
    mask, edges = setup["mask"], setup["edges"]
    inner = rng.integers(0, n, size=(n_inner, n))
    out = {}
    for label, shrink in (("gls_bootstrap_cov", False), ("gls_shrunk", True)):
        fit, lam = _estimate_c(values, mask, edges, target, inner, shrink=shrink)
        lnp = np.array(fit["log_parameters"]) if F.accepted(fit) else np.full(3, np.nan)
        sig = np.sqrt(np.diag(fit["cov_log_parameters"]))
        boot = np.full((n_boot, 3), np.nan)
        for k in range(n_boot):
            r = rng.integers(0, n, n)
            sub = values[r]
            fk, _ = _estimate_c(sub, mask, edges, target, inner, start=fit["parameters"], shrink=shrink)
            if F.accepted(fk):
                boot[k] = fk["log_parameters"]
        out[label] = {"lnp": lnp, "sigma": sig, "sigma_d": fit["sigma_d"], "lambda": lam, "boot": boot}
    # diagonal objectives for reference
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    fa = F.objective_a("b12", mean, std, mask, edges=edges, jac="analytic")
    out["diag_a_lnp"] = np.array(fa["log_parameters"])
    # cross-fitting
    def builder(cov_values, mean_values):
        Cd = C.delta_log_mean_covariance(cov_values, mask, n_mean=len(mean_values), mean=cov_values[:, mask].mean(axis=0))
        Cs, _ = C.shrunk_covariance(Cd, target[np.ix_(mask, mask)], X=cov_values[:, mask], rng=rng)
        return Cs
    cf = F.cross_fit("b12", values, mask, builder, rng, n_splits=2, edges=edges)
    out["crossfit_lnp"] = np.array(cf["log_parameters"]) if cf["parameters"] is not None else np.full(3, np.nan)
    return out


def test_gls_coverage():
    t = time.perf_counter()
    setup = _mock_setup()
    n, mocks, n_boot, n_inner = 300, (60 if QUICK else 200), (40 if QUICK else 100), 300
    # Pooled-like target: linear-F correlation from an independent large mock sample.
    big = _draw(setup, 20000, np.random.default_rng(99))
    target = np.eye(len(setup["model"]))
    target[np.ix_(setup["mask"], setup["mask"])] = np.corrcoef(big[:, setup["mask"]], rowvar=False)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        outs = pool.map(_coverage_worker, [(1000 + i, setup, target, n, n_boot, n_inner) for i in range(mocks)])
    truth = np.log(THETA_TRUE)
    details = {"mocks": mocks, "haloes_per_mock": n, "bootstrap_replicates": n_boot, "inner_draws": n_inner,
               "shells": setup["n_shells"], "theta_true": THETA_TRUE}
    passed = True
    se = np.sqrt(0.683 * 0.317 / mocks)
    for label in ("gls_bootstrap_cov", "gls_shrunk"):
        lnp = np.array([o[label]["lnp"] for o in outs])
        sig = np.array([o[label]["sigma"] for o in outs])
        boot = np.array([o[label]["boot"] for o in outs])
        lo, hi = np.nanpercentile(boot, 15.865, axis=1), np.nanpercentile(boot, 84.135, axis=1)
        analytic = np.mean(np.abs(lnp - truth) <= sig, axis=0)
        bootstrap = np.mean((lo <= truth) & (truth <= hi), axis=0)
        bias = np.nanmean(lnp - truth, axis=0)
        details[label] = {"analytic_68_coverage": analytic, "bootstrap_68_coverage": bootstrap,
                          "mean_ln_bias": bias, "mean_ln_bias_over_rms_error": bias / np.nanstd(lnp - truth, axis=0),
                          "rms_ln_error": np.sqrt(np.nanmean((lnp - truth)**2, axis=0)),
                          "fraction_sigma_d_positive": float(np.mean([o[label]["sigma_d"] > 0 for o in outs])),
                          "mean_lambda": float(np.mean([o[label]["lambda"] for o in outs]))}
        passed &= bool(np.all(np.abs(analytic - 0.683) <= 3 * se + 0.02) and np.all(np.abs(bootstrap - 0.683) <= 3 * se + 0.02))
    diag = np.array([o["diag_a_lnp"] for o in outs])
    cross = np.array([o["crossfit_lnp"] for o in outs])
    details["diagonal_a_rms_ln_error"] = np.sqrt(np.mean((diag - truth)**2, axis=0))
    details["crossfit_mean_ln_bias"] = np.nanmean(cross - truth, axis=0)
    details["crossfit_rms_ln_error"] = np.sqrt(np.nanmean((cross - truth)**2, axis=0))
    details["pass_rule"] = "|coverage - 0.683| <= 3 binomial SE + 0.02 for every parameter, analytic and bootstrap"
    record("gls_parameter_recovery_and_coverage", passed, t, **details)


def _perm_worker(args):
    seed, corr_a, corr_b, na, nb, lnx, n_perm = args
    rng = np.random.default_rng(seed)
    p = len(lnx)
    Za = rng.multivariate_normal(np.zeros(p), corr_a, size=na)
    Zb = rng.multivariate_normal(np.zeros(p), corr_b, size=nb)
    Za, Zb = ST.residualise(Za), ST.residualise(Zb)
    res = ST.permutation_test(Za, Zb, lnx, n_perm=n_perm, seed=seed + 1)
    return [res["p_values"][m] for m in ST.METRICS] + [res["p_combined_minp"]]


def test_permutation_null():
    t = time.perf_counter()
    # Residualised ln-F correlations of two real bins on their common shells.
    def corr_of(b, mask):
        rows = aio.bin_rows(DATA, b)
        logs, kept = C.log_values(aio.profile(DATA, "pressure_noagn", "all", rows), mask)
        return np.corrcoef(ST.residualise(logs, DATA["halos"]["logmass"][rows][kept]), rowvar=False)
    m9 = msk.resolution_masks(DATA, aio.bin_rows(DATA, 9))["strict"]
    m12 = msk.resolution_masks(DATA, aio.bin_rows(DATA, 12))["strict"]
    common = m9 & m12
    lnx = np.log(aio.shell_centres(DATA["edges"]))[common]
    R9, R12 = corr_of(9, common), corr_of(12, common)
    pairs, n_perm = (60 if QUICK else 200), (100 if QUICK else 200)
    cases = {"null_equal_N_300_300": (R9, R9, 300, 300), "null_unequal_N_300_21": (R9, R9, 300, 21),
             "power_bin9_vs_bin12_corr_300_300": (R9, R12, 300, 300)}
    details = {"pairs_per_case": pairs, "permutations": n_perm, "shells": int(common.sum()),
               "frobenius_rms_true_difference_bin9_bin12": ST.frobenius_rms(R9, R12)}
    passed = True
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for name, (ra, rb, na, nb) in cases.items():
            ps = np.array(pool.map(_perm_worker, [(7000 + 13 * i, ra, rb, na, nb, lnx, n_perm) for i in range(pairs)]))
            rate = np.mean(ps <= 0.05, axis=0)
            details[name] = dict(zip(list(ST.METRICS) + ["combined_minp"], rate.tolist()))
            if name.startswith("null"):
                # binomial 99.7% band around 5% for this many pairs
                band = 3 * np.sqrt(0.05 * 0.95 / pairs)
                passed &= bool(np.all(rate <= 0.05 + band) and np.all(rate >= max(0.0, 0.05 - band)))
    details["pass_rule"] = "null rejection rates at alpha=0.05 within 0.05 +- 3 binomial SE for every metric"
    record("permutation_null_false_positive_rate", passed, t, **details)


def test_shrinkage():
    t = time.perf_counter()
    setup = _mock_setup()
    rng = np.random.default_rng(21)
    mask = setup["mask"]
    big = _draw(setup, 40000, rng)[:, mask]
    true_cov = np.cov(big, rowvar=False)
    true_corr = C.cov_to_corr(true_cov)
    # imperfect target: correlation of a different real bin's linear profiles on the same shells
    rows = aio.bin_rows(DATA, 9)
    target = np.corrcoef(aio.profile(DATA, "pressure_noagn", "all", rows)[:, mask], rowvar=False)
    details = {"shells": int(mask.sum()), "target_vs_truth_frobenius_rms": ST.frobenius_rms(target, true_corr)}
    ratios = {}
    off = ~np.eye(len(true_corr), dtype=bool)
    for n in (10, 20, 40, 63, 120, 300, 900):
        es, eh, eh_ss, lams, lams_ss, samples = [], [], [], [], [], []
        for _ in range(100 if QUICK else 300):
            X = big[rng.integers(0, len(big), n)]
            Rs = np.corrcoef(X, rowvar=False)
            Sm = np.cov(X, rowvar=False)
            _, lam = C.shrunk_covariance(Sm, target, X=X, rng=rng)
            lam_ss = C.schafer_strimmer_lambda(X, target)
            es.append(np.linalg.norm(Rs - true_corr))
            eh.append(np.linalg.norm(C.shrink_correlation(Rs, target, lam) - true_corr))
            eh_ss.append(np.linalg.norm(C.shrink_correlation(Rs, target, lam_ss) - true_corr))
            lams.append(lam)
            lams_ss.append(lam_ss)
            samples.append(Rs)
        samples = np.array(samples)
        variance = samples.var(axis=0)[off].sum()
        oracle = variance / (variance + np.sum((target - true_corr)[off]**2))
        ratios[n] = {"mean_error_ratio_shrunk_over_sample": float(np.mean(eh) / np.mean(es)),
                     "mean_lambda_bootstrap": float(np.mean(lams)), "oracle_lambda": float(oracle),
                     "mean_lambda_schafer_strimmer_analytic": float(np.mean(lams_ss)),
                     "error_ratio_with_analytic_lambda": float(np.mean(eh_ss) / np.mean(es))}
    details["by_N"] = ratios
    small_ok = all(ratios[n]["mean_error_ratio_shrunk_over_sample"] < 1 for n in (10, 20, 40, 63, 120, 300))
    monotone = all(ratios[a]["mean_lambda_bootstrap"] >= ratios[b]["mean_lambda_bootstrap"] - 0.02
                   for a, b in [(40, 63), (63, 120), (120, 300), (300, 900)])
    near_oracle = all(abs(ratios[n]["mean_lambda_bootstrap"] - ratios[n]["oracle_lambda"]) < 0.15 for n in (63, 120, 300, 900))
    details["pass_rule"] = ("default (bootstrap-variance) shrinkage reduces the correlation error for N <= 300, "
                            "lambda non-increasing with N for N >= 40 (0.02 slack), |lambda - oracle| < 0.15 for N >= 63")
    record("shrinkage_reduces_error_for_small_N", small_ok and monotone and near_oracle, t,
           lambda_decreases_with_N=monotone, lambda_near_oracle=near_oracle, **details)


def test_factor_model():
    t = time.perf_counter()
    rng = np.random.default_rng(31)
    p, k, n = 20, 2, 300
    L = rng.normal(size=(p, k)) * np.linspace(1, 0.3, p)[:, None]
    psi = rng.uniform(0.05, 0.3, p)
    Ct = L @ L.T + np.diag(psi)
    errs_fa, errs_s = [], []
    for _ in range(50):
        X = rng.multivariate_normal(np.zeros(p), Ct, size=n)
        Sm = np.cov(X, rowvar=False)
        fa = C.factor_analysis(Sm, k)
        errs_fa.append(np.linalg.norm(fa["cov"] - Ct) / np.linalg.norm(Ct))
        errs_s.append(np.linalg.norm(Sm - Ct) / np.linalg.norm(Ct))
    X = rng.multivariate_normal(np.zeros(p), Ct, size=15)        # N < p: rank-deficient
    fa_small = C.factor_analysis(np.cov(X, rowvar=False), k)
    pd = bool(np.all(np.linalg.eigvalsh(fa_small["cov"]) > 0))
    record("factor_model_recovers_low_rank_covariance", np.mean(errs_fa) < np.mean(errs_s) and pd, t,
           mean_relative_error_factor_model=float(np.mean(errs_fa)),
           mean_relative_error_sample=float(np.mean(errs_s)), positive_definite_when_N_lt_p=pd)


def test_sigma_d():
    t = time.perf_counter()
    setup = _mock_setup()
    rng = np.random.default_rng(41)
    mask, edges = setup["mask"], setup["edges"]
    sigma_true = 0.03
    est, chi = [], []
    n_inner = 300
    for i in range(20 if QUICK else 60):
        values = _draw(setup, 300, rng)
        mean = values.mean(axis=0)
        mean[mask] *= np.exp(rng.normal(0, sigma_true, mask.sum()))
        inner = rng.integers(0, 300, size=(n_inner, 300))
        cb = C.bootstrap_log_mean_covariance(values, mask, inner)["cov"]
        fit = F.objective_c("b12", mean, cb, mask, edges=edges, sigma_d="solve")
        est.append(fit["sigma_d"])
        chi.append(fit["chi2_per_dof"])
    est = np.array(est)
    chi = np.array(chi)
    ok = abs(np.sqrt(np.mean(est**2)) / sigma_true - 1) < 0.2 and np.all(np.abs(chi - 1) < 1e-3)
    record("sigma_d_solver_recovers_injected_discrepancy", ok, t, sigma_true=sigma_true,
           rms_sigma_d=float(np.sqrt(np.mean(est**2))), median_sigma_d=float(np.median(est)),
           max_abs_chi2_per_dof_minus_1=float(np.max(np.abs(chi - 1))),
           note="discrepancy injected on ln(mean) of 21 shells; 3 fitted parameters absorb part of it")


def test_nearest_correlation():
    t = time.perf_counter()
    rng = np.random.default_rng(17)
    out, ok = {}, True
    for trial in range(20):
        A = np.corrcoef(rng.normal(size=(25, 14)), rowvar=False)
        i, j = rng.choice(14, 2, replace=False)
        A[i, j] = A[j, i] = rng.choice([-0.99, 0.99])           # make it indefinite
        clip = C.nearest_correlation(A)
        high = C.nearest_correlation(A, method="higham")
        for B in (clip, high):
            ok &= bool(np.linalg.eigvalsh(B).min() > 0 and np.allclose(np.diag(B), 1, atol=1e-12)
                       and np.allclose(B, B.T, atol=1e-14))
        ok &= bool(np.linalg.norm(high - A) <= np.linalg.norm(clip - A) + 1e-9)
    psd = np.corrcoef(rng.normal(size=(200, 10)), rowvar=False)
    out["psd_input_unchanged_max"] = float(np.max(np.abs(C.nearest_correlation(psd, method="higham") - psd)))
    ok &= out["psd_input_unchanged_max"] < 1e-8
    record("nearest_correlation_clip_and_higham", ok, t, trials=20, **out,
           pass_rule="both PSD, unit diagonal, symmetric; Higham Frobenius change <= clip; PSD input unchanged")


def test_hartlap():
    t = time.perf_counter()
    ok = abs(C.hartlap(300, 21) - (300 - 21 - 2) / 299) < 1e-15 and C.hartlap(21, 26) < 0
    record("hartlap_factor_formula", ok, t, example_N300_p21=C.hartlap(300, 21), example_N21_p26=C.hartlap(21, 26),
           note="reported only; a global rescaling of C^-1 cannot move GLS point estimates at fixed sigma_d")


# ============================================================================ extension: families and objectives (d)-(g)
def test_family_jacobians_and_identities():
    t = time.perf_counter()
    rng = np.random.default_rng(13)
    edges = DATA["model_edges"]
    spline = FAM.b12_spline_family([0.1, 0.4, 1.2, 3.0])
    worst = {}
    for fam, centre in ((FAM.GNFW5, [8.0, 0.8, 1.2, 5.6, 0.4]), (FAM.B12ENV, [15.8, 0.76, 5.6, 3e-3, 2.0]),
                        (spline, [15.8, 0.76, 5.6, 0.05, -0.1, 0.2, 0.1]), (FAM.B12, [15.8, 0.76, 5.6]),
                        (FAM.B16, [2000.0, 1.25, 3.9])):
        errs = []
        for _ in range(20):
            p = np.array(centre, float)
            p = np.where(fam.log, p * np.exp(rng.normal(0, 0.2, fam.n)), p + rng.normal(0, 0.05, fam.n))
            theta = fam.to_theta(p)
            _, J = FAM.shell_model_jacobian(fam, p, edges)
            Jn = np.empty_like(J)
            for k in range(fam.n):
                up, down = theta.copy(), theta.copy()
                up[k] += 1e-6
                down[k] -= 1e-6
                Jn[:, k] = (np.log(FAM.shell_model(fam, fam.to_params(up), edges))
                            - np.log(FAM.shell_model(fam, fam.to_params(down), edges))) / 2e-6
            errs.append(np.max(np.abs(J - Jn) / np.maximum(np.max(np.abs(J), axis=0), 1e-12)))
        worst[fam.name] = float(np.max(errs))
    p = np.array([15.8, 0.76, 5.6])
    gnfw_identity = float(np.max(np.abs(FAM.shell_model("gnfw5", [0.5176 * 15.8, 0.76, 1.0, 5.9, 0.3], edges)
                                        / FAM.shell_model("b12", p, edges) - 1)))
    legacy_identity = bool(np.array_equal(FAM.shell_model("b12", p, edges), F.shell_model("b12", p, edges)))
    cut_edge = FAM.shell_model_truncated("b12", p, edges, 2.0)
    full = FAM.shell_model("b12", p, edges)
    edge_ok = bool(np.array_equal(cut_edge[edges[1:] <= 2], full[edges[1:] <= 2]) and np.all(cut_edge[edges[:-1] >= 2] == 0))
    cut = 2.4
    s = int(np.flatnonzero((edges[:-1] < cut) & (edges[1:] > cut))[0])
    xx = np.linspace(edges[s], cut, 400001)
    exact = np.trapezoid(FAM.B12.shape(xx, p) * 3 * xx**2, xx) / (edges[s + 1]**3 - edges[s]**3)
    split_err = float(abs(FAM.shell_model_truncated("b12", p, edges, cut)[s] / exact - 1))
    passed = max(worst.values()) < 1e-6 and gnfw_identity < 1e-12 and legacy_identity and edge_ok and split_err < 1e-8
    record("families_jacobians_identities_truncation", passed, t, max_relative_jacobian_error=worst,
           gnfw5_equals_b12_special_case=gnfw_identity, b12_family_equals_legacy_shell_model=legacy_identity,
           truncation_at_edge_exact=edge_ok, split_shell_relative_error=split_err)


def test_general_objectives():
    t = time.perf_counter()
    edges = DATA["model_edges"]
    rows = aio.bin_rows(DATA, 10)
    base = msk.resolution_masks(DATA, rows)["base"]
    values = aio.profile(DATA, "pressure_noagn", "all", rows)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    details, ok = {}, True
    for cut in (2.0, 4.0):
        mask = msk.fit_mask(base, edges, cut, mean)
        inner = S.bootstrap_rows(len(rows), 1000, 5)
        cb = C.bootstrap_log_mean_covariance(values, mask, inner)["cov"]
        legacy_a = F.objective_a("b12", mean, std, mask, edges=edges)
        general_a = F.objective_a_general("b12", mean, std, mask, edges=edges)
        sigma = np.maximum(std / mean, 0.05)
        as_d = F.objective_d("b12", mean, 1 / sigma**2, mask, edges=edges)
        gls = F.objective_c("b12", mean, cb, mask, edges=edges, sigma_d=0.0)
        as_e = F.objective_e("b12", mean, np.linalg.inv(cb), mask, edges=edges)
        legacy_c = F.objective_c("b12", mean, cb, mask, edges=edges)
        general_c = F.objective_c_general("b12", mean, cb, mask, edges=edges)
        w = np.ones(len(mean))
        K = np.diag(w[mask])
        tiny = F.objective_f("b12", mean, cb, K, mask, 1e-9, edges=edges)
        huge = F.objective_f("b12", mean, cb, K, mask, 1e9, edges=edges)
        unit = F.objective_d("b12", mean, w, mask, edges=edges)
        pinned = F.objective_g("b12", mean, mask, [lambda p: np.array([(np.log(p[0]) - np.log(20.0)) / 1e-4])],
                               sigma=sigma, edges=edges)
        rel = lambda a, b: float(np.max(np.abs(np.array(a["parameters"]) / np.array(b["parameters"]) - 1)))  # noqa: E731
        r = {"general_a_vs_legacy_a": rel(general_a, legacy_a), "d_with_1_over_sigma2_vs_legacy_a": rel(as_d, legacy_a),
             "e_with_Cinv_vs_legacy_gls": rel(as_e, gls), "general_c_vs_legacy_c": rel(general_c, legacy_c),
             "sigma_d_general_minus_legacy": abs(general_c["sigma_d"] - legacy_c["sigma_d"]),
             "f_lambda_1e-9_vs_gls": rel(tiny, gls), "f_lambda_1e9_vs_unit_weights": rel(huge, unit),
             "g_pinned_P0": pinned["parameters"][0]}
        details[f"{cut:g}R"] = r
        ok &= (r["general_a_vs_legacy_a"] < 1e-6 and r["d_with_1_over_sigma2_vs_legacy_a"] < 1e-6
               and r["e_with_Cinv_vs_legacy_gls"] < 1e-6 and r["general_c_vs_legacy_c"] < 1e-6
               and r["sigma_d_general_minus_legacy"] < 1e-7 and r["f_lambda_1e-9_vs_gls"] < 1e-4
               and r["f_lambda_1e9_vs_unit_weights"] < 1e-3 and abs(r["g_pinned_P0"] / 20 - 1) < 1e-3)
    # observable metric: a linear observable O = sum a_s F_s gives K = (a F)(a F)^T / O^2
    a = np.random.default_rng(2).uniform(0.5, 2.0, len(mean))
    Fm = mean
    K = F.observable_metric(a[None, :], [a @ Fm], [1.0], Fm)
    g = a * Fm
    details["observable_metric_identity"] = float(np.max(np.abs(K - np.outer(g, g) / (a @ Fm)**2)))
    ok &= details["observable_metric_identity"] < 1e-15
    record("general_objectives_d_e_f_g_consistency", ok, t, **details,
           pass_rule="general paths agree with legacy (a)/(c) to 1e-6, lambda limits of (f), pinned extra residual")


def test_new_family_recovery():
    t = time.perf_counter()
    edges = DATA["model_edges"]
    mask = (edges[:-1] >= 0.0896) & (edges[1:] <= 4.0 + 1e-12)
    out, ok = {}, True
    for fam, truth, start in ((FAM.GNFW5, [9.0, 0.7, 1.3, 5.2, 0.4], None),
                              (FAM.B12ENV, [15.8, 0.76, 5.66, 2e-3, 1.5], None),
                              (FAM.b12_spline_family([0.15, 0.5, 1.5, 3.5], prior_sigma=10.0),
                               [15.8, 0.76, 5.66, 0.05, -0.08, 0.1, 0.0], None)):
        mean = FAM.shell_model(fam, truth, edges)
        fit = F.fit_general(fam, mean, mask, sigma=np.full(len(mean), 0.05), edges=edges, start=start)
        dev = np.abs(fam.to_theta(fit["parameters"]) - fam.to_theta(truth))
        out[fam.name] = {"max_abs_theta_error": float(dev.max()), "accepted": fit["accepted"],
                         "rms_percent": fit["rms_percent"]}
        if fam.name == "b12spline":
            # B12 parameters and a smooth multiplicative correction trade off: the profile is
            # recovered, the decomposition is not unique (the weak prior picks smaller |c|).
            ok &= fit["rms_percent"] < 1e-3
        else:
            ok &= dev.max() < 1e-3 and fit["rms_percent"] < 1e-3
    record("new_families_recover_noise_free_truth", ok, t, **out,
           note="noise-free shell-averaged truths on 0.09-4 R200c; gnfw5/b12env: parameters recovered; "
                "b12spline (prior width 10): profile recovered, parameters degenerate with the correction")


def main():
    global DATA
    started = time.perf_counter()
    DATA = aio.load_radii234()
    for test in (test_frozen_sources, test_normalisation, test_masks_parity, test_objective_a_bit_exact,
                 test_weighted_stack_parity, test_bootstrap_rows, test_rebinning, test_jacobian, test_hartlap,
                 test_nearest_correlation,
                 test_factor_model, test_shrinkage, test_sigma_d, test_family_jacobians_and_identities,
                 test_general_objectives, test_new_family_recovery, test_permutation_null, test_gls_coverage):
        try:
            test()
        except Exception as error:  # a crash is a failure, recorded with its message
            import traceback
            RESULTS[test.__name__] = {"passed": False, "error": repr(error), "traceback": traceback.format_exc()}
            print(f"[FAIL] {test.__name__}: {error!r}", flush=True)
    summary = {"library_version": analysis.__version__, "quick_mode": QUICK,
               "all_passed": all(r["passed"] for r in RESULTS.values()),
               "passed": sum(r["passed"] for r in RESULTS.values()), "total": len(RESULTS),
               "wall_seconds": time.perf_counter() - started, "workers": WORKERS,
               "peak_RSS_MiB_parent": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "python": sys.version.split()[0], "numpy": np.__version__,
               "tests": RESULTS}
    (HERE / "test_results.json").write_text(json.dumps(plain(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("all_passed", "passed", "total", "wall_seconds")}))


if __name__ == "__main__":
    main()
