"""Validation of analysis.newdata / analysis.variants on the rehearsal fixture (or a real product set).

Usage:
  <flamingo python> -B analysis/tests/test_newdata.py                       # rehearsal fixture
  <flamingo python> -B analysis/tests/test_newdata.py --data <study root> \\
        --radii234 <radii234_contamination_L1_m9_0077_20261008/measurements.npz>   # real data, later
Writes analysis/tests/test_newdata_results.json (fixture) or test_newdata_results_<dirname>.json.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from analysis import io as aio  # noqa: E402
from analysis import newdata as ND  # noqa: E402
from analysis import variants as V  # noqa: E402

RESULTS: dict = {}
TOL = 1e-12


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
    print(f"[{'PASS' if passed else 'FAIL'}] {name} ({RESULTS[name]['seconds']} s)", flush=True)


def rel(a, b):
    """max |a - b| / max|b| per halo (shell axis), over finite entries."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ref = np.nanmax(np.abs(b), axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(a - b) / np.maximum(ref, 1e-300)
    return float(np.nanmax(r)) if np.isfinite(r).any() else 0.0


# ============================================================================ (i) + (ii)
def test_legacy_equivalence(d):
    t = time.perf_counter()
    legacy = d.legacy_dataset()
    new = V.profile(d, "pressure", "geometric", "kept", "all", "old").values
    old = aio.profile(legacy, "pressure", "all")
    i_dev = rel(new, old)
    new_noagn = V.profile(d, "pressure", "geometric", "removed", "all", "old").values
    old_noagn = aio.profile(legacy, "pressure_noagn", "all")
    ii_dev = rel(new_noagn, old_noagn)
    every = V.compare_with_legacy(d)
    record("i_all_gas_agn_kept_geometric_equals_legacy", i_dev < TOL, t, max_relative_deviation=i_dev,
           note="owner200 (17 owners x 3 phases, 67 shells) rebinned to the old 36 shells BEFORE normalising, vs "
                "measurements_legacy.npz with radii234 normalisation; float sums are reordered, so equality is to "
                "float precision (counts are exact)")
    record("ii_agn_removed_equals_legacy_y_noagn", ii_dev < TOL, t, max_relative_deviation=ii_dev)
    record("legacy_equivalence_all_fields_and_radii234_selections", max(every.values()) < TOL, t,
           max_relative_deviation=max(every.values()), per_case=every)
    # counts are exact
    o200 = d.array("owner200")
    qc = d.index("quantities", "count")
    counts = d.rebin(o200[:, :, :, qc].sum(axis=(1, 2)), d.subgrid_index("old")[1])
    exact = bool(np.array_equal(counts, d.array("legacy_sums")[:, :, 0].sum(axis=1)))
    record("owner_counts_exactly_equal_legacy_counts", exact, t)


# ============================================================================ (iii)
def test_block_consistency(d):
    t = time.perf_counter()
    o200 = d.array("owner200")
    ex = d.array("excision")
    part = d.array("partition")
    out = {}
    worst = 0.0
    for q in d.names["reduced"]:
        qi, ri = d.index("quantities", q), d.index("reduced", q)
        owner_total = o200[:, :, :, qi].sum(axis=(1, 2))
        for v, name in enumerate(d.names["variants"]):
            out[f"excision_{name}_{q}"] = rel(ex[:, v, :, :, :, ri].sum(axis=(1, 2, 3)), owner_total)
        out[f"partition_{q}"] = rel(part[..., ri, :].sum(axis=(1, 2, 3)), owner_total)
        # members and AGN axes against the owner/phase axes
        out[f"excision_member_central_{q}"] = rel(ex[:, 0, :, 0, :, ri].sum(axis=(1, 2)), o200[:, 0, :, qi].sum(axis=1))
        out[f"excision_member_overlapper_{q}"] = rel(ex[:, 0, :, 1, :, ri].sum(axis=(1, 2)), o200[:, 1:13, :, qi].sum(axis=(1, 2)))
        out[f"excision_member_rest_{q}"] = rel(ex[:, 0, :, 2, :, ri].sum(axis=(1, 2)), o200[:, 13:, :, qi].sum(axis=(1, 2)))
        out[f"excision_agn_{q}"] = rel(ex[:, 0, :, :, 1, ri].sum(axis=(1, 2)), o200[:, :, 0, qi].sum(axis=1))
        out[f"partition_agn_{q}"] = rel(part[:, :, :, 1, ri].sum(axis=(1, 2)), o200[:, :, 0, qi].sum(axis=1))
    for key, value in out.items():
        worst = max(worst, value)
    # auxiliary composition sums vs legacy (all components) on the old grid
    aux = d.array("aux200")
    legacy = d.array("legacy_sums")
    sum_names = list(d.mod.P.SUM_NAMES)
    idx = d.subgrid_index("old")[1]
    for k, name in enumerate(d.names["aux"]):
        out[f"aux_{name}_vs_legacy"] = rel(d.rebin(aux[:, :, k].sum(axis=1), idx), legacy[:, :, sum_names.index(name)].sum(axis=1))
        worst = max(worst, out[f"aux_{name}_vs_legacy"])
    record("iii_blocks_sum_consistently", worst < TOL, t, max_relative_deviation=worst, per_check=out)


# ============================================================================ (iv)
def test_excision_zero_spheres(d):
    t = time.perf_counter()
    lens = np.diff(d.tables["exc_ptr"]).reshape(d.n, 4)
    worst, cases = 0.0, 0
    retained_ok = True
    for v, name in enumerate(d.names["variants"]):
        rows = np.flatnonzero(lens[:, v] == 0)
        if not len(rows):
            continue
        for agn in ("kept", "removed"):
            base = V.profile(d, "pressure", "geometric", agn, "all", "x200", rows=rows).values
            for kappa in (0.5, 1.0, 2.0):
                p = V.profile(d, "pressure", "geometric", agn, "all", "x200",
                              excision={"variant": name, "kappa": kappa}, rows=rows)
                worst = max(worst, rel(p.values, base))
                retained_ok &= bool(np.all(p.retained_fraction == 1.0))
                cases += len(rows)
                for field in ("electrons", "gas"):
                    q = V.profile(d, field, "geometric", agn, "all", "old", excision={"variant": name, "kappa": kappa}, rows=rows)
                    worst = max(worst, rel(q.values, V.profile(d, field, "geometric", agn, "all", "old", rows=rows).values))
    # monotonic: larger kappa never retains more volume (haloes with spheres)
    mono = True
    for v, name in enumerate(d.names["variants"]):
        f = [V.excision_fraction(d, name, k) for k in (0.5, 1.0, 2.0)]
        mono &= bool(np.all(f[0] <= f[1] + 1e-6) and np.all(f[1] <= f[2] + 1e-6))
    # masked shells: excised profiles are NaN exactly where < 50% of the volume is retained
    p = V.profile(d, "pressure", "geometric", "kept", "all", "x200", excision={"variant": "all", "kappa": 2.0})
    nan_rule = bool(np.array_equal(np.isnan(p.values), p.retained_fraction < 0.5))
    common = V.excision_common_set(d, ("all", "placebo_iso"), (0.5, 1.0, 2.0))
    p2 = V.profile(d, "pressure", "geometric", "kept", "all", "x200", excision={"variant": "placebo_iso", "kappa": 1.0})
    subset = bool(np.all(~common | (np.isfinite(p.values) & np.isfinite(p2.values))))
    record("iv_excision_with_zero_spheres_returns_unexcised_profile", worst < TOL and retained_ok and mono and nan_rule and subset, t,
           max_relative_deviation=worst, halo_variant_kappa_cases=cases, retained_fraction_exactly_one=retained_ok,
           excised_fraction_monotonic_in_kappa=mono, nan_exactly_where_retained_below_half=nan_rule,
           common_halo_set_is_subset_of_finite=subset)


# ============================================================================ (v)
def _cored_model(m, x):
    """Smooth synthetic overlapper profile (finite core, steep outer fall-off), mild mass dependence."""
    x = np.asarray(x, float)
    return (m / 1e14)**0.1 * 3.0 / (1 + (x / 0.3)**2)**1.5


def test_template_recovery(d):
    t = time.perf_counter()
    GEO = d.mod.GEO
    ov = d.overlapper_table()
    counts = np.bincount(ov["halo"][ov["mu"] >= 0.1], minlength=d.n)
    rows = np.flatnonzero(counts >= 2)[:4]
    if not len(rows):
        rows = np.flatnonzero(counts >= 1)[:4]
    base, idx = d.subgrid_index("x200")
    edges = d.grid(base)[idx]
    scales = V._scales(d, base, "auto", rows)
    spec = {"model": _cored_model, "mu_min": 0.1, "reach": "any", "kind": "all", "truncation": 1.0, "n_seg": 800}
    tmpl = V.template_profiles(d, "pressure", base, edges, scales, spec, "kept", rows)
    # independent injection: Monte Carlo shell averages of the same physical profiles
    rng_truth = np.random.default_rng(3)
    truth = np.exp(rng_truth.normal(0, 0.3, (len(rows), len(edges) - 1))) * 1e-2
    injected = np.zeros_like(truth)
    r_norm = d.radius(base, rows)
    norm_h = scales["pressure"]
    for i, h in enumerate(rows):
        for k in range(ov["ptr"][h], ov["ptr"][h + 1]):
            if ov["mu"][k] < 0.1:
                continue
            m_j, r_j, dist = float(ov["mass_msun"][k]), float(ov["radius_mpc"][k]), float(ov["distance_mpc"][k])
            amp = float(V._norm_of(d, "pressure", m_j, r_j)) / float(norm_h[i])
            scale = r_norm[i] / r_j

            def F(s, a=amp, sc=scale, mj=m_j):
                return a * _cored_model(mj, s * sc)
            injected[i] += GEO.mc_offset_shell_average(F, edges, dist / r_norm[i], r_j / r_norm[i], n=200000, seed=11 + k)
    recovered = truth + injected - tmpl
    err_mc = float(np.max(np.abs(recovered - truth)) / injected.max())
    rel_mc_bulk = float(np.median(np.abs(recovered - truth)[injected > 0.05 * injected.max()]
                                  / injected[injected > 0.05 * injected.max()]))
    # exact injection (same quadrature): subtraction is an identity to rounding
    exact = truth + tmpl - V.template_profiles(d, "pressure", base, edges, scales, spec, "kept", rows)
    err_exact = float(np.max(np.abs(exact - truth) / truth))
    # profile() integration: profile(template=spec) = profile() - template_profiles()
    p_t = V.profile(d, "pressure", "geometric", "kept", "all", "x200", template=spec, rows=rows)
    p_0 = V.profile(d, "pressure", "geometric", "kept", "all", "x200", rows=rows)
    integ = float(np.max(np.abs(p_t.values - (p_0.values - tmpl))))
    # measured-bound amplitude: the template's integral equals the target bound gas
    k0 = int(np.flatnonzero(ov["mu"] >= 0.1)[0])
    h0 = int(ov["halo"][k0])
    sc0 = V._scales(d, base, "auto", np.array([h0]))
    m_j, r_j = float(ov["mass_msun"][k0]), float(ov["radius_mpc"][k0])
    amp0 = float(V._norm_of(d, "pressure", m_j, r_j)) / float(sc0["pressure"][0])
    rn = float(d.radius(base, np.array([h0]))[0])
    target = 2.5e-7
    f = V.template_amplitude_scale(d, "pressure", _cored_model, m_j, amp0, rn / r_j, r_j / rn, rn, float(sc0["pressure"][0]), target)
    achieved = V.model_integral(d, "pressure", _cored_model, m_j, amp0 * f, rn / r_j, r_j / rn, rn, float(sc0["pressure"][0]))
    amp_err = abs(achieved / target - 1)
    passed = err_mc < 0.01 and err_exact < 1e-12 and integ == 0.0 and amp_err < 1e-12
    record("v_template_subtraction_recovers_injected_truth", passed, t, haloes=rows, overlappers=int(np.sum((ov["mu"] >= 0.1) & np.isin(ov["halo"], rows))),
           max_error_over_max_injected_monte_carlo=err_mc, median_relative_error_where_injected_gt_5pct_of_max=rel_mc_bulk,
           max_relative_error_exact_injection=err_exact,
           profile_template_integration_max_abs=integ, measured_amplitude_relative_error=amp_err,
           note="Monte Carlo (200,000 points per shell, geometry.mc_offset_shell_average) is an independent "
                "injection; tolerance: max |recovered - truth| < 1% of the largest injected shell value")


# ============================================================================ (vi)
def test_weighted_stack(d):
    t = time.perf_counter()
    v = V.profile(d, "pressure", "geometric", "removed", "all", "old").values
    equal = V.stack(v)
    const = V.stack(v, weights=np.full(d.n, 3.7))
    plain_mean = np.nanmean(v, axis=0)
    dev = float(np.nanmax(np.abs(const["mean"] - equal["mean"]) / np.abs(equal["mean"])))
    dev_np = float(np.nanmax(np.abs(equal["mean"] - plain_mean) / np.abs(plain_mean)))
    neff_ok = bool(np.allclose(const["n_eff"], equal["n"], rtol=1e-12))
    w = d.inclusion_weights()
    weighted = V.stack(v, weights=w)
    manual = (w[:, None] * v).sum(axis=0) / w.sum()
    dev_w = float(np.nanmax(np.abs(weighted["mean"] - manual) / np.abs(manual)))
    # excision-style NaNs are excluded per shell
    vn = v.copy()
    vn[::3, 5] = np.nan
    s = V.stack(vn)
    nan_ok = bool(np.isclose(s["mean"][5], np.nanmean(vn[:, 5]), rtol=1e-12) and s["n"][5] == np.isfinite(vn[:, 5]).sum())
    # 1e-12: summation roundoff over ~4,600 real haloes reaches ~1e-13 (8.9e-14 measured); 1e-14 held only on the fixture
    record("vi_inclusion_weighted_stack_reduces_to_equal_weights", dev < 1e-12 and dev_np < 1e-12 and neff_ok and dev_w < 1e-12 and nan_ok, t,
           constant_weights_vs_equal=dev, equal_vs_numpy_mean=dev_np, n_eff_equals_n_for_equal_weights=neff_ok,
           inclusion_weighted_vs_manual=dev_w, nan_shells_excluded=nan_ok,
           inclusion_weight_range=[float(w.min()), float(w.max())])


# ============================================================================ census classes
def test_census_classes(d):
    t = time.perf_counter()
    main = np.asarray(d.halos["census_class_main"])
    out = {"main_definition_equal": bool(np.array_equal(d.census_class(0.1), main))}
    for mu in (0.03, 0.1, 0.3):
        for kind in ("all", "own_satellites", "neighbours"):
            out[f"mu{mu}_{kind}_ov_lists_equal_max_mu"] = bool(np.array_equal(d.census_class(mu, kind), d.census_class_from_max_mu(mu, kind)))
        out[f"mu{mu}_counts"] = np.bincount(d.census_class(mu), minlength=3)
    nested = bool(np.all((d.census_class(0.3) > 0) <= (d.census_class(0.1) > 0)) and np.all((d.census_class(0.1) > 0) <= (d.census_class(0.03) > 0)))
    out["contamination_nested_in_threshold"] = nested
    record("census_classes_main_and_derived", all(v for k, v in out.items() if not k.endswith("counts")), t, **out)


# ============================================================================ radii234 reference
def test_radii234_reference(d, path):
    t = time.perf_counter()
    path = Path(path)
    if not path.exists():
        record("radii234_reference_reproduction", False, t, error=f"missing {path}")
        return
    ref = V.radii234_reference_dataset(d, path)
    n_ref = len(ref["sums"])
    rows = np.arange(n_ref)
    dev = V.compare_with_legacy(d, legacy=ref, rows=rows)
    legacy_equal = bool(np.array_equal(d.array("legacy_sums")[:n_ref], ref["sums"]))
    stored = {}
    with np.load(path) as z:
        plan_path = path.parent / "metadata_plan.json"
        if "pressure_components" in z.files and plan_path.exists():
            fb_old = float(json.loads(plan_path.read_text())["baryon_fraction"])
            ratio = d.baryon_fraction / fb_old             # normalised profiles scale as 1/f_b
            for key, field, agn in (("pressure_components", "pressure", "kept"), ("electron_components", "electrons", "kept"),
                                    ("gas_components", "gas", "kept")):
                new = V.profile(d, field, "geometric", agn, "all", "old", rows=rows).values * ratio
                stored[key] = rel(new, z[key].sum(axis=1))
            new = V.profile(d, "pressure", "geometric", "removed", "all", "old", rows=rows).values * ratio
            stored["pressure_noagn_ratio"] = rel(new, z["pressure_noagn_ratio"])
            stored["baryon_fraction_new_over_radii234"] = ratio
    stored_ok = all(v < 1e-12 for k, v in stored.items() if k != "baryon_fraction_new_over_radii234")
    record("radii234_reference_reproduction", max(dev.values()) < TOL and legacy_equal and stored_ok, t, reference=str(path),
           rows=n_ref, max_relative_deviation=max(dev.values()), legacy_sums_array_equal_reference=legacy_equal,
           stored_normalised_arrays=stored or "not in the reference file (rehearsal fixture: sums only)",
           per_case=dev,
           note="rows 0..n_old-1: profiles from the new owner blocks (old grid) vs the radii234 sums with the same "
                "normalisation; on real data pass --radii234 radii234_contamination_L1_m9_0077_20261008/measurements.npz")


# ============================================================================ grids
def test_grids(d):
    t = time.perf_counter()
    out = {}
    fine = V.profile(d, "pressure", "geometric", "kept", "all", "x200")
    for grid in ("old", "lee22"):
        coarse = V.profile(d, "pressure", "geometric", "kept", "all", grid)
        idx = d.subgrid_index(grid)[1]
        vol = np.diff(d.grid("x200")**3)
        expect = d.rebin(fine.values * vol, idx) / d.rebin(np.broadcast_to(vol, fine.values.shape), idx)
        out[f"x200_to_{grid}_volume_weighted_identity"] = rel(coarse.values, expect)
    custom = d.grid("x200")[[0, 9, 20, 31, 45, 52, 60, 67]]
    c = V.profile(d, "gas", "geometric", "kept", "all", ("x200", custom))
    idx = d.subgrid_index(("x200", custom))[1]
    g_fine = V.profile(d, "gas", "geometric", "kept", "all", "x200")
    vol = np.diff(d.grid("x200")**3)
    out["custom_subgrid_identity"] = rel(c.values, d.rebin(g_fine.values * vol, idx) / d.rebin(np.broadcast_to(vol, g_fine.values.shape), idx))
    f5 = V.profile(d, "pressure", "geometric", "kept", "all", "x500")
    b24 = V.profile(d, "pressure", "geometric", "kept", "all", "b24")
    idx5 = d.subgrid_index("b24")[1]
    vol5 = np.diff(d.grid("x500")**3)
    out["x500_to_b24_identity"] = rel(b24.values, d.rebin(f5.values * vol5, idx5) / d.rebin(np.broadcast_to(vol5, f5.values.shape), idx5))
    out["grid_sizes"] = {g: len(d.edges(g)) - 1 for g in ("x200", "old", "lee22", "x500", "b24")}
    try:
        d.subgrid_index(("x200", [0.0, 0.333, 4.0]))
        out["non_subset_rejected"] = False
    except ValueError:
        out["non_subset_rejected"] = True
    worst = max(v for k, v in out.items() if isinstance(v, float))
    record("grids_rebin_linear_sums_before_normalising", worst < TOL and out["non_subset_rejected"]
           and out["grid_sizes"] == {"x200": 67, "old": 36, "lee22": 27, "x500": 40, "b24": 30}, t, **out)


# ============================================================================ estimators on a synthetic block
def test_estimators_synthetic(d):
    """Volume-filling synthetic owner blocks with uniform particles: every pressure estimator = P0, every
    temperature = T0, clumping = 1, geometric densities = their known values."""
    t = time.perf_counter()
    P = d.mod.P
    MO = d.mod.MO
    names = list(d.names["quantities"])
    out_ok, details = True, {}
    for base in ("x200", "x500"):
        edges = d.grid(base)
        R = d.radius(base)
        V_shell = 4 * np.pi / 3 * R[:, None]**3 * np.diff(edges**3)       # Mpc^3
        n_part = 50.0
        Vp = V_shell / n_part                                             # particle volume
        rho = 1e13                                                        # Msun / Mpc^3
        m = rho * Vp
        T0 = 3e7
        n0 = 1e-3 * 1e6                                                   # m^-3
        P0 = n0 * P.Boltzmann * T0                                        # Pa
        y = P0 * Vp / MO.PE_FACTOR                                        # so that y / V PE_FACTOR = P0
        ne = n0 * P.MPC_M**3 * Vp                                         # n_e V (count)
        L0 = 1e40
        q = {"count": n_part + 0 * Vp, "mass": n_part * m, "volume": n_part * Vp, "y": n_part * y,
             "electrons": n_part * ne, "electron_kT": n_part * ne * P.Boltzmann * T0,
             "gas_pressure_volume": n_part * P0 * Vp, "mass_pe": n_part * m * P0, "volume_pe2": n_part * Vp * P0**2,
             "mass_t": n_part * m * T0, "volume_t": n_part * Vp * T0, "mass_rho": n_part * m * rho,
             "lx_rosat": n_part * L0 + 0 * Vp, "lx_pe": n_part * L0 * P0 + 0 * Vp, "lx_t": n_part * L0 * T0 + 0 * Vp,
             "y2": n_part * y * y}
        block = np.zeros((d.n, 17, 3, len(names), len(edges) - 1))
        for k, name in enumerate(names):
            block[:, 0, 1, k] = q[name]                                   # central-bound, X-ray-selected phase
        key = "owner200" if base == "x200" else "owner500"
        saved = d._cache.get(key)
        d._cache[key] = block
        try:
            norm = d.p200() if base == "x200" else d.p500("B24")
            for est in V.ESTIMATORS["pressure"]:
                v = V.profile(d, "pressure", est, "removed", "all", base).values
                details[f"{base}_pressure_{est}"] = rel(v * norm[:, None], np.full_like(v, P0))
            n_e_norm = (d.rho_c("x200") * d.baryon_fraction / P.MU_XGP_MSUN if base == "x200"
                        else d.ne500("B24") * P.MPC_M**3)
            rho_norm = (d.rho_c("x200") * d.baryon_fraction if base == "x200" else 500 * ND.B24["f_B"] * d.rho_c("x500"))
            for est in V.ESTIMATORS["electrons"]:
                v = V.profile(d, "electrons", est, "removed", "all", base).values
                details[f"{base}_electrons_{est}"] = rel(v * n_e_norm[:, None], np.full_like(v, n0 * P.MPC_M**3))
            for est in ("geometric", "Q_V"):
                v = V.profile(d, "gas", est, "removed", "all", base).values
                details[f"{base}_gas_{est}"] = rel(v * rho_norm[:, None], np.full_like(v, rho))
            v = V.profile(d, "gas", "Q_M", "removed", "all", base).values
            details[f"{base}_gas_Q_M"] = rel(v * rho_norm[:, None], np.full_like(v, rho))
            for kind, value in (("T_M", T0), ("T_V", T0), ("T_X", T0), ("clumping_density", 1.0), ("clumping_pressure", 1.0)):
                details[f"{base}_{kind}"] = rel(V.diagnostic(d, kind, "removed", "all", base), np.full(V_shell.shape, value))
            # AGN: moving everything to the recent-AGN phase empties 'removed' and keeps 'kept'
            block_agn = np.zeros_like(block)
            block_agn[:, 0, 0] = block[:, 0, 1]
            d._cache[key] = block_agn
            kept = V.profile(d, "pressure", "geometric", "kept", "all", base).values
            removed = V.profile(d, "pressure", "geometric", "removed", "all", base).values
            details[f"{base}_agn_kept"] = rel(kept * norm[:, None], np.full_like(kept, P0))
            details[f"{base}_agn_removed_is_zero"] = float(np.max(np.abs(removed)))
        finally:
            if saved is None:
                d._cache.pop(key, None)
            else:
                d._cache[key] = saved
    worst = max(details.values())
    out_ok = worst < 1e-12
    record("estimators_on_synthetic_volume_filling_block", out_ok, t, max_relative_deviation=worst, per_case=details,
           note="uniform particles filling every shell: geometric = Q_V = Q_V_y = Q_M = Q_X = n_e k T_X = P0; "
                "temperatures = T0; clumping = 1; B24 normalisation on X500, P200 on X200")


# ============================================================================ normalisations
def test_normalisations(d):
    t = time.perf_counter()
    P = d.mod.P
    out = {}
    ratio = d.p500("B24") / d.p500("self_similar_500")
    expect = ND.B24["mu"] / ND.B24["mu_e"] * ND.B24["f_B"] / d.baryon_fraction
    out["B24_over_self_similar_minus_expected"] = float(np.max(np.abs(ratio / expect - 1)))
    h70 = d.hubble() / 0.7
    m_ref = 3e14 / h70                                                   # Msun: 3e14 h70^-1 Msun
    a10 = ND.A10_P500_KEV_CM3 * d.e_of_z()**(8 / 3) * (m_ref * h70 / 3e14)**(2 / 3) * h70**2 * ND.KEV_CM3_IN_PA
    out["A10_pivot_value_Pa"] = a10
    out["A10_pivot_expected_Pa"] = 1.65e-3 * h70**2 * 1.602176634e-10
    out["A10_pivot_relative"] = abs(a10 / out["A10_pivot_expected_Pa"] - 1)
    rho = d.rho_c("x500") * P.MSUN_KG / P.MPC_M**3
    out["ne500_formula_relative"] = float(np.max(np.abs(d.ne500() / (500 * 0.159 * rho / (1.14 * P.proton_mass)) - 1)))
    out["p200_equals_pilot_physics"] = bool(np.array_equal(d.p200(), P.p200(d.halos["m200c_msun"], d.halos["r_200_crit_mpc"], d.baryon_fraction)))
    passed = (out["B24_over_self_similar_minus_expected"] < 1e-12 and out["A10_pivot_relative"] < 1e-12
              and out["ne500_formula_relative"] < 1e-12 and out["p200_equals_pilot_physics"])
    record("normalisations_P200_B24_A10", passed, t, **out)


# ============================================================================ selections
def test_selections(d):
    t = time.perf_counter()
    o200 = d.array("owner200")
    qy = d.index("quantities", "y")
    total = o200[:, :, 1:, qy].sum(axis=2)                               # [n, owner, shell], AGN removed
    out = {}
    s = V.selection("remove_overlappers", mu_min=0.03, reach="any", flavour="remove_all")
    removed = V.owner_sum(d, "x200", "y", V.selection("all"), "removed") - V.owner_sum(d, "x200", "y", s, "removed")
    out["remove_all_mu0p03_equals_owners_1_12"] = rel(removed, total[:, 1:13].sum(axis=1))
    s = V.selection("remove_overlappers", mu_min=0.1, reach="any", flavour="keep_own_satellites")
    removed = V.owner_sum(d, "x200", "y", V.selection("all"), "removed") - V.owner_sum(d, "x200", "y", s, "removed")
    out["keep_own_satellites_mu0p1_equals_neighbour_codes"] = rel(removed, total[:, [6, 8, 10, 12]].sum(axis=1))
    s = V.selection("remove_overlappers", mu_min=0.3, reach="reach1R", flavour="remove_all")
    removed = V.owner_sum(d, "x200", "y", V.selection("all"), "removed") - V.owner_sum(d, "x200", "y", s, "removed")
    out["remove_all_mu0p3_reach1R_equals_codes_9_10"] = rel(removed, total[:, [9, 10]].sum(axis=1))
    # partition ownership never exceeds all gas; A2-i masks only remove painted neighbour codes and owner 14
    po = V.profile(d, "gas", "geometric", "kept", "partition_owned", "x200").values
    al = V.profile(d, "gas", "geometric", "kept", "all", "x200").values
    out["partition_owned_le_all"] = bool(np.all(po <= al * (1 + 1e-12)))
    masks = V.owner_masks(d, V.selection("remove_bound_to_other_painted"))
    out["a2i_keeps_central_own_satellites_unbound"] = bool(masks[:, [0, 1, 3, 5, 7, 9, 11, 13, 15, 16]].all())
    out["a2i_removes_owner_14"] = bool(not masks[:, 14].any())
    out["a2i_ambiguity"] = V.a2i_ambiguity(d)
    worst = max(v for k, v in out.items() if isinstance(v, float))
    record("gas_selections_owner_algebra", worst < TOL and out["partition_owned_le_all"]
           and out["a2i_keeps_central_own_satellites_unbound"] and out["a2i_removes_owner_14"], t, **out)


def test_resolution_masks(d):
    t = time.perf_counter()
    rows = np.arange(d.n)
    out = {}
    m = V.resolution(d, rows, "old")
    # particle counts used by the mask equal the legacy counts exactly
    o200 = d.array("owner200")
    qc = d.index("quantities", "count")
    idx = d.subgrid_index("old")[1]
    counts = d.rebin(o200[:, :, :, qc].sum(axis=(1, 2)), idx)
    out["counts_equal_legacy"] = bool(np.array_equal(counts, d.array("legacy_sums")[:, :, 0].sum(axis=1)))
    # upper medians of rebinned histograms equal brute-force medians of the expanded bin samples
    hs = d.array("hs200")
    edges_h = d.array("hs_edges")
    rng = np.random.default_rng(4)
    ok = True
    for _ in range(40):
        h = int(rng.integers(0, d.n))
        s = int(rng.integers(0, len(idx) - 1))
        hist = hs[h, idx[s]:idx[s + 1]].sum(axis=0).astype(np.int64)
        n = int(hist.sum())
        if n == 0:
            continue
        upper_edges = np.repeat(edges_h[1:], hist)           # each particle at its bin's upper edge
        expected = np.sort(upper_edges)[n // 2]
        from analysis import io as aio
        got = aio.upper_median_from_hist(hist[None, None, :], edges_h)[0, 0]
        ok &= bool(got == expected)
    out["rebinned_histogram_upper_median_exact"] = ok
    out["strict_subset_of_geometric"] = bool(np.all(m["strict"] <= m["geometric"].all(axis=0)))
    out["supported_flag"] = m["supported"]
    record("resolution_masks_on_new_grids", out["counts_equal_legacy"] and ok and out["strict_subset_of_geometric"], t, **out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ND.FIXTURE_DIR))
    parser.add_argument("--radii234", default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    d = ND.load(args.data)
    fixture = Path(args.data).resolve() == ND.FIXTURE_DIR.resolve()
    reference = args.radii234 or (str(ND.FIXTURE_DIR / "radii234_reference_measurements.npz") if fixture else None)
    for test in (test_legacy_equivalence, test_block_consistency, test_excision_zero_spheres, test_template_recovery,
                 test_weighted_stack, test_census_classes, test_grids, test_estimators_synthetic, test_normalisations,
                 test_selections, test_resolution_masks):
        try:
            test(d)
        except Exception as error:
            import traceback
            RESULTS[test.__name__] = {"passed": False, "error": repr(error), "traceback": traceback.format_exc()}
            print(f"[FAIL] {test.__name__}: {error!r}", flush=True)
    if reference is not None:
        try:
            test_radii234_reference(d, reference)
        except Exception as error:
            import traceback
            RESULTS["radii234_reference_reproduction"] = {"passed": False, "error": repr(error), "traceback": traceback.format_exc()}
    summary = {"data": str(Path(args.data).resolve()), "fixture": fixture, "haloes": d.n, "old_rows": d.n_old,
               "all_passed": all(r["passed"] for r in RESULTS.values()), "passed": sum(r["passed"] for r in RESULTS.values()),
               "total": len(RESULTS), "wall_seconds": time.perf_counter() - started, "tests": RESULTS}
    name = "test_newdata_results.json" if fixture else f"test_newdata_results_{Path(args.data).resolve().name}.json"
    (HERE / name).write_text(json.dumps(plain(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("all_passed", "passed", "total", "wall_seconds")}))


if __name__ == "__main__":
    main()
