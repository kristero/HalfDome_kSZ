"""Tests of the stacking cache and stackcore (run: python -B core/test_core.py; also pytest-compatible).

Writes core/test_core_results.json (per-test status, max deviations, timings).  Exit code 1 if any test fails.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stackcore as sc  # noqa: E402

STUDY = sc.STUDY
BRES = STUDY / "B_covariance" / "results"
FRES = STUDY / "F_observables" / "results"
RTOL = 1e-10
N = 4588
RESULTS: dict = {}


def _maxrel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    fin = np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        return 0.0
    den = np.maximum(np.abs(b[fin]), 1e-300)
    return float(np.max(np.abs(a[fin] - b[fin]) / den))


def _same(a, b, rtol=RTOL):
    """allclose with identical NaN pattern; returns (ok, max relative deviation, nan-pattern equal)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    nan_eq = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    ok = a.shape == b.shape and nan_eq and bool(np.allclose(a, b, rtol=rtol, atol=0, equal_nan=True))
    return ok, _maxrel(a, b), nan_eq


# ============================================================================ 1. cache integrity
def test_cache_files():
    idx = sc.load_index()["variants"]
    n_checked, problems = 0, []
    nan_resolved = {}
    ratio_extra = {}
    resolved = {g: sc.resolution_detail(np.arange(N), g)["resolved"] for g in ("x200", "fit", "lee22", "old", "x500", "b24")}
    for key, v in idx.items():
        path = sc.CACHE / v["file"]
        S = len(sc.edges(v["grid"])) - 1
        a = np.load(path, mmap_mode="r")
        if a.shape != (N, S) or a.dtype != np.float64 or not a.flags["C_CONTIGUOUS"]:
            problems.append(f"{key}: shape/dtype {a.shape} {a.dtype}")
        if path.stat().st_size != a.offset + 8 * N * S:   # uncompressed .npy: header + raw float64 data
            problems.append(f"{key}: unexpected file size {path.stat().st_size}")
        arr = np.asarray(a)
        if np.isinf(arr).any():
            problems.append(f"{key}: inf")
        nan = np.isnan(arr)
        if v["estimator"] == "geometric" and not v["selection"].startswith("exc"):
            if nan.any():
                problems.append(f"{key}: {int(nan.sum())} NaN in a geometric variant")
        elif v["selection"].startswith("exc"):
            ret = np.asarray(sc.retained(v["selection"], v["grid"]))
            if not np.array_equal(nan, ret < 0.5):
                problems.append(f"{key}: NaN pattern != retained < 0.5")
        else:
            # ratio variants: NaN exactly where the denominator vanishes (no gas of the selection / phase in the
            # halo-shell), read from the matching geometric gas profile
            exp, exact = _expected_nan(v)
            k = int((nan & resolved[v["grid"]]).sum())
            nan_resolved[key] = k
            if exact and not np.array_equal(nan, exp):
                problems.append(f"{key}: NaN pattern != zero denominator ({int((nan ^ exp).sum())} differ)")
            if not exact:
                extra = int((nan & ~exp).sum())
                ratio_extra[key] = extra
                if (exp & ~nan).any():
                    problems.append(f"{key}: finite where the X-ray phase is empty")
        n_checked += 1
    for g in ("x200", "fit", "lee22", "old", "x500", "b24"):
        S = len(sc.edges(g)) - 1
        for arr, name in ((sc.counts(g), "counts"), (sc.hsml_x(g), "hsml")):
            if arr.shape != (N, S):
                problems.append(f"{name}_{g} shape {arr.shape}")
        c, h = np.asarray(sc.counts(g)), np.asarray(sc.hsml_x(g))
        if not np.array_equal(np.isnan(h), c == 0):
            problems.append(f"hsml_{g}: NaN pattern != empty shells")
        if np.any(c < 0) or not np.allclose(c, np.round(c)):
            problems.append(f"counts_{g}: not non-negative integers")
    tab = sc.halo_table()
    for k, v in tab.items():
        if v.shape[:1] != (N,) and k not in ("mass_edges", "block_fold"):
            problems.append(f"halo_table {k} shape {v.shape}")
        if np.issubdtype(v.dtype, np.floating) and k not in ("concentration_200c", "dm_concentration_200c") \
                and not np.all(np.isfinite(v)):
            problems.append(f"halo_table {k} not finite")
    assert not problems, problems[:20]
    worst = max(nan_resolved.items(), key=lambda kv: kv[1]) if nan_resolved else None
    return {"variants": n_checked, "ratio_variants_checked": len(nan_resolved),
            "max_nan_in_resolved_halo_shells": worst,
            "xray_estimators_nan_with_xray_gas_present (lx = 0)": ratio_extra}


def _gas_key(agn, sel, phase, grid):
    return f"G|geometric|{agn}|{sel}|{phase}|{grid}|{'B24' if grid in ('x500', 'b24') else 'rhoc_fb'}"


def _expected_nan(v):
    """(expected NaN mask, exact?) of a ratio variant: zero denominator gas in the halo-shell."""
    g, agn, sel, phase = v["grid"], v["agn"], v["selection"], v["phase"]
    if v["field"] == "E_fullion":
        return np.zeros((N, len(sc.edges(g)) - 1), bool), True
    xray_based = v["estimator"] in ("Q_X", "nT_X") or v["field"] == "T_X"
    if xray_based:
        gas = np.asarray(sc.per_halo(_gas_key("removed", "all", "xray", g)))
        return gas == 0, False
    gas = np.asarray(sc.per_halo(_gas_key(agn, sel, phase, g)))
    exp = gas == 0
    if v["field"] == "C_P":            # denominator sum(V Pe)^2 = (sum y)^2: gas with zero y (cold / star-forming)
        Pg = np.asarray(sc.per_halo(f"P|geometric|{agn}|{sel}|{phase}|{g}|{'B24' if g in ('x500', 'b24') else 'P200'}"))
        exp = Pg == 0
    if v["field"] == "K":
        E = np.asarray(sc.per_halo(f"E|geometric|{agn}|{sel}|{phase}|{g}|{'B24' if g in ('x500', 'b24') else 'rhoc_fb'}"))
        exp = exp | (E <= 0)
    if v["field"] == "T_e":
        E = np.asarray(sc.per_halo(f"E|geometric|{agn}|{sel}|{phase}|{g}|{'B24' if g in ('x500', 'b24') else 'rhoc_fb'}"))
        exp = exp | (E <= 0)
    return exp, True


# ============================================================================ 2. reproduce Test B per-halo arrays
B_MAP_FIT = {
    "P_geo": "P|geometric|removed|all|all|fit|P200", "P_geo_agnkept": "P|geometric|kept|all|all|fit|P200",
    "E_geo": "E|geometric|removed|all|all|fit|rhoc_fb", "E_geo_agnkept": "E|geometric|kept|all|all|fit|rhoc_fb",
    "G_geo": "G|geometric|removed|all|all|fit|rhoc_fb",
    "P_QV": "P|Q_V|removed|all|all|fit|P200", "P_QVy": "P|Q_V_y|removed|all|all|fit|P200",
    "P_QM": "P|Q_M|removed|all|all|fit|P200", "P_QX": "P|Q_X|removed|all|all|fit|P200",
    "P_nTX": "P|nT_X|removed|all|all|fit|P200", "E_QV": "E|Q_V|removed|all|all|fit|rhoc_fb",
    "G_QV": "G|Q_V|removed|all|all|fit|rhoc_fb", "G_QM": "G|Q_M|removed|all|all|fit|rhoc_fb",
    "C_rho": "C_rho|weighted|removed|all|all|fit|none", "C_P": "C_P|weighted|removed|all|all|fit|none",
    "P_bound_all": "P|geometric|removed|rm_ov_mu0.1|all|fit|P200",
    "E_bound_all": "E|geometric|removed|rm_ov_mu0.1|all|fit|rhoc_fb",
    "P_bound_keep": "P|geometric|removed|rm_ov_mu0.1_keepsat|all|fit|P200",
    "E_bound_keep": "E|geometric|removed|rm_ov_mu0.1_keepsat|all|fit|rhoc_fb",
    "P_exc_all": "P|geometric|removed|exc_all_k1|all|fit|P200", "E_exc_all": "E|geometric|removed|exc_all_k1|all|fit|rhoc_fb",
    "P_exc_keep": "P|geometric|removed|exc_nbr_k1|all|fit|P200",
    "E_exc_keep": "E|geometric|removed|exc_nbr_k1|all|fit|rhoc_fb",
    "P_A2i": "P|geometric|removed|A2i|all|fit|P200", "E_A2i": "E|geometric|removed|A2i|all|fit|rhoc_fb",
    "P_A2ii": "P|geometric|removed|A2ii|all|fit|P200", "E_A2ii": "E|geometric|removed|A2ii|all|fit|rhoc_fb",
}
B_TEMPS = {"T_M": "T_M|weighted|removed|all|all|fit|T200", "T_V": "T_V|weighted|removed|all|all|fit|T200",
           "T_X": "T_X|weighted|removed|all|all|fit|T200"}


def test_reproduce_B_profiles():
    bt = np.load(BRES / "halo_table.npz")
    order = np.asarray(bt["orig_row"])                 # B row i <-> study row order[i]
    out, fails = {}, []
    with np.load(BRES / "profiles_fit.npz") as z:
        if not np.array_equal(z["stored_edges"], sc.edges("fit")):
            fails.append("fit edges")
        for bkey, key in B_MAP_FIT.items():
            ok, dev, nan_eq = _same(np.asarray(sc.per_halo(key))[order], z[bkey])
            out[f"fit:{bkey}"] = dev
            if not ok:
                fails.append(f"fit:{bkey} dev {dev} nan_eq {nan_eq}")
        T200 = sc.halo_table()["T200_K"][order]
        for bkey, key in B_TEMPS.items():
            ok, dev, nan_eq = _same(np.asarray(sc.per_halo(key))[order] * T200[:, None], z[bkey])
            out[f"fit:{bkey}"] = dev
            if not ok:
                fails.append(f"fit:{bkey} dev {dev}")
        for bkey, sel in (("retained_all", "exc_all_k1"), ("retained_keep", "exc_nbr_k1")):
            ok, dev, _ = _same(np.asarray(sc.retained(sel, "fit"))[order], z[bkey])
            out[f"fit:{bkey}"] = dev
            if not ok:
                fails.append(f"fit:{bkey}")
        not_cached = [k for k in z.files if k not in B_MAP_FIT and k not in B_TEMPS
                      and k not in ("edges", "stored_edges", "retained_all", "retained_keep")]
    for grid in ("lee22", "old"):
        with np.load(BRES / f"profiles_{grid}.npz") as z:
            if not np.array_equal(z["stored_edges"], sc.edges(grid)):
                fails.append(f"{grid} edges")
            for bkey, key in (("P_geo", f"P|geometric|removed|all|all|{grid}|P200"),
                              ("P_geo_agnkept", f"P|geometric|kept|all|all|{grid}|P200"),
                              ("E_geo", f"E|geometric|removed|all|all|{grid}|rhoc_fb"),
                              ("E_geo_agnkept", f"E|geometric|kept|all|all|{grid}|rhoc_fb")):
                ok, dev, _ = _same(np.asarray(sc.per_halo(key))[order], z[bkey])
                out[f"{grid}:{bkey}"] = dev
                if not ok:
                    fails.append(f"{grid}:{bkey} dev {dev}")
    assert not fails, fails
    return {"arrays_compared": len(out), "max_rel_dev": max(out.values()), "per_array_max_rel_dev": out,
            "B_arrays_not_in_cache": not_cached}


# ============================================================================ 3. reproduce Test F stacks
def test_reproduce_F_stacks():
    tab = sc.halo_table()
    fails, devs = [], {}
    with np.load(FRES / "profiles.npz") as z:
        if not np.array_equal(z["halo_rows"], np.arange(N)) or not np.array_equal(z["weight"], tab["weight"]):
            fails.append("F rows / weights differ")
        P = np.asarray(sc.per_halo("P|geometric|removed|all|all|fit|P200"))
        E = np.asarray(sc.per_halo("E|geometric|removed|all|all|fit|rhoc_fb"))
        for b in range(15):                               # every 0.2-dex bin (calibrated bins 5..14 included)
            rows = np.flatnonzero(tab["mass_bin"] == b)
            for fld, arr in (("P", P), ("E", E)):
                r = sc.stack(arr[rows], tab["weight"][rows], "mean", n_boot=0)
                for stat, mine in (("mean", r["value"]), ("sd", r["sd"])):
                    ref = z[f"{stat}{fld}_main"][b]
                    ok, dev, _ = _same(mine, ref)
                    devs[f"{stat}{fld}_b{b}"] = dev
                    if not ok:
                        fails.append(f"{stat}{fld}_main bin {b}: dev {dev}")
                if not np.allclose(r["n_eff"], z["neff_main"][b], rtol=RTOL, atol=0):
                    fails.append(f"neff bin {b}")
    assert not fails, fails
    cal = [v for k, v in devs.items() if int(k.split("_b")[1]) >= 5]
    return {"bins": 15, "calibrated_bins": "5..14", "max_rel_dev_calibrated": max(cal), "max_rel_dev_all": max(devs.values())}


# ============================================================================ 4. reproduce Test B resolution masks
def test_reproduce_B_masks():
    fails, n = [], 0
    bins = {b["index"]: b for b in sc.mass_bins("M200c_0.2")}
    for grid in ("fit", "lee22", "old"):
        with np.load(BRES / f"masks_{grid}.npz") as z:
            for b in range(15):
                rows = bins[b]["rows"]
                if not np.array_equal(np.sort(rows), np.sort(z[f"b{b}__rows"])):
                    fails.append(f"{grid} b{b} rows")
                det = sc.resolution_detail(rows, grid)
                for k in ("strict", "diagnostic"):
                    if not np.array_equal(sc.resolution_mask(rows, grid, k), z[f"b{b}__{k}"]):
                        fails.append(f"{grid} b{b} {k}")
                    n += 1
                if not np.allclose(det["resolved_fraction"], z[f"b{b}__resolved_fraction"], rtol=1e-12, atol=0):
                    fails.append(f"{grid} b{b} resolved_fraction")
                ph = sc.resolution_mask(rows, grid, "per_halo")
                if ph.shape != (len(rows), len(sc.edges(grid)) - 1) or not np.array_equal(ph.all(axis=0), z[f"b{b}__strict"]):
                    fails.append(f"{grid} b{b} per_halo")
            if grid != "old":
                for L, label in (("L0", "13.2-13.4"), ("L1", "13.8-14.0")):     # Lee22 h-unit bins
                    lb = sc.find_bin("Lee22_h", label)
                    if not np.array_equal(np.sort(lb["rows"]), np.sort(z[f"{L}__rows"])):
                        fails.append(f"{grid} {L} rows")
                    if not np.array_equal(sc.resolution_mask(lb["rows"], grid, "strict"), z[f"{L}__strict"]):
                        fails.append(f"{grid} {L} strict")
                    n += 1
    with np.load(FRES / "profiles.npz") as z:            # Test F masks (same grid)
        for b in range(15):
            if not np.array_equal(sc.resolution_mask(bins[b]["rows"], "fit", "strict"), z["strict_mask"][b]):
                fails.append(f"F strict b{b}")
            if not np.array_equal(sc.resolution_mask(bins[b]["rows"], "fit", "diagnostic"), z["diagnostic_mask"][b]):
                fails.append(f"F diagnostic b{b}")
            n += 2
    assert not fails, fails
    return {"masks_compared": n}


# ============================================================================ 5. mass bins
def test_mass_bins():
    h = np.load(STUDY / "selected_halos.npz")
    tab = sc.halo_table()
    bins = sc.mass_bins("M200c_0.2")
    counts = np.bincount(h["mass_bin"], minlength=15)
    mine = np.zeros(15, int)
    for b in bins:
        mine[b["index"]] = b["N"]
        assert np.all(h["mass_bin"][b["rows"]] == b["index"]), b["label"]
        assert np.array_equal(b["weights"], h["weight"][b["rows"]])
    assert np.array_equal(mine, counts), (mine, counts)
    out = {"M200c_0.2_counts": mine.tolist()}
    for s in sc.SCHEMES:
        bs = sc.mass_bins(s)
        rows = np.concatenate([b["rows"] for b in bs])
        assert len(rows) == len(np.unique(rows)), s           # disjoint
        assert all(b["N"] > 0 for b in bs)
        for b in bs:
            assert abs(b["N_eff"] - b["weights"].sum()**2 / (b["weights"]**2).sum()) < 1e-9
        out[s] = {"bins": len(bs), "haloes": int(len(rows)), "labels": [b["label"] for b in bs],
                  "N": [b["N"] for b in bs], "N_eff": [round(b["N_eff"], 1) for b in bs]}
    # M200c schemes cover every halo (12.5 <= log M < 15.5)
    for s in ("M200c_0.1", "M200c_0.3", "M200c_0.5"):
        assert out[s]["haloes"] == N, s
    assert abs(sc.scheme_definition("Lee22_h")["h"] - 0.681) < 1e-12
    out["M500c_0.2_dropped_below_12.5"] = int(np.sum(tab["logm500c"] < 12.5))
    return out


# ============================================================================ 6. internal consistency
def test_phase_and_agn_additivity():
    """xray + cold_sf = agn removed; removed + only = kept (linear sums with the same normalisation)."""
    devs = {}
    for f, norm in (("P", "P200"), ("E", "rhoc_fb"), ("G", "rhoc_fb")):
        for g in ("x200", "fit", "lee22", "old"):
            for sel in ("all", "central_bound"):
                a = np.asarray(sc.per_halo(f"{f}|geometric|removed|{sel}|xray|{g}|{norm}")) + \
                    np.asarray(sc.per_halo(f"{f}|geometric|removed|{sel}|cold_sf|{g}|{norm}"))
                ref = np.asarray(sc.per_halo(f"{f}|geometric|removed|{sel}|all|{g}|{norm}"))
                assert np.allclose(a, ref, rtol=1e-12, atol=1e-300), (f, g, sel)
                devs[f"{f}:{g}:{sel}:phases"] = _maxrel(a, ref)
            a = np.asarray(sc.per_halo(f"{f}|geometric|removed|all|all|{g}|{norm}")) + \
                np.asarray(sc.per_halo(f"{f}|geometric|only|all|all|{g}|{norm}"))
            ref = np.asarray(sc.per_halo(f"{f}|geometric|kept|all|all|{g}|{norm}"))
            assert np.allclose(a, ref, rtol=1e-12, atol=1e-300), (f, g)
            devs[f"{f}:{g}:agn"] = _maxrel(a, ref)
    return {"max_rel_dev": max(devs.values())}


def test_phase_sum_equals_library():
    """The build's custom phase sum equals variants.owner_sum bitwise for the library's own phase vectors."""
    sys.path.insert(0, str(HERE))
    import build_cache as BC
    from analysis import variants as V
    b = BC.Builder.__new__(BC.Builder)
    from analysis import newdata
    b.d = newdata.load(STUDY)
    sel = V.selection("all")
    sel2 = V.selection("remove_bound_to_other_painted")
    out = {}
    for q in ("y", "electrons"):
        for agn, w in (("removed", (0.0, 1.0, 1.0)), ("kept", (1.0, 1.0, 1.0))):
            for s in (sel, sel2):
                mine = b.phase_sum("x200", q, s, w)
                ref = V.owner_sum(b.d, "x200", q, s, agn)
                assert np.array_equal(mine, ref), (q, agn, s.name)
                out[f"{q}:{agn}:{s.name}"] = "bitwise equal"
    b.d.release()
    return out


def test_integrals_against_apertures():
    """Own Y inside R200c / R500c (sum of particle y, x200 / x500 shells) = the particle-pass apertures (R200c, R500c,
    all gas, y_noagn); the normalised x200 / x500 pressure integrates back to the same Y (units check)."""
    tab = sc.halo_table()
    with np.load(STUDY / "apertures.npz") as z:
        ap = z["apertures"]
    schema = json.loads((STUDY / "schema.json").read_text())["names"]
    qi = schema["aperture_quantities"].index("y_noagn")
    ei = schema["aperture_quantities"].index("electrons")
    out = {}
    out["Y200_vs_aperture"] = _maxrel(tab["Y200_own_mpc2"], ap[:, 0, 0, qi])
    out["Y500_vs_aperture"] = _maxrel(tab["Y500_own_mpc2"], ap[:, 1, 0, qi])
    out["Ne200_vs_aperture_electrons_note"] = "aperture 'electrons' includes recent-AGN gas"
    out["Y200_vs_SOAP_noagn_median_ratio"] = float(np.median(tab["Y200_own_mpc2"] / tab["y200_soap_noagn_mpc2"]))
    assert out["Y200_vs_aperture"] < 1e-10 and out["Y500_vs_aperture"] < 1e-10, out
    # pressure integral -> y: y = sum_s P_s P200 V_s MPC_M sigma_T / (m_e c^2), V_s = R^3 4pi/3 dx^3
    from scipy.constants import c, electron_mass, physical_constants
    sigma_t = physical_constants["Thomson cross section"][0]
    mpc_m = 3.08567758e22
    for grid, rkey, pkey, ykey, norm in (("x200", "r200c", "p200_pa", "Y200_own_mpc2", "P200"),
                                          ("x500", "r500c", "p500_B24_pa", "Y500_own_mpc2", "B24")):
        P = np.asarray(sc.per_halo(f"P|geometric|removed|all|all|{grid}|{norm}"))
        integ = sc.integral(P, grid, 1.0)
        y = integ * tab[pkey] * tab[rkey]**3 * mpc_m * sigma_t / (electron_mass * c**2)
        out[f"{grid}_pressure_integral_vs_Y"] = _maxrel(y, tab[ykey])
        assert out[f"{grid}_pressure_integral_vs_Y"] < 1e-10, out
    return out


def test_normalisation_relations():
    tab = sc.halo_table()
    out = {}
    for grid in ("x500", "b24"):
        b24 = np.asarray(sc.per_halo(f"E|geometric|removed|all|all|{grid}|B24"))
        for norm in ("A10", "self_similar_500"):
            assert np.array_equal(np.asarray(sc.per_halo(f"E|geometric|removed|all|all|{grid}|{norm}")), b24)
        # pressure ratios between normalisations are per-halo constants
        pb = np.asarray(sc.per_halo(f"P|geometric|removed|all|all|{grid}|B24"))
        for norm, col in (("A10", "p500_A10_pa"), ("self_similar_500", "p500_ss_pa"), ("P200", "p200_pa")):
            pn = np.asarray(sc.per_halo(f"P|geometric|removed|all|all|{grid}|{norm}"))
            out[f"{grid}:{norm}"] = _maxrel(pn * tab[col][:, None], pb * tab["p500_B24_pa"][:, None])
            assert out[f"{grid}:{norm}"] < 1e-12
    # entropy: K/K200 = (T_e/T200) / (n_e/n_e,200)^(2/3); n_e/n_e,200 = E mu_e m_p / (200 mu_XGP)
    from scipy.constants import electron_mass, proton_mass
    mu_xgp_kg = (electron_mass + proton_mass * 2 / 1.76) / 0.9
    for g in ("fit", "x200"):
        te = np.asarray(sc.per_halo(f"T_e|weighted|removed|all|all|{g}|T200"))
        E = np.asarray(sc.per_halo(f"E|geometric|removed|all|all|{g}|rhoc_fb"))
        K = np.asarray(sc.per_halo(f"K|weighted|removed|all|all|{g}|K200"))
        with np.errstate(divide="ignore", invalid="ignore"):
            Kr = te / (E * 1.14 * proton_mass / (200 * mu_xgp_kg))**(2 / 3)
        ok, dev, nan_eq = _same(K, np.where(E > 0, Kr, np.nan), rtol=1e-9)
        out[f"K_{g}"] = dev
        assert ok, (g, dev, nan_eq)
    # T_e of the X-ray phase vs all phases in hot, resolved shells: ratio near 1 at x ~ 0.5 R200c for massive haloes
    rows = np.flatnonzero(tab["logm200c"] > 14.5)
    s = int(np.searchsorted(sc.edges("fit"), 0.5)) - 1
    r = np.asarray(sc.per_halo("T_e|weighted|removed|all|xray|fit|T200"))[rows, s] / \
        np.asarray(sc.per_halo("T_e|weighted|removed|all|all|fit|T200"))[rows, s]
    out["median_Te_xray_over_all_at_0.5R_logM>14.5"] = float(np.nanmedian(r))
    assert 0.95 < out["median_Te_xray_over_all_at_0.5R_logM>14.5"] < 1.1
    return out


# ============================================================================ 7. statistics
def test_statistics():
    rng = np.random.default_rng(1)
    out = {}
    v = rng.lognormal(0.0, 0.5, size=(401, 7))
    w1 = np.ones(401)
    r = sc.stack(v, w1, "median", n_boot=0)
    assert np.allclose(r["value"], np.median(v, axis=0), rtol=1e-14)
    v2 = v[:400]
    r = sc.stack(v2, np.ones(400), "median", n_boot=0)
    assert np.allclose(r["value"], np.median(v2, axis=0), rtol=1e-14)        # even n: tie averaging
    r = sc.stack(v, w1, "geomean", n_boot=0)
    assert np.allclose(r["value"], np.exp(np.log(v).mean(axis=0)), rtol=1e-13)
    w = rng.uniform(0.5, 3, size=401)
    r0 = sc.stack(v, w, "trimmed_mean", n_boot=0, trim=0.0)
    rm = sc.stack(v, w, "mean", n_boot=0)
    assert np.allclose(r0["value"], rm["value"], rtol=1e-12)
    # trimmed mean with equal weights and n*trim integer = mean of the central order statistics
    v3 = rng.normal(size=(200, 3))
    rt = sc.stack(v3, np.ones(200), "trimmed_mean", n_boot=0, trim=0.05)
    ref = np.sort(v3, axis=0)[10:190].mean(axis=0)
    assert np.allclose(rt["value"], ref, rtol=1e-12)
    # weighted median: integer weights = replicated sample
    wi = rng.integers(1, 4, size=401)
    rr = sc.stack(v, wi.astype(float), "median", n_boot=0)
    rep = np.repeat(v, wi, axis=0)
    assert np.allclose(rr["value"], np.median(rep, axis=0), rtol=1e-14)
    # y-weighted mean = weighted mean with w * y
    y = rng.uniform(0, 2, size=401)
    ry = sc.stack(v, w, "y_weighted_mean", n_boot=0, y200=y)
    assert np.allclose(ry["value"], (w * y) @ v / (w * y).sum(), rtol=1e-12)
    # NaN handling equals analysis.variants.stack
    sys.path.insert(0, str(STUDY))
    from analysis import variants as V
    tab = sc.halo_table()
    b = sc.find_bin("M200c_0.2", "14.5-14.7")
    P = np.asarray(sc.per_halo("P|geometric|removed|exc_all_k2|all|fit|P200"))[b["rows"]]
    mine = sc.stack(P, b["weights"], "mean", n_boot=0)
    lib = V.stack(P, b["weights"])
    assert np.allclose(mine["value"], lib["mean"], rtol=1e-12, equal_nan=True)
    assert np.array_equal(mine["n_used"], lib["n"]) and np.allclose(mine["n_eff"], lib["n_eff"], rtol=1e-12)
    out["nan_handling_equals_variants.stack"] = True
    # bootstrap: mean error ~ sd / sqrt(n_eff); pairing reproducible
    Pm = np.asarray(sc.per_halo("P|geometric|removed|all|all|fit|P200"))[b["rows"]]
    r1 = sc.stack(Pm, b["weights"], "mean", n_boot=400, seed=5, strata=b["strata"], return_boot=True)
    r2 = sc.stack(Pm * 1.01, b["weights"], "mean", n_boot=400, seed=5, strata=b["strata"], return_boot=True)
    assert np.allclose(r2["boot"], 1.01 * r1["boot"], rtol=1e-12)            # same resamples -> paired
    s = (sc.edges("fit")[:-1] >= 0.1) & (sc.edges("fit")[1:] <= 2)
    ratio = r1["err"][s] / (r1["sd"][s] / np.sqrt(r1["n_eff"][s]))
    out["boot_err_over_sd_sqrt_neff_median"] = float(np.median(ratio))
    assert 0.7 < np.median(ratio) < 1.3
    # weighted percentiles bracket the median
    rmed = sc.stack(Pm, b["weights"], "median", n_boot=0)
    assert np.all((rmed["q16"] <= rmed["value"] + 1e-15) & (rmed["value"] <= rmed["q84"] + 1e-15))
    # per_halo mask: shells all-resolved under strict keep every halo
    ph = sc.resolution_mask(b["rows"], "fit", "per_halo")
    rph = sc.stack(Pm, b["weights"], "mean", halo_shell_mask=ph, n_boot=0)
    st = sc.resolution_mask(b["rows"], "fit", "strict")
    full = sc.stack(Pm, b["weights"], "mean", n_boot=0)["value"]
    assert np.allclose(rph["value"][st], full[st], rtol=1e-12)
    return out


def test_stack_bins_and_records():
    recs = sc.stack_bins("P|geometric|removed|all|all|fit|P200", "M200c_0.2", "median", "strict", n_boot=50)
    txt = json.dumps(recs, allow_nan=False)
    assert len(recs) == 15 and all(len(r["value"]) == 35 for r in recs)
    rec = recs[8]
    for k in ("variant", "field", "scheme", "bin", "x_edges", "value", "err", "q16", "q84", "n_used", "N", "N_eff",
              "resolved", "statistic", "rule"):
        assert k in rec, k
    v = [x for x in rec["value"] if x not in (None, 0.0)]
    assert all(len(f"{abs(x):.10g}".replace(".", "").lstrip("0").rstrip("0")) <= 4 for x in v[:10])
    recs2 = sc.stack_bins("E|geometric|removed|A2ii|all|fit|rhoc_fb", "Lee22_h", "y_weighted_mean", "per_halo",
                          n_boot=20)
    json.dumps(recs2, allow_nan=False)
    return {"records": len(recs), "json_bytes_15_bins": len(txt)}


# ============================================================================ 8. observable propagation
def test_observable_impact():
    summ = json.loads((FRES / "importance_summary.json").read_text())
    names = {"tSZ_yy_total_ell1000": "yytot_1000", "tSZ_yy_total_ell100": "yytot_100", "tSZ_yy_1h_ell3000": "yy1h_3000",
             "y_mean": "ymean", "kSZ_1h_ell1000": "ksz1h_1000", "DM_mean": "dm_mean", "DM_var": "dm_var",
             "DM_q50": "dm_q50", "DM_Cl_total_ell1000": "dmcltot_1000", "tSZxDM_total_ell1000_P": "ydmtot_P_1000",
             "tSZxDM_total_ell1000_E": "ydmtot_E_1000"}
    out = {}
    for fld in ("P", "E"):
        res = sc.observable_impact(np.full((10, 35), 0.01), fld)
        for k, r in res.items():
            assert abs(r["dO_over_O"] - r["k"] * np.expm1(0.01) * r["calibrated_share"]) < 1e-14
            assert abs(r["dlnO"] - np.log1p(r["dO_over_O"])) < 1e-15
            if k in names:
                assert abs(r["calibrated_share"] - summ[names[k]]["calibrated_share"]) < 1e-12, k
            out[k] = {"calibrated_share": round(r["calibrated_share"], 4), "k": r["k"]}
    d = np.zeros((10, 35))
    d[3, :] = np.nan
    r = sc.observable_impact(d, "P")["tSZ_yy_total_ell1000"]
    assert r["dlnO"] == 0 and 0 < r["missing_share"] < 1
    full = np.zeros((16, 35))
    full[5:15] = 0.02
    r = sc.observable_impact(full, "E")["DM_mean"]
    assert abs(r["dO_over_O"] - np.expm1(0.02) * r["calibrated_share"]) < 1e-14
    a = np.ones((10, 35)) * 1.03
    r = sc.impact_from_stacks(a, np.ones((10, 35)), "P")["tSZ_yy_total_ell1000"]
    assert abs(r["dO_over_O"] - 2 * 0.03 * r["calibrated_share"]) < 1e-12
    return out


def test_observable_impact_vs_F_truth():
    """First-order propagation of the stacked-mean change (F selections a2i, a2ii, agn_kept vs main) against the
    exact F engine truths (F_observables/results/truth.npz, truth_summary.json): linear observables must agree to
    1e-4 in d ln O, quadratic ones to 15% of the change + 1e-3."""
    F = np.load(FRES / "profiles.npz")
    T = np.load(FRES / "truth.npz")
    S = json.loads((FRES / "truth_summary.json").read_text())
    ell = np.load(sc.IMPORTANCE_FILE)["ell"]
    cal = list(sc.CAL_BINS)
    tmap = {"tSZ_yy_total_ell100": ("yytot", 100), "tSZ_yy_total_ell1000": ("yytot", 1000),
            "tSZ_yy_total_ell3000": ("yytot", 3000), "tSZ_yy_1h_ell1000": ("yy1h", 1000), "y_mean": ("ymean", None),
            "kSZ_1h_ell100": ("ksz1h", 100), "kSZ_1h_ell1000": ("ksz1h", 1000), "kSZ_1h_ell3000": ("ksz1h", 3000),
            "DM_mean": ("dm_mean", None), "DM_var": ("dm_var", None), "DM_Cl_total_ell1000": ("dmcltot", 1000)}
    linear_obs = {"y_mean", "DM_mean"}
    out, fails = {}, []
    for sel in ("agn_kept", "a2i", "a2ii"):
        lin = {**sc.impact_from_stacks(F[f"meanP_{sel}"][cal], F["meanP_main"][cal], "P"),
               **sc.impact_from_stacks(F[f"meanE_{sel}"][cal], F["meanE_main"][cal], "E")}
        rows = {}
        for name, (ok, at) in tmap.items():
            a, b = T[f"truth_{sel}__{ok}"], T[f"truth_main__{ok}"]
            if at is not None:
                i = int(np.flatnonzero(ell == at)[0])
                a, b = a[i], b[i]
            exact = float(np.log(a / b))
            rows[name] = (lin[name]["dlnO"], exact)
        for qi, qn in ((0, "DM_q16"), (1, "DM_q50"), (2, "DM_q84")):
            rows[qn] = (lin[qn]["dlnO"], float(np.log(S["pdf"][f"truth_{sel}"]["quantiles"][qi]
                                                      / S["pdf"]["truth_main"]["quantiles"][qi])))
        cross = sc.impact_cross(np.log(F[f"meanP_{sel}"][cal] / F["meanP_main"][cal]),
                                np.log(F[f"meanE_{sel}"][cal] / F["meanE_main"][cal]))
        i = int(np.flatnonzero(ell == 1000)[0])
        rows["tSZxDM_total_ell1000"] = (cross["dlnO"], float(np.log(T[f"truth_{sel}__ydmtot"][i]
                                                                   / T["truth_main__ydmtot"][i])))
        for name, (l, e) in rows.items():
            tol = 1e-4 if name in linear_obs else 0.15 * abs(e) + 1e-3
            if abs(l - e) > tol:
                fails.append(f"{sel} {name}: linear {l:+.5f} exact {e:+.5f}")
        out[sel] = {k: {"linear": round(l, 5), "exact": round(e, 5)} for k, (l, e) in rows.items()}
    assert not fails, fails
    return out


TESTS = [test_cache_files, test_reproduce_B_profiles, test_reproduce_F_stacks, test_reproduce_B_masks, test_mass_bins,
         test_phase_and_agn_additivity, test_phase_sum_equals_library, test_integrals_against_apertures,
         test_normalisation_relations, test_statistics, test_stack_bins_and_records, test_observable_impact,
         test_observable_impact_vs_F_truth]


def main():
    t_all = time.perf_counter()
    failed = 0
    for t in TESTS:
        t0 = time.perf_counter()
        try:
            info = t()
            RESULTS[t.__name__] = {"status": "pass", "seconds": round(time.perf_counter() - t0, 2), "info": info}
            print(f"PASS {t.__name__} ({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception as exc:       # noqa: BLE001
            failed += 1
            RESULTS[t.__name__] = {"status": "FAIL", "seconds": round(time.perf_counter() - t0, 2),
                                   "error": repr(exc)[:2000], "traceback": traceback.format_exc()[-3000:]}
            print(f"FAIL {t.__name__}: {exc!r}"[:1500], flush=True)
    import resource
    RESULTS["_summary"] = {"tests": len(TESTS), "failed": failed, "seconds": round(time.perf_counter() - t_all, 1),
                           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    (HERE / "test_core_results.json").write_text(json.dumps(RESULTS, indent=1, default=str) + "\n")
    print(json.dumps(RESULTS["_summary"]))
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
