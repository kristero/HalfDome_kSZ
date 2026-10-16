"""Test K, K5 smoke and parity checks (local; small).

  python -B k_parity.py [--julia] [--results DIR]
Checks (written to results/k5_parity.json):
  integral      k_fit_core.cumulative (closed form: incomplete beta / 2F1 / series) vs 40-digit mpmath quadrature,
                ~1,000 random and 288 targeted parameter sets of both forms: alpha 0.1-30, x/x_c up to 400, outer
                slopes steeper and shallower than x^-3 (b = -3..15, incl. b ~ 0 and negative integers)
  xgpaint       (--julia) XGPaint (validated checkout) evaluated by k_xgpaint_check.jl vs k_fit_core: parameter laws
                (published, all-power-law, free gamma, free x_c, BreakModel), the 3D shape, the line-of-sight shape
                integral, the P200 prefactor and R200c, the tau rho_2d normalisation (incl. f_b); tolerance 1e-6
  compression   32-node quadrature vs every halo (computed on maas in K1 for the published and a perturbed law;
                here also for the fitted laws on the stored 256-halo subsamples)
  particles     z = 0: SOAP per-halo normalised apertures (results/z0_study_rows.npz) vs the particle-pass
                cumulative of the same haloes (stacking_choices research/soap/particle_cumulative.npz, PCHIP in ln x)
  shells        the B12 shell average from k_fit_core vs the study's models.py shell() (Tests B-I) on the fit grid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k_fit_core as K  # noqa: E402

JULIA = "/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia"
DEPOT = "/home/kn18001/.julia"
G_SI = 6.67430e-11
MPC_M = 3.08567758e22


def _ref_incomplete_beta(a, b, ls, dps=40):
    """40-digit reference of Int_0^{s_x} s^(a-1) (1+s)^(-(a+b)) ds (= B(t; a, b), t = s_x/(1+s_x), ln s_x = ls) by
    mpmath quadrature in ln u, split at u = 1/2 so that the part near t -> 1 is integrated in w = 1 - u with its exact
    lower limit 1/(1 + s_x)."""
    import mpmath as mp
    mp.mp.dps = dps
    a, b, ls = mp.mpf(a), mp.mpf(b), mp.mpf(ls)
    t = 1 / (1 + mp.exp(-ls))
    half = mp.mpf(1) / 2
    f1 = lambda v: mp.exp(a * v) * (1 - mp.exp(v)) ** (b - 1)          # noqa: E731  (u = e^v)
    top = mp.log(t) if t <= half else mp.log(half)
    out = mp.quad(f1, mp.linspace(top - 120 / a, top, 60))
    if t > half:
        y = 1 / (1 + mp.exp(ls))
        f2 = lambda v: mp.exp(b * v) * (1 - mp.exp(v)) ** (a - 1)      # noqa: E731  (w = 1 - u = e^v)
        out += mp.quad(f2, mp.linspace(mp.log(y), mp.log(half), 120))
    return out


def check_integral(n=1400, seed=6):
    """k_fit_core.cumulative vs 40-digit quadrature over a wide parameter range: alpha 0.1-30 (log-uniform), gamma
    -2.5..-0.001, x_c 0.01-3.2, x 0.01-4 (x/x_c up to 400, ln s_x up to 180), outer exponent chosen so that
    b = e - (3+gamma)/alpha covers -3.03..15 with extra weight near b = 0 and b = -1, -2, -3 (the 2F1 / series
    branches), plus a targeted grid alpha 2-30 x beta 2.8-4 (B16, gamma -0.55) at x/x_c = 13.75, 72, 1 and 400,
    which contains the case of the run-2 bug (alpha 11.6, x/x_c 13.75)."""
    rng = np.random.default_rng(seed)
    cases = []
    for t in range(n):
        form = "B12" if t % 2 else "B16"
        al = 10 ** rng.uniform(-1, math.log10(30)); xc = 10 ** rng.uniform(-2, 0.5); x = 10 ** rng.uniform(-2, math.log10(4))
        ga = -rng.uniform(0.001, 2.5)
        a = (3 + ga) / al
        kind = t % 7
        if kind in (0, 1):
            b = rng.uniform(-0.05, 0.05)
        elif kind == 2:
            b = -rng.integers(1, 4) + rng.uniform(-0.03, 0.03)
        elif kind == 3:
            b = rng.uniform(-3, 0.02)
        elif kind == 4:
            b = rng.uniform(0.02, 0.3)
        else:
            b = rng.uniform(-2, 15)
        e = a + b
        be = e if form == "B12" else e * al - ga
        if (form == "B12" and not 0.5 <= be <= 20) or (form == "B16" and not 1.0 <= be <= 25):
            continue
        cases.append((form, xc, al, be, ga, x))
    for al in (2.0, 4.0, 6.0, 8.0, 10.0, 11.6, 15.0, 20.0, 30.0):
        for be in (2.8, 2.95, 2.99, 3.0, 3.01, 3.05, 3.2, 4.0):
            for xc, x in ((0.24, 3.3), (0.05, 3.6), (0.5, 0.5), (0.01, 4.0)):
                cases.append(("B16", xc, al, be, -0.55, x))
    errs, worst, wpar = [], 0.0, None
    lsmax = 0.0
    for form, xc, al, be, ga, x in cases:
        ls = al * math.log(x / xc)
        if ls > 690:
            continue
        e = be if form == "B12" else (be + ga) / al
        a = (3 + ga) / al
        pref = K.E_OVER_TH if form == "B12" else 1.0
        ref = float(_ref_incomplete_beta(a, e - a, ls)) * pref * xc ** 3 / al
        got = float(K.cumulative(x, form, 1.0, xc, al, be, ga))
        err = abs(got / ref - 1) if np.isfinite(got) else np.inf
        errs.append(err)
        lsmax = max(lsmax, ls)
        if err > worst:
            worst, wpar = err, dict(form=form, xc=xc, alpha=al, beta=be, gamma=ga, x=x, a=a, b=e - a, ln_s=ls)
    errs = np.array(errs)
    return {"n": int(len(errs)), "max_rel_err": float(worst), "median_rel_err": float(np.median(errs)),
            "n_above_1e-12": int(np.sum(errs > 1e-12)), "worst_case": wpar, "max_ln_s": lsmax,
            "reference": "mpmath 1.3 quadrature at 40 digits (k_parity._ref_incomplete_beta)",
            "range": "alpha 0.1-30, gamma -2.5..-0.001, x_c 0.01-3.2, x 0.01-4, b -3.03..15 (incl. b ~ 0, -1, -2, -3), "
                     "targeted grid alpha 2-30 x beta 2.8-4 at x/x_c 13.75 / 72 / 1 / 400",
            "pass": worst < 1e-10}


def jl(v):
    if isinstance(v, dict):
        return "Dict{String,Any}(" + ", ".join(f'"{k}" => {jl(x)}' for k, x in v.items()) + ")"
    if isinstance(v, (list, tuple)):
        return "Any[" + ", ".join(jl(x) for x in v) + "]"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(float(v))


def xgp_cases():
    """Parameter sets: dict q -> (A, alpha_m, alpha_z) in XGPaint's argument convention."""
    pub12 = {"P0": (18.1, 0.154, -0.758), "xc": (0.497, -0.00865, 0.731), "alpha": (1.0, 0.0, 0.0),
             "beta": (4.35, 0.0393, 0.415), "gamma": (-0.3, 0.0, 0.0)}
    pub16 = {"P0": (4.0e3, 0.29, -0.66), "xc": (0.5, 0.0, 0.0), "alpha": (0.88, -0.03, 0.19),
             "beta": (3.83, 0.04, -0.025), "gamma": (-0.2, 0.0, 0.0)}
    t12 = {"P0": (7.3, 0.31, -0.42), "xc": (0.83, -0.11, 0.2), "alpha": (1.0, 0.0, 0.0),
           "beta": (5.6, 0.07, 0.3), "gamma": (-0.55, 0.05, -0.1)}
    t16 = {"P0": (812.0, 0.41, -0.2), "xc": (0.31, 0.06, -0.15), "alpha": (1.4, -0.08, 0.1),
           "beta": (2.6, 0.03, 0.08), "gamma": (-0.45, -0.07, 0.12)}
    return {"y_published": {"kind": "y", "pars": pub12}, "y_free": {"kind": "y", "pars": t12},
            "tau_published": {"kind": "tau", "pars": pub16}, "tau_free": {"kind": "tau", "pars": t16},
            "break_test": {"kind": "break", "pars": dict(t12, gamma=(-0.55, 0.0, 0.0)), "abrk": 0.37, "mbrk": 2.5e14}}


def python_law_values(pars, M, z, form):
    """k_fit_core-side profile parameters for an XGPaint argument set (pl laws for every parameter)."""
    m, z1 = M / 1e14, 1 + z
    v = {q: pars[q][0] * m ** pars[q][1] * z1 ** pars[q][2] for q in ("P0", "xc", "alpha", "beta", "gamma")}
    return v


def python_law_via_core(pars, M, z, form, kind):
    """Same numbers through k_fit_core.Law (free/brk_amp), to prove the Law -> XGPaint mapping."""
    law = K.Law(form, "free" if kind != "break" else "brk_amp")
    th = law.start()
    for q, xq in (("amp", "P0"), ("xc", "xc"), ("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma")):
        if law.qkind[q] == "pl":
            A, am, az = pars[xq]
            th[law.index[f"{q}:lnA"]] = math.log(abs(A)); th[law.index[f"{q}:am"]] = am; th[law.index[f"{q}:az"]] = az
        else:
            law.fixed[q] = pars[xq][0]
    return law, th


def check_xgpaint(work):
    cases = xgp_cases()
    masses = [10 ** 12.8, 10 ** 13.5, 2.0e14, 1.0e15]
    zs = [0.0, 0.5, 2.0, 3.0]
    x3d = [0.03, 0.1, 0.3, 0.7, 1.0, 2.0, 3.3]
    xproj = [0.05, 0.3, 1.0, 2.5]
    inp = {"cases": cases, "masses": masses, "redshifts": zs, "x3d": x3d, "xproj": xproj}
    work.mkdir(parents=True, exist_ok=True)
    (work / "xgp_cases.jl").write_text("inp = " + jl(inp) + "\n")
    os.chmod(work, 0o777)
    for f in work.iterdir():
        os.chmod(f, 0o666)
    out_json = work / "xgp_out.json"
    cmd = ["runuser", "-u", "kn18001", "--", "env", f"JULIA_DEPOT_PATH={DEPOT}", "JULIA_NUM_THREADS=1", JULIA,
           "--startup-file=no", str(HERE / "k_xgpaint_check.jl"), str(work / "xgp_cases.jl"), str(out_json)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode:
        return {"pass": False, "error": (r.stderr + r.stdout)[-3000:]}
    X = json.loads(out_json.read_text())
    worst = {}
    rows_out = []

    def upd(key, a, b):
        e = abs(a / b - 1) if b != 0 else abs(a - b)
        worst[key] = max(worst.get(key, 0.0), e)

    for name, case in cases.items():
        form = "B16" if case["kind"] == "tau" else "B12"
        for row in X[name]:
            M, z = row["M"], row["z"]
            v = python_law_values(case["pars"], M, z, form)
            law, th = python_law_via_core(case["pars"], M, z, form, case["kind"])
            if case["kind"] == "break":
                th[law.index["amp:abrk"]] = case["abrk"]; th[law.index["log10Mb"]] = math.log10(case["mbrk"])
            ph = law.phi(th, np.array([math.log(M)]), np.array([math.log1p(z)]))
            ph = {q: float(ph[q][0]) for q in K.QNAMES}
            amp_x = row["amp"] * (row.get("break_factor", 1.0))
            upd("amp(core Law) vs XGPaint", ph["amp"], amp_x)
            upd("x_c", ph["xc"], row["xc"]); upd("alpha", ph["alpha"], row["alpha"]); upd("gamma", ph["gamma"], row["gamma"])
            beta_x = (ph["alpha"] * ph["beta"] - ph["gamma"]) if form == "B12" else ph["beta"]
            upd("beta (XGPaint internal)", beta_x, row["beta_xgp"])
            upd("law formula vs XGPaint P0", v["P0"], row["amp"])
            shape = np.array([K.profile(x, form, 1.0, ph["xc"], ph["alpha"], ph["beta"], ph["gamma"]) for x in x3d])
            if form == "B12":
                shape = shape / K.E_OVER_TH
            for a, b in zip(shape, row["gnfw3d"]):
                upd("3D shape", a, b)
            for xp, b in zip(xproj, row["los"]):
                f = lambda y: K.profile(math.sqrt(y * y + xp * xp), form, 1.0, ph["xc"], ph["alpha"], ph["beta"], ph["gamma"])  # noqa: E731
                br = [0, xp, 10 * xp, 100 * xp, 1e3, 1e4, 1e5]
                val = 2 * sum(quad(f, br[i], br[i + 1], epsabs=0, epsrel=1e-12, limit=400)[0] for i in range(len(br) - 1))
                if form == "B12":
                    val /= K.E_OVER_TH
                upd("line-of-sight shape integral", val, b)
                if case["kind"] == "tau":
                    r200 = row["R200_Mpc"] * MPC_M
                    pred = ph["amp"] * row["f_b"] * row["rho_crit_kg_m3"] * r200 * val
                    upd("tau rho_2d normalisation (rho0 f_b rho_c R200 * LOS)", pred, row["rho2d_kg_m2"][xproj.index(xp)])
            r200 = row["R200_Mpc"] * MPC_M
            Mkg = M * 1.98847e30
            upd("R200c = (3M/(4 pi 200 rho_c))^(1/3)", (3 * Mkg / (4 * math.pi * 200 * row["rho_crit_kg_m3"])) ** (1 / 3), r200)
            if case["kind"] != "tau":
                pref = G_SI * Mkg * 200 * row["rho_crit_kg_m3"] * row["f_b"] / 2
                upd("P_th200 R200 prefactor G M 200 rho_c f_b / 2", pref, row["pth_prefactor_Pa_m"])
                p200_k = G_SI * Mkg * (Mkg / (4 * math.pi / 3 * r200 ** 3)) * row["f_b"] / (2 * r200)   # Test K's P200
                upd("Test K P200 (M/(4pi/3 R^3) form) vs XGPaint G M 200 rho_c f_b/(2 R200)", p200_k, row["pth_prefactor_Pa_m"] / r200)
            rows_out.append({"case": name, "M": M, "z": z})
    # G differs slightly between CODATA 2018 (XGPaint) and the 6.67430e-11 used here: both are CODATA 2018
    return {"pass": all(v < 1e-6 for v in worst.values()), "max_rel_diff": worst, "n_evaluations": len(rows_out),
            "cases": {k: {kk: vv for kk, vv in v.items()} for k, v in cases.items()},
            "masses": masses, "redshifts": zs, "x3d": x3d, "xproj": xproj, "tolerance": 1e-6,
            "xgpaint_checkout": "/home/kn18001/.julia/dev/XGPaint (branch cluster)", "julia": JULIA}


def check_shells():
    """k_fit_core shell average vs the study's models.py shell() (Tests B-I) for the published B12/B16."""
    sys.path.insert(0, str(HERE.parent))
    try:
        import models as M
        import pilot_physics as PP
    except Exception as exc:   # noqa: BLE001
        return {"pass": None, "error": repr(exc)}
    e = np.asarray(PP.EDGES)
    out = {}
    for fam, form in (("b12", "B12"), ("b16", "B16")):
        par = PP.B12[:3] if fam == "b12" else PP.B16[:3]
        ref = M.shell(fam, par, edges=e)
        if form == "B12":
            phi = dict(amp=par[0], xc=par[1], alpha=1.0, beta=par[2], gamma=-0.3)
        else:
            phi = dict(amp=par[0], xc=0.5, alpha=par[1], beta=par[2], gamma=-0.2)
        c = K.cumulative(np.maximum(e, 1e-12), form, *(phi[q] for q in K.QNAMES))
        c[0] = 0.0
        got = 3 * np.diff(c) / np.diff(e ** 3)
        out[form] = float(np.max(np.abs(got / ref - 1)))
    return {"pass": max(out.values()) < 1e-6, "max_rel_diff": out, "note": "models.py uses 32-point Gauss-Legendre per shell"}


def check_particles(results):
    """z = 0: Test K's normalised SOAP apertures of the study haloes (results/z0_study_rows.npz, written in K1) vs the
    particle-pass cumulative of the same haloes, normalised independently (selected_halos M200c / R200c, the study's
    pilot_physics.p200 with the plan's f_b), at each halo's own SOAP radius (monotone PCHIP in ln x through the 67
    X200 edges, as in the SOAP audit).  Ratios per aperture and mass group (median, 16-84%)."""
    zr = Path(results) / "z0_study_rows.npz"
    if not zr.exists():
        return {"pass": None, "note": "results/z0_study_rows.npz not available"}
    from scipy.interpolate import PchipInterpolator
    sys.path.insert(0, str(HERE.parent))
    import pilot_physics as PP
    Z = np.load(zr)
    study = HERE.parent
    H = np.load(study / "selected_halos.npz")
    plan = json.loads((study / "metadata_plan.json").read_text())
    fb = float(plan["baryon_fraction"])
    PC = np.load(study / "stacking_choices_20261014" / "research" / "soap" / "particle_cumulative.npz")
    found = Z["found"].astype(bool)
    assert np.array_equal(Z["study_soap_row"], H["soap_row"])
    X200 = PC["x200"]
    inc = PC["shells200_inc"]                      # [n, 3 phases, 4 q, 67]; q = count, mass, y, electrons
    cum = np.concatenate([np.zeros(inc.shape[:-1] + (1,)), np.cumsum(inc, -1)], -1)
    M = H["m200c_msun"]; R = H["r_200_crit_mpc"]
    p200 = PP.p200(M, R, fb)
    yconv = 8.1871057769e-14 / 6.6524587321e-29 * MPC_M ** 2
    out = {"n_found": int(found.sum()), "n_study": int(len(found)), "f_b_plan": fb,
           "M200c_soapK_over_study_median": float(np.median(np.exp(Z["lnM"][found]) / M[found])),
           "R200c_soapK_over_study_median": float(np.median(Z["R200c"][found] / R[found])), "apertures": {}}
    names = list(json.loads((HERE / "results" / "k0_inventory.json").read_text())["files"]) if False else None
    import k_common as KC
    lm = np.log10(M)
    groups = {"12.5-13.5": lm < 13.5, "13.5-14.5": (lm >= 13.5) & (lm < 14.5), ">=14.5": lm >= 14.5}
    worst = 0.0
    for j, ap in enumerate(KC.AP_NAMES):
        x = Z["x"][:, j].astype(float)
        ok = found & np.isfinite(x) & (x >= X200[1]) & (x <= X200[-1]) & Z["valid"][:, j]
        if ok.sum() < 10:
            continue
        pv = {"Yn": np.full(len(x), np.nan), "Mg": np.full(len(x), np.nan)}
        for i in np.nonzero(ok)[0]:
            P = PchipInterpolator(np.log(X200[1:]), cum[i, :, :, 1:], axis=-1)
            v = P(np.log(x[i]))                   # [3, 4]
            pv["Yn"][i] = v[1:, 2].sum() * yconv / (p200[i] * (R[i] * MPC_M) ** 3)
            pv["Mg"][i] = v[:, 1].sum() / (fb * M[i])
        e = {}
        for q in ("Yn", "Mg"):
            if q == "Yn" and j >= KC.N_SO:
                continue
            soap = Z[q][:, j].astype(float)
            for g, gm in groups.items():
                m = ok & gm & np.isfinite(pv[q]) & (pv[q] > 0)
                if m.sum() < 5:
                    continue
                r = soap[m] / pv[q][m]
                e[f"{q}|{g}"] = {"n": int(m.sum()), "median": float(np.median(r)), "p16": float(np.percentile(r, 16)),
                                 "p84": float(np.percentile(r, 84))}
                if g != "12.5-13.5":
                    worst = max(worst, abs(e[f"{q}|{g}"]["median"] - 1))
        out["apertures"][ap] = e
    out["max_abs_median_dev_above_13.5"] = worst
    out["pass"] = worst < 0.005
    out["note"] = ("SOAP per-halo Ytilde_noAGN and Mtilde from Test K's K1 (own M200c, R200c, P200 with f_b = 0.0486/0.306) vs "
                   "the particle cumulative normalised with the study's selected_halos M200c/R200c and pilot_physics.p200; "
                   "PCHIP interpolation scatter is ~0.5% (16-84%, SOAP audit), so medians are the test")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--julia", action="store_true")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    res["integral"] = check_integral()
    print("integral", res["integral"]["max_rel_err"], flush=True)
    res["shells"] = check_shells()
    print("shells", res["shells"], flush=True)
    res["particles"] = check_particles(a.results)
    print("particles", res["particles"].get("max_abs_median_dev_above_13.5"), res["particles"].get("pass"), flush=True)
    if a.julia:
        res["xgpaint"] = check_xgpaint(HERE / "parity_work")
        print("xgpaint", json.dumps(res["xgpaint"].get("max_rel_diff", res["xgpaint"]), indent=1), flush=True)
    out = Path(a.out) if a.out else Path(a.results) / "k5_parity_local.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
