"""F1 gate: the in-basis machinery reproduces kernels/ engine outputs exactly (reference set V).

With g_b = the kernels/ P4 and E4 fits and F_s = <g_b>_s, every in-basis observable must equal the
engine's value in kernels/jacobians.npz (same grid nodes inside 4R, same b quadrature, same fallback),
and the in-basis Jacobians (plus the builder's fallback pieces) must equal the engine Jacobians summed
from the 67 X200 shells onto the 35 F shells.  The DM PDF pass (f_pdf.py) is checked the same way
(quantiles, PDF mean/variance, Mecke quantile Jacobians).  Output: products/validation_V.json.
"""
from __future__ import annotations

import f_basis as FB
import f_common as FC
import f_pdf as FP
import numpy as np


def x200_to_f():
    mids = .5 * (FC.X200[:-1] + FC.X200[1:])
    return np.searchsorted(FC.F_EDGES, mids) - 1


def aggregate(J, mapping, axis=0, n=FC.N_F):
    moved = np.moveaxis(np.asarray(J), axis, 0)
    out = np.zeros((n,) + moved.shape[1:])
    np.add.at(out, mapping, moved)
    return np.moveaxis(out, 0, axis)


def full_bins(Jcal, Jfb):
    """[c, s, ...] calibrated + [16, s, ...] fallback -> [s, 16 engine bins, ...]."""
    out = np.array(Jfb, float, copy=True)
    for c, b in enumerate(FC.CAL_BINS):
        out[b] += Jcal[c]
    return np.moveaxis(out, 0, 1)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a / b - 1)))


def engine_reference():
    """Engine values and Jacobians of the kernels/ P4 and E4 fits on the run's sky, multipoles and support.

    full: kernels/jacobians.npz (computed by kernels/jacobians.py with exactly this setup); its Jacobians are
    per X200 shell.  smoke: a fresh engine run (1 worker) on the reduced sky / multipole grid / support; its
    Jacobians are per grid segment.  Returns (dict of arrays, map unit -> F shell)."""
    if not FC.SMOKE:
        jac = np.load(FC.ROOT / "kernels" / "jacobians.npz")
        return {k: jac[k] for k in jac.files}, x200_to_f()
    import f_kernels as K
    E = FC.E
    base, _, _, _ = K.basis_grid()
    sky = FC.make_sky()
    support = FC.SUPPORT
    variants = [E.Variant("P4", E.pressure_fit(4, support), base, 4., jacobian=True),
                E.Variant("E4", E.electron_fit(4, support), base, 4., jacobian=True)]
    engine = E.Engine(sky)
    first = engine.run(variants, ell=FC.ELL, workers=1, pairs=[("P4", "E4")], verbose=False)
    ref = {"E4": first["dm_pdf_summaries"]["E4"]}
    second = engine.run([variants[1]], ell=FC.ELL[:1], workers=1, pdf_reference=ref, verbose=False)
    J, JP = first["jacobians"], first["pair_jacobians"]["P4xE4"]
    out = {"O_cl_1h__P4": first["cl_1h"][0], "O_cl_2h__P4": first["cl_2h"][0], "O_mean__P4": first["mean"][0],
           "O_ksz_cl_1h__E4": first["ksz_cl_1h"][1], "O_ksz_cl_2h_approx__E4": first["ksz_cl_2h_approx"][1],
           "O_dm_cl_1h__E4": first["dm_cl_1h"][1], "O_dm_cl_2h__E4": first["dm_cl_2h"][1],
           "O_cross_cl_1h__P4xE4": first["cross_cl_1h"][0], "O_cross_cl_2h__P4xE4": first["cross_cl_2h"][0],
           "O_dm_mean__E4": first["dm_mean"][1], "O_dm_var__E4": first["dm_var"][1],
           "O_dm_quantiles__E4": first["dm_quantiles"][1],
           "J_cl_1h__P4": J["P4"]["one"], "J_cl_2h__P4": J["P4"]["two"], "J_mean__P4": J["P4"]["mean"],
           "J_ksz_cl_1h__E4": J["E4"]["ksz_one"], "J_ksz_cl_2h_approx__E4": J["E4"]["ksz_two"],
           "J_dm_cl_1h__E4": J["E4"]["dm_one"], "J_dm_cl_2h__E4": J["E4"]["dm_two"], "J_dm_mean__E4": J["E4"]["dm_mean"],
           "J_dm_var__E4": J["E4"]["dm_var"], "J_cross_1h_P__P4xE4": JP["cross_1h_P"], "J_cross_1h_E__P4xE4": JP["cross_1h_E"],
           "J_cross_2h_P__P4xE4": JP["cross_2h_P"], "J_cross_2h_E__P4xE4": JP["cross_2h_E"],
           "J_dm_quantiles__E4": np.moveaxis(second["jacobians"]["E4"]["dm_quantile"], 0, 0)}
    # engine quantile Jacobian is [quantile, segment, bin]; the others [segment, bin(, ell)]
    mids = np.array([.5 * (base.x[a] + base.x[b]) for a, b in base.segments])
    return out, np.searchsorted(FC.F_EDGES, mids) - 1


def main():
    timer = FC.Timer()
    jac, mp = engine_reference()
    B = FB.Basis("V")
    P, Ee = B.gbar_P, B.gbar_E
    o = B.observables(P, Ee)
    res = {"mode": FC.MODE, "observables_max_relative": {}, "jacobians_max_abs_over_kO": {}}
    pairs = {"yy1h": "O_cl_1h__P4", "yy2h": "O_cl_2h__P4", "ksz1h": "O_ksz_cl_1h__E4",
             "ksz2h": "O_ksz_cl_2h_approx__E4", "dmcl1h": "O_dm_cl_1h__E4", "dmcl2h": "O_dm_cl_2h__E4",
             "ydm1h": "O_cross_cl_1h__P4xE4", "ydm2h": "O_cross_cl_2h__P4xE4", "ymean": "O_mean__P4",
             "dm_mean": "O_dm_mean__E4", "dm_var": "O_dm_var__E4"}
    for k, key in pairs.items():
        res["observables_max_relative"][k] = rel(o[k], jac[key])
    # Jacobians
    Jb = B.jacobians(P, Ee)
    fb = B.fb
    jfb2 = B.Jfb2h
    checks = {
        "yy1h": (Jb["yy1h"], fb["J_yy1h"], "J_cl_1h__P4", 2, "O_cl_1h__P4"),
        "yy2h": (Jb["yy2h"], jfb2["Jfb_yy2h"], "J_cl_2h__P4", 2, "O_cl_2h__P4"),
        "ymean": (Jb["ymean"], fb["J_mean_y"], "J_mean__P4", 1, "O_mean__P4"),
        "ksz1h": (Jb["ksz1h"], fb["J_ksz1h"], "J_ksz_cl_1h__E4", 2, "O_ksz_cl_1h__E4"),
        "ksz2h": (Jb["ksz2h"], jfb2["Jfb_ksz2h"], "J_ksz_cl_2h_approx__E4", 2, "O_ksz_cl_2h_approx__E4"),
        "dmcl1h": (Jb["dmcl1h"], fb["J_dm1h"], "J_dm_cl_1h__E4", 2, "O_dm_cl_1h__E4"),
        "dmcl2h": (Jb["dmcl2h"], jfb2["Jfb_dm2h"], "J_dm_cl_2h__E4", 2, "O_dm_cl_2h__E4"),
        "dm_mean": (Jb["dm_mean"], fb["J_dm_mean"], "J_dm_mean__E4", 1, "O_dm_mean__E4"),
        "dm_var": (Jb["dm_var"], fb["J_dm_var"], "J_dm_var__E4", 2, "O_dm_var__E4"),
        "ydm1h_P": (Jb["ydm1h_P"], fb["J_ydm1h_P"], "J_cross_1h_P__P4xE4", 1, "O_cross_cl_1h__P4xE4"),
        "ydm1h_E": (Jb["ydm1h_E"], fb["J_ydm1h_E"], "J_cross_1h_E__P4xE4", 1, "O_cross_cl_1h__P4xE4"),
        "ydm2h_P": (Jb["ydm2h_P"], jfb2["Jfb_ydm2h_P"], "J_cross_2h_P__P4xE4", 1, "O_cross_cl_2h__P4xE4"),
        "ydm2h_E": (Jb["ydm2h_E"], jfb2["Jfb_ydm2h_E"], "J_cross_2h_E__P4xE4", 1, "O_cross_cl_2h__P4xE4"),
    }
    for name, (jc, jf, key, k, okey) in checks.items():
        mine = full_bins(jc, jf)
        eng = aggregate(jac[key], mp, axis=0)
        O = np.asarray(jac[okey], float)
        res["jacobians_max_abs_over_kO"][name] = float(np.max(np.abs(mine - eng) / (k * np.abs(O))))
        res.setdefault("jacobian_sum_rules", {})[name] = float(np.max(np.abs(mine.sum(axis=(0, 1)) / (k * O) - 1)))
    FC.log("moments and Jacobians compared", timer.mark("moments"))
    # DM PDF pass (V): quantiles, PDF mean/variance, Mecke Jacobians
    ent = FP.Entries(B.ns)
    h = ent.new_hist("E4")
    rows = {}
    for c in range(FC.N_CAL):
        rows[c] = ent.add(c, Ee[c], h, 1.0)
    ent.freeze()
    first = FP.run(ent, "V", keep_density=True)
    s = first["summaries"][0]
    res["pdf"] = {"quantiles_mine": s["quantiles"], "quantiles_engine": jac["O_dm_quantiles__E4"],
                  "quantiles_max_relative": rel(s["quantiles"], jac["O_dm_quantiles__E4"]),
                  "pdf_mean_vs_analytic": float(s["mean"] / o["dm_mean"] - 1),
                  "pdf_variance_vs_analytic": float(s["variance"] / o["dm_var"] - 1),
                  "mass_on_grid": s["mass_on_grid"], "timing_pass1": first["timing"]}
    plan = dict(rows)
    plan["fallback"] = True
    second = FP.run(ent, "V", mecke=[(s, plan)], include_fallback=True)
    Jq = np.zeros((len(FC.QUANTILES), FC.N_F, 16))
    for p in range(len(FC.QUANTILES)):
        Jq[p] = full_bins(second["J"][0][p], second["Jfb"][0][p])
    eng = aggregate(np.moveaxis(jac["J_dm_quantiles__E4"], 1, 0), mp, axis=0)          # [s, p, bin]
    eng = np.moveaxis(eng, 1, 0)                                                         # [p, s, bin]
    Q = np.asarray(jac["O_dm_quantiles__E4"], float)
    res["pdf"]["quantile_jacobian_max_abs_over_Q"] = float(np.max(np.abs(Jq - eng) / Q[:, None, None]))
    res["pdf"]["quantile_jacobian_sum_rule"] = (Jq.sum(axis=(1, 2)) / Q - 1).tolist()
    res["pdf"]["timing_pass2"] = second["timing"]
    tol = {"observables": 1e-9, "jacobians": 1e-9, "quantiles": 1e-6, "quantile_jacobians": 1e-5}
    res["passed"] = {
        "observables": max(res["observables_max_relative"].values()) < tol["observables"],
        "jacobians": max(res["jacobians_max_abs_over_kO"].values()) < tol["jacobians"],
        "quantiles": res["pdf"]["quantiles_max_relative"] < tol["quantiles"],
        "quantile_jacobians": res["pdf"]["quantile_jacobian_max_abs_over_Q"] < tol["quantile_jacobians"]}
    res["tolerances"] = tol
    res["builder_timing"] = B.timing
    res["passed_all"] = bool(all(res["passed"].values()))
    res["seconds"] = timer.mark("total")
    FC.write_json(FC.PRODUCTS / "validation_V.json", res)
    return res
    FC.log(FC.json.dumps(FC.plain({k: v for k, v in res.items() if k != "pdf"}), indent=1))
    FC.log(FC.json.dumps(FC.plain({k: v for k, v in res["pdf"].items() if not k.startswith("timing")}), indent=1))


if __name__ == "__main__":
    main()
