"""Parameter tables of the recommended J2 paintings (local, numpy only): post_j2/J2_recommended_parameters.json/.csv.

For every definition the recommended painting, its compact alternative and the best per-bin B16 painting
(post_j2/J2_summary.json: recommendations) are written as numbers a painter can use directly:
  B16 per bin   rho0 [E units], alpha, beta per calibrated bin (E(x) = rho0 q^-0.2 (1 + q^alpha)^-((beta-0.2)/alpha),
                q = x/0.5, x = r/R200c), from results_j2/evaluate_items.json (the parameters actually painted,
                including the (a) substitutes of (L*))
  ladder law    the law's parameters (G's names: ln amplitudes, mass slopes a_m (below M_cut) / a_m' (above), log10 M_cut,
                gamma) from results_j2/ladder_<D>.npz, and the per-bin profile parameters A [E units], x_c [R200c],
                beta, gamma of F(x) = A (x/x_c)^gamma (1 + x/x_c)^-beta evaluated at each bin's inclusion-weighted mean
                log10 M200c (as painted in j2_eval.py)
Check: the ladder profiles rebuilt here from the per-bin parameters equal the painted shells of evaluate.npz (exact
32-point Gauss-Legendre shell averages; max relative difference written to the json).
Units: M in Msun; R_cut is the 3D cut of the painting (the definition's boundary). z = 0 profiles; HalfDome applies the
published redshift response e_E(M, z, x) of J1 (J.2).
"""
from __future__ import annotations

import json

import j2_common as C
import numpy as np

R = C.J2_RESULTS
POST = C.HERE / ("post_j2_smoke" if C.SMOKE else "post_j2")
LN10 = np.log(10.0)
LEE22_E = {False: 13.61, True: 13.75}           # h^-1 Msun (g_selection.LEE22_MCUT, field E)
H_TNG = 0.6774


def law_params(names, theta, model, logm, lnc, frozen=None):
    th = dict(zip(names, theta))
    th.update(frozen or {})                     # PL1s: shape (xc_0, beta_0) frozen from G's first fit
    n = len(logm)
    lnMp = LN10 * (np.asarray(logm) - 14.5)
    if "log10Mcut" in th:
        mc = th["log10Mcut"]
    else:
        withc = any(k.endswith("_ac") for k in th)
        mc = LEE22_E[withc] - np.log10(H_TNG)
    lnMc = LN10 * (np.asarray(logm) - mc)
    out = {}
    for q in ("A", "xc", "beta"):
        if f"{q}_b0" in th:
            v = np.array([th[f"{q}_b{b}"] for b in range(n)])
        elif f"{q}_amp" in th:
            v = th[f"{q}_0"] + np.where(lnMc < 0, th[f"{q}_am"], th[f"{q}_amp"]) * lnMc
        elif f"{q}_am" in th:
            v = th[f"{q}_0"] + th[f"{q}_am"] * lnMp
        else:
            v = np.full(n, th[f"{q}_0"])
        if f"{q}_ac" in th:
            v = v + th[f"{q}_ac"] * (np.asarray(lnc) - np.log(10.0))
        out[q] = np.exp(v)
    if "g_am" in th:
        out["gamma"] = th["g0"] * np.exp(th["g_am"] * lnMp)
    elif "g0" in th:
        out["gamma"] = np.full(n, th["g0"])
    else:
        out["gamma"] = np.full(n, -0.3)
    out["log10Mcut"] = mc if any(k.endswith("_amp") for k in th) else None
    return out


def shell_avg(F, edges, order=32):
    t, wts = np.polynomial.legendre.leggauss(order)
    out = np.zeros(len(edges) - 1)
    for s in range(len(edges) - 1):
        a, b = edges[s], edges[s + 1]
        x = a + (b - a) * (t + 1) / 2
        out[s] = np.sum(F(x) * x**2 * wts) * (b - a) / 2 * 3 / (b**3 - a**3)
    return out


def main():
    summ = C.read_json(POST / "J2_summary.json")
    items = {it["label"]: it for it in C.read_json(R / "evaluate_items.json")["items"]}
    ev = C.load_npz(R / "evaluate.npz")
    inp = C.load_npz(C.J2_INPUT, keys=["logm200c", "weight", "mass_bin", "concentration_200c"])
    jb = C.J_BINS
    logm, lnc = [], []
    for b in jb:
        r = np.flatnonzero(inp["mass_bin"] == b)
        w = inp["weight"][r] / inp["weight"][r].sum()
        logm.append(float(w @ inp["logm200c"][r]))
        lnc.append(float(w @ np.log(inp["concentration_200c"][r])))
    X = np.asarray(ev["x_edges"], float)
    out, rows = {"definitions": __doc__, "bins": [C.bin_label(b) for b in jb], "bin_mean_log10_M200c": logm}, []
    for D in C.DEFS:
        rec = summ["recommendations"][D]
        for kind in ("primary", "compact", "best_B16"):
            lab = rec[kind]["label"]
            it = items[lab]
            ent = {"label": lab, "R_fit": it.get("R_fit"), "R_cut": it["R_cut"], "k": it["k"],
                   "DM_family_error_S1": rec[kind]["rms_DM"], "DM_family_error_profile_only": rec[kind]["prof_DM"]}
            if lab.startswith(("a|", "rc|")):
                P = np.asarray(it["params"], float)
                ent.update(form="B16", per_bin={"rho0": P[:, 0], "alpha": P[:, 1], "beta": P[:, 2]},
                           a_substituted_bins=it.get("a_substituted_bins"),
                           rc_fallback_to_S400_bins=it.get("rc_fallback_to_S400_bins"))
                for c, b in enumerate(jb):
                    rows.append({"D": D, "kind": kind, "label": lab, "bin": C.bin_label(b), "form": "B16",
                                 "rho0_or_A": P[c, 0], "alpha_or_xc": P[c, 1], "beta": P[c, 2], "gamma": None})
            else:
                _, model, dset, rf = lab.split("|")[1:]
                res = json.loads(str(np.load(R / f"ladder_{D}.npz")["results_json"]))
                r = [x for x in res if x["model"] == model and x["dset"] == dset
                     and abs(float(x["R_fit"]) - float(rf[2:])) < 1e-9][0]
                names, theta = r["names"], r["fits"]["floor"]["theta"]
                frozen = r.get("frozen_shape")
                pp = law_params(names, theta, model, logm, lnc, frozen)
                M4 = np.asarray(ev[f"M4__{lab}"], float)
                worst = 0.0 if model.endswith("+O0") else None      # O1/O2: the painted profile adds an outer term
                for c in range(len(jb)):
                    A, xc, be, ga = pp["A"][c], pp["xc"][c], pp["beta"][c], pp["gamma"][c]
                    if worst is not None:
                        S = shell_avg(lambda x: A * (x / xc)**ga * (1 + x / xc)**(-be), X)
                        worst = max(worst, float(np.max(np.abs(S / M4[c] - 1))))
                    rows.append({"D": D, "kind": kind, "label": lab, "bin": C.bin_label(jb[c]), "form": "gNFW law",
                                 "rho0_or_A": A, "alpha_or_xc": xc, "beta": be, "gamma": ga})
                ent.update(form=f"Lee22 ladder law {model} ({dset} cells)",
                           law_parameters=dict(zip(names, theta)) | dict(frozen or {}),
                           frozen_parameters=sorted(frozen or {}),
                           log10Mcut=pp["log10Mcut"],
                           per_bin={"A": pp["A"], "x_c": pp["xc"], "beta": pp["beta"], "gamma": pp["gamma"]},
                           rebuild_vs_painted_max_rel=worst)
                if not model.endswith("+O0"):
                    th = dict(zip(names, theta))
                    lnMp = LN10 * (np.asarray(logm) - 14.5)
                    ent["outer_option"] = model.split("+")[1]
                    ent["outer_note"] = (
                        "painted profile = F_in(x) f_tr(x) + E_cos [1 + T2h_c(x)], F_in from the per-bin A, x_c, beta, "
                        "gamma; f_tr = [1 + (x/x_t)^4]^-1.5 for O2 (1 for O1); E_cos = G's ebar (cosmic mean electron "
                        "density, E units); T2h_c = the bin's inclusion-weighted linear two-halo template b_T(M) "
                        "xi_lin(x R200c) (g_physics.two_halo_template).  Not representable by the inner parameters "
                        "alone: the exact painted shell averages are given in painted_shells_x200 (67 X200 shells "
                        "to 4 R200c, E units).")
                    if "ln_xt0" in th:
                        ent["per_bin"]["x_t"] = np.exp(th["ln_xt0"] + th["a_xt"] * lnMp)
                    ent["painted_shells_x200"] = M4
                    ent["x200_edges"] = X
            out.setdefault(D, {})[kind] = ent
    C.write_json(POST / "J2_recommended_parameters.json", out)
    C.write_csv(POST / "J2_recommended_parameters.csv", rows)
    print({D: {k: v.get("rebuild_vs_painted_max_rel") for k, v in out[D].items()} for D in C.DEFS})


if __name__ == "__main__":
    main()
