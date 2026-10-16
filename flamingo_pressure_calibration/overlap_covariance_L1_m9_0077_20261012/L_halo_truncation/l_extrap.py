"""L stage 'extrap' (local, stack level; added 2026-10-16 after the user's clarification): the region the painting
extrapolates into.

User clarification (2026-10-16): the truncation radius only decides WHICH DATA constrain the halo-profile fit (all gas
inside R_t, as in observations); HalfDome still paints the fitted analytic law out to 4 R200c.  The part of each painted
quantity that comes from R_t - 4 R200c is therefore set by the law's extrapolation, not by data.  This stage measures,
per mass bin and field, how large that part is and what it would be under three references:
    upper   the painted A2-ii gas (Test J's D0) beyond R_t,
    lower   the host-bound gas only (Test J's D3) beyond R_t (variant: the nested bound-and-A2-ii upper bracket),
    pub     the published B12 (pressure) / B16 (electrons) law that HalfDome paints now, with its shape only (rescaled to
            the same content inside R_t, as a law fitted inside R_t would be).
Inside R_t every reference is the measured all-gas (= A2-ii, G2) stack.

Stack-level quantities (x = r / R200c; X_q(<R) = sum_s V_s S_q,s with the shell-constant partial last shell,
l_common.cumulative_at; S_q the inclusion-weighted stack of quantity q, rows = sample + 200 paired halo resamples):
    rho_q = [X_q(<4) - X_q(<R)] / X_in,   X_in = X_a2ii(<R)            (q = a2ii: upper; hb: lower; nest_hi: variant)
    rho_pub = [X_pub(<4) - X_pub(<R)] / X_pub(<R)                      (published law, shape only)
    s_q = rho_q / (1 + rho_q)  = share of the 4R-painted content (Y or N_e) that comes from R - 4 R200c when the inside
                                 is matched to the data
    t_pub = (rho_pub - rho_hb) / (rho_a2ii - rho_hb)   position of the published shape in the bracket (0 = bound only,
                                 1 = all A2-ii gas; inside the bracket iff 0 <= t_pub <= 1)
    n_pub = X_pub(<R) / X_in   the published law's own normalisation inside R (not a shape quantity)
R = the common global fit radius x_t (global.json: global.B32.common.R) and, as a variant, each bin's own B32 radius
(per bootstrap row).  The published shell averages are exact: F_s = 3 [C(x_s+1) - C(x_s)] / (x_s+1^3 - x_s^3) with
C(x) = Int_0^x F x'^2 dx' in closed form (K_soap_battaglia/k_fit_core.cumulative, published laws at z = 0, each halo at
its own M200c; identical to the engine's published_profile, gate G11).

Observable level (exact for the linear terms, Test J's per-bin own curves; x_t common):
    D0 share beyond x_t:  1 - T_O(x_t) for every additive term O (the part of the 4R value that depends on R > x_t if
                          the extrapolation followed all A2-ii gas);
    mean DM bracket:      lower reference L = fb + sum_b [own_D0,b(x_t) + own_D3,b(4) - own_D3,b(x_t)] (all gas inside
                          x_t, bound gas beyond); share beyond x_t = sum_b [own_D3,b(4) - own_D3,b(x_t)] / L; L / D0(4R).
                          fb = the fallback haloes' term (unchanged).  Quadratic terms (variance, C_ell) need the exact
                          painting of the composite profile (Test L2).
    yy1h fallback (added 2026-10-16, review round 2): J stores no trunc_all__D0__yy1h, so the tSZ 1-halo shares above
                          carry no fallback term (fb = 0: shares of the calibrated haloes' yy1h only).  With the fallback
                          (unchanged, not cut) the share beyond x_t is share_nofb x (1 - phi), phi = fallback share of
                          yy1h(4R).  phi is bracketed (ESTIMATE):
                            phi_lo = Test H's importance share of the >= 10^15.5 row (H results/sens_summary.json:
                                     shares['a2ii__yy1h@<ell>']['>=15.5']; H's fallback also covers 15.388-15.5, which
                                     sits in H's 15.3-15.5 row, so this is a lower bound; C10 configuration);
                            phi_hi = Y0 / (Y0 + sum_b own_b(4)), Y0 = J's trunc_all__D0__yytot[0, ell] = the total
                                     (1h + 2h) C_ell^yy of the fallback haloes alone (an upper bound on their 1h term).
Output: results/extrap.json, results/extrap.npz, tables/L1_extrapolation.csv.
"""
from __future__ import annotations

import csv
import sys

import l_common as LC
import l_global as LG
import l_stacks as LS
import numpy as np

if str(LC.STUDY / "K_soap_battaglia") not in sys.path:
    sys.path.append(str(LC.STUDY / "K_soap_battaglia"))
import k_fit_core as K  # noqa: E402

FORM = {"P": "B12", "E": "B16"}
REFS = ("a2ii", "hb", "nest_hi", "all")
PLOT_BINS = (3, 10)                       # 13.1-13.3 and 14.5-14.7 (also the smoke bins)


def published_shells(field, m200c):
    """Exact shell averages of the published law at z = 0 for haloes of mass m200c [n, 67] (P/P200 or E units)."""
    form = FORM[field]
    ph = K.published_phi(form, np.log(np.asarray(m200c, float)), 0.0)
    xe = LC.X[1:][None, :]
    C = K.cumulative(xe, form, *(np.asarray(ph[q], float)[:, None] for q in K.QNAMES))
    C = np.c_[np.zeros(len(m200c)), C]                                  # C(0) = 0
    return 3.0 * np.diff(C, axis=1) / np.diff(LC.X ** 3)[None, :]


def cum_rows(S, R):
    """X(<R_r) for every row r of S [R, 67] (shell-constant inside the shell holding R_r)."""
    S = np.atleast_2d(np.asarray(S, float))
    R = np.clip(np.broadcast_to(np.asarray(R, float), (S.shape[0],)), 0.0, LC.X[-1])
    cum = np.c_[np.zeros(S.shape[0]), np.cumsum(S * LC.VOL[None, :], axis=1)]
    k = np.minimum(np.searchsorted(LC.X, R, side="right") - 1, LC.N_X - 1)
    r = np.arange(S.shape[0])
    return cum[r, k] + S[r, k] * 4 * np.pi / 3 * (R ** 3 - LC.X[k] ** 3)


def bin_quantities(b, field, R):
    st = LS.bin_stacks(b)
    S = st["S"]
    rows = np.asarray(LS.bins()[b]["rows"])
    Wd = LC.boot_matrix(LS.bins()[b])
    Fp = published_shells(field, LS.halo_table()["m200c"][rows])
    Sp = LC.wstack(Wd, Fp)
    q = {}
    with np.errstate(invalid="ignore", divide="ignore"):         # R = 4 (gate G11) gives 0/0 in the ratios
        X_in = cum_rows(S[f"{field}_a2ii"], R)
        X4 = {}
        for ref in REFS:
            Sref = S[f"{field}_{ref}"]
            X4[ref] = cum_rows(Sref, 4.0)
            q[f"rho_{ref}"] = (X4[ref] - cum_rows(Sref, R)) / X_in
            q[f"s_{ref}"] = q[f"rho_{ref}"] / (1 + q[f"rho_{ref}"])
        Xp_in = cum_rows(Sp, R)
        Xp4 = cum_rows(Sp, 4.0)
        q["rho_pub"] = (Xp4 - Xp_in) / Xp_in
        q["s_pub"] = q["rho_pub"] / (1 + q["rho_pub"])
        q["n_pub"] = Xp_in / X_in
        q["pub_out_over_a2ii_out"] = (Xp4 - Xp_in) / (X4["a2ii"] - cum_rows(S[f"{field}_a2ii"], R))
        q["t_pub"] = (q["rho_pub"] - q["rho_hb"]) / (q["rho_a2ii"] - q["rho_hb"])
        q["t_pub_all"] = (q["rho_pub"] - q["rho_hb"]) / (q["rho_all"] - q["rho_hb"])
        q["hb_over_a2ii_out"] = q["rho_hb"] / q["rho_a2ii"]
    return q, Sp


def summarise(q):
    out = {}
    for k, v in q.items():
        out[k] = LC.boot_summary(v[1:], float(v[0]))
    t = q["t_pub"]
    out["pub_in_bracket"] = bool(0.0 <= t[0] <= 1.0)
    out["pub_in_bracket_fraction_boot"] = float(np.mean((t[1:] >= 0) & (t[1:] <= 1)))
    out["pub_below_bound_fraction_boot"] = float(np.mean(t[1:] < 0))
    out["pub_above_a2ii_fraction_boot"] = float(np.mean(t[1:] > 1))
    ta = q["t_pub_all"]
    out["pub_in_bracket_all"] = bool(0.0 <= ta[0] <= 1.0)
    out["pub_in_bracket_all_fraction_boot"] = float(np.mean((ta[1:] >= 0) & (ta[1:] <= 1)))
    return out


def yy1h_fallback_estimate(sens, x_t):
    """Bracket of the fallback haloes' share phi of C_ell^yy 1h and the share beyond x_t with that term (ESTIMATE; see
    the module docstring)."""
    hs = LC.jload(LC.STUDY / "H_sensitivity" / "results" / "sens_summary.json")["shares"]
    yt = np.asarray(sens["trunc_all__D0__yytot"], float)
    out = {"definition": yy1h_fallback_estimate.__doc__ + "  phi_lo: Test H's >= 10^15.5 importance share (lower "
           "bound); phi_hi: J's fallback-only C_ell^yy (1h + 2h) / (that + sum_b own_b(4)) (upper bound); "
           "share_beyond_with_fallback = share_beyond_no_fallback x (1 - phi)"}
    for ell in (100, 1000, 3000):
        li = int(np.argmin(np.abs(sens["ell"] - ell)))
        c = LG.own_curve(sens, "D0", "yy1h", ell)
        own4 = float(c[:, -1].sum())
        ins = float(sum(LG.own_at(c[k], x_t) for k in range(len(c))))
        s0 = 1 - ins / own4
        phi_lo = float(hs[f"a2ii__yy1h@{ell}"][">=15.5"])
        phi_hi = float(yt[0, li] / (yt[0, li] + own4))
        out[str(ell)] = {"phi_lo": phi_lo, "phi_hi": phi_hi, "share_beyond_no_fallback": float(s0),
                         "share_beyond_with_fallback_lo": float(s0 * (1 - phi_hi)),
                         "share_beyond_with_fallback_hi": float(s0 * (1 - phi_lo))}
    return out


def observable_level(x_t, rad):
    """Exact 1-halo / linear terms (Test J's own curves): D0 dependence on R > x_t and the mean-DM bracket."""
    sens = np.load(LC.J_DIR / "results" / "sens.npz")
    res = {"x_t": x_t, "D0_share_beyond_xt": {}, "mean_dm_bracket": {}}
    for f, obs in LG.OWN_OBS.items():
        for name, ell in obs:
            tag = f"{name}@{ell}" if ell else name
            c = LG.own_curve(sens, "D0", name, ell)
            fb = 0.0
            if name in ("ymean", "dm_mean", "dm_var", "ksz1h", "dmcl1h"):
                al = np.asarray(sens[f"trunc_all__D0__{name}"], float)
                fb = float(al[0, int(np.argmin(np.abs(sens["ell"] - ell)))]) if al.ndim == 2 else float(al[0])
            ins = sum(LG.own_at(c[k], x_t) for k in range(len(c)))
            res["D0_share_beyond_xt"][tag] = float(1 - (fb + ins) / (fb + c[:, -1].sum()))
    # mean DM: all gas inside x_t, host-bound gas beyond (exact, linear in the profile)
    c0 = LG.own_curve(sens, "D0", "dm_mean", None)
    c3 = LG.own_curve(sens, "D3", "dm_mean", None)
    fb0 = float(np.asarray(sens["trunc_all__D0__dm_mean"], float)[0])
    fb3 = float(np.asarray(sens["trunc_all__D3__dm_mean"], float)[0])
    labs = [LC.LABELS[b] for b in LC.J_BINS]
    for which in ("global", "perbin"):
        if which == "global":
            Rb = np.full(len(labs), x_t)
        else:
            if not all(lab in rad["bins"] for lab in labs):
                continue
            Rb = np.array([rad["bins"][lab]["rules"]["B32"]["E"]["value"] for lab in labs])
        ins0 = np.array([LG.own_at(c0[k], Rb[k]) for k in range(len(labs))])
        ins3 = np.array([LG.own_at(c3[k], Rb[k]) for k in range(len(labs))])
        out3 = c3[:, -1] - ins3
        D0 = fb0 + c0[:, -1].sum()
        L = fb0 + ins0.sum() + out3.sum()
        U = D0
        res["mean_dm_bracket"][which] = {
            "R": Rb.tolist(),
            "upper_over_D0": 1.0, "lower_over_D0": float(L / D0),
            "inside_over_D0": float((fb0 + ins0.sum()) / D0),
            "share_beyond_upper": float(1 - (fb0 + ins0.sum()) / U),
            "share_beyond_lower": float(out3.sum() / L),
            "D3_over_D0": float((fb3 + c3[:, -1].sum()) / D0),
            "lower_over_D3": float(L / (fb3 + c3[:, -1].sum())),
            "per_bin": {lab: {"share_beyond_upper": float((c0[k, -1] - ins0[k]) / c0[k, -1]),
                              "share_beyond_lower": float(out3[k] / (ins0[k] + out3[k])),
                              "inside_D3_over_inside_D0": float(ins3[k] / ins0[k])}
                        for k, lab in enumerate(labs)}}
    res["fallback"] = {"D0": fb0, "D3": fb3}
    res["yy1h_fallback_ESTIMATE"] = yy1h_fallback_estimate(sens, x_t)
    res["definition"] = ("D0_share_beyond_xt: 1 - O(every calibrated halo cut at x_t)/O(4R) for the painted A2-ii "
                         "target (the part of the 4R value that depends on R > x_t if the extrapolation followed all "
                         "A2-ii gas); mean_dm_bracket: lower = all gas inside, host-bound beyond (D0 inside + D3 "
                         "beyond), upper = D0; share_beyond = beyond part / that reference's total")
    return res


def main():
    t0 = LC.time.perf_counter()
    rad = LC.jload(LC.RESULTS / "radii.json")
    glob = LC.jload(LC.RESULTS / "global.json")
    rnpz = np.load(LC.RESULTS / "radii.npz")
    x_t = float(glob["global"]["B32"]["common"]["R"])
    out = {"definitions": __doc__, "x_t": x_t, "bins": {}, "family_weighted": {}}
    arr = {}
    rows_csv = []
    for b in LC.BINS:
        lab = LC.LABELS[b]
        out["bins"][lab] = {"mean_logm200c": rad["bins"][lab]["mean_logm200c"]}
        for f in ("P", "E"):
            for which in ("global", "perbin"):
                R = x_t if which == "global" else np.asarray(rnpz[f"R__B32__{f}__{b}"], float)
                q, Sp = bin_quantities(b, f, R)
                s = summarise(q)
                out["bins"][lab].setdefault(f, {})[which] = s
                for k in ("s_a2ii", "s_hb", "s_nest_hi", "s_all", "s_pub", "t_pub", "n_pub"):
                    arr[f"{k}__{f}__{which}__{b}"] = q[k].astype(np.float32)
                if which == "global":
                    arr[f"Spub__{f}__{b}"] = Sp[0]
                    st = LS.bin_stacks(b)["S"]
                    for ref in REFS:
                        arr[f"S{ref}__{f}__{b}"] = st[f"{f}_{ref}"][0]
                rows_csv.append([lab, f, which, f"{(x_t if which == 'global' else float(R[0])):.4f}"]
                                + [f"{s[k]['value']:.5g}" for k in ("s_a2ii", "s_all", "s_hb", "s_nest_hi", "s_pub",
                                                                     "t_pub", "t_pub_all", "n_pub",
                                                                     "pub_out_over_a2ii_out")]
                                + [s["pub_in_bracket"], f"{s['pub_in_bracket_fraction_boot']:.3f}",
                                   s["pub_in_bracket_all"]])
    # family-weighted summaries over the bins (D0 family importance weights, as for the global radius)
    labs_ok = [LC.LABELS[b] for b in LC.BINS]
    if len(labs_ok) == len(LC.J_BINS):
        for f in ("P", "E"):
            w = np.array([glob["weights"][f"{f}:family"][lab] for lab in labs_ok])
            w = w / w.sum()
            for which in ("global", "perbin"):
                d = {}
                for k in ("s_a2ii", "s_hb", "s_nest_hi", "s_pub"):
                    vals = np.array([arr[f"{k}__{f}__{which}__{b}"] for b in LC.BINS], float)     # [nb, rows]
                    d[k] = LC.boot_summary(w @ vals[:, 1:], float(w @ vals[:, 0]))
                inb = [out["bins"][lab][f][which]["pub_in_bracket"] for lab in labs_ok]
                tp = [out["bins"][lab][f][which]["t_pub"]["value"] for lab in labs_ok]
                d["n_bins_pub_in_bracket"] = int(sum(inb))
                d["n_bins_pub_in_bracket_all"] = int(sum(out["bins"][lab][f][which]["pub_in_bracket_all"]
                                                         for lab in labs_ok))
                d["n_bins"] = len(inb)
                d["bins_pub_below_bound"] = [lab for lab, t in zip(labs_ok, tp) if t < 0]
                d["bins_pub_above_a2ii"] = [lab for lab, t in zip(labs_ok, tp) if t > 1]
                d["t_pub_range"] = [float(min(tp)), float(max(tp))]
                d["weight_in_bracket"] = float(sum(wi for wi, ok in zip(w, inb) if ok))
                for k in ("s_a2ii", "s_hb", "s_all", "s_pub"):
                    v = [out["bins"][lab][f][which][k]["value"] for lab in labs_ok]
                    d[f"{k}_range"] = [float(min(v)), float(max(v))]
                hi = [lab for lab in labs_ok if float(lab.split("-")[0]) >= 13.5 - 1e-9]
                lo = [lab for lab in labs_ok if float(lab.split("-")[0]) < 13.5 - 1e-9]
                for nm, sel in (("ge13.5", hi), ("lt13.5", lo)):
                    tv = [out["bins"][lab][f][which]["t_pub"]["value"] for lab in sel]
                    d[f"t_pub_range_{nm}"] = [float(min(tv)), float(max(tv))]
                out["family_weighted"].setdefault(f, {})[which] = d
        out["observable_level"] = observable_level(x_t, rad)
    # what the published B12/B16 HalfDome maps get from beyond x_t (Codex shell cross-spectra; linear in R between the
    # 1.0 and 1.5 edges, ESTIMATE)
    cx = LC.jload(LC.RESULTS / "codex.json") if (LC.RESULTS / "codex.json").exists() else None
    if cx:
        sh = np.array([0.0, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
        sens = np.load(LC.J_DIR / "results" / "sens.npz")
        hd = {}
        for fld, o in (("tsz", "yytot"), ("ksz", "ksztot"), ("dm", "dmcltot")):
            hd[fld] = {}
            tr = np.asarray(sens[f"trunc_all__D0__{o}"], float)                  # [68 fine edges, ell]
            for band in ("77", "325", "1025", "3025"):
                T = cx["halfdome_maps"][fld][band]["T_at_edges"]
                eng = {"77": 100, "325": 300, "1025": 1000, "3025": 3000}[band]
                li = int(np.argmin(np.abs(sens["ell"] - eng)))
                Te = float(np.interp(x_t, LC.X, tr[:, li]) / tr[-1, li])
                hd[fld][band] = {"ell_centre": cx["halfdome_maps"][fld][band]["ell_centre"],
                                 "halfdome_B12B16_share_beyond_xt_ESTIMATE": float(1 - np.interp(x_t, sh, T)),
                                 "engine_D0_share_beyond_xt": float(1 - Te), "engine_ell": eng}
        out["maps_share_beyond_xt"] = {"values": hd, "definition": (
            "1 - C_ell(every halo cut at x_t) / C_ell(4R): the part of the 4R-painted map power that depends on the "
            "region beyond x_t.  HalfDome = the published B12/B16 painting (Codex shell cross-spectra; linear in R "
            "between the 1.0 and 1.5 R200c shell edges, ESTIMATE); engine = FLAMINGO A2-ii stacks (Test J D0, C14) "
            "on the same lightcone catalogue, Test J's trunc_all curves (linear in R between fine edges)")}
    # gate inputs (checked in verify_l G11)
    g = {}
    m = LS.halo_table()["m200c"][:5]
    for fi, f in enumerate(("P", "E")):
        Fk = published_shells(f, m)
        gt, gw = np.polynomial.legendre.leggauss(48)
        dev = 0.0
        for s in (45, 55, 63):                                         # shells near 0.7, 1.4 and 3.0 R200c
            a, c_ = LC.X[s], LC.X[s + 1]
            xx = 0.5 * (c_ - a) * gt + 0.5 * (c_ + a)
            vals = LC.JC.A.published_profile(fi, m, 0.0, xx) * xx[None, :] ** 2
            avg = 3 * (vals @ (0.5 * (c_ - a) * gw)) / (c_ ** 3 - a ** 3)
            dev = max(dev, float(np.max(np.abs(Fk[:, s] / avg - 1))))
        g[f"{f}_published_shell_avg_vs_engine_quadrature_max_rel"] = dev
    b0 = LC.BINS[0]
    q4, _ = bin_quantities(b0, "E", 4.0)
    g["shares_at_R4_max_abs"] = float(max(np.nanmax(np.abs(q4[k])) for k in ("s_a2ii", "s_hb", "s_pub")))
    if "observable_level" in out:
        sens = np.load(LC.J_DIR / "results" / "sens.npz")
        c3 = LG.own_curve(sens, "D3", "dm_mean", None)
        al3 = np.asarray(sens["trunc_all__D3__dm_mean"], float)
        g["D3_fallback_plus_own_at_4R_vs_trunc_all_rel"] = float(abs((al3[0] + c3[:, -1].sum()) / al3[-1] - 1))
        ol = out["observable_level"]["mean_dm_bracket"]["global"]
        g["lower_between_D3_and_D0"] = bool(ol["D3_over_D0"] - 1e-12 <= ol["lower_over_D0"] <= 1 + 1e-12)
    out["gates"] = g
    out["seconds"] = LC.time.perf_counter() - t0
    LC.jdump(out, LC.RESULTS / "extrap.json")
    arr["bins"] = np.array(LC.BINS)
    arr["x_t"] = np.array(x_t)
    np.savez_compressed(LC.RESULTS / "extrap.npz", **arr)
    with open(LC.TABLES / "L1_extrapolation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bin", "field", "fit_radius", "R_fit_R200c", "s_upper_a2ii", "s_all_gas", "s_lower_hostbound",
                    "s_lower_nested_hi", "s_published_shape", "t_published", "t_published_vs_all_gas",
                    "published_norm_inside", "published_out_over_a2ii_out", "published_in_bracket_hb_a2ii",
                    "in_bracket_fraction_boot", "published_in_bracket_hb_all"])
        w.writerows(rows_csv)
    LC.log(f"extrap done {out['seconds']:.1f} s", g)


if __name__ == "__main__":
    main()
