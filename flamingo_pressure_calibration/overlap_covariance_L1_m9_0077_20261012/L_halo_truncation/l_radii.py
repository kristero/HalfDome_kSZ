"""L stage 'radii' (local): truncation radii R_t of every rule, per mass bin and field, with paired-bootstrap ranges.

Rules (x = r/R200c; every rule is evaluated on the sample (row 0) and on each of the 200 paired halo resamples):
  Bound-fraction thresholds, f(x) = stack of the bound gas / stack of the reference gas in each (merged) shell:
    B32      f_hb/all(x) falls to 0.32 for good (PRIMARY: host-bound over all gas, nested since hb is a subset of all)
    B32u     f_hb/a2ii = 0.32 (upper bound of the nested (hb AND a2ii)/a2ii)
    B32lo/hi the nested bracket max(0, hb - p10)/a2ii and min(hb, a2iia2i)/a2ii
    B50      f_hb/all = 0.50;   B32cb / B50cb: central-bound only, cb/all
    (crossing: l_common.crossing_down, last downward crossing at x >= 0.3, linear in f vs ln x between shell radii)
  Cumulative bound gas (variant reading 'C68' and the bound-component convergence radii):
    C68 / C90 / C95  X_hb(<R) = 0.68 / 0.90 / 0.95 X_hb(<4R), X(<R) = sum_s V_s F_s, shell-constant inversion
  Budget matching (alternative): the painted A2-ii gas inside R equals the whole bound component inside 4 R200c,
    Req      X_a2ii(<R) = X_hb(<4R)        (Req_all: X_all(<R) = X_hb(<4R); Req_nest: with the nested bracket)
  Outer edge from the profile shape (alternative):
    SB       the steepest logarithmic slope d ln F / d ln x of the stacked all-gas profile in [max(0.7, x_h), 3.6] R200c
             (F = P for pressure, E for electrons; SB_G: gas mass density; SB_a2ii: the painted A2-ii profiles),
             local quadratic regression in ln x with a Gaussian kernel, sigma = 0.15 (0.10 and 0.25 as checks)
    SH_K0    the entropy maximum, d ln K / d ln x = 0 (first downward zero crossing in [max(1.0, x_h), 3.6])
    SH_TM / SH_K   the steepest drop of the mass-weighted temperature / entropy in [max(1.0, x_h), 3.6] R200c
             (shock signature; a minimum at the range edge = not detected)
  Two-halo floor (alternative): ratio of the stacked all-gas profile to the stacked linear field expectation,
    2H2      F_all / F_2h falls to 2 (the halo excess F_all - F_2h equals the correlated field F_2h)
    2H1      F_all / F_2h falls to 1 (the profile meets the field expectation);  2H2_a2ii: the painted A2-ii profile
  Baryon budget (closure-radius proxy): CLg = the radius where the all-gas mass inside R reaches f_b M200c
    (sum_s V_s G_s = 800 pi/3); stars ignored (pushes CLg out) and M_tot(<R) > M200c beyond R200c ignored (pulls it
    in) relative to the closure radius R_c of Ayromlou, Nelson & Pillepich (2023, MNRAS 524, 5391); ESTIMATE.
  Reference (not measured here): MDK15 = the dark-matter splashback radius from More, Diemer & Kravtsov (2015, ApJ
    810, 36) Eq. (7), R_sp/R200m = 0.81 (1 + 0.97 exp(-nu200m/2.44)), per halo (ESTIMATE).
Unit conversions per row: R in R200m units = R x <R200c/R200m>_w, in R500c units = R x <R200c/R500c>_w (each halo's
own SOAP radii, inclusion-weighted over the bin's haloes and over the same resamples).
Inside fractions (per row): the share of the painted A2-ii stack's Y(<4R) (pressure) or N_e(<4R) (electrons) inside R_t,
and the share of the bound component's X_hb(<4R) inside R_t.
Output: results/radii.npz (rule x field x bin x row), results/radii.json (summaries), tables/L1_radii.csv.
"""
from __future__ import annotations

import csv

import l_common as LC
import l_stacks as LS
import numpy as np

B_RULES = {"B32": ("hb/all", 0.32), "B32u": ("hb/a2ii", 0.32), "B32lo": ("hbnest_lo/a2ii", 0.32),
           "B32hi": ("hbnest_hi/a2ii", 0.32), "B50": ("hb/all", 0.50), "B32cb": ("cb/all", 0.32),
           "B50cb": ("cb/all", 0.50), "B32cbu": ("cb/a2ii", 0.32)}
C_RULES = {"C68": 0.68, "C90": 0.90, "C95": 0.95}
SB_RULES = {"SB": "{f}_all", "SB_a2ii": "{f}_a2ii", "SB_G": "G_all"}
SH_RULES = {"SH_TM": "T_M", "SH_K": "K"}          # + SH_K0 (entropy maximum), computed with SH_K
TWOH_RULES = {"2H2": ("all", 2.0), "2H1": ("all", 1.0), "2H2_a2ii": ("a2ii", 2.0), "2H1_a2ii": ("a2ii", 1.0)}
RULE_ORDER = (list(B_RULES) + list(C_RULES) + ["Req", "Req_all", "Req_nest_lo", "Req_nest_hi"] + list(SB_RULES)
              + list(SH_RULES) + ["SH_K0"] + list(TWOH_RULES) + ["MDK15", "CLg"] + [f"SB_s{int(100 * s)}" for s in LC.SLOPE_SIGMAS])
DESCRIPTION = {
    "B32": "host-bound / all gas falls to 0.32 (primary)",
    "B32u": "host-bound / A2-ii falls to 0.32 (upper bound of the nested fraction)",
    "B32lo": "nested lower bracket max(0, hb - p10) / A2-ii falls to 0.32",
    "B32hi": "nested upper bracket min(hb, A2-ii and A2-i) / A2-ii falls to 0.32",
    "B50": "host-bound / all gas falls to 0.50",
    "B32cb": "central-bound / all gas falls to 0.32",
    "B50cb": "central-bound / all gas falls to 0.50",
    "B32cbu": "central-bound / A2-ii falls to 0.32",
    "C68": "68% of the host-bound gas inside 4R lies inside R (variant reading of '1 sigma of bound gas')",
    "C90": "90% of the host-bound gas inside 4R (bound-component convergence, stack level)",
    "C95": "95% of the host-bound gas inside 4R (bound-component convergence, stack level)",
    "Req": "A2-ii gas inside R equals the host-bound gas inside 4R (budget matching)",
    "Req_all": "all gas inside R equals the host-bound gas inside 4R",
    "Req_nest_lo": "A2-ii gas inside R equals the nested lower bracket of the bound gas inside 4R",
    "Req_nest_hi": "A2-ii gas inside R equals the nested upper bracket of the bound gas inside 4R",
    "SB": "steepest log slope of the all-gas profile of the field (P or E), sigma_ln x = 0.15",
    "SB_a2ii": "steepest log slope of the A2-ii profile of the field",
    "SB_G": "steepest log slope of the all-gas mass density (field-independent)",
    "SH_TM": "steepest drop of the mass-weighted temperature (shock signature; field-independent)",
    "SH_K": "steepest drop of the entropy K (shock signature; field-independent)",
    "SH_K0": "entropy maximum: d ln K / d ln x falls to 0 (inner edge of the stacked shock region; field-independent)",
    "2H2": "all-gas profile / linear field expectation falls to 2 (halo excess = field)",
    "CLg": "gas-only baryon budget: all gas inside R = f_b M200c (closure-radius proxy, field-independent; ESTIMATE)",
    "MDK15": "reference: DM splashback radius, More, Diemer & Kravtsov 2015 Eq. 7 in nu200m (not a gas measurement)",
    "2H1": "all-gas profile / linear field expectation falls to 1",
    "2H2_a2ii": "A2-ii profile / linear field expectation falls to 2",
    "2H1_a2ii": "A2-ii profile / linear field expectation falls to 1",
}
for _s in LC.SLOPE_SIGMAS:
    DESCRIPTION[f"SB_s{int(100 * _s)}"] = f"as SB with sigma_ln x = {_s}"
SLOPE_GRID = np.exp(np.linspace(np.log(0.5), np.log(3.9), 160))


def slope_operator(xr, sigma, grid=SLOPE_GRID):
    """Linear map L [grid, points]: slope(grid) = L @ ln F(points) (local quadratic regression, Gaussian kernel)."""
    u = np.log(np.asarray(xr, float))
    L = np.full((len(grid), len(u)), np.nan)
    for i, u0 in enumerate(np.log(grid)):
        k = np.exp(-0.5 * ((u - u0) / sigma) ** 2)
        if np.sum(k > 1e-3 * k.max()) < 5:
            continue
        A = np.vstack([np.ones_like(u), u - u0, (u - u0) ** 2]).T
        Aw = A * k[:, None]
        M = A.T @ Aw
        if np.linalg.cond(M) > 1e12:
            continue
        L[i] = np.linalg.solve(M, Aw.T)[1]
    return L


def slopes(vals, xr, sigma):
    """d ln F / d ln x on SLOPE_GRID for every row of vals [R, G] (rows with non-positive values: per-row fallback)."""
    L = slope_operator(xr, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.log(vals)
    good = np.all(np.isfinite(lv), axis=1)
    out = np.full((vals.shape[0], len(SLOPE_GRID)), np.nan)
    if good.any():
        out[good] = lv[good] @ L.T
    for r in np.flatnonzero(~good):
        ok = np.isfinite(lv[r])
        if ok.sum() >= 6:
            out[r] = lv[r, ok] @ slope_operator(xr[ok], sigma).T
    return out


def row_radii_crossing(fr, xr, thr):
    res = [LC.crossing_down(xr, fr[r], thr) for r in range(fr.shape[0])]
    R = np.array([q["R"] for q in res])
    st = np.array([q["status"] for q in res])
    Rf = np.array([q["R_first"] for q in res])
    nc = np.array([q["n_cross"] for q in res])
    return R, st, Rf, nc


def evaluate_bin(b):
    st = LS.bin_stacks(b)
    S, g, meta = st["S"], st["groups"], st["meta"]
    nrow = S["E_all"].shape[0]
    out = {}                  # (rule, field) -> dict(R [rows], status, extra)
    # slope searches start outside the hydrodynamic-resolution radius x_h (shape unresolved inside it)
    lo_sb = max(LC.SLOPE_RANGE[0], meta["x_h"])
    lo_sh = max(LC.SHOCK_RANGE[0], meta["x_h"])
    for field in ("P", "E"):
        # ---- bound-fraction thresholds
        for rule, (fname, thr) in B_RULES.items():
            fr, xr, _ = LS.fraction(b, field, fname)
            R, stt, Rf, nc = row_radii_crossing(fr, xr, thr)
            out[(rule, field)] = dict(R=R, status=stt, R_first=Rf, n_cross=nc)
        # ---- cumulative bound gas (merged with the hb groups)
        Fhb = LC.merged_shells(S[f"{field}_hb"], g["hb"])
        for rule, lev in C_RULES.items():
            R = np.array([LC.cumulative_radius(Fhb[r], lev) for r in range(nrow)])
            out[(rule, field)] = dict(R=R, status=np.where(np.isfinite(R), "ok", "nan"))
        # ---- budget matching
        Fa = LC.merged_shells(S[f"{field}_a2ii"], g["a2ii"])
        Fall = LC.merged_shells(S[f"{field}_all"], g["all"])
        tot_hb = np.array([LC.cumulative_at(Fhb[r], 4.0) for r in range(nrow)])
        for rule, Fp, tot in (("Req", Fa, tot_hb), ("Req_all", Fall, tot_hb),
                              ("Req_nest_lo", Fa, np.array([LC.cumulative_at(v, 4.0) for v in
                                                            LC.merged_shells(S[f"{field}_nest_lo"], g["hb"])])),
                              ("Req_nest_hi", Fa, np.array([LC.cumulative_at(v, 4.0) for v in
                                                            LC.merged_shells(S[f"{field}_nest_hi"], g["hb"])]))):
            R = np.array([LC.budget_radius(Fp[r], tot[r]) for r in range(nrow)])
            out[(rule, field)] = dict(R=R, status=np.where(np.isfinite(R), "ok", "nan"))
        # ---- gas-only baryon budget (lower-bound proxy of the closure radius of Ayromlou et al. 2023): the all-gas
        #      mass inside R reaches f_b M200c, i.e. sum_s V_s G_s (< R) = 800 pi / 3 (G = rho_gas / (rho_c f_b))
        Gall = LC.merged_shells(S["G_all"], g["all"])
        R = np.array([LC.budget_radius(Gall[r], 800 * np.pi / 3) for r in range(nrow)])
        out[("CLg", field)] = dict(R=R, status=np.where(np.isfinite(R), "ok", "above"))
        # ---- steepest slope (splashback-like edge)
        for rule, qn in SB_RULES.items():
            q = qn.format(f=field)
            gk = "a2ii" if q.endswith("a2ii") else "all"
            v, xr = LC.group_points(S[q], g[gk])
            sl = slopes(v, xr, LC.SLOPE_SIGMA)
            res = [LC.steepest(SLOPE_GRID, sl[r], lo_sb, LC.SLOPE_RANGE[1]) for r in range(nrow)]
            out[(rule, field)] = dict(R=np.array([z["R"] for z in res]), slope=np.array([z["slope"] for z in res]),
                                      status=np.array(["edge" if z["edge"] else "ok" for z in res]), curve=sl[0],
                                      depth=np.array([z["depth"] for z in res]))
        for s_ in LC.SLOPE_SIGMAS:
            v, xr = LC.group_points(S[f"{field}_all"], g["all"])
            sl = slopes(v, xr, s_)
            res = [LC.steepest(SLOPE_GRID, sl[r], lo_sb, LC.SLOPE_RANGE[1]) for r in range(nrow)]
            out[(f"SB_s{int(100 * s_)}", field)] = dict(R=np.array([z["R"] for z in res]),
                                                         slope=np.array([z["slope"] for z in res]),
                                                         status=np.array(["edge" if z["edge"] else "ok" for z in res]))
        # ---- shock signatures (temperature, entropy; all gas; raw fine shells, geometric-mean stacks)
        for rule, q in SH_RULES.items():
            v, xr = LC.group_points(S[q], g["all"]) if b in LC.LOW_BINS else (S[q], LC.shell_radius(LC.X[:-1], LC.X[1:]))
            sl = slopes(np.asarray(v), np.asarray(xr), LC.SLOPE_SIGMA)
            res = [LC.steepest(SLOPE_GRID, sl[r], lo_sh, LC.SHOCK_RANGE[1]) for r in range(nrow)]
            out[(rule, field)] = dict(R=np.array([z["R"] for z in res]), slope=np.array([z["slope"] for z in res]),
                                      status=np.array(["edge" if z["edge"] else "ok" for z in res]), curve=sl[0],
                                      depth=np.array([z["depth"] for z in res]))
            if rule == "SH_K":                          # entropy maximum: d ln K / d ln x falls to 0
                zc = [LC.zero_cross_down(SLOPE_GRID, sl[r], lo_sh, LC.SHOCK_RANGE[1]) for r in range(nrow)]
                out[("SH_K0", field)] = dict(R=np.array([z["R"] for z in zc]),
                                             status=np.array(["ok" if z["status"] == "ok" else "above" for z in zc]))
        # ---- two-halo floor
        for rule, (sel, thr) in TWOH_RULES.items():
            vn, xr = LC.group_points(S[f"{field}_{sel}"], g[sel])
            vd, _ = LC.group_points(S[f"{field}_2h"], g[sel])
            ratio = vn / vd
            R, stt, Rf, nc = row_radii_crossing(ratio, xr, thr)
            out[(rule, field)] = dict(R=R, status=stt, R_first=Rf, n_cross=nc, ratio=ratio[0], xr=xr)
    # ---- reference: the dark-matter splashback radius of More, Diemer & Kravtsov (2015, ApJ 810, 36), Eq. (7),
    #      R_sp / R200m = 0.81 (1 + 0.97 exp(-nu200m / 2.44)), nu200m = delta_c / sigma(M200m) from G's CLASS table
    #      (LinearHalo.nu_of), M200m = 200 Omega_m rho_c 4pi/3 R200m^3 from each halo's SOAP R200m (ESTIMATE: a DM
    #      calibration for Omega_m = 0.27, applied per halo and averaged over the same weights/resamples).
    rows = np.asarray(LS.bins()[b]["rows"])
    t = LS.halo_table()
    r200m = LS.inputs()["r200m_mpc"][rows]
    m200m = t["m200c"][rows] * (r200m / t["r200c"][rows]) ** 3 * LC.OMEGA_M          # 200 rho_m R^3 vs 200 rho_c R^3
    nu = LS.linear_halo().nu_of(m200m)
    rsp_m = 0.81 * (1 + 0.97 * np.exp(-nu / 2.44))
    Wd = LC.boot_matrix(LS.bins()[b])
    wsum = Wd.sum(axis=1)
    out[("MDK15", "E")] = dict(R=(Wd @ (rsp_m * r200m / t["r200c"][rows])) / wsum, status=np.full(nrow, "ok"),
                               rsp_over_r200m=(Wd @ rsp_m) / wsum, nu200m=(Wd @ nu) / wsum)
    out[("MDK15", "P")] = dict(out[("MDK15", "E")])
    # ---- unit conversions and inside fractions
    q200m, q500 = meta["q200m"], meta["q500"]
    for (rule, field), d in out.items():
        R = d["R"]
        d["R200m"] = R * q200m
        d["R500c"] = R * q500
        Fa = LC.merged_shells(S[f"{field}_a2ii"], g["a2ii"])
        Fhb = LC.merged_shells(S[f"{field}_hb"], g["hb"])
        ins_a, ins_hb, fabs = [], [], []
        for r in range(nrow):
            Rr = min(R[r], 4.0) if np.isfinite(R[r]) else 4.0
            a4 = LC.cumulative_at(Fa[r], 4.0)
            ins_a.append(LC.cumulative_at(Fa[r], Rr) / a4)
            ins_hb.append(LC.cumulative_at(Fhb[r], Rr) / LC.cumulative_at(Fhb[r], 4.0))
            fabs.append(LC.cumulative_at(Fa[r], Rr) / (800 * np.pi / 3))
        d["inside_a2ii"] = np.array(ins_a)
        d["inside_hb"] = np.array(ins_hb)
        d["X_a2ii_over_200V"] = np.array(fabs)      # E: N_e mu_XGP/(f_b M200c);  P: Y/(P200 R200c^3) / (800 pi/3)
    return out


def summarise(d, x_last):
    R = np.asarray(d["R"], float)
    st = np.asarray(d["status"])
    cens = st == "above"
    Rb = np.where(cens, 4.0, R)[1:]
    s = LC.boot_summary(Rb, float(R[0]) if np.isfinite(R[0]) else None)
    s["status"] = str(st[0])
    s["censored_fraction_boot"] = float(np.mean(cens[1:])) if len(cens) > 1 else 0.0
    if cens[0]:
        s["lower_limit"] = float(x_last)
    for k in ("R200m", "R500c", "inside_a2ii", "inside_hb", "X_a2ii_over_200V"):
        v = np.asarray(d[k], float)
        vb = np.where(cens, np.nan, v)[1:]
        s[k] = LC.boot_summary(vb, float(v[0]) if np.isfinite(v[0]) and not cens[0] else None)
    if "R_first" in d:
        s["R_first"] = float(d["R_first"][0]) if np.isfinite(d["R_first"][0]) else None
        s["n_cross"] = int(d["n_cross"][0])
        s["non_monotonic"] = bool(d["n_cross"][0] > 1)
        s["non_monotonic_fraction_boot"] = float(np.mean(np.asarray(d["n_cross"][1:]) > 1)) if len(R) > 1 else 0.0
    if "slope" in d:
        s["slope_at_R"] = float(d["slope"][0])
        s["slope_q16_q84"] = [float(np.nanpercentile(d["slope"][1:], 16)), float(np.nanpercentile(d["slope"][1:], 84))]
        s["edge_fraction_boot"] = float(np.mean(np.asarray(st[1:]) == "edge"))
    if "depth" in d:
        s["depth"] = float(d["depth"][0])          # min(slope at the range ends) - slope at the minimum
        s["shallow"] = bool(d["depth"][0] < 0.3)
    return s


def main():
    t0 = LC.time.perf_counter()
    summ = {"rules": DESCRIPTION, "order": RULE_ORDER, "bins": {}, "settings": {
        "X_CORE": LC.X_CORE, "SLOPE_SIGMA": LC.SLOPE_SIGMA, "SLOPE_SIGMAS": LC.SLOPE_SIGMAS,
        "SLOPE_RANGE": LC.SLOPE_RANGE, "SHOCK_RANGE": LC.SHOCK_RANGE, "n_boot": LC.N_BOOT,
        "censoring": "a replicate with f >= threshold at the last point (x = 3.82) is censored and enters the "
                     "percentiles as 4.0 (lower limit); its unit conversions and inside fractions are excluded"}}
    arr = {}
    curves = {}
    rows = []
    for b in LC.BINS:
        lab = LC.LABELS[b]
        LC.log("radii", lab)
        res = evaluate_bin(b)
        meta = LS.bin_stacks(b)["meta"]
        summ["bins"][lab] = {"engine_bin": b, "mean_logm200c": meta["mean_logm200c"],
                             "R200c_over_R200m": float(meta["q200m"][0]), "R200c_over_R500c": float(meta["q500"][0]),
                             "rules": {}}
        for (rule, field), d in res.items():
            s = summarise(d, 3.823)
            summ["bins"][lab]["rules"].setdefault(rule, {})[field] = s
            arr[f"R__{rule}__{field}__{b}"] = np.asarray(d["R"], np.float32)
            arr[f"R200m__{rule}__{field}__{b}"] = np.asarray(d["R200m"], np.float32)
            arr[f"R500c__{rule}__{field}__{b}"] = np.asarray(d["R500c"], np.float32)
            arr[f"ins__{rule}__{field}__{b}"] = np.asarray(d["inside_a2ii"], np.float32)
            if "curve" in d:
                curves[f"slope__{rule}__{field}__{b}"] = np.asarray(d["curve"], np.float32)
            if "ratio" in d:
                curves[f"ratio2h__{rule}__{field}__{b}"] = np.asarray(d["ratio"], np.float32)
                curves[f"ratio2hx__{rule}__{field}__{b}"] = np.asarray(d["xr"], np.float32)
            rows.append(dict(bin=lab, mean_logm200c=round(meta["mean_logm200c"], 4), rule=rule, field=field,
                             R_R200c=s["value"], q16=s["q16"], q84=s["q84"], status=s["status"],
                             censored_boot=s["censored_fraction_boot"],
                             R_R200m=s["R200m"]["value"], R_R500c=s["R500c"]["value"],
                             inside_A2ii_share=s["inside_a2ii"]["value"], inside_hb_share=s["inside_hb"]["value"],
                             non_monotonic=s.get("non_monotonic", "")))
    arr["slope_grid"] = SLOPE_GRID
    np.savez_compressed(LC.RESULTS / "radii.npz", **arr)
    np.savez_compressed(LC.RESULTS / "curves.npz", **curves, slope_grid=SLOPE_GRID)
    summ["seconds"] = LC.time.perf_counter() - t0
    summ["peak_rss_gb"] = LC.rss_gb()
    LC.jdump(summ, LC.RESULTS / "radii.json")
    with open(LC.TABLES / "L1_radii.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    LC.log(f"radii done {summ['seconds']:.0f} s rss {summ['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
