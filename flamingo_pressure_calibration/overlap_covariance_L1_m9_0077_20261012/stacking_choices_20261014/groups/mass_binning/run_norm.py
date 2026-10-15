"""Mass-binning group, part 2: own-halo normalisation (reference) vs bin-mean normalisation.

Variant: each halo's profile is put back into physical units (P_e = (P/P200)_h P200_h at r = x R200c_h), stacked at
fixed physical radius in units of the bin's mean-mass radius Rbar = R200c(10^<log M200c>), and divided by the
mean-mass scale P200(10^<log M200c>) (T200 for T_e, K200 for K; E and G are normalised by rho_c f_b, which does not
depend on mass, so only the radius mapping changes).  Rbar and P200(Mbar) are the inclusion-weighted geometric means
of the haloes' own R200c and P200 (exact, because R200c ~ M^(1/3) and P200 ~ M^(2/3) at fixed z).
Rescaling: rebin.rescale (cumulative PCHIP in ln r); its accuracy on analytic gNFW / beta profiles is measured by
run_rebin_test.py (work/res_rebin_test.json).
Source: the x200 grid (67 shells, contains every fit edge, so s = 1 reproduces the fit-grid reference exactly).
Explanation of the bias (jensen_decomposition): with u = ln s_h (inclusion-weighted mean 0, variance sigma^2),
a = mass power of the normalising scale in units of s (P200, T200, K200: 2; rho_c f_b: 0), alpha = -dlnF/dlnx and
gamma = d2lnF/dlnx2 of the stack, and the within-bin residual mass trend beta(x) = dlnF_h/du at fixed x (weighted LS
slope of ln F_h on u per shell), second order in sigma:
    bin-mean / own - 1 = exp(0.5 sigma^2 [(a + alpha)(a + alpha + 2 beta) + gamma - 2 dbeta/dlnx]) - 1.
(a + alpha)^2 is the pure Jensen (convexity) term, 2 (a + alpha) beta the covariance of the rescaling with the
residual mass trend (can be negative), gamma - 2 dbeta/dlnx the curvature terms.  Also given: the pure-Jensen term
alone, and a numerical trend model (ln F_h(x) = c0(x) + beta(x) u_h, rescaled exactly per halo).
Outputs: work/web_norm.json, work/res_norm.json, work/jensen_shells.npz.
"""
from mbcommon import (BAND_IDX, BAND_LABELS, CAL_LABELS, EDGES_FIT, FIELDS_WEB, N_BOOT, OBS_FIELDS, REF_SCHEME, TAB, W,
                      WORK, X_FIT, band_ratios, dump, impacts, interp_lnF, load, record_from_stack, ref_key, sc, sig,
                      web_record, wmean_boot)
from rebin import rescale
import numpy as np

E200 = np.asarray(sc.edges("x200"), float)
r200 = TAB["r200c"]
scale_amp = {"P": TAB["p200_pa"], "T_e": TAB["T200_K"], "K": TAB["T200_K"]}   # K200 ~ T200 (n_e,200 mass-independent)

# constant c with n_e / n_e,200 = c * E (checked below on the cache)
Kx = load(ref_key("K", "x200"))
Tx = load(ref_key("T_e", "x200"))
Ex = load(ref_key("E", "x200"))
Gx = load(ref_key("G", "x200"))
Px = load(ref_key("P", "x200"))
with np.errstate(divide="ignore", invalid="ignore"):
    cc = (Tx / Kx)**1.5 / Ex
ok = np.isfinite(cc) & (cc > 0)
C_NE = float(np.median(cc[ok]))
C_SPREAD = float(np.percentile(np.abs(cc[ok] / C_NE - 1), 99.9))
print("c_ne", C_NE, "99.9% |dev|", C_SPREAD)


def physical_stack_inputs(rows, Rbar, amp_bar):
    """Per-halo rescaled profiles [n, 35] on the fit grid in Rbar units, for every web field."""
    s = r200[rows] / Rbar
    out = {}
    out["E"] = rescale(Ex[rows], E200, s, EDGES_FIT)
    out["G"] = rescale(Gx[rows], E200, s, EDGES_FIT)
    aP = TAB["p200_pa"][rows] / amp_bar["P"]
    out["P"] = rescale(Px[rows] * aP[:, None], E200, s, EDGES_FIT)
    aT = TAB["T200_K"][rows] / amp_bar["T_e"]
    num = rescale(np.where(np.isfinite(Tx[rows]), Ex[rows] * Tx[rows], 0.0) * aT[:, None], E200, s, EDGES_FIT)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["T_e"] = np.where(out["E"] > 0, num / out["E"], np.nan)
        out["K"] = np.where(out["E"] > 0, out["T_e"] / (C_NE * out["E"])**(2 / 3), np.nan)
    return out, s


results = {"c_ne_check": {"median": sig(C_NE), "p99.9_abs_rel_dev": sig(C_SPREAD)}, "bins": {}, "impacts": {},
           "notes": {}}
records = []
NS = {}
ST = np.load(WORK / "scheme_stacks.npz")
for S, vid in (("M200c_0.2", "mass_binning:binmean_norm"), ("M200c_0.5", "mass_binning:binmean_norm_0.5")):
    per_field_vals = {f: [] for f in FIELDS_WEB}
    per_field_boot = {f: [] for f in FIELDS_WEB}
    rows_out = []
    for b in sc.mass_bins(S):
        rows, w = b["rows"], b["weights"]
        Rbar = 10**np.average(np.log10(r200[rows]), weights=w)
        amp_bar = {k: 10**np.average(np.log10(v[rows]), weights=w) for k, v in scale_amp.items()}
        # arithmetic-mean-mass alternative (amplitude only): P200(<M>) / P200(10^<log M>)
        m_arith = np.average(TAB["m200c"][rows], weights=w)
        alt_amp = (m_arith / 10**b["mean_logm200c"])**(2 / 3)
        prof, s = physical_stack_inputs(rows, Rbar, amp_bar)
        ref_strict = np.asarray(sc.resolution_mask(rows, "fit", "strict"), bool)
        # shells resolved for the variant: inner edge beyond the reference's first resolved edge x max(s),
        # outer edge inside every halo's 4 R200c coverage
        first = EDGES_FIT[:-1][np.argmax(ref_strict)] if ref_strict.any() else np.inf
        res_var = ref_strict & (EDGES_FIT[:-1] >= first * s.max()) & (EDGES_FIT[1:] <= 4.0 * s.min() + 1e-9)
        entry = {"bin": b["label"], "N": b["N"], "mean_logm200c": sig(b["mean_logm200c"]),
                 "s_range": [sig(s.min()), sig(s.max())],
                 "sd_ln_s": sig(np.sqrt(np.average(np.log(s)**2, weights=w))),
                 "Pbar_arith_over_geo": sig(alt_amp), "fields": {}, "_res_var": res_var.tolist()}
        for field in FIELDS_WEB:
            # web record via the core's stack (default seed of the bin; same resample as stack_bin)
            seed = sc.STUDY_SEED + 1000 * int(b["index"])
            rr = sc.stack(prof[field], w, "mean", n_boot=N_BOOT, seed=seed, strata=b["strata"])
            rr["resolved"] = res_var
            rr["edges"] = EDGES_FIT
            records.append(record_from_stack(vid, field, S, b, rr))
            # paired global-bootstrap stacks for the comparison
            Fv = np.full((len(W), 35), np.nan)
            Fv[rows] = prof[field]
            v, bt, _ = wmean_boot(Fv, rows)
            per_field_vals[field].append(v)
            per_field_boot[field].append(bt)
            i = list(ST[f"{field}__{S}__labels"]).index(b["label"])
            v0, b0 = ST[f"{field}__{S}__value"][i], ST[f"{field}__{S}__boot"][i]
            entry["fields"][field] = band_ratios(v, v0, bt, b0, res_var)
        rows_out.append(entry)
    results["bins"][S] = rows_out
    NS[S] = {f: (np.array(per_field_vals[f]), np.array(per_field_boot[f])) for f in FIELDS_WEB}

# ------------------------------------------------------------------ explanation: Jensen + residual-mass-trend terms
AMP_POWER = {"P": 2.0, "E": 0.0, "G": 0.0, "T_e": 2.0, "K": 2.0}
LNX = np.log(X_FIT)
jensen = {}
jshell = {}
for S in ("M200c_0.2", "M200c_0.5"):
    for b in sc.mass_bins(S):
        rows, w = b["rows"], b["weights"]
        Rbar = 10**np.average(np.log10(r200[rows]), weights=w)
        u = np.log(r200[rows] / Rbar)
        sig2 = float(np.average(u**2, weights=w))
        entry = {"sd_ln_s": sig(np.sqrt(sig2)), "fields": {}}
        meas_bins = {e["bin"]: e for e in results["bins"][S]}
        for field in FIELDS_WEB:
            D = np.asarray(load(ref_key(field))[rows], float)
            ok = np.isfinite(D) & (D > 0)
            st = np.array([np.average(D[ok[:, j], j], weights=w[ok[:, j]]) if ok[:, j].any() else np.nan
                           for j in range(D.shape[1])])
            with np.errstate(divide="ignore", invalid="ignore"):
                lnF = np.log(st)
            alpha = -np.gradient(lnF, LNX)
            gamma = np.gradient(np.gradient(lnF, LNX), LNX)
            beta, c0 = np.full(35, np.nan), np.full(35, np.nan)
            for j in range(35):
                m = ok[:, j]
                if m.sum() > 10:
                    X = np.vstack([np.ones(m.sum()), u[m]]).T
                    c0[j], beta[j] = np.linalg.solve(X.T @ (w[m][:, None] * X), X.T @ (w[m] * np.log(D[m, j])))
            dbeta = np.gradient(beta, LNX)
            a = AMP_POWER[field]
            pure = np.expm1(0.5 * (a + alpha)**2 * sig2)
            second = np.expm1(0.5 * sig2 * ((a + alpha) * (a + alpha + 2 * beta) + gamma - 2 * dbeta))
            jensen_cov = np.expm1(0.5 * sig2 * (a + alpha) * (a + alpha + 2 * beta))
            good = np.isfinite(c0) & np.isfinite(beta)
            num, den = np.zeros(35), np.zeros(35)
            if good.sum() < 2:
                num[:] = np.nan
            for hh in (range(len(rows)) if good.sum() >= 2 else ()):
                L = LNX - u[hh]
                num += w[hh] * np.exp(a * u[hh] + np.interp(L, LNX[good], c0[good], left=np.nan, right=np.nan)
                                      + np.interp(L, LNX[good], beta[good], left=np.nan, right=np.nan) * u[hh])
                den += w[hh] * np.exp(c0 + beta * u[hh])
            model = num / den - 1
            resolved = np.asarray(meas_bins[b["label"]]["_res_var"], bool)
            jshell[f"{field}__{S}__{b['label']}"] = np.vstack([pure, second, model, beta, alpha, resolved])
            band = {}
            for lab, idx in zip(BAND_LABELS, BAND_IDX):
                sel = idx[resolved[idx]]
                mv = meas_bins[b["label"]]["fields"][field].get(lab)
                if not len(sel) or mv is None:
                    band[lab] = None
                    continue
                band[lab] = {"measured_pct": sig(100 * (mv["ratio"] - 1)),
                             "measured_err_pct": sig(100 * mv["err"]) if mv["err"] is not None else None,
                             "pure_jensen_pct": sig(100 * np.nanmedian(pure[sel])),
                             "jensen_plus_trend_cov_pct": sig(100 * np.nanmedian(jensen_cov[sel])),
                             "second_order_pct": sig(100 * np.nanmedian(second[sel])),
                             "trend_model_pct": sig(100 * np.nanmedian(model[sel])),
                             "a_plus_alpha": sig(a + np.nanmedian(alpha[sel])), "beta": sig(np.nanmedian(beta[sel]))}
            entry["fields"][field] = band
        jensen.setdefault(S, {})[b["label"]] = entry
results["jensen_decomposition"] = jensen
results["notes"]["jensen_decomposition"] = (
    "per bin / field / band (median over the variant's resolved shells): measured bin-mean/own - 1 and predictions "
    "with u = ln(R200c_h/Rbar), sigma^2 = weighted var(u): pure_jensen = exp(0.5 (a+alpha)^2 sigma^2) - 1; "
    "jensen_plus_trend_cov = exp(0.5 (a+alpha)(a+alpha+2 beta) sigma^2) - 1; second_order = exp(0.5 sigma^2 "
    "[(a+alpha)(a+alpha+2beta) + gamma - 2 dbeta/dlnx]) - 1; trend_model: per-halo ln F_h(x) = c0(x) + beta(x) u_h "
    "(weighted LS per shell, F_h > 0) rescaled exactly (s^a F(x'/s)) and stacked.  a: P, T_e, K 2; E, G 0.  alpha, "
    "gamma from the own-normalised stack (ln F vs ln x); beta = dlnF_h/du at fixed x/R200c (= 3 dlnF/dlnM).")
for S in results["bins"]:
    for e in results["bins"][S]:
        e.pop("_res_var", None)
np.savez(WORK / "jensen_shells.npz", **jshell)

# observable impacts
for S in ("M200c_0.2", "M200c_0.5"):
    labs = [b["label"] for b in sc.mass_bins(S)]
    m_nodes = np.array([b["mean_logm200c"] for b in sc.mass_bins(S)])
    results["impacts"][S] = {}
    for field in OBS_FIELDS:
        Vv, Vb = NS[S][field]
        R0, Rb0 = ST[f"{field}__{S}__value"], ST[f"{field}__{S}__boot"]
        if S == "M200c_0.2":
            idx = [labs.index(c) for c in CAL_LABELS]
            Fv, Bv = Vv[idx], np.moveaxis(Vb[idx], 0, 1)
            Fr, Br = R0[idx], np.moveaxis(Rb0[idx], 0, 1)
        else:   # smooth painting through the 0.5-dex nodes, both own-normalised and bin-mean-normalised
            mt = [sc.find_bin(REF_SCHEME, c)["mean_logm200c"] for c in CAL_LABELS]
            Fv = np.array([interp_lnF(m_nodes, Vv, m)[0] for m in mt])
            Bv = np.moveaxis(np.array([interp_lnF(m_nodes, np.moveaxis(Vb, 1, 0), m)[0] for m in mt]), 0, 1)
            Fr = np.array([interp_lnF(m_nodes, R0, m)[0] for m in mt])
            Br = np.moveaxis(np.array([interp_lnF(m_nodes, np.moveaxis(Rb0, 1, 0), m)[0] for m in mt]), 0, 1)
        results["impacts"][S][field] = impacts(Fv, Fr, Bv, Br, field)

results["notes"]["bins"] = (
    "per bin and field: band ratios (bin-mean-normalised stack / own-normalised stack of the same haloes and the same "
    "scheme), median over shells strict-resolved for the reference with inner edge >= first resolved edge x max(s) and "
    "outer edge <= 4 min(s); err = paired global-bootstrap SD.  s = R200c_h / Rbar.  Pbar_arith_over_geo = "
    "P200(<M200c>) / P200(10^<log M200c>), the extra amplitude factor if the bin's mean mass were the arithmetic mean.")
results["notes"]["impacts"] = (
    "per cent; M200c_0.2: the calibrated bins' bin-mean-normalised stacks vs the reference; M200c_0.5: both "
    "normalisations painted smoothly through the 0.5-dex nodes (ln F linear in mean log M200c) at the calibrated bins' "
    "mean masses, so only the normalisation differs.  All 35 shells (NaN cells beyond coverage count as no change).")
dump(results, WORK / "res_norm.json")
dump(records, WORK / "web_norm.json")
print("done", len(records))
