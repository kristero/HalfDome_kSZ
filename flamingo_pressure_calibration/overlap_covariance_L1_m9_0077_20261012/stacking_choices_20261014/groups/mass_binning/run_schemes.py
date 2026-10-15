"""Mass-binning group, part 1: bin schemes, within-bin mass trend, M200c vs M500c, painting assignment.

Outputs: work/scheme_stacks.npz (stacks + paired global-bootstrap replicates of every scheme / field),
         work/web_schemes.json (web records), work/res_schemes.json (results).
Observable metrics for the schemes:
  (A) 'at reference node masses': the scheme's stacks interpolated to the calibrated bins' mean log M200c, compared
      with the reference stacks.  The reference is exact there by construction, so (A) favours the 0.2-dex nodes.
  (B) 'continuous painting vs truth' (mbcommon.continuous_paint): every calibrated halo painted at its own log M200c
      from the scheme's nodes; painted bin mean vs the true bin mean.  Applies to the reference too.
  (C) = (B) of the scheme relative to (B) of the reference (paired): the cost of the scheme vs 0.2-dex bins when
      both are painted the same way.
"""
from mbcommon import *  # noqa: F401,F403
from mbcommon import (BAND_LABELS, CAL_LABELS, EDGES_FIT, FIELDS_WEB, N_BOOT, OBS_FIELDS, REF_SCHEME, TAB, W, WORK,
                      X_FIT, band_ratios, continuous_paint, dump, impacts, interp_lnF, load, record_from_stack, ref_key,
                      sc, sig, strict_mask, wmean_boot, BAND_IDX)
import numpy as np

SCHEMES = sc.SCHEMES
VID = {s: ("ref" if s == REF_SCHEME else f"mass_binning:{s}") for s in SCHEMES}

# ------------------------------------------------------------------------------------------- stacks
records = []
ST = {}           # (field, scheme) -> dict of arrays
bootcheck = {}    # global-bootstrap SD (core design / census design) vs the core's per-bin stack_bin SD
for field in FIELDS_WEB:
    key = ref_key(field)
    F = load(key)
    for S in SCHEMES:
        bins = sc.mass_bins(S)
        vals, boots, strict, nused = [], [], [], []
        for b in bins:
            res = sc.stack_bin(key, b, "mean", "strict", n_boot=N_BOOT)
            records.append(record_from_stack(VID[S], field, S, b, res))
            v, bt, nu = wmean_boot(F, b["rows"])
            assert np.allclose(v, res["value"], equal_nan=True, rtol=1e-12)
            if S == REF_SCHEME:
                _, btc, _ = wmean_boot(F, b["rows"], design="core")
                with np.errstate(divide="ignore", invalid="ignore"):
                    rc = np.std(btc, axis=0, ddof=1) / np.asarray(res["err"], float)
                    rn_ = np.std(bt, axis=0, ddof=1) / np.asarray(res["err"], float)
                rs_ = np.asarray(res["resolved"], bool)
                bootcheck.setdefault(field, {})[b["label"]] = {
                    lab: ({"core_design_over_stack_bin": sig(np.nanmedian(rc[idx[rs_[idx]]])),
                           "census_design_over_stack_bin": sig(np.nanmedian(rn_[idx[rs_[idx]]]))}
                          if rs_[idx].any() else None) for lab, idx in zip(BAND_LABELS, BAND_IDX)}
            vals.append(v)
            boots.append(bt)
            strict.append(np.asarray(res["resolved"], bool))
            nused.append(nu)
        ST[(field, S)] = {"value": np.array(vals), "boot": np.array(boots), "strict": np.array(strict),
                          "labels": [b["label"] for b in bins],
                          "m_scheme": np.array([b["mean_log_mass"] for b in bins]),
                          "m200": np.array([b["mean_logm200c"] for b in bins]),
                          "m500": np.array([b["mean_logm500c"] for b in bins]),
                          "N": np.array([b["N"] for b in bins]), "N_eff": np.array([b["N_eff"] for b in bins]),
                          "sum_weight": np.array([b["sum_weight"] for b in bins])}
    print("stacked", field, flush=True)

np.savez(WORK / "scheme_stacks.npz", **{f"{f}__{S}__{k}": np.asarray(v) for (f, S), d in ST.items()
                                        for k, v in d.items()})
dump(records, WORK / "web_schemes.json")

ref_bins = sc.mass_bins(REF_SCHEME)
res = {"notes": {}}

# --------------------------------------------------------------- (1a) each scheme vs the reference at the same mass
# Reference evaluated at a variant bin's inclusion-weighted mean log M200c: ln F of the M200c_0.2 stacks linear in
# their mean log M200c (paired replicates interpolated the same way).  Shells: strict-resolved in the variant bin
# and in both bracketing reference bins.
scheme_vs_ref = {}
for field in FIELDS_WEB:
    R = ST[(field, REF_SCHEME)]
    out_f = {}
    for S in SCHEMES:
        if S == REF_SCHEME:
            continue
        V = ST[(field, S)]
        rows_out = []
        for j, lab in enumerate(V["labels"]):
            m = V["m200"][j]
            Fr, i0, t, extr = interp_lnF(R["m200"], R["value"], m)
            Br, _, _, _ = interp_lnF(R["m200"], np.moveaxis(R["boot"], 1, 0), m)
            order = np.argsort(R["m200"])
            ia, ib = order[np.clip(np.searchsorted(R["m200"][order], m) - 1, 0, len(order) - 2)], None
            k = int(np.clip(np.searchsorted(R["m200"][order], m) - 1, 0, len(order) - 2))
            ia, ib = order[k], order[k + 1]
            resolved = V["strict"][j] & R["strict"][ia] & R["strict"][ib]
            rows_out.append({"bin": lab, "mean_log_mass": sig(V["m_scheme"][j]), "mean_logm200c": sig(m),
                             "N": int(V["N"][j]), "N_eff": sig(V["N_eff"][j]), "extrapolated_reference": extr,
                             "bands": band_ratios(V["value"][j], Fr, V["boot"][j], Br, resolved)})
        out_f[S] = rows_out
    scheme_vs_ref[field] = out_f
res["scheme_vs_reference_same_mass"] = scheme_vs_ref
res["notes"]["scheme_vs_reference_same_mass"] = (
    "ratio of each scheme bin's stack to the M200c_0.2 reference interpolated (ln F linear in mean log M200c) to the "
    "bin's inclusion-weighted mean log M200c; per radial band (R200c) the median over shells strict-resolved in the "
    "variant bin and both bracketing reference bins; err = SD over 200 paired global-bootstrap replicates")


# --------------------------------------------------- (1b) observable impact of each scheme (smooth-trend painting)
def painted_cal(field, S):
    """[10,35] profile a painter would use for each calibrated 0.2-dex bin: the scheme's stacks with ln F linear in
    their mean log M200c, evaluated at the calibrated bin's mean log M200c (linear extrapolation beyond the end
    nodes).  Also the [B,10,35] replicates."""
    V = ST[(field, S)]
    vals, boots, extr = [], [], []
    for lab in CAL_LABELS:
        m = sc.find_bin(REF_SCHEME, lab)["mean_logm200c"]
        f, _, _, e = interp_lnF(V["m200"], V["value"], m)
        bt, _, _, _ = interp_lnF(V["m200"], np.moveaxis(V["boot"], 1, 0), m)
        vals.append(f)
        boots.append(bt)
        extr.append(e)
    return np.array(vals), np.moveaxis(np.array(boots), 0, 1), extr


def ref_cal(field):
    R = ST[(field, REF_SCHEME)]
    idx = [R["labels"].index(lab) for lab in CAL_LABELS]
    return R["value"][idx], np.moveaxis(R["boot"][idx], 0, 1)


scheme_impacts = {}
for field in OBS_FIELDS:
    Fr, Br = ref_cal(field)
    for S in SCHEMES:
        Fv, Bv, extr = painted_cal(field, S)
        imp = impacts(Fv, Fr, Bv, Br, field)
        scheme_impacts.setdefault(S, {})[field] = {"impact": imp, "extrapolated_bins": [lab for lab, e in
                                                                                       zip(CAL_LABELS, extr) if e]}
res["scheme_impacts_smooth_painting"] = scheme_impacts
res["notes"]["scheme_impacts_smooth_painting"] = (
    "METRIC A (at the reference node masses): first-order impact (sc.impact_from_stacks, per cent) of painting the 10 "
    "calibrated 0.2-dex M200c bins with each scheme's stacks interpolated (ln F linear in mean log M200c, linear "
    "extrapolation beyond the end nodes) to the calibrated bins' mean log M200c, relative to the reference stacks "
    "themselves; all 35 shells used; err = paired global-bootstrap SD.  M200c_0.2 itself reproduces the reference "
    "exactly (closure check) - the reference has zero interpolation error at its own nodes by construction, so this "
    "metric charges every other scheme with node-placement error (see scheme_impacts_continuous_painting).")

# ------------------------------------------- (1b') continuous painting (metric B: vs truth; metric C: vs reference)
cal_bins = [sc.find_bin(REF_SCHEME, lab) for lab in CAL_LABELS]
cont = {}
painted = {}
for field in OBS_FIELDS:
    Fr, Br = ref_cal(field)                       # truth: the mean of the haloes' own profiles in each bin
    for S in SCHEMES:
        V = ST[(field, S)]
        Pv, Pb, n_ext = continuous_paint(V["m200"], V["value"], V["boot"], cal_bins)
        painted[(field, S)] = (Pv, Pb)
        cont.setdefault(S, {})[field] = {"vs_truth": impacts(Pv, Fr, Pb, Br, field),
                                         "n_extrapolated_haloes": dict(zip(CAL_LABELS, [int(v) for v in n_ext]))}
    P0, P0b = painted[(field, REF_SCHEME)]
    for S in SCHEMES:
        Pv, Pb = painted[(field, S)]
        cont[S][field]["vs_reference_painted_same_way"] = impacts(Pv, P0, Pb, P0b, field)
res["scheme_impacts_continuous_painting"] = cont
res["notes"]["scheme_impacts_continuous_painting"] = (
    "METRIC B (vs_truth): every halo of the 10 calibrated 0.2-dex bins is painted with the scheme's node stacks "
    "interpolated (ln F linear in log M200c between the bracketing nodes at their inclusion-weighted mean log M200c; "
    "linear extrapolation beyond the end nodes) to its OWN log M200c; the painted bin mean (inclusion-weighted) is "
    "compared with the true bin mean (the reference stack, i.e. the mean of the haloes' own normalised profiles); "
    "first-order impact via sc.impact_from_stacks, per cent.  The reference (M200c_0.2) is painted the same way, so no "
    "scheme is exact by construction.  METRIC C (vs_reference_painted_same_way): scheme painted continuously vs the "
    "reference painted continuously (the cost of the scheme relative to 0.2-dex bins).  err = paired global-bootstrap "
    "SD (census design; replicate r uses draw r for the nodes, the painted haloes' weights and the truth).")
np.savez(WORK / "painted_continuous.npz", **{f"{f}__{S}__{k}": (v[0] if k == "value" else v[1])
                                              for (f, S), v in painted.items() for k in ("value", "boot")})

# ------------------------------------------------------------- (1c) wide bins vs the 0.1-dex bins they contain
fine = "M200c_0.1"
wide_vs_fine = {}
for field in FIELDS_WEB:
    Fn = ST[(field, fine)]
    out_f = {}
    for S in ("M200c_0.2", "M200c_0.3", "M200c_0.5"):
        V = ST[(field, S)]
        rows_out = []
        for j, lab in enumerate(V["labels"]):
            lo, hi = map(float, lab.split("-"))
            sub = [i for i, l2 in enumerate(Fn["labels"]) if float(l2.split("-")[0]) >= lo - 1e-9
                   and float(l2.split("-")[1]) <= hi + 1e-9]
            if len(sub) < 2:
                continue
            wsum = Fn["sum_weight"][sub]
            # (A) inclusion-weighted average of the sub-bin stacks (weights = summed inclusion weights)
            avgA = np.einsum("k,ks->s", wsum, Fn["value"][sub]) / wsum.sum()
            avgB = np.einsum("k,kbs->bs", wsum, Fn["boot"][sub]) / wsum.sum()
            # (B) the 0.1-dex stack at the same mean mass (ln F linear in mean log M200c between 0.1-dex bins)
            m = V["m200"][j]
            Fi, _, _, extr = interp_lnF(Fn["m200"], Fn["value"], m)
            Bi, _, _, _ = interp_lnF(Fn["m200"], np.moveaxis(Fn["boot"], 1, 0), m)
            order = np.argsort(Fn["m200"])
            k = int(np.clip(np.searchsorted(Fn["m200"][order], m) - 1, 0, len(order) - 2))
            ia, ib = order[k], order[k + 1]
            resolvedB = V["strict"][j] & Fn["strict"][ia] & Fn["strict"][ib]
            # weighted variance of log M200c inside the wide bin
            b = sc.find_bin(S, lab)
            lm = TAB["logm200c"][b["rows"]]
            var_m = float(np.average((lm - b["mean_logm200c"])**2, weights=b["weights"]))
            first_res = lambda msk: sig(float(EDGES_FIT[:-1][np.argmax(msk)])) if msk.any() else None  # noqa: E731
            rows_out.append({
                "bin": lab, "mean_logm200c": sig(m), "N": int(V["N"][j]), "sd_logm_within": sig(np.sqrt(var_m)),
                "sub_bins": [Fn["labels"][i] for i in sub],
                "vs_weighted_avg_of_subbins": band_ratios(V["value"][j], avgA, V["boot"][j], avgB, V["strict"][j]),
                "vs_0p1_at_same_mean_mass": band_ratios(V["value"][j], Fi, V["boot"][j], Bi, resolvedB),
                "first_resolved_edge_wide": first_res(V["strict"][j]),
                "first_resolved_edge_subbins": [first_res(Fn["strict"][i]) for i in sub],
                "extrapolated": extr})
        out_f[S] = rows_out
    wide_vs_fine[field] = out_f
res["wide_vs_fine"] = wide_vs_fine
res["notes"]["wide_vs_fine"] = (
    "A: wide-bin stack / summed-inclusion-weight average of its 0.1-dex sub-bin stacks (identity for the mean "
    "statistic up to NaN halo-shells; strict mask of the wide bin).  B: wide-bin stack / 0.1-dex stacks interpolated "
    "(ln F linear in mean log M200c) to the wide bin's mean log M200c (shells resolved in the wide bin and the two "
    "bracketing 0.1-dex bins).  first_resolved_edge: inner edge (R200c) of the first strict-resolved fit shell.")


# ------------------------------------------------------------------- (1d) residual mass trend of the normalised profile
def band_median(arr2, resolved):
    """arr2 [..., S] -> per band median over resolved shells ([..., n_bands])."""
    out = []
    for idx in BAND_IDX:
        sel = idx[resolved[idx]]
        out.append(np.nanmedian(arr2[..., sel], axis=-1) if len(sel) else np.full(arr2.shape[:-1], np.nan))
    return np.stack(out, axis=-1)


mass_trend = {}
for field in FIELDS_WEB:
    R = ST[(field, REF_SCHEME)]
    Fn = ST[(field, fine)]
    with np.errstate(divide="ignore", invalid="ignore"):
        lnR = np.log(np.where(R["value"] > 0, R["value"], np.nan))
        lnRb = np.log(np.where(R["boot"] > 0, R["boot"], np.nan))
        lnN = np.log(np.where(Fn["value"] > 0, Fn["value"], np.nan))
        lnNb = np.log(np.where(Fn["boot"] > 0, Fn["boot"], np.nan))
    rows_out = []
    for i, lab in enumerate(R["labels"]):
        entry = {"bin": lab, "mean_logm200c": sig(R["m200"][i])}
        # (a) central difference of the 0.2-dex stacks, per 0.1 dex
        if 0 < i < len(R["labels"]) - 1:
            dm = R["m200"][i + 1] - R["m200"][i - 1]
            res_ = R["strict"][i - 1] & R["strict"][i] & R["strict"][i + 1]
            s0 = (lnR[i + 1] - lnR[i - 1]) / dm * 0.1
            sb = (lnRb[i + 1] - lnRb[i - 1]) / dm * 0.1
            d2 = (((lnR[i + 1] - lnR[i]) / (R["m200"][i + 1] - R["m200"][i])
                   - (lnR[i] - lnR[i - 1]) / (R["m200"][i] - R["m200"][i - 1])) / (0.5 * dm))
            bm, bb, bc = band_median(s0, res_), band_median(sb, res_), band_median(d2, res_)
            entry["slope_central_per_0p1dex"] = {lab2: ({"value": sig(bm[k]), "err": sig(np.nanstd(bb[:, k], ddof=1))}
                                                       if np.isfinite(bm[k]) else None)
                                                 for k, lab2 in enumerate(BAND_LABELS)}
            entry["curvature_per_dex2"] = {lab2: sig(bc[k]) for k, lab2 in enumerate(BAND_LABELS)}
        # (b) the two 0.1-dex halves of this 0.2-dex bin
        lo, hi = map(float, lab.split("-"))
        sub = [k for k, l2 in enumerate(Fn["labels"]) if float(l2.split("-")[0]) >= lo - 1e-9
               and float(l2.split("-")[1]) <= hi + 1e-9]
        if len(sub) == 2:
            a, b2 = sub
            dm = Fn["m200"][b2] - Fn["m200"][a]
            res_ = Fn["strict"][a] & Fn["strict"][b2]
            s0 = (lnN[b2] - lnN[a]) / dm * 0.1
            sb = (lnNb[b2] - lnNb[a]) / dm * 0.1
            bm, bb = band_median(s0, res_), band_median(sb, res_)
            entry["slope_halves_per_0p1dex"] = {lab2: ({"value": sig(bm[k]), "err": sig(np.nanstd(bb[:, k], ddof=1))}
                                                      if np.isfinite(bm[k]) else None)
                                                for k, lab2 in enumerate(BAND_LABELS)}
            entry["halves_dm"] = sig(dm)
        # (c) share of the per-halo ln-variance that the mass trend explains inside this 0.2-dex bin
        b = sc.find_bin(REF_SCHEME, lab)
        lm = TAB["logm200c"][b["rows"]]
        var_m = float(np.average((lm - b["mean_logm200c"])**2, weights=b["weights"]))
        st = sc.stack(load(ref_key(field))[b["rows"]], b["weights"], "mean", n_boot=0)
        if "slope_central_per_0p1dex" in entry:
            sl = np.array([np.nan if entry["slope_central_per_0p1dex"][k] is None else
                           entry["slope_central_per_0p1dex"][k]["value"] for k in BAND_LABELS], float) * 10
            sdl = band_median(st["sd_ln"][None, :], R["strict"][i])[0]
            entry["trend_var_share"] = {lab2: sig(sl[k]**2 * var_m / sdl[k]**2) if np.isfinite(sl[k] * sdl[k])
                                        else None for k, lab2 in enumerate(BAND_LABELS)}
            entry["sd_ln_per_halo"] = {lab2: sig(sdl[k]) for k, lab2 in enumerate(BAND_LABELS)}
        entry["sd_logm_within"] = sig(np.sqrt(var_m))
        rows_out.append(entry)
    mass_trend[field] = rows_out
res["mass_trend"] = mass_trend
res["notes"]["mass_trend"] = (
    "d ln F per 0.1 dex of M200c (F = normalised stacked mean): 'central' from the neighbouring 0.2-dex reference bins, "
    "'halves' from the two 0.1-dex sub-bins of the bin; band = median over shells resolved in all bins used; "
    "err = paired global-bootstrap SD.  curvature = second difference of ln F (per dex^2).  trend_var_share = "
    "(slope per dex)^2 x var(log M200c inside the 0.2-dex bin) / sd_ln^2 (per-halo lognormal scatter of the bin).")

# ------------------------------------------------------------------------------ (3) M500c vs M200c binning variable
lm2, lm5 = TAB["logm200c"], TAB["logm500c"]
dlm = lm2 - lm5
lc = np.log10(TAB["concentration_200c"])
m5 = []
for b in ref_bins:
    r, w = b["rows"], b["weights"]
    mu = np.average(dlm[r], weights=w)
    sd = np.sqrt(np.average((dlm[r] - mu)**2, weights=w))
    ok = np.isfinite(lc[r])
    cc = np.corrcoef(dlm[r][ok], lc[r][ok])[0, 1]
    m5.append({"bin": b["label"], "mean_dlogm_200c_minus_500c": sig(mu), "sd_dlogm": sig(sd),
               "corr_with_log_c200c": sig(cc), "sd_logm500c_at_fixed_m200c_bin": sig(
                   np.sqrt(np.average((lm5[r] - np.average(lm5[r], weights=w))**2, weights=w)))})
# migration: M500c_0.2 bin -> M200c_0.2 bins (inclusion-weighted fractions); expected (diagonal) partner = the
# M200c bin containing (M500c-bin mean log M500c + mean offset)
mig = {}
for Sv in ("M500c_0.2", "M500c_0.5", "Lee22_h", "M200c_0.5"):
    rows_out = []
    for b in sc.mass_bins(Sv):
        r, w = b["rows"], b["weights"]
        fr = {}
        for rb in ref_bins:
            s = w[np.isin(r, rb["rows"])].sum()
            if s > 0:
                fr[rb["label"]] = sig(s / w.sum())
        top = max(fr.items(), key=lambda kv: kv[1])
        rows_out.append({"bin": b["label"], "N": b["N"], "mean_logm200c": sig(b["mean_logm200c"]),
                         "mean_logm500c": sig(b["mean_logm500c"]),
                         "sd_logm200c": sig(np.sqrt(np.average((lm2[r] - b["mean_logm200c"])**2, weights=w))),
                         "fractions_by_M200c_0.2_bin": fr, "largest_share": {top[0]: top[1]},
                         "mean_log_c200c": sig(np.average(lc[r][np.isfinite(lc[r])],
                                                          weights=w[np.isfinite(lc[r])]))})
    mig[Sv] = rows_out
# reverse view: share of each calibrated 0.2-dex M200c bin's haloes in its most-populated M500c_0.5 bin
rev = {}
for rb in ref_bins:
    fr = {}
    for b in sc.mass_bins("M500c_0.5"):
        s = rb["weights"][np.isin(rb["rows"], b["rows"])].sum()
        if s > 0:
            fr[b["label"]] = sig(s / rb["weights"].sum())
    rev[rb["label"]] = fr
# concentration of M500c-selected vs M200c-selected haloes at the same mean M200c
ref_lc = np.array([np.average(lc[b["rows"]][np.isfinite(lc[b["rows"]])], weights=b["weights"][np.isfinite(
    lc[b["rows"]])]) for b in ref_bins])
ref_m = np.array([b["mean_logm200c"] for b in ref_bins])
for Sv in ("M500c_0.2", "M500c_0.5"):
    for e in mig[Sv]:
        e["dlog_c200c_vs_M200c_bins_at_same_mean_m200"] = sig(e["mean_log_c200c"] - np.interp(e["mean_logm200c"],
                                                                                               ref_m, ref_lc))
h = sc._hubble()
lee = []
for b in sc.mass_bins("Lee22_h"):
    lee.append({"bin_h^-1Msun": b["label"], "N": b["N"],
                "edges_Msun": [sig(b["lo"] - np.log10(h)), sig(b["hi"] - np.log10(h))],
                "mean_logm200c_Msun": sig(b["mean_logm200c"]), "mean_log_mass_h^-1Msun": sig(b["mean_log_mass"])})
res["m500_vs_m200"] = {"per_M200c_bin": m5, "composition": mig, "M200c_bin_into_M500c_0.5_bins": rev,
                       "lee22_rule_dropped_bins": [{"bin": b["label"], "N": b["N"]} for b in sc.mass_bins("Lee22_h")
                                                   if b["N"] <= 29],
                       "lee22_bins_in_Msun": {"h": sig(h), "log10_h": sig(np.log10(h)),
                                              "rule": "log M[h^-1 Msun] = log M[Msun] + log h, so an h^-1 Msun edge "
                                                      "E corresponds to E - log h = E + 0.167 dex in Msun",
                                              "bins": lee}}
res["bootstrap_consistency"] = bootcheck
res["notes"]["bootstrap_consistency"] = (
    "per M200c_0.2 bin, field and band: median over the band's strict-resolved shells of (SD of the global-bootstrap "
    "replicates) / (err of sc.stack_bin, the core's per-bin stratified bootstrap with its default seed).  "
    "core_design: the global bootstrap with the core's strata (mass_bin x class_main) - differs only by the random "
    "draws; census_design: plain within-bin resampling in the census bins 14.5-15.5 (used for every error in "
    "results.json).")

# ----------------------------------------------------- (F) painting assignment: step vs smooth vs per-halo truth
# Integrated per-halo amounts within 4 R200c: Y_h = P200 R200c^3 int P/P200 dV_x; N_e,h = R200c^3 int E dV_x (E
# normalisation is mass independent).  Proxies over the calibrated haloes (13.5 <= log M200c < 15.5):
# mean y ~ sum w Y, low-ell 1-halo tSZ ~ sum w Y^2, mean DM ~ sum w N_e, low-ell kSZ 1h ~ sum w N_e^2.
cal = (lm2 >= 13.5) & (lm2 < 15.5)
IP = sc.integral(load(ref_key("P")), "fit", 4.0)
IE = sc.integral(load(ref_key("E")), "fit", 4.0)
ampP = TAB["p200_pa"] * TAB["r200c"]**3
ampE = TAB["r200c"]**3


def assigned(I, S, how):
    out = np.full(len(I), np.nan)
    bins = sc.mass_bins(S)
    means = np.array([np.average(I[b["rows"]], weights=b["weights"]) for b in bins])
    if how == "step":
        for b, mu in zip(bins, means):
            out[b["rows"]] = mu
        return out
    m_nodes = np.array([b["mean_logm200c"] for b in bins])
    o = np.argsort(m_nodes)
    m_nodes, lnI = m_nodes[o], np.log(means[o])
    # linear in log M200c, linear extrapolation beyond the end nodes
    k = np.clip(np.searchsorted(m_nodes, lm2) - 1, 0, len(m_nodes) - 2)
    t = (lm2 - m_nodes[k]) / (m_nodes[k + 1] - m_nodes[k])
    return np.exp((1 - t) * lnI[k] + t * lnI[k + 1])


def proxies(IPh, IEh, valid):
    w = W[valid]
    Y = (ampP * IPh)[valid]
    N = (ampE * IEh)[valid]
    return {"y_mean": np.sum(w * Y), "tSZ_1h_lowell": np.sum(w * Y**2), "DM_mean": np.sum(w * N),
            "kSZ_1h_lowell": np.sum(w * N**2)}


KEYS4 = ("y_mean", "DM_mean", "tSZ_1h_lowell", "kSZ_1h_lowell")
s01P, s01E = assigned(IP, "M200c_0.1", "smooth"), assigned(IE, "M200c_0.1", "smooth")
assign = {}
for S in SCHEMES:
    for how in ("step", "smooth"):
        aP, aE = assigned(IP, S, how), assigned(IE, S, how)
        valid = cal & np.isfinite(aP) & np.isfinite(aE)
        p = proxies(aP, aE, valid)
        truth = proxies(IP, IE, valid)
        smooth01 = proxies(s01P, s01E, valid)
        assign[f"{S}|{how}"] = {
            "vs_truth_pct": {k: sig(100 * (p[k] / truth[k] - 1)) for k in KEYS4},
            "vs_smooth_M200c_0.1_pct": {k: sig(100 * (p[k] / smooth01[k] - 1)) for k in KEYS4},
            "n_calibrated_haloes_without_bin": int((cal & ~valid).sum())}
res["painting_assignment_estimate"] = assign
res["notes"]["painting_assignment_estimate"] = (
    "ESTIMATE with integrated per-halo amounts (Y and N_e within 4 R200c, all 35 fit shells, mean profile only: no "
    "radial shape, no 2-halo term, no velocities).  Each calibrated halo (13.5-15.5) gets the bin-mean normalised "
    "integral of its scheme bin ('step') or the bin means interpolated linearly in log M200c to its own mass "
    "('smooth').  vs_truth: linear proxies (mean y, mean DM) measure the within-bin covariance of mass and profile "
    "(the per-halo truth has the same scatter); the quadratic proxies vs truth also contain the missing halo-to-halo "
    "scatter (1 + sigma^2), so compare quadratic proxies via vs_smooth_M200c_0.1.")

dump(res, WORK / "res_schemes.json")
print("done; records", len(records))

