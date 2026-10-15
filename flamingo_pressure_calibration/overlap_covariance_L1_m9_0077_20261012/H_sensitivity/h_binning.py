"""H5: propagation of the mass-binning choice alone (A2-ii stacks, no profile fit) to the observables.

For a binning scheme (stackcore.mass_bins: M200c_0.1 / 0.2 (reference) / 0.3 / 0.5 in log10 M200c, M500c_0.2 / 0.5 in
log10 M500c) the stack of scheme bin k is the inclusion-weighted mean A2-ii profile F_k(x) (x = r/R200c, fine
grid) of all haloes in the bin, at the bin's inclusion-weighted mean log mass mu_k (of the scheme variable).
Every calibrated FLAMINGO halo i (engine bins 13.5-15.5 in M200c) is painted with
  piecewise    F_i = F_k(i), k(i) the scheme bin containing the halo's mass variable v_i;
  interpolated ln F_i(s) linear in v between the two bin means bracketing v_i (constant beyond the first/last mean;
               linear in F where a stack value is not positive);
both truncated at R_cut = 1, 1.5, 2, 3, 4 R200c.  The painted population of engine bin c gives the moments the
lightcone engine needs: mean_c = sum_i w_i F_i, and second moments
  det : <F_s F_t>_c = sum_i w_i F_i,s F_i,t                     (no intrinsic scatter: only the mass-dependent
                                                                painting varies within the engine bin)
  S1  : <F_s F_t>_c = sum_i w_i exp(sigma2_k(i)) F_i,s F_i,t,   sigma2_k = ln <X^2>/<X>^2 of X = Y(<R_cut) (pressure)
        or N_e(<R_cut) (electrons) over the scheme bin's haloes; cross term <Y N>/(<Y><N>) (Test F / G's S1 recipe
        applied per scheme bin; interpolated in v for the interpolated painting),
w_i = inclusion weights normalised within engine bin c.  The truth (scatter-inclusive A2-ii, to 4 R200c) uses each
halo's own measured profile.  M200c_0.2 piecewise det equals the truth mean exactly (check).
DM quantiles: DM-PDF passes with one entry per distinct painted profile (piecewise: per scheme bin; interpolated:
haloes grouped in 0.02-dex cells of v, the group's mean painted profile; validated against per-halo entries for
M200c_0.1 at R_cut = 4); det at R_cut = 2, 3, 4 and S1 at R_cut = 4.
Outputs: results/binning.npz, results/binning_summary.json, tables/H5_binning.csv.
"""
from __future__ import annotations

import json

import h_common as HC
import h_pdf as HP
import h_truth as HT
import numpy as np

FC, FB = HC.FC, HC.FB
SCHEMES = ("M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5")
R_CUTS = (1.0, 1.5, 2.0, 3.0, 4.0)
PDF_DET_CUTS = (2.0, 3.0, 4.0)
GROUP_DEX = 0.02
MODES = ("piecewise", "interp")
OBS_KEYS = [f"yytot_{e}" for e in HC.SEL_ELL] + [f"ksz1h_{e}" for e in HC.SEL_ELL] + ["dm_mean", "dm_var", "ymean"] + \
           [f"ydmtot_{e}" for e in HC.SEL_ELL] + [f"dmcltot_{e}" for e in HC.SEL_ELL]


def scheme_bins(meta, scheme, pr_all):
    sd = meta["schemes"][scheme]
    var = "logm200c" if sd["variable"] == "logm200c" else "logm500c"
    return np.asarray(sd["edges"], float), var


def stacks(h0, rows_all, v_all, edges, cut_idx):
    """Per scheme bin: A2-ii mean P/E (fine), mean v, S1 moments at the R_cut aperture; NaN-free arrays."""
    w = h0["weight"]
    out = []
    vol = HC.VOL_FINE[:HC.N_X]
    for k in range(len(edges) - 1):
        r = rows_all[(v_all[rows_all] >= edges[k]) & (v_all[rows_all] < edges[k + 1])]
        if not len(r):
            continue
        ww = w[r] / w[r].sum()
        P = np.nan_to_num(h0["P_a2ii"][r], nan=0.0)
        E = np.nan_to_num(h0["E_a2ii"][r], nan=0.0)
        mP, mE = ww @ P, ww @ E
        s1 = {}
        for rc, k_cut in cut_idx.items():
            Y = P[:, :k_cut] @ vol[:k_cut]
            N = E[:, :k_cut] @ vol[:k_cut]
            s1[rc] = (np.log((ww @ Y**2) / (ww @ Y)**2), np.log((ww @ N**2) / (ww @ N)**2),
                      (ww @ (Y * N)) / ((ww @ Y) * (ww @ N)))
        out.append({"k": k, "lo": edges[k], "hi": edges[k + 1], "mu": float(ww @ v_all[r]), "P": mP, "E": mE,
                    "s1": s1, "N": int(len(r))})
    return out


def interp_profiles(sb, v, field):
    """ln-linear interpolation of the bin stacks in v at the masses v [n] -> [n, 67]."""
    mu = np.array([b["mu"] for b in sb])
    F = np.array([b[field] for b in sb])                      # [K, 67]
    order = np.argsort(mu)
    mu, F = mu[order], F[order]
    if len(mu) == 1:
        return np.repeat(F, len(v), axis=0)
    j = np.clip(np.searchsorted(mu, v) - 1, 0, len(mu) - 2)
    gap = mu[j + 1] - mu[j]
    t = np.clip(np.divide(v - mu[j], gap, out=np.zeros_like(v, dtype=float), where=gap > 0), 0.0, 1.0)
    a, b = F[j], F[j + 1]
    pos = (a > 0) & (b > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.exp(np.log(np.where(pos, a, 1.0)) * (1 - t[:, None]) + np.log(np.where(pos, b, 1.0)) * t[:, None])
    lin = a * (1 - t[:, None]) + b * t[:, None]
    return np.where(pos, lg, lin)


def interp_s1(sb, v, rc):
    mu = np.array([b["mu"] for b in sb])
    S = np.array([b["s1"][rc] for b in sb])                   # [K, 3]
    order = np.argsort(mu)
    if len(mu) == 1:
        return np.repeat(S, len(v), axis=0)
    return np.stack([np.interp(v, mu[order], S[order, q]) for q in range(3)], axis=1)


def painted(pr, sb, v_all, mode, rc, k_cut):
    """Per calibrated engine bin: per-halo painted P, E [n_c, 67], S1 [n_c, 3], weights, group labels (for the PDF)."""
    out = []
    mu = np.array([b["mu"] for b in sb])
    los = np.array([b["lo"] for b in sb])
    for c in range(FC.N_CAL):
        rows = HT.bin_rows(pr, c)
        hrows = pr["halo_rows"][rows]
        v = v_all[hrows]
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        if mode == "piecewise":
            j = np.clip(np.searchsorted(los, v, side="right") - 1, 0, len(sb) - 1)
            P = np.array([sb[q]["P"] for q in j])
            E = np.array([sb[q]["E"] for q in j])
            S1 = np.array([sb[q]["s1"][rc] for q in j])
            group = j
        else:
            P = interp_profiles(sb, v, "P")
            E = interp_profiles(sb, v, "E")
            S1 = interp_s1(sb, v, rc)
            group = np.floor(v / GROUP_DEX).astype(int)
        P = P.copy()
        E = E.copy()
        P[:, k_cut:] = 0.0
        E[:, k_cut:] = 0.0
        out.append({"P": P, "E": E, "S1": S1, "w": w, "group": group, "v": v, "mu": mu})
    return out


def moments_of(paint, scatter):
    nb, ns = FC.N_CAL, HC.N_X
    P = np.zeros((nb, ns))
    E = np.zeros((nb, ns))
    SPP = np.zeros((nb, ns, ns))
    SEE = np.zeros_like(SPP)
    SPE = np.zeros_like(SPP)
    for c, p in enumerate(paint):
        w = p["w"]
        P[c], E[c] = w @ p["P"], w @ p["E"]
        if scatter == "det":
            aP = aE = x = np.ones(len(w))
        else:
            aP, aE, x = np.exp(p["S1"][:, 0]), np.exp(p["S1"][:, 1]), p["S1"][:, 2]
        SPP[c] = np.einsum("h,hs,ht->st", w * aP, p["P"], p["P"])
        SEE[c] = np.einsum("h,hs,ht->st", w * aE, p["E"], p["E"])
        SPE[c] = np.einsum("h,hs,ht->st", w * x, p["P"], p["E"])
    return dict(P=P, E=E, SPP=SPP, SEE=SEE, SPE=SPE)


def pdf_entries(paint, scatter, per_halo=False):
    """Entries per calibrated bin: one per group (mean painted profile of the group, its weight share)."""
    out = []
    t, wq = np.polynomial.hermite.hermgauss(9)
    wq = wq / np.sqrt(np.pi)
    for p in paint:
        if per_halo:
            groups = [np.array([i]) for i in range(len(p["w"]))]
        else:
            groups = [np.flatnonzero(p["group"] == g) for g in np.unique(p["group"])]
        rows, wts = [], []
        for idx in groups:
            ww = p["w"][idx]
            W = ww.sum()
            F = (ww @ p["E"][idx]) / W
            if scatter == "det":
                rows.append(F[None, :])
                wts.append(np.array([W]))
            else:
                s2 = float((ww @ p["S1"][idx, 1]) / W)
                s = np.sqrt(max(s2, 0.0))
                amp = np.exp(s * np.sqrt(2) * t - s2 / 2)
                rows.append(amp[:, None] * F[None, :])
                wts.append(W * wq)
        out.append((np.concatenate(rows), np.concatenate(wts)))
    return out


def family_rms(err):
    out = {}
    for fam, keys in FC.FAMILIES.items():
        vals = [err[k] for k in keys if FC.primary(k) and k in err and np.isfinite(err[k])]
        out[fam] = float(np.sqrt(np.mean(np.square(vals)))) if vals else np.nan
    return out


def main():
    timer = HC.Timer()
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    h0 = HC.load_npz(HC.H0_PROFILES)
    meta = json.loads(HC.H0_META.read_text())
    tr = HC.load_npz(HC.RESULTS / "truth.npz")
    ts = HC.read_json(HC.RESULTS / "truth_summary.json")
    fallback = np.load(HC.WORK / "pdf_fallback_h.npz")
    fallback = (fallback["hist"], float(fallback["overflow"]))
    B = FB.Basis("HA")
    truth = dict(ts["observables"]["truth_a2ii"])
    tmean = dict(ts["observables"]["truthmean_a2ii"])
    for p, qn in enumerate(HC.Q_NAMES):
        truth[f"dm_{qn}"] = float(ts["pdf"]["truth_a2ii"]["quantiles"][p])
        tmean[f"dm_{qn}"] = float(ts["pdf"]["truthmean_a2ii"]["quantiles"][p])
    rows_all = np.arange(len(h0["weight"])) if not HC.SMOKE else np.asarray(pr["halo_rows"])
    cut_idx = {rc: HC.edge_index(HC.X200, rc) for rc in R_CUTS}
    out, summary, rows, specs, keys = {}, {"definitions": __doc__, "schemes": {}, "errors": {}}, [], [], []
    paints = {}
    for scheme in SCHEMES:
        edges, var = scheme_bins(meta, scheme, h0)
        v_all = np.asarray(h0[var], float)
        sb = stacks(h0, rows_all, v_all, edges, cut_idx)
        summary["schemes"][scheme] = {"variable": var, "bins": [{"lo": b["lo"], "hi": b["hi"], "mu": b["mu"], "N": b["N"]}
                                                                  for b in sb]}
        out[f"stack_mu__{scheme}"] = np.array([b["mu"] for b in sb])
        out[f"stack_P__{scheme}"] = np.array([b["P"] for b in sb])
        out[f"stack_E__{scheme}"] = np.array([b["E"] for b in sb])
        for mode in MODES:
            for rc in R_CUTS:
                paint = painted(pr, sb, v_all, mode, rc, cut_idx[rc])
                for scatter in ("det", "S1"):
                    mom = moments_of(paint, scatter)
                    sc_ = B.scalars(B.observables(**mom))
                    err_truth = {k: sc_[k] / truth[k] - 1 for k in sc_ if k in truth and truth[k]}
                    err_mean = {k: sc_[k] / tmean[k] - 1 for k in sc_ if k in tmean and tmean[k]}
                    lab = f"{scheme}|{mode}|{rc:g}|{scatter}"
                    summary["errors"][lab] = {"vs_truth": {k: err_truth[k] for k in OBS_KEYS if k in err_truth},
                                              "vs_truth_mean": {k: err_mean[k] for k in OBS_KEYS if k in err_mean}}
                    if scheme == "M200c_0.2" and mode == "piecewise" and rc == 4.0 and scatter == "det":
                        ref = HT.cal_moments(pr, "a2ii")
                        summary["reference_check_mean_profiles_max_rel"] = max(
                            HT.rel(mom[k][ref[k] != 0], ref[k][ref[k] != 0]) for k in ("P", "E"))
                        summary["reference_check_observables_max_rel_vs_truthmean"] = max(
                            abs(err_mean[k]) for k in err_mean)
                    if (scatter == "det" and rc in PDF_DET_CUTS) or (scatter == "S1" and rc == 4.0):
                        specs.append((lab, pdf_entries(paint, scatter)))
                        keys.append(lab)
                    if scheme == "M200c_0.1" and mode == "interp" and rc == 4.0 and scatter == "det":
                        specs.append((lab + "|perhalo", pdf_entries(paint, scatter, per_halo=True)))
                        keys.append(lab + "|perhalo")
        HC.log(f"binning moments {scheme}", timer.mark(scheme))
    del B                                      # no basis in memory at the DM-PDF fork
    HC.mem("binning after moments")
    res, _, tim = HP.run_specs(specs, "HA", fallback=fallback, label="pdf binning")
    for lab, s in res.items():
        q = np.asarray(s["quantiles"], float)
        out[f"dmq__{lab}"] = q
        base = lab.replace("|perhalo", "")
        e = summary["errors"][base]
        tagt = "vs_truth" if not lab.endswith("perhalo") else "vs_truth_perhalo"
        tagm = "vs_truth_mean" if not lab.endswith("perhalo") else "vs_truth_mean_perhalo"
        e.setdefault(tagt, {})
        e.setdefault(tagm, {})
        for p, qn in enumerate(HC.Q_NAMES):
            e[tagt][f"dm_{qn}"] = float(q[p] / truth[f"dm_{qn}"] - 1)
            e[tagm][f"dm_{qn}"] = float(q[p] / tmean[f"dm_{qn}"] - 1)
    summary["pdf_timing"] = tim
    for lab, e in summary["errors"].items():
        scheme, mode, rc, scatter = lab.split("|")
        fr = family_rms(e["vs_truth"])
        fm = family_rms(e["vs_truth_mean"])
        e["family_rms_vs_truth"] = fr
        e["family_rms_vs_truth_mean"] = fm
        e["dm_quantiles_included"] = "dm_q50" in e["vs_truth"]
        rows.append(dict(scheme=scheme, mode=mode, R_cut=float(rc), scatter=scatter,
                         **{f"rms_{k}_vs_truth": v for k, v in fr.items()},
                         **{f"rms_{k}_vs_truthmean": v for k, v in fm.items()},
                         **{f"err_{k}": v for k, v in e["vs_truth"].items()}))
    # grouping validation (interpolated painting, per-halo vs 0.02-dex groups)
    lab = "M200c_0.1|interp|4|det"
    if lab in summary["errors"] and "vs_truth_perhalo" in summary["errors"][lab]:
        e = summary["errors"][lab]
        summary["grouping_check_max_abs_quantile_diff"] = max(
            abs(e["vs_truth"][f"dm_{qn}"] - e["vs_truth_perhalo"][f"dm_{qn}"]) for qn in HC.Q_NAMES)
    np.savez_compressed(HC.RESULTS / "binning.npz", **out)
    summary["seconds"] = timer.mark("total")
    HC.write_json(HC.RESULTS / "binning_summary.json", summary)
    HC.write_csv(HC.TABLES / "H5_binning.csv", rows)
    HC.log("binning done", summary["seconds"])
    return summary


if __name__ == "__main__":
    main()
