"""H3: where the signal of each probe comes from, at the fine resolution (67 X200 shells to 4 R200c), per engine mass bin.

Definitions (x = r/R200c; F_{b,s} = the inclusion-weighted mean profile of mass bin b in fine shell s; every halo of
bin b is scaled by (1 + eps) in shell s for a derivative "d ln F_{b,s}"):
  importance        I_O[b, s] = (dO/d ln F_{b,s}) / (k O),  k = 2 for quadratic observables (auto spectra 1h and 2h,
                    kSZ, DM variance, DM-field C_ell), 1 for means, DM quantiles and each side of the y x DM cross.
                    Calibrated bins (13.5-15.5): exact in-basis Jacobians at the scatter-inclusive truth; fallback
                    bins (published B12/B16 below 10^13.5 and above 10^15.388): the kernel builder's per-(bin, shell)
                    pieces; DM quantiles: Mecke Jacobians (per-halo subsample and truth mean).
                    Sum rule: O is homogeneous of degree k in the profiles, so sum_{b,s} I_O[b, s] = 1 (Euler).
  contribution      linear terms F_s K_s / O; quadratic 1-halo terms sum_t K_st <F_s F_t> / O (the symmetric split
                    of the cross terms).  By Euler's theorem this equals the importance exactly; the diagonal part
                    K_ss <F_s^2> / O (the shell's own signal) and the cross part are stored separately.
  truncation curve  T_O(R_t) = O[calibrated haloes cut at R_t] / O[truth to 4 R200c] (fallback haloes unchanged, as in
                    Test F's / G's R_cut evaluations); calibrated-signal cumulative fraction
                    C_O(R_t) = (O(R_t) - O(0)) / (O(4R) - O(0)), O(0) = no calibrated haloes; per-bin curves for the
                    1-halo and linear terms (each bin's own contribution, exact) and for every observable with only
                    that bin cut ("bin-cut" curves).
  radius R_X        first radius where C_O reaches X = 50 / 80 / 90 / 95 / 99% (linear in ln x between edges).
  tolerance         the coherent fractional error of the stacked profile in one fine shell (all calibrated bins)
                    that changes O by 1%:  tol_s = 0.01 / |k sum_b I_O[b, s]|; per 0.1 dex of radius:
                    tol(x) = 0.01 / |k i(x) 0.1 ln 10| with the importance density i(x) = sum_b I_O[b, s] / dln x_s.
DM quantile truncation curves come from DM-PDF passes: the truth mean cut at every fine edge, the per-halo truth
cut at selected radii (A2-ii) and at 1/2/3 R200c (all gas).
Outputs: results/sens.npz, results/sens_summary.json, tables/H3_*.csv.
"""
from __future__ import annotations

import h_common as HC
import h_pdf as HP
import h_truth as HT
import numpy as np

FC, FB = HC.FC, HC.FB
import f1_importance as FI  # noqa: E402

RADII_X = (0.5, 0.8, 0.9, 0.95, 0.99)
PDF_HALO_CUTS = {"a2ii": (0.5185, 1.0, 1.5, 2.0, 3.0) if not HC.SMOKE else (1.0, 2.0),
                 "main": (1.0, 2.0) if not HC.SMOKE else (2.0,)}
# observables: name -> (family, jacobian key, truth key, k, truncated field(s), description)
OBS = {
    "yytot": ("tSZ", "yytot", "yytot", 2, "P", "tSZ C_ell^yy total"),
    "yy1h": ("tSZ", "yy1h", "yy1h", 2, "P", "tSZ C_ell^yy 1-halo"),
    "yy2h": ("tSZ", "yy2h", "yy2h", 2, "P", "tSZ C_ell^yy 2-halo"),
    "ymean": ("tSZ", "ymean", "ymean", 1, "P", "mean Compton y"),
    "ksz1h": ("kSZ", "ksz1h", "ksz1h", 2, "E", "kSZ C_ell 1-halo (catalogue v_LOS)"),
    "ksz2h": ("kSZ", "ksz2h", "ksz2h", 2, "E", "kSZ C_ell 2-halo (approximate)"),
    "ksztot": ("kSZ", "ksztot", "ksztot", 2, "E", "kSZ C_ell 1h + 2h"),
    "dm_mean": ("DM", "dm_mean", "dm_mean", 1, "E", "mean halo DM (z_s = 1)"),
    "dm_var": ("DM", "dm_var", "dm_var", 2, "E", "DM variance (compound Poisson)"),
    "dmcltot": ("DM", "dmcltot", "dmcltot", 2, "E", "DM-field C_ell total"),
    "dmcl1h": ("DM", "dmcl1h", "dmcl1h", 2, "E", "DM-field C_ell 1-halo"),
    "dmcl2h": ("DM", "dmcl2h", "dmcl2h", 2, "E", "DM-field C_ell 2-halo"),
    "ydmtot_P": ("tSZxDM", "ydmtot_P", "ydmtot", 1, "P", "y x DM C_ell total, pressure side"),
    "ydmtot_E": ("tSZxDM", "ydmtot_E", "ydmtot", 1, "E", "y x DM C_ell total, electron side"),
    "ydm1h_P": ("tSZxDM", "ydm1h_P", "ydm1h", 1, "P", "y x DM C_ell 1-halo, pressure side"),
    "ydm1h_E": ("tSZxDM", "ydm1h_E", "ydm1h", 1, "E", "y x DM C_ell 1-halo, electron side"),
}
FB_EXTRA = {"ksztot": ("ksz1h", "ksz2h")}


def fallback_J(B, key):
    if key in FB_EXTRA:
        return sum(FI.fallback_J(B, k) for k in FB_EXTRA[key])
    return FI.fallback_J(B, key)


def full_map(B, J, key):
    out = np.array(fallback_J(B, key), float, copy=True)
    for c, b in enumerate(FC.CAL_BINS):
        out[b] += J[key][c, :HC.N_X]
    return out


# ============================================================================ truncation curves (prefix sums)
def _cum2(A):
    """C[c, k, (l)] = sum_{s, t < k} A[c, s, t, (l)] for k = 0..ns (A [c, s, t(, l)])."""
    C = A.cumsum(axis=1).cumsum(axis=2)
    diag = np.stack([C[:, k - 1, k - 1] for k in range(1, A.shape[1] + 1)], axis=1)
    return np.concatenate([np.zeros_like(diag[:, :1]), diag], axis=1)


def _cum2_side(A, side):
    """Cross form with one side cut: side 'P' -> sum_{s<k, all t}; 'E' -> sum_{all s, t<k}."""
    if side == "P":
        C = A.sum(axis=2).cumsum(axis=1)
    else:
        C = A.sum(axis=1).cumsum(axis=1)
    return np.concatenate([np.zeros_like(C[:, :1]), C], axis=1)


def truncation_curves(B, mom):
    """All moment observables with the calibrated profiles cut at every fine edge (k = 0..67 shells kept), for the
    all-bin cut and for single-bin cuts.  Returns {'all': {obs: [68(, l)]}, 'bin': {obs: [N_CAL, 68(, l)]},
    'own': {obs: [N_CAL, 68(, l)]} (each bin's own 1h / linear contribution)}."""
    ns = HC.N_X
    P, E = B.pad(mom["P"])[:, :ns], B.pad(mom["E"])[:, :ns]
    SPP, SEE, SPE = (B.pad2(mom[k])[:, :ns, :ns] for k in ("SPP", "SEE", "SPE"))
    K = {k: getattr(B, k)[:, :ns, :ns] for k in ("K_yy1h", "K_ksz1h", "K_dm1h", "K_ydm1h")}
    own = {}
    own["yy1h"] = _cum2(K["K_yy1h"] * SPP[..., None])
    own["ksz1h"] = _cum2(K["K_ksz1h"] * SEE[..., None])
    own["dmcl1h"] = _cum2(K["K_dm1h"] * SEE[..., None])
    A = K["K_ydm1h"] * SPE[..., None]
    own["ydm1h"] = _cum2(A)
    own["ydm1h_Pcut"] = _cum2_side(A, "P")
    own["ydm1h_Ecut"] = _cum2_side(A, "E")
    own["dm_var"] = _cum2((B.K_dmvar[:, :ns, :ns] * SEE)[..., None])[..., 0]
    zero = np.zeros((FC.N_CAL, 1))
    own["ymean"] = np.concatenate([zero, np.cumsum(P * B.mean_y[:, :ns], axis=1)], axis=1)
    own["dm_mean"] = np.concatenate([zero, np.cumsum(E * B.dm_mean[:, :ns], axis=1)], axis=1)
    full1 = {k: v[:, -1] for k, v in own.items()}
    fb = B.fb
    out_all, out_bin = {}, {}
    # ---- 1-halo / linear parts (fallback constant)
    base = {"yy1h": fb["yy1h"], "ksz1h": fb["ksz1h"], "dmcl1h": fb["dm1h"], "ydm1h": fb["ydm1h"],
            "ydm1h_Pcut": fb["ydm1h"], "ydm1h_Ecut": fb["ydm1h"], "dm_var": fb["dm_var"], "ymean": fb["mean_y"],
            "dm_mean": fb["dm_mean"]}
    for k, v in own.items():
        out_all[k] = base[k] + v.sum(axis=0)
        tot = base[k] + full1[k].sum(axis=0)
        out_bin[k] = tot[None] - full1[k][:, None] + v
    # ---- 2-halo windows: prefix sums over shells of F_cs W[z, c, s, l]
    GP = np.cumsum(np.einsum("cs,zcsl->zcsl", P, B.W_y[:, :, :ns], optimize=True), axis=2)
    GE = np.cumsum(np.einsum("cs,zcsl->zcsl", E, B.W_tau[:, :, :ns], optimize=True), axis=2)
    nz, _, _, nl = GP.shape
    zP = np.zeros((nz, FC.N_CAL, 1, nl))
    GP = np.concatenate([zP, GP], axis=2)                 # [z, c, k = 0..67, l]
    GE = np.concatenate([zP, GE], axis=2)
    WPf = B.W_fb_y + GP[:, :, -1].sum(axis=1)             # full windows [z, l]
    WEf = B.W_fb_tau + GE[:, :, -1].sum(axis=1)
    pv, s2, dz = B.pv, B.s2_lin[:, None], B.dmz[:, None]

    def two_h(WP, WE):
        return {"yy2h": np.sum(pv * WP * WP, axis=0), "ksz2h": np.sum(s2 * pv * WE * WE, axis=0),
                "dmcl2h": np.sum(pv * (dz * WE)**2, axis=0), "ydm2h": np.sum(pv * WP * dz * WE, axis=0),
                "ydm2h_Pcut": np.sum(pv * WP * dz * WEf, axis=0), "ydm2h_Ecut": np.sum(pv * WPf * dz * WE, axis=0)}

    nk = ns + 1
    two_all = {k: np.zeros((nk, nl)) for k in ("yy2h", "ksz2h", "dmcl2h", "ydm2h", "ydm2h_Pcut", "ydm2h_Ecut")}
    two_bin = {k: np.zeros((FC.N_CAL, nk, nl)) for k in two_all}
    for k in range(nk):
        WP = B.W_fb_y + GP[:, :, k].sum(axis=1)
        WE = B.W_fb_tau + GE[:, :, k].sum(axis=1)
        for name, v in two_h(WP, WE).items():
            two_all[name][k] = v
        for c in range(FC.N_CAL):
            WPc = WPf - GP[:, c, -1] + GP[:, c, k]
            WEc = WEf - GE[:, c, -1] + GE[:, c, k]
            for name, v in two_h(WPc, WEc).items():
                two_bin[name][c, k] = v
    out_all.update(two_all)
    out_bin.update(two_bin)
    # ---- totals
    for tot, a, b in (("yytot", "yy1h", "yy2h"), ("ksztot", "ksz1h", "ksz2h"), ("dmcltot", "dmcl1h", "dmcl2h"),
                      ("ydmtot", "ydm1h", "ydm2h"), ("ydmtot_Pcut", "ydm1h_Pcut", "ydm2h_Pcut"),
                      ("ydmtot_Ecut", "ydm1h_Ecut", "ydm2h_Ecut")):
        out_all[tot] = out_all[a] + out_all[b]
        out_bin[tot] = out_bin[a] + out_bin[b]
    return {"all": out_all, "bin": out_bin, "own": own}


def radii_containing(xr, frac_curve):
    return {f"R{int(round(100 * X))}": HC.interp_crossing(xr, frac_curve, X) for X in RADII_X}


# ============================================================================ main
def main():
    timer = HC.Timer()
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    tr = HC.load_npz(HC.RESULTS / "truth.npz")
    ts = HC.read_json(HC.RESULTS / "truth_summary.json")
    xr = HC.X200[1:]                                  # truncation radii (outer edges), k = 1..67
    xk = HC.X200                                      # k = 0..67 (k shells kept: R_t = X200[k])
    dln = HC.dlnx(HC.X200)
    li = [int(np.flatnonzero(FC.ELL == e)[0]) for e in HC.SEL_ELL]
    out = {"x_edges": HC.X200, "dlnx": dln, "ell": FC.ELL, "sel_ell": np.array(HC.SEL_ELL), "engine_bin_labels":
           np.array(HC.ENGINE_LABELS), "cal_bins": np.array(FC.CAL_BINS), "quantile_levels": HC.QUANTILES}
    summary = {"definitions": __doc__, "radii": {}, "shares": {}, "sum_rules": {}, "tolerance": {}, "peaks": {},
               "bands": {}, "self_fraction": {}}
    rows_r, rows_tol, rows_share, rows_band = [], [], [], []
    bands = ((0, .1051), (.1051, .2335), (.2335, .5185), (.5185, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))
    band_labels = ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")
    mid = np.sqrt(np.maximum(HC.X200[:-1], 1e-6) * HC.X200[1:])
    band_of = np.searchsorted(np.array([b[1] for b in bands]), mid, side="left")
    fallback = np.load(HC.WORK / "pdf_fallback_h.npz")
    fallback = (fallback["hist"], float(fallback["overflow"]))
    cal_mask = np.zeros(16, bool)
    cal_mask[FC.CAL_BINS] = True
    for sel in HC.SELECTIONS:
        B = FB.Basis(HT.BASIS_OF[sel])
        mom = HT.cal_moments(pr, sel)
        J = B.jacobians(**mom)
        J["ksztot"] = J["ksz1h"] + J["ksz2h"]
        Jd = B.jacobians(mom["P"], mom["E"])                 # deterministic (truth mean)
        Jd["ksztot"] = Jd["ksz1h"] + Jd["ksz2h"]
        O = {k: tr[f"truth_{sel}__{k}"] for k in ("yytot", "yy1h", "yy2h", "ymean", "ksz1h", "ksz2h", "ksztot",
                                                     "dm_mean", "dm_var", "dmcltot", "dmcl1h", "dmcl2h", "ydmtot", "ydm1h")}
        Od = {k: tr[f"truthmean_{sel}__{k}"] for k in O}
        # ---------------- importance maps (scatter-inclusive and deterministic)
        for name, (fam, jkey, tkey, k, fld, desc) in OBS.items():
            for tag, JJ, OO in (("imp", J, O), ("impdet", Jd, Od)):
                M = full_map(B, JJ, jkey)
                Ov = OO[tkey]
                norm = M / (k * (Ov if np.ndim(Ov) == 0 else Ov[None, None, :]))
                out[f"{tag}__{sel}__{name}"] = norm
                s = norm.sum(axis=(0, 1))
                summary["sum_rules"][f"{tag}__{sel}__{name}"] = float(np.max(np.abs(s - 1)))
        # quantiles (Mecke): per-halo subsample and truth mean
        for tag, lab in (("imp", f"truth_{sel}_subsample"), ("impdet", f"truthmean_{sel}")):
            Jq, Jqfb = tr[f"Jq_{lab}"], tr[f"Jqfb_{lab}"]
            Q = np.asarray(ts["pdf"][lab.replace("_subsample", "")]["quantiles"], float)
            for p, qn in enumerate(HC.Q_NAMES):
                M = np.array(Jqfb[p], float, copy=True)
                for c, b in enumerate(FC.CAL_BINS):
                    M[b] += Jq[p, c, :HC.N_X]
                norm = M / Q[p]
                out[f"{tag}__{sel}__dm_{qn}"] = norm
                summary["sum_rules"][f"{tag}__{sel}__dm_{qn}"] = float(abs(norm.sum() - 1))
        # ---------------- contribution split for the 1-halo forms (diagonal = the shell's own signal)
        for name, Kname, Sname, tkey in (("yy1h", "K_yy1h", "SPP", "yy1h"), ("ksz1h", "K_ksz1h", "SEE", "ksz1h"),
                                         ("dmcl1h", "K_dm1h", "SEE", "dmcl1h")):
            Kd = np.einsum("cssl->csl", getattr(B, Kname)[:, :HC.N_X, :HC.N_X])
            Sd = np.einsum("css->cs", B.pad2(mom[Sname])[:, :HC.N_X, :HC.N_X])
            diag = Kd * Sd[..., None] / O[tkey][None, None, :]
            out[f"self__{sel}__{name}"] = diag
            # share of the calibrated haloes' 1-halo signal carried by the diagonal (same-shell) terms
            summary["self_fraction"][f"{sel}__{name}"] = {str(e): float(diag[..., i].sum() * O[tkey][i]
                                                                       / tr[f"truthcal_{sel}__{name}"][i])
                                                         for e, i in zip(HC.SEL_ELL, li)}
        Kd = np.einsum("css->cs", B.K_dmvar[:, :HC.N_X, :HC.N_X])
        out[f"self__{sel}__dm_var"] = Kd * np.einsum("css->cs", B.pad2(mom["SEE"])[:, :HC.N_X, :HC.N_X]) / O["dm_var"]
        # ---------------- truncation curves (moment observables)
        tc = truncation_curves(B, mom)
        for scope in ("all", "bin", "own"):
            for name, v in tc[scope].items():
                out[f"trunc_{scope}__{sel}__{name}"] = v
        # check: the k = 67 point equals the truth
        chk = {}
        for name in ("yytot", "ksz1h", "ksztot", "dm_mean", "dm_var", "dmcltot", "ydmtot", "ymean"):
            chk[name] = float(np.max(np.abs(tc["all"][name][-1] / O[name] - 1)))
        summary["sum_rules"][f"truncation_endpoint__{sel}"] = chk
        # zero point vs the fallback-only observables
        calonly = {k: tr[f"truthcal_{sel}__{k}"] for k in O}
        chk0 = {}
        for name in ("yy1h", "ksz1h", "dm_mean", "dm_var", "ymean"):
            fbv = O[name] - calonly[name]
            if name in ("dm_mean", "ymean", "dm_var"):
                chk0[name] = float(abs(tc["all"][name][0] / fbv - 1))
            else:
                chk0[name] = float(np.max(np.abs(tc["all"][name][0] / fbv - 1)))
        summary["sum_rules"][f"truncation_zero_point__{sel}"] = chk0
        # ---------------- radii containing X% (calibrated signal) and the truncation loss
        summary["radii"][sel] = {}
        for name in ("yytot", "yy1h", "yy2h", "ymean", "ksz1h", "ksz2h", "ksztot", "dm_mean", "dm_var", "dmcltot",
                     "dmcl1h", "ydmtot", "ydmtot_Pcut", "ydmtot_Ecut"):
            curve = tc["all"][name]
            if curve.ndim == 1:
                C = (curve - curve[0]) / (curve[-1] - curve[0])
                T = curve / curve[-1]
                entries = {"scalar": (C, T)}
            else:
                entries = {}
                for e, i in zip(HC.SEL_ELL, li):
                    c_ = curve[:, i]
                    entries[str(e)] = ((c_ - c_[0]) / (c_[-1] - c_[0]), c_ / c_[-1])
            for at, (C, T) in entries.items():
                rad = radii_containing(xk, C)
                loss = {f"loss_at_{r:g}R": float(1 - T[HC.edge_index(xk, r)]) for r in (1.0, 1.5, 2.0, 2.5, 3.0)}
                summary["radii"][sel][f"{name}@{at}"] = dict(rad, **loss)
                rows_r.append(dict(selection=sel, observable=name, ell=at, scope="all bins", mass_bin="all", **rad,
                                   **loss))
            # per-bin (own contribution for 1h / linear terms)
            if name in tc["own"]:
                own = tc["own"][name]
                for c in range(FC.N_CAL):
                    if own.ndim == 2:
                        C = (own[c] - own[c, 0]) / (own[c, -1] - own[c, 0])
                        rad = radii_containing(xk, C)
                        rows_r.append(dict(selection=sel, observable=name, ell="scalar", scope="own bin",
                                           mass_bin=HC.CAL_LABELS[c], **rad))
                    else:
                        for e, i in zip(HC.SEL_ELL, li):
                            C = (own[c, :, i] - own[c, 0, i]) / (own[c, -1, i] - own[c, 0, i])
                            rad = radii_containing(xk, C)
                            rows_r.append(dict(selection=sel, observable=name, ell=str(e), scope="own bin",
                                               mass_bin=HC.CAL_LABELS[c], **rad))
        # ---------------- tolerance, shares, peaks, bands
        names = list(OBS) + [f"dm_{q}" for q in HC.Q_NAMES]
        for name in names:
            M = out[f"imp__{sel}__{name}"]
            k = OBS[name][3] if name in OBS else 1
            ats = [(str(e), i) for e, i in zip(HC.SEL_ELL, li)] if M.ndim == 3 else [("scalar", None)]
            for at, i in ats:
                m = M[..., i] if i is not None else M
                per_shell = m[cal_mask].sum(axis=0)                  # calibrated bins, per fine shell
                dens = m.sum(axis=0) / dln                           # all bins, per unit ln x
                with np.errstate(divide="ignore"):
                    tol_shell = 0.01 / np.abs(k * per_shell)
                    tol_01dex = 0.01 / np.abs(k * (per_shell / dln) * 0.1 * np.log(10))
                out[f"tol_shell__{sel}__{name}@{at}"] = tol_shell
                out[f"tol_01dex__{sel}__{name}@{at}"] = tol_01dex
                j = int(np.argmin(tol_01dex))
                summary["tolerance"][f"{sel}__{name}@{at}"] = {
                    "min_tol_per_0.1dex": float(tol_01dex[j]), "at_x": float(mid[j]),
                    "tol_per_0.1dex_at": {f"{r:g}": float(np.interp(np.log(r), np.log(mid), tol_01dex))
                                          for r in (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 3.8)}}
                jp = int(np.argmax(dens))
                summary["peaks"][f"{sel}__{name}@{at}"] = {"x_peak_density": float(mid[jp]),
                                                           "peak_density_per_lnx": float(dens[jp])}
                shares = m.sum(axis=1)
                summary["shares"][f"{sel}__{name}@{at}"] = {HC.ENGINE_LABELS[b]: float(shares[b]) for b in range(16)}
                summary["shares"][f"{sel}__{name}@{at}"]["calibrated"] = float(shares[cal_mask].sum())
                bsum = np.array([m[:, band_of == q].sum() for q in range(len(bands))])
                summary["bands"][f"{sel}__{name}@{at}"] = dict(zip(band_labels, bsum.tolist()))
                for b in range(16):
                    rows_share.append(dict(selection=sel, observable=name, ell=at, mass_bin=HC.ENGINE_LABELS[b],
                                           calibrated=bool(cal_mask[b]), share=float(shares[b])))
                for q, lab in enumerate(band_labels):
                    rows_band.append(dict(selection=sel, observable=name, ell=at, band_R200c=lab,
                                          importance=float(bsum[q])))
                for s in range(HC.N_X):
                    rows_tol.append(dict(selection=sel, observable=name, ell=at, x_lo=HC.X200[s], x_hi=HC.X200[s + 1],
                                         importance_calibrated=float(per_shell[s]), importance_all=float(m[:, s].sum()),
                                         density_per_lnx=float(dens[s]), tol_shell=float(tol_shell[s]),
                                         tol_per_0p1dex=float(tol_01dex[s])))
        HC.log(f"sens moments {sel}", timer.mark(f"mom_{sel}"))
        del B, J, Jd, tc                       # no basis in memory at the DM-PDF forks below
        HC.mem(f"sens after moments {sel}")
    # ---------------- DM quantile truncation curves (PDF passes)
    zero_entries = [(np.zeros(HC.N_X), 1.0) for _ in range(FC.N_CAL)]
    for sel in HC.SELECTIONS:
        which = HT.BASIS_OF[sel]
        mom = HT.cal_moments(pr, sel)
        rows_bin = [HT.bin_rows(pr, c) for c in range(FC.N_CAL)]
        w_bin = [pr["weight"][r] / pr["weight"][r].sum() for r in rows_bin]
        specs = [("zero", zero_entries)]
        for k in range(1, HC.N_X):                                    # truth mean cut at every edge below 4R
            Ek = mom["E"].copy()
            Ek[:, k:] = 0.0
            specs.append((f"mean_k{k}", HP.det_entries(Ek)))
        for r in PDF_HALO_CUTS[sel]:
            k = HC.edge_index(HC.X200, r, tol=1e-3)
            per = []
            for c in range(FC.N_CAL):
                v = np.nan_to_num(pr[f"E_{sel}"][rows_bin[c]], nan=0.0).copy()
                v[:, k:] = 0.0
                per.append((v, w_bin[c]))
            specs.append((f"halo_k{k}", per))
        res, _, tim = HP.run_specs(specs, which, fallback=fallback, label=f"pdf cuts {sel}")
        qmean = np.zeros((HC.N_X + 1, len(HC.QUANTILES)))
        qmean[0] = res["zero"]["quantiles"]
        for k in range(1, HC.N_X):
            qmean[k] = res[f"mean_k{k}"]["quantiles"]
        qmean[HC.N_X] = ts["pdf"][f"truthmean_{sel}"]["quantiles"]
        out[f"trunc_all__{sel}__dmq_mean"] = qmean
        mo_mean = np.array([[res[f"mean_k{k}"]["mean"], res[f"mean_k{k}"]["variance"]] for k in range(1, HC.N_X)])
        out[f"trunc_all__{sel}__dmmoments_mean"] = mo_mean
        ks = [HC.edge_index(HC.X200, r, tol=1e-3) for r in PDF_HALO_CUTS[sel]]
        qh = np.array([res[f"halo_k{k}"]["quantiles"] for k in ks] + [ts["pdf"][f"truth_{sel}"]["quantiles"]])
        out[f"trunc_halo__{sel}__dmq"] = qh
        out[f"trunc_halo__{sel}__k"] = np.array(ks + [HC.N_X])
        out[f"dmq_zero__{sel}"] = np.asarray(res["zero"]["quantiles"], float)
        summary.setdefault("pdf_timing", {})[sel] = tim
        for p, qn in enumerate(HC.Q_NAMES):
            C = (qmean[:, p] - qmean[0, p]) / (qmean[-1, p] - qmean[0, p])
            rad = radii_containing(xk, C)
            T = qmean[:, p] / qmean[-1, p]
            loss = {f"loss_at_{r:g}R": float(1 - T[HC.edge_index(xk, r)]) for r in (1.0, 1.5, 2.0, 2.5, 3.0)}
            summary["radii"][sel][f"dm_{qn}@truthmean"] = dict(rad, **loss)
            rows_r.append(dict(selection=sel, observable=f"dm_{qn}", ell="scalar", scope="all bins (truth mean)",
                               mass_bin="all", **rad, **loss))
            Th = qh[:, p] / qh[-1, p]
            summary["radii"][sel][f"dm_{qn}@per_halo_loss"] = {f"loss_at_{HC.X200[k]:.4g}R": float(1 - Th[j])
                                                               for j, k in enumerate(ks)}
        HC.log(f"pdf cuts {sel}", timer.mark(f"pdf_{sel}"))
    np.savez_compressed(HC.RESULTS / "sens.npz", **out)
    summary["seconds"] = timer.mark("total")
    HC.write_json(HC.RESULTS / "sens_summary.json", summary)
    HC.write_csv(HC.TABLES / "H3_radii.csv", rows_r)
    HC.write_csv(HC.TABLES / "H3_importance_shells.csv", rows_tol)
    HC.write_csv(HC.TABLES / "H3_mass_shares.csv", rows_share)
    HC.write_csv(HC.TABLES / "H3_bands.csv", rows_band)
    HC.log("sens done", summary["seconds"])
    return summary


if __name__ == "__main__":
    main()
