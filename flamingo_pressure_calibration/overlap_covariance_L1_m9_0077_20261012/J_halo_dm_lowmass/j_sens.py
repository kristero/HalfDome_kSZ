"""J stage 'sens' (J1d): where the signal of each probe comes from under each halo-component definition D0-D3, at the
finest radial resolution (67 X200 shells to 4 R200c), per engine mass bin, with the low-mass bins 12.7-13.5 calibrated
(configuration C14, treatment T1).

Definitions (x = r/R200c; F_{b,s} = the inclusion-weighted mean profile of mass bin b in fine shell s; a derivative
"d ln F_{b,s}" scales every halo of bin b by (1 + eps) in shell s):
  importance     I_O[b, s] = (dO/d ln F_{b,s}) / (k_O O), k_O = 2 for quadratic observables (auto spectra, kSZ, DM
                 variance, DM-field C_ell), 1 for means, DM quantiles and each side of y x DM.  Sum rule (Euler):
                 sum_{b,s} I_O[b, s] = 1.  Calibrated bins: exact in-basis Jacobians at the scatter-inclusive truth
                 (f_basis); fallback haloes above 10^15.388: the per-(bin, shell) pieces of j_fallback with the
                 definition's cut and total 2-halo windows; DM quantiles: Mecke Jacobians (truth stage).
                 Shells beyond a definition's boundary have F = 0, hence I = 0 there.
  truncation     T_O(R_t) = O[calibrated haloes cut in 3D at R_t] / O[definition], fallback haloes unchanged;
                 C_O(R_t) = (O(R_t) - O(0)) / (O(R_D) - O(0)), O(0) = no calibrated halo, R_D = the boundary of the
                 definition (1, 2 or 4 R200c): C_O(R_D) = 1 by construction.
  radius R_X     first radius where C_O reaches X = 50/80/90/95/99% (linear in ln x between fine edges).
  tolerance      the coherent fractional error of the stacks in one fine shell (all calibrated bins) that changes O
                 by 1%: tol_s = 0.01/|k_O sum_b I_O[b, s]|; per 0.1 dex: tol(x) = 0.01/|k_O i(x) 0.1 ln 10|,
                 i(x) = sum_b I_O[b, s]/dln x_s (calibrated bins).
DM-quantile truncation curves come from DM-PDF passes: the bin-mean profiles cut at every fine edge inside the boundary,
and the per-halo truth cut at selected radii.
Outputs: results/sens.npz (maps at the selected multipoles, float32), results/sens_summary.json, tables/J1d_*.csv.
"""
from __future__ import annotations

import j_common as JC
import j_fallback as JF
import j_obs as JO
import j_pdf as JP
import j_truth as JT
import numpy as np

FC, FB, FI = JC.FC, JC.FB, JC.FI
RADII_X = (0.5, 0.8, 0.9, 0.95, 0.99)
# name -> (family, jacobian key, truth key, k)
OBS = {"yytot": ("tSZ", "yytot", "yytot", 2), "ymean": ("tSZ", "ymean", "ymean", 1),
       "ksz1h": ("kSZ", "ksz1h", "ksz1h", 2), "ksz2h": ("kSZ", "ksz2h", "ksz2h", 2),
       "ksztot": ("kSZ", "ksztot", "ksztot", 2), "dm_mean": ("DM", "dm_mean", "dm_mean", 1),
       "dm_var": ("DM", "dm_var", "dm_var", 2), "dmcltot": ("DM", "dmcltot", "dmcltot", 2),
       "dmcl1h": ("DM", "dmcl1h", "dmcl1h", 2), "ydmtot_P": ("tSZxDM", "ydmtot_P", "ydmtot", 1),
       "ydmtot_E": ("tSZxDM", "ydmtot_E", "ydmtot", 1)}
E_SIDE = ("ksz1h", "ksz2h", "ksztot", "dm_mean", "dm_var", "dmcltot", "dmcl1h", "ydmtot_E")
P_SIDE = ("yytot", "ymean", "ydmtot_P")            # pressure is the same in every definition: reported for D0 only
BANDS = ((0.0, 0.2335), (0.2335, 0.5185), (0.5185, 1.0), (1.0, 2.0), (2.0, 4.0))
BAND_LABELS = ("<0.233R", "0.233-0.519R", "0.519-1R", "1-2R", "2-4R")
PDF_HALO_CUTS = {"D0": (1.0, 2.0), "D1": (0.5185,), "D2": (1.0,), "D3": (1.0, 2.0)}   # per-halo checks of the shortcut
if JC.SMOKE:
    PDF_HALO_CUTS = {"D0": (1.0, 2.0), "D1": (0.5185,), "D2": (1.0,), "D3": (2.0,)}


def fallback_J(B, key):
    if key == "ksztot":
        return FI.fallback_J(B, "ksz1h") + FI.fallback_J(B, "ksz2h")
    return FI.fallback_J(B, key)


def full_map(B, J, key):
    out = np.array(fallback_J(B, key), float, copy=True)
    for c, b in enumerate(FC.CAL_BINS):
        out[b] += J[key][c, :JC.N_X]
    return out


# ---------------------------------------------------------------------------- truncation curves (h_sens.py, identical)
def _cum2(A):
    C = A.cumsum(axis=1).cumsum(axis=2)
    diag = np.stack([C[:, k - 1, k - 1] for k in range(1, A.shape[1] + 1)], axis=1)
    return np.concatenate([np.zeros_like(diag[:, :1]), diag], axis=1)


def _cum2_side(A, side):
    C = A.sum(axis=2).cumsum(axis=1) if side == "P" else A.sum(axis=1).cumsum(axis=1)
    return np.concatenate([np.zeros_like(C[:, :1]), C], axis=1)


def truncation_curves(B, mom):
    """Test H's h_sens.truncation_curves (identical code; uses the installed fallback)."""
    ns = JC.N_X
    P, E = B.pad(mom["P"])[:, :ns], B.pad(mom["E"])[:, :ns]
    SPP, SEE, SPE = (B.pad2(mom[k])[:, :ns, :ns] for k in ("SPP", "SEE", "SPE"))
    K = {k: getattr(B, k)[:, :ns, :ns] for k in ("K_yy1h", "K_ksz1h", "K_dm1h", "K_ydm1h")}
    own = {}
    own["yy1h"] = _cum2(K["K_yy1h"] * SPP[..., None])
    own["ksz1h"] = _cum2(K["K_ksz1h"] * SEE[..., None])
    own["dmcl1h"] = _cum2(K["K_dm1h"] * SEE[..., None])
    A_ = K["K_ydm1h"] * SPE[..., None]
    own["ydm1h"] = _cum2(A_)
    own["ydm1h_Pcut"] = _cum2_side(A_, "P")
    own["ydm1h_Ecut"] = _cum2_side(A_, "E")
    own["dm_var"] = _cum2((B.K_dmvar[:, :ns, :ns] * SEE)[..., None])[..., 0]
    zero = np.zeros((FC.N_CAL, 1))
    own["ymean"] = np.concatenate([zero, np.cumsum(P * B.mean_y[:, :ns], axis=1)], axis=1)
    own["dm_mean"] = np.concatenate([zero, np.cumsum(E * B.dm_mean[:, :ns], axis=1)], axis=1)
    full1 = {k: v[:, -1] for k, v in own.items()}
    fb = B.fb
    out_all, out_bin = {}, {}
    base = {"yy1h": fb["yy1h"], "ksz1h": fb["ksz1h"], "dmcl1h": fb["dm1h"], "ydm1h": fb["ydm1h"],
            "ydm1h_Pcut": fb["ydm1h"], "ydm1h_Ecut": fb["ydm1h"], "dm_var": fb["dm_var"], "ymean": fb["mean_y"],
            "dm_mean": fb["dm_mean"]}
    for k, v in own.items():
        out_all[k] = base[k] + v.sum(axis=0)
        tot = base[k] + full1[k].sum(axis=0)
        out_bin[k] = tot[None] - full1[k][:, None] + v
    GP_ = np.cumsum(np.einsum("cs,zcsl->zcsl", P, B.W_y[:, :, :ns], optimize=True), axis=2)
    GE_ = np.cumsum(np.einsum("cs,zcsl->zcsl", E, B.W_tau[:, :, :ns], optimize=True), axis=2)
    nz, _, _, nl = GP_.shape
    zP = np.zeros((nz, FC.N_CAL, 1, nl))
    GP_ = np.concatenate([zP, GP_], axis=2)
    GE_ = np.concatenate([zP, GE_], axis=2)
    WPf = B.W_fb_y + GP_[:, :, -1].sum(axis=1)
    WEf = B.W_fb_tau + GE_[:, :, -1].sum(axis=1)
    pv, s2, dz = B.pv, B.s2_lin[:, None], B.dmz[:, None]

    def two_h(WP, WE):
        return {"yy2h": np.sum(pv * WP * WP, axis=0), "ksz2h": np.sum(s2 * pv * WE * WE, axis=0),
                "dmcl2h": np.sum(pv * (dz * WE)**2, axis=0), "ydm2h": np.sum(pv * WP * dz * WE, axis=0),
                "ydm2h_Pcut": np.sum(pv * WP * dz * WEf, axis=0), "ydm2h_Ecut": np.sum(pv * WPf * dz * WE, axis=0)}

    nk = ns + 1
    two_all = {k: np.zeros((nk, nl)) for k in ("yy2h", "ksz2h", "dmcl2h", "ydm2h", "ydm2h_Pcut", "ydm2h_Ecut")}
    two_bin = {k: np.zeros((FC.N_CAL, nk, nl)) for k in two_all}
    for k in range(nk):
        WP = B.W_fb_y + GP_[:, :, k].sum(axis=1)
        WE = B.W_fb_tau + GE_[:, :, k].sum(axis=1)
        for name, v in two_h(WP, WE).items():
            two_all[name][k] = v
        for c in range(FC.N_CAL):
            WPc = WPf - GP_[:, c, -1] + GP_[:, c, k]
            WEc = WEf - GE_[:, c, -1] + GE_[:, c, k]
            for name, v in two_h(WPc, WEc).items():
                two_bin[name][c, k] = v
    out_all.update(two_all)
    out_bin.update(two_bin)
    for tot, a, b in (("yytot", "yy1h", "yy2h"), ("ksztot", "ksz1h", "ksz2h"), ("dmcltot", "dmcl1h", "dmcl2h"),
                      ("ydmtot", "ydm1h", "ydm2h"), ("ydmtot_Pcut", "ydm1h_Pcut", "ydm2h_Pcut"),
                      ("ydmtot_Ecut", "ydm1h_Ecut", "ydm2h_Ecut")):
        out_all[tot] = out_all[a] + out_all[b]
        out_bin[tot] = out_bin[a] + out_bin[b]
    return {"all": out_all, "bin": out_bin, "own": own}


def radii(xk, C):
    return {f"R{int(round(100 * X))}": JC.interp_crossing(xk, C, X) for X in RADII_X}


# ============================================================================ main
def main():
    timer = JC.Timer()
    pr = JC.load_npz(JC.WORK / "profiles_j.npz")
    tr = JC.load_npz(JC.RESULTS / "truth.npz")
    ts = JC.read_json(JC.RESULTS / "truth_summary.json")
    xk = JC.X200
    dln = JC.dlnx(JC.X200)
    mid = np.sqrt(np.maximum(JC.X200[:-1], 1e-6) * JC.X200[1:])
    band_of = np.searchsorted(np.array([b[1] for b in BANDS]), mid, side="left")
    li = [int(np.flatnonzero(FC.ELL == e)[0]) for e in JC.SEL_ELL]
    out = {"x_edges": JC.X200, "dlnx": dln, "ell": FC.ELL, "sel_ell": np.array(JC.SEL_ELL), "j_bins": np.array(JC.J_BINS),
           "engine_bin_labels": np.array(JC.ENGINE_LABELS), "quantile_levels": JC.QUANTILES}
    summary = {"definitions": __doc__, "sum_rules": {}, "radii": {}, "shares": {}, "bands": {}, "peaks": {},
               "tolerance": {}, "self_fraction": {}, "own_radii": {}, "checks": {}}
    rows_r, rows_tol, rows_share, rows_band = [], [], [], []
    cal_mask = np.zeros(16, bool)
    cal_mask[FC.CAL_BINS] = True
    for d, v in JC.DEFS.items():
        tag = f"C14|{d}|T1"
        cut = v["cut"]
        kB = JC.K_CUT[cut]
        B = FB.Basis(v["basis"])
        mom = JO.moments(pr, v["sel"], cut, "T1", "C14")
        JF.install(B, "C14", cut)
        WP, WE = B.windows(B.pad(mom["P"]), B.pad(mom["E"]))
        B.Jfb2h = JF.jfb2h(B, cut, WP, WE)
        J = B.jacobians(**mom)
        J["ksztot"] = J["ksz1h"] + J["ksz2h"]
        Jd = B.jacobians(mom["P"], mom["E"])
        Jd["ksztot"] = Jd["ksz1h"] + Jd["ksz2h"]
        O = {k: tr[f"{tag}|truth__{k}"] for k in JO.OBS_KEYS}
        Od = {k: tr[f"{tag}|truthmean__{k}"] for k in JO.OBS_KEYS}
        maps = {}
        for name, (fam, jkey, tkey, k) in OBS.items():
            if name in P_SIDE and d != "D0":
                continue
            for kind, JJ, OO in (("imp", J, O), ("impdet", Jd, Od)):
                M = full_map(B, JJ, jkey)
                Ov = OO[tkey]
                norm = M / (k * (Ov if np.ndim(Ov) == 0 else Ov[None, None, :]))
                s = norm.sum(axis=(0, 1))
                summary["sum_rules"][f"{kind}__{d}__{name}"] = float(np.max(np.abs(s - 1)))
                if name in E_SIDE:
                    summary["checks"][f"zero_beyond_boundary__{kind}__{d}__{name}"] = float(np.max(np.abs(norm[:, kB:]))) \
                        if kB < JC.N_X else 0.0
                keep = norm[..., li] if norm.ndim == 3 else norm
                out[f"{kind}__{d}__{name}"] = keep.astype(np.float32)
                if kind == "imp":
                    maps[name] = norm
        # quantiles (Mecke): per-halo subsample (imp) and truth mean (impdet)
        for kind, lab in (("imp", f"{tag}|truth_subsample"), ("impdet", f"{tag}|truthmean")):
            Jq, Jqfb = tr[f"Jq|{lab}"], tr[f"Jqfb|{lab}"]
            Q = np.asarray(ts["pdf"][lab.replace("_subsample", "")]["quantiles"], float)
            for p, qn in enumerate(JC.Q_NAMES):
                M = np.array(Jqfb[p], float, copy=True)
                for c, b in enumerate(FC.CAL_BINS):
                    M[b] += Jq[p, c, :JC.N_X]
                norm = M / Q[p]
                out[f"{kind}__{d}__dm_{qn}"] = norm.astype(np.float32)
                summary["sum_rules"][f"{kind}__{d}__dm_{qn}"] = float(abs(norm.sum() - 1))
                summary["checks"][f"zero_beyond_boundary__{kind}__{d}__dm_{qn}"] = float(np.max(np.abs(norm[:, kB:]))) \
                    if kB < JC.N_X else 0.0
                if kind == "imp":
                    maps[f"dm_{qn}"] = norm
        # contribution split: diagonal (own-shell) fraction of the calibrated 1-halo signal
        for name, Kname, Sname in (("ksz1h", "K_ksz1h", "SEE"), ("dmcl1h", "K_dm1h", "SEE")):
            Kd = np.einsum("cssl->csl", getattr(B, Kname)[:, :JC.N_X, :JC.N_X])
            Sd = np.einsum("css->cs", B.pad2(mom[Sname])[:, :JC.N_X, :JC.N_X])
            diag = (Kd * Sd[..., None]).sum(axis=(0, 1))
            summary["self_fraction"][f"{d}__{name}"] = {str(e): float(diag[i] / tr[f"{tag}|truthcal__{name}"][i])
                                                       for e, i in zip(JC.SEL_ELL, li)}
        # truncation curves (moment observables)
        tc = truncation_curves(B, mom)
        chk_end, chk_flat = {}, {}
        names_t = ("ksz1h", "ksz2h", "ksztot", "dm_mean", "dm_var", "dmcltot", "dmcl1h", "ydmtot_Ecut") + \
            (("yytot", "ymean", "ydmtot") if d == "D0" else ())
        for name in names_t:
            curve = tc["all"][name]
            out[f"trunc_all__{d}__{name}"] = curve
            ref = O.get(name, O.get(name.replace("_Ecut", "")))
            chk_end[name] = float(np.max(np.abs(curve[-1] / ref - 1)))
            if name not in ("yytot", "ymean", "ydmtot"):
                chk_flat[name] = float(np.max(np.abs(curve[kB:] / curve[kB] - 1)))
        summary["checks"][f"truncation_endpoint_equals_truth__{d}"] = chk_end
        summary["checks"][f"truncation_flat_beyond_boundary__{d}"] = chk_flat
        for name, v_ in tc["own"].items():
            if name in ("ymean", "dm_mean", "dm_var"):
                out[f"trunc_own__{d}__{name}"] = v_
            else:
                out[f"trunc_own__{d}__{name}"] = v_[..., li]
        summary["radii"][d] = {}
        for name in names_t:
            curve = tc["all"][name]
            ent = {"scalar": curve[:kB + 1]} if curve.ndim == 1 else {str(e): curve[:kB + 1, i] for e, i in zip(JC.SEL_ELL, li)}
            for at, c_ in ent.items():
                C = (c_ - c_[0]) / (c_[-1] - c_[0])
                T = c_ / c_[-1]
                rad = radii(xk[:kB + 1], C)
                loss = {f"loss_at_{r:g}R": float(1 - T[JC.edge_index(xk, r, tol=1e-3)]) for r in (0.5185, 1.0, 1.5, 2.0, 3.0)
                        if r <= cut + 1e-9}
                summary["radii"][d][f"{name}@{at}"] = dict(rad, **loss, C_max=float(np.max(C)))
                rows_r.append(dict(definition=d, observable=name, ell=at, scope="all bins", mass_bin="all", **rad, **loss))
            if name in tc["own"]:
                own = tc["own"][name]
                for c in range(FC.N_CAL):
                    ats = [("scalar", None)] if own.ndim == 2 else [(str(e), i) for e, i in zip(JC.SEL_ELL, li)]
                    for at, i in ats:
                        o_ = own[c, :kB + 1] if i is None else own[c, :kB + 1, i]
                        if o_[-1] == o_[0]:
                            continue
                        C = (o_ - o_[0]) / (o_[-1] - o_[0])
                        rad = radii(xk[:kB + 1], C)
                        summary["own_radii"][f"{d}__{name}@{at}__{JC.J_LABELS[c]}"] = rad
                        rows_r.append(dict(definition=d, observable=name, ell=at, scope="own bin",
                                           mass_bin=JC.J_LABELS[c], **rad))
        # tolerance, shares, peaks, bands
        for name, M in maps.items():
            k = OBS[name][3] if name in OBS else 1
            ats = [(str(e), i) for e, i in zip(JC.SEL_ELL, li)] if M.ndim == 3 else [("scalar", None)]
            for at, i in ats:
                m = M[..., i] if i is not None else M
                per_shell = m[cal_mask].sum(axis=0)
                dens = m.sum(axis=0) / dln
                with np.errstate(divide="ignore"):
                    tol_shell = 0.01 / np.abs(k * per_shell)
                    tol_01dex = 0.01 / np.abs(k * (per_shell / dln) * 0.1 * np.log(10))
                out[f"tol01dex__{d}__{name}@{at}"] = tol_01dex.astype(np.float32)
                inside_ = np.arange(JC.N_X) < kB
                j = int(np.argmin(np.where(inside_, tol_01dex, np.inf)))
                summary["tolerance"][f"{d}__{name}@{at}"] = {
                    "min_tol_per_0.1dex": float(tol_01dex[j]), "at_x": float(mid[j]),
                    "tol_per_0.1dex_at": {f"{r:g}": float(np.interp(np.log(r), np.log(mid), tol_01dex))
                                          for r in (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 3.8) if r < cut}}
                jp = int(np.argmax(dens))
                summary["peaks"][f"{d}__{name}@{at}"] = {"x_peak_density": float(mid[jp]),
                                                         "peak_density_per_lnx": float(dens[jp])}
                shares = m.sum(axis=1)
                sh = {JC.ENGINE_LABELS[b]: float(shares[b]) for b in range(16)}
                sh["calibrated_engine_bins"] = float(shares[cal_mask].sum())
                sh["low_mass_12.7-13.5"] = float(shares[JC.LOW_BINS].sum()) if JC.LOW_BINS else 0.0
                summary["shares"][f"{d}__{name}@{at}"] = sh
                bsum = np.array([m[:, band_of == q].sum() for q in range(len(BANDS))])
                summary["bands"][f"{d}__{name}@{at}"] = dict(zip(BAND_LABELS, bsum.tolist()))
                for b in range(16):
                    rows_share.append(dict(definition=d, observable=name, ell=at, mass_bin=JC.ENGINE_LABELS[b],
                                           calibrated=bool(cal_mask[b]), share=float(shares[b])))
                for q, lab in enumerate(BAND_LABELS):
                    rows_band.append(dict(definition=d, observable=name, ell=at, band=lab, importance=float(bsum[q])))
                for s in range(kB):
                    rows_tol.append(dict(definition=d, observable=name, ell=at, x_lo=JC.X200[s], x_hi=JC.X200[s + 1],
                                         importance_calibrated=float(per_shell[s]), importance_all=float(m[:, s].sum()),
                                         density_per_lnx=float(dens[s]), tol_shell=float(tol_shell[s]),
                                         tol_per_0p1dex=float(tol_01dex[s])))
        # exact share of the fallback haloes (above 10^15.388) for linear and 1-halo terms: O_fb / O
        summary.setdefault("fallback_share", {})[d] = {
            "dm_mean": float(B.fb["dm_mean"] / O["dm_mean"]), "dm_var": float(B.fb["dm_var"] / O["dm_var"]),
            **{f"ksz1h@{e}": float(B.fb["ksz1h"][i] / O["ksz1h"][i]) for e, i in zip(JC.SEL_ELL, li)},
            **{f"dmcl1h@{e}": float(B.fb["dm1h"][i] / O["dmcl1h"][i]) for e, i in zip(JC.SEL_ELL, li)}}
        JC.log(f"sens moments {d}", timer.mark(f"mom_{d}"))
        del B, J, Jd, tc, maps
        JC.mem(f"sens after {d}")
    # ---------------------------------------------------------------- DM quantile truncation curves (PDF passes)
    for d, v in JC.DEFS.items():
        tag = f"C14|{d}|T1"
        cut = v["cut"]
        kB = JC.K_CUT[cut]
        ek = JO.SEL_KEY[v["sel"]]
        mom = JO.moments(pr, v["sel"], cut, "T1", "C14")
        specs = [("zero", [(np.zeros(JC.N_FINE), 1.0) for _ in range(FC.N_CAL)])]
        for k in range(1, kB):
            Ek = mom["E"].copy()
            Ek[:, k:] = 0.0
            specs.append((f"mean_k{k}", [(JC.pad75(Ek[c]), 1.0) for c in range(FC.N_CAL)]))
        ks = [JC.edge_index(JC.X200, r, tol=1e-3) for r in PDF_HALO_CUTS[d]]
        for k in ks:
            per = []
            for b in JC.J_BINS:
                rows = JT.bin_rows(pr, b)
                F = pr[f"{ek}_T1"][rows].copy()
                F[:, k:] = 0.0
                w = pr["weight"][rows]
                per.append((JC.pad75(F), w / w.sum()))
            specs.append((f"halo_k{k}", per))
        res, tim = JP.run_specs(specs, v["basis"], cut, "C14", label=f"pdf cuts {d}")
        qmean = np.zeros((kB + 1, len(JC.QUANTILES)))
        qmean[0] = res["zero"]["quantiles"]
        for k in range(1, kB):
            qmean[k] = res[f"mean_k{k}"]["quantiles"]
        qmean[kB] = ts["pdf"][f"{tag}|truthmean"]["quantiles"]
        out[f"trunc_all__{d}__dmq_mean"] = qmean
        qh = np.array([res[f"halo_k{k}"]["quantiles"] for k in ks] + [ts["pdf"][f"{tag}|truth"]["quantiles"]])
        out[f"trunc_halo__{d}__dmq"] = qh
        out[f"trunc_halo__{d}__k"] = np.array(ks + [kB])
        summary.setdefault("pdf_timing", {})[d] = tim
        for p, qn in enumerate(JC.Q_NAMES):
            C = (qmean[:, p] - qmean[0, p]) / (qmean[-1, p] - qmean[0, p])
            T = qmean[:, p] / qmean[-1, p]
            rad = radii(xk[:kB + 1], C)
            loss = {f"loss_at_{r:g}R": float(1 - T[JC.edge_index(xk, r, tol=1e-3)]) for r in (0.5185, 1.0, 1.5, 2.0, 3.0)
                    if r <= cut + 1e-9}
            summary["radii"][d][f"dm_{qn}@truthmean"] = dict(rad, **loss)
            rows_r.append(dict(definition=d, observable=f"dm_{qn}", ell="scalar", scope="all bins (truth mean)",
                               mass_bin="all", **rad, **loss))
            Th = qh[:, p] / qh[-1, p]
            summary["radii"][d][f"dm_{qn}@per_halo_loss"] = {f"loss_at_{JC.X200[k]:.4g}R": float(1 - Th[j])
                                                            for j, k in enumerate(ks)}
        JC.log(f"pdf cuts {d}", timer.mark(f"pdf_{d}"))
        del res
    np.savez_compressed(JC.RESULTS / "sens.npz", **out)
    summary["seconds"] = timer.mark("total")
    JC.write_json(JC.RESULTS / "sens_summary.json", summary)
    JC.write_csv(JC.TABLES / "J1d_radii.csv", rows_r)
    JC.write_csv(JC.TABLES / "J1d_importance_shells.csv", rows_tol)
    JC.write_csv(JC.TABLES / "J1d_mass_shares.csv", rows_share)
    JC.write_csv(JC.TABLES / "J1d_bands.csv", rows_band)
    JC.log("sens done", summary["seconds"])
    return summary
