"""F1: which radial shells and which haloes carry each observable (at the measured FLAMINGO truth).

Normalised importance = (dO/dln F_s,b) / (k O_truth), k = 2 for quadratic forms (auto 1h/2h, kSZ, DM
variance), 1 for means, quantiles and each side of the y x DM cross (kernels/ convention; maps sum to 1).
Calibrated bins (13.5-15.388): exact in-basis Jacobians at the scatter-inclusive truth (per-halo second
moments: every halo of the bin scaled in one shell); DM quantiles: Mecke Jacobians of the per-halo truth
(64-halo subsample per bin, reweighted).  Fallback haloes (published profiles): the builder's per-(engine
bin, shell) pieces.  Also the deterministic (truth-mean) Jacobians that define the observable metric K.
Outputs: products/importance_F.npz, tables/F1_importance_{bands,mass,shells}.csv, F1 figures.
"""
from __future__ import annotations

import f_basis as FB
import f_common as FC
import f_fit as FF
import numpy as np

QUAD = {"yy1h", "yy2h", "yytot", "ksz1h", "ksz2h", "ksztot", "dmcl1h", "dmcl2h", "dmcltot", "dm_var"}
OBS_ROWS = [  # (output name, jacobian key, truth key, family, description)
    ("yy1h", "yy1h", "yy1h", "tSZ", "tSZ yy 1-halo"), ("yy2h", "yy2h", "yy2h", "tSZ", "tSZ yy 2-halo"),
    ("yytot", "yytot", "yytot", "tSZ", "tSZ yy total"), ("ymean", "ymean", "ymean", "tSZ", "mean Compton y"),
    ("ksz1h", "ksz1h", "ksz1h", "kSZ", "kSZ 1-halo (catalogue v_LOS)"), ("ksz2h", "ksz2h", "ksz2h", "kSZ", "kSZ 2-halo (approx.)"),
    ("dmcltot", "dmcltot", "dmcltot", "DM", "DM-field C_ell total"),
    ("dm_mean", "dm_mean", "dm_mean", "DM", "mean halo DM"), ("dm_var", "dm_var", "dm_var", "DM", "DM variance"),
    ("ydmtot_P", "ydmtot_P", "ydmtot", "tSZxDM", "y x DM total, pressure side"),
    ("ydmtot_E", "ydmtot_E", "ydmtot", "tSZxDM", "y x DM total, electron side"),
    ("ydm1h_P", "ydm1h_P", "ydm1h", "tSZxDM", "y x DM 1h, pressure side"), ("ydm1h_E", "ydm1h_E", "ydm1h", "tSZxDM", "y x DM 1h, electron side"),
]
FB_KEY = {"yy1h": ("fb", "J_yy1h"), "yy2h": ("2h", "Jfb_yy2h"), "ymean": ("fb", "J_mean_y"),
          "ksz1h": ("fb", "J_ksz1h"), "ksz2h": ("2h", "Jfb_ksz2h"), "dm_mean": ("fb", "J_dm_mean"),
          "dm_var": ("fb", "J_dm_var"), "ydm1h_P": ("fb", "J_ydm1h_P"), "ydm1h_E": ("fb", "J_ydm1h_E"),
          "dmcl1h": ("fb", "J_dm1h"), "dmcl2h": ("2h", "Jfb_dm2h"), "ydm2h_P": ("2h", "Jfb_ydm2h_P"),
          "ydm2h_E": ("2h", "Jfb_ydm2h_E")}
SUMS = {"yytot": ("yy1h", "yy2h"), "dmcltot": ("dmcl1h", "dmcl2h"), "ydmtot_P": ("ydm1h_P", "ydm2h_P"),
        "ydmtot_E": ("ydm1h_E", "ydm2h_E")}


def fallback_J(B, key):
    if key in SUMS:
        return sum(fallback_J(B, k) for k in SUMS[key])
    src, name = FB_KEY[key]
    return B.fb[name] if src == "fb" else B.Jfb2h[name]


def full_map(B, Jcal, key):
    """[16 engine bins, 35 shells(, ell)]: calibrated in-basis part + fallback pieces."""
    out = np.array(fallback_J(B, key), float, copy=True)
    for c, b in enumerate(FC.CAL_BINS):
        out[b] += Jcal[key][c, :FC.N_F]
    return out


def main():
    timer = FC.Timer()
    pr = dict(np.load(FC.PRODUCTS / "profiles.npz"))
    tr = dict(np.load(FC.PRODUCTS / "truth.npz"))
    B = FB.Basis("T")
    cal = FC.CAL_BINS
    P, Ee = pr["meanP_main"][cal], pr["meanE_main"][cal]
    S = dict(SPP=pr["SPP_main"][cal], SEE=pr["SEE_main"][cal], SPE=pr["SPE_main"][cal])
    Jt = B.jacobians(P, Ee, **S)                   # scatter-inclusive (truth)
    Jd = B.jacobians(P, Ee)                        # deterministic (truth mean)
    truth = {k.split("__")[1]: tr[k] for k in tr if k.startswith("truth_main__")}
    truthmean = {k.split("__")[1]: tr[k] for k in tr if k.startswith("truthmean_main__")}
    Qt = tr["pdf_truth_main__quantiles"]
    Qd = tr["pdf_truthmean_main__quantiles"]
    out = {"shell_edges": FC.F_EDGES, "band_edges": FC.BAND_EDGES, "engine_bin_labels": np.array(FC.ENGINE_BIN_LABELS),
           "ell": B.ell, "quantile_levels": FC.QUANTILES}
    rows_band, rows_mass, rows_shell = [], [], []
    band = FC.band_of_shell(FC.F_EDGES)
    base_full = np.zeros((16, FC.N_F), bool)
    base_full[:15] = pr["base_mask"]
    cal_mask = np.zeros(16, bool)
    cal_mask[cal] = True
    summary = {}
    for name, jkey, tkey, fam, text in OBS_ROWS:
        M = full_map(B, Jt, jkey)                                   # [16, 35(, ell)]
        k = 2 if jkey in QUAD else 1
        O = truth[tkey]
        norm = M / (k * (O if np.ndim(O) == 0 else O[None, None, :]))
        out[f"imp__{name}"] = norm
        for at, idx in ((e, i) for e, i in zip(FC.SEL_ELL, FC.SEL_INDEX)) if np.ndim(O) else [(None, None)]:
            m = norm[..., idx] if idx is not None else norm
            label = f"{name}_{at}" if at is not None else name
            bands = np.array([m[:, band == b].sum() for b in range(len(FC.BAND_LABELS))])
            masses = m.sum(axis=1)
            summary[label] = {"family": fam, "bands": bands.tolist(), "calibrated_share": float(masses[cal_mask].sum()),
                              "unresolved_calibrated_share": float(np.sum(m[cal_mask][~base_full[cal_mask]])),
                              "top_bins": [FC.ENGINE_BIN_LABELS[i] for i in np.argsort(-np.abs(masses))[:3]]}
            for b, lab in enumerate(FC.BAND_LABELS):
                rows_band.append({"observable": label, "family": fam, "description": text, "band_R200c": lab,
                                  "normalised_importance": bands[b]})
            for b in range(16):
                rows_mass.append({"observable": label, "family": fam, "mass_bin": FC.ENGINE_BIN_LABELS[b],
                                  "calibrated": bool(cal_mask[b]), "normalised_importance": masses[b]})
            for s in range(FC.N_F):
                rows_shell.append({"observable": label, "family": fam, "x_lo": FC.F_EDGES[s], "x_hi": FC.F_EDGES[s + 1],
                                   "normalised_importance": float(m[:, s].sum()),
                                   "calibrated_bins": float(m[cal_mask, s].sum())})
    # DM quantiles (per-halo subsample Mecke for calibrated bins + fallback pieces)
    Jq = tr["Jq_truth_main_subsample"]
    Jqfb = tr["Jqfb_truth_main_subsample"]
    for p, qn in enumerate(FC.Q_NAMES):
        M = np.array(Jqfb[p], float, copy=True)
        for c, b in enumerate(cal):
            M[b] += Jq[p, c, :FC.N_F]
        norm = M / Qt[p]
        out[f"imp__dm_{qn}"] = norm
        bands = np.array([norm[:, band == b].sum() for b in range(len(FC.BAND_LABELS))])
        masses = norm.sum(axis=1)
        label = f"dm_{qn}"
        summary[label] = {"family": "DM", "bands": bands.tolist(), "calibrated_share": float(masses[cal_mask].sum()),
                          "unresolved_calibrated_share": float(np.sum(norm[cal_mask][~base_full[cal_mask]])),
                          "top_bins": [FC.ENGINE_BIN_LABELS[i] for i in np.argsort(-np.abs(masses))[:3]],
                          "sum_rule": float(norm.sum())}
        for b, lab in enumerate(FC.BAND_LABELS):
            rows_band.append({"observable": label, "family": "DM", "description": f"DM PDF quantile {FC.QUANTILES[p]:g}",
                              "band_R200c": lab, "normalised_importance": bands[b]})
        for b in range(16):
            rows_mass.append({"observable": label, "family": "DM", "mass_bin": FC.ENGINE_BIN_LABELS[b],
                              "calibrated": bool(cal_mask[b]), "normalised_importance": masses[b]})
        for s in range(FC.N_F):
            rows_shell.append({"observable": label, "family": "DM", "x_lo": FC.F_EDGES[s], "x_hi": FC.F_EDGES[s + 1],
                               "normalised_importance": float(norm[:, s].sum()), "calibrated_bins": float(norm[cal_mask, s].sum())})
    # Y and N_e budgets (z = 0, per calibrated bin): share of Y(<4R) per shell
    vol = FC.SHELL_VOLUME[:FC.N_F]
    for name, mean in (("Y", P), ("Ne", Ee)):
        share = mean * vol / (mean @ vol)[:, None]
        out[f"budget__{name}"] = share
        for c in range(FC.N_CAL):
            bands = np.array([share[c, band == b].sum() for b in range(len(FC.BAND_LABELS))])
            summary[f"{name}_{FC.CAL_LABELS[c]}"] = {"family": "budget", "bands": bands.tolist()}
            for b, lab in enumerate(FC.BAND_LABELS):
                rows_band.append({"observable": f"{name}(<4R) {FC.CAL_LABELS[c]}", "family": "budget",
                                  "description": f"z=0 {name} budget", "band_R200c": lab, "normalised_importance": bands[b]})
    # weights (objective d) and metric (objective e) from these maps
    # weights: scatter-inclusive moment Jacobians, deterministic (truth-mean) Mecke Jacobians for the quantiles
    # (the same recipe is reproducible on every CV training set); metric K: deterministic throughout
    gs = FF.jacobian_maps(Jt, tr["Jq_truthmean_main"], truth, Qt)
    gd = FF.jacobian_maps(Jd, tr["Jq_truthmean_main"], truth, Qt)
    for f in ("P", "E"):
        w, Wf = FF.field_weights(gs, f)
        out[f"w_{f}"] = w
        for fam, W in Wf.items():
            out[f"Wfam_{f}_{fam}"] = W
        out[f"K_{f}"] = FF.field_metric(gd, f)
        wbin = w / w.sum(axis=1, keepdims=True)
        out[f"wbin_{f}"] = wbin
    # comparison with kernels/ (P4/E4 fits): band shares of the same observables
    kim = np.load(FC.ROOT / "kernels" / "importance.npz")
    comp = {}
    for mine, theirs in (("yytot", "normalised__P4__cl_total"), ("ksz1h", "normalised__E4__ksz_cl_1h"),
                         ("dm_mean", "normalised__E4__dm_mean"), ("dm_var", "normalised__E4__dm_var")):
        their = kim[theirs]
        comp[mine] = {"kernels_P4E4_bands": (their.sum(axis=1)).tolist(), "kernels_shell_edges": kim["shell_edges"].tolist()}
    summary["_kernels_comparison"] = comp
    np.savez_compressed(FC.PRODUCTS / "importance_F.npz", **out)
    FC.write_csv(FC.TABLES / "F1_importance_bands.csv", rows_band)
    FC.write_csv(FC.TABLES / "F1_importance_mass.csv", rows_mass)
    FC.write_csv(FC.TABLES / "F1_importance_shells.csv", rows_shell)
    summary["_seconds"] = timer.mark("total")
    FC.write_json(FC.PRODUCTS / "importance_summary.json", summary)
    FC.log("importance done", summary["_seconds"])


if __name__ == "__main__":
    main()
