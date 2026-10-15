"""P1: recompute radial / mass importance summaries from the stored maps of earlier tests (read only).

Sources
-------
* F_observables/results/importance_F.npz  (Test F, F1): imp__<obs> [16 engine bins, 35 F shells(, 106 ell)],
  normalised importance  I = (dO / d ln F_{b,s}) / (k O_truth), k = 2 quadratic, 1 linear; evaluated at the measured
  FLAMINGO truth (per-halo profiles to 4 R200c, scatter-inclusive), all gas, recent-AGN gas removed.
  budget__Y / budget__Ne [10 calibrated bins, 35]: share of Y(<4R) / N_e(<4R) per shell (z = 0).
* kernels/jacobians.npz (Step 0.1): J_<obs>__<P4|E4> [67 x200 shells, 16 bins(, 106 ell)] at the 4R B12/B16 fits
  (deterministic, finest stored radial grid).  Normalised here exactly like F: J / (k O).

Outputs (H_sensitivity/prior/tables/)
-------
* imp_bands_F.csv        share per band (all masses, calibrated-only renormalised, fallback share)
* imp_bandmass_F.csv     share per (engine bin, band), plus the within-bin band profile
* imp_radii_F.csv        R50/R80/R90/R95/R99, shares beyond 1/1.5/2/3 R, peak of dI/dlnx, last-shell / peak ratio
                         (all masses, calibrated-only, and per calibrated bin)
* imp_x200_kernels.csv   the same radii from the 67-shell kernels Jacobians (P4/E4 fits) + band shares
* budget_bands_F.csv     Y(<4R), N_e(<4R) band shares per calibrated bin
* p1_summary.json        compact numbers for the report
"""
from __future__ import annotations

import numpy as np

import prior_common as C

FR = (0.5, 0.8, 0.9, 0.95, 0.99)
BEYOND = (1.0, 1.5, 2.0, 3.0)
CAL = list(range(5, 15))   # engine bins 13.5-13.7 ... 15.3-15.5 (calibrated support 13.5-15.388)

# (family, key, ell or None, label)
OBS = []
for e in C.SEL_ELL:
    OBS.append(("tSZ", "yytot", e, f"tSZ yy total, ell={e}"))
for e in (100, 1000, 3000):
    OBS.append(("tSZ", "yy1h", e, f"tSZ yy 1h, ell={e}"))
OBS.append(("tSZ", "ymean", None, "mean y"))
for e in C.SEL_ELL:
    OBS.append(("kSZ", "ksz1h", e, f"kSZ 1h, ell={e}"))
for k, lab in (("dm_mean", "mean DM"), ("dm_var", "DM variance"), ("dm_q16", "DM Q16"), ("dm_q50", "DM Q50"),
               ("dm_q84", "DM Q84"), ("dm_q975", "DM Q97.5"), ("dm_q999", "DM Q99.9")):
    OBS.append(("DM", k, None, lab))
for e in (100, 1000, 3000):
    OBS.append(("DM", "dmcltot", e, f"DM-field C_ell total, ell={e}"))
for e in (100, 1000, 3000):
    OBS.append(("tSZxDM", "ydmtot_P", e, f"tSZxDM total, pressure side, ell={e}"))
    OBS.append(("tSZxDM", "ydmtot_E", e, f"tSZxDM total, electron side, ell={e}"))


def obs_name(key, ell):
    return key if ell is None else f"{key}_{ell}"


def radii_row(edges, imp):
    out = {f"R{int(f * 100)}": C.cumulative_radius(edges, imp, f) for f in FR}
    for x in BEYOND:
        out[f"share_beyond_{x:g}R"] = C.share_beyond(edges, imp, x)
    w = np.log(edges[2:] / edges[1:-1])                      # shells 1..: log widths (shell 0 starts at x = 0)
    dens = imp[1:] / w
    k = int(np.argmax(dens))
    out["peak_dI_dlnx_x"] = float(np.sqrt(edges[1 + k] * edges[2 + k]))
    out["peak_dI_dlnx"] = float(dens[k])
    out["last_shell_dI_dlnx_over_peak"] = float(dens[-1] / dens[k])
    return out


def main():
    d = np.load(C.F_IMP)
    edges = d["shell_edges"]
    ell = d["ell"]
    labels = [str(x) for x in d["engine_bin_labels"]]
    band = C.band_index(edges, np.array([0, .1051, .2335, .5185, 1, 1.5, 2, 3, 4]))
    assert band.min() == 0 and band.max() == 7

    rows_band, rows_bm, rows_rad, summary = [], [], [], {"source": str(C.F_IMP.relative_to(C.STUDY))}
    checks = {}
    for fam, key, e, lab in OBS:
        arr = d[f"imp__{key}"]
        M = arr[:, :, int(np.flatnonzero(ell == e)[0])] if e is not None else arr        # [16, 35]
        name = obs_name(key, e)
        tot = float(M.sum())
        checks[name] = tot
        cal = M[CAL].sum()
        bands_all = np.array([M[:, band == j].sum() for j in range(8)]) / tot
        bands_cal = np.array([M[CAL][:, band == j].sum() for j in range(8)]) / cal
        for j in range(8):
            rows_band.append({"family": fam, "observable": name, "label": lab, "band_R200c": C.BAND_LABELS[j],
                              "share_all_masses": bands_all[j], "share_calibrated_renorm": bands_cal[j]})
        rows_band.append({"family": fam, "observable": name, "label": lab, "band_R200c": "calibrated_share",
                          "share_all_masses": cal / tot})
        for b in range(16):
            mb = M[b].sum()
            for j in range(8):
                rows_bm.append({"family": fam, "observable": name, "mass_bin": labels[b], "calibrated": b in CAL,
                                "band_R200c": C.BAND_LABELS[j], "share_of_total": M[b, band == j].sum() / tot,
                                "share_within_bin": (M[b, band == j].sum() / mb) if mb != 0 else np.nan,
                                "bin_share_of_total": mb / tot})
        r_all = radii_row(edges, M.sum(0))
        r_cal = radii_row(edges, M[CAL].sum(0))
        rows_rad.append({"family": fam, "observable": name, "label": lab, "masses": "all", **r_all})
        rows_rad.append({"family": fam, "observable": name, "label": lab, "masses": "calibrated 13.5-15.5", **r_cal})
        for b in CAL:
            if abs(M[b].sum()) > 1e-6:
                rows_rad.append({"family": fam, "observable": name, "label": lab, "masses": labels[b],
                                 "bin_share_of_total": M[b].sum() / tot, **radii_row(edges, M[b])})
        summary[name] = {"label": lab, "bands_all": dict(zip(C.BAND_LABELS, bands_all)),
                         "inside_1R": bands_all[:4].sum(), "1_2R": bands_all[4:6].sum(), "2_4R": bands_all[6:].sum(),
                         "calibrated_share": cal / tot, "radii_all": r_all, "radii_calibrated": r_cal,
                         "dominant_bin": labels[int(np.argmax(M.sum(1)))],
                         "dominant_bin_share": float(M.sum(1).max() / tot)}
    summary["sum_rule_max_abs_dev"] = max(abs(v - 1) for v in checks.values())

    # cross-check against Test F's own band table (F1_importance_bands.csv)
    import csv
    ref = {}
    with open(C.F_TABLES / "F1_importance_bands.csv") as fh:
        for r in csv.DictReader(fh):
            ref[(r["observable"], r["band_R200c"])] = float(r["normalised_importance"])
    dev = []
    for r in rows_band:
        k = (r["observable"], r["band_R200c"])
        if k in ref:
            dev.append(abs(ref[k] - r["share_all_masses"]))
    summary["check_vs_F1_importance_bands_csv"] = {"n": len(dev), "max_abs_dev": max(dev)}

    # Y / N_e budgets (z = 0)
    rows_bud = []
    for q in ("Y", "Ne"):
        Bq = d[f"budget__{q}"]
        for c, b in enumerate(CAL):
            sh = np.array([Bq[c, band == j].sum() for j in range(8)])
            # local growth rate of the enclosed integral at the 4R edge: d ln I(<x) / d ln x over the last shell
            growth = Bq[c, -1] / np.log(edges[-1] / edges[-2]) / Bq[c].sum()
            rows_bud.append({"quantity": f"{q}(<4R)", "mass_bin": labels[b], **dict(zip(C.BAND_LABELS, sh)),
                             "inside_1R": sh[:4].sum(), "inside_2R": sh[:6].sum(), "2_4R": sh[6:].sum(),
                             "dlnI_dlnx_last_shell_3.63_4R": growth})
    C.write_csv(C.TABLES / "budget_bands_F.csv", rows_bud)

    # finest stored grid: kernels Jacobians at the P4 / E4 fits (deterministic)
    K = np.load(C.K_JAC)
    xe = K["x200_edges"]
    kell = K["ell"]
    kband = C.band_index(xe, C.BAND_EDGES_NOMINAL)
    rows_k = []

    def kmap(name, e):
        ie = None if e is None else int(np.flatnonzero(kell == e)[0])
        if name == "yytot":
            J, O, k = K["J_cl_total__P4"][..., ie], K["O_cl_total__P4"][ie], 2
        elif name == "yy1h":
            J, O, k = K["J_cl_1h__P4"][..., ie], K["O_cl_1h__P4"][ie], 2
        elif name == "ymean":
            J, O, k = K["J_mean__P4"], K["O_mean__P4"], 1
        elif name == "ksz1h":
            J, O, k = K["J_ksz_cl_1h__E4"][..., ie], K["O_ksz_cl_1h__E4"][ie], 2
        elif name == "dm_mean":
            J, O, k = K["J_dm_mean__E4"], K["O_dm_mean__E4"], 1
        elif name == "dm_var":
            J, O, k = K["J_dm_var__E4"], K["O_dm_var__E4"], 2
        elif name.startswith("dm_q"):
            qi = ["dm_q16", "dm_q50", "dm_q84", "dm_q975", "dm_q999"].index(name)
            J, O, k = K["J_dm_quantiles__E4"][qi], K["O_dm_quantiles__E4"][qi], 1
        elif name == "dmcltot":
            J = K["J_dm_cl_1h__E4"][..., ie] + K["J_dm_cl_2h__E4"][..., ie]
            O, k = K["O_dm_cl_1h__E4"][ie] + K["O_dm_cl_2h__E4"][ie], 2
        elif name in ("ydmtot_P", "ydmtot_E"):
            s = name[-1]
            J = K[f"J_cross_1h_{s}__P4xE4"][..., ie] + K[f"J_cross_2h_{s}__P4xE4"][..., ie]
            O, k = K["O_cross_cl_1h__P4xE4"][ie] + K["O_cross_cl_2h__P4xE4"][ie], 1
        else:
            raise KeyError(name)
        return (J / (k * O)).T                                   # [16, 67]

    kcheck = {}
    for fam, key, e, lab in OBS:
        M = kmap(key, e)
        name = obs_name(key, e)
        kcheck[name] = float(M.sum())
        b_all = np.array([M[:, kband == j].sum() for j in range(8)]) / M.sum()
        Fb = np.array(list(summary[name]["bands_all"].values()))
        rows_k.append({"family": fam, "observable": name, "label": lab, "masses": "all",
                       **radii_row(xe, M.sum(0)), **{f"band_{C.BAND_LABELS[j]}": b_all[j] for j in range(8)},
                       "max_abs_band_diff_vs_F_truth": float(np.max(np.abs(b_all - Fb)))})
        rows_k.append({"family": fam, "observable": name, "label": lab, "masses": "calibrated 13.5-15.5",
                       **radii_row(xe, M[CAL].sum(0))})
        summary[name]["x200_kernels_P4E4"] = {**radii_row(xe, M.sum(0)),
                                              "bands_all": dict(zip(C.BAND_LABELS, b_all))}
    summary["kernels_sum_rule_max_abs_dev"] = max(abs(v - 1) for v in kcheck.values())
    summary["kernels_source"] = str(C.K_JAC.relative_to(C.STUDY))

    C.write_csv(C.TABLES / "imp_bands_F.csv", rows_band)
    C.write_csv(C.TABLES / "imp_bandmass_F.csv", rows_bm)
    C.write_csv(C.TABLES / "imp_radii_F.csv", rows_rad)
    C.write_csv(C.TABLES / "imp_x200_kernels.csv", rows_k)
    C.write_json(C.PRIOR / "p1_summary.json", summary)
    print("sum rule max dev F:", summary["sum_rule_max_abs_dev"], " kernels:", summary["kernels_sum_rule_max_abs_dev"])
    print("vs F1 csv:", summary["check_vs_F1_importance_bands_csv"])


if __name__ == "__main__":
    main()
