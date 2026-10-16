"""L stage 'codex' (local, read-only): the Codex analyses of 2026-10-15/16 confronted with the truncation radii.

Inputs (never modified; read with numpy only):
  halfdome_sz_contributions_20261015/spectra_tsz.npz, spectra_ksz.npz   full HalfDome lightcone painted with the
      published B12 (pressure) / B16 (electrons) to 4 R200c: C_ell of every pair of 3D shells (shell_cross_matrix
      [8, 8, ell], shells 0-0.2-0.5-0.75-1-1.5-2-3-4 R200c), mass and redshift cross_total, full
  flamingo_lightcone_sz_comparison_20261015/results/HalfDome/dm_spectra.npz   the same for the HalfDome DM map (z_s = 1)
  flamingo_lightcone_sz_comparison_20261015/stageA_summary.json               FLAMINGO / HalfDome band powers near ell 1025
  flamingo_lightcone_sz_comparison_20261015/results/<sim>/redshift_<f>.npz    redshift cross_total of the maps
  flamingo_lightcone_sz_comparison_20261015/phaseB_pilot/<sim>_profiles_maps.npz, pilot_assessment.json
      the particle pilot (one massive z ~ 0.2 halo per simulation): shell pressure by category
      (central, own_satellites, other_host, unbound, unknown), radial edges in R200c
User clarification (2026-10-16): R_t is only the FIT range; HalfDome still paints the fitted law to 4 R200c.  The
quantities T(R) below are therefore NOT a painting plan: T(R) is the share of the 4R-painted power that a hard cut at R
would keep (equivalently, 1 - T(R) is the part that depends on the extrapolated region R - 4 R200c), and the
'ratio_<rule>_ESTIMATE' entries are what a hard cut at R_t would imply (not planned; reference only).  The 4R
FLAMINGO / HalfDome ratios (ratio_4R) and the HalfDome shell fractions are the reference painting and its comparison.
Computed:
  1. Share of the 4R-painted HalfDome map power from inside a radius (a hard cut there would keep it), exact at the
     shell edges:
        T(R_k) = sum_{i,j < k} C_ij / C_full,   C_ij = shell_cross_matrix,
     band-averaged with Codex's own band rule (mode-weighted D_ell over [a, a + 50)); between edges linear in R
     (ESTIMATE).  Gate: sum_ij C_ij = C_full, and the additive shell fractions sum_j C_ij / C_full reproduce the
     published CSV tables.
  2. Our engine's analogue on the same lightcone catalogue (FLAMINGO A2-ii stacks, Test J D0):
        T_ours(R) = trunc_all__D0__<O>[edge(R)] / trunc_all__D0__<O>[4R]   (every calibrated halo cut at R)
     and the additive band shares of Test J's importance maps (imp__D0__<O>, summed over bins) on the common shells
     0-1, 1-1.5, 1.5-2, 2-3, 3-4 R200c.
  3. FLAMINGO / HalfDome band-power ratios near ell = 1025: ratio_4R (the reference comparison, both painted to 4R /
     all gas), and, NOT planned, what a hard cut of HalfDome's painting at R_t would imply:
        ratio(R_t) = ratio(4R) / T_HD(R_t)  (ESTIMATE: the HalfDome map scaled by its own shell decomposition).
  4. Redshift and mass shares (additive, C(component, total) / C(total)) of HalfDome and FLAMINGO at the same bands.
  5. Particle pilot: differential bound fraction f(x) = (central + own satellites) / all per shell, its B32/B50
     crossings (same rule as ours), cumulative fractions at 1-4 R200c (reproduce pilot_assessment.json), against our
     stacks and per-halo values in 15.1-15.3 and 15.3-15.5.
Output: results/codex.json, tables/L1_codex_*.csv.
"""
from __future__ import annotations

import csv

import l_common as LC
import numpy as np

SHELLS = np.array([0.0, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
BAND_STARTS = {"77": 50, "126": 100, "325": 300, "1025": 1000, "3025": 3000}
FILES = {"tsz": LC.CODEX_HD / "spectra_tsz.npz", "ksz": LC.CODEX_HD / "spectra_ksz.npz",
         "dm": LC.CODEX_FL / "results" / "HalfDome" / "dm_spectra.npz"}
CSVS = {"tsz": LC.CODEX_HD / "tsz_shell_fractions.csv", "ksz": LC.CODEX_HD / "ksz_shell_fractions.csv"}
J_OBS = {"tsz": "yytot", "ksz": "ksztot", "dm": "dmcltot"}
COMMON = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]
Z_EDGES_HD = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
M_EDGES_HD = [12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5]


def band(ell, values, a, width=50):
    """Codex's band rule (make_figures.bands): mode-count weighted mean of D_ell over [a, a + width)."""
    sl = slice(a, a + width)
    w = 2 * ell[sl] + 1
    power = values * (ell * (ell + 1) / (2 * np.pi))
    return float(np.average(ell[sl], weights=w)), np.average(power[..., sl], axis=-1, weights=w)


def retention(Cm, full, ell):
    out = {}
    for name, a in BAND_STARTS.items():
        c, F = band(ell, full, a)
        _, M = band(ell, Cm, a)                       # [8, 8]
        T = [float(M[:k, :k].sum() / F) for k in range(0, 9)]
        add = (M.sum(axis=1) / F).tolist()
        out[name] = {"ell_centre": c, "T_at_edges": T, "additive_shell_fractions": add}
    return out


def interp_T(T_edges, R):
    return float(np.interp(R, SHELLS, T_edges))


def main():
    t0 = LC.time.perf_counter()
    rad = LC.jload(LC.RESULTS / "radii.json")
    glob = LC.jload(LC.RESULTS / "global.json")
    sens = np.load(LC.J_DIR / "results" / "sens.npz")
    out = {"definitions": __doc__, "sources": {}, "halfdome_maps": {}, "engine": {}, "gates": {}}
    Rg = {r: {f: glob["global"][r][f]["R"]["family"].get("value") for f in ("P", "E")}
          for r in ("B32", "B50", "Req", "C68", "SB", "2H2")}
    # ---------------------------------------------------------------- 1. HalfDome maps
    for fld, path in FILES.items():
        d = np.load(path, allow_pickle=False)
        out["sources"][str(path.relative_to(LC.FLAM))] = LC.sha256(path)
        ell = d["ell"].astype(float)
        Cm, full = d["shell_cross_matrix"], d["full"]
        out["gates"][f"{fld}_shell_matrix_sum_over_full_max_dev"] = float(
            np.nanmax(np.abs(Cm.sum(axis=(0, 1))[50:] / full[50:] - 1)))
        ret = retention(Cm, full, ell)
        if fld in CSVS:
            rows = list(csv.reader(open(CSVS[fld])))
            ref = {round(float(r[0]), 3): [float(x) for x in r[1:]] for r in rows[1:]}
            dev = 0.0
            for name, v in ret.items():
                k = round(v["ell_centre"], 3)
                if k in ref:
                    dev = max(dev, float(np.max(np.abs(np.array(v["additive_shell_fractions"]) - ref[k]))))
            out["gates"][f"{fld}_additive_fractions_vs_csv_max_abs"] = dev
            out["sources"][str(CSVS[fld].relative_to(LC.FLAM))] = LC.sha256(CSVS[fld])
        f_rt = "P" if fld == "tsz" else "E"
        for name, v in ret.items():
            v["T_at_rule_radius_ESTIMATE"] = {r: (interp_T(v["T_at_edges"], Rg[r][f_rt]) if Rg[r][f_rt] else None)
                                              for r in Rg}
            v["common_shell_fractions"] = [float(sum(v["additive_shell_fractions"][i] for i in range(8)
                                                     if SHELLS[i] >= lo - 1e-9 and SHELLS[i + 1] <= hi + 1e-9))
                                           for lo, hi in COMMON]
        out["halfdome_maps"][fld] = ret
        # mass and redshift additive fractions (cross_total / full) at the same bands
        names = d["names"].tolist() if "names" in d.files else None
        for grp, key in (("mass", "mass_cross_total"), ("redshift", "redshift_cross_total")):
            if key in d.files:
                ct = d[key]
            elif names is not None:
                ct = d["cross_total"][[i for i, n in enumerate(names) if n.startswith(grp + "_")]]
            else:
                continue
            out["halfdome_maps"][fld][f"{grp}_fractions"] = {
                name: (band(ell, ct, a)[1] / band(ell, full, a)[1]).tolist() for name, a in BAND_STARTS.items()}
    out["halfdome_maps"]["mass_edges_log10"] = M_EDGES_HD
    out["halfdome_maps"]["redshift_edges"] = Z_EDGES_HD
    out["halfdome_maps"]["dm_redshift_edges_note"] = "the DM map's 4 redshift groups cover z <= 1 (sources at z_s = 1)"
    # ---------------------------------------------------------------- 2. our engine on the same catalogue
    jell = sens["ell"]
    X = LC.X
    for fld, o in J_OBS.items():
        tr = np.asarray(sens[f"trunc_all__D0__{o}"], float)               # [68 edges, 106 ell]
        imp = np.asarray(sens[f"imp__D0__{o}"], float).sum(axis=0)        # [67 shells, 5 sel_ell]
        res = {}
        for e in (100, 300, 1000, 3000):
            li = int(np.argmin(np.abs(jell - e)))
            si = list(sens["sel_ell"]).index(e)
            T = {f"{R:g}": float(tr[int(np.argmin(np.abs(X - R))), li] / tr[-1, li]) for R in (1.0, 1.5, 2.0, 3.0, 4.0)}
            shares = [float(imp[(X[:-1] >= lo - 1e-9) & (X[1:] <= hi + 1e-9), si].sum()) for lo, hi in COMMON]
            res[str(e)] = {"T_at": T, "band_importance_shares": shares,
                           "T_at_rule_radius": {r: (float(np.interp(Rg[r]["P" if fld == "tsz" else "E"], X, tr[:, li]) / tr[-1, li])
                                                     if Rg[r]["P" if fld == "tsz" else "E"] else None) for r in Rg}}
        out["engine"][fld] = res
    out["engine"]["note"] = ("FLAMINGO A2-ii stacks (C14, Test J D0) painted on the HalfDome lightcone catalogue with "
                             "the engine (Limber, flat sky, approximate kSZ 2-halo); T between edges: linear in R")
    # ---------------------------------------------------------------- 3. FLAMINGO / HalfDome ratios
    sa = LC.jload(LC.CODEX_FL / "stageA_summary.json")
    out["sources"]["flamingo_lightcone_sz_comparison_20261015/stageA_summary.json"] = LC.sha256(LC.CODEX_FL / "stageA_summary.json")
    fr = {}
    for fld in ("tsz", "ksz", "dm"):
        T1025 = out["halfdome_maps"][fld]["1025"]
        f_rt = "P" if fld == "tsz" else "E"
        fr[fld] = {"ell": sa[fld]["HalfDome"]["ell"]}
        for sim in ("L1_m9", "L2p8_m9"):
            r4 = sa[fld][sim]["ratio_to_halfdome"]
            fr[fld][sim] = {"ratio_4R": r4}
            for r in Rg:
                Rv = Rg[r][f_rt]
                fr[fld][sim][f"ratio_{r}_ESTIMATE"] = (r4 / interp_T(T1025["T_at_edges"], Rv)) if Rv else None
            fr[fld][sim]["ratio_cut_1R"] = r4 / T1025["T_at_edges"][4]
            fr[fld][sim]["ratio_cut_1.5R"] = r4 / T1025["T_at_edges"][5]
        fr[fld]["mean_native"] = {s: sa[fld][s]["mean_native"] for s in ("HalfDome", "L1_m9", "L2p8_m9")}
    out["flamingo_over_halfdome_1025"] = fr
    # redshift shares of the FLAMINGO maps (additive, common depth) and of the HalfDome maps used by Codex
    zs = {}
    for sim in ("L1_m9", "L2p8_m9", "HalfDome"):
        for fld in ("tsz", "ksz", "dm"):
            p = LC.CODEX_FL / "results" / sim / f"redshift_{fld}.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=False)
            out["sources"][str(p.relative_to(LC.FLAM))] = LC.sha256(p)
            ell = d["ell"].astype(float)
            if "cross_total" in d.files:
                ct = d["common_cross_total"] if "common_cross_total" in d.files else d["cross_total"]
                full = d["full_common"] if "full_common" in d.files else d["full_native"]
            else:
                ct = d["cross_matrix"].sum(axis=1)
                full = d["full_common"]
            edges = d["redshift_edges"].tolist()
            zs.setdefault(sim, {})[fld] = {"edges": edges, "bands": {
                name: (band(ell, ct, a)[1] / band(ell, full, a)[1]).tolist() for name, a in BAND_STARTS.items()}}
    out["redshift_shares_maps"] = zs
    # ---------------------------------------------------------------- 5. particle pilot
    pa = LC.jload(LC.CODEX_FL / "phaseB_pilot" / "pilot_assessment.json")
    out["sources"]["flamingo_lightcone_sz_comparison_20261015/phaseB_pilot/pilot_assessment.json"] = LC.sha256(
        LC.CODEX_FL / "phaseB_pilot" / "pilot_assessment.json")
    pil = {}
    for sim in ("L1_m9", "L2p8_m9"):
        p = LC.CODEX_FL / "phaseB_pilot" / f"{sim}_profiles_maps.npz"
        out["sources"][str(p.relative_to(LC.FLAM))] = LC.sha256(p)
        d = np.load(p, allow_pickle=False)
        e = d["radial_edges"]
        c = d["pressure_components_pa"]
        cats = d["category_names"].tolist()
        v = 4 * np.pi / 3 * np.diff(e**3)
        tot = c.sum(axis=0)
        hb = c[cats.index("central")] + c[cats.index("own_satellites")]
        cen = c[cats.index("central")]
        xr = LC.shell_radius(e[:-1], e[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            fb = np.where(tot > 0, hb / tot, np.nan)
            fc = np.where(tot > 0, cen / tot, np.nan)
        cum = {f"{R}": float((hb[e[1:] <= R + 1e-9] @ v[e[1:] <= R + 1e-9]) / (tot[e[1:] <= R + 1e-9] @ v[e[1:] <= R + 1e-9]))
               for R in (1, 2, 3, 4)}
        ref = {k: pa["runs"][sim]["spherical_radius_metrics"][k]["central_plus_satellites_fraction"] for k in cum}
        out["gates"][f"pilot_{sim}_cumulative_reproduced_max_abs"] = float(max(abs(cum[k] - ref[k]) for k in cum))
        pil[sim] = {"mass_msun": pa["runs"][sim]["mass_msun"], "log10_m200c": float(np.log10(pa["runs"][sim]["mass_msun"])),
                    "redshift": pa["runs"][sim]["redshift"], "x": xr.tolist(), "f_hb": fb.tolist(), "f_central": fc.tolist(),
                    "B32": LC.crossing_down(xr, fb, 0.32), "B50": LC.crossing_down(xr, fb, 0.50),
                    "B32_central": LC.crossing_down(xr, fc, 0.32),
                    "cumulative_hb_over_all": cum, "category_fractions_4R": dict(zip(cats, pa["runs"][sim]["pressure_volume_category_fractions"])),
                    "alternative_snapshot_cumulative": {k: pa["runs"][sim]["spherical_radius_metrics"][k]["alternative_snapshot_bound_fraction"] for k in cum}}
        # the bump that makes the last B32 crossing differ from the first (added 2026-10-16 after review)
        r1 = pil[sim]["B32"]["R_first"]
        beyond = (xr > r1) & np.isfinite(fb)
        jm = int(np.flatnonzero(beyond)[np.argmax(fb[beyond])])
        above = beyond & (fb >= 0.32)
        pil[sim]["bump_beyond_first_B32"] = {
            "f_hb_max": float(fb[jm]), "x_of_max": float(xr[jm]),
            "x_range_f_hb_ge_0.32": [float(xr[above].min()), float(xr[above].max())] if above.any() else None,
            "f_central_max": float(np.nanmax(fc[beyond])),
            "definition": "largest host-bound (central + own satellites) / all pressure of a shell beyond the first B32 "
                          "crossing, its shell radius (volume-weighted mean, R200c), the radii of the shells beyond the "
                          "first crossing with f_hb >= 0.32, and the largest central-only fraction there"}
    # our stacks and haloes in the top bins
    tab = LC.SC.halo_table()
    Pall = np.asarray(LC.SC.per_halo(LC.key("P", "all")))
    Phb = np.asarray(LC.SC.per_halo(LC.key("P", "hb")))
    Pcb = np.asarray(LC.SC.per_halo(LC.key("P", "cb")))
    ours = {}
    bins = LC.mass_bins()
    for b in (12, 13, 14):
        mb = bins[b]
        rows = np.asarray(mb["rows"])
        w = np.asarray(mb["weights"], float)
        per = {}
        stack = {}
        for R in (1, 2, 3, 4):
            m = LC.X[1:] <= R + 1e-9
            num = Phb[rows][:, m] @ LC.VOL[m]
            numc = Pcb[rows][:, m] @ LC.VOL[m]
            den = Pall[rows][:, m] @ LC.VOL[m]
            ph = num / den
            per[f"{R}"] = {"median": float(np.median(ph)), "q16": float(np.percentile(ph, 16)),
                           "q84": float(np.percentile(ph, 84)), "min": float(ph.min()), "max": float(ph.max())}
            stack[f"{R}"] = {"hb": float((w @ num) / (w @ den)), "cb": float((w @ numc) / (w @ den))}
        lab = LC.LABELS[b]
        ours[lab] = {"N": int(len(rows)), "mean_logm200c": float(mb["mean_logm200c"]), "stack_cumulative": stack,
                     "per_halo_cumulative_hb": per,
                     "B32": rad["bins"].get(lab, {}).get("rules", {}).get("B32", {}).get("P"),
                     "B32cb": rad["bins"].get(lab, {}).get("rules", {}).get("B32cb", {}).get("P")}
    out["pilot"] = {"codex": pil, "ours_pressure": ours,
                    "caveats": ["pilot haloes at z ~ 0.21 (snapshot 73), ours at z = 0", "pilot pressure includes recently "
                                "AGN-heated gas, ours removes it", "pilot 'all' = every gas particle in the sphere, like our 'all'",
                                "one halo per simulation"]}
    out["labels"] = {
        "T_at_edges / T_at_rule_radius_ESTIMATE / engine T_at": "share of the 4R-painted power from inside the radius "
            "(what a hard cut there would keep); a hard cut is NOT the painting plan (the fitted law is painted to "
            "4 R200c), so 1 - T is the part of the painted signal that depends on the extrapolated region",
        "ratio_4R": "FLAMINGO / HalfDome band power with HalfDome painted with B12/B16 to 4 R200c (reference comparison)",
        "ratio_<rule>_ESTIMATE, ratio_cut_*": "what a hard cut of HalfDome's painting at that radius would imply - not "
            "planned, reference only"}
    out["seconds"] = LC.time.perf_counter() - t0
    LC.jdump(out, LC.RESULTS / "codex.json")
    with open(LC.TABLES / "L1_codex_halfdome_retention.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "band_centre", "T_1R", "T_1.5R", "T_2R", "T_3R", "T_4R", "T_inside_B32_hardcut_not_planned_ESTIMATE",
                    "engine_T_1R", "engine_T_1.5R", "engine_T_2R", "engine_T_3R"])
        for fld in ("tsz", "ksz", "dm"):
            for name, v in out["halfdome_maps"][fld].items():
                if name not in BAND_STARTS:
                    continue
                Te = v["T_at_edges"]
                eng = {"77": "100", "126": "100", "325": "300", "1025": "1000", "3025": "3000"}[name]
                et = out["engine"][fld][eng]["T_at"]
                w.writerow([fld, f"{v['ell_centre']:.1f}", Te[4], Te[5], Te[6], Te[7], Te[8],
                            v["T_at_rule_radius_ESTIMATE"]["B32"], et["1"], et["1.5"], et["2"], et["3"]])
    LC.log("codex done", out["gates"])


if __name__ == "__main__":
    main()
