"""L1 gates -> verification.json (verification_smoke.json in smoke mode).

G0  inputs: partition closure all = a2ii + p10 + p11 (<= 1e-12 relative), the library's A2-ii equals the cache
    exactly, no A2-ii complement inside R200c, streamed host-bound counts equal Test J's J0 export exactly, J0 profile
    arrays equal the cache, R200m > R200c for every halo
G1  stacks: row 0 equals stackcore.stack_bin (<= 1e-12 relative) and the bootstrap rows equal stack_bin's replicates
    exactly (paired draws); merged groups in 12.7-13.5 equal Test J's stored groups
G2  fractions: inside R200c every fraction definition with the A2-ii or all-gas denominator coincides (a2ii = all
    there); nested bracket lo <= hi <= hb/a2ii; cb <= hb; 0 <= f <= 1 (+1e-12) in every shell and bootstrap row
G3  rule functions on synthetic profiles with known answers: crossing_down, cumulative_radius, budget_radius,
    local slope of a power law (exact for a quadratic in ln x), steepest point of a known minimum, zero crossing
G4  primary rule: B32 has a crossing (not censored) in every bin and both fields; B50 <= B32 in every bin
G5  additive terms: fallback + sum of Test J's per-bin own curves at 4R equals J's trunc_all at 4R (<= 1e-12), and
    T = 1 when every bin is cut at 4R
G6  Codex inputs: shell cross matrices sum to the full spectra, our band fractions reproduce the published CSV tables,
    the pilot's cumulative fractions are reproduced from its shell components
G7  smoke vs full: central values (row 0) of every rule in the smoke bins equal the full run's (full mode only, if the
    smoke results exist)
G8  report-number registry: every entry of results/report_numbers.json re-resolved from its source field.  This is a
    consistency check of the registry against the result files only; it does NOT parse the report prose or tables
    (that is G12 for the tables; clarified 2026-10-16 after review round 2)
G9  read-only: no file of the earlier tests or of the stacking cache was modified after this test's inputs were built
G10 observable-optimal single cut (full mode; added 2026-10-16 after review): every optimum lies strictly inside the
    search grid, its RMS is not above the RMS at the signal-weighted radius (+1e-5 for the 1e-4 grid step), and the
    vectorised own-curve interpolation equals the scalar one (own_at) at 200 random radii in [0, 4] (<= 1e-14 relative)
G11 extrapolated region (added 2026-10-16 after the user's clarification that R_t is the fit range and the law is still
    painted to 4 R200c): the published B12/B16 shell averages (closed-form cumulative) equal a 48-point quadrature of
    the engine's published_profile (<= 1e-10 relative); every share is 0 at R = 4 R200c; Test J's D3 fallback + own
    curves at 4R equal its trunc_all (<= 1e-12); the mean-DM lower reference lies between D3 and D0 (full mode);
    s_a2ii and s_hb of every bin recomputed from results/stacks_L.npz (row 0) with l_common.cumulative_at
    (<= 1e-10 relative); ordering s_hb <= s_all and s_a2ii <= s_all in every bin, field and bootstrap row
G12 report tables vs files (full mode; added 2026-10-16 after review round 2, which found two mis-rounded cells that G8
    could not catch): (a) every numeric table of the report is re-formatted here from the JSON by code independent of
    l_numbers.report_tables (3 decimals) and must equal tables/L1_report_tables.json row by row; (b) if the environment
    variable L_REPORT_TEXT names the report text, every row of every table must occur verbatim (leading whitespace
    stripped) as a line of that text; its SHA256 is recorded.  Without L_REPORT_TEXT only (a) runs and the gate says so.
"""
from __future__ import annotations

import os

import l_common as LC
import l_radii as LR
import l_stacks as LS
import numpy as np


def _expected_tables():
    """Independent re-formatting of the report tables from the result JSON (G12a)."""
    R = LC.jload(LC.RESULTS / "radii.json")["bins"]
    Gl = LC.jload(LC.RESULTS / "global.json")
    X = LC.jload(LC.RESULTS / "extrap.json")
    C = LC.jload(LC.RESULTS / "codex.json")
    N = LC.jload(LC.RESULTS / "stacks_summary.json")["bins"]

    def c(v):
        return "–" if v is None else format(float(v), ".3f")

    def cr(d):
        return "{} [{}, {}]".format(c(d["value"]), c(d["q16"]), c(d["q84"]))

    def row(*cells):
        return "| " + " | ".join(str(x) for x in cells) + " |"
    out = {}
    b = Gl["global"]["B32"]
    out["L1.0_global_radius"] = ["| | R200c | R200m | R500c |", "|---|---|---|---|"] + [
        row(f, cr(b[f]["R"]["family"]), c(b[f]["R200m"]["family"]["value"]), c(b[f]["R500c"]["family"]["value"]))
        for f in ("P", "E")] + [row("common", c(b["common"]["R"]), c(b["common"]["R200m"]), c(b["common"]["R500c"]))]
    d0 = X["observable_level"]["D0_share_beyond_xt"]
    sl = lambda key: " / ".join(c(d0[f"{key}@{e}"]) for e in (100, 1000, 3000))      # noqa: E731
    out["L1.0_D0_share_beyond_xt"] = ["| term | share from beyond x_t |", "|---|---|", row("mean y", c(d0["ymean"])),
                                      row("tSZ 1-halo, ell = 100 / 1000 / 3000", sl("yy1h")),
                                      row("mean DM", c(d0["dm_mean"])), row("DM variance", c(d0["dm_var"])),
                                      row("kSZ 1-halo, ell = 100 / 1000 / 3000", sl("ksz1h")),
                                      row("DM-field C_ell 1-halo, ell = 1000", c(d0["dmcl1h@1000"]))]
    rows = ["| bin | N | B32 P | B32 E | B32cb P / E | B50 P / E | C68 P / E | inside B32, P / E |",
            "|---|---|---|---|---|---|---|---|"]
    for lab in R:
        ru = R[lab]["rules"]

        def pe(rule, k=None):
            v = [ru[rule][f]["value"] if k is None else ru[rule][f][k]["value"] for f in ("P", "E")]
            return "{} / {}".format(c(v[0]), c(v[1]))
        rows.append(row(lab, N[lab]["N"], cr(ru["B32"]["P"]), cr(ru["B32"]["E"]), pe("B32cb"), pe("B50"), pe("C68"),
                        pe("B32", "inside_a2ii")))
    out["L1.1_per_bin_radii"] = rows
    keys = ("s_a2ii", "s_all", "s_hb", "s_pub", "t_pub")
    rows = ["| bin | P: A2-ii | P: all gas | P: bound | P: B12 shape | t_B12 | E: A2-ii | E: all gas | E: bound | "
            "E: B16 shape | t_B16 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ("12.7-12.9", "13.1-13.3", "13.5-13.7", "13.9-14.1", "14.5-14.7", "14.9-15.1", "15.1-15.3"):
        rows.append(row(lab, *[c(X["bins"][lab][f]["global"][k]["value"]) for f in ("P", "E") for k in keys]))
    fw = X["family_weighted"]
    rows.append(row("family-weighted", *[v for f in ("P", "E") for v in (
        c(fw[f]["global"]["s_a2ii"]["value"]), "–", c(fw[f]["global"]["s_hb"]["value"]),
        c(fw[f]["global"]["s_pub"]["value"]), "–")]))
    out["L1.2_extrapolated_shares"] = rows
    A = Gl["additive_terms"]
    rows = ["| term | field radius (P {:.4f} / E {:.4f}) | common x_t = {:.4f} | per-bin B32 radii |".format(
        A["ymean"]["B32"]["R_global"], A["dm_mean"]["B32"]["R_global"], X["x_t"]), "|---|---|---|---|"]
    for tag, nm in (("ymean", "mean y"), ("yy1h@1000", "tSZ 1-halo, ell = 1000"), ("ksz1h@100", "kSZ 1-halo, ell = 100"),
                    ("dm_mean", "mean DM")):
        rows.append(row(nm, c(A[tag]["B32"]["T_global"]), c(1 - d0[tag]), c(A[tag]["B32"]["T_perbin"])))
    out["L1.2_global_vs_perbin"] = rows
    names = (("tsz", "tSZ"), ("ksz", "kSZ"), ("dm", "DM"))
    out["L1.4_shell_shares"] = ["| | HalfDome B12/B16 (ell ≈ 77) | engine, FLAMINGO A2-ii (ell = 100) |", "|---|---|---|"] + [
        row(nm, " / ".join(map(c, C["halfdome_maps"][k]["77"]["common_shell_fractions"])),
            " / ".join(map(c, C["engine"][k]["100"]["band_importance_shares"]))) for k, nm in names]
    out["L1.4_flamingo_over_halfdome"] = ["| | L1_m9 | L2p8 |", "|---|---|---|"] + [
        row(nm, c(C["flamingo_over_halfdome_1025"][k]["L1_m9"]["ratio_4R"]),
            c(C["flamingo_over_halfdome_1025"][k]["L2p8_m9"]["ratio_4R"])) for k, nm in names]
    return out


def g12_report_tables():
    import hashlib
    exp = _expected_tables()
    p = LC.TABLES / "L1_report_tables.json"
    got = LC.jload(p) if p.exists() else {}
    mism = []
    for name, rows in exp.items():
        g_ = got.get(name)
        if g_ != rows:
            mism.append({"table": name, "expected": rows, "l_numbers": g_})
    res = {"n_tables": len(exp), "n_rows": sum(len(v) for v in exp.values()), "regeneration_mismatches": mism[:5]}
    ok = not mism and set(got) == set(exp)
    rp = os.environ.get("L_REPORT_TEXT")
    if rp and os.path.exists(rp):
        raw = open(rp, "rb").read()
        lines = {ln.strip() for ln in raw.decode("utf-8").splitlines()}
        missing = [(name, r) for name, rows in exp.items() for r in rows if r.strip() not in lines]
        res.update({"report_text_checked": True, "report_text_sha256": hashlib.sha256(raw).hexdigest(),
                    "report_text_chars": len(raw.decode("utf-8")), "rows_missing_from_report": missing[:20],
                    "n_rows_missing": len(missing)})
        ok = ok and not missing
    else:
        res.update({"report_text_checked": False,
                    "note": "L_REPORT_TEXT not set: only the regeneration part (a) ran"})
    res["passed"] = bool(ok)
    return res


def main():
    gates = {}
    meta = LC.jload(LC.INPUT_META)
    g = meta["gates"]
    ok0 = (all(g[f"{f}_all_eq_a2ii_p10_p11_max_rel"] <= 1e-12 for f in "PEG")
           and all(g[f"{f}_library_a2ii_equals_cache"] == 0 for f in "PEG")
           and all(g[f"{f}_complement_inside_R200c_max"] == 0 for f in "PEG")
           and g["count_hb_equals_J0_max_abs"] == 0 and g["count_cb_le_hb"]
           and all(g[f"J0_{k}_equals_cache"] == 0 for k in ("P_a2ii", "E_a2ii", "E_hb", "E_all"))
           and g["r200m_gt_r200c"])
    gates["G0_inputs"] = {"passed": bool(ok0), "detail": g}
    # G1
    ss = LC.jload(LC.RESULTS / "stacks_summary.json")
    worst, boot_exact, grp = 0.0, True, True
    for lab, d in ss["bins"].items():
        p = d["stackcore_parity_max_rel"]
        worst = max(worst, max(v for k, v in p.items() if k != "E_all_boot_replicates_max_rel"))
        boot_exact &= p["E_all_boot_replicates_max_rel"] == 0
        if d["groups_parity_with_J"] is not None:
            grp &= all(d["groups_parity_with_J"].values())
    gates["G1_stacks"] = {"passed": bool(worst <= 1e-12 and boot_exact and grp), "value_max_rel": worst,
                          "bootstrap_rows_exact": bool(boot_exact), "groups_equal_J": bool(grp)}
    # G2
    bad = []
    for b in LC.BINS:
        for f in ("P", "E"):
            fr = {n: LS.fraction(b, f, n) for n in LS.FRACTIONS}
            for n, (v, xr, grp_) in fr.items():
                if np.nanmin(v) < -1e-12 or np.nanmax(v) > 1 + 1e-12:
                    bad.append((b, f, n, "range", float(np.nanmin(v)), float(np.nanmax(v))))
            # inside R200c (hb groups): hb/all == hb/a2ii and the nested bracket coincides
            v1, x1, _ = fr["hb/all"]
            v2, _, _ = fr["hb/a2ii"]
            lo, _, _ = fr["hbnest_lo/a2ii"]
            hi, _, _ = fr["hbnest_hi/a2ii"]
            ins = x1 <= 1.0
            if np.nanmax(np.abs(v1[:, ins] - v2[:, ins])) > 1e-12 or np.nanmax(np.abs(lo[:, ins] - hi[:, ins])) > 1e-12:
                bad.append((b, f, "inside_R200c_mismatch"))
            if np.nanmax(lo - hi) > 1e-12 or np.nanmax(hi - v2) > 1e-12:
                bad.append((b, f, "bracket_order"))
            c1, _, _ = fr["cb/all"]
            st = LS.bin_stacks(b)["S"]
            if np.nanmax(st[f"{f}_cb"] - st[f"{f}_hb"]) > 1e-12 * np.nanmax(st[f"{f}_hb"]):
                bad.append((b, f, "cb_gt_hb"))
    gates["G2_fractions"] = {"passed": not bad, "failures": bad[:20]}
    # G3 synthetic tests
    t = {}
    xs = np.exp(np.linspace(np.log(0.3), np.log(3.8), 60))
    fx = 1 / (1 + (xs / 1.4) ** 6)                        # f = 0.5 at 1.4
    t["crossing_0.5"] = abs(LC.crossing_down(xs, fx, 0.5)["R"] - 1.4)
    Fc = np.ones(LC.N_X)                                   # constant density: X(<R) = 4pi/3 R^3
    t["cumulative_0.5"] = abs(LC.cumulative_radius(Fc, 0.5) - 4 * 0.5 ** (1 / 3))
    t["budget"] = abs(LC.budget_radius(Fc, 4 * np.pi / 3 * 1.7 ** 3) - 1.7)
    t["cumulative_at"] = abs(LC.cumulative_at(Fc, 2.3) - 4 * np.pi / 3 * 2.3 ** 3)
    sl = LR.slopes(np.exp(-3 * np.log(xs))[None, :], xs, 0.15)[0]
    m = (LR.SLOPE_GRID > 0.6) & (LR.SLOPE_GRID < 3.5)
    t["powerlaw_slope"] = float(np.nanmax(np.abs(sl[m] + 3)))
    yq = -2 - 0.8 * (np.log(LR.SLOPE_GRID) - np.log(1.6)) ** 2 * -1  # minimum -2 at 1.6
    t["steepest"] = abs(LC.steepest(LR.SLOPE_GRID, yq, 0.7, 3.6)["R"] - 1.6) / 1.6
    yz = 1 - LR.SLOPE_GRID / 2.2                           # zero at 2.2
    t["zero_cross"] = abs(LC.zero_cross_down(LR.SLOPE_GRID, yz, 1.0, 3.6)["R"] - 2.2) / 2.2
    tol = {"crossing_0.5": 5e-3, "cumulative_0.5": 1e-12, "budget": 1e-12, "cumulative_at": 1e-9,
           "powerlaw_slope": 1e-9, "steepest": 0.01, "zero_cross": 0.01}
    gates["G3_rule_functions"] = {"passed": all(t[k] <= tol[k] for k in t), "errors": t, "tolerances": tol}
    # G4
    rad = LC.jload(LC.RESULTS / "radii.json")
    fails = []
    for lab, d in rad["bins"].items():
        for f in ("P", "E"):
            b32 = d["rules"]["B32"][f]
            b50 = d["rules"]["B50"][f]
            if b32["value"] is None or b32["status"] != "ok":
                fails.append((lab, f, "B32 censored"))
            elif b50["value"] is not None and b50["value"] > b32["value"] + 1e-12:
                fails.append((lab, f, "B50 > B32"))
    gates["G4_primary_rule"] = {"passed": not fails, "failures": fails}
    # G5
    if not LC.SMOKE:
        sens = np.load(LC.J_DIR / "results" / "sens.npz")
        dev = 0.0
        for name in ("ymean", "dm_mean", "dm_var"):
            al = np.asarray(sens[f"trunc_all__D0__{name}"], float)
            own = np.asarray(sens[f"trunc_own__D0__{name}"], float)
            dev = max(dev, abs((al[0] + own[:, -1].sum()) / al[-1] - 1))
        for name in ("ksz1h", "dmcl1h"):
            al = np.asarray(sens[f"trunc_all__D0__{name}"], float)
            own = np.asarray(sens[f"trunc_own__D0__{name}"], float)
            for j, e in enumerate((100, 300, 1000, 3000, 8000)):
                li = int(np.argmin(np.abs(sens["ell"] - e)))
                dev = max(dev, abs((al[0, li] + own[:, -1, j].sum()) / al[-1, li] - 1))
        import l_global as LG
        c = LG.own_curve(sens, "D0", "ksz1h", 1000)
        t4 = sum(LG.own_at(c[k], 4.0) for k in range(len(c))) / c[:, -1].sum()
        gates["G5_additive_terms"] = {"passed": bool(dev <= 1e-12 and abs(t4 - 1) <= 1e-14), "max_rel_dev": dev,
                                      "T_at_4R_minus_1": float(t4 - 1)}
    # G6
    cx = LC.RESULTS / "codex.json"
    if cx.exists():
        cg = LC.jload(cx)["gates"]
        ok6 = (all(v <= 1e-7 for k, v in cg.items() if "shell_matrix" in k)
               and all(v <= 1e-8 for k, v in cg.items() if "vs_csv" in k)
               and all(v <= 1e-12 for k, v in cg.items() if k.startswith("pilot")))
        gates["G6_codex_inputs"] = {"passed": bool(ok6), "detail": cg}
    # G7
    sm = LC.HERE / "results_smoke" / "radii.json"
    if not LC.SMOKE and sm.exists():
        s = LC.jload(sm)
        dev, n = 0.0, 0
        for lab, d in s["bins"].items():
            for rule, ff in d["rules"].items():
                for f, v in ff.items():
                    a, b_ = v["value"], rad["bins"][lab]["rules"][rule][f]["value"]
                    if a is None and b_ is None:
                        continue
                    n += 1
                    if (a is None) != (b_ is None):
                        dev = np.inf
                    else:
                        dev = max(dev, abs(a - b_))
        gates["G7_smoke_vs_full"] = {"passed": bool(dev <= 1e-12), "max_abs_dev": dev, "compared": n}
    # G8
    nums = LC.jload(LC.RESULTS / "report_numbers.json") if (LC.RESULTS / "report_numbers.json").exists() else {}
    bad8 = []
    for name, e in nums.items():
        src = LC.jload(LC.HERE / e["source"])
        v = src
        try:
            for k in e["field"].split("/"):
                v = v[int(k)] if isinstance(v, list) else v[k]
        except (KeyError, IndexError, ValueError):
            bad8.append(name)
            continue
        if (v is None) != (e["value"] is None) or (v is not None and abs(float(v) * e["scale"] - e["value"]) >
                                                   1e-12 * max(1.0, abs(e["value"]))):
            bad8.append(name)
    gates["G8_report_numbers"] = {"passed": not bad8 and len(nums) > 0, "n": len(nums), "failures": bad8[:20]}
    # G9
    t_ref = os.path.getmtime(LC.INPUT_FILE)
    newer = []
    for sub in ("J_halo_dm_lowmass", "H_sensitivity", "I_fitting_a2ii", "K_soap_battaglia", "G_physical_outer",
                "F_observables", "stacking_choices_20261014/cache", "stacking_choices_20261014/core", "analysis"):
        for root, dirs, files in os.walk(LC.STUDY / sub):
            dirs[:] = [d_ for d_ in dirs if d_ != "__pycache__"]
            for fn in files:
                p = os.path.join(root, fn)
                if os.path.getmtime(p) > t_ref:
                    newer.append(os.path.relpath(p, LC.STUDY))
    for p in ("measurements_owner.npz", "measurements_partition.npz", "cluster_tests.py"):
        if os.path.getmtime(LC.STUDY / p) > t_ref:
            newer.append(p)
    gates["G9_read_only"] = {"passed": not newer, "modified_after_inputs": newer[:20]}
    # G10
    if not LC.SMOKE:
        import l_global as LG
        gl = LC.jload(LC.RESULTS / "global.json")
        oo = gl.get("observable_optimal", {})
        bad10, n10 = [], 0
        lo_g, hi_g = oo.get("grid", [1.0, 2.5])
        for rule in LG.OPT_RULES:
            for fam, r in oo.get(rule, {}).items():
                n10 += 1
                if not (lo_g < r["x_opt"] < hi_g):
                    bad10.append((rule, fam, "x_opt at grid edge"))
                if r["rms_at_signal_weighted"] is not None and r["rms_at_opt"] > r["rms_at_signal_weighted"] + 1e-5:
                    bad10.append((rule, fam, "rms_at_opt > rms at the signal-weighted radius"))
        sens = np.load(LC.J_DIR / "results" / "sens.npz")
        c = LG.own_curve(sens, "D0", "dm_mean", None)
        xs = np.random.default_rng(20261016).uniform(0, 4, 200)
        dv = max(abs(float(LG.own_at_vec(c[k], xs)[i]) / LG.own_at(c[k], xs[i]) - 1)
                 for k in range(len(c)) for i in range(len(xs)) if LG.own_at(c[k], xs[i]) != 0)
        gates["G10_observable_optimal"] = {"passed": bool(not bad10 and n10 > 0 and dv <= 1e-14), "n_optima": n10,
                                           "failures": bad10, "own_at_vec_vs_own_at_max_rel": dv}
    # G11
    ex_p = LC.RESULTS / "extrap.json"
    if ex_p.exists():
        ex = LC.jload(ex_p)
        eg = ex["gates"]
        bad11 = []
        if max(eg["P_published_shell_avg_vs_engine_quadrature_max_rel"],
               eg["E_published_shell_avg_vs_engine_quadrature_max_rel"]) > 1e-10:
            bad11.append("published shell averages")
        if eg["shares_at_R4_max_abs"] > 1e-12:
            bad11.append("shares at 4R")
        if "D3_fallback_plus_own_at_4R_vs_trunc_all_rel" in eg and eg["D3_fallback_plus_own_at_4R_vs_trunc_all_rel"] > 1e-12:
            bad11.append("D3 closure")
        if "lower_between_D3_and_D0" in eg and not eg["lower_between_D3_and_D0"]:
            bad11.append("mean-DM lower reference outside [D3, D0]")
        st = np.load(LC.RESULTS / "stacks_L.npz")
        x_t = ex["x_t"]
        dev11 = 0.0
        for i, b in enumerate([int(v) for v in st["bins"]]):
            lab = LC.LABELS[b]
            for f in ("P", "E"):
                Sa, Sh = st[f"stack_{f}_a2ii"][i], st[f"stack_{f}_hb"][i]
                xin = LC.cumulative_at(Sa, x_t)
                for k, S in (("s_a2ii", Sa), ("s_hb", Sh)):
                    rho = (LC.cumulative_at(S, 4.0) - LC.cumulative_at(S, x_t)) / xin
                    ref = ex["bins"][lab][f]["global"][k]["value"]
                    dev11 = max(dev11, abs(rho / (1 + rho) / ref - 1))
        en = np.load(LC.RESULTS / "extrap.npz")
        order_bad = 0
        for b in [int(v) for v in st["bins"]]:
            for f in ("P", "E"):
                for which in ("global", "perbin"):
                    sa, sh, s_all = (en[f"s_a2ii__{f}__{which}__{b}"], en[f"s_hb__{f}__{which}__{b}"],
                                     en[f"s_all__{f}__{which}__{b}"])
                    order_bad += int(np.sum(sh > s_all + 1e-6) + np.sum(sa > s_all + 1e-6))     # float32 storage
        if dev11 > 1e-10:
            bad11.append("s recomputation")
        if order_bad:
            bad11.append(f"ordering ({order_bad})")
        gates["G11_extrapolated_region"] = {"passed": not bad11, "failures": bad11, "detail": eg,
                                            "s_recomputed_max_rel": dev11, "ordering_violations": order_bad}
    # G12
    if not LC.SMOKE:
        gates["G12_report_tables"] = g12_report_tables()
    out = {"all_passed": all(v["passed"] for v in gates.values()), "mode": LC.MODE, "gates": gates,
           "definitions": __doc__}
    LC.jdump(out, LC.HERE / ("verification_smoke.json" if LC.SMOKE else "verification.json"))
    LC.log("verification", {k: v["passed"] for k, v in gates.items()}, "ALL" if out["all_passed"] else "FAILED")


if __name__ == "__main__":
    main()
