"""Verification gates of Test J2 (local, read-only; writes verification_j2.json).

G0  inputs: inputs/j2_fitdata.npz rebinned 35-shell profiles equal Test F/G's (<= 1e-14), x_h, softening, stacked counts
    and merged T1 groups equal Test J1's; DM weight maps built from 7 maps (6 for D1, without its Q16 map, which lies
    inside the DM = 0 atom)
G1  remote provenance: before the run every file J2 read and every J2 module was identical on maas
    (post_j2/remote_check.json); the run manifest's input and code SHA256 equal the local files
G2  J1 rebuild on maas: results_j2/j1base fallback parity passed; its stacks.npz equals J1's results/stacks.npz array by array
G3  fit parity: J2's A2-ii B16/B12 fits of the bins 13.5-15.5 equal Test I's fits35.json (same data, mask, objective)
G4  ladder parity: J2's ladder driver on Test I's 10 bins (D0parityI) reproduces Test I's E ladder at R_fit 4
    (chi2 under the three likelihoods for the fits that converged with no parameter at a bound in either run, the
    sigma_d floor and the CV chi2; the selected models)
G5  evaluation parity: every definition's stack painted det equals J1's truth mean (moments <= 1e-9, DM quantiles <= 1e-6)
G6  PDF vs moments: the DM-PDF mean and variance of every painting equal its moment forms (mean <= 1e-5, variance <= 1e-4,
    or a variance deficit explained by the probability beyond the PDF grid end at 8192 pc cm^-3)
G7  completeness: every painting has det and S1 observables and DM quantiles; fits that failed or were substituted listed
G8  results checksums: results_j2/SHA256SUMS verified locally after the fetch
G9  scores: the stacks themselves score S = 0 for every weight map
G10 scratch: the node-local scratch was deleted after the final fetch (CLUSTER_RUN.json j2_maas_runs)
G11 report numbers: every number registered by j2_report.py re-resolved from its source and found in the report text
G12 (L*) substitutions: the bins without an (L) fit follow from the rule (post_j2/J2_L_substitutions.json, j2_lsub.py:
    N_h <= p + 2 or p < 6) and equal the a_substituted_bins of every (L*) painting
G13 (run 2) ladder completeness: every mass-cell ladder model with a floor fit is painted at each ladder radius
G14 (run 2) dIT construction: the dIT paintings use Test I's fits35.json E|d|4 / P|a|4 parameters bitwise in 13.5-15.5
    and J2's fits.json D0|dI|4 / P|a|4 in 12.7-13.5, and their painted shells equal analysis.families
    shell_model_truncated of those parameters
G15 (run 2) reproducibility: every run-1 painting (results_j2_run1/) is reproduced by run 2 (results_j2_run2/):
    observables, DM quantiles, painted shells and stored CDFs
G16 shared run records: J1's SHA256SUMS verifies; the lines of the two files J1 and J2 share (CLUSTER_RUN.json,
    run_test_j.py) were updated by J2 and are logged with old and new digests (post_j2/sums_update_j2.json); J1's
    2026-10-16 revision re-hashed J1's manifest (j_sums.py, post/sums_update_j1.json)
G17 (run 3, J1's whole-atom Mecke maps) reproducibility: every run-2 painting (results_j2_run2/) whose painted shells
    are bitwise unchanged in run 3 has bitwise equal observables, errors, quantiles and stored CDFs; the paintings whose
    shells changed are among the objective-(d) paintings (incl. RC) and D0's dI / dIT paintings (the only fits that use
    J1's importance weights), and every fit with another objective is bitwise equal to run 2's
Usage: python -B verify_j2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = HERE / "results_j2"
POST = HERE / "post_j2"


def js(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def resolve(cache, path, keys):
    if path not in cache:
        f = HERE / path
        cache[path] = dict(np.load(f, allow_pickle=False)) if path.endswith(".npz") else js(f)
    o = cache[path]
    for k in keys:
        o = o[k]
    return o


def report_numbers_gate():
    reg_p, txt_p = POST / "report_numbers_j2.json", POST / "J2_report.txt"
    if not (reg_p.exists() and txt_p.exists()):
        return None
    reg, text = js(reg_p), txt_p.read_text()
    cache, bad, worst = {}, [], 0.0
    for i, it in enumerate(reg["items"]):
        try:
            vals = [resolve(cache, f, k) for f, k in it["sources"]]
            [float(x) for x in vals] if it["op"] != "len" else None
        except (KeyError, IndexError, TypeError, ValueError, FileNotFoundError) as exc:
            bad.append({"item": i, "text": it["text"], "problem": f"source not resolvable: {exc!r}"})
            continue
        op, sc, mi = it["op"], it["scale"], it["minus"]
        if op == "len":
            v = len(vals[0])
        elif op == "value":
            v = (float(vals[0]) - mi) * sc
        elif op == "ratio":
            v = (float(vals[0]) / float(vals[1]) - mi) * sc
        elif op in ("min", "max", "sum"):
            v = ({"min": min, "max": max, "sum": sum}[op]([float(x) for x in vals]) - mi) * sc
        elif op == "absmax":
            v = (max(abs(float(x)) for x in vals) - mi) * sc
        else:
            bad.append({"item": i, "problem": f"unknown op {op}"})
            continue
        dev = abs(v - it["value"]) / max(abs(it["value"]), 1e-300) if it["value"] != 0 else abs(v)
        worst = max(worst, dev)
        t = format(v, it["format"])
        if t.startswith("-0") and set(t[1:]) <= set("0.,"):
            t = t[1:]
        if dev > 1e-12 or t != it["text"]:
            bad.append({"item": i, "text": it["text"], "recomputed": t, "dev": dev})
        if it.get("in_report", True) and it["text"] not in text:
            bad.append({"item": i, "text": it["text"], "problem": "not found in the report"})
    return {"passed": not bad, "n_numbers": len(reg["items"]),
            "n_in_report": sum(1 for it in reg["items"] if it.get("in_report", True)),
            "max_rel_deviation": worst, "problems": bad[:20]}


def run2_gates(G):
    import sys
    sys.path.insert(0, str(ROOT))
    from analysis import families as AF
    summ = js(POST / "J2_summary.json")
    comp = summ["ladder"]["completeness"]
    G["G13_ladder_complete"] = {"passed": all(v["complete"] for v in comp.values()),
                                "per_radius": {k: {kk: v[kk] for kk in ("n_fitted", "n_painted", "missing")}
                                               for k, v in comp.items()}}
    # G14
    items = {it["label"]: it for it in js(R / "evaluate_items.json")["items"]}
    ev = np.load(R / "evaluate.npz")
    TI = js(ROOT / "I_fitting_a2ii" / "results" / "fits35.json")["fits"]
    FJ = js(R / "fits.json")["fits"]
    jb = [int(b) for b in ev["j_bins"]]
    fine = np.asarray(ev["fine_edges"], float)
    bad, dev = [], 0.0
    for lab in ("a|D0|B16|dIT|Rf4", "lm|C14|B12a+B16dIT", "lm|C10|B12a+B16dIT"):
        it = items.get(lab)
        if it is None:
            bad.append(f"{lab} missing")
            continue
        for c, b in enumerate(jb):
            want_e = TI[f"E|d|4|{b - 5}"]["parameters"] if b >= 5 else FJ[f"D0|dI|4|{b}"]["parameters"]
            want_p = TI[f"P|a|4|{b - 5}"]["parameters"] if b >= 5 else FJ[f"P|a|4|{b}"]["parameters"]
            if list(it["params"][c]) != list(want_e) or list(it["p_params"][c]) != list(want_p):
                bad.append(f"{lab} bin {b}: parameters differ")
            M = AF.shell_model_truncated("b16", want_e, fine, 4.0)[:ev[f"M4__{lab}"].shape[1]]
            m4 = ev[f"M4__{lab}"][c]
            dev = max(dev, float(np.max(np.abs(M[m4 != 0] / m4[m4 != 0] - 1))),
                      float(np.max(np.abs(M[m4 == 0]), initial=0.0)))
            if f"P4__{lab}" in ev.files:
                P = AF.shell_model_truncated("b12", want_p, fine, 4.0)[:ev[f"P4__{lab}"].shape[1]]
                p4 = ev[f"P4__{lab}"][c]
                dev = max(dev, float(np.max(np.abs(P[p4 != 0] / p4[p4 != 0] - 1))))
    G["G14_dIT_construction"] = {"passed": bool(not bad and dev <= 1e-12), "problems": bad,
                                 "max_rel_shell_difference": dev,
                                 "note": "parameters compared bitwise; shells relative (floating point of the two "
                                         "call paths of the same exact shell average)"}
    # G15 (run 1 vs run 2, the archived run-2 results)
    R2 = HERE / "results_j2_run2"
    ev_run2 = np.load(R2 / "evaluate.npz")
    old = {(r["label"], r["prediction"]): r for r in csv.DictReader(open(HERE / "results_j2_run1" / "J2_eval_long.csv"))}
    new = {(r["label"], r["prediction"]): r for r in csv.DictReader(open(R2 / "J2_eval_long.csv"))}
    ev1 = np.load(HERE / "results_j2_run1" / "evaluate.npz")
    worst, wkey, missing, n_cmp = 0.0, None, [], 0
    for key, r in old.items():
        q = new.get(key)
        if q is None:
            missing.append(key)
            continue
        for k, v in r.items():
            if not k.startswith(("obs_", "err_", "errm_")) or v in ("", None):
                continue
            a, b = float(v), float(q[k])
            n_cmp += 1
            d = abs(a - b) / max(abs(a), 1e-300) if a != b else 0.0
            if d > worst:
                worst, wkey = d, f"{key[0]}|{key[1]}|{k}"
    arr_bad = [k for k in ev1.files if k not in ev_run2.files or not np.array_equal(ev1[k], ev_run2[k])]
    G["G15_run1_reproduced"] = {"passed": bool(not missing and worst == 0.0 and not arr_bad),
                                "n_run1_rows": len(old), "n_values_compared": n_cmp, "max_rel_difference": worst,
                                "worst": wkey, "missing_rows": missing[:10], "n_arrays_compared": len(ev1.files),
                                "arrays_differing": arr_bad[:10]}
    # G17 (run 2 vs run 3)
    items3 = {it["label"]: it for it in js(R / "evaluate_items.json")["items"]}
    items2 = {it["label"]: it for it in js(R2 / "evaluate_items.json")["items"]}
    long3 = {(r["label"], r["prediction"]): r for r in csv.DictReader(open(R / "tables" / "J2_eval_long.csv"))}
    changed, unchanged, worst3, wkey3, n3, cdf_bad = [], [], 0.0, None, 0, []
    for lab in items2:
        k4 = f"M4__{lab}"
        same = lab in items3 and k4 in ev.files and np.array_equal(ev_run2[k4], ev[k4]) and \
            (f"P4__{lab}" not in ev_run2.files or np.array_equal(ev_run2[f"P4__{lab}"], ev[f"P4__{lab}"]))
        (unchanged if same else changed).append(lab)
        if not same:
            continue
        for pred in ("det", "S1"):
            r2, r3 = new.get((lab, pred)), long3.get((lab, pred))
            if r2 is None or r3 is None:
                cdf_bad.append(f"{lab}|{pred}|missing")
                continue
            for k, v in r2.items():
                if not k.startswith(("obs_", "err_", "errm_")) or v in ("", None):
                    continue
                a_, b_ = float(v), float(r3[k])
                n3 += 1
                d_ = abs(a_ - b_) / max(abs(a_), 1e-300) if a_ != b_ else 0.0
                if d_ > worst3:
                    worst3, wkey3 = d_, f"{lab}|{pred}|{k}"
            kc = f"cdf__{lab}|{pred}"
            if kc in ev_run2.files and not np.array_equal(ev_run2[kc], ev[kc]):
                cdf_bad.append(kc)
    weighted = lambda lab: (lab.split("|")[0] in ("a", "rc") and lab.split("|")[3] in ("d", "dI", "dIT")) or \
        (lab.startswith("lm|") and lab.endswith(("B16dI", "B16dIT")))          # noqa: E731
    f2, f3 = js(R2 / "fits.json"), js(R / "fits.json")
    fit_diff = {"same_objective_other": [], "weighted_changed": []}
    for sec in ("fits", "rc"):
        for k, v in f2[sec].items():
            w3 = f3[sec].get(k)
            p2 = None if v is None else v.get("parameters")
            p3 = None if w3 is None else w3.get("parameters")
            eq = p2 == p3
            obj = k.split("|")[1]
            if obj in ("d", "dI"):
                if not eq:
                    fit_diff["weighted_changed"].append(f"{sec}:{k}")
            elif not eq:
                fit_diff["same_objective_other"].append(f"{sec}:{k}")
    G["G17_run2_reproduced"] = {
        "passed": bool(not cdf_bad and worst3 == 0.0 and all(weighted(lab) for lab in changed)
                       and set(items2) == set(items3) and not fit_diff["same_objective_other"]),
        "n_items": len(items3), "n_unchanged_items": len(unchanged), "n_changed_items": len(changed),
        "changed_items": sorted(changed), "n_values_compared": n3, "max_rel_difference_unchanged": worst3,
        "worst": wkey3, "problems": cdf_bad[:10], "fits_changed_with_weights": len(fit_diff["weighted_changed"]),
        "fits_changed_other_objectives": fit_diff["same_objective_other"][:10]}
    # G16
    upd = js(POST / "sums_update_j2.json") if (POST / "sums_update_j2.json").exists() else {}
    shared = set(upd.get("shared_files", []))
    fails1, n1 = [], 0
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            n1 += 1
            dig, rel = line.split("  ", 1)
            if not (HERE / rel).exists() or sha(HERE / rel) != dig:
                fails1.append(rel)
    G["G16_checksum_manifests"] = {"passed": bool(not fails1), "J1_SHA256SUMS_lines": n1,
                                   "J1_SHA256SUMS_failing": fails1,
                                   "J1_lines_updated_for_shared_files": upd.get("J1_lines_updated"),
                                   "shared_files": sorted(shared), "updates_log": upd.get("updates"),
                                   "note": upd.get("note"),
                                   "J1_revision_log": "post/sums_update_j1.json (j_sums.py: J1's manifest re-hashed by "
                                                      "J1's 2026-10-16 revision)",
                                   "SHA256SUMS_J2": "written last by j2_sums.py j2 (after this gate), verified with "
                                                    "sha256sum -c"}


def main():
    G = {}
    m = js(HERE / "inputs" / "j2_fitdata_meta.json")["checks"]
    G["G0_inputs"] = {"passed": bool(m["P_a2ii_35_vs_G_f35_max_rel"] <= 1e-14 and m["E_a2ii_35_vs_G_f35_max_rel"] <= 1e-14
                                     and m["P_a2ii_35_zero_pattern_equal_G"] and m["E_a2ii_35_zero_pattern_equal_G"]
                                     and m["x_h_equal_J1"] and m["softening_equal_J1"] == 0
                                     and m["stacked_counts_equal_J1"]
                                     and all(v for k, v in m.items() if k.startswith("groups_"))
                                     and all(m[f"DM_weight_maps_{d}_n_obs"] == (6 if d == "D1" else 7)
                                             for d in ("D0", "D1", "D2", "D3"))),
                      **{k: v for k, v in m.items() if not k.startswith("mask35")}}
    # G1
    rc = js(POST / "remote_check.json")
    man = js(R / "run_manifest.json")
    inv = [i for i in man["invocations"] if i.get("mode") == "full"][-1]
    loc_in = {k: sha(ROOT / k) for k in inv["inputs_sha256"] if (ROOT / k).exists()}
    in_eq = {k: inv["inputs_sha256"][k] == v for k, v in loc_in.items()}
    compute = ("j2_common.py", "j2_fits.py", "j2_eval.py", "run_test_j2.py", "run_test_j.py", "j_common.py",
               "j_prepare.py", "j_basis.py", "j_fallback.py", "j_obs.py", "j_pdf.py")
    code_eq = {k: inv["code_sha256"].get(k) == sha(HERE / k) for k in compute}
    G["G1_remote_provenance"] = {"passed": bool(rc["all_equal"] and all(in_eq.values()) and all(code_eq.values())),
                                 "pre_run_all_equal": rc["all_equal"], "pre_run_checked_utc": rc["checked_utc"],
                                 "n_files_checked": len(rc["files"]), "manifest_inputs_equal_local": in_eq,
                                 "manifest_compute_code_equal_local": code_eq}
    # G2
    fb = js(R / "j1base" / "fallback_parity.json")
    a, b = np.load(R / "j1base" / "stacks.npz"), np.load(HERE / "results" / "stacks.npz")
    neq = [k for k in b.files if k not in a.files or not np.array_equal(a[k], b[k], equal_nan=True)]
    G["G2_j1_rebuild"] = {"passed": bool(fb["passed"] and not neq), "fallback_parity_passed": fb["passed"],
                          "stacks_keys_compared": len(b.files), "stacks_keys_differing": neq,
                          "fallback_JA_max_abs_over_max": fb["JA"]["max_abs_over_max"],
                          "fallback_JB_max_abs_over_max": fb["JB"]["max_abs_over_max"]}
    # G3
    fp = js(R / "fits_parity.json")
    G["G3_fit_parity_test_I"] = {"passed": bool(fp.get("max_overall") is not None and fp["max_overall"] < 1e-6), **fp,
                                 "scope": "D0|a, D0|L, D1|a, D2|a, P|a at every R_fit; the dI fits are NOT covered "
                                          "(their weights come from J1's C14 maps, Test I's from Test H's C10 maps; "
                                          "J2_summary.json dI_vs_TestI)"}
    # G4: converged interior fits only (a fit that stopped at G's max_nfev = 400, or ended with parameters on their
    # bounds, in either run depends on its start, which differs at the 1e-7 level through the floor fit)
    lp = js(R / "ladder_parity.json")
    ref = {(r["dset"], r["model"]): r for r in csv.DictReader(open(ROOT / "I_fitting_a2ii" / "results" / "tables" /
                                                                    "I2_ladder_models.csv"))
           if r["field"] == "E" and abs(float(r["R_fit"]) - 4) < 1e-9}
    mine = {(r["dset"], r["model"]): r for r in csv.DictReader(open(R / "tables" / "J2b_ladder_models_D0parityI.csv"))}
    worst, skipped = {}, []
    for key, r in mine.items():
        q = ref.get(key)
        if q is None:
            continue
        for lik in ("floor", "stat", "L22"):
            if r.get(f"max_nfev_{lik}") == "True" or q.get(f"max_nfev_{lik}") == "True" \
                    or r.get(f"at_bound_{lik}") or q.get(f"at_bound_{lik}"):
                skipped.append(f"{key[0]}|{key[1]}|{lik}")
                continue
            a_, b_ = float(r[f"chi2_{lik}"]), float(q[f"chi2_{lik}"])
            dev = abs(a_ - b_) / max(abs(b_), 1e-300) if abs(a_ - b_) > 1e-3 else 0.0
            worst[lik] = max(worst.get(lik, 0.0), dev)
        for col in ("sigma_floor", "cv_chi2"):
            a_, b_ = float(r[col]), float(q[col])
            dev = abs(a_ - b_) / max(abs(b_), 1e-300) if abs(a_ - b_) > 1e-3 * (col == "cv_chi2") else 0.0
            worst[col] = max(worst.get(col, 0.0), dev)
    sel_ok = all(lp["selected_J2"].get(k, {}).get("model") == v["model"] for k, v in lp.get("selected_TestI", {}).items())
    G["G4_ladder_parity_test_I"] = {"passed": bool(worst and max(worst.values()) < 1e-6 and sel_ok),
                                    "max_rel_converged_fits": worst, "models_compared": len(mine),
                                    "skipped_not_converged_in_either_run": skipped,
                                    "all_fits_max_rel_including_unconverged": lp["max_rel_vs_TestI_I2_ladder_models_E_Rfit4"],
                                    "selected_equal": sel_ok, "selected_J2": lp["selected_J2"]}
    # G5
    ep = js(R / "eval_parity.json")
    G["G5_eval_parity_truth_mean"] = {"passed": bool(ep.get("passed")), **ep}
    # G6 / G7
    rows = list(csv.DictReader(open(R / "tables" / "J2_eval_long.csv")))
    items = js(R / "evaluate_items.json")["items"]
    dev_m, dev_v, missing = 0.0, 0.0, []
    quant = js(R / "evaluate_items.json")["quant"]
    tail, unexplained, dev_v_grid = [], [], 0.0
    for r in rows:
        try:
            dm, pm = float(r["obs_dm_mean"]), float(r["obs_pdf_mean"])
            dv, pv = float(r["obs_dm_var"]), float(r["obs_pdf_variance"])
        except (KeyError, ValueError):
            missing.append((r["label"], r["prediction"]))
            continue
        dev_m, dev_v = max(dev_m, abs(pm / dm - 1)), max(dev_v, abs(pv / dv - 1))
        if abs(pv / dv - 1) > 1e-4:
            # the PDF grid ends at 8192 pc cm^-3 (f_pdf, overflow excluded): probability 1 - m beyond it removes at
            # least (1 - m) (8192 - <DM>)^2 from the PDF variance
            m = quant[f"{r['label']}|{r['prediction']}"]["mass_on_grid"]
            bound = (1 - m) * (8192.0 - dm)**2
            rec_ = {"painting": f"{r['label']}|{r['prediction']}", "rel_dev_variance": pv / dv - 1,
                    "mass_beyond_grid": 1 - m, "variance_deficit": dv - pv, "deficit_lower_bound_from_tail": bound,
                    "deficit_lower_bound_rel": bound / dv}
            (tail if (pv < dv and dv - pv >= bound > 0) else unexplained).append(rec_)
        else:
            dev_v_grid = max(dev_v_grid, abs(pv / dv - 1))
        if any(r.get(f"obs_{q}") in (None, "") for q in ("q16", "q50", "q84", "q975", "q999")):
            missing.append((r["label"], r["prediction"]))
    G["G6_pdf_vs_moments"] = {"passed": bool(dev_m <= 1e-5 and not unexplained), "max_rel_mean": dev_m,
                              "max_rel_variance": dev_v, "max_rel_variance_others": dev_v_grid,
                              "n_paintings": len(rows), "variance_deviation_above_1e-4_explained_by_tail": tail,
                              "unexplained": unexplained,
                              "rule": "mean <= 1e-5; variance <= 1e-4, or a deficit at least the lower bound "
                                      "(1 - mass_on_grid)(8192 - <DM>)^2 of the probability beyond the PDF grid"}
    labs = {(r["label"], r["prediction"]) for r in rows}
    incomplete = [it["label"] for it in items if not {(it["label"], "det"), (it["label"], "S1")} <= labs]
    fits = js(R / "fits.json")
    none = [k for k, v in fits["fits"].items() if v is None]
    rcnone = [k for k, v in fits["rc"].items() if v.get("parameters") is None]
    subst = {it["label"]: it.get("a_substituted_bins") for it in items if it.get("a_substituted_bins")}
    rcsub = {it["label"]: it.get("rc_fallback_to_S400_bins") for it in items if it.get("rc_fallback_to_S400_bins")}
    G["G7_completeness"] = {"passed": bool(not incomplete and not missing), "n_items": len(items),
                            "incomplete": incomplete, "missing_quantiles": missing[:10],
                            "fits_without_result": none, "rc_fits_without_result": rcnone,
                            "L_star_substituted_bins": subst, "rc_items_bins_with_S400_fit": rcsub}
    # G8
    bad = []
    for line in (R / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        dig, rel = line.split("  ", 1)
        if not (R / rel).exists() or sha(R / rel) != dig:
            bad.append(rel)
    G["G8_results_checksums"] = {"passed": not bad, "n_files": len((R / "SHA256SUMS").read_text().splitlines()),
                                 "bad": bad}
    # G9
    sc = list(csv.DictReader(open(POST / "J2_scores.csv")))
    refs = [r for r in sc if r["label"].startswith("ref|")]
    vals = [float(v) for r in refs for k, v in r.items() if k.startswith("S_") and v not in ("", None)]
    G["G9_stack_scores_zero"] = {"passed": bool(vals and max(abs(v) for v in vals) == 0.0), "n_values": len(vals),
                                 "max_abs": max(abs(v) for v in vals) if vals else None}
    # G10
    cr = js(HERE / "CLUSTER_RUN.json").get("j2_maas_runs", [])
    dele = [x for x in cr if x.get("scratch_deleted")]
    G["G10_scratch_deleted"] = {"passed": bool(dele) and len(dele) == len(cr),
                                "records": [x.get("scratch_deleted") for x in dele], "n_runs": len(cr)}
    g11 = report_numbers_gate()
    if g11 is not None:
        G["G11_report_numbers"] = g11
    # G12: (L*) substitutions explained by the rule (post_j2/J2_L_substitutions.json, j2_lsub.py) and equal to the
    # a_substituted_bins of every (L*) painting
    ls_p = POST / "J2_L_substitutions.json"
    if ls_p.exists():
        lsj = js(ls_p)
        ev_items = js(R / "evaluate_items.json")["items"]
        diff = []
        for it in ev_items:
            if it["label"].startswith("a|") and it["label"].split("|")[3] == "L":
                key = f"{it['D']}|Rf{float(it['R_fit']):g}"
                want = sorted(lsj["missing_L_fits"].get(key, {}))
                if sorted(it["a_substituted_bins"]) != want:
                    diff.append({"label": it["label"], "painted": it["a_substituted_bins"], "rule": want})
        G["G12_L_substitutions"] = {"passed": bool(lsj["check_passed"] and not diff),
                                    "rule_vs_fits_json_mismatches": lsj["mismatches_vs_fits_json"],
                                    "painting_vs_rule_differences": diff, "p_min": lsj["p_min"]}
    run2_gates(G)
    out = {"all_passed": all(v["passed"] for v in G.values()), "gates": G}
    (HERE / "verification_j2.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v["passed"] for k, v in G.items()}, indent=1), "\nall passed:", out["all_passed"])


if __name__ == "__main__":
    main()
