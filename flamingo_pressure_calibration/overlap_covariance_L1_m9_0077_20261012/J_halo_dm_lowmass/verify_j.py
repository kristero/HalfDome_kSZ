"""Verification gates of Test J (local; reads the fetched results/, inputs/, post/ and Test H's results/).
Writes verification.json (or verification_smoke.json with --smoke).

G0  J0 inputs: P_a2ii, E_a2ii, E_all bitwise equal Test H's export; owned counts consistent (j0_meta.json)
G1  fallback reconstruction (j_fallback, C14 at 4 R200c) = the totals f_kernels accumulated in the bases (<= 1e-10)
G2  Test H parity, configuration C10 / D0 = Test H's A2-ii truth: moment observables and DM-PDF quantiles <= 1e-6;
    deposit identity exactly 0
G3  per-bin parity with Test H in the 10 common bins: each bin's own mean DM, DM variance, kSZ / DM-field / yy 1-halo
    terms (C10|D0) against Test H's trunc_own__a2ii__* end points (sens.npz) <= 1e-6
G4  sum rules: importance maps sum to 1 (moment observables <= 1e-10); Mecke DM-quantile maps (whole DM = 0 atom, J1
    revision 2026-10-16) within max(0.05, 3 sigma_boot) (per-halo subsample maps 'imp'; sigma_boot = the bootstrap
    SD of the subsample map sum, results_mecke_checks/) and 0.01 (truth-mean maps 'impdet') for every
    quantile outside the atom (Q > 0.125 pc cm^-3; D1's Q16 lies inside it), all four definitions; radial
    distribution: central finite differences (eps = 0.05, results_mecke_checks/mecke_fd.json) of the truth-mean
    quantiles for each radial band (and the outermost fine shell) of the calibrated bins against Q x the band sum of the
    truth-mean map, |ratio - 1| <= 0.03 for Q16-Q97.5 outside the atom and bands carrying >= 5% of the map; the
    outer-slab removal check against the exact truncation passes (trunc_all__D__dmq_mean) is reported (it includes the
    second-order response to a whole shell's removal; finite_difference.outermost_shell_q50_second_order: first and
    second central differences of Q50 in the outermost shell's scale factor and the second-order removal estimate);
    Q99.9 is reported only (finite_difference.q999_*)
G5  truncation curves: end point = the definition's truth; flat beyond the boundary; importance zero beyond it;
    cumulative curves reach exactly 1 at each boundary
G6  low-mass rule: every stacked shell of the 12.7-13.5 bins holds >= 400 particles; merging conserves each halo's
    integral per merged shell (<= 1e-12); no merged shell straddles 1, 2 R200c or x_h
G7  DM PDF vs moment forms (mean, variance) <= 2e-3 for every evaluated PDF (grid / overflow effects)
G8  spherical chord: engine column of a 3D-cut profile = the HalfDome chord integral (<= 1e-6 for the same nodes) and the
    unit chain DM = (2 R200c/(1+z)) int n_e dl (<= 1e-12)
G9  zero-hit probability: the D1 / D2 PDF atom (P(DM < 0.0625)) against exp(-lambda) from the halo-crossing count
G10 configuration checks: no painted lightcone halo in engine bin 12.5-12.7; C14 calibrates 14 bins; the z <= 1
    lightcone-mass summary (post/lightcone_mass_z.json) uses the histogram of the maas run (SHA256, per-bin counts)
G11 the compute modules on maas equal the local ones (post/remote_code_check.json)
G12 mean DM of every evaluation recomputed locally from ell = 0 kernels only (j_local_mean.py) <= 1e-10
G13 report transcription: every number registered by j_report.py (post/report_numbers.json) is re-resolved here from
    its source file and field with the declared operation (<= 1e-12 relative), re-formatted, and found in
    post/J1_report.txt (numbers flagged in_report)
G14 Mecke whole-atom revision (2026-10-16): the maas run results_mecke/ reproduced J1's reference PDFs and J1's
    (half-atom) Jacobians (mecke_parity.json, <= 1e-12) and its SHA256SUMS verifies; j_mecke_update.py reproduced the
    stored sens products byte for byte from J1's Jacobians before rebuilding them; the five rebuilt files equal the
    logged digests and the archive equals the maas originals; the added atom term is non-negative; the subsample-noise
    run (results_mecke_checks/: per-halo Mecke sums, SHA256SUMS) reproduces the whole-atom subsample map sums of the main
    pass to <= 1e-5 (float32 reference density); a toy compound
    Poisson (lambda = 1.058, D1's crossings; CIC grid, FFT) gives whole-atom map sums within 0.01 of 1 at Q50, Q84,
    Q97.5 while the half-atom formula falls short
Usage: python -B verify_j.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFS = ("D0", "D1", "D2", "D3")
CUT = {"D0": 4.0, "D1": 1.0, "D2": 2.0, "D3": 4.0}


def js(p):
    return json.loads(Path(p).read_text())


def resolve(src_cache, path, keys):
    """Independent re-resolution of a report number's source field (json walk or npz array index)."""
    if path not in src_cache:
        f = HERE / path
        src_cache[path] = dict(np.load(f, allow_pickle=False)) if path.endswith(".npz") else js(f)
    o = src_cache[path]
    for k in keys:
        o = o[k]
    return o


def report_numbers_gate():
    reg_p, txt_p = HERE / "post" / "report_numbers.json", HERE / "post" / "J1_report.txt"
    if not (reg_p.exists() and txt_p.exists()):
        return None
    reg, text = js(reg_p), txt_p.read_text()
    cache, bad, worst = {}, [], 0.0
    for i, it in enumerate(reg["items"]):
        try:
            vals = [resolve(cache, f, k) for f, k in it["sources"]]
        except (KeyError, IndexError, TypeError, FileNotFoundError) as exc:
            bad.append({"item": i, "text": it["text"], "problem": f"source not found: {exc!r}"})
            continue
        op, sc, mi = it["op"], it["scale"], it["minus"]
        if op == "len":
            v = len(vals[0])
        elif op == "value":
            v = (float(vals[0]) - mi) * sc
        elif op == "ratio":
            v = (float(vals[0]) / float(vals[1]) - mi) * sc
        elif op in ("min", "max", "sum"):
            a = {"min": min, "max": max, "sum": sum}[op]([float(x) for x in vals])
            v = (a - mi) * sc
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


def toy_mecke(lam_tot=1.058, n=200000, seed=1):
    """Compound Poisson with lognormal single-halo DM (median 20 pc cm^-3): Mecke sums (1/Q) sum lambda x f(Q - x) / f(Q)
    with the half-atom density (np.interp(Q - x, grid, f, left=0), J1's formula) and the whole-atom density (f shifted
    by one node with a zero at DM = 0, quantile + DX; j_mecke_atom.py); exact value 1 (homogeneity)."""
    dx, npts = 0.125, 2**15
    grid = np.arange(npts) * dx
    rng = np.random.default_rng(seed)
    x = np.exp(rng.normal(np.log(20.0), 1.2, n))
    x = x[x < grid[-1] / 4]
    lam = np.full(len(x), lam_tot / len(x))
    inten = np.zeros(npts)
    pos = x / dx
    i = np.floor(pos).astype(int)
    f = pos - i
    np.add.at(inten, i, lam * (1 - f))
    np.add.at(inten, i + 1, lam * f)
    pmf = np.maximum(np.fft.irfft(np.exp(np.fft.rfft(inten, 2 * npts) - inten.sum()), 2 * npts)[:npts], 0.0)
    cdf = np.cumsum(pmf)
    dens = pmf / dx
    P = np.array([0.5, 0.84, 0.975])
    Q = np.interp(P, cdf, grid + 0.5 * dx)
    fQ = np.interp(Q, grid, dens)
    sh = np.zeros_like(dens)
    sh[1:] = dens[:-1]
    half = [float(np.sum(lam * x * np.interp(q - x, grid, dens, left=0.0, right=0.0)) / fq / q) for q, fq in zip(Q, fQ)]
    whole = [float(np.sum(lam * x * np.interp(q + dx - x, grid, sh, left=0.0, right=0.0)) / fq / q) for q, fq in zip(Q, fQ)]
    return {"lambda": lam_tot, "pi0": float(pmf[0]), "levels": P.tolist(), "Q": Q.tolist(), "half_atom_sum": half,
            "whole_atom_sum": whole}


def mecke_revision_gate():
    mp_p, mu_p = HERE / "results_mecke" / "mecke_parity.json", HERE / "post" / "mecke_update.json"
    if not (mp_p.exists() and mu_p.exists()):
        return None
    mp, mu = js(mp_p), js(mu_p)
    import hashlib
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()       # noqa: E731
    bad_sums = []
    for line in (HERE / "results_mecke" / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            dig, rel = line.split("  ", 1)
            p = HERE / "results_mecke" / rel
            if not p.exists() or sha(p) != dig:
                bad_sums.append(rel)
    files_ok = {f: sha(HERE / "results" / f) == v["new_sha256"] for f, v in mu["files"].items()}
    arch_ok = {f: sha(HERE / mu["archive"] / f) == v for f, v in mu["maas_original_sha256"].items()}
    atom_min = min(v["atom_term_min"] for v in mu["map_change"].values())
    toy = toy_mecke()
    toy_ok = all(abs(v - 1) <= 0.01 for v in toy["whole_atom_sum"]) and all(v < 0.9 for v in toy["half_atom_sum"][:2])
    nz_p = HERE / "results_mecke_checks" / "mecke_noise.json"
    noise_ok, noise_dev = False, None
    if nz_p.exists():
        nz = js(nz_p)
        noise_dev = max(v["per_halo_total_vs_main_pass_max_abs"] for v in nz["per_definition"].values())
        for line in (HERE / "results_mecke_checks" / "SHA256SUMS").read_text().splitlines():
            if line.strip():
                dig, rel = line.split("  ", 1)
                p = HERE / "results_mecke_checks" / rel
                if not p.exists() or sha(p) != dig:
                    bad_sums.append("results_mecke_checks/" + rel)
        noise_ok = noise_dev is not None and noise_dev <= 1e-5
    fdb_p = HERE / "results_mecke_checks" / "mecke_fd.json"
    fd_base = max(v["base_vs_J1_truthmean_max_abs"] for v in js(fdb_p)["per_definition"].values()) \
        if fdb_p.exists() else None
    noise_ok = noise_ok and fd_base is not None and fd_base <= 1e-9
    worst_ref = max(r["quantiles_max_rel"] for rec in mp["reference_pdf"].values() for r in rec.values())
    worst_half = max(x["Jq_max_abs_diff_over_max"] for rec in mp["half_vs_J1"].values() for x in rec.values())
    return {"passed": bool(mp["passed"] and not bad_sums and all(files_ok.values()) and all(arch_ok.values())
                           and mu["reproduce_stored_from_J1_jacobians"]["passed"] and atom_min >= -1e-6 and toy_ok
                           and noise_ok),
            "noise_per_halo_sums_vs_main_pass_max_abs": noise_dev, "fd_base_vs_J1_truthmean_max_abs": fd_base,
            "maas_parity_passed": mp["passed"], "reference_quantiles_max_rel": worst_ref,
            "half_atom_jacobians_max_rel": worst_half, "results_mecke_SHA256SUMS_failing": bad_sums,
            "rebuilt_files_equal_logged": files_ok, "archive_equals_maas_originals": arch_ok,
            "reproduce_stored_from_J1_jacobians": mu["reproduce_stored_from_J1_jacobians"]["passed"],
            "atom_term_min": atom_min, "toy": toy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = HERE / ("results_smoke" if a.smoke else "results")
    G = {}
    meta = js(HERE / "inputs" / "j0_meta.json")
    c = meta["checks"]
    G["G0_inputs"] = {"passed": bool(c["P_a2ii_bitwise_equal_H0_P_a2ii"] and c["E_a2ii_bitwise_equal_H0_E_a2ii"]
                                     and c["E_all_bitwise_equal_H0_E_main"] and c["x200_edges_equal_H0_cache_edges"]
                                     and c["count_a2ii_le_count_all_violations"] == 0
                                     and c["count_hb_le_count_all_violations"] == 0
                                     and c["count_a2ii_zero_iff_E_a2ii_zero"]["count0_value_nonzero"] == 0
                                     and c["count_hb_zero_iff_E_hb_zero"]["count0_value_nonzero"] == 0),
                      "checks": c}
    fb = js(res / "fallback_parity.json")
    G["G1_fallback_parity"] = {"passed": bool(fb["passed"]), "JA": fb["JA"]["max_abs_over_max"],
                               "JB": fb["JB"]["max_abs_over_max"], "Jfb2h_JA": fb["JA"]["Jfb2h_max_abs_over_max"],
                               "Jfb2h_JB": fb["JB"]["Jfb2h_max_abs_over_max"]}
    par = js(res / "parity.json")
    G["G2_test_H_parity"] = {"passed": bool(par["passed"]), "moments_max_rel": par.get("H_moments_max_rel"),
                             "pdf_max_rel": par.get("H_pdf_max_rel"),
                             "deposit_identity_max_abs": par["deposit_identity_max_abs"],
                             "note": par.get("note")}
    tr = dict(np.load(res / "truth.npz"))
    ts = js(res / "truth_summary.json")
    prep = js(res / "prepare.json")
    jb = prep["j_bins"]
    # G3 per-bin parity with Test H
    if not a.smoke:
        hs = np.load(ROOT / "H_sensitivity" / "results" / "sens.npz")
        sel = [jb.index(b) for b in range(5, 15)]
        worst = {}
        for name, key in (("dm_mean", "dm_mean"), ("dm_var", "dm_var"), ("ksz1h", "ksz1h"), ("dmcl1h", "dmcl1h"),
                          ("yy1h", "yy1h"), ("ymean", "ymean")):
            mine = tr[f"C10|D0|T1|perbin__{key}"][sel]
            ref = hs[f"trunc_own__a2ii__{name}"][:, -1]
            with np.errstate(all="ignore"):
                r = np.abs(mine / ref - 1)
            worst[name] = float(np.nanmax(r[np.isfinite(r)]))
        G["G3_per_bin_parity_with_H"] = {"passed": bool(max(worst.values()) < 1e-6), "max_rel": worst}
    else:
        G["G3_per_bin_parity_with_H"] = {"passed": True, "note": "smoke: not applicable"}
    # G4 sum rules
    ss = js(res / "sens_summary.json")
    mom = {k: v for k, v in ss["sum_rules"].items() if "dm_q" not in k}
    q = {}
    noise_p = HERE / "results_mecke_checks" / "mecke_noise.json"
    noise = js(noise_p) if noise_p.exists() and not a.smoke else None
    for k, v in ss["sum_rules"].items():
        if "dm_q" not in k:
            continue
        kind, d, name = k.split("__")
        lab = f"C14|{d}|T1|truth" if kind == "imp" else f"C14|{d}|T1|truthmean"
        Q = ts["pdf"][lab]["quantiles"][("q16", "q50", "q84", "q975", "q999").index(name[3:])]
        p_atom = ts["pdf"][f"C14|{d}|T1|truth"]["p_first_bin"]
        thr = 0.05 if kind == "imp" else 0.01
        sig = None
        if kind == "imp" and noise is not None:
            sig = noise["per_definition"][d]["sigma_boot"][("q16", "q50", "q84", "q975", "q999").index(name[3:])]
            thr = max(thr, 3 * sig)
        applicable = bool(Q > 0.125)
        q[k] = {"deviation": v, "quantile": Q, "p_atom": p_atom, "threshold": thr, "sigma_boot": sig,
                "applicable": applicable, "passed": (v <= thr) if applicable else None}
    q_ok = all(r["passed"] for r in q.values() if r["applicable"])
    mu_p = HERE / "post" / "mecke_update.json"
    slab = {}
    if mu_p.exists() and not a.smoke:
        for d, rec in js(mu_p)["outer_slab_first_order"].items():
            r0 = rec["whole_atom"]["q50"][0]
            slab[d] = {"R_t": r0["R_t"], "R_D": r0["R_D"], "ratio": r0["ratio"], "exact_dQ": r0["exact_dQ"],
                       "ratio_half_atom_J1": rec["half_atom_J1"]["q50"][0]["ratio"]}
    # finite-difference test of the radial distribution (results_mecke_checks/mecke_fd.json): central differences of the
    # truth-mean quantiles for (1 +- 0.05) scalings of each radial band of the calibrated bins vs Q x the band sum of the
    # truth-mean map; applicable to Q16-Q97.5 outside the atom and bands carrying >= 5% of the map; |ratio - 1| <= 0.03
    fd_p = HERE / "results_mecke_checks" / "mecke_fd.json"
    fd = {}
    if fd_p.exists() and not a.smoke:
        S0 = dict(np.load(res / "sens.npz"))
        mu_ = js(mu_p)
        S_old = dict(np.load(HERE / mu_["archive"] / "sens.npz"))
        jb_ = [int(b) for b in S0["j_bins"]]
        for d, rec in js(fd_p)["per_definition"].items():
            Qd = np.asarray(ts["pdf"][f"C14|{d}|T1|truthmean"]["quantiles"], float)
            for name, st in rec["sets"].items():
                a0, b0 = st["shells"]
                for i, qn in enumerate(("q16", "q50", "q84", "q975", "q999")):
                    if Qd[i] <= 0.125:
                        continue
                    imp_new = float(np.asarray(S0[f"impdet__{d}__dm_{qn}"], float)[jb_][:, a0:b0].sum())
                    imp_old = float(np.asarray(S_old[f"impdet__{d}__dm_{qn}"], float)[jb_][:, a0:b0].sum())
                    fdv = st["dQ_dlnF_central"][i]
                    applicable = bool(abs(imp_new) >= 0.05 and qn != "q999" and fdv != 0)
                    r_new = Qd[i] * imp_new / fdv if fdv else None
                    fd[f"{d}|{name}|{qn}"] = {"band_importance": imp_new, "ratio_whole_atom": r_new,
                                              "ratio_half_atom_J1": Qd[i] * imp_old / fdv if fdv else None,
                                              "applicable": applicable,
                                              "passed": bool(abs(r_new - 1) <= 0.03) if applicable else None}
    fd_ok = bool(fd) and all(v["passed"] for v in fd.values() if v["applicable"])
    fd_app = [v for v in fd.values() if v["applicable"]]
    fd_summary = {"n_applicable": len(fd_app),
                  "max_abs_ratio_minus_1_whole_atom": max((abs(v["ratio_whole_atom"] - 1) for v in fd_app), default=None),
                  "max_abs_ratio_minus_1_half_atom_J1": max((abs(v["ratio_half_atom_J1"] - 1) for v in fd_app), default=None)}
    for d in DEFS:
        sel = [v for k, v in fd.items() if k.startswith(d + "|") and v["applicable"]]
        fd_summary[d] = {"n": len(sel),
                         "max_abs_ratio_minus_1_whole_atom": max((abs(v["ratio_whole_atom"] - 1) for v in sel), default=None),
                         "max_abs_ratio_minus_1_half_atom_J1": max((abs(v["ratio_half_atom_J1"] - 1) for v in sel), default=None)}
    q999 = [v for k, v in fd.items() if k.endswith("|q999") and abs(v["band_importance"]) >= 0.05]
    fd_summary["q999_max_abs_ratio_minus_1_whole_atom"] = max((abs(v["ratio_whole_atom"] - 1) for v in q999), default=None)
    # Q99.9 is reported, not gated: its pairs (bands with >= 5% of the map), their number and the worst pair
    q999_keys = [k for k, v in fd.items() if k.endswith("|q999") and abs(v["band_importance"]) >= 0.05]
    fd_summary["q999_n"] = len(q999_keys)
    fd_summary["q999_worst_pair"] = (max(q999_keys, key=lambda k: abs(fd[k]["ratio_whole_atom"] - 1))
                                     if q999_keys else None)
    # second-order response of the truth-mean Q50 to the outermost fine shell (the shell of the outer-slab check):
    # Q' = (Q+ - Q-)/(2 eps) and Q'' = (Q+ + Q- - 2 Q)/eps^2 are the first and second central differences in that shell's
    # profile scale factor s at s = 1 (Q+- at s = 1 +- eps, results_mecke_checks/mecke_fd.json); removing the shell
    # (s: 1 -> 0) changes Q50 by Q' to first order and by Q' - Q''/2 to second order, against the exact change of the
    # truncation passes (post/mecke_update.json outer_slab_first_order, exact_dQ)
    so = {}
    if fd_p.exists() and mu_p.exists() and not a.smoke:
        fdj = js(fd_p)
        eps = float(fdj["eps"])
        for d, rec in fdj["per_definition"].items():
            st = rec["sets"]["outermost_shell"]
            q0, qp, qm = float(rec["quantiles"][1]), float(st["Q_plus"][1]), float(st["Q_minus"][1])
            d1, d2 = (qp - qm) / (2 * eps), (qp + qm - 2 * q0) / eps ** 2
            ex = slab[d]["exact_dQ"]
            so[d] = {"x_range": st["x_range"], "Q50": q0, "dQ_ds": d1, "d2Q_ds2": d2, "d2Q_over_dQ": d2 / d1,
                     "removal_dQ_first_order": d1, "removal_dQ_second_order": d1 - d2 / 2, "removal_dQ_exact": ex,
                     "removal_dQ_exact_over_Q50": ex / q0,
                     "map_first_order_over_exact": slab[d]["ratio"],
                     "second_order_over_exact": (d1 - d2 / 2) / ex}
    fd_summary["outermost_shell_q50_second_order"] = so
    G["G4_sum_rules"] = {"passed": bool(max(mom.values()) < 1e-10 and ((q_ok and fd_ok) or a.smoke)),
                         "moments_max_deviation": max(mom.values()), "quantiles": q, "outer_slab_q50": slab,
                         "finite_difference": fd_summary, "finite_difference_detail": fd,
                         "note": "Mecke quantile maps with the whole DM = 0 atom (j_mecke_atom.py): per-halo subsample "
                                 "maps (imp) within max(0.05, 3 sigma_boot) (sigma_boot: bootstrap SD of the subsample "
                                 "map sum, results_mecke_checks/mecke_noise.json), truth-mean maps (impdet) within 0.01, for "
                                 "every quantile outside the atom (Q > 0.125 pc cm^-3; D1's Q16 lies inside it: 34.7% of "
                                 "sightlines cross no R200c sphere); radial distribution: central finite differences "
                                 "(eps = 0.05) per band within 3% (Q16-Q97.5, bands with >= 5% of the map); the outer-slab "
                                 "removal check is reported only (second-order; outermost_shell_q50_second_order); Q99.9 "
                                 "finite differences are reported only (q999_*); not applicable on the smoke sky"}
    # G5 truncation
    chk = ss["checks"]
    end = max(max(v.values()) for k, v in chk.items() if k.startswith("truncation_endpoint"))
    flat = max(max(v.values()) for k, v in chk.items() if k.startswith("truncation_flat"))
    zero = max(v for k, v in chk.items() if k.startswith("zero_beyond_boundary"))
    S = dict(np.load(res / "sens.npz"))
    e = S["x_edges"]
    reach = {}
    for d in DEFS:
        kB = int(np.flatnonzero(np.isclose(e, CUT[d]))[0])
        for name in ("dm_mean", "dm_var"):
            cvec = S[f"trunc_all__{d}__{name}"][:kB + 1]
            C = (cvec - cvec[0]) / (cvec[-1] - cvec[0])
            reach[f"{d}__{name}"] = float(abs(C[-1] - 1))
        qm = S[f"trunc_all__{d}__dmq_mean"]
        reach[f"{d}__dmq_endpoint_equals_truthmean"] = float(np.max(np.abs(
            qm[-1] - np.asarray(ts["pdf"][f"C14|{d}|T1|truthmean"]["quantiles"]))))
    G["G5_truncation"] = {"passed": bool(end < 1e-10 and flat < 1e-12 and zero < 1e-12 and max(reach.values()) < 1e-12),
                          "endpoint_max_rel": end, "flat_beyond_boundary_max": flat, "importance_beyond_boundary_max": zero,
                          "cumulative_reaches_1": reach}
    # G6 low-mass rule
    rule = {"min_stacked_count": {}, "merge_conservation_max_rel": prep["merge_integral_conservation_max_rel"]}
    straddle = 0
    x = e
    for lab, rec in prep["J1a_by_bin"].items():
        b = ["12.5-12.7", "12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5"].index(lab) if lab in (
            "12.5-12.7", "12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5") else None
        if b is None or b not in prep["low_bins"]:
            continue
        for v in ("P_a2ii", "E_a2ii", "E_hb", "E_all"):
            rule["min_stacked_count"][f"{lab}|{v}"] = rec[f"merge_{v}"]["min_stacked_count"]
            for lo, hi, n in rec[f"merge_{v}"]["merged_groups"]:
                for edge in (1.0, 2.0, rec["x_h"]):
                    if lo < edge - 1e-9 and hi > edge + 1e-9:
                        straddle += 1
    rule["groups_straddling_1R_2R_xh"] = straddle
    G["G6_low_mass_rule"] = {"passed": bool(min(rule["min_stacked_count"].values()) >= 400 and straddle == 0
                                            and rule["merge_conservation_max_rel"] < 1e-12), **rule}
    # G7 PDF vs moments
    pvm = ts["pdf_vs_moments"]
    worst = max(max(abs(v["mean"]), abs(v["variance"])) for v in pvm.values())
    G["G7_pdf_vs_moments"] = {"passed": bool(worst < 2e-3), "max_abs_rel": worst,
                              "worst": max(pvm, key=lambda k: max(abs(pvm[k]["mean"]), abs(pvm[k]["variance"])))}
    # G8 chord
    ch = par["chord"]
    G["G8_spherical_chord"] = {"passed": bool(ch["engine_vs_chord_integral_same_nodes_max_rel"] < 1e-6
                                              and ch["engine_DM_vs_HalfDome_formula_SI_max_rel"] < 1e-12), **ch}
    # G9 zero-hit probability
    cr = ts["crossings"]
    z = {}
    for d in ("D1", "D2"):
        p_pdf = ts["pdf"][f"C14|{d}|T1|truth"]["p_first_bin"]
        p_cr = cr[f"{CUT[d]:g}R"]["p_no_crossing"]
        z[d] = {"pdf_P_DM_lt_0.0625": p_pdf, "exp_minus_lambda": p_cr, "rel": p_pdf / p_cr - 1}
    G["G9_zero_hit_probability"] = {"passed": bool(all(abs(v["rel"]) < 0.02 for v in z.values())), **z,
                                    "note": "the first PDF bin also holds sightlines whose summed DM is below 0.0625"}
    # G10 configuration
    lc = prep["lightcone"]
    G["G10_configuration"] = {"passed": bool(lc["engine_bin_12.5-12.7_haloes"] == 0 and (len(jb) == 14 or a.smoke)),
                              "engine_bin_12.5-12.7_haloes": lc["engine_bin_12.5-12.7_haloes"],
                              "calibrated_bins": jb, "min_logm200c_painted": lc["min_logm200c_painted"],
                              "haloes_above_15.388": lc["haloes_above_15.388"]}
    # z <= 1 lightcone masses (post/lightcone_mass_z.json, j_lightcone_mass.py): the local histogram must be the one the
    # maas run read (SHA256) and reproduce its per-bin counts and mean masses
    lmz = HERE / "post" / "lightcone_mass_z.json"
    if lmz.exists() and not a.smoke:
        par = js(lmz)["parity_with_results_prepare_lightcone"]
        G["G10_configuration"]["lightcone_mass_z_parity"] = par
        G["G10_configuration"]["passed"] = bool(G["G10_configuration"]["passed"] and par["passed"])
    # G12 independent local recomputation of the linear observable mean DM (j_local_mean.py, ell = 0 kernels only)
    lm = HERE / ("post_smoke" if a.smoke else "post") / "local_mean" / "mean_dm.json"
    if lm.exists():
        L = js(lm)["evaluations"]
        dev = {t: abs(v["dm_mean"] / ts["evaluations"][f"{t}|truth"]["dm_mean"] - 1) for t, v in L.items()
               if f"{t}|truth" in ts["evaluations"]}
        G["G12_local_mean_dm_recomputation"] = {"passed": bool(max(dev.values()) < 1e-10), "max_rel": max(dev.values()),
                                                "n_evaluations": len(dev)}
    rc = HERE / "post" / "remote_code_check.json"
    if rc.exists() and not a.smoke:
        r = js(rc)
        G["G11_remote_code_check"] = {"passed": bool(r.get("compute_modules_equal")), **r}
    g13 = None if a.smoke else report_numbers_gate()
    if g13 is not None:
        G["G13_report_numbers"] = g13
    g14 = None if a.smoke else mecke_revision_gate()
    if g14 is not None:
        G["G14_mecke_whole_atom_revision"] = g14
    out = {"all_passed": all(v["passed"] for v in G.values()), "gates": G}
    path = HERE / ("verification_smoke.json" if a.smoke else "verification.json")
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v["passed"] for k, v in G.items()}, indent=1), "\nall passed:", out["all_passed"])


if __name__ == "__main__":
    main()
