"""J stage 'truth' (J1c): observables of the halo electron component for the definitions D0-D3, the low-mass
treatments T1-T3, the bin-1 subset in the painted mass range, and the comparison configurations C10 (Test H: published
fallback below 10^13.5) and C0 (published B12/B16 everywhere), with the parity gates against Test H.

Evaluation tags '<configuration>|<definition>|<treatment>':
  C14|D0..D3|T1   the production truths (14 calibrated bins 12.7-15.388; fallback above 10^15.388 cut like the definition)
  C14|D0..D3|T2,T3  low-mass treatments (bins 12.7-13.5 only; >= 13.5 identical)
  C14|D0,D1,D3|S82  bin 12.7-12.9 stacked from its haloes with M200c >= 6.6e12 only (the painted mass range)
  C14|ALL|T1      all gas to 4 R200c (comparison);  C14|D3inJA|T1  D3 in the A2-ii-shape basis (within-shell shapes)
  C10|D0..D3|T1   Test H's support: the published fallback paints 12.82-13.5 (cut like the definition); C10|D0|T1 is
                  Test H's A2-ii truth (parity);  C10|ALL|T1  all gas (for the first-order host-bound estimate)
  C0|cutR|-       every halo published B12/B16, electrons cut at R = 1, 2, 4 R200c
Kinds: truth (scatter-inclusive: per-halo second moments, per-halo DM-PDF columns), truthmean (bin-mean profiles),
truthcal (calibrated haloes only).  DM-PDF passes: j_pdf (z_s = 1); Mecke quantile Jacobians for C14|D0..D3|T1.
Outputs: results/truth.npz, results/truth_summary.json, results/parity.json.
"""
from __future__ import annotations

import j_common as JC
import j_fallback as JF
import j_obs as JO
import j_pdf as JP
import numpy as np

FC, FB, FK, E, A = JC.FC, JC.FB, JC.FK, JC.E, JC.A
DEF_BASIS = {d: v["basis"] for d, v in JC.DEFS.items()}


def eval_list():
    ev = []
    for t in ("T1", "T2", "T3"):
        for d, v in JC.DEFS.items():
            ev.append((f"C14|{d}|{t}", v["basis"], "C14", v["sel"], v["cut"], t, False))
    if 1 in JC.J_BINS:
        for d in ("D0", "D1", "D3"):
            v = JC.DEFS[d]
            ev.append((f"C14|{d}|S82", v["basis"], "C14", v["sel"], v["cut"], "T1", True))
    ev.append(("C14|ALL|T1", "JA", "C14", "all", 4.0, "T1", False))
    ev.append(("C14|D3inJA|T1", "JA", "C14", "hb", 4.0, "T1", False))
    for d, v in JC.DEFS.items():
        ev.append((f"C10|{d}|T1", v["basis"], "C10", v["sel"], v["cut"], "T1", False))
    ev.append(("C10|ALL|T1", "JA", "C10", "all", 4.0, "T1", False))
    for cut in JC.CUTS:
        ev.append((f"C0|cut{cut:g}|-", "JA", "C0", "a2ii", cut, "T1", False))
    return ev


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(a / b - 1)
    r = r[np.isfinite(r) & (b != 0)]
    return float(r.max()) if r.size else 0.0


# ============================================================================ spherical-chord equivalence
def chord_check(logm=13.6, z=0.3, radius=1.0, n_b=120):
    """The engine column of a profile cut in 3D at ``radius`` (f_pdf: sum over the basis segments inside the cut of
    the exact piecewise-linear projection operator) against the HalfDome SphericalChordDMProfile formula
        DM(b) = (2 R200c / (1+z)) int_0^{sqrt(X^2 - b^2)} n_e(sqrt(b^2 + l^2)) dl,   X = radius, b and l in R200c,
    with n_e = E rho_c(z) f_b / mu_XGP (E: dimensionless electron profile, here the published B16 at log M = ``logm``).
    Returns the max relative differences (a) engine vs the chord integral of the same piecewise-linear node profile,
    (b) vs the chord integral of the smooth B16 (node discretisation), (c) the unit chain column_scale x Sigma vs
    the formula in physical units (pc cm^-3)."""
    from scipy.integrate import quad
    _, grid, seg_shell, _ = FK.basis_grid()
    n_cut = JC.n_seg_upto(radius)
    stop = grid.index[grid.segments_upto(radius)]
    b_all, _ = E.b_quadrature(grid.x[:stop + 1], E.DM_ORDER)
    rng = np.random.default_rng(JC.SEED)
    b = np.sort(rng.choice(b_all[(b_all > 1e-4) & (b_all < radius * 0.999)], size=n_b, replace=False))
    mass = np.array([10**logm])
    prof = A.published_profile(1, mass, 0.0, grid.x)[0]
    sig_engine = np.zeros(len(b))
    for a, c in grid.segments[:n_cut]:
        sig_engine += E.projection_operator(grid.x[a:c + 1], b) @ prof[a:c + 1]
    lin = lambda r: np.interp(r, grid.x, prof)                                     # noqa: E731
    smooth = lambda r: A.published_profile(1, mass, 0.0, np.atleast_1d(r))[0, 0]   # noqa: E731
    sig_lin, sig_smooth = np.zeros(len(b)), np.zeros(len(b))
    for i, bb in enumerate(b):
        L = np.sqrt(radius**2 - bb**2)
        brk = np.sqrt(np.maximum(grid.x[(grid.x > bb) & (grid.x < radius)]**2 - bb**2, 0))
        brk = brk[(brk > 0) & (brk < L)]
        pts = brk if len(brk) < 400 else brk[:: int(np.ceil(len(brk) / 400))]
        sig_lin[i] = 2 * quad(lambda l: lin(np.sqrt(bb**2 + l**2)), 0, L, points=pts, limit=2000, epsabs=0,
                              epsrel=1e-11)[0]
        sig_smooth[i] = 2 * quad(lambda l: smooth(np.sqrt(bb**2 + l**2)), 0, L, limit=2000, epsabs=0, epsrel=1e-11)[0]
    # unit chain: engine DM = column_scale * Sigma (f_pdf / engine.column_terms) vs the formula in SI
    ing = E.Lightcone().ingredients
    da, R, y_amp, tau_amp = ing.geometry(z, mass)
    column_scale = tau_amp * (da / R)**2 / (E.DM_FACTOR * (1 + z))
    P = A.C.P
    rho = np.exp(ing.rho_spline(z)) * A.H**2                                     # Msun / Mpc^3 (physical)
    n_unit_m3 = A.FB * rho / P.MU_XGP_MSUN / P.MPC_M**3                          # electrons per m^3 for E = 1
    dm_formula = 2 * R[0] * P.MPC_M / (1 + z) * (sig_engine / 2) * n_unit_m3 / A.M2_PER_PC_CM3
    dm_engine = column_scale[0] * sig_engine
    return {"logm200c": logm, "z": z, "sphere_radius_R200c": radius, "n_b": int(n_b),
            "engine_vs_chord_integral_same_nodes_max_rel": rel(sig_engine, sig_lin),
            "engine_vs_chord_integral_smooth_B16_max_rel": rel(sig_engine, sig_smooth),
            "engine_DM_vs_HalfDome_formula_SI_max_rel": rel(dm_engine, dm_formula),
            "column_scale_pc_cm3_per_unit_Sigma": float(column_scale[0]),
            "f_b_engine_(lightcone_analytic.FB)": float(A.FB), "f_b_study": JC.F_B,
            "n_basis_segments_inside_cut": n_cut}


# ============================================================================ halo crossings per sightline
def crossings():
    """Expected number of painted haloes whose 3D sphere of radius X R200c a sightline to z_s = 1 crosses (the engine's
    'lam': sum over lightcone cells z <= 1 of N (1 - cos(X R200c / d_A)) / 2), per engine bin, X = 1, 2, 4; the
    fraction of sightlines crossing none is exp(-lambda) (Poisson)."""
    sky = FC.make_sky()
    ing = sky.ingredients
    out = {}
    for X in JC.CUTS:
        lam = np.zeros(16)
        for j in np.flatnonzero((sky.z <= 1.0) & (sky.counts.sum(axis=1) > 0)):
            idx = np.flatnonzero(sky.counts[j] > 0)
            da, R, _, _ = ing.geometry(sky.z[j], sky.mass[idx])
            ang = np.minimum(np.pi, X * R / da)
            lam += np.bincount(sky.mass_bin[idx], weights=sky.counts[j, idx] * (1 - np.cos(ang)) / 2, minlength=16)
        out[f"{X:g}R"] = {"lambda_total": float(lam.sum()), "p_no_crossing": float(np.exp(-lam.sum())),
                          "lambda_per_engine_bin": lam.tolist()}
    return out


# ============================================================================ entries
def bin_rows(pr, b):
    return np.flatnonzero(pr["mass_bin"] == b)


def halo_entries(pr, ek, treat, cut, config, sub82=False):
    """Per-halo DM-PDF entries of the bins of ``config`` (J order restricted to the configuration)."""
    out = []
    for b in JC.SUPPORT_BINS[config]:
        if sub82 and b == 1:
            rows = pr["sub82_rows"]
            F = pr[f"sub82_{ek}"]
            w = pr["weight"][rows]
        else:
            rows = bin_rows(pr, b)
            F = pr[f"{ek}_{treat}"][rows]
            w = pr["weight"][rows]
        out.append((JC.pad75(JC.cut_profile(F, cut)), w / w.sum()))
    return out


def det_entries(mom, config):
    idx = [JC.J_BINS.index(b) for b in JC.SUPPORT_BINS[config]]
    return [(JC.pad75(mom["E"][c]), 1.0) for c in idx]


def subsample(pr, ek, cut):
    """Test F / H Mecke recipe: <= MECKE_SUBSAMPLE haloes per bin (default_rng(f_common.SEED + 5)), weights rescaled."""
    rng = np.random.default_rng(FC.SEED + 5)
    out = []
    for b in JC.J_BINS:
        rows = bin_rows(pr, b)
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        pick = rng.choice(len(rows), size=min(FC.MECKE_SUBSAMPLE, len(rows)), replace=False)
        F = JC.cut_profile(pr[f"{ek}_T1"][rows[pick]], cut)
        out.append((JC.pad75(F), w[pick] * (w.sum() / w[pick].sum())))
    return out


# ============================================================================ main
def main():
    timer = JC.Timer()
    pr = JC.load_npz(JC.WORK / "profiles_j.npz")
    out, summary, parity = {}, {"evaluations": {}, "pdf": {}, "per_bin": {}}, {}
    parity["deposit_identity_max_abs"] = JP.GP.deposit_identity_check()
    parity["chord"] = chord_check()
    summary["crossings"] = crossings()
    JC.log("chord check", parity["chord"])
    evs = eval_list()
    moms = {}
    # ---------------------------------------------------------------- moment observables, one basis at a time
    for which in ("JA", "JB"):
        B = FB.Basis(which)
        for tag, basis, config, sel, cut, treat, sub in evs:
            if basis != which:
                continue
            mom = JO.moments(pr, sel, cut, treat, config, sub)
            moms[tag] = mom
            res = JO.evaluate(B, mom, config, cut)
            for kind, o in res.items():
                for k, v in o.items():
                    out[f"{tag}|{kind}__{k}"] = np.asarray(v)
                summary["evaluations"][f"{tag}|{kind}"] = JO.scalars(o)
            if treat == "T1" and not sub and config in ("C14", "C10"):
                pb = JO.per_bin(B, mom)
                fpb = JO.fallback_per_bin(B)
                for k, v in pb.items():
                    out[f"{tag}|perbin__{k}"] = v
                for k, v in fpb.items():
                    out[f"{tag}|fbperbin__{k}"] = v
            # Y(<4R) and N_e(<4R) budgets of the calibrated bins (dimensionless)
            bud = B.budgets(mom["P"], mom["E"])
            out[f"{tag}|Y4"], out[f"{tag}|Ne4"] = bud["Y"], bud["Ne"]
        del B
        JC.mem(f"after basis {which}")
    # parity: C10|D0|T1 is Test H's A2-ii configuration (full mode only: the smoke configurations differ)
    htr = {} if JC.SMOKE else JC.load_npz(JC.H_RESULTS / "truth.npz", keys=lambda k: k.startswith(("truth_a2ii__", "truthmean_a2ii__",
                                                                                "truthcal_a2ii__")))
    worst = {}
    for kind in ("truth", "truthmean", "truthcal"):
        for k in JO.OBS_KEYS:
            ref = htr.get(f"{kind}_a2ii__{k}")
            if ref is not None:
                worst[f"{kind}__{k}"] = rel(out[f"C10|D0|T1|{kind}__{k}"], ref)
    parity["H_moments_max_rel"] = max(worst.values()) if worst else None
    parity["H_moments_by_key"] = worst
    JC.log("parity with Test H (moments)", parity["H_moments_max_rel"])
    JC.mem("before pdf passes")
    # ---------------------------------------------------------------- DM PDF passes
    passes = {}
    for tag, basis, config, sel, cut, treat, sub in evs:
        if config == "C0":
            continue
        ek = JO.SEL_KEY[sel]
        which = basis if config == "C14" else basis + "10"
        key = (which, cut, config)
        passes.setdefault(key, [])
        passes[key].append((f"{tag}|truth", halo_entries(pr, ek, treat, cut, config, sub)))
        if treat == "T1" and not sub and tag.split("|")[1] in JC.DEFS:
            passes[key].append((f"{tag}|truthmean", det_entries(moms[tag], config)))
    for cut in JC.CUTS:
        passes[("JA10", cut, "C0")] = [(f"C0|cut{cut:g}|-|truth", [])]
    keep_density = {f"C14|{d}|T1|truth" for d in JC.DEFS} | {f"C10|{d}|T1|truth" for d in JC.DEFS} | \
        {f"C14|{d}|T1|truthmean" for d in JC.DEFS} | {f"C0|cut{c:g}|-|truth" for c in JC.CUTS}
    full_refs = {}
    for (which, cut, config), specs in passes.items():
        labs = {lab for lab, _ in specs}
        res, tim = JP.run_specs(specs, which, cut, config, keep_density=bool(labs & keep_density),
                                keep_full={lab for lab in labs if lab.startswith("C14|") and "|T1|" in lab
                                           and lab.split("|")[1] in JC.DEFS}, label=f"pdf {config} {which} {cut:g}")
        for lab, s in res.items():
            if "_full" in s:
                full_refs[lab] = s.pop("_full")
            dens = s.pop("density", None)
            if dens is not None and lab in keep_density:
                out[f"pdf|{lab}|density"] = dens
            out[f"pdf|{lab}|quantiles"] = np.asarray(s["quantiles"], float)
            out[f"pdf|{lab}|mean"] = s["mean"]
            out[f"pdf|{lab}|variance"] = s["variance"]
            summary["pdf"][lab] = s
        summary.setdefault("pdf_timing", {})[f"{config}|{which}|{cut:g}"] = tim
        del res
        JC.mem(f"after pdf pass {config} {which} {cut:g}")
    # PDF vs moments (the PDF mean / variance must equal the moment forms; differences = grid / overflow effects)
    pvm = {}
    for lab, s in summary["pdf"].items():
        tag, kind = lab.rsplit("|", 1)
        ev = summary["evaluations"].get(f"{tag}|{kind}")
        if ev is not None:
            pvm[lab] = {"mean": s["mean"] / ev["dm_mean"] - 1, "variance": s["variance"] / ev["dm_var"] - 1}
    summary["pdf_vs_moments"] = pvm
    hts = None if JC.SMOKE else JC.read_json(JC.H_RESULTS / "truth_summary.json")
    pq = {}
    for kind in (() if JC.SMOKE else ("truth", "truthmean")):
        q = np.asarray(summary["pdf"][f"C10|D0|T1|{kind}"]["quantiles"], float)
        ref = np.asarray(hts["pdf"][f"{kind}_a2ii"]["quantiles"], float)
        pq[kind] = {"J_C10_D0": q, "H": ref, "max_rel": rel(q, ref),
                    "mean_rel": rel(summary["pdf"][f"C10|D0|T1|{kind}"]["mean"], hts["pdf"][f"{kind}_a2ii"]["mean"])}
    parity["H_pdf"] = pq
    parity["H_pdf_max_rel"] = max(max(v["max_rel"], v["mean_rel"]) for v in pq.values()) if pq else None
    JC.log("parity with Test H (DM PDF)", parity["H_pdf_max_rel"])
    # ---------------------------------------------------------------- Mecke quantile Jacobians (C14 production)
    mecke_sum = {}
    for d, v in JC.DEFS.items():
        ek = JO.SEL_KEY[v["sel"]]
        tag = f"C14|{d}|T1"
        items = [(f"{tag}|truth_subsample", full_refs[f"{tag}|truth"], subsample(pr, ek, v["cut"])),
                 (f"{tag}|truthmean", full_refs[f"{tag}|truthmean"], det_entries(moms[tag], "C14"))]
        m, tim = JP.run_mecke(items, v["basis"], v["cut"], "C14", label=f"mecke {d}")
        for lab, (J, Jfb) in m.items():
            out[f"Jq|{lab}"] = J
            out[f"Jqfb|{lab}"] = Jfb
            Q = np.asarray(full_refs[lab.replace("_subsample", "")]["quantiles"], float)
            mecke_sum[lab] = ((J.sum(axis=(1, 2)) + Jfb.sum(axis=(1, 2))) / Q - 1).tolist()
        summary.setdefault("mecke_timing", {})[d] = tim
        del m
    summary["mecke_sum_rule"] = mecke_sum
    del full_refs
    # ---------------------------------------------------------------- headline comparisons
    summary["comparisons"] = comparisons(summary)
    if JC.SMOKE:
        parity["note"] = "smoke: the C10 smoke configuration is not Test H's; no Test H parity"
        parity["passed"] = bool(parity["deposit_identity_max_abs"] == 0.0)
    else:
        parity["passed"] = bool(parity["deposit_identity_max_abs"] == 0.0 and parity["H_moments_max_rel"] < 1e-6
                                and parity["H_pdf_max_rel"] < 1e-6)
    summary["seconds"] = timer.mark("total")
    parity["seconds"] = summary["seconds"]
    np.savez_compressed(JC.RESULTS / "truth.npz", **out)
    JC.write_json(JC.RESULTS / "truth_summary.json", summary)
    JC.write_json(JC.RESULTS / "parity.json", parity)
    JC.log("truth stage done; parity passed:", parity["passed"])
    return parity


DM_STATS = ("dm_mean", "dm_var", "q16", "q50", "q84", "q975", "q999", "dmcltot_100", "dmcltot_300", "dmcltot_1000",
            "dmcltot_3000", "ydmtot_100", "ydmtot_300", "ydmtot_1000", "ydmtot_3000")
KSZ_STATS = tuple(f"{k}_{e}" for k in ("ksz1h", "ksz2h", "ksztot") for e in (100, 300, 1000, 3000))


def stat(summary, tag, name):
    """A statistic of an evaluation: moment observables from the scatter-inclusive truth, quantiles from its PDF."""
    if name.startswith("q"):
        s = summary["pdf"].get(f"{tag}|truth")
        if s is None:
            return None
        return float(s["quantiles"][JC.Q_NAMES.index(name)])
    return summary["evaluations"][f"{tag}|truth"].get(name)


def comparisons(summary):
    """Ratios of the headline statistics (all from this stage's outputs)."""
    out = {"definitions_vs_D0_C14": {}, "measured_vs_fallback_C14_over_C10": {}, "treatments_vs_T1": {},
           "subset82_vs_T1": {}, "kSZ_D3_over_D0": {}, "first_order_check_C10_hb_over_all": {},
           "C14_over_C0_published": {}, "D3_JA_vs_JB_shape": {}}
    for d in JC.DEFS:
        out["definitions_vs_D0_C14"][d] = {k: _ratio(stat(summary, f"C14|{d}|T1", k), stat(summary, "C14|D0|T1", k))
                                           for k in DM_STATS}
        out["measured_vs_fallback_C14_over_C10"][d] = {k: _ratio(stat(summary, f"C14|{d}|T1", k),
                                                                 stat(summary, f"C10|{d}|T1", k)) for k in DM_STATS}
        for t in ("T2", "T3"):
            out["treatments_vs_T1"][f"{d}|{t}"] = {k: _ratio(stat(summary, f"C14|{d}|{t}", k),
                                                             stat(summary, f"C14|{d}|T1", k)) for k in DM_STATS}
        if f"C14|{d}|S82|truth" in summary["evaluations"]:
            out["subset82_vs_T1"][d] = {k: _ratio(stat(summary, f"C14|{d}|S82", k), stat(summary, f"C14|{d}|T1", k))
                                        for k in DM_STATS}
        cut = JC.DEFS[d]["cut"]
        out["C14_over_C0_published"][d] = {k: _ratio(stat(summary, f"C14|{d}|T1", k),
                                                     stat(summary, f"C0|cut{cut:g}|-", k)) for k in DM_STATS}
    for cfg in ("C14", "C10"):
        out["kSZ_D3_over_D0"][cfg] = {k: _ratio(stat(summary, f"{cfg}|D3|T1", k), stat(summary, f"{cfg}|D0|T1", k))
                                      for k in KSZ_STATS}
    out["first_order_check_C10_hb_over_all"] = {k: _ratio(stat(summary, "C10|D3|T1", k), stat(summary, "C10|ALL|T1", k))
                                                for k in KSZ_STATS + DM_STATS}
    out["D3_JA_vs_JB_shape"] = {k: _ratio(stat(summary, "C14|D3inJA|T1", k), stat(summary, "C14|D3|T1", k))
                                for k in KSZ_STATS + DM_STATS}
    return out


def _ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return float(a / b)
