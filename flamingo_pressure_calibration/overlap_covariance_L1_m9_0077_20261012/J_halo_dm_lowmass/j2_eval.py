"""Test J2 stage 'eval': every fitted painting of J2a / J2b evaluated in Test J1's configuration (fine 75-shell basis,
calibrated support C14 = 14 bins 12.7-15.388, J1's fallback haloes cut like each definition) against J1's truth of
each definition.

Configuration (environment set by run_test_j2.py before import): J_MODE, J_RESULTS = <J2 results>/j1base (the J1
stages prepare / basis / fallback re-run there, bases in the node-local scratch J_SCRATCH), J_WORKERS.  Test J1's
modules (j_common, j_obs, j_pdf, j_fallback) and Test F/G's modules are imported read-only.

Paintings (items).  For every item and calibrated bin c the painted electron profile is the exact shell average of
the fitted form on the 75 fine shells, M[c, s], hard-cut in 3D at R_cut = the definition's boundary (D1: 1, D2: 2,
D0 / D3: 4 R200c):
  a|D|B16|obj|Rf     B16 per bin (j2_fits 'fits'; S400 / strict masks), objective a, d, L (L*: (a) where no L fit), dI
  rc|D|B16|obj       as above with the RC fits in the low-mass bins (resolved shape + core integral)
  l|D|model|dset|Rf  Lee22-ladder models (j2_fits 'ladder'): run 1 painted the selected model and the best-BIC outer
                     option of PB, PL1, PL2, PL3, BPLall, BPLall_L22, the free-gamma laws and the concentration laws
                     (painted at the bin-mean log M and mean ln c, as Test I); run 2 adds every other mass-cell model
                     with a floor fit at the definition's ladder radii (role 'complete_ladder', batch 2)
  a|D0|B16|dIT|Rf4, lm|cfg|B12a+B16dIT   run 2: Test I's own published fits in 13.5-15.5 (I_fitting_a2ii/results/
                     fits35.json E|d|4, P|a|4) with J2's dI / P|a fits in 12.7-13.5 (batch 2)
  ref|D|stack        the definition's own stacked mean (T1) profiles: det = the truth mean (parity gate), S1 = the
                     scatter-model floor of the definition
  lm|cfg|B12a+B16obj  Test I's recipe for the newly calibrated bins (pressure B12 (a) and electrons B16 (obj) at
                     R_fit = 4, D0 target), painted in C14 (all 14 bins fitted) and C10 (bins 12.7-13.5 painted with
                     the published B12/B16 fallback, J1's configuration of Tests F/H); obj = dI: Test I's weight
                     formula on J1's C14 maps, refitted in all 14 bins; obj = a; obj = dIT (run 2, above)
Batches: the run-2 items are deposited in their own DM-PDF groups after all run-1 groups (run-1 items unchanged).
Predictions: det (one profile per bin) and S1 (lognormal amplitude scatter of the whole profile per bin,
  <F_s G_t> = M_FG F_s G_t with M_EE = <N^2>/<N>^2, N = the haloes' electron count inside the definition's boundary,
  M_PP = <Y^2>/<Y>^2, Y = Y(<4R), M_PE = <Y N>/(<Y><N>); inclusion-weighted over the bin's haloes, T1 profiles).
The pressure paired with an electron model is the truth's mean A2-ii pressure (field-separated scoring, Test I);
lm items paint both fields.
Observables (f_basis, kernels as J1): mean DM, DM variance, DM-field C_ell (1h, 2h, total), y x DM, kSZ (1h, 2h), tSZ;
DM quantiles 16/50/84/97.5/99.9% from the compound-Poisson DM PDF (j_pdf / f_pdf, z_s = 1, observer frame).
Errors: O / O_truth - 1 against J1's scatter-inclusive truth 'C14|D|T1|truth' (and the profile-only error of det
against 'C14|D|T1|truthmean').
Outputs (J2 results): evaluate.npz (painted shells, PDF CDFs on a log grid, scatter tables), evaluate_items.json,
eval_parity.json.
"""
from __future__ import annotations

import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import j2_common as C  # noqa: E402
import j_common as JC  # noqa: E402
import j_fallback as JF  # noqa: E402
import j_pdf as JP  # noqa: E402

FC, FB = JC.FC, JC.FB
FAM = FC.FAM
import g_physics as PH  # noqa: E402
import g_selection as GS  # noqa: E402

assert list(JC.J_BINS) == list(C.J_BINS), (JC.J_BINS, C.J_BINS)
assert tuple(FC.SEL_ELL) == C.SEL_ELL and tuple(FC.Q_NAMES) == C.Q_NAMES
N_J = JC.N_J
LOW_J = [JC.J_BINS.index(b) for b in JC.LOW_BINS]
C10_J = [JC.J_BINS.index(b) for b in JC.SUPPORT_BINS["C10"]]
DM_GRID = np.logspace(-1, np.log10(2000.0), 400)          # pc cm^-3: CDFs of every PDF are stored on this grid
GH_T, GH_W = np.polynomial.hermite.hermgauss(9)


# ============================================================================ scatter tables
def s1_table(pr, ekey, R):
    """[N_J, 3]: ln<Y^2>/<Y>^2, ln<N^2>/<N>^2, <Y N>/(<Y><N>) with Y = Y(<4R) (A2-ii pressure), N = electrons inside
    the 3D radius R (selection ekey), per-halo T1 profiles, inclusion weights."""
    V = JC.VOL
    ins4, insR = JC.inside(JC.X200, 4.0), JC.inside(JC.X200, R)
    out = np.zeros((N_J, 3))
    for c, b in enumerate(JC.J_BINS):
        r = np.flatnonzero(pr["mass_bin"] == b)
        w = pr["weight"][r] / pr["weight"][r].sum()
        Y = pr["P_a2ii_T1"][r][:, ins4] @ V[ins4]
        N = pr[f"{ekey}_T1"][r][:, insR] @ V[insR]
        X = np.c_[Y, N]
        m = w @ X
        M = np.einsum("h,hi,hj->ij", w, X, X) / np.outer(m, m)
        out[c] = (np.log(M[0, 0]), np.log(M[1, 1]), M[0, 1])
    return out


def s1_moments(Pm, Em, s1):
    o = lambda a, b: np.einsum("cs,ct->cst", a, b)          # noqa: E731
    return dict(P=Pm, E=Em, SPP=np.exp(s1[:, 0])[:, None, None] * o(Pm, Pm),
                SEE=np.exp(s1[:, 1])[:, None, None] * o(Em, Em), SPE=s1[:, 2][:, None, None] * o(Pm, Em))


def s1_entries(F, sigma2, idx):
    out = []
    for c in idx:
        s = np.sqrt(max(float(sigma2[c]), 0.0))
        amp = np.exp(s * np.sqrt(2) * GH_T - sigma2[c] / 2)
        out.append((amp[:, None] * JC.pad75(F[c])[None, :], GH_W / np.sqrt(np.pi)))
    return out


def det_entries(F, idx):
    return [(JC.pad75(F[c]), 1.0) for c in idx]


def cut75(M, R):
    M = np.array(JC.pad75(M), float, copy=True)
    M[..., ~(JC.FINE[1:] <= R + 1e-9)] = 0.0
    return M


# ============================================================================ paintings
def b_shells(family, recs):
    fam = FAM.get(family)
    return np.array([FAM.shell_model_truncated(fam, np.asarray(r["parameters"], float), JC.FINE, 4.0) for r in recs])


def build_items(pr, fits, lad):
    items = []

    def add(label, block, D, M4, k, pdf=True, field="E", Pm=None, config="C14", **meta):
        R = C.DEFS[D]["R"] if D in C.DEFS else 4.0
        items.append(dict(label=label, block=block, D=D, M4=np.asarray(M4, float), k=int(k), pdf=pdf, field=field,
                          Pm=Pm, config=config, R_cut=R, **meta))

    F = fits["fits"]
    # ---- J2a: B16 per bin (S400 / strict masks)
    for D, spec in C.DEFS.items():
        for rf in spec["rfit"]:
            objs = ["a", "L"] + (["d"] if rf <= spec["R"] + 1e-9 else []) + (["dI"] if D == "D0" and rf >= 3.0 else [])
            for obj in objs:
                recs, star = [], []
                for b in JC.J_BINS:
                    r = F.get(f"{D}|{obj}|{rf:g}|{b}")
                    if r is None or r.get("parameters") is None:
                        r = F.get(f"{D}|a|{rf:g}|{b}")
                        star.append(C.bin_label(b))
                    recs.append(r)
                if any(r is None or r.get("parameters") is None for r in recs):
                    C.log("skip (missing fit)", D, obj, rf)
                    continue
                add(f"a|{D}|B16|{obj}|Rf{rf:g}", "J2a", D, b_shells("b16", recs), 3 * N_J, model="B16",
                    objective=obj, R_fit=rf, rule="S400", a_substituted_bins=star,
                    params=[r["parameters"] for r in recs])
    # ---- J2a: RC variant (low bins)
    for D, spec in C.DEFS.items():
        rf = spec["R"]
        for obj in ("a", "d"):
            recs, sub = [], []
            for b in JC.J_BINS:
                r = None
                if b in JC.LOW_BINS:
                    r = fits["rc"].get(f"{D}|{obj}|{rf:g}|{b}")
                    if r is None or r.get("parameters") is None:
                        sub.append(C.bin_label(b))
                        r = None
                if r is None:
                    r = F.get(f"{D}|{obj}|{rf:g}|{b}")
                recs.append(r)
            if any(r is None or r.get("parameters") is None for r in recs):
                continue
            add(f"rc|{D}|B16|{obj}|Rf{rf:g}", "J2a", D, b_shells("b16", recs), 3 * N_J, model="B16", objective=obj,
                R_fit=rf, rule="RC", rc_fallback_to_S400_bins=sub, params=[r["parameters"] for r in recs])
    # ---- reference: the stacks themselves
    for D, spec in C.DEFS.items():
        ek = "E_a2ii" if spec["sel"] == "a2ii" else "E_hb"
        add(f"ref|{D}|stack", "ref", D, JC.pad75(pr[f"mean_{ek}_T1"][JC.J_BINS]), 0, model="stack", objective="-",
            R_fit=None, rule="T1")
    # ---- low-mass calibration with Test I's recipe (D0 target, R_fit 4)
    precs = [F.get(f"P|a|4|{b}") for b in JC.J_BINS]
    if all(r is not None for r in precs):
        Pm = b_shells("b12", precs)
        for obj in ("dI", "a"):
            erecs = [F.get(f"D0|{obj}|4|{b}") for b in JC.J_BINS]
            if any(r is None for r in erecs):
                continue
            Em = b_shells("b16", erecs)
            for cfg in ("C14", "C10") if not JC.SMOKE else ("C14",):
                add(f"lm|{cfg}|B12a+B16{obj}", "LM", "D0", Em, 6 * N_J, field="PE", Pm=Pm, config=cfg,
                    model=f"B12(a)+B16({obj})", objective=obj, R_fit=4.0, rule="S400")
    # ---- J2b: ladder models
    if lad:
        xF, wqF = PH.quad_nodes(JC.FINE)
        lh = PH.LinearHalo()
        prep = np.load(C.G_PREPARE)
        ctx = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"])}
        rows = [np.flatnonzero(pr["mass_bin"] == b) for b in JC.J_BINS]
        T2h = np.array([PH.two_halo_template(lh, pr["m200c"][r], pr["r200c"][r], pr["weight"][r], xF) for r in rows])
        conc = lad["conc"]
        w = [pr["weight"][r] / pr["weight"][r].sum() for r in rows]
        logm = np.array([ww @ pr["logm200c"][r] for ww, r in zip(w, rows)])
        lnc = np.array([ww @ np.log(conc[pr["rows"][r]]) for ww, r in zip(w, rows)])
        view = types.SimpleNamespace(bin=np.arange(N_J), logm=logm, lnc=lnc, n=N_J)

        def lshells(dset, name, theta, frozen=None):
            inner, outer = name.split("+")
            spec = [s for s in GS.model_specs("E", dset) if s[0] == inner][0]
            model = GS.build_model("E", spec, outer, N_J, frozen=frozen)
            Fin, Fout = model.evaluate(np.asarray(theta, float), view, xF, wqF, JC.N_FINE, T2h, ctx, jac=False,
                                       parts=True)
            M = Fin + Fout
            M[:, ~(JC.FINE[1:] <= 4.0 + 1e-9)] = 0.0
            return M

        def bic(r):
            f = r["fits"]["floor"]
            return f["chi2"] + r["k"] * np.log(f["n_data"])

        extra = []
        for D, res in lad["results"].items():
            spec = C.DEFS[D]
            by = {(r["dset"], float(r["R_fit"]), r["model"]): r for r in res}
            for rf in spec["lad_rfit"]:
                picks = {}
                ch = lad["chosen"][D].get(f"E|mass|{rf:g}")
                if ch is not None:
                    picks["selected"] = by.get(("mass", rf, ch["model"]))
                chc = lad["chosen"][D].get(f"E|massc|{rf:g}")
                if chc is not None and not chc["model"].startswith("PBcell"):
                    picks["selected_c"] = by.get(("massc", rf, chc["model"]))

                def best(dset, inners):
                    cands = [r for (ds, r_, m), r in by.items() if ds == dset and abs(r_ - rf) < 1e-9
                             and m.split("+")[0] in inners and "floor" in r["fits"]]
                    return min(cands, key=bic) if cands else None

                for inner in ("PB", "PL1", "PL2", "PL3", "BPLall", "BPLall_L22"):
                    picks[inner] = best("mass", (inner,))
                picks["gamma"] = best("mass", ("PL3g", "PL3gM"))
                picks["cbest"] = best("massc", ("PL1cA", "PL3cA", "PL3c3", "L22best", "L22best_L22"))
                seen = set()
                for role, r in picks.items():
                    if r is None or "floor" not in r["fits"]:
                        continue
                    key = (r["dset"], r["model"])
                    if key in seen:
                        continue
                    seen.add(key)
                    M = lshells(r["dset"], r["model"], r["fits"]["floor"]["theta"], r.get("frozen_shape"))
                    roles = [ro for ro, rr in picks.items() if rr is r]
                    add(f"l|{D}|{r['model']}|{r['dset']}|Rf{rf:g}", "J2b", D, M, int(r["k"]), model=r["model"],
                        dset=r["dset"], roles=roles, objective="floor", R_fit=rf, rule="S400",
                        selected=("selected" in roles) or ("selected_c" in roles))
                # run 2: every other mass-cell model with a floor fit (painted after all run-1 items, see below)
                rest = [r for (ds, r_, m), r in by.items() if ds == "mass" and abs(r_ - rf) < 1e-9
                        and "floor" in r["fits"] and ("mass", r["model"]) not in seen]
                extra += [(D, rf, r) for r in sorted(rest, key=lambda r: (int(r["k"]), r["model"]))]
        # ---- run 2 (2026-10-15, verifier 7c): the complete mass-cell ladder.  Appended after every run-1 item and
        # deposited in their own DM-PDF groups (batch 2), so that every run-1 painting is computed exactly as in run 1.
        for D, rf, r in extra:
            M = lshells(r["dset"], r["model"], r["fits"]["floor"]["theta"], r.get("frozen_shape"))
            add(f"l|{D}|{r['model']}|mass|Rf{rf:g}", "J2b", D, M, int(r["k"]), model=r["model"], dset="mass",
                roles=["complete_ladder"], objective="floor", R_fit=rf, rule="S400", selected=False, batch=2)
    # ---- run 2 (verifier 7b): Test I's own published fits in 13.5-15.5 (E|d|4 and P|a|4 of I_fitting_a2ii/results/
    # fits35.json, weights from Test H's C10 importance maps) + J2's new fits in 12.7-13.5 (D0|dI|4 and P|a|4: Test I's
    # weight formula on J1's C14 maps, S400 masks): the kSZ recipe unchanged where Test I calibrated it ('dIT')
    ifp = C.I_RESULTS / "fits35.json"
    if ifp.exists() and all(F.get(f"P|a|4|{b}") is not None for b in JC.LOW_BINS) \
            and all(F.get(f"D0|dI|4|{b}") is not None for b in JC.LOW_BINS):
        TI = C.read_json(ifp)["fits"]
        erecs, prc, src = [], [], []
        for b in JC.J_BINS:
            if b >= 5:                                    # Test I's index c = engine bin - 5 (its 10 bins 13.5-15.5)
                erecs.append(TI[f"E|d|4|{b - 5}"])
                prc.append(TI[f"P|a|4|{b - 5}"])
                src.append("Test I fits35.json E|d|4 / P|a|4")
            else:
                erecs.append(F[f"D0|dI|4|{b}"])
                prc.append(F[f"P|a|4|{b}"])
                src.append("J2 fits.json D0|dI|4 / P|a|4")
        Em, PmT = b_shells("b16", erecs), b_shells("b12", prc)
        hyb = dict(objective="dIT", R_fit=4.0, rule="S400", source_per_bin=src,
                   params=[r["parameters"] for r in erecs], p_params=[r["parameters"] for r in prc], batch=2)
        add("a|D0|B16|dIT|Rf4", "J2a", "D0", Em, 3 * N_J, model="B16", a_substituted_bins=[], **hyb)
        for cfg in ("C14", "C10") if not JC.SMOKE else ("C14",):
            add(f"lm|{cfg}|B12a+B16dIT", "LM", "D0", Em, 6 * N_J, field="PE", Pm=PmT, config=cfg,
                model="B12(a)+B16(dIT)", **hyb)
    return items


# ============================================================================ main
def truths(D):
    """J1's scatter-inclusive truth and truth mean of definition D (C14, T1): moment scalars + DM quantiles."""
    ts = C.read_json(C.J1_TRUTH)
    out = {}
    for kind in ("truth", "truthmean"):
        o = dict(ts["evaluations"][f"C14|{D}|T1|{kind}"])
        q = ts["pdf"][f"C14|{D}|T1|{kind}"]["quantiles"]
        for p, qn in enumerate(C.Q_NAMES):
            o[qn] = float(q[p])
        out[kind] = o
    return out


def cdf_on_grid(density):
    d = np.asarray(density, float)
    cdf = np.cumsum(d) * JP.FP.DX
    x = np.arange(len(d)) * JP.FP.DX + 0.5 * JP.FP.DX
    return np.interp(DM_GRID, x, cdf).astype(np.float32)


def main():
    t0 = time.perf_counter()
    pr = JC.load_npz(JC.WORK / "profiles_j.npz")
    fits = C.read_json(C.J2_RESULTS / "fits.json")
    lad = {}
    summ_path = C.J2_RESULTS / "ladder_summary.json"
    if summ_path.exists():
        s = C.read_json(summ_path)
        lad = {"chosen": s["chosen"], "results": {}}
        for D in C.DEFS:
            p = C.J2_RESULTS / f"ladder_{D}.npz"
            if p.exists():
                lad["results"][D] = C.json.loads(str(np.load(p)["results_json"]))
        d_in = C.load_npz(C.J2_INPUT, keys=["concentration_200c", "rows"])
        conc = np.zeros(int(np.max(pr["rows"])) + 1)
        conc[d_in["rows"]] = d_in["concentration_200c"]
        lad["conc"] = conc
    items = build_items(pr, fits, lad)
    C.log(f"eval: {len(items)} paintings")
    # ---- scatter tables per definition (and pressure)
    s1 = {}
    for D, spec in C.DEFS.items():
        s1[D] = s1_table(pr, "E_a2ii" if spec["sel"] == "a2ii" else "E_hb", spec["R"])
    Pt = JC.pad75(pr["mean_P_a2ii_T1"][JC.J_BINS])
    # ---- DM PDF passes (forked workers; no basis loaded yet)
    groups = {}
    for it in items:
        if not it["pdf"]:
            continue
        D, cfg = it["D"], it["config"]
        R = it["R_cut"]
        M = cut75(it["M4"], R)
        idx = list(range(N_J)) if cfg == "C14" else C10_J
        which = C.DEFS[D]["basis"] + ("" if cfg == "C14" else "10")
        key = (which, R, cfg, int(it.get("batch", 1)))
        groups.setdefault(key, [])
        groups[key].append((f"{it['label']}|det", det_entries(M, idx)))
        groups[key].append((f"{it['label']}|S1", s1_entries(M, s1[D][:, 1], idx)))
    quant, cdfs, pdf_timing = {}, {}, {}
    # run-1 groups first (in run 1's order), then the batch-2 groups of run 2: every run-1 histogram is deposited in
    # the same group, with the same companions and the same fallback pass as in run 1
    for (which, R, cfg, batch), specs in sorted(groups.items(), key=lambda kv: kv[0][3]):
        res, tim = JP.run_specs(specs, which, R, cfg, keep_density=True,
                                label=f"J2 pdf {cfg} {which} {R:g} batch {batch}")
        for lab, s in res.items():
            dens = s.pop("density")
            cdfs[lab] = cdf_on_grid(dens)
            quant[lab] = {"quantiles": np.asarray(s["quantiles"], float).tolist(), "mean": float(s["mean"]),
                          "variance": float(s["variance"]), "p_first_cell": float(s.get("p_first_bin", np.nan)),
                          "mass_on_grid": float(s["mass_on_grid"])}
        pdf_timing[f"{cfg}|{which}|{R:g}" + ("" if batch == 1 else f"|batch{batch}")] = tim
        JC.mem(f"after J2 pdf {cfg} {which} {R:g} batch {batch}")
    C.log(f"pdf passes done: {len(quant)} histograms, {time.perf_counter() - t0:.0f} s")
    # ---- moment observables (one basis at a time, serial)
    T = {D: truths(D) for D in C.DEFS} if C.J1_TRUTH.exists() else {}
    obs = {}
    parity = {}
    for which in ("JA", "JB"):
        B = FB.Basis(which)
        for it in items:
            D = it["D"]
            if C.DEFS[D]["basis"] != which:
                continue
            R, cfg = it["R_cut"], it["config"]
            JF.install(B, cfg, R)
            E_ = cut75(it["M4"], R)
            P_ = Pt if it["field"] == "E" else JC.pad75(it["Pm"])
            if cfg == "C10":
                E_, P_ = E_.copy(), P_.copy()
                E_[LOW_J], P_[LOW_J] = 0.0, 0.0
            s1d = s1[D]
            for v in ("det", "S1"):
                mom = dict(P=P_, E=E_) if v == "det" else s1_moments(P_, E_, s1d)
                o = B.scalars(B.observables(**mom))
                q = quant.get(f"{it['label']}|{v}")
                if q is not None:
                    for p, qn in enumerate(C.Q_NAMES):
                        o[qn] = float(q["quantiles"][p])
                    o["pdf_mean"], o["pdf_variance"] = q["mean"], q["variance"]
                obs[f"{it['label']}|{v}"] = o
        del B
        JC.mem(f"after J2 moments {which}")
    # ---- parity: the stacks painted det = J1's truth mean (moments and DM quantiles)
    for D in C.DEFS:
        o = obs.get(f"ref|{D}|stack|det")
        if o is None or D not in T:
            continue
        ref = T[D]["truthmean"]
        rel = {k: abs(o[k] / ref[k] - 1) for k in o if k in ref and ref[k] not in (0, None) and np.isfinite(ref[k])}
        parity[D] = {"max_rel_all": max(rel.values()), "max_rel_moments": max(v for k, v in rel.items()
                                                                              if not k.startswith("q")),
                     "max_rel_quantiles": max(v for k, v in rel.items() if k.startswith("q")),
                     "worst_key": max(rel, key=rel.get)}
    parity["passed"] = bool(parity) and all(v["max_rel_moments"] < 1e-9 and v["max_rel_quantiles"] < 1e-6
                                            for k, v in parity.items() if isinstance(v, dict))
    # ---- errors and outputs
    rows = []
    store = {"x_edges": JC.X200, "fine_edges": JC.FINE, "dm_grid": DM_GRID, "j_bins": np.array(JC.J_BINS),
             "Pt": Pt}
    for D in C.DEFS:
        store[f"s1__{D}"] = s1[D]
    meta_items = []
    for n, it in enumerate(items):
        D = it["D"]
        store[f"M4__{it['label']}"] = it["M4"][:, :JC.N_X]
        if it["Pm"] is not None:
            store[f"P4__{it['label']}"] = np.asarray(it["Pm"])[:, :JC.N_X]
        meta = {k: v for k, v in it.items() if k not in ("M4", "Pm")}
        meta_items.append(meta)
        for v in ("det", "S1"):
            lab = f"{it['label']}|{v}"
            o = obs[lab]
            if lab in cdfs:
                store[f"cdf__{lab}"] = cdfs[lab]
            row = dict(label=it["label"], prediction=v, D=D, block=it["block"], config=it["config"],
                       model=it.get("model"), objective=it.get("objective"), R_fit=it.get("R_fit"),
                       R_cut=it["R_cut"], k=it["k"], rule=it.get("rule"), dset=it.get("dset"),
                       selected=it.get("selected"))
            if D in T:
                for kind, tag in (("truth", "err"), ("truthmean", "errm")):
                    ref = T[D][kind]
                    for key, val in o.items():
                        if key in ref and ref[key] not in (0, None):
                            row[f"{tag}_{key}"] = val / ref[key] - 1
            for key, val in o.items():
                row[f"obs_{key}"] = val
            rows.append(row)
    np.savez_compressed(C.J2_RESULTS / "evaluate.npz", **store)
    C.write_json(C.J2_RESULTS / "evaluate_items.json", {"items": meta_items, "truths": T, "quant": quant,
                                                        "pdf_timing": pdf_timing, "definitions": __doc__,
                                                        "seconds": time.perf_counter() - t0})
    C.write_csv(C.J2_RESULTS / "tables" / "J2_eval_long.csv", rows)
    C.write_json(C.J2_RESULTS / "eval_parity.json", parity)
    C.log(f"eval done in {time.perf_counter() - t0:.0f} s; parity {parity}")


if __name__ == "__main__":
    main()
