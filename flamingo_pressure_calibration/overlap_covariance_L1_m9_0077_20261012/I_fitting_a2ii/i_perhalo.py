"""Test I run 3 (maas): the per-halo resolution rule tested on its own resolved shells.

Why: in runs 1-2 the I5 variant 'perhalo' (A2-ii stacked with the per-halo resolution rule) was fitted on the STRICT
resolved shells only: i_g35.stage_fits35 ANDs every 0.2-dex variant's resolved mask with Test G's base_mask and
stage_lvariants with strict_mask.  On those shells every halo is resolved, so the per-halo stack equals A2-ii exactly
(inputs/i0_perhalo_x200.json) and the variant reproduced the reference by construction; it did not test the rule.
This run fits the per-halo stack on its own resolved shells (26-31 of the 35 fit shells per bin, instead of the strict
18-31) and paints the per-halo stack itself.

Resolution rules (stacking core; x = r/R200c): a halo resolves shell [x_s, x_{s+1}] if x_s >= max(0.04, 4 eps/R200c),
the shell holds >= 20 gas particles and x_s >= 2 h_med/R200c (h_med = upper-median smoothing length).  'strict' uses a
shell for a bin only if every halo of the bin resolves it (all haloes enter it); 'per-halo' lets each halo enter only
the shells it resolves (a shell is used if at least one halo resolves it).

Stages (each in its own process through run_test_i.py --stages phfits,pheval --scratch DIR):
  phfits  (Test G's 35-shell configuration, as i_g35)  on inputs/i0_variants.npz 'perhalo':
            B12 (P) / B16 (E) per calibrated bin, objective (a)
              chi2 = sum_s [(ln S_s - ln m_s)/sigma_s]^2, sigma_s = max(SD_s/m_s, 5%)
              (S_s = exact shell average of the model, m_s = per-halo-rule stack, SD_s = its reliability-weighted
              halo-to-halo SD), R_fit in {2, 4}, on the OWN mask (resolved & finite & > 0) and, as a parity check,
              on the strict mask (must reproduce run 1's fits35.json variants 'perhalo|...');
            the selected Lee22-ladder model (mass cells) and PL3 with the same outer option, R_fit 2 and 4, diagonal
              weights sigma_s as above (I5 convention of stage_lvariants), on the own mask and, as parity, on the
              strict mask (must reproduce run 1's lvariants.json 'perhalo|...').
            Output: results/perhalo/phfits.json (fits, masks, parity).
  pheval  (Test H's fine configuration, as i_supp; fine A2-ii basis HA in DIR/f_base_work, left there by run 2)
            i_fine.stage_evaluate unchanged on these paintings (outputs redirected to results/perhalo/):
              I5|perhalo_own|<f>|B|Rf<rf>|Rc<rc>       B12 / B16 own-mask fits, R_cut in {R_fit, 4}
              I5|perhalo_own|<f>|<model>|Rf<rf>|Rc<rc>  Lee22 own-mask fits, R_cut in {R_fit, 4}
              I5|stack_a2ii|<f>|stack|Rc4             the A2-ii fine stack itself (= Test H's truth mean painting:
                                                      det vs truth mean must be 0; its S1 error is the exact S1
                                                      scatter-model error at the target profile)
              I5|stack_perhalo|<f>|stack|Rc4          the per-halo-rule fine stack (A2-ii mean where no halo
                                                      resolves a shell): the observable effect of the rule itself
              run-1 labels I5|perhalo|P|B|Rf4|Rc4, I5|perhalo|E|B|Rf4|Rc4, I1|f35|E|a|Rf4|Rc4 (parity with run 1)
            DM PDF (h_pdf passes): det at every cut, S1 at R_cut = 4 (as I5).
            Output: results/perhalo/ (evaluate.npz, evaluate_items.json, tables/I_scores.csv, tables/I_errors_long.csv,
            pheval_items.json, basis_check.json).
Nothing outside results/perhalo, results/work/{g35,h_perhalo} and DIR is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import i_common as IC  # noqa: E402

PH = IC.RESULTS / "perhalo"
PH_INPUT = IC.INPUTS / "i0_perhalo_x200.npz"
HDIR = IC.WORK / "h_perhalo"
H_FBASE = IC.H_RESULTS / "work" / "g" / "f_base"
R_FITS = IC.R_FIT_I5
PARITY_LABELS = ("I5|perhalo|P|B|Rf4|Rc4", "I5|perhalo|E|B|Rf4|Rc4", "I1|f35|E|a|Rf4|Rc4")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cal_rows(bins, mass_edges, cal_bins):
    """Variant bin index of every calibrated engine bin (full: 10 bins 13.5-15.5; smoke: Test F's two bins)."""
    eng = [int(np.argmin(np.abs(np.asarray(mass_edges)[:-1] - lo))) for lo in np.asarray(bins)[:, 0]]
    return [eng.index(b) for b in cal_bins]


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return float(np.max(np.abs(a[ok] / b[ok] - 1))) if ok.any() else None


# ============================================================================ phfits (35-shell, Test G configuration)
def stage_phfits(workers):
    import i_g35 as IG
    IG.configure(workers)
    import g_common as GC
    import g_physics as PHY
    import g_selection as GS
    FF = GC.FF
    PH.mkdir(parents=True, exist_ok=True)
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    var = dict(np.load(IC.VARIANTS_NPZ))
    f35 = np.asarray(GC.F_EDGES)
    v = "perhalo"
    run1_b = IC.read_json(IC.RESULTS / "fits35.json")["variants"]
    run1_l = IC.read_json(IC.RESULTS / "lvariants.json")
    out = {"masks": {}, "B": {}, "B_strict": {}, "lee": {}, "lee_strict": {}, "parity": {}, "definitions": __doc__}
    bins_all = var[f"bins_{v}"]
    sel = cal_rows(bins_all, GC.FC.MASS_EDGES, GC.CAL_BINS)        # full: 0..9
    bins = bins_all[sel]
    n = len(sel)
    assert n == GC.N_CAL
    bidx = [int(np.argmin(np.abs(GC.FC.MASS_EDGES[:-1] - bins[i, 0]))) for i in range(n)]
    masks = {}
    for field in ("P", "E"):
        mean, sd, res = (var[f"{q}_{v}_{field}"][sel] for q in ("mean", "sd", "res"))
        for i in range(n):
            own = res[i].astype(bool) & np.isfinite(mean[i]) & (mean[i] > 0)
            strict_b = own & np.asarray(pr["base_mask"][bidx[i]], bool)       # = run 1 (stage_fits35)
            strict_l = own & np.asarray(pr["strict_mask"][bidx[i]], bool)     # = run 1 (stage_lvariants)
            masks[(field, i)] = {"own": own, "strict_b": strict_b, "strict_l": strict_l}
            out["masks"][f"{field}|{i}"] = {
                "own": own.tolist(), "strict": strict_b.tolist(), "n_own": int(own.sum()),
                "n_strict": int(strict_b.sum()), "strict_b_equals_strict_l": bool(np.array_equal(strict_b, strict_l)),
                "x_inner_own": float(f35[np.argmax(own)]), "x_inner_strict": float(f35[np.argmax(strict_b)])}
            for kind, base in (("B", own), ("B_strict", strict_b)):
                bd = types.SimpleNamespace(mean=np.nan_to_num(mean[i]), sd=np.nan_to_num(sd[i]), base=base,
                                           family=FF.FIELD_FAMILY[field],
                                           mask=lambda r, base=base: base & GC.FC.shells_within(GC.F_EDGES, r))
                for rf in R_FITS:
                    nfit = int((base & GC.FC.shells_within(GC.F_EDGES, rf)).sum())
                    out[kind][f"{field}|{rf:g}|{i}"] = IG.fit_rec(FF.fit_objective(bd, "a", rf),
                                                                  {"bin_lo": bins[i, 0], "bin_hi": bins[i, 1],
                                                                   "n_fit_shells": nfit})
        IC.log("phfits B", field)
    # parity of the strict-mask B refits with run 1
    worst = 0.0
    for key, r in out["B_strict"].items():
        field, rf, i = key.split("|")
        old = run1_b.get(f"{v}|{field}|{rf}|{sel[int(i)]}")
        if r is None or old is None:
            continue
        worst = max(worst, rel(r["parameters"], old["parameters"]) or 0.0)
    out["parity"]["B_strict_vs_run1_params_max_rel"] = worst
    # ---- Lee22 ladder models (selected + PL3), diagonal weights, own and strict masks
    summ = IC.read_json(IC.RESULTS / "ladder_selection_summary.json")
    results = json.loads(str(np.load(IC.RESULTS / "ladder_selection.npz")["results_json"]))
    res_by = {(r["field"], r["dset"], float(r["R_fit"]), r["model"]): r for r in results}
    prep = GC.load_prepare()
    ctx = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"])}
    lh = PHY.LinearHalo()
    x35, _ = PHY.quad_nodes(GC.F_EDGES)
    rows = [np.flatnonzero((np.log10(pr["m200c_msun"]) >= bins[i, 0]) & (np.log10(pr["m200c_msun"]) < bins[i, 1]))
            for i in range(n)]
    T2h = np.array([PHY.two_halo_template(lh, pr["m200c_msun"][r], pr["r200c_mpc"][r], pr["weight"][r], x35)
                    for r in rows])
    worst_l = 0.0
    for field in ("P", "E"):
        mean, sd = var[f"mean_{v}_{field}"][sel], var[f"sd_{v}_{field}"][sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = np.maximum(np.where(mean > 0, sd / mean, 1.0), 0.05)
        sig = np.where(np.isfinite(sig), sig, 1.0)
        cells = types.SimpleNamespace(n=n, bin=np.arange(n), logm=var[f"logm_{v}"][sel], lnc=var[f"lnc_{v}"][sel],
                                      mean=np.where(mean > 0, mean, 1.0), T2h=T2h)
        for rf in R_FITS:
            ch = summ["chosen"].get(f"{field}|mass|{rf:g}")
            if ch is None:
                continue
            outer = ch["model"].split("+")[1]
            names = [ch["model"]] + [m for m in (f"PL3+{outer}",) if m != ch["model"]]
            for name in names:
                r0 = res_by.get((field, "mass", rf, name))
                if r0 is None or "floor" not in r0["fits"]:
                    continue
                inner = name.split("+")[0]
                spec = [s for s in GS.model_specs(field, "mass") if s[0] == inner][0]
                theta0 = np.asarray(r0["fits"]["floor"]["theta"], float)
                for kind, mk in (("lee", "own"), ("lee_strict", "strict_l")):
                    model = GS.build_model(field, spec, outer, n)
                    covs = []
                    for i in range(n):
                        m_ = masks[(field, i)][mk] & GC.x_hi_le(GC.F_EDGES, rf)
                        idx = np.flatnonzero(m_)
                        covs.append({"mask": m_, "C": np.diag(sig[i, idx]**2), "C_L22": None, "hartlap": np.nan})
                    prob = GS.Problem(model, cells, covs, "stat", 0.0, ctx, rf)
                    if prob.n_data <= model.k:
                        continue
                    sol = prob.fit([theta0])
                    if sol is None:
                        continue
                    out[kind][f"{field}|{rf:g}|{name}"] = {
                        "theta": sol.x.tolist(), "names": list(model.names),
                        "chi2": float(np.sum(prob.residuals(sol.x)**2)), "n_data": prob.n_data, "k": model.k,
                        "selected": name == ch["model"]}
                    if kind == "lee_strict":
                        old = run1_l.get(f"{v}|{field}|{rf:g}|{name}")
                        if old is not None and "theta" in old and len(old["theta"]) == len(sol.x):
                            d = np.max(np.abs(np.asarray(sol.x) - np.asarray(old["theta"])))
                            worst_l = max(worst_l, float(d))
                IC.log("phfits lee", field, rf, name)
    out["parity"]["lee_strict_vs_run1_theta_max_abs"] = worst_l
    IC.write_json(PH / "phfits.json", out)
    IC.log("phfits done", out["parity"])


# ============================================================================ pheval (fine grid, Test H configuration)
def configure_h(workers, scratch):
    fbase = HDIR / "work" / "g" / "f_base"
    fbase.mkdir(parents=True, exist_ok=True)
    PH.mkdir(parents=True, exist_ok=True)
    prof = fbase / "profiles.npz"
    if not prof.exists():
        prof.symlink_to((H_FBASE / "profiles.npz").resolve())
    if scratch:
        os.environ["H_SCRATCH"] = str(Path(scratch).resolve())
    else:                                                  # smoke: Test H's (smoke) basis, read-only symlink
        (fbase / "work").mkdir(parents=True, exist_ok=True)
        dst = fbase / "work" / "basis_HA.npz"
        if not dst.exists():
            dst.symlink_to((H_FBASE / "work" / "basis_HA.npz").resolve())
    os.environ["H_MODE"] = IC.MODE
    os.environ["H_RESULTS"] = str(HDIR)
    os.environ["H_WORKERS"] = str(workers)
    os.environ["STUDY_ROOT"] = str(IC.ROOT)
    sys.path.insert(0, str(IC.ROOT / "H_sensitivity"))


def ph_items(HC, IF):
    import h_fits as HF
    import g_physics as PHY
    import g_selection as GS
    FC = HC.FC
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    results = json.loads(str(np.load(IC.RESULTS / "ladder_selection.npz")["results_json"]))
    chosen = IC.read_json(IC.RESULTS / "ladder_selection_summary.json")["chosen"]
    orig = IF.build_items(HC, pr, (results, chosen))            # run 1's paintings (parity items)
    phf = IC.read_json(PH / "phfits.json")
    items, recs = [], {}
    # ---- B12 / B16 own-mask fits
    for field in ("P", "E"):
        for rf in R_FITS:
            rr = [phf["B"].get(f"{field}|{rf:g}|{i}") for i in range(FC.N_CAL)]
            if any(r is None for r in rr):
                IC.log("missing own-mask B fit", field, rf)
                continue
            M4 = np.array([HF.model_shells(field, r["parameters"], cut=4.0) for r in rr])
            for rc in sorted({rf, 4.0}):
                lab = f"I5|perhalo_own|{field}|B|Rf{rf:g}|Rc{rc:g}"
                items.append(dict(label=lab, block="I5", field=field, M4=M4, R_cut=float(rc),
                                  pdf=("det", "S1") if rc == 4.0 else ("det",), grid="f35", variant="perhalo_own",
                                  model="B12" if field == "P" else "B16", R_fit=rf, k=3 * FC.N_CAL, objective="a"))
                recs[lab] = {"kind": "perhalo_own_B", "n_fit_shells": [r["n_fit_shells"] for r in rr]}
    # ---- Lee22 own-mask fits: fine-shell painting exactly as i_fine.build_items (bin-mean log M, ln c)
    xF, wqF = PHY.quad_nodes(HC.FINE)
    lh = PHY.LinearHalo()
    prep = np.load(IC.ROOT / "G_physical_outer" / ("results_smoke" if IC.SMOKE else "results") / "prepare.npz")
    ctx = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"])}
    rows = [np.flatnonzero(pr["mass_bin"] == b) for b in FC.CAL_BINS]
    T2h = np.array([PHY.two_halo_template(lh, pr["m200c_msun"][r], pr["r200c_mpc"][r], pr["weight"][r], xF)
                    for r in rows])
    w = [pr["weight"][r] / pr["weight"][r].sum() for r in rows]
    logm = np.array([ww @ np.log10(pr["m200c_msun"][r]) for ww, r in zip(w, rows)])
    lnc = np.array([ww @ np.log(pr["concentration_200c"][r]) for ww, r in zip(w, rows)])
    view = types.SimpleNamespace(bin=np.arange(FC.N_CAL), logm=logm, lnc=lnc, n=FC.N_CAL)
    for key, r in phf["lee"].items():
        field, rf, name = key.split("|")
        rf = float(rf)
        inner, outer = name.split("+")
        spec = [s for s in GS.model_specs(field, "mass") if s[0] == inner][0]
        model = GS.build_model(field, spec, outer, FC.N_CAL)
        Fin, Fout = model.evaluate(np.asarray(r["theta"], float), view, xF, wqF, HC.N_FINE, T2h, ctx, jac=False,
                                   parts=True)
        M4 = Fin + Fout
        M4[:, ~(HC.FINE[1:] <= 4.0 + 1e-9)] = 0.0
        for rc in sorted({rf, 4.0}):
            lab = f"I5|perhalo_own|{field}|{name}|Rf{rf:g}|Rc{rc:g}"
            items.append(dict(label=lab, block="I5", field=field, M4=M4, R_cut=float(rc),
                              pdf=("det", "S1") if rc == 4.0 else ("det",), grid="f35", variant="perhalo_own",
                              model=name, R_fit=rf, k=int(model.k), objective="diag", selected=bool(r["selected"])))
            recs[lab] = {"kind": "perhalo_own_lee"}
    # ---- the stacks themselves on the fine grid (no fit)
    z = np.load(PH_INPUT)
    zsel = cal_rows(z["bins"], FC.MASS_EDGES, FC.CAL_BINS)
    for field in ("P", "E"):
        ms = np.asarray(z[f"mean_strict_{field}"], float)[zsel]
        mp = np.asarray(z[f"mean_perhalo_{field}"], float)[zsel]
        mstack = pr[f"mean{field}_a2ii"][FC.CAL_BINS][:, :HC.N_X]
        recs["stack_input_parity_" + field] = {"strict_x200_vs_profiles_max_rel": rel(ms, mstack)}
        filled = np.where(np.isfinite(mp), mp, ms)
        for tag, M in (("stack_a2ii", ms), ("stack_perhalo", filled)):
            M4 = np.zeros((FC.N_CAL, HC.N_FINE))
            M4[:, :HC.N_X] = M
            lab = f"I5|{tag}|{field}|stack|Rc4"
            items.append(dict(label=lab, block="I5", field=field, M4=M4, R_cut=4.0, pdf=("det", "S1"), grid="x200",
                              variant=tag, model="stack", objective="none"))
            recs[lab] = {"kind": tag, "n_shells_filled_with_a2ii": int(np.sum(~np.isfinite(mp))) if
                         tag == "stack_perhalo" else 0}
    # ---- parity items (run-1 labels)
    by = {it["label"]: it for it in orig}
    for lab in PARITY_LABELS:
        if lab in by:
            items.append(dict(by[lab]))
            recs[lab] = {"kind": "parity_run1"}
    return items, recs


def stage_pheval(workers, scratch):
    configure_h(workers, scratch)
    import h_common as HC
    import i_fine as IF
    t0 = time.perf_counter()
    path = HC.FC.WORK / "basis_HA.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing: the fine A2-ii basis must be rebuilt first (i_supp.py suppbasis)")
    chk = {"path": str(path), "resolved": str(path.resolve()), "bytes": path.stat().st_size,
           "sha256_file": sha256(path)}
    IC.write_json(PH / "basis_check.json", chk)
    items, recs = ph_items(HC, IF)
    IC.write_json(PH / "pheval_items.json", {"n_items": len(items), "items": recs, "definitions": __doc__})
    IC.log(f"pheval: {len(items)} paintings")
    IF.build_items = lambda *_a, **_k: items
    IC.RESULTS, IC.TABLES, IC.WORK = PH, PH / "tables", PH / "work"
    for d in (IC.TABLES, IC.WORK):
        d.mkdir(parents=True, exist_ok=True)
    IF.stage_evaluate()
    IC.log(f"pheval done in {time.perf_counter() - t0:.0f} s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("phfits", "pheval"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args(argv)
    if a.stage == "phfits":
        stage_phfits(a.workers)
    else:
        stage_pheval(a.workers, a.scratch)


if __name__ == "__main__":
    main()
