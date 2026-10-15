"""Test I supplement (maas run 2): completes I1 objective (L) and extends I3 to the full R_cut grid.

Why: in run 1 the Lee22-convention fits (objective L, i_g35.stage_fits35) exist only where the bin has N > p + 2 haloes
(Lee22 / Hartlap convention: the precision is undefined otherwise); the two most massive calibrated bins (15.1-15.3,
15.3-15.5) have too few haloes for the 7-29 fitted shells, so i_fine could not paint objective L (one painting needs all
ten bins) and I1(L) had no observables.  Run 1 also deleted Test H's node-local scratch with the fine A2-ii basis
(--cleanup-scratch), so this supplement first rebuilds that basis with Test H's own code.

Stages (each in its own process through run_test_i.py --stages suppbasis,supp --scratch DIR):
  suppbasis  Test H's fine A2-ii basis HA rebuilt into DIR/f_base_work/basis_HA.npz with h_basis.build and exactly
             Test H's HA specification (h_truth.specs_for_basis: shapes = log-log interpolation through the A2-ii fine
             means, coefficients = the A2-ii fine means), set HA only.  Parity gate: the in-basis moment observables of
             the A2-ii bin-mean profiles must reproduce Test H's truth_summary 'truthmean_a2ii' (max |rel| recorded in
             supp/basis_rebuild.json).
  supp       i_fine.stage_evaluate (unchanged; its outputs redirected to results/supp/) on
               I1 objective 'Lfb': the objective-(L) fits, with the bins that have no L fit (N <= p + 2) painted with
                   the objective-(a) fit of the same R_fit (flagged per bin in supp/supp_items.json); R_cut in
                   {1, 1.5, 2, 3, 4}; DM PDF det at every cut, S1 where R_cut = R_fit or 4 (as I1)
               I3 extra cuts: every ladder painting of run 1 (roles selected, PB, PL1, PL3, BPLall, gamma-free best,
                   +c best; mass bins) at the R_cut values of {1, 1.5, 2, 3, 4} not evaluated in run 1 (run 1 had
                   R_cut = R_fit and 4); duplicate paintings (same model, data set, R_fit under two roles) once;
                   DM PDF det only
               parity items: five run-1 paintings re-evaluated with their run-1 labels (compared locally with
                   results/tables/I_scores.csv: the rebuilt basis must reproduce run 1)
Outputs: results/supp/ (evaluate.npz, evaluate_items.json, tables/I_scores.csv, tables/I_errors_long.csv,
supp_items.json, basis_rebuild.json).  Nothing outside results/supp, results/work/h_supp and DIR is written.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import i_common as IC  # noqa: E402

SUPP = IC.RESULTS / "supp"
HDIR = IC.WORK / "h_supp"
FBASE = HDIR / "work" / "g" / "f_base"
H_FBASE = IC.H_RESULTS / "work" / "g" / "f_base"
PARITY_LABELS = ("I1|f35|P|a|Rf4|Rc4", "I1|f35|E|a|Rf4|Rc4", "I1|f35|E|d|Rf2|Rc2",
                 "I3|E|selected|PL3g+O0|mass|Rf4|Rc4", "I3|P|selected|BPLall+O2|mass|Rf4|Rc4")


def configure(workers, scratch):
    if not scratch:
        raise SystemExit("i_supp needs --scratch (node-local directory for the rebuilt fine basis)")
    FBASE.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    prof = FBASE / "profiles.npz"
    if not prof.exists():
        prof.symlink_to((H_FBASE / "profiles.npz").resolve())
    os.environ["H_SCRATCH"] = str(Path(scratch).resolve())
    os.environ["H_MODE"] = IC.MODE
    os.environ["H_RESULTS"] = str(HDIR)
    os.environ["H_WORKERS"] = str(workers)
    os.environ["STUDY_ROOT"] = str(IC.ROOT)
    sys.path.insert(0, str(IC.ROOT / "H_sensitivity"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================================ suppbasis
def stage_suppbasis():
    import h_common as HC
    import h_basis as HB
    import h_pdf as HP
    import h_truth as HT
    FC, FB = HC.FC, HC.FB
    path = FC.WORK / "basis_HA.npz"
    rec = {"path": str(path), "scratch": os.environ["H_SCRATCH"]}
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    cal = FC.CAL_BINS
    if path.exists():
        rec["rebuilt"] = False
    else:
        _, gridobj, _, _ = HB.grid()
        x = gridobj.x
        mP, mE = pr["meanP_a2ii"][cal], pr["meanE_a2ii"][cal]
        spec = {"HA": dict(gP=HT.shapes(mP, x, HC.X200), gE=HT.shapes(mE, x, HC.X200), coefP=HP.pad(mP),
                           coefE=HP.pad(mE))}
        rec["timing"] = HB.build(spec)
        rec["rebuilt"] = True
    rec["sha256_file"] = sha256(path)
    rec["bytes"] = path.stat().st_size
    # parity gate: in-basis moment observables of the A2-ii bin means vs Test H's truth mean
    B = FB.Basis("HA")
    o = B.scalars(B.observables(P=HP.pad(pr["meanP_a2ii"][cal]), E=HP.pad(pr["meanE_a2ii"][cal])))
    ts = IC.read_json(IC.H_RESULTS / "truth_summary.json")
    tm = ts["observables"]["truthmean_a2ii"]
    rel = {k: float(abs(o[k] / tm[k] - 1)) for k in o if k in tm and tm[k]}
    rec["parity_truthmean_a2ii_max_rel"] = max(rel.values()) if rel else None
    rec["parity_truthmean_a2ii_n"] = len(rel)
    rec["parity_truthmean_a2ii_worst"] = max(rel, key=rel.get) if rel else None
    IC.write_json(SUPP / "basis_rebuild.json", rec)
    IC.log("suppbasis done", {k: rec[k] for k in ("rebuilt", "parity_truthmean_a2ii_max_rel", "bytes")})
    if rec["parity_truthmean_a2ii_max_rel"] is None or rec["parity_truthmean_a2ii_max_rel"] > 1e-8:
        raise SystemExit("rebuilt basis HA does not reproduce Test H's truth mean (parity gate 1e-8)")


# ============================================================================ supp
def supp_items(HC, IF):
    """(items, records): the new paintings (dicts in i_fine's item format) and their provenance."""
    import json
    import h_fits as HF
    FC = HC.FC
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    lad = None
    if (IC.RESULTS / "ladder_selection.npz").exists():
        results = json.loads(str(np.load(IC.RESULTS / "ladder_selection.npz")["results_json"]))
        chosen = IC.read_json(IC.RESULTS / "ladder_selection_summary.json")["chosen"]
        lad = (results, chosen)
    orig = IF.build_items(HC, pr, lad)                  # run 1's paintings (reads results/fits35.json etc.)
    items, recs = [], {}
    # ---- I1 objective L with the objective-(a) fit in bins without an L fit
    fits = IC.read_json(IC.RESULTS / "fits35.json")["fits"]
    for field in ("P", "E"):
        for rf in IC.R_FIT_I1:
            recL = [fits.get(f"{field}|L|{rf:g}|{c}") for c in range(FC.N_CAL)]
            recA = [fits.get(f"{field}|a|{rf:g}|{c}") for c in range(FC.N_CAL)]
            use = [r if r is not None else a for r, a in zip(recL, recA)]
            if any(u is None for u in use):
                continue
            M4 = np.array([HF.model_shells(field, u["parameters"], cut=4.0) for u in use])
            fb_bins = [c for c in range(FC.N_CAL) if recL[c] is None]
            for rc in IC.R_CUT_I1:
                diag = abs(rc - rf) < 1e-9 or rc == 4.0
                lab = f"I1|f35|{field}|Lfb|Rf{rf:g}|Rc{rc:g}"
                items.append(dict(label=lab, block="I1", field=field, M4=M4, R_cut=float(rc),
                                  pdf=("det", "S1") if diag else ("det",), grid="f35",
                                  model="B12" if field == "P" else "B16", objective="Lfb", R_fit=rf, k=3 * FC.N_CAL))
                recs[lab] = {"kind": "I1_Lfb", "fallback_a_bins": fb_bins,
                             "fallback_a_bin_labels": [FC.CAL_LABELS[c] for c in fb_bins]}
    # ---- I3 at the remaining cuts (templates: run 1's R_cut = 4 items), duplicates once
    seen = {}
    for it in orig:
        if it["block"] != "I3" or it["R_cut"] != 4.0:
            continue
        key = (it["field"], it.get("dset"), it.get("model"), float(it["R_fit"]))
        if key in seen:
            recs[seen[key]]["roles"].append(it.get("role"))
            continue
        rf = float(it["R_fit"])
        cuts = [rc for rc in sorted(set(IC.R_CUT_I1) | {rf}) if abs(rc - rf) > 1e-9 and rc != 4.0]
        first = None
        for rc in cuts:
            lab = it["label"].replace("|Rc4", f"|Rc{rc:g}")
            assert lab != it["label"]
            items.append(dict(it, label=lab, R_cut=float(rc), pdf=("det",)))
            recs[lab] = {"kind": "I3_cut", "template": it["label"], "roles": [it.get("role")]}
            first = first or lab
        if first:
            seen[key] = first
            for rc in cuts[1:]:                          # the same roles list object for every cut of this painting
                recs[it["label"].replace("|Rc4", f"|Rc{rc:g}")]["roles"] = recs[first]["roles"]
    # ---- parity items (run-1 labels)
    by = {it["label"]: it for it in orig}
    for lab in PARITY_LABELS:
        if lab in by:
            items.append(dict(by[lab]))
            recs[lab] = {"kind": "parity_run1"}
    return items, recs


def stage_supp():
    import h_common as HC
    import i_fine as IF
    t0 = time.perf_counter()
    items, recs = supp_items(HC, IF)
    IC.write_json(SUPP / "supp_items.json", {"n_items": len(items), "items": recs, "definitions": __doc__})
    IC.log(f"supp: {len(items)} paintings ({sum(r['kind'] == 'I1_Lfb' for r in recs.values())} I1 Lfb, "
           f"{sum(r['kind'] == 'I3_cut' for r in recs.values())} I3 cuts, "
           f"{sum(r['kind'] == 'parity_run1' for r in recs.values())} parity)")
    # i_fine.stage_evaluate unchanged, its paintings and output paths redirected to results/supp
    IF.build_items = lambda *_a, **_k: items
    IC.RESULTS, IC.TABLES, IC.WORK = SUPP, SUPP / "tables", SUPP / "work"
    for d in (IC.TABLES, IC.WORK):
        d.mkdir(parents=True, exist_ok=True)
    IF.stage_evaluate()
    IC.log(f"supp done in {time.perf_counter() - t0:.0f} s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("suppbasis", "supp"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args(argv)
    configure(a.workers, a.scratch)
    {"suppbasis": stage_suppbasis, "supp": stage_supp}[a.stage]()


if __name__ == "__main__":
    main()
