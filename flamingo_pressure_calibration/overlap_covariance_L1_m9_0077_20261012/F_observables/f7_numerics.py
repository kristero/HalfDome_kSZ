"""Numerics check after Test D's finding (D_literature/REPORT.md, "Numerics"): local post-processing only.

The frozen engine kernel (lightcone_analytic.linear_radial_kernel, used by kernels/engine.radial_weights)
loses precision on the innermost radial nodes for q >= 0.2 (q x << 1).  In Test F the kernel integrated:
  (i)  the basis reference shapes g_b (set T: log-log interpolation of each bin's measured mean profile,
       inner slope clipped to [-1, 0]; set V: the kernels/ P4 / E4 fits).  Every truth and model observable
       is a linear or quadratic form in shell averages with these basis kernels, so a kernel error enters
       every number through g_b -- for pressure AND electrons (kSZ, DM-field C_ell, y x DM);
  (ii) the B12 / B16 fits of the engine-check stage (shallow cores).
Flexible-family and direct-fit models never went through the kernel themselves (only their shell averages).

Step 1 (indicator, as td_fixcl.transform_error): max over q in [1e-3, 1e3] of |u_frozen / u_stable - 1| for
every reference shape, with D_literature/td_stable.stable_radial_weights (imported read-only).
Step 2 (if any indicator >= 1e-3): rebuild the production basis (set T) twice at 25 multipoles -- frozen
weights (reproducibility check against the maas values) and stable weights (E.radial_weights replaced in
memory, kernels/engine.py untouched) -- and re-evaluate every model and the truth in-basis.  DM moments and
quantiles use projected columns and q = 0 series and are not affected.
Outputs: results_local/numerics.json, results_local/basis_T_{frozen,stable}.npz (work), corrected tables.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("F_MODE", "full")
os.environ.setdefault("F_WORKERS", "2")
os.environ["F_RESULTS"] = str(HERE / "results")          # inputs: the maas results (read-only here)
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import f_common as FC  # noqa: E402
import f_kernels as K  # noqa: E402

E = FC.E
LOCAL = HERE / "results_local"
LOCAL_WORK = LOCAL / "work"
ELL_FIX = np.unique(np.r_[np.asarray(FC.SEL_ELL, float), np.geomspace(20., 2e4, 21)])


def load_stable():
    path = FC.ROOT / "D_literature" / "td_stable.py"
    spec = importlib.util.spec_from_file_location("td_stable_readonly", path)
    mod = importlib.util.module_from_spec(spec)
    flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = flag
    return mod, path


def transforms(grid, table, n_seg, q, weights):
    """u(q) [rows, q] of node tables [rows, nodes] over the first n_seg segments with a weight function."""
    u = np.zeros((table.shape[0], len(q)))
    pieces = []
    for s_i, (a, b) in enumerate(grid.segments[:n_seg]):
        w = weights(grid.x[a:b + 1], q)
        part = table[:, a:b + 1] @ w.T
        u += part
        pieces.append(part)
    return u, pieces


def indicators(STB):
    base, grid, seg_shell, n_seg_4 = K.basis_grid()
    pr = np.load(FC.PRODUCTS / "profiles.npz")
    q = np.geomspace(1e-3, 1e3, 241)
    shapes = {
        "T_pressure": (K.truth_shapes(pr["meanP_main"][FC.CAL_BINS], grid.x), len(grid.segments)),
        "T_electrons": (K.truth_shapes(pr["meanE_main"][FC.CAL_BINS], grid.x), len(grid.segments)),
        "V_pressure_P4": (E.pressure_fit(4, FC.SUPPORT).calibrated(np.array(FC.CAL_BINS), grid.x), n_seg_4),
        "V_electrons_E4": (E.electron_fit(4, FC.SUPPORT).calibrated(np.array(FC.CAL_BINS), grid.x), n_seg_4),
    }
    out = {}
    for name, (table, n_seg) in shapes.items():
        uf, pf = transforms(grid, table, n_seg, q, E.radial_weights)
        us, ps = transforms(grid, table, n_seg, q, STB.stable_radial_weights)
        with np.errstate(divide="ignore", invalid="ignore"):
            ind = np.nanmax(np.abs(uf / us - 1), axis=1)
        # where the error lives: the innermost basis shell (0 - 0.025 R200c) holds every cancelling node
        shell0 = [s for s in range(n_seg) if seg_shell[s] == 0]
        d0 = sum(pf[s] - ps[s] for s in shell0)
        rest = sum(pf[s] - ps[s] for s in range(n_seg) if seg_shell[s] != 0)
        inner_slope = None
        if name.startswith("T"):
            mean = pr["meanP_main" if "pressure" in name else "meanE_main"][FC.CAL_BINS]
            edges = FC.F_EDGES
            c0, c1 = 0.716 * edges[1], np.sqrt(edges[1] * edges[2])
            inner_slope = (np.log(mean[:, 1] / mean[:, 0]) / np.log(c1 / c0)).tolist()
        out[name] = {"indicator_per_bin": dict(zip(FC.CAL_LABELS, ind.tolist())),
                     "max_indicator": float(ind.max()),
                     "shell0_share_of_error": float(np.nanmax(np.abs(d0)) / max(np.nanmax(np.abs(d0 + rest)), 1e-300)),
                     "max_abs_error_outside_shell0_over_u": float(np.nanmax(np.abs(rest) / np.abs(us))),
                     "measured_inner_slope_before_clip": inner_slope}
    return out


def model_indicators(STB):
    """Indicator of the fitted model profiles themselves (continuous, cut at 4R), as they would enter the frozen
    engine if propagated directly.  In Test F they were NOT: only their shell averages entered the basis."""
    import csv
    import f_fit as FF
    base, grid, seg_shell, n_seg_4 = K.basis_grid()
    q = np.geomspace(1e-3, 1e3, 241)
    pars = {}
    with open(FC.TABLES / "F_fit_records.csv") as handle:
        for r in csv.DictReader(handle):
            if r.get("parameters") and (r["method"].startswith(("flex_", "direct_")) or (r["method"] == "a" and r["R_fit"] in ("2.0", "4.0"))):
                pars.setdefault((r["method"], r["R_fit"], r["field"]), {})[r["bin"]] = np.array([float(v) for v in r["parameters"].split(";")])
    out = {}
    x4 = grid.x <= 4.0 * (1 + 1e-12)
    for (method, rf, field), per_bin in sorted(pars.items()):
        if method.startswith("flex_"):
            name = method.split("_")[1]
            fam = FF.FLEX[field][name]
        else:
            fam = FF.family_obj(FF.FIELD_FAMILY[field])
        table = np.array([np.where(x4, fam.shape(grid.x, per_bin[lab]), 0.0) for lab in FC.CAL_LABELS])
        uf, _ = transforms(grid, table, n_seg_4, q, E.radial_weights)
        us, _ = transforms(grid, table, n_seg_4, q, STB.stable_radial_weights)
        with np.errstate(divide="ignore", invalid="ignore"):
            ind = np.nanmax(np.abs(uf / us - 1), axis=1)
        label = method if method != "a" else f"grid_a_Rf{float(rf):g}"
        out[f"{label}|{field}"] = {"max_indicator": float(ind.max()),
                                  "bins_ge_1e-3": [lab for lab, v in zip(FC.CAL_LABELS, ind) if v >= 1e-3],
                                  "per_bin": dict(zip(FC.CAL_LABELS, ind.tolist()))}
    return out


def rebuild(kind, STB, workers):
    """Basis set T at ELL_FIX with frozen or stable radial weights -> LOCAL_WORK/basis_T_<kind>.npz."""
    FC.ELL = ELL_FIX
    FC.SEL_INDEX = [int(np.flatnonzero(ELL_FIX == e)[0]) for e in FC.SEL_ELL]
    FC.WORK = LOCAL_WORK
    frozen = E.radial_weights
    columns = K._columns
    K._columns = lambda *a, **k: None                       # DM columns (projections) are not affected: skipped
    if kind == "stable":
        E.radial_weights = STB.stable_radial_weights          # in memory only; kernels/engine.py is unchanged
    try:
        timing = K.build(sets=("T",), workers=workers)
    finally:
        E.radial_weights = frozen
        K._columns = columns
    os.replace(LOCAL_WORK / "basis_T.npz", LOCAL_WORK / f"basis_T_{kind}.npz")
    return timing


SPECTRA = ("yy1h", "yy2h", "yytot", "ksz1h", "ksz2h", "dmcl1h", "dmcl2h", "dmcltot", "ydm1h", "ydm2h", "ydmtot")


def effect():
    """Every model (deterministic and lognormal-1R predictions) and the truth with the frozen and the stable
    rebuild: max |stable/frozen - 1| of each spectrum at the selected multipoles, the change of every relative
    error (model vs truth), and the frozen rebuild against the maas values (reproducibility)."""
    import csv
    import f_basis as FB
    FC.WORK = LOCAL_WORK
    Bf, Bs = FB.Basis("T_frozen"), FB.Basis("T_stable")
    pr = np.load(FC.PRODUCTS / "profiles.npz")
    tr = np.load(FC.PRODUCTS / "truth.npz")
    mm = np.load(FC.PRODUCTS / "models_main.npz")
    cal = FC.CAL_BINS
    S = dict(SPP=pr["SPP_main"][cal], SEE=pr["SEE_main"][cal], SPE=pr["SPE_main"][cal])
    Pt, Et = pr["meanP_main"][cal], pr["meanE_main"][cal]
    mom = tr["lognormal_main_1R"]
    sel = FC.SEL_INDEX
    out = {"spectra_max_abs_stable_over_frozen_minus_1": {}, "error_shift_max_abs": {}, "per_model": {}}
    tf, ts = Bf.observables(Pt, Et, **S), Bs.observables(Pt, Et, **S)
    truth_shift = {k: (ts[k][sel] / tf[k][sel] - 1) for k in SPECTRA}
    out["truth_stable_over_frozen_minus_1"] = {k: v.tolist() for k, v in truth_shift.items()}
    worst = {k: 0.0 for k in SPECTRA}
    worst_err = {k: 0.0 for k in SPECTRA}
    rows = []
    for i, label in enumerate(mm["labels"]):
        P, E_ = mm["P"][i], mm["E"][i]
        for pred in ("deterministic", "lognormal_1R"):
            kw = {}
            if pred == "lognormal_1R":
                kw = dict(SPP=np.exp(mom[:, 0])[:, None, None] * np.einsum("cs,ct->cst", P, P),
                          SEE=np.exp(mom[:, 1])[:, None, None] * np.einsum("cs,ct->cst", E_, E_),
                          SPE=mom[:, 2][:, None, None] * np.einsum("cs,ct->cst", P, E_))
            of, os_ = Bf.observables(P, E_, **kw), Bs.observables(P, E_, **kw)
            for k in SPECTRA:
                d = os_[k][sel] / of[k][sel] - 1
                e = (os_[k][sel] / ts[k][sel]) - (of[k][sel] / tf[k][sel])
                worst[k] = max(worst[k], float(np.max(np.abs(d))))
                worst_err[k] = max(worst_err[k], float(np.max(np.abs(e))))
                for e_, i_ in zip(FC.SEL_ELL, range(len(sel))):
                    rows.append({"model": str(label), "prediction": pred, "observable": f"{k}_{e_}",
                                 "frozen": of[k][sel][i_], "stable": os_[k][sel][i_], "stable_over_frozen_minus_1": d[i_],
                                 "rel_error_shift": e[i_]})
    out["spectra_max_abs_stable_over_frozen_minus_1"] = worst
    out["error_shift_max_abs"] = worst_err
    FC.write_csv(LOCAL / "numerics_per_model.csv", rows)
    # reproducibility: frozen rebuild vs the maas values (deterministic predictions)
    maas = {}
    with open(FC.TABLES / "F_errors_long.csv") as handle:
        for r in csv.DictReader(handle):
            if r["prediction"] == "deterministic":
                maas[(r["model"], r["observable"])] = float(r["value"])
    dev = [abs(r["frozen"] / maas[(r["model"], r["observable"])] - 1) for r in rows
           if r["prediction"] == "deterministic" and (r["model"], r["observable"]) in maas]
    out["frozen_rebuild_vs_maas_max_abs"] = float(max(dev))
    out["frozen_rebuild_vs_maas_compared"] = len(dev)
    return out


def main():
    t0 = time.perf_counter()
    LOCAL.mkdir(exist_ok=True)
    LOCAL_WORK.mkdir(exist_ok=True)
    STB, path = load_stable()
    res = {"td_stable": str(path.relative_to(FC.ROOT)), "td_stable_sha256": FC.sha256(path)}
    res["indicators"] = indicators(STB)
    res["model_indicators_not_used_by_the_run"] = model_indicators(STB)
    worst = max(v["max_indicator"] for v in res["indicators"].values())
    res["max_indicator"] = worst
    res["threshold"] = 1e-3
    FC.log("indicators", json.dumps({k: v["max_indicator"] for k, v in res["indicators"].items()}))
    if worst >= 1e-3 or "--force" in sys.argv:
        res["rebuild"] = {}
        for kind in ("frozen", "stable"):
            res["rebuild"][kind] = rebuild(kind, STB, FC.WORKERS)
            FC.log(f"rebuild {kind} done")
        res["ell"] = ELL_FIX.tolist()
    if "rebuild" in res:
        res["effect"] = effect()
    res["wall_seconds"] = time.perf_counter() - t0
    FC.write_json(LOCAL / "numerics.json", res)
    FC.log("numerics step done", res["wall_seconds"])


if __name__ == "__main__":
    main()
