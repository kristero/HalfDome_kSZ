"""Test I stages on Test F's 35-shell fit grid, in Test G's configuration (one process per stage, run by run_test_i.py).

Configuration (before any F / G import; nothing is written into F_observables/ or G_physical_outer/):
  G_RESULTS = <I results>/work/g35, F_RESULTS = <I results>/work/g35/f_base (= g_common.FBASE)
  <g35>/f_base/profiles.npz  derived from Test G's 35-shell per-halo profiles (G_physical_outer/results/f_base/
                             profiles.npz): every '*_main' array (all gas) is replaced by the corresponding '*_a2ii'
                             array (A2-ii painting partition, recent-AGN gas removed) and the original all-gas arrays
                             are kept as '*_allgas'.  Test G's selection machinery (g_selection: Cells, covariances,
                             Problem, the ladder, CV and the selection rule), which reads the 'main' stacks, therefore
                             runs unchanged on the A2-ii stacks (I2).
  <g35>/prepare.npz          Test G's prepare products (ebar, kappa_P for the O1 / O2 outer options), read-only copy.

Stages
  fits35   I1 on the 35-shell grid: B12 (P) / B16 (E) per calibrated bin, R_fit in {1, 1.34, 1.5, 2, 3, 4},
           objectives
             (a) chi2 = sum_s [(ln S_s - ln m_s) / sigma_s]^2, sigma_s = max(SD_s / m_s, 5%)  (Test F / G / H)
             (d) chi2 = sum_s w_s (ln S_s - ln m_s)^2, w_s = Test H's A2-ii family importance of the bin summed
                 into the F shell (P: tSZ; E: (kSZ + DM)/2, each family map normalised to unit sum), started
                 from the (a) fit
             (L) chi2 = r^T (C_L22 / h)^-1 r, r = ln S - ln m, C_L22 = inclusion-weighted sample covariance of
                 the haloes' ln F divided by N_eff, h = (N - p - 2)/(N - 1) (Lee22's convention; shells where any
                 halo has F <= 0 are dropped from this likelihood)
           (S_s = exact shell average of the model, m_s = inclusion-weighted A2-ii mean, ln = natural log)
           and I5's B12 / B16 refits (objective a) of the stacking-choice variants (inputs/i0_variants.npz) at
           R_fit in {2, 4}.
  ladder   I2: g_selection.main() on the derived (A2-ii) profiles, its observable propagation replaced by a no-op
           (Test I propagates on Test H's fine A2-ii basis instead, in i_fine).  Outputs in <g35>/: tables/
           G_selection_*.csv, selection.npz, selection_summary.json.
  lvariants I5: the selected Lee22-ladder model (mass data set, R_fit 2 and 4) and PL3 refitted to every variant's
           stacks with diagonal weights sigma_s = max(SD_s / m_s, 5%) (objective (a) convention: the variants have
           no bootstrap covariances), with the outer option of the selected model.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import i_common as IC  # noqa: E402

G35 = IC.WORK / "g35"
FB35 = G35 / "f_base"
G_SRC = IC.ROOT / "G_physical_outer" / ("results_smoke" if IC.SMOKE else "results")


def prepare_inputs():
    FB35.mkdir(parents=True, exist_ok=True)
    src = G_SRC / "f_base" / "profiles.npz"
    dst = FB35 / "profiles.npz"
    if not dst.exists():
        with np.load(src, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        out = dict(d)
        for k in list(d):
            if k.endswith("_main") or "_main_" in k:
                alt = k.replace("_main", "_a2ii")
                if alt in d:
                    out[k.replace("_main", "_allgas")] = d[k]
                    out[k] = d[alt]
        tmp = FB35 / "profiles.tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, dst)
    if not (G35 / "prepare.npz").exists():
        shutil.copyfile(G_SRC / "prepare.npz", G35 / "prepare.npz")


def configure(workers):
    prepare_inputs()
    os.environ["G_MODE"] = IC.MODE
    os.environ["G_RESULTS"] = str(G35)
    os.environ["G_WORKERS"] = str(workers)
    os.environ["STUDY_ROOT"] = str(IC.ROOT)
    sys.path.insert(0, str(IC.ROOT / "G_physical_outer"))
    sys.path.insert(0, str(IC.ROOT / "F_observables"))


# ============================================================================ fits35
def sum_to_f35(M, x200, f35):
    idx = np.array([int(np.argmin(np.abs(x200 - e))) for e in f35])
    assert np.allclose(x200[idx], f35, atol=1e-12)
    return np.add.reduceat(M[..., :len(x200) - 1], idx[:-1], axis=-1)


def weight_maps(GC):
    sens = dict(np.load(IC.H_RESULTS / "sens.npz"))
    W = IC.family_weights(sens, GC.CAL_BINS, n_x=len(sens["x_edges"]) - 1)
    W35 = {f: sum_to_f35(v, np.asarray(sens["x_edges"]), np.asarray(GC.F_EDGES)) for f, v in W.items()}
    norm = {f: v / v.sum(axis=1, keepdims=True) for f, v in W35.items()}
    return {"P": norm["tSZ"], "E": 0.5 * norm["kSZ"] + 0.5 * norm["DM"]}


def l22_cov(pr, field, c, GC):
    """C_L22 [35, 35] (NaN rows / cols on shells with any F <= 0), Hartlap needs p: returned with N, N_eff, ok."""
    rows = np.flatnonzero(pr["mass_bin"] == GC.CAL_BINS[c])
    v = np.asarray(pr[f"{field}_a2ii"][rows], float)
    w = pr["weight"][rows] / pr["weight"][rows].sum()
    ok = np.all(np.isfinite(v) & (v > 0), axis=0)
    L = np.log(np.where(v > 0, v, 1.0))
    mu = w @ L
    d = L - mu
    C = (d.T * w) @ d / max(1 - np.sum(w**2), 1e-12)
    neff = 1 / np.sum(w**2)
    return C / neff, len(rows), ok


def fit_rec(fit, extra=None):
    if fit is None:
        return None
    rec = {k: fit.get(k) for k in ("parameters", "chi2", "chi2_per_dof", "n_shells", "rms_percent", "accepted")}
    rec["parameters"] = np.asarray(fit["parameters"], float).tolist()
    rec.update(extra or {})
    return rec


def stage_fits35():
    import g_common as GC
    FF = GC.FF
    AF = FF.F
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    wmap = weight_maps(GC)
    out = {"fits": {}, "variants": {}, "f_edges": np.asarray(GC.F_EDGES).tolist(), "definitions": __doc__}
    for field in ("P", "E"):
        fam = FF.FIELD_FAMILY[field]
        for rf in IC.R_FIT_I1:
            for c in range(GC.N_CAL):
                bd = FF.BinData(pr, field, "a2ii", c)
                fa = FF.fit_objective(bd, "a", rf)
                out["fits"][f"{field}|a|{rf:g}|{c}"] = fit_rec(fa)
                st = fa["parameters"] if fa is not None else None
                fd = FF.fit_objective(bd, "d", rf, start=st, weights=wmap[field][c])
                out["fits"][f"{field}|d|{rf:g}|{c}"] = fit_rec(fd)
                C, N, okL = l22_cov(pr, field, c, GC)
                mask = bd.mask(rf) & okL
                p = int(mask.sum())
                h = (N - p - 2) / (N - 1) if N > p + 2 else np.nan
                fl = None
                if np.isfinite(h) and p >= 4:
                    try:
                        fl = AF.fit_general(fam, bd.mean, mask, cov=C / h, start=st, edges=GC.F_EDGES)
                    except Exception as exc:                      # noqa: BLE001
                        IC.log("L fit failed", field, rf, c, repr(exc))
                out["fits"][f"{field}|L|{rf:g}|{c}"] = fit_rec(fl, {"hartlap": h, "N": N, "p": p})
            IC.log("fits35", field, rf)
    # ---- I5 variants (objective a)
    var = dict(np.load(IC.VARIANTS_NPZ))
    for v in IC.VARIANTS:
        for field in ("P", "E"):
            mean, sd, res = var[f"mean_{v}_{field}"], var[f"sd_{v}_{field}"], var[f"res_{v}_{field}"]
            bins = var[f"bins_{v}"]
            for i in range(len(mean)):
                b = int(np.argmin(np.abs(GC.FC.MASS_EDGES[:-1] - bins[i, 0])))
                base = res[i] & np.isfinite(mean[i]) & (mean[i] > 0)
                if abs(bins[i, 1] - bins[i, 0] - 0.2) < 1e-6:
                    base &= np.asarray(pr["base_mask"][b], bool)
                bd = types.SimpleNamespace(mean=np.nan_to_num(mean[i]), sd=np.nan_to_num(sd[i]), base=base,
                                           family=FF.FIELD_FAMILY[field],
                                           mask=lambda r, base=base: base & GC.FC.shells_within(GC.F_EDGES, r))
                for rf in IC.R_FIT_I5:
                    out["variants"][f"{v}|{field}|{rf:g}|{i}"] = fit_rec(FF.fit_objective(bd, "a", rf),
                                                                         {"bin_lo": bins[i, 0], "bin_hi": bins[i, 1]})
        IC.log("variants", v)
    IC.write_json(IC.RESULTS / "fits35.json", out)


# ============================================================================ ladder (I2)
def stage_ladder():
    import g_common as GC
    import g_selection as GS
    GS.propagate = lambda *a, **k: []            # observables: i_fine (Test H's fine A2-ii basis)
    GS.main()
    for name in ("selection_summary.json", "selection.npz"):
        shutil.copyfile(GC.RESULTS / name, IC.RESULTS / f"ladder_{name}")
    for p in sorted((GC.TABLES).glob("G_selection_*.csv")):
        shutil.copyfile(p, IC.TABLES / p.name.replace("G_selection", "I2_ladder"))


# ============================================================================ lvariants (I5, Lee22 model)
def stage_lvariants():
    import g_common as GC
    import g_physics as PH
    import g_selection as GS
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    summ = IC.read_json(IC.RESULTS / "ladder_selection_summary.json")
    import json
    results = json.loads(str(np.load(IC.RESULTS / "ladder_selection.npz")["results_json"]))
    res_by = {(r["field"], r["dset"], float(r["R_fit"]), r["model"]): r for r in results}
    prep = GC.load_prepare()
    ctx = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"])}
    lh = PH.LinearHalo()
    x35, wq35 = PH.quad_nodes(GC.F_EDGES)
    var = dict(np.load(IC.VARIANTS_NPZ))
    out = {}
    for field in ("P", "E"):
        for rf in IC.R_FIT_I5:
            ch = summ["chosen"].get(f"{field}|mass|{rf:g}")
            if ch is None:
                continue
            outer = ch["model"].split("+")[1]
            names = [ch["model"]] + [n for n in (f"PL3+{outer}",) if n != ch["model"]]
            for name in names:
                r0 = res_by.get((field, "mass", rf, name))
                if r0 is None or "floor" not in r0["fits"]:
                    continue
                inner = name.split("+")[0]
                spec = [s for s in GS.model_specs(field, "mass") if s[0] == inner][0]
                theta0 = np.asarray(r0["fits"]["floor"]["theta"], float)
                for v in IC.VARIANTS:
                    mean, sd, resv = var[f"mean_{v}_{field}"], var[f"sd_{v}_{field}"], var[f"res_{v}_{field}"]
                    bins = var[f"bins_{v}"]
                    n = len(mean)
                    rows = [np.flatnonzero((np.log10(pr["m200c_msun"]) >= bins[i, 0]) &
                                           (np.log10(pr["m200c_msun"]) < bins[i, 1])) for i in range(n)]
                    T2h = np.array([PH.two_halo_template(lh, pr["m200c_msun"][r], pr["r200c_mpc"][r], pr["weight"][r],
                                                         x35) for r in rows])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sig = np.maximum(np.where(mean > 0, sd / mean, 1.0), 0.05)
                    strict = []
                    for i in range(n):
                        m = resv[i] & np.isfinite(mean[i]) & (mean[i] > 0)
                        if abs(bins[i, 1] - bins[i, 0] - 0.2) < 1e-6:
                            b = int(np.argmin(np.abs(GC.FC.MASS_EDGES[:-1] - bins[i, 0])))
                            m &= np.asarray(pr["strict_mask"][b], bool)
                        strict.append(m)
                    strict = np.array(strict)
                    cells = types.SimpleNamespace(n=n, bin=np.arange(n) if n == GC.N_CAL else np.zeros(n, int),
                                                  logm=var[f"logm_{v}"], lnc=var[f"lnc_{v}"],
                                                  mean=np.where(mean > 0, mean, 1.0), T2h=T2h)
                    if n != GC.N_CAL and "bin" in spec[1].values():
                        out[f"{v}|{field}|{rf:g}|{name}"] = {"skipped": "per-bin law on a different binning"}
                        continue
                    model = GS.build_model(field, spec, outer, n)
                    covs = []
                    S = int(np.sum(GC.x_hi_le(GC.F_EDGES, rf)))
                    for i in range(n):
                        mk = strict[i] & GC.x_hi_le(GC.F_EDGES, rf)
                        idx = np.flatnonzero(mk)
                        covs.append({"mask": mk, "C": np.diag(sig[i, idx]**2), "C_L22": None, "hartlap": np.nan})
                    prob = GS.Problem(model, cells, covs, "stat", 0.0, ctx, rf)
                    if prob.n_data <= model.k:
                        continue
                    starts = [theta0]
                    sol = prob.fit(starts)
                    if sol is None:
                        continue
                    out[f"{v}|{field}|{rf:g}|{name}"] = {"theta": sol.x.tolist(), "names": list(model.names),
                                                         "chi2": float(np.sum(prob.residuals(sol.x)**2)),
                                                         "n_data": prob.n_data, "k": model.k,
                                                         "selected": name == ch["model"], "S": S}
                IC.log("lvariants", field, rf, name)
    IC.write_json(IC.RESULTS / "lvariants.json", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fits35", "ladder", "lvariants"))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    configure(a.workers)
    {"fits35": stage_fits35, "ladder": stage_ladder, "lvariants": stage_lvariants}[a.stage]()


if __name__ == "__main__":
    main()
