"""LOCAL, light: the exact mean halo DM (z_s = 1) of every Test J evaluation, while the maas run is pending.

Mean DM is linear in the profiles, so it needs only the volume integral (the ell = 0 term) of every basis function in
every lightcone cell z <= 1 - no Fourier transforms at ell > 0 and no projected columns.  The arithmetic is f_kernels'
(the ell = 0 column of engine.radial_weights; the same reference shapes, redshift response and amplitudes):
    <DM> = sum_c sum_s E_c,s m_c,s + <DM>_fb ,
    m_c,s = sum_{cells z <= 1, bin c} s_DM(z) N/(4 pi) tau_amp <phi_s>_0,     <phi_s>_0 = 4 pi int x^2 g_c(x) e(x) dx / <g_c>_s
(N: catalogue haloes of the cell; tau_amp: lightcone_analytic electron amplitude; s_DM = 1/(DM_FACTOR (1+z)); g_c: the
bin's reference shape; e(x) = B16(M, z, x)/B16(M, 0, x) the declared redshift response; <g_c>_s its shell average), and
the fallback cells contribute their published B16 integrated over the shells inside the cut.  Validated against Test
H's truth (configuration C10 / D0: mean DM of the A2-ii truth) and, in smoke mode, against the smoke run's j_truth.
Outputs (post/local_mean/): mean_dm.json (every evaluation, per-bin terms, importance maps, cumulative curves, R_X,
tolerance), kernels.npz.
Usage: OMP_NUM_THREADS=2 python -B j_local_mean.py [--smoke]   (needs a local 'prepare' in post/local_mean/prepare)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ARGS = ap.parse_args()
OUT = HERE / ("post_smoke" if ARGS.smoke else "post") / "local_mean"
os.environ["J_MODE"] = "smoke" if ARGS.smoke else "full"
os.environ["J_RESULTS"] = str(OUT / "prepare")
os.environ["J_WORKERS"] = "1"
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import j_basis as JBm  # noqa: E402
import j_common as JC  # noqa: E402

FC, FK, E, A = JC.FC, JC.FK, JC.E, JC.A
CONFIGS = ("C14", "C10", "C0")


def kernels(pr):
    """m[set][c, s] (calibrated, J bins) and the fallback per (config, cut): total and per (engine bin, shell)."""
    sky = FC.make_sky()
    _, grid, seg_shell, n_seg_4 = FK.basis_grid()
    x = grid.x
    sp = JBm.specs(pr)
    gE = {r: np.asarray(sp[r]["gE"]) for r in sp}
    gbar = {r: FK.shell_averages(grid, seg_shell, gE[r], JC.N_FINE) for r in sp}
    w0 = [E.radial_weights(x[a:b + 1], np.zeros((1, 1)))[0, 0] for a, b in grid.segments]    # 4 pi int x^2 phi dx
    profiles = {}
    for cfg in CONFIGS:
        mask = np.zeros(JC.N_ENGINE, bool)
        mask[JC.SUPPORT_BINS[cfg]] = True
        profiles[cfg] = E.Profile(1, "bins", FC.SUPPORT["mass_edges"], mask, FC.SUPPORT["max_logmass"])
    cal_index = np.full(JC.N_ENGINE, -1)
    cal_index[JC.J_BINS] = np.arange(JC.N_J)
    m = {r: np.zeros((JC.N_J, JC.N_FINE)) for r in sp}
    fb = {(cfg, cut): np.zeros((16, JC.N_X)) for cfg in CONFIGS for cut in JC.CUTS}
    for j in np.flatnonzero((sky.z <= 1.0) & (sky.counts.sum(axis=1) > 0)):
        index = np.flatnonzero(sky.counts[j] > 0)
        zz = sky.z[j]
        mass = sky.mass[index]
        n4 = sky.counts[j, index].astype(float) / (4 * np.pi)
        da, radius, y_amp, tau_amp = sky.ingredients.geometry(zz, mass)
        dm_scale = 1. / (E.DM_FACTOR * (1 + zz))
        pub0 = A.published_profile(1, mass, 0., x)
        evol = A.published_profile(1, mass, zz, x) / pub0
        safe, valid = profiles["C14"].bins(mass)
        cal = np.flatnonzero(valid)
        cidx = cal_index[safe[cal]]
        for r in sp:
            prof = gE[r][cidx] * evol[cal]
            piece = np.zeros((len(cal), JC.N_FINE))
            for s_i, (a, b) in enumerate(grid.segments):
                piece[:, seg_shell[s_i]] += prof[:, a:b + 1] @ w0[s_i]
            Bt0 = tau_amp[cal, None] * piece / gbar[r][cidx]
            np.add.at(m[r], cidx, dm_scale * n4[cal, None] * Bt0)
        fbprof = pub0 * evol
        piece = np.zeros((len(index), JC.N_X))
        for s_i, (a, b) in enumerate(grid.segments[:n_seg_4]):
            piece[:, seg_shell[s_i]] += fbprof[:, a:b + 1] @ w0[s_i]
        yE0 = tau_amp[:, None] * piece
        for cfg in CONFIGS:
            f = np.flatnonzero(~profiles[cfg].bins(mass)[1])
            for cut in JC.CUTS:
                k = JC.K_CUT[cut]
                np.add.at(fb[(cfg, cut)], (sky.mass_bin[index][f], slice(None)),
                          dm_scale * n4[f, None] * np.where(np.arange(JC.N_X) < k, yE0[f], 0.0))
    return m, fb


def main():
    import j_obs as JO
    pr = JC.load_npz(JC.WORK / "profiles_j.npz")
    m, fb = kernels(pr)
    np.savez_compressed(OUT / "kernels.npz", m_JA=m["JA"], m_JB=m["JB"],
                        **{f"fb|{c}|{k:g}": v for (c, k), v in fb.items()})
    import j_truth as JT
    res = {"note": "LOCAL computation of the linear observable mean DM only (exact; see module docstring)",
           "evaluations": {}, "per_bin": {}, "fallback_per_bin": {}}
    for tag, basis, config, sel, cut, treat, sub in JT.eval_list():
        mom = JO.moments(pr, sel, cut, treat, config, sub)
        Ecal = JC.pad75(mom["E"])
        cal_terms = (Ecal * m[basis]).sum(axis=1)
        fbt = fb[(config, cut)]
        tot = float(cal_terms.sum() + fbt.sum())
        res["evaluations"][tag] = {"dm_mean": tot, "calibrated": float(cal_terms.sum()), "fallback": float(fbt.sum())}
        if treat == "T1" and not sub and config in ("C14", "C10"):
            res["per_bin"][tag] = cal_terms.tolist()
            res["fallback_per_bin"][tag] = fbt.sum(axis=1).tolist()
    # validation
    val = {}
    if not JC.SMOKE:
        h = np.load(JC.H_RESULTS / "truth.npz")
        val["C10|D0|T1_vs_TestH_truth_a2ii_dm_mean"] = res["evaluations"]["C10|D0|T1"]["dm_mean"] / float(
            h["truth_a2ii__dm_mean"]) - 1
    else:
        ts = json.loads((HERE / "results_smoke" / "truth_summary.json").read_text())
        val["vs_smoke_j_truth_max_rel"] = max(abs(v["dm_mean"] / ts["evaluations"][f"{t}|truth"]["dm_mean"] - 1)
                                              for t, v in res["evaluations"].items())
    res["validation"] = val
    # importance, cumulative curves, radii, tolerance (C14, T1)
    xk = JC.X200
    dln = JC.dlnx(JC.X200)
    mid = np.sqrt(np.maximum(JC.X200[:-1], 1e-6) * JC.X200[1:])
    bands = ((0.0, 0.2335), (0.2335, 0.5185), (0.5185, 1.0), (1.0, 2.0), (2.0, 4.0))
    band_of = np.searchsorted(np.array([b[1] for b in bands]), mid, side="left")
    res["sens"] = {}
    for d, v in JC.DEFS.items():
        tag = f"C14|{d}|T1"
        mom = JO.moments(pr, v["sel"], v["cut"], "T1", "C14")
        J = JC.pad75(mom["E"])[:, :JC.N_X] * m[v["basis"]][:, :JC.N_X]               # [c, s]
        O = res["evaluations"][tag]["dm_mean"]
        imp = np.array(fb[("C14", v["cut"])], copy=True)
        for c, b in enumerate(JC.J_BINS):
            imp[b] += J[c]
        imp /= O
        kB = JC.K_CUT[v["cut"]]
        cum = np.concatenate([[0.0], np.cumsum(J.sum(axis=0))])[:kB + 1]                # calibrated haloes cut at R_t
        C = cum / cum[-1]
        rad = {f"R{int(round(100 * X))}": JC.interp_crossing(xk[:kB + 1], C, X) for X in (0.5, 0.8, 0.9, 0.95, 0.99)}
        per_shell = imp[JC.J_BINS].sum(axis=0)
        with np.errstate(divide="ignore"):
            tol = 0.01 / np.abs(per_shell / dln * 0.1 * np.log(10))
        jmin = int(np.argmin(np.where(np.arange(JC.N_X) < kB, tol, np.inf)))
        res["sens"][d] = {"sum_rule": float(imp.sum()), "bands": {f"{a:g}-{b:g}R": float(imp[:, band_of == q].sum())
                                                                 for q, (a, b) in enumerate(bands)},
                          "radii": rad, "loss_at": {f"{r:g}R": float(1 - (cum[JC.edge_index(xk, r, 1e-3)] + fb[("C14", v["cut"])].sum())
                                                                      / O) for r in (0.5185, 1.0, 1.5, 2.0, 3.0) if r <= v["cut"]},
                          "shares": {JC.ENGINE_LABELS[b]: float(imp[b].sum()) for b in range(16)},
                          "tolerance_min_per_0.1dex": float(tol[jmin]), "tolerance_at_x": float(mid[jmin]),
                          "importance_map": imp.tolist(), "cumulative": C.tolist()}
    (OUT / "mean_dm.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"validation": val, **{t: round(v["dm_mean"], 4) for t, v in res["evaluations"].items()}}, indent=1))


if __name__ == "__main__":
    main()
