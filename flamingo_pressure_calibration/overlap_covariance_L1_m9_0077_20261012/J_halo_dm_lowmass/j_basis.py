"""J stage 'basis' (J1b): Test F's in-basis observable kernels on Test H's fine grid with the calibrated support extended
to 12.7-15.388 (configuration C14: 14 calibrated engine bins).

The per-redshift-shell kernel code is Test F's (f_kernels._chunk / _shell / _columns, imported read-only and run in this
process with the fine grid and the C14 support set on the f_common module object, j_common).  This module only sets up
the job, exactly as Test H's h_basis.build does (same grid, same reference-shape construction), except that chunk
results are summed as they arrive (same order, so the same sums) to keep the parent's memory low.

Sets (both on the same 75 fine shells and radial nodes; reference shapes g_b = log-log interpolation through the
fine-shell centres of the bin's T1 stacked mean, inner slope clipped to [-1, 0], outer slope <= -1; coefficients =
those means):
  JA  A2-ii pressure and A2-ii electrons (definitions D0, D1, D2; the 10 bins 13.5-15.388 are bitwise Test H's HA)
  JB  A2-ii pressure and host-bound electrons (definition D3)
Every basis function has unit shell average, so any shell-average profile is represented exactly whatever g_b is;
only the within-shell shape comes from g_b (Test H: changing it moves observables by <= 3.8e-4).
Also writes basis_JA10.npz / basis_JB10.npz: the four arrays f_pdf reads (n_shell, n_seg, g_E, gbar_E) restricted to
the 10 bins of configuration C10, so that DM-PDF passes can run with Test H's support.
Output: <F_RESULTS>/work/basis_{JA,JB,JA10,JB10}.npz (scratch), results/basis.json.
"""
from __future__ import annotations

import resource
import time

import j_common as JC
import numpy as np

FC, FK, E = JC.FC, JC.FK, JC.E


def truth_shapes(mean, x, edges):
    """Test H's h_basis.truth_shapes (identical code)."""
    edges = np.asarray(edges, float)
    centre = np.sqrt(np.maximum(edges[:-1], 1e-300) * edges[1:])
    centre[0] = 0.716 * edges[1]
    lc = np.log(centre)
    lx = np.log(x)
    out = np.zeros((len(mean), len(x)))
    for i, m in enumerate(mean):
        ok = np.isfinite(m) & (m > 0)
        lm = np.interp(lc, lc[ok], np.log(m[ok]))
        g = np.interp(lx, lc, lm)
        lo_slope = float(np.clip((lm[1] - lm[0]) / (lc[1] - lc[0]), -1.0, 0.0))
        hi_slope = min((lm[-1] - lm[-2]) / (lc[-1] - lc[-2]), -1.0)
        g = np.where(lx < lc[0], lm[0] + lo_slope * (lx - lc[0]), g)
        g = np.where(lx > lc[-1], lm[-1] + hi_slope * (lx - lc[-1]), g)
        out[i] = np.exp(g)
    return out


def specs(pr):
    _, gridobj, _, _ = FK.basis_grid()
    x = gridobj.x
    jb = JC.J_BINS
    mP = pr["mean_P_a2ii_T1"][jb]
    out = {}
    for name, ekey in (("JA", "mean_E_a2ii_T1"), ("JB", "mean_E_hb_T1")):
        mE = pr[ekey][jb]
        out[name] = dict(gP=truth_shapes(mP, x, JC.X200), gE=truth_shapes(mE, x, JC.X200), coefP=JC.pad75(mP),
                         coefE=JC.pad75(mE))
    return out


def build(sp, workers=None, chunks_per_worker=2):
    """Test H's h_basis.build with running chunk sums (C14 support must be active)."""
    workers = JC.WORKERS if workers is None else workers
    assert JC.CONFIG["name"] == "C14"
    started = time.perf_counter()
    sky = FC.make_sky()
    ell = FC.ELL
    base, gridobj, seg_shell, n_seg_4 = FK.basis_grid()
    support = FC.SUPPORT
    profile = E.Profile(0, "bins", support["mass_edges"], np.asarray(support["support"], bool), support["max_logmass"])
    cal_index = np.full(len(support["support"]), -1)
    cal_index[FC.CAL_BINS] = np.arange(FC.N_CAL)
    sets = list(sp)
    g, gbar, coef, n_shell, n_seg = {}, {}, {}, {}, {}
    for r, s in sp.items():
        g[r] = [np.asarray(s["gP"], float), np.asarray(s["gE"], float)]
        gbar[r] = [FK.shell_averages(gridobj, seg_shell, v, JC.N_FINE) for v in g[r]]
        coef[r] = [np.asarray(s["coefP"], float), np.asarray(s["coefE"], float)]
        n_shell[r], n_seg[r] = JC.N_FINE, len(gridobj.segments)
    cs = E.column_setup(gridobj, FC.X_END, True)
    FK._JOB.clear()
    FK._JOB.update(sky=sky, grid=gridobj, ell=ell, seg_shell=seg_shell, n_seg_4=n_seg_4, profile=profile,
                   cal_index=cal_index, sets=sets, g=g, gbar=gbar, coef=coef, n_shell=n_shell, n_seg=n_seg,
                   A_seg=cs["A_seg"], area=cs["area"])
    chunks = E._balanced_chunks(sky, workers * chunks_per_worker)
    JC.log(f"J basis build: sets {sets}, N_CAL {FC.N_CAL}, {len(gridobj.x)} nodes, {len(gridobj.segments)} segments, "
           f"{len(cs['b'])} b nodes, {len(chunks)} chunks on {workers} workers")
    nz, nl = len(sky.z), len(ell)
    acc = {r: None for r in sets}
    W = {r: (np.zeros((nz, FC.N_CAL, n_shell[r], nl)), np.zeros((nz, FC.N_CAL, n_shell[r], nl))) for r in sets}
    fbacc = None
    WfP = np.zeros((nz, nl))
    WfE = np.zeros((nz, nl))
    JC.mem("before fork: basis build")
    for k, (rows, a) in enumerate(FC.pool_map(FK._chunk, chunks, workers)):
        for r in sets:
            acc[r] = {kk: v.copy() for kk, v in a[r].items()} if acc[r] is None else {kk: acc[r][kk] + a[r][kk] for kk in acc[r]}
        fbacc = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in a["fb"].items()} if fbacc is None \
            else {kk: fbacc[kk] + a["fb"][kk] for kk in fbacc}
        for row in rows:
            for r in sets:
                W[r][0][row["j"]], W[r][1][row["j"]] = row["W"][r]
            WfP[row["j"]], WfE[row["j"]] = row["W_fb"]
        del rows, a
        JC.log(f"chunk {k + 1}/{len(chunks)}: {time.perf_counter() - started:.0f} s")
    FK._JOB.clear()
    ing = sky.ingredients
    pk = np.array([ing.pk(z, ell) for z in sky.z])
    common = dict(ell=ell, z=sky.z, volume=sky.volume, pk=pk, use_dm=sky.z <= 1.,
                  dm_scale=1. / (E.DM_FACTOR * (1 + sky.z)), s2_lin=(sky.sigma_v_lin / E.C_KMS)**2, W_fb_y=WfP,
                  W_fb_tau=WfE, grid_x=gridobj.x, grid_breaks=gridobj.breaks, seg_shell=seg_shell, n_seg_4=n_seg_4,
                  basis_edges=FC.BASIS_EDGES, f_edges=FC.F_EDGES, shell_volume=FC.SHELL_VOLUME,
                  cal_bins=np.array(FC.CAL_BINS),
                  catalogue_haloes_per_engine_bin=np.bincount(sky.mass_bin, weights=sky.counts.sum(axis=0), minlength=16),
                  catalogue_haloes_per_engine_bin_z_le_1=np.bincount(
                      sky.mass_bin, weights=sky.counts[sky.z <= 1].sum(axis=0), minlength=16))
    common.update({f"fb_{k}": v for k, v in fbacc.items()})
    timing = {"wall_seconds": time.perf_counter() - started, "workers": workers, "chunks": len(chunks),
              "nodes": len(gridobj.x), "segments": len(gridobj.segments), "b_nodes": len(cs["b"]), "sets": sets,
              "n_cal": FC.N_CAL, "cal_bins": FC.CAL_BINS,
              "peak_rss_parent_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "peak_rss_largest_child_GiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2}
    for r in sets:
        out = dict(common)
        out.update(acc[r], W_y=W[r][0], W_tau=W[r][1], g_P=g[r][0], g_E=g[r][1], gbar_P=gbar[r][0], gbar_E=gbar[r][1],
                   coef_P=coef[r][0], coef_E=coef[r][1], n_shell=n_shell[r], n_seg=n_seg[r])
        out["timing_json"] = np.array(FC.json.dumps(timing))
        np.savez(JC.BIG / f"basis_{r}.npz", **out)
        # f_pdf-only file for configuration C10 (Test H's support): the 10 bins 13.5-15.388
        small = dict(n_shell=n_shell[r], n_seg=n_seg[r], g_E=g[r][1][JC.C10_IN_J], gbar_E=gbar[r][1][JC.C10_IN_J],
                     g_P=g[r][0][JC.C10_IN_J], gbar_P=gbar[r][0][JC.C10_IN_J])
        np.savez(JC.BIG / f"basis_{r}10.npz", **small)
        del out
    JC.log(f"J basis build done: {timing}")
    return timing


def main():
    pr = JC.load_npz(JC.WORK / "profiles_j.npz", keys=lambda k: k.startswith("mean_"))
    sp = specs(pr)
    timing = build(sp)
    # identity of this module's truth_shapes with Test F's f_kernels.truth_shapes on Test F's own grid
    _, gridobj, _, _ = FK.basis_grid()
    fpr = JC.load_npz(JC.F_PROFILES, keys=["meanE_main"])
    saved = dict(F_EDGES=FC.F_EDGES, N_F=FC.N_F)
    try:
        FC.F_EDGES, FC.N_F = JC.F35, len(JC.F35) - 1
        ref = FK.truth_shapes(fpr["meanE_main"][list(range(5, 15)) if not JC.SMOKE else [10]], gridobj.x)
        mine = truth_shapes(fpr["meanE_main"][list(range(5, 15)) if not JC.SMOKE else [10]], gridobj.x, JC.F35)
    finally:
        FC.F_EDGES, FC.N_F = saved["F_EDGES"], saved["N_F"]
    timing["truth_shapes_identity_vs_f_kernels"] = float(np.max(np.abs(mine - ref)))
    JC.write_json(JC.RESULTS / "basis.json", timing)
    return timing
