"""H1: Test F's in-basis observable kernels on the fine grid (67 X200 shells to 4 R200c + 8 extension shells to 12 R200c).

The per-redshift-shell kernel code is Test F's (f_kernels._chunk / _shell / _columns, imported read-only and run in
this process with the fine grid on the f_common module object, h_common).  This module only sets up the job:
reference shapes g_b(x) (the within-shell shape of each basis function phi_s = g_b 1[x in s] / <g_b>_s; shell
averages are represented exactly whatever g_b is) and the reference coefficients used by the fallback 2-halo
Jacobian pieces (they need the total 2-halo window of a reference profile).

Basis sets (all on the same 75 fine shells and the same radial nodes):
  FP  g_b = Test F's own production shapes (log-log interpolation through the F-shell centres of the all-gas F-grid
      means, f_kernels.truth_shapes on the F grid); coefficients = Test F's all-gas means mapped to the fine shells
      through g_b.  With these, every in-basis observable equals Test F's to rounding: the parity set.
  HA  g_b = log-log interpolation through the fine-shell centres of the A2-ii fine means; coefficients = the A2-ii
      fine means (production basis of the painting target).
  HM  the same for the all-gas fine means (production basis of the comparison).
Output: <F_RESULTS>/work/basis_{FP,HA,HM}.npz in Test F's basis format (f_basis.Basis(name) loads them).
"""
from __future__ import annotations

import resource
import time

import h_common as HC
import numpy as np

FC, FK, E = HC.FC, HC.FK, HC.E


def truth_shapes(mean, x, edges):
    """f_kernels.truth_shapes with the shell edges as an argument (identical code; checked against Test F's function
    on the F grid in check_truth_shapes): log-log interpolation through effective shell centres (geometric mean;
    0.716 x the outer edge for the first shell), inner slope clipped to [-1, 0], outer slope <= -1."""
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


def check_truth_shapes(mean35, x):
    """Max |difference| between this module's truth_shapes and f_kernels.truth_shapes on Test F's grid (must be 0)."""
    with HC.f_grid():
        ref = FK.truth_shapes(mean35, x)
    return float(np.max(np.abs(truth_shapes(mean35, x, HC.F35) - ref)))


def grid():
    """(base, grid, seg_shell, n_seg_4) of f_kernels.basis_grid on the fine grid."""
    return FK.basis_grid()


def fine_averages(g, gridobj, seg_shell):
    """Fine-shell averages <g>_s [.., 75] of node profiles g [.., nodes]."""
    return FK.shell_averages(gridobj, seg_shell, g, HC.N_FINE)


def fp_ratios(gF, gridobj, seg_shell):
    """r[c, s] = <g>_s / <g>_S(s): the share of an F-shell average carried by its fine shell s (F-shell profile
    F_S phi_S(x) = sum_{s in S} F_S r_s phi_s(x) exactly); extension shells map to F's extension shells."""
    gbar = fine_averages(gF, gridobj, seg_shell)                        # [nb, 75]
    num = np.add.reduceat(gbar * HC.VOL_FINE, HC.IDX42[:-1], axis=-1)   # F-basis-shell integrals [nb, 42]
    den = np.add.reduceat(HC.VOL_FINE, HC.IDX42[:-1])
    gF42 = num / den
    return gbar / gF42[:, HC.FINE_OF_B42], gbar, gF42


def build(specs: dict, workers=None, chunks_per_worker=2):
    """specs: {name: dict(gP [nb, nodes], gE, coefP [nb, 75], coefE [nb, 75])}; writes basis_<name>.npz."""
    workers = HC.WORKERS if workers is None else workers
    started = time.perf_counter()
    sky = FC.make_sky()
    ell = FC.ELL
    base, gridobj, seg_shell, n_seg_4 = grid()
    support = FC.SUPPORT
    profile = E.Profile(0, "bins", support["mass_edges"], np.asarray(support["support"], bool), support["max_logmass"])
    cal_index = np.full(len(support["support"]), -1)
    cal_index[FC.CAL_BINS] = np.arange(FC.N_CAL)
    sets = list(specs)
    g, gbar, coef, n_shell, n_seg = {}, {}, {}, {}, {}
    for r, sp in specs.items():
        g[r] = [np.asarray(sp["gP"], float), np.asarray(sp["gE"], float)]
        gbar[r] = [fine_averages(v, gridobj, seg_shell) for v in g[r]]
        coef[r] = [np.asarray(sp["coefP"], float), np.asarray(sp["coefE"], float)]
        n_shell[r], n_seg[r] = HC.N_FINE, len(gridobj.segments)
    cs = E.column_setup(gridobj, FC.X_END, True)
    FK._JOB.clear()
    FK._JOB.update(sky=sky, grid=gridobj, ell=ell, seg_shell=seg_shell, n_seg_4=n_seg_4, profile=profile,
                   cal_index=cal_index, sets=sets, g=g, gbar=gbar, coef=coef, n_shell=n_shell, n_seg=n_seg,
                   A_seg=cs["A_seg"], area=cs["area"])
    chunks = E._balanced_chunks(sky, workers * chunks_per_worker)
    HC.log(f"fine basis build: sets {sets}, {len(gridobj.x)} nodes, {len(gridobj.segments)} segments, "
           f"{len(cs['b'])} b nodes, {len(chunks)} chunks on {workers} workers")
    results = []
    for k, item in enumerate(FC.pool_map(FK._chunk, chunks, workers)):
        results.append(item)
        HC.log(f"chunk {k + 1}/{len(chunks)}: {time.perf_counter() - started:.0f} s")
    FK._JOB.clear()
    nz, nl = len(sky.z), len(ell)
    products = {}
    for r in sets:
        acc = None
        for _, a in results:
            acc = {k: v.copy() for k, v in a[r].items()} if acc is None else {k: acc[k] + a[r][k] for k in acc}
        WP = np.zeros((nz, FC.N_CAL, n_shell[r], nl))
        WE = np.zeros_like(WP)
        for rows, _ in results:
            for row in rows:
                WP[row["j"]], WE[row["j"]] = row["W"][r]
        products[r] = dict(acc, W_y=WP, W_tau=WE, g_P=g[r][0], g_E=g[r][1], gbar_P=gbar[r][0], gbar_E=gbar[r][1],
                           coef_P=coef[r][0], coef_E=coef[r][1], n_shell=n_shell[r], n_seg=n_seg[r])
    fbacc = None
    for _, a in results:
        fbacc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a["fb"].items()} if fbacc is None \
            else {k: fbacc[k] + a["fb"][k] for k in fbacc}
    WfP = np.zeros((nz, nl))
    WfE = np.zeros((nz, nl))
    for rows, _ in results:
        for row in rows:
            WfP[row["j"]], WfE[row["j"]] = row["W_fb"]
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
              "peak_rss_parent_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "peak_rss_largest_child_GiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2}
    for r in sets:
        out = dict(common)
        out.update(products[r])
        out["timing_json"] = np.array(FC.json.dumps(timing))
        np.savez(FC.WORK / f"basis_{r}.npz", **out)
    HC.log(f"fine basis build done: {timing}")
    return timing
