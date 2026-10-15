"""F1a: exact per-shell basis kernels of the frozen lightcone halo model (kernels/engine.py, read-only).

In every calibrated catalogue cell (z, M) a profile is written as F(x) = sum_s F_s phi_s(x), with
phi_s = g_b(x) 1[x in shell s] / <g_b>_s (unit shell average; g_b a reference shape of the cell's mass
bin).  Every engine observable is then an exact linear (means, the 2h window W) or quadratic (1h,
DM variance) form in the shell averages F_s, e.g.
    C^1h(ell) = sum_b sum_st K_b[s, t, ell] <F_s F_t>_b + fallback,
    C^2h(ell) = sum_z P(ell, z)/V_z [W_fb(z, ell) + sum_bs F_bs W_bs(z, ell)]^2,
so per-halo second moments give the 1h terms with halo-to-halo scatter exactly (the mean over the
bin's haloes of |y_ell,i|^2).  The loop body mirrors engine._evaluate_shell line by line (same
geometry, bias, P_lin, redshift response, radial weights, DM scaling and catalogue velocities).

Reference sets
  V (validation): g_b = the kernels/ P4 and E4 fits on the grid of kernels/jacobians.py, extent 4R.
     With F_s = <g_b>_s the in-basis observables must equal the engine's (gate in f1_validate.py).
  T (production): g_b = log-log interpolation of the bin's measured mean profile (main selection)
     through the F-shell centres (power-law ends: inner slope clipped to [-1, 0], outer slope <= -1
     beyond the last centre); 42 shells to 12 R200c.  Only the within-shell shape comes from g_b: every
     basis function has unit shell average, so shell averages are represented exactly.
Fallback cells (outside the calibrated support): published B12/B16 at the cell mass, cut at 4R, exactly
as engine.Profile; their contributions and per-(engine bin, shell) importance pieces are stored.
Output: products/basis_T.npz, products/basis_V.npz.
"""
from __future__ import annotations

import multiprocessing
import resource
import time

import f_common as FC
import numpy as np

E = FC.E
A = E.A
BATCH = 24                      # cells per column batch (memory bound)
_JOB: dict = {}


# ----------------------------------------------------------------------------- grid
def basis_grid():
    """kernels/jacobians.jacobian_grid() nodes (<= 4R, identical) + log nodes 4..12R; extension breaks."""
    base = E.make_grid(4., breaks=np.unique(np.r_[FC.X200[1:], E.SHELL_EDGES[1:]]))
    n_ext = int(np.ceil(np.log(FC.X_END / 4.) / E.LOG_STEP)) + 1
    ext = np.geomspace(4., FC.X_END, n_ext)
    grid = E.Grid(np.r_[base.x, ext[1:]], breaks=np.unique(np.r_[base.breaks, FC.EXT_EDGES]))
    assert np.array_equal(grid.x[:len(base.x)], base.x)
    mids = np.array([.5 * (grid.x[a] + grid.x[b]) for a, b in grid.segments])
    seg_shell = np.searchsorted(FC.BASIS_EDGES, mids) - 1
    n_seg_4 = int(np.sum(grid.x[[b for _, b in grid.segments]] <= 4. * (1 + 1e-12)))
    assert seg_shell[n_seg_4 - 1] == FC.N_F - 1 and seg_shell[n_seg_4] == FC.N_F
    return base, grid, seg_shell, n_seg_4


def segment_volumes(grid, profile):
    """4pi int x^2 F dx per segment (piecewise-linear F on the nodes): [..., segment]."""
    out = []
    for a, b in grid.segments:
        out.append(profile[..., a:b + 1] @ A.linear_radial_kernel(grid.x[a:b + 1], np.zeros(1))[0])
    return np.stack(out, axis=-1)


def shell_averages(grid, seg_shell, profile, n_shell):
    """Shell averages of the node-interpolated profile: segment integrals summed per shell / V_s."""
    seg = segment_volumes(grid, profile)
    out = np.zeros(profile.shape[:-1] + (n_shell,))
    for s in range(seg.shape[-1]):
        if seg_shell[s] < n_shell:
            out[..., seg_shell[s]] += seg[..., s]
    return out / FC.SHELL_VOLUME[:n_shell]


def truth_shapes(mean, x):
    """Reference shapes g_b(x) [bin, node] from shell means [bin, 35]: log-log interpolation through
    effective shell centres (geometric mean; 0.716 x 0.025 for the first shell), power-law ends."""
    edges = FC.F_EDGES
    centre = np.sqrt(np.maximum(edges[:-1], 1e-300) * edges[1:])
    centre[0] = 0.716 * edges[1]
    lc = np.log(centre)
    lx = np.log(x)
    out = np.zeros((len(mean), len(x)))
    for i, m in enumerate(mean):
        ok = np.isfinite(m) & (m > 0)
        lm = np.interp(lc, lc[ok], np.log(m[ok]))            # fill any non-positive shell from neighbours
        g = np.interp(lx, lc, lm)
        lo_slope = float(np.clip((lm[1] - lm[0]) / (lc[1] - lc[0]), -1.0, 0.0))
        hi_slope = min((lm[-1] - lm[-2]) / (lc[-1] - lc[-2]), -1.0)
        g = np.where(lx < lc[0], lm[0] + lo_slope * (lx - lc[0]), g)
        g = np.where(lx > lc[-1], lm[-1] + hi_slope * (lx - lc[-1]), g)
        out[i] = np.exp(g)
    return out


# ----------------------------------------------------------------------------- per-shell evaluation
def _bmm(w, a, b):
    """sum_m w_m a[m, s, l] b[m, t, l] -> [s, t, l] via batched BLAS matmul over l."""
    aw = (a * w[:, None, None]).transpose(2, 1, 0)          # [l, s, m]
    return np.matmul(aw, b.transpose(2, 0, 1)).transpose(1, 2, 0)


def _shell(j, acc):
    job = _JOB
    sky, grid, ell = job["sky"], job["grid"], job["ell"]
    x = grid.x
    index = np.flatnonzero(sky.counts[j] > 0)
    if not len(index):
        return None
    nl = len(ell)
    zz = sky.z[j]
    mass = sky.mass[index]
    N = sky.counts[j, index].astype(float)
    ing = sky.ingredients
    da, radius, y_amp, tau_amp = ing.geometry(zz, mass)
    bias = ing.bias(zz, mass)
    pk = ing.pk(zz, ell)
    vol = sky.volume[j]
    q_all = np.c_[np.zeros(len(mass)), radius[:, None] / da * (ell[None] + .5)]
    use_dm = zz <= 1.
    dm_scale = 1. / (E.DM_FACTOR * (1 + zz))
    v2 = sky.v2_sum[j, index] / E.C_KMS**2
    s2_lin = (sky.sigma_v_lin[j] / E.C_KMS)**2
    column_scale = tau_amp * (da / radius)**2 * dm_scale
    area_scale = N / (4 * np.pi) * (radius / da)**2
    safe, valid = job["profile"].bins(mass)
    cal, fb = np.flatnonzero(valid), np.flatnonzero(~valid)
    cidx = job["cal_index"][safe[cal]]
    evolution = {f: A.published_profile(f, mass, zz, x) / A.published_profile(f, mass, 0., x) for f in (0, 1)}
    sets = job["sets"]
    prof = {(r, f): job["g"][r][f][cidx] * evolution[f][cal] for r in sets for f in (0, 1)}
    fbprof = {f: A.published_profile(f, mass[fb], 0., x) * evolution[f][fb] for f in (0, 1)}
    pieces = {(r, f): np.zeros((len(cal), job["n_shell"][r], nl + 1)) for r in sets for f in (0, 1)}
    fbp = {f: np.zeros((len(fb), FC.N_F, nl + 1)) for f in (0, 1)}
    for s_i, (a, b) in enumerate(grid.segments):
        sh = job["seg_shell"][s_i]
        weights = E.radial_weights(x[a:b + 1], q_all)                      # [m, ell+1, n]
        wc = weights[cal]
        for r in sets:
            if s_i < job["n_seg"][r]:
                for f in (0, 1):
                    pieces[(r, f)][:, sh] += np.einsum("mln,mn->ml", wc, prof[(r, f)][:, a:b + 1])
        if s_i < job["n_seg_4"] and len(fb):
            wf = weights[fb]
            for f in (0, 1):
                fbp[f][:, sh] += np.einsum("mln,mn->ml", wf, fbprof[f][:, a:b + 1])
    out = {"j": j, "W": {}}
    n4 = N / (4 * np.pi)
    nb4 = N * bias / (4 * np.pi)
    # ---- calibrated cells, per reference set
    for r in sets:
        gp, ge = job["gbar"][r]
        Bp = y_amp[cal, None, None] * pieces[(r, 0)] / gp[cidx][:, :, None]
        Bt = tau_amp[cal, None, None] * pieces[(r, 1)] / ge[cidx][:, :, None]
        a_r = acc[r]
        Wy = np.zeros((FC.N_CAL, job["n_shell"][r], nl))
        Wt = np.zeros_like(Wy)
        for c in np.unique(cidx):
            m = cidx == c
            rows = cal[m]
            bp, bt = Bp[m, :, 1:], Bt[m, :, 1:]
            a_r["K_yy1h"][c] += _bmm(n4[rows], bp, bp)
            a_r["K_ksz1h"][c] += _bmm(v2[rows] / (4 * np.pi), bt, bt)
            a_r["mean_y"][c] += n4[rows] @ Bp[m, :, 0]
            a_r["mean_tau"][c] += n4[rows] @ Bt[m, :, 0]
            if use_dm:
                a_r["K_dm1h"][c] += dm_scale**2 * _bmm(n4[rows], bt, bt)
                a_r["K_ydm1h"][c] += dm_scale * _bmm(n4[rows], bp, bt)
                a_r["dm_mean"][c] += dm_scale * (n4[rows] @ Bt[m, :, 0])
            Wy[c] = np.einsum("m,msl->sl", nb4[rows], bp)
            Wt[c] = np.einsum("m,msl->sl", nb4[rows], bt)
        out["W"][r] = (Wy, Wt)
    # ---- fallback cells (shared by both sets; 2h importance needs each set's total window)
    yP = y_amp[fb, None, None] * fbp[0]                                  # [mf, 35, ell+1]
    yE = tau_amp[fb, None, None] * fbp[1]
    tP, tE = yP.sum(axis=1), yE.sum(axis=1)                               # [mf, ell+1]
    nf = n4[fb]
    a_f = acc["fb"]
    a_f["yy1h"] += nf @ tP[:, 1:]**2
    a_f["ksz1h"] += (v2[fb] / (4 * np.pi)) @ tE[:, 1:]**2
    a_f["mean_y"] += nf @ tP[:, 0]
    a_f["mean_tau"] += nf @ tE[:, 0]
    onehot = np.zeros((len(fb), 16))
    onehot[np.arange(len(fb)), sky.mass_bin[index][fb]] = 1.
    a_f["J_yy1h"] += 2 * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yP[:, :, 1:], tP[:, 1:], optimize=True)
    a_f["J_ksz1h"] += 2 * np.einsum("mb,m,msl,ml->bsl", onehot, v2[fb] / (4 * np.pi), yE[:, :, 1:], tE[:, 1:],
                                    optimize=True)
    a_f["J_mean_y"] += np.einsum("mb,m,ms->bs", onehot, nf, yP[:, :, 0], optimize=True)
    WfP = nb4[fb] @ tP[:, 1:]
    WfE = nb4[fb] @ tE[:, 1:]
    out["W_fb"] = (WfP, WfE)
    pieceP = np.einsum("mb,m,msl->bsl", onehot, nb4[fb], yP[:, :, 1:], optimize=True)   # per-bin-shell window
    pieceE = np.einsum("mb,m,msl->bsl", onehot, nb4[fb], yE[:, :, 1:], optimize=True)
    for r in sets:
        cP, cE = job["coef"][r]
        WtotP = WfP + np.einsum("cs,csl->l", cP, out["W"][r][0])
        WtotE = WfE + np.einsum("cs,csl->l", cE, out["W"][r][1])
        a_r = acc[r]
        a_r["Jfb_yy2h"] += 2 * (pk / vol * WtotP)[None, None] * pieceP
        a_r["Jfb_ksz2h"] += 2 * (s2_lin * pk / vol * WtotE)[None, None] * pieceE
        if use_dm:
            a_r["Jfb_dm2h"] += 2 * dm_scale**2 * (pk / vol * WtotE)[None, None] * pieceE
            a_r["Jfb_ydm2h_P"] += dm_scale * (pk / vol * WtotE)[None, None] * pieceP
            a_r["Jfb_ydm2h_E"] += dm_scale * (pk / vol * WtotP)[None, None] * pieceE
    if use_dm:
        a_f["dm1h"] += dm_scale**2 * (nf @ tE[:, 1:]**2)
        a_f["ydm1h"] += dm_scale * (nf @ (tP[:, 1:] * tE[:, 1:]))
        a_f["dm_mean"] += dm_scale * (nf @ tE[:, 0])
        a_f["J_dm1h"] += 2 * dm_scale**2 * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yE[:, :, 1:], tE[:, 1:],
                                                     optimize=True)
        a_f["J_ydm1h_P"] += dm_scale * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yP[:, :, 1:], tE[:, 1:], optimize=True)
        a_f["J_ydm1h_E"] += dm_scale * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yE[:, :, 1:], tP[:, 1:], optimize=True)
        a_f["J_dm_mean"] += dm_scale * np.einsum("mb,m,ms->bs", onehot, nf, yE[:, :, 0], optimize=True)
        _columns(job, acc, cal, fb, cidx, prof, fbprof, column_scale, area_scale, onehot)
    return out


def _columns(job, acc, cal, fb, cidx, prof, fbprof, column_scale, area_scale, onehot):
    """DM variance forms K_dmvar[b, s, t] = sum_cells lam (scale)^2 sum_k area_k c_s(k) c_t(k), fallback
    DM variance and its per-(bin, shell) importance pieces."""
    grid = job["grid"]
    A_seg, area = job["A_seg"], job["area"]
    sqrt_area = np.sqrt(area)
    for r in job["sets"]:
        ns = job["n_shell"][r]
        ge = job["gbar"][r][1]
        for start in range(0, len(cal), BATCH):
            sl = slice(start, start + BATCH)
            p = prof[(r, 1)][sl]
            col = np.zeros((p.shape[0], ns, len(area)))
            for s_i, (a, b) in enumerate(grid.segments[:job["n_seg"][r]]):
                col[:, job["seg_shell"][s_i]] += p[:, a:b + 1] @ A_seg[s_i].T
            col /= ge[cidx[sl]][:, :, None]
            cw = col * sqrt_area[None, None, :]
            kv = np.matmul(cw, cw.transpose(0, 2, 1)) * (area_scale[cal][sl] * column_scale[cal][sl]**2)[:, None, None]
            for c in np.unique(cidx[sl]):
                acc[r]["K_dmvar"][c] += kv[cidx[sl] == c].sum(axis=0)
    a_f = acc["fb"]
    for start in range(0, len(fb), BATCH):
        sl = slice(start, start + BATCH)
        p = fbprof[1][sl]
        col = np.zeros((p.shape[0], FC.N_F, len(area)))
        for s_i, (a, b) in enumerate(grid.segments[:job["n_seg_4"]]):
            col[:, job["seg_shell"][s_i]] += p[:, a:b + 1] @ A_seg[s_i].T
        scale = column_scale[fb][sl]
        xb = scale[:, None] * col.sum(axis=1)                               # DM per b node
        lam = area_scale[fb][sl][:, None] * area[None, :]
        a_f["dm_var"] += float(np.sum(lam * xb * xb))
        a_f["J_dm_var"] += 2 * np.einsum("mb,mk,msk->bs", onehot[sl], lam * xb, scale[:, None, None] * col,
                                         optimize=True)


def _chunk(shells):
    job = _JOB
    nl = len(job["ell"])
    acc = {}
    for r in job["sets"]:
        ns = job["n_shell"][r]
        acc[r] = {k: np.zeros((FC.N_CAL, ns, ns, nl)) for k in ("K_yy1h", "K_ksz1h", "K_dm1h", "K_ydm1h")}
        acc[r].update(K_dmvar=np.zeros((FC.N_CAL, ns, ns)), mean_y=np.zeros((FC.N_CAL, ns)),
                      mean_tau=np.zeros((FC.N_CAL, ns)), dm_mean=np.zeros((FC.N_CAL, ns)))
        for k in ("Jfb_yy2h", "Jfb_ksz2h", "Jfb_dm2h", "Jfb_ydm2h_P", "Jfb_ydm2h_E"):
            acc[r][k] = np.zeros((16, FC.N_F, nl))
    acc["fb"] = {k: np.zeros(nl) for k in ("yy1h", "ksz1h", "dm1h", "ydm1h")}
    acc["fb"].update(mean_y=0., mean_tau=0., dm_mean=0., dm_var=0.)
    for k in ("J_yy1h", "J_ksz1h", "J_dm1h", "J_ydm1h_P", "J_ydm1h_E"):
        acc["fb"][k] = np.zeros((16, FC.N_F, nl))
    for k in ("J_mean_y", "J_dm_mean", "J_dm_var"):
        acc["fb"][k] = np.zeros((16, FC.N_F))
    rows = [_shell(j, acc) for j in shells]
    return [r for r in rows if r is not None], acc


def build(sets=("V", "T"), workers=FC.WORKERS, chunks_per_worker=2):
    started = time.perf_counter()
    sky = FC.make_sky()
    ell = FC.ELL
    base, grid, seg_shell, n_seg_4 = basis_grid()
    support = FC.SUPPORT
    profile = E.Profile(0, "bins", support["mass_edges"], np.asarray(support["support"], bool), support["max_logmass"])
    cal_index = np.full(len(support["support"]), -1)
    cal_index[FC.CAL_BINS] = np.arange(FC.N_CAL)
    pr = np.load(FC.PRODUCTS / "profiles.npz")
    g, gbar, coef, n_shell, n_seg = {}, {}, {}, {}, {}
    if "V" in sets:
        gV = [E.pressure_fit(4, support).calibrated(np.array(FC.CAL_BINS), grid.x),
              E.electron_fit(4, support).calibrated(np.array(FC.CAL_BINS), grid.x)]
        gV = [np.where(grid.x[None, :] <= 4. * (1 + 1e-12), v, 0.) for v in gV]
        g["V"] = gV
        gbar["V"] = [shell_averages(grid, seg_shell, v, FC.N_F) for v in gV]
        coef["V"] = gbar["V"]                                                 # the P4/E4 profiles themselves
        n_shell["V"], n_seg["V"] = FC.N_F, n_seg_4
    if "T" in sets:
        mP = pr["meanP_main"][FC.CAL_BINS]
        mE = pr["meanE_main"][FC.CAL_BINS]
        gT = [truth_shapes(mP, grid.x), truth_shapes(mE, grid.x)]
        g["T"] = gT
        gbar["T"] = [shell_averages(grid, seg_shell, v, FC.N_B) for v in gT]
        coef["T"] = [np.pad(mP, ((0, 0), (0, FC.N_B - FC.N_F))), np.pad(mE, ((0, 0), (0, FC.N_B - FC.N_F)))]
        n_shell["T"], n_seg["T"] = FC.N_B, len(grid.segments)
    cs = E.column_setup(grid, FC.X_END, True)
    _JOB.clear()
    _JOB.update(sky=sky, grid=grid, ell=ell, seg_shell=seg_shell, n_seg_4=n_seg_4, profile=profile,
                cal_index=cal_index, sets=list(sets), g=g, gbar=gbar, coef=coef, n_shell=n_shell, n_seg=n_seg,
                A_seg=cs["A_seg"], area=cs["area"])
    chunks = E._balanced_chunks(sky, workers * chunks_per_worker)
    FC.log(f"basis build: sets {sets}, {len(grid.x)} nodes, {len(grid.segments)} segments, {len(cs['b'])} b nodes, "
           f"{len(chunks)} chunks on {workers} workers")
    results = []
    for k, item in enumerate(FC.pool_map(_chunk, chunks, workers)):
        results.append(item)
        FC.log(f"chunk {k + 1}/{len(chunks)}: {time.perf_counter() - started:.0f} s")
    _JOB.clear()
    nz, nl = len(sky.z), len(ell)
    products = {}
    for r in sets:
        acc = None
        for _, a in results:                       # chunk order: deterministic sums
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
    common = dict(ell=ell, z=sky.z, volume=sky.volume, pk=pk, use_dm=sky.z <= 1., dm_scale=1. / (E.DM_FACTOR * (1 + sky.z)),
                  s2_lin=(sky.sigma_v_lin / E.C_KMS)**2, W_fb_y=WfP, W_fb_tau=WfE, grid_x=grid.x, grid_breaks=grid.breaks,
                  seg_shell=seg_shell, n_seg_4=n_seg_4, basis_edges=FC.BASIS_EDGES, f_edges=FC.F_EDGES,
                  shell_volume=FC.SHELL_VOLUME, cal_bins=np.array(FC.CAL_BINS),
                  catalogue_haloes_per_engine_bin=np.bincount(sky.mass_bin, weights=sky.counts.sum(axis=0), minlength=16),
                  catalogue_haloes_per_engine_bin_z_le_1=np.bincount(
                      sky.mass_bin, weights=sky.counts[sky.z <= 1].sum(axis=0), minlength=16))
    common.update({f"fb_{k}": v for k, v in fbacc.items()})
    timing = {"wall_seconds": time.perf_counter() - started, "workers": workers, "chunks": len(chunks),
              "nodes": len(grid.x), "segments": len(grid.segments), "b_nodes": len(cs["b"]),
              "peak_rss_parent_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "peak_rss_largest_child_GiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2}
    for r in sets:
        out = dict(common)
        out.update(products[r])
        out["timing_json"] = np.array(FC.json.dumps(timing))
        np.savez_compressed(FC.WORK / f"basis_{r}.npz", **out)
    FC.log(f"basis build done: {timing}")
    return timing


if __name__ == "__main__":
    import sys
    build(sets=tuple(sys.argv[1:]) if len(sys.argv) > 1 else ("V", "T"))
