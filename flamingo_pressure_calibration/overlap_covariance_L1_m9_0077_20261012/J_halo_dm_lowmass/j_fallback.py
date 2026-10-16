"""J stage 'fallback': the published-fallback (B12 pressure / B16 electrons) haloes of each configuration, with the
electron profile cut in 3D at 1, 2 or 4 R200c.

The engine (and f_kernels) paints every lightcone cell outside the calibrated support with the published profile at the
cell's mass, cut at 4 R200c.  Test J needs the same fallback for three supports and three electron cuts, so this module
repeats f_kernels' fallback lines (f_kernels._shell, 'fallback cells' block, and f_kernels._columns, fallback part)
with the cut as a parameter:
  configurations  C14 (fallback = haloes above 10^15.388), C10 (Test H: also every halo below 10^13.5), C0 (all haloes)
  electron cut k  number of fine shells kept (1 R200c: 50, 2 R200c: 60, 4 R200c: 67); pressure always to 4 R200c.
For each (configuration, cut) and lightcone redshift shell z, with the fallback cells m (N_m haloes, bias b_m, summed
catalogue v_LOS^2 v2_m, angular-diameter distance d_A, the cell's y / tau amplitudes) and per-shell transforms
y^P_m,s(ell), y^E_m,s(ell) (Fourier transform at ell of the profile restricted to fine shell s; ell index 0 = the
volume integral), t^X_m = sum_{s < k_X} y^X_m,s:
  yy1h  += sum_m N_m/(4pi) |t^P_m(ell)|^2         ksz1h += sum_m v2_m/(4pi c^2) |t^E_m(ell)|^2
  mean_y += sum_m N_m/(4pi) t^P_m(0)               mean_tau += sum_m N_m/(4pi) t^E_m(0)
  W_fb^X(z, ell) = sum_m N_m b_m/(4pi) t^X_m(ell)   (2-halo window)
  for z <= 1 (s = 1/(DM_FACTOR (1+z)) converts tau to DM in pc cm^-3):
  dm1h += s^2 sum_m N_m/(4pi) |t^E_m|^2, ydm1h += s sum_m N_m/(4pi) t^P_m t^E_m, dm_mean += s sum_m N_m/(4pi) t^E_m(0),
  dm_var += sum_{m, b-node} lambda_m,b DM_m,b^2 with DM_m,b = column_scale_m sum_{s<k} c_m,s(b) (projected column of
  the cut profile at impact parameter b) and lambda_m,b = N_m/(4pi) (R_m/d_A)^2 area_b (expected crossings),
and the per-(engine bin, shell) importance pieces J_* (derivative with respect to a scaling of the fallback profile in one
shell, as f_kernels).  The 2-halo window pieces per (z, engine bin, shell) are kept for configuration C14 (fallback
bins only) so that the fallback 2-halo importance can be formed with any total window.
Parity: configuration C14 at 4 R200c must equal the fb_* totals and the fallback windows that f_kernels accumulated in
basis_JA (results/fallback_parity.json).
Output: <work>/fallback_j.npz (scratch), results/fallback_parity.json.
"""
from __future__ import annotations

import time

import j_common as JC
import numpy as np

FC, FK, E, A = JC.FC, JC.FK, JC.E, JC.A
CONFIGS = ("C14", "C10", "C0")
SUM_KEYS = ("yy1h", "ksz1h", "dm1h", "ydm1h", "mean_y", "mean_tau", "dm_mean", "dm_var")
J_KEYS_L = ("J_yy1h", "J_ksz1h", "J_dm1h", "J_ydm1h_P", "J_ydm1h_E")
J_KEYS_S = ("J_mean_y", "J_dm_mean", "J_dm_var")
_JOB: dict = {}


def _profiles():
    out = {}
    for cfg in CONFIGS:
        mask = np.zeros(JC.N_ENGINE, bool)
        mask[JC.SUPPORT_BINS[cfg]] = True
        out[cfg] = E.Profile(0, "bins", FC.SUPPORT["mass_edges"], mask, FC.SUPPORT["max_logmass"])
    return out


def _new_acc(nl):
    acc = {}
    for cfg in CONFIGS:
        for cut in JC.CUTS:
            a = {k: np.zeros(nl) for k in ("yy1h", "ksz1h", "dm1h", "ydm1h")}
            a.update(mean_y=0., mean_tau=0., dm_mean=0., dm_var=0.)
            for k in J_KEYS_L:
                a[k] = np.zeros((16, JC.N_X, nl))
            for k in J_KEYS_S:
                a[k] = np.zeros((16, JC.N_X))
            acc[(cfg, cut)] = a
    return acc


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
    q_all = np.c_[np.zeros(len(mass)), radius[:, None] / da * (ell[None] + .5)]
    use_dm = zz <= 1.
    dm_scale = 1. / (E.DM_FACTOR * (1 + zz))
    v2 = sky.v2_sum[j, index] / E.C_KMS**2
    column_scale = tau_amp * (da / radius)**2 * dm_scale
    area_scale = N / (4 * np.pi) * (radius / da)**2
    fbsets = {cfg: np.flatnonzero(~job["profiles"][cfg].bins(mass)[1]) for cfg in CONFIGS}
    union = fbsets["C0"]                                              # every cell (C0 has no calibrated bin)
    assert len(union) == len(index)
    evolution = {f: A.published_profile(f, mass, zz, x) / A.published_profile(f, mass, 0., x) for f in (0, 1)}
    fbprof = {f: A.published_profile(f, mass, 0., x) * evolution[f] for f in (0, 1)}
    fbp = {f: np.zeros((len(index), JC.N_X, nl + 1)) for f in (0, 1)}
    for s_i, (a, b) in enumerate(grid.segments):
        if s_i >= job["n_seg_4"]:
            break
        sh = job["seg_shell"][s_i]
        weights = E.radial_weights(x[a:b + 1], q_all)                   # [m, ell+1, n]
        for f in (0, 1):
            fbp[f][:, sh] += np.einsum("mln,mn->ml", weights, fbprof[f][:, a:b + 1])
    out = {"j": j, "W": {}, "pieces": None}
    n4 = N / (4 * np.pi)
    nb4 = N * bias / (4 * np.pi)
    mbin = sky.mass_bin[index]
    for cfg in CONFIGS:
        fb = fbsets[cfg]
        if not len(fb):
            for cut in JC.CUTS:
                out["W"][(cfg, cut)] = (np.zeros(nl), np.zeros(nl))
            continue
        yP = y_amp[fb, None, None] * fbp[0][fb]                          # [mf, 67, ell+1]
        yE = tau_amp[fb, None, None] * fbp[1][fb]
        tP = yP.sum(axis=1)
        nf = n4[fb]
        onehot = np.zeros((len(fb), 16))
        onehot[np.arange(len(fb)), mbin[fb]] = 1.
        if cfg == "C14":
            pieceP = np.einsum("mb,m,msl->bsl", onehot, nb4[fb], yP[:, :, 1:], optimize=True)
            pieceE = np.einsum("mb,m,msl->bsl", onehot, nb4[fb], yE[:, :, 1:], optimize=True)
            out["pieces"] = (pieceP[job["fb_bins"]], pieceE[job["fb_bins"]])
        for cut in JC.CUTS:
            k = JC.K_CUT[cut]
            a_ = acc[(cfg, cut)]
            tE = yE[:, :k].sum(axis=1)
            a_["yy1h"] += nf @ tP[:, 1:]**2
            a_["ksz1h"] += (v2[fb] / (4 * np.pi)) @ tE[:, 1:]**2
            a_["mean_y"] += nf @ tP[:, 0]
            a_["mean_tau"] += nf @ tE[:, 0]
            a_["J_yy1h"] += 2 * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yP[:, :, 1:], tP[:, 1:], optimize=True)
            a_["J_ksz1h"][:, :k] += 2 * np.einsum("mb,m,msl,ml->bsl", onehot, v2[fb] / (4 * np.pi), yE[:, :k, 1:],
                                                  tE[:, 1:], optimize=True)
            a_["J_mean_y"] += np.einsum("mb,m,ms->bs", onehot, nf, yP[:, :, 0], optimize=True)
            out["W"][(cfg, cut)] = (nb4[fb] @ tP[:, 1:], nb4[fb] @ tE[:, 1:])
            if use_dm:
                a_["dm1h"] += dm_scale**2 * (nf @ tE[:, 1:]**2)
                a_["ydm1h"] += dm_scale * (nf @ (tP[:, 1:] * tE[:, 1:]))
                a_["dm_mean"] += dm_scale * (nf @ tE[:, 0])
                a_["J_dm1h"][:, :k] += 2 * dm_scale**2 * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yE[:, :k, 1:],
                                                                   tE[:, 1:], optimize=True)
                a_["J_ydm1h_P"] += dm_scale * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yP[:, :, 1:], tE[:, 1:],
                                                        optimize=True)
                a_["J_ydm1h_E"][:, :k] += dm_scale * np.einsum("mb,m,msl,ml->bsl", onehot, nf, yE[:, :k, 1:],
                                                               tP[:, 1:], optimize=True)
                a_["J_dm_mean"][:, :k] += dm_scale * np.einsum("mb,m,ms->bs", onehot, nf, yE[:, :k, 0], optimize=True)
    # DM variance: projected columns per fine shell (z <= 1), batched over cells as f_kernels._columns
    if use_dm:
        A_seg, area = job["A_seg"], job["area"]
        onehot_all = np.zeros((len(index), 16))
        onehot_all[np.arange(len(index)), mbin] = 1.
        member = {cfg: np.isin(np.arange(len(index)), fbsets[cfg]) for cfg in CONFIGS}
        for start in range(0, len(index), FK.BATCH):
            sl = np.arange(start, min(start + FK.BATCH, len(index)))
            p = fbprof[1][sl]
            col = np.zeros((len(sl), JC.N_X, len(area)))
            for s_i, (a, b) in enumerate(grid.segments[:job["n_seg_4"]]):
                col[:, job["seg_shell"][s_i]] += p[:, a:b + 1] @ A_seg[s_i].T
            for cfg in CONFIGS:
                use = member[cfg][sl]
                if not use.any():
                    continue
                rows_ = sl[use]
                scale = column_scale[rows_]
                lam = area_scale[rows_][:, None] * area[None, :]
                oh = onehot_all[rows_]
                for cut in JC.CUTS:
                    k = JC.K_CUT[cut]
                    c_ = col[use, :k]
                    xb = scale[:, None] * c_.sum(axis=1)
                    a_ = acc[(cfg, cut)]
                    a_["dm_var"] += float(np.sum(lam * xb * xb))
                    a_["J_dm_var"][:, :k] += 2 * np.einsum("mb,mk,msk->bs", oh, lam * xb, scale[:, None, None] * c_,
                                                           optimize=True)
    return out


def _chunk(shells):
    acc = _new_acc(len(_JOB["ell"]))
    rows = [_shell(j, acc) for j in shells]
    return [r for r in rows if r is not None], acc


def build(workers=None, chunks_per_worker=2):
    workers = JC.WORKERS if workers is None else workers
    started = time.perf_counter()
    sky = FC.make_sky()
    ell = FC.ELL
    _, grid, seg_shell, n_seg_4 = FK.basis_grid()
    cs = E.column_setup(grid, FC.X_END, True)
    profiles = _profiles()
    lm = np.log10(sky.mass)
    fb14 = ~profiles["C14"].bins(sky.mass)[1] & (sky.counts.sum(axis=0) > 0)
    fb_bins = np.unique(sky.mass_bin[fb14]).astype(int)
    _JOB.clear()
    _JOB.update(sky=sky, grid=grid, ell=ell, seg_shell=seg_shell, n_seg_4=n_seg_4, A_seg=cs["A_seg"], area=cs["area"],
                profiles=profiles, fb_bins=fb_bins)
    chunks = E._balanced_chunks(sky, workers * chunks_per_worker)
    nz, nl = len(sky.z), len(ell)
    acc = None
    W = {(cfg, cut): (np.zeros((nz, nl)), np.zeros((nz, nl))) for cfg in CONFIGS for cut in JC.CUTS}
    piecesP = np.zeros((nz, len(fb_bins), JC.N_X, nl))
    piecesE = np.zeros_like(piecesP)
    JC.log(f"fallback pass: configurations {CONFIGS}, cuts {JC.CUTS}, C14 fallback engine bins {fb_bins.tolist()}, "
           f"{len(chunks)} chunks on {workers} workers")
    JC.mem("before fork: fallback")
    for k, (rows, a) in enumerate(FC.pool_map(_chunk, chunks, workers)):
        if acc is None:
            acc = {key: {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in d.items()} for key, d in a.items()}
        else:
            for key, d in a.items():
                for kk in d:
                    acc[key][kk] = acc[key][kk] + d[kk]
        for row in rows:
            for key, (wp, we) in row["W"].items():
                W[key][0][row["j"]], W[key][1][row["j"]] = wp, we
            if row["pieces"] is not None:
                piecesP[row["j"]], piecesE[row["j"]] = row["pieces"]
        del rows, a
        JC.log(f"fallback chunk {k + 1}/{len(chunks)}: {time.perf_counter() - started:.0f} s")
    _JOB.clear()
    out = {"fb_bins_C14": fb_bins, "piecesP_C14": piecesP, "piecesE_C14": piecesE}
    for (cfg, cut), d in acc.items():
        for kk, v in d.items():
            out[f"{cfg}|{cut:g}|{kk}"] = np.asarray(v)
        out[f"{cfg}|{cut:g}|W_fb_y"] = W[(cfg, cut)][0]
        out[f"{cfg}|{cut:g}|W_fb_tau"] = W[(cfg, cut)][1]
    np.savez(JC.BIG / "fallback_j.npz", **out)
    timing = {"wall_seconds": time.perf_counter() - started, "workers": workers, "chunks": len(chunks),
              "C14_fallback_bins": fb_bins.tolist(),
              "C14_fallback_haloes": float(sky.counts.sum(axis=0)[fb14].sum()),
              "C14_fallback_logm_min": float(lm[fb14].min()) if fb14.any() else None}
    return timing


def load(cfg, cut, piece_cut=None):
    """Fallback dict for Basis.fb (keys as f_kernels' fb_*), plus 'W_fb_y' / 'W_fb_tau' windows [nz, ell]."""
    z = np.load(JC.BIG / "fallback_j.npz")
    pre = f"{cfg}|{cut:g}|"
    d = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
    for k in ("mean_y", "mean_tau", "dm_mean", "dm_var"):
        d[k] = float(d[k])
    # the pressure side never depends on the electron cut (sanity)
    return d


def jfb2h(B, cut, WP, WE):
    """Fallback 2-halo importance pieces [16, 67, ell] of configuration C14 for total windows WP, WE [z, ell] (the
    f_kernels formulas: Jfb_yy2h = sum_z 2 P/V W_P piece_P, ...), with the electron pieces cut at ``cut``."""
    z = np.load(JC.BIG / "fallback_j.npz")
    bins = z["fb_bins_C14"]
    pP = z["piecesP_C14"]
    pE = np.array(z["piecesE_C14"], copy=True)
    pE[:, :, JC.K_CUT[cut]:] = 0.0
    nl = WP.shape[1]
    out = {k: np.zeros((16, JC.N_X, nl)) for k in ("Jfb_yy2h", "Jfb_ksz2h", "Jfb_dm2h", "Jfb_ydm2h_P", "Jfb_ydm2h_E")}
    pv = B.pv
    dz = B.dmz[:, None]
    out["Jfb_yy2h"][bins] = 2 * np.einsum("zl,zbsl->bsl", pv * WP, pP, optimize=True)
    out["Jfb_ksz2h"][bins] = 2 * np.einsum("zl,zbsl->bsl", B.s2_lin[:, None] * pv * WE, pE, optimize=True)
    out["Jfb_dm2h"][bins] = 2 * np.einsum("zl,zbsl->bsl", dz**2 * pv * WE, pE, optimize=True)
    out["Jfb_ydm2h_P"][bins] = np.einsum("zl,zbsl->bsl", dz * pv * WE, pP, optimize=True)
    out["Jfb_ydm2h_E"][bins] = np.einsum("zl,zbsl->bsl", dz * pv * WP, pE, optimize=True)
    return out


def install(B, cfg, cut, W_total=None):
    """Replace the basis' fallback by configuration ``cfg`` with the electron profiles cut at ``cut``; with W_total =
    (WP, WE) also its 2-halo importance pieces (configuration C14 only)."""
    d = load(cfg, cut)
    B.W_fb_y = d.pop("W_fb_y")
    B.W_fb_tau = d.pop("W_fb_tau")
    B.fb = d
    if W_total is not None:
        assert cfg == "C14"
        B.Jfb2h = jfb2h(B, cut, *W_total)
    return B


def parity():
    """C14 at 4 R200c against the totals f_kernels accumulated in basis_JA / basis_JB."""
    res = {}
    d = load("C14", 4.0)
    for which in ("JA", "JB"):
        z = np.load(JC.BIG / f"basis_{which}.npz")
        worst = {}
        for k in list(d):
            ref = z[f"fb_{k}"] if f"fb_{k}" in z.files else (z[k] if k in ("W_fb_y", "W_fb_tau") else None)
            if ref is None:
                continue
            a, b = np.asarray(d[k], float), np.asarray(ref, float)
            scale = np.max(np.abs(b)) if np.any(b != 0) else 1.0
            worst[k] = float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a)))
        res[which] = {"max_abs_over_max": max(worst.values()), "by_key": worst}
        # 2-halo pieces with the basis' own reference windows (coefficients = the set's mean profiles)
        B = JC.FB.Basis(which)
        WP, WE = B.windows(B.coef_P, B.coef_E)
        J = jfb2h(B, 4.0, WP, WE)
        w2 = {}
        for k, v in J.items():
            ref = z[k]
            scale = np.max(np.abs(ref))
            w2[k] = float(np.max(np.abs(v - ref)) / scale) if scale > 0 else float(np.max(np.abs(v)))
        res[which]["Jfb2h_max_abs_over_max"] = max(w2.values())
        res[which]["Jfb2h_by_key"] = w2
        del B, z
        JC.release()
    return res


def main():
    timing = build()
    par = parity()
    par["timing"] = timing
    par["passed"] = bool(all(par[w]["max_abs_over_max"] < 1e-10 and par[w]["Jfb2h_max_abs_over_max"] < 1e-8
                             for w in ("JA", "JB")))
    JC.write_json(JC.RESULTS / "fallback_parity.json", par)
    JC.log("fallback done; parity", par["passed"], {w: par[w]["max_abs_over_max"] for w in ("JA", "JB")})
    return par
