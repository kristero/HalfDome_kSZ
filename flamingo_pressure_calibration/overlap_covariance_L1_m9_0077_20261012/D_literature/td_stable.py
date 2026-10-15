"""Cancellation-free radial weights for the kernels/ engine (Test D post-processing; engine file untouched).

The frozen kernel lightcone_analytic.linear_radial_kernel integrates 4 pi int x^2 F(x) sinc(q x) dx for a
piecewise-linear F with an 8-term moment series for |q| < 0.2 and the closed-form antiderivatives for
|q| >= 0.2 (kernels/engine.radial_weights adds a guard for small q on long grids).  For q >= 0.2 the closed
form is also used on the innermost nodes, where q x << 1: the terms of i2 = ... + 2 (cos q hi - cos q lo)/q^3
cancel to ~1e-8 of their size, which is harmless for profiles with a shallow core (B12/B16, gamma <= 0.3) but
not for the steep unresolved cusps that free-gamma gNFW fits produce in a few Test D subsets (F(1e-5 R200c) up
to 1e5-1e6 x the reference).

stable_radial_weights(x, q) uses, per interval, the same moment series (13 terms) wherever q * x_hi <= 1.5 and
the frozen closed form elsewhere -- the engine's own guard rule applied to every q.  It returns the frozen
kernel's shape q.shape + (len(x),).  self_test() checks it against the frozen kernel on regular profiles and
against 50-digit mpmath on a cusp.
"""
from __future__ import annotations

import math

import numpy as np

N_TERMS = 13
QX_MAX = 1.5


def _series_parts(lo, hi):
    """[N_TERMS, n_int] left/right moment integrals (q independent), as the frozen series."""
    dx = hi - lo
    left = np.empty((N_TERMS, len(lo)))
    right = np.empty((N_TERMS, len(lo)))
    for m in range(N_TERMS):
        power = 2 * m + 2
        a = (hi**(power + 1) - lo**(power + 1)) / (power + 1)
        b = (hi**(power + 2) - lo**(power + 2)) / (power + 2)
        left[m] = (hi * a - b) / dx
        right[m] = (b - lo * a) / dx
    return left, right


def stable_radial_weights(x, q):
    x = np.asarray(x, float)
    q = np.asarray(q, float)
    flat = q.ravel()
    lo, hi = x[:-1], x[1:]
    dx = hi - lo
    aq = np.abs(flat)
    series = aq[:, None] * hi[None, :] <= QX_MAX                      # [nq, n_int]
    left, right = _series_parts(lo, hi)
    coeff = np.stack([(-1.0)**m * aq**(2 * m) / float(math.factorial(2 * m + 1)) for m in range(N_TERMS)], axis=1)
    sl_ = coeff @ left                                                 # [nq, n_int]
    sr_ = coeff @ right
    W = np.zeros((len(flat), len(x)))
    cl_ = np.zeros_like(sl_)
    cr_ = np.zeros_like(sr_)
    need = ~series
    if need.any():
        rows = np.flatnonzero(need.any(axis=1))
        ql = aq[rows, None]
        sL, sH = np.sin(ql * lo), np.sin(ql * hi)
        cL, cH = np.cos(ql * lo), np.cos(ql * hi)
        i1 = (sH - sL) / ql**2 - (hi * cH - lo * cL) / ql
        i2 = -(hi**2 * cH - lo**2 * cL) / ql + 2 * (hi * sH - lo * sL) / ql**2 + 2 * (cH - cL) / ql**3
        cl_[rows] = (hi * i1 - i2) / dx / ql
        cr_[rows] = (i2 - lo * i1) / dx / ql
    lw = np.where(series, sl_, cl_)
    rw = np.where(series, sr_, cr_)
    W[:, :-1] += lw
    W[:, 1:] += rw
    W *= 4 * np.pi
    return W.reshape(q.shape + (len(x),))


def mp_weights(x, q, dps=50):
    """Reference weights with mpmath (closed form at dps digits: no cancellation issue)."""
    import mpmath as mp
    mp.mp.dps = dps
    out = np.zeros((len(q), len(x)))
    for i, qq in enumerate(q):
        Q = mp.mpf(qq)
        for j in range(len(x) - 1):
            L, H = mp.mpf(x[j]), mp.mpf(x[j + 1])
            D = H - L
            if Q == 0:
                a = (H**3 - L**3) / 3
                b = (H**4 - L**4) / 4
                lw, rw = (H * a - b) / D, (b - L * a) / D
            else:
                sL, sH, cL, cH = mp.sin(Q * L), mp.sin(Q * H), mp.cos(Q * L), mp.cos(Q * H)
                i1 = (sH - sL) / Q**2 - (H * cH - L * cL) / Q
                i2 = -(H**2 * cH - L**2 * cL) / Q + 2 * (H * sH - L * sL) / Q**2 + 2 * (cH - cL) / Q**3
                lw, rw = (H * i1 - i2) / D / Q, (i2 - L * i1) / D / Q
            out[i, j] += float(4 * mp.pi * lw)
            out[i, j + 1] += float(4 * mp.pi * rw)
    return out


def self_test(E, grid, cusp_table, regular_table):
    """Frozen vs stable on a regular profile (all q) and stable/frozen vs mpmath on a cusp (innermost nodes)."""
    q = np.geomspace(1e-3, 1e3, 121)
    out = {}
    for name, prof in (("regular", regular_table), ("cusp", cusp_table)):
        uf = np.zeros(len(q))
        us = np.zeros(len(q))
        for a, b in grid.segments:
            xs = grid.x[a:b + 1]
            uf += E.radial_weights(xs, q) @ prof[a:b + 1]
            us += stable_radial_weights(xs, q) @ prof[a:b + 1]
        out[f"{name}: max |frozen/stable - 1| over q in [1e-3, 1e3]"] = float(np.max(np.abs(uf / us - 1)))
    # mpmath on the first 60 nodes (where the cancellation lives) at q around the series switch
    xs = grid.x[:60]
    qs = np.array([0.0, 0.1, 0.19, 0.21, 0.23, 0.5, 1.0, 5.0, 50.0])
    ref = mp_weights(xs, qs)
    st = stable_radial_weights(xs, qs)
    fr = E.radial_weights(xs, qs)
    scale = np.abs(ref).max(axis=1, keepdims=True)
    out["innermost 60 nodes: max |stable - mpmath| / max|w|"] = float(np.max(np.abs(st - ref) / scale))
    out["innermost 60 nodes: max |frozen - mpmath| / max|w|"] = float(np.max(np.abs(fr - ref) / scale))
    u_ref = ref @ cusp_table[:60]
    out["cusp, innermost 60 nodes: max |u_stable/u_mpmath - 1|"] = float(np.max(np.abs(st @ cusp_table[:60] / u_ref - 1)))
    out["cusp, innermost 60 nodes: max |u_frozen/u_mpmath - 1|"] = float(np.max(np.abs(fr @ cusp_table[:60] / u_ref - 1)))
    return out
