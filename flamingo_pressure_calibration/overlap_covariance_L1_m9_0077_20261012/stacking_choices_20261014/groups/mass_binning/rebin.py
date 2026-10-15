"""Radial rescaling of per-halo shell profiles (mass-binning group, part 2).

A per-halo profile F_h given as shell averages on source edges e (in units of the halo's own R200c) is moved to target
edges t in units of a common radius Rbar: x' = x s_h with s_h = R200c_h / Rbar.  The cumulative amount
Q_h(u) = sum_j F_j 4pi/3 (e_{j+1}^3 - e_j^3) up to own radius u is interpolated with a monotone PCHIP in ln u
(constant density inside the first shell), so the transfer conserves the integral and keeps F >= 0.
Target shells that reach beyond the source coverage (4 R200c_h) are NaN.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def cumulative(F, e):
    V = 4 * np.pi / 3 * np.diff(e**3)
    Fz = np.where(np.isfinite(F), F, 0.0)
    return np.concatenate([np.zeros(F.shape[:-1] + (1,)), np.cumsum(Fz * V, axis=-1)], axis=-1)


def rescale(F, e_src, s, t_edges):
    """F [n, S_src] (own units), s [n] = R_h / Rbar -> [n, S_tgt] shell averages on t_edges (Rbar units)."""
    F = np.asarray(F, float)
    e_src = np.asarray(e_src, float)
    t = np.asarray(t_edges, float)
    Vt = 4 * np.pi / 3 * np.diff(t**3)
    Q = cumulative(F, e_src)                     # [n, S+1]
    out = np.full((F.shape[0], len(t) - 1), np.nan)
    lx = np.log(e_src[1:])
    for h in range(F.shape[0]):
        u = t / s[h]                             # target edges in own units
        q = np.full(len(t), np.nan)
        inside = u <= e_src[-1] * (1 + 1e-12)
        f = PchipInterpolator(lx, Q[h, 1:], extrapolate=False)
        small = u < e_src[1]
        mid = inside & ~small
        q[mid] = f(np.log(np.clip(u[mid], e_src[1], e_src[-1])))
        q[small] = Q[h, 1] * (u[small] / e_src[1])**3
        out[h] = s[h]**3 * np.diff(q) / Vt
    return out
