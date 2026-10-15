"""Geometry helpers for the periodic-boundary audit (2026-10-07).

f_in(c, r, L): fraction of the surface of a sphere of radius r centred at c that lies inside the primary cube
[0, L)^3.  A profile code that does NOT wrap separations (no minimum image, no periodic tree, no ghost particles)
sees, at radius r, only the gas in that fraction of the shell; for a statistically isotropic halo its shell sums are
therefore biased by f_in(r) on average.  Used as a surrogate for the 'no-wrap' failure mode; the surrogate itself is
checked against a particle-level no-wrap recomputation (camels_wrap_particle_test.py).
"""
import numpy as np


def fibonacci_dirs(n=4000):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


DIRS = fibonacci_dirs()


def f_in_sphere(c, r, L, dirs=DIRS):
    """c: (3,) centre, r: (nr,) radii -> (nr,) fraction of each sphere surface inside [0, L)^3."""
    c = np.asarray(c, float)
    r = np.atleast_1d(np.asarray(r, float))
    dmin = np.min(np.minimum(c, L - c))
    out = np.ones(len(r))
    cut = r > dmin
    if cut.any():
        p = c[None, None, :] + r[cut, None, None] * dirs[None, :, :]
        inside = np.all((p >= 0) & (p < L), axis=2)
        out[cut] = inside.mean(1)
    return out


def f_in_shells(c, r_edges, L, nsub=6):
    """Volume-weighted mean of f_in over each shell [r_edges[i], r_edges[i+1]] (log sub-sampling, weight r^3 dlnr)."""
    r_edges = np.asarray(r_edges, float)
    lo = np.maximum(r_edges[:-1], 1e-6 * r_edges[1:])
    hi = r_edges[1:]
    t = (np.arange(nsub) + 0.5) / nsub
    rs = np.exp(np.log(lo)[:, None] + (np.log(hi) - np.log(lo))[:, None] * t[None, :])     # (nb, nsub)
    f = f_in_sphere(c, rs.ravel(), L).reshape(rs.shape)
    w = rs ** 3
    return (f * w).sum(1) / w.sum(1)


def touches_face(c, r, L):
    """True where a sphere of radius r around c crosses a face of [0, L)^3."""
    c = np.atleast_2d(c)
    dmin = np.min(np.minimum(c, L - c), axis=1)
    return np.asarray(r) > dmin
