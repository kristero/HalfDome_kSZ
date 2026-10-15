"""Excision and offset-template geometry for Test A (validated against Monte Carlo in tests/).

All lengths in one consistent unit (physical Mpc or R200c units); spheres are centred on the central
(origin) unless an offset is given.

  lens_volume(a, b, d)                 exact volume common to two balls (containment and disjoint cases)
  ball_shell_volume(r1, r2, d, rho)    exact volume of ball(offset d, radius rho) inside the shell [r1, r2]
  union_shell_fraction(edges, offsets, radii)
        fraction of each shell [edges[s], edges[s+1]] covered by the UNION of balls. Exact (analytic) when
        the active balls in a shell are pairwise disjoint; otherwise piecewise Gauss-Legendre in u = r^3
        between the radial breakpoints |d_k -+ rho_k|, with the covered solid-angle fraction at each
        radius from a Fibonacci sphere of directions (error ~1e-3 at n_dir = 4096).
  offset_shell_average(G, edges, d, cut)
        shell average of a radial profile F centred at distance d from the origin and truncated at `cut`:
        <F>(r; d) = [G(min(r + d, cut)) - G(min(|r - d|, cut))] / (2 r d),  G(s) = int_0^s s' F(s') ds'
        (exact angular average), then volume-averaged over each shell by piecewise Gauss-Legendre.
  make_G(F, cut)                       G(s) on a fine log grid (power-law start below cut * 1e-7)
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def shell_volume(r1, r2):
    return 4 * np.pi / 3 * (np.asarray(r2, float) ** 3 - np.asarray(r1, float) ** 3)


def lens_volume(a, b, d):
    """Exact volume common to balls of radii a and b whose centres are d apart."""
    a, b, d = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(d, float))
    out = np.zeros(d.shape)
    contained = d <= np.abs(a - b)
    out[contained] = 4 * np.pi / 3 * np.minimum(a, b)[contained] ** 3
    lens = (d < a + b) & ~contained
    al, bl, dl = a[lens], b[lens], d[lens]
    out[lens] = np.pi * (al + bl - dl) ** 2 * (dl ** 2 + 2 * dl * (al + bl) - 3 * (al - bl) ** 2) / (12 * dl)
    return out


def ball_shell_volume(r1, r2, d, rho):
    """Volume of ball(centre at distance d, radius rho) inside the origin-centred shell [r1, r2]."""
    return lens_volume(r2, rho, d) - lens_volume(r1, rho, d)


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    z = 1 - 2 * i / n
    phi = np.pi * (3 - np.sqrt(5)) * i
    s = np.sqrt(1 - z * z)
    return np.column_stack((s * np.cos(phi), s * np.sin(phi), z))


_DIRECTIONS = {}


def _directions(n):
    if n not in _DIRECTIONS:
        _DIRECTIONS[n] = fibonacci_sphere(n)
    return _DIRECTIONS[n]


def covered_fraction(r, centres, radii, n_dir=4096):
    """Fraction of the sphere of radius r (origin-centred) inside the union of balls."""
    centres, radii = np.asarray(centres, float).reshape(-1, 3), np.asarray(radii, float)
    if r <= 0:
        return float(np.any(np.linalg.norm(centres, axis=1) < radii))
    directions = _directions(n_dir)
    inside = np.zeros(len(directions), bool)
    for c, rho in zip(centres, radii):
        d = float(np.linalg.norm(c))
        if d == 0:
            if r < rho:
                return 1.0
            continue
        cos_limit = (r * r + d * d - rho * rho) / (2 * r * d)
        if cos_limit >= 1:
            continue
        if cos_limit <= -1:
            return 1.0
        inside |= directions @ (c / d) > cos_limit
    return float(inside.mean())


def union_shell_fraction(edges, centres, radii, n_dir=4096, n_gl=12):
    """Covered (excised) volume fraction of every shell for the union of balls (see module docstring)."""
    edges = np.asarray(edges, float)
    centres, radii = np.asarray(centres, float).reshape(-1, 3), np.asarray(radii, float)
    out = np.zeros(len(edges) - 1)
    if not len(radii):
        return out
    d = np.linalg.norm(centres, axis=1)
    lo_reach, hi_reach = np.maximum(d - radii, 0), d + radii
    pair_overlap = np.linalg.norm(centres[:, None] - centres[None], axis=-1) < radii[:, None] + radii[None]
    np.fill_diagonal(pair_overlap, False)
    nodes, weights = leggauss(n_gl)
    for s in range(len(edges) - 1):
        r1, r2 = edges[s], edges[s + 1]
        active = np.flatnonzero((lo_reach < r2) & (hi_reach > r1) & (radii > 0))
        if not len(active):
            continue
        volume = shell_volume(r1, r2)
        if len(active) == 1 or not pair_overlap[np.ix_(active, active)].any():
            out[s] = min(float(np.sum(ball_shell_volume(r1, r2, d[active], radii[active]))) / volume, 1.0)
            continue
        cuts = np.r_[r1, r2, lo_reach[active], hi_reach[active]]
        cuts = np.unique(cuts[(cuts >= r1) & (cuts <= r2)])
        covered = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            u0, u1 = a ** 3, b ** 3
            u = u0 + (u1 - u0) * (nodes + 1) / 2
            f = np.array([covered_fraction(np.cbrt(x), centres[active], radii[active], n_dir) for x in u])
            covered += np.sum(f * weights) * (u1 - u0) / 2
        out[s] = covered / (r2 ** 3 - r1 ** 3)
    return out


def make_G(F, cut, n_seg=4000, n_gl=6, start=1e-7):
    """G(s) = int_0^s s' F(s') ds' for 0 <= s <= cut (F integrable as s^p with p > -2 at the origin)."""
    s_edges = np.geomspace(start * cut, cut, n_seg + 1)
    nodes, weights = leggauss(n_gl)
    a, b = s_edges[:-1, None], s_edges[1:, None]
    x = a + (b - a) * (nodes + 1) / 2
    seg = np.sum(x * F(x) * weights, axis=1) * (b - a)[:, 0] / 2
    s0 = s_edges[0]
    f1, f2 = F(np.array([s0]))[0], F(np.array([2 * s0]))[0]
    if f1 > 0 and f2 > 0:
        p = np.log(f2 / f1) / np.log(2)
        g0 = s0 * s0 * f1 / (2 + p)
    else:
        # a template with zero amplitude (e.g. a satellite with no bound gas) has no inner power-law term;
        # log(0/0) would otherwise make G, and every shell average built from it, NaN
        p, g0 = 0.0, 0.0
    values = g0 + np.r_[0, np.cumsum(seg)]
    log_s = np.log(s_edges)

    def G(s):
        s = np.minimum(np.asarray(s, float), cut)
        out = np.empty_like(s)
        small = s < s0
        out[small] = g0 * (np.maximum(s[small], 0) / s0) ** (2 + p)
        out[~small] = np.interp(np.log(s[~small]), log_s, values)
        return out
    G.cut = cut
    return G


def offset_profile_average(G, r, d, cut):
    """Angular average over the sphere of radius r of F(|x - d z|), F truncated at `cut` (d > 0)."""
    r = np.asarray(r, float)
    return (G(np.minimum(r + d, cut)) - G(np.minimum(np.abs(r - d), cut))) / (2 * r * d)


def offset_shell_average(G, edges, d, cut, n_gl=16):
    """Volume average over each shell of the angular average above (piecewise at d, |cut - d|, cut + d)."""
    edges = np.asarray(edges, float)
    nodes, weights = leggauss(n_gl)
    out = np.zeros(len(edges) - 1)
    for s in range(len(edges) - 1):
        r1, r2 = edges[s], edges[s + 1]
        cuts = np.r_[r1, r2, d, abs(cut - d), cut + d]
        cuts = np.unique(cuts[(cuts >= r1) & (cuts <= r2)])
        total = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            r = a + (b - a) * (nodes + 1) / 2
            total += np.sum(r * r * offset_profile_average(G, r, d, cut) * weights) * (b - a) / 2
        out[s] = 3 * total / (r2 ** 3 - r1 ** 3)
    return out


# ------------------------------------------------------------------ Monte Carlo references (tests only)
def mc_union_shell_fraction(edges, centres, radii, n=400000, seed=1):
    rng = np.random.default_rng(seed)
    centres, radii = np.asarray(centres, float).reshape(-1, 3), np.asarray(radii, float)
    out = np.zeros(len(edges) - 1)
    for s in range(len(edges) - 1):
        r = np.cbrt(rng.uniform(edges[s] ** 3, edges[s + 1] ** 3, n))
        v = rng.normal(size=(n, 3))
        p = v / np.linalg.norm(v, axis=1)[:, None] * r[:, None]
        inside = np.zeros(n, bool)
        for c, rho in zip(centres, radii):
            inside |= np.einsum('ij,ij->i', p - c, p - c) < rho * rho
        out[s] = inside.mean()
    return out


def mc_offset_shell_average(F, edges, d, cut, n=400000, seed=2):
    rng = np.random.default_rng(seed)
    out = np.zeros(len(edges) - 1)
    for s in range(len(edges) - 1):
        r = np.cbrt(rng.uniform(edges[s] ** 3, edges[s + 1] ** 3, n))
        v = rng.normal(size=(n, 3))
        p = v / np.linalg.norm(v, axis=1)[:, None] * r[:, None]
        q = np.linalg.norm(p - np.array([0, 0, d]), axis=1)
        out[s] = np.mean(np.where(q < cut, F(np.maximum(q, 1e-300)), 0.0))
    return out
