"""Seeded compound-Poisson Monte Carlo of halo DM sightlines (independent check).

Same model as the engine's FFT PDF and as sensitivity_projects frb_poisson_control.py:
independent (Poisson) halo centres from the HalfDome lightcone count histogram,
z <= 1 cells (z_s = 1), observer-frame DM = column/(1+z), flat-sky impact
parameters uniform in area within the projected extent. For every cell the
number of hits over all rays is Poisson(rays x lambda_cell); hits are assigned
to random rays, impact parameters are drawn uniformly in area and the column
is interpolated from the exact piecewise-linear projection on a fine b grid.
"""
from __future__ import annotations

import sys

import numpy as np

E = sys.modules["kernel_engine"]


def sample(sky, profile, grid, extent, rays=120000, seed=1, nb=600, window=None):
    """DM per ray [pc cm^-3] for one electron profile (hard cut at extent or windowed)."""
    rng = np.random.default_rng(seed)
    dm = np.zeros(rays)
    # Fine area-uniform lookup: u = (b/extent)^2 in [0,1], plus a log-spaced core.
    b_table = extent*np.unique(np.r_[0., np.geomspace(1e-5, 1., nb), np.sqrt(np.linspace(0., 1., nb))])
    b_table[-1] = extent*(1-1e-12)
    operator = E.projection_operator(grid.x, b_table, E.interval_mask(grid, 0., extent))
    base = profile.base(sky.mass, grid.x)
    if window is not None:
        base = base*(window if window.ndim == 2 else window[None])
    events = 0
    for j in np.flatnonzero(sky.z <= 1.):
        index = np.flatnonzero(sky.counts[j] > 0)
        if not len(index):
            continue
        zz = sky.z[j]
        mass = sky.mass[index]
        N = sky.counts[j, index].astype(float)
        da, radius, _, tau_amp = sky.ingredients.geometry(zz, mass)
        evolution = E.A.published_profile(1, mass, zz, grid.x)/E.A.published_profile(1, mass, 0., grid.x)
        sigma = (base[index]*evolution)@operator.T                       # [m, b_table]
        scale = tau_amp*(da/radius)**2/(E.DM_FACTOR*(1+zz))
        lam = N/(4*np.pi)*(radius/da)**2*np.pi*extent**2                 # flat-sky hits per ray
        hits = rng.poisson(rays*lam)
        for m in np.flatnonzero(hits):
            u = rng.random(hits[m])
            b = extent*np.sqrt(u)
            value = scale[m]*np.interp(b, b_table, sigma[m])
            np.add.at(dm, rng.integers(0, rays, hits[m]), value)
            events += hits[m]
    return dm, events


def summary(dm, levels=E.QUANTILES):
    rng = np.random.default_rng(0)
    boot = np.array([np.quantile(dm[rng.integers(0, len(dm), len(dm))], levels) for _ in range(100)])
    return dict(quantiles=np.quantile(dm, levels), quantile_bootstrap_sd=boot.std(axis=0, ddof=1),
                mean=float(dm.mean()), variance=float(dm.var(ddof=1)), rays=len(dm))
