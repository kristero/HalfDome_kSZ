"""Resolution masks generalised to any shell grid (radii234 fit_study.py:60-70).

A halo-shell is *resolved* when (i) its inner edge is at least
max(0.04, 4 softening / R) in units of the normalising radius R, (ii) it holds at
least 20 gas particles (all membership components), and (iii) its inner edge is
at least twice the upper-median smoothing length in that shell.  ``strict``
requires every halo of the bin to be resolved; ``diagnostic`` keeps the
geometric criterion for every halo and asks for >= 200 particles in total and
particles in >= 50% of the haloes.  A bin is *supported* when >= 6 strict shells
have outer edge <= 2 (radii234 convention, no tolerance on the 2).
"""
from __future__ import annotations

import numpy as np

from . import io as aio

DEFAULTS = dict(min_edge=0.04, softening_factor=4.0, min_count=20, hsml_factor=2.0,
                support_outer=2.0, min_supported_shells=6,
                diagnostic_total_count=200, diagnostic_nonzero_fraction=0.5)


def particle_counts(data: dict, rows=None) -> np.ndarray:
    """[halo, shell] particle counts summed over every middle axis (all gas)."""
    values, _ = aio._take_quantity(data, "count", rows)
    return values.reshape(values.shape[0], -1, values.shape[-1]).sum(axis=1) \
        if values.ndim > 2 else values


def resolution_masks(data: dict, rows, **options) -> dict:
    """Masks for the haloes ``rows`` on ``data['edges']``.

    Returns dict(base, resolved[halo, shell], strict, diagnostic, geometric, supported):
    ``base`` is ``strict`` if supported else ``diagnostic`` (as in fit_study).
    """
    o = dict(DEFAULTS, **options)
    rows = np.asarray(rows)
    edges = np.asarray(data["edges"], float)
    x = edges[:-1]
    count = particle_counts(data, rows)
    radius = data["radius_mpc"][rows]
    geometric = x[None, :] >= np.maximum(o["min_edge"], o["softening_factor"] * data["softening_mpc"] / radius[:, None])
    resolved = geometric & (count >= o["min_count"])
    if data.get("hsml_median_x") is not None:
        resolved &= x[None, :] >= o["hsml_factor"] * data["hsml_median_x"][rows]
    strict = np.all(resolved, axis=0)
    diagnostic = (np.all(geometric, axis=0) & (count.sum(axis=0) >= o["diagnostic_total_count"])
                  & (np.mean(count > 0, axis=0) >= o["diagnostic_nonzero_fraction"]))
    supported = int(np.sum(strict & (edges[1:] <= o["support_outer"]))) >= o["min_supported_shells"]
    return {"base": strict if supported else diagnostic, "resolved": resolved, "strict": strict,
            "diagnostic": diagnostic, "geometric": geometric, "supported": bool(supported),
            "strict_shells_to_support_outer": int(np.sum(strict & (edges[1:] <= o["support_outer"]))),
            "resolved_fraction": resolved.mean(axis=0)}


def fit_mask(base: np.ndarray, edges, cutoff: float, mean=None) -> np.ndarray:
    """Fit shells: base mask, outer edge <= cutoff (+1e-12), finite positive mean."""
    edges = np.asarray(edges, float)
    mask = np.asarray(base, bool) & (edges[1:] <= cutoff + 1e-12)
    if mean is not None:
        mask &= np.isfinite(mean) & (mean > 0)
    return mask


def common_mask(*masks) -> np.ndarray:
    out = np.asarray(masks[0], bool).copy()
    for m in masks[1:]:
        out &= np.asarray(m, bool)
    return out
