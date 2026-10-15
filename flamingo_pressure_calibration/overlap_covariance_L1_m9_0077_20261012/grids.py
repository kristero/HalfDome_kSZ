"""Radial grids of the overlap/covariance pass (plan Step 1.2) and exact sub-grid rebinning.

X200 (x = r/R200c, 0..4): union of the radii234 36-shell edges (pilot_physics.EDGES), Lee22's 27 log
shells over 0.04-1.34 and {1.5, 1.75, 2.5}. The old edges are an exact (bitwise) subset (gate G2), so
the old 36-shell profiles are recovered with np.add.reduceat. The union contains a few thin slivers
(narrowest 0.24% in radius at 0.3205-0.3212); they are storage shells only - analyses use sub-grids.
X500 (x = r/R500c, 0..6): B24's 31 log edges over 0.01-3 plus 0, 0.03, 0.15, 1, 3.5, 4, 4.5, 5, 5.5, 6.
A particle at exactly the outer edge belongs to the last shell (as radii234).
"""
import numpy as np

import pilot_physics as P

X200 = np.unique(np.r_[P.EDGES, np.geomspace(.04, 1.34, 28), 1.5, 1.75, 2.5])
X500 = np.unique(np.r_[0, np.geomspace(.01, 3, 31), .03, .15, 1, 3.5, 4, 4.5, 5, 5.5, 6])
N200, N500 = len(X200) - 1, len(X500) - 1
OLD_EDGES = P.EDGES
LEE22_EDGES = np.geomspace(.04, 1.34, 28)       # 27 shells over 0.04-1.34 R200c
B24_EDGES = np.geomspace(.01, 3, 31)            # 30 shells over 0.01-3 R500c


def subgrid_index(grid, edges):
    """Indices of `edges` in `grid`; raises unless every edge is bitwise present."""
    edges = np.asarray(edges, dtype=np.float64)
    index = np.searchsorted(grid, edges)
    if np.any(index >= len(grid)) or not np.array_equal(grid[np.minimum(index, len(grid) - 1)], edges):
        raise ValueError('edges are not an exact subset of the grid')
    return index


OLD_INDEX = subgrid_index(X200, OLD_EDGES)
LEE22_INDEX = subgrid_index(X200, LEE22_EDGES)
B24_INDEX = subgrid_index(X500, B24_EDGES)
assert len(OLD_INDEX) == 37 and OLD_INDEX[0] == 0 and OLD_INDEX[-1] == N200     # G2
assert X200[-1] == 4.0 and X500[-1] == 6.0


def rebin(values, grid, edges, axis=-1):
    """Sum shells of `grid` into the shells of the exact sub-grid `edges` (linear sums only)."""
    index = subgrid_index(grid, edges)
    values = np.moveaxis(np.asarray(values), axis, -1)[..., index[0]:index[-1]]
    out = np.add.reduceat(values, index[:-1] - index[0], axis=-1)
    return np.moveaxis(out, -1, axis)


def to_old(values, axis=-1):
    """X200 shells -> the radii234 36 shells."""
    return rebin(values, X200, OLD_EDGES, axis)


def to_lee22(values, axis=-1):
    return rebin(values, X200, LEE22_EDGES, axis)


def to_b24(values, axis=-1):
    return rebin(values, X500, B24_EDGES, axis)


def shell_index(x, grid):
    """Shell of each x in [0, grid[-1]] (outer edge inclusive, as measure.accumulate)."""
    return np.minimum(np.searchsorted(grid, x, side='right') - 1, len(grid) - 2)


def g2_report():
    return {'X200_edges': len(X200), 'X500_edges': len(X500), 'old_edges_exact_subset': True,
            'old_index': OLD_INDEX.tolist(), 'lee22_exact_subset': True, 'b24_exact_subset': True,
            'thinnest_X200_log10_width': float(np.min(np.log10(X200[2:] / X200[1:-1]))),
            'thinnest_X500_log10_width': float(np.min(np.log10(X500[2:] / X500[1:-1])))}
