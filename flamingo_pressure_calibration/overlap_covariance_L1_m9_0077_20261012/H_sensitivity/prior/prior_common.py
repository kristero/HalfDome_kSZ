"""Shared paths and helpers for the H_sensitivity 'prior results' collection (read-only use of earlier tests).

Nothing here writes outside H_sensitivity/prior/.  Earlier products are only read (np.load / csv / json).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[2]
PRIOR = Path(__file__).resolve().parent
TABLES = PRIOR / "tables"
FIGURES = PRIOR / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# earlier products (read only)
F_IMP = STUDY / "F_observables/results/importance_F.npz"
F_TABLES = STUDY / "F_observables/results/tables"
K_JAC = STUDY / "kernels/jacobians.npz"
K_IMP = STUDY / "kernels/importance.npz"
G_TABLES = STUDY / "G_physical_outer/results/tables"
B_TABLES = STUDY / "B_covariance/tables"
B_RESULTS = STUDY / "B_covariance/results"
MB_WORK = STUDY / "stacking_choices_20261014/groups/mass_binning/work"

# radial bands requested by the user (R200c); on the F grid 0.1/0.25/0.5 are the nearest edges 0.1051/0.2335/0.5185
BAND_EDGES_NOMINAL = np.array([0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
BAND_LABELS = ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")
SEL_ELL = (100, 300, 1000, 3000, 8000)

# normalisation exponent k of the importance maps: 2 for quadratic observables, 1 for linear ones
QUADRATIC = {"yy1h", "yy2h", "yytot", "ksz1h", "ksz2h", "dmcltot", "dm_var"}


def band_index(edges: np.ndarray, band_edges: np.ndarray) -> np.ndarray:
    """Band (0..7) of every shell, by its geometric midpoint; -1 beyond the last band edge."""
    lo = np.maximum(edges[:-1], 1e-6)
    mid = np.sqrt(lo * edges[1:])
    out = np.searchsorted(band_edges, mid) - 1
    out[edges[1:] > band_edges[-1] + 1e-9] = -1
    return out


def cumulative_radius(edges: np.ndarray, imp: np.ndarray, frac: float) -> float:
    """Smallest radius x at which the cumulative importance (from the centre) first reaches `frac` of the total.

    imp[s] is the importance of shell s (edges[s], edges[s+1]); inside a shell the cumulative is interpolated
    linearly in x.  Signed importances are allowed (first crossing is returned); NaN if never reached.
    """
    tot = float(np.sum(imp))
    if tot == 0:
        return float("nan")
    c = np.r_[0.0, np.cumsum(imp)] / tot
    for s in range(len(imp)):
        if c[s] < frac <= c[s + 1]:
            t = (frac - c[s]) / (c[s + 1] - c[s])
            return float(edges[s] + t * (edges[s + 1] - edges[s]))
    return float("nan")


def share_beyond(edges: np.ndarray, imp: np.ndarray, x: float) -> float:
    """Fraction of the total importance in shells whose inner edge is >= x (x must be an edge)."""
    tot = float(np.sum(imp))
    k = int(np.argmin(np.abs(edges - x)))
    assert abs(edges[k] - x) < 1e-3 * max(1, x), (x, edges[k])
    return float(np.sum(imp[k:]) / tot)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.6g}"
    return v


def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(path: Path, obj) -> None:
    with open(path, "w") as fh:
        json.dump(plain(obj), fh, indent=1)
