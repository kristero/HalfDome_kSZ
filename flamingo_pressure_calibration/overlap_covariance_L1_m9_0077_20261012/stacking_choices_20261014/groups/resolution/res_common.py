"""Shared definitions of the 'resolution' group (stacking_choices_20261014/groups/resolution).

Resolution settings, a local resolution mask that reproduces stackcore.resolution_detail exactly for the core's
rules and adds
  * 'off' semantics for hsml_factor = 0 (the core keeps an implicit ">= 1 particle" test there, because an empty
    shell has a NaN smoothing length and NaN comparisons are False),
  * fraction rules ('frac'): a shell is kept when the inclusion-weighted fraction of haloes resolved in it is
    >= frac_min; every halo enters the mean (like 'strict'),
  * 'per_halo_frac': per-halo masks (unresolved halo-shells dropped) and a shell is kept only if >= frac_min of the
    weight is resolved there.
Everything else (stacking, bootstrap seeds, strata) goes through stackcore.stack with stack_bin's conventions, so
bootstrap replicates are paired with every other group.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GROUP = Path(__file__).resolve().parent
OUT = GROUP.parents[1]
STUDY = OUT.parent
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

GROUP_NAME = "resolution"
SCHEME = "M200c_0.2"
GRID = "fit"
N_BOOT = 200
CAL_LABELS = ("13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9",
              "14.9-15.1", "15.1-15.3", "15.3-15.5")
BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))
BAND_LABELS = tuple(f"{a:g}-{b:g}" for a, b in BANDS)
FIELDS_WEB_ALL = ("P", "E", "G", "T_e", "K")

REF = dict(rule="strict", min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04)


def field_key(field: str, grid: str = GRID) -> str:
    est = "weighted" if field in ("T_e", "T_M", "T_V", "T_X", "K", "C_rho", "C_P") else "geometric"
    return sc.make_key(field, estimator=est, grid=grid)


def _s(short, dimension, label, description, **kw):
    o = dict(REF)
    o.update(kw)
    return dict(id="ref" if short == "ref" else f"{GROUP_NAME}:{short}", short=short, dimension=dimension,
                label=label, description=description, opts=o)


D_MC = "Resolution: minimum gas particles per halo-shell"
D_MCO = "Resolution: minimum gas particles per halo-shell (count criterion alone)"
D_SF = "Resolution: softening criterion factor"
D_HF = "Resolution: smoothing-length criterion factor"
D_ME = "Resolution: minimum inner edge"
D_RULE = "Resolution: masking rule"

SETTINGS = [
    _s("ref", "Reference", "Reference (strict, >=20 particles, 4 eps, 2 h, x>=0.04)",
       "Reference stack: geometric estimator, recent-AGN gas removed, all gas, fit grid, inclusion-weighted mean; a "
       "shell is kept only if every halo of the bin has >= 20 gas particles in it, its inner edge is >= 4 softening "
       "lengths, >= 2 upper-median smoothing lengths and >= 0.04 R200c."),
]
for mc in (0, 5, 10, 32, 50, 100, 200, 500):
    SETTINGS.append(_s(f"mc{mc}", D_MC, f">= {mc} particles per halo-shell" + (" (L2p8 choice)" if mc == 32 else ""),
                       f"As the reference, but a halo-shell needs >= {mc} gas particles instead of 20 (softening, "
                       f"smoothing-length and 0.04 R200c criteria unchanged). Same stacked values; only the kept "
                       f"shells change.", min_count=mc))
for mc in (0, 5, 10, 20, 50, 100, 200, 500):
    SETTINGS.append(_s(f"mconly{mc}", D_MCO, f">= {mc} particles, other criteria off",
                       f"Only the particle-count criterion (>= {mc} gas particles in every halo of the bin); the "
                       f"softening, smoothing-length and minimum-edge criteria are switched off. Same stacked "
                       f"values; only the kept shells change.",
                       min_count=mc, softening_factor=0.0, hsml_factor=0.0, min_edge=0.0))
for sf in (0, 2, 3, 6):
    SETTINGS.append(_s(f"sf{sf}", D_SF, f"inner edge >= {sf} softening lengths",
                       f"As the reference, but the inner edge must be >= {sf} x the 5.7 kpc softening (reference 4). "
                       f"Same stacked values; only the kept shells change.", softening_factor=float(sf)))
for hf in (0, 1, 3):
    SETTINGS.append(_s(f"hf{hf}", D_HF, f"inner edge >= {hf} smoothing lengths",
                       f"As the reference, but the inner edge must be >= {hf} x the upper-median SPH smoothing "
                       f"length of the shell (reference 2; 0 = criterion off). Same stacked values; only the kept "
                       f"shells change.", hsml_factor=float(hf)))
for me in (0.0, 0.1):
    SETTINGS.append(_s(f"me{me:g}", D_ME, f"inner edge >= {me:g} R200c",
                       f"As the reference, but the minimum inner edge is {me:g} R200c (reference 0.04). Same "
                       f"stacked values; only the kept shells change.", min_edge=me))
SETTINGS += [
    _s("frac95", D_RULE, ">= 95% of haloes resolved (all haloes averaged)",
       "Shell kept if >= 95% of the bin's inclusion weight is resolved there (reference criteria); every halo enters "
       "the mean, as in the reference. Unlike 'strict', this does not get stricter as the bin holds more haloes.",
       rule="frac", frac_min=0.95),
    _s("frac84", D_RULE, ">= 84% of haloes resolved (all haloes averaged)",
       "Shell kept if >= 84% of the bin's inclusion weight is resolved there (reference criteria); every halo enters "
       "the mean.", rule="frac", frac_min=0.84),
    _s("diagnostic", D_RULE, "diagnostic rule (pooled counts)",
       "Shell kept if every halo passes the geometric (softening, 0.04 R200c) criterion, the bin holds >= 200 "
       "particles in total and >= 50% of haloes have >= 1 particle there; no per-halo count or smoothing-length "
       "test. Same stacked values.", rule="diagnostic"),
    _s("per_halo", D_RULE, "per-halo masks (unresolved halo-shells dropped)",
       "Each halo-shell that fails the reference criteria is dropped individually, so partially resolved inner "
       "shells average only the resolved (denser) haloes; a shell is shown if at least one halo is resolved.",
       rule="per_halo"),
    _s("per_halo_f50", D_RULE, "per-halo masks, shell kept if >= 50% resolved",
       "As 'per-halo masks', but a shell is kept only where >= 50% of the bin's inclusion weight is resolved.",
       rule="per_halo_frac", frac_min=0.5),
    _s("none", D_RULE, "no resolution mask",
       "Every shell is kept down to the centre, including shells with no particles or with fewer particles than "
       "needed; same stacked values as the reference.", rule="none"),
]
SETTING_BY_ID = {s["id"]: s for s in SETTINGS}
SETTING_BY_SHORT = {s["short"]: s for s in SETTINGS}


# ============================================================================ masks
def resolution_detail(rows, grid: str, opts: dict) -> dict:
    """stackcore.resolution_detail semantics; hsml_factor == 0 switches the smoothing-length test off entirely
    (no implicit non-empty test); per-criterion masks are returned for the 'which criterion binds' analysis."""
    rows = np.asarray(rows)
    x = sc.edges(grid)[:-1]
    tab = sc.halo_table()
    eps = (tab["eps_over_r500c"] if grid in sc.X500_GRIDS else tab["eps_over_r200c"])[rows]
    c = np.asarray(sc.counts(grid)[rows])
    h = np.asarray(sc.hsml_x(grid)[rows])
    n, S = c.shape
    c_min_edge = np.broadcast_to(x[None, :] >= opts["min_edge"], (n, S))
    c_soft = x[None, :] >= opts["softening_factor"] * eps[:, None]
    c_count = c >= opts["min_count"]
    with np.errstate(invalid="ignore"):
        c_hsml = (np.ones((n, S), bool) if opts["hsml_factor"] == 0 else (x[None, :] >= opts["hsml_factor"] * h))
    geometric = c_min_edge & c_soft
    resolved = geometric & c_count & c_hsml
    return {"resolved": resolved, "geometric": geometric, "c_min_edge": c_min_edge, "c_soft": c_soft,
            "c_count": c_count, "c_hsml": c_hsml, "counts": c, "hsml": h}


def weighted_fraction(mask2d, w):
    return (np.asarray(mask2d, float) * w[:, None]).sum(0) / w.sum()


def kept_and_halo_mask(mass_bin: dict, grid: str, opts: dict):
    """(kept [S] shells used downstream, halo_shell_mask for the stack or None, detail)."""
    rows, w = mass_bin["rows"], mass_bin["weights"]
    rule = opts["rule"]
    S = len(sc.edges(grid)) - 1
    det = resolution_detail(rows, grid, opts)
    res = det["resolved"]
    if rule == "none":
        return np.ones(S, bool), None, det
    if rule == "strict":
        return res.all(axis=0), None, det
    if rule == "diagnostic":
        c = det["counts"]
        o = sc.RES_DEFAULTS
        kept = (det["geometric"].all(axis=0) & (c.sum(axis=0) >= o["diagnostic_total_count"])
                & (np.mean(c > 0, axis=0) >= o["diagnostic_nonzero_fraction"]))
        return kept, None, det
    if rule == "frac":
        return weighted_fraction(res, w) >= opts["frac_min"] - 1e-12, None, det
    if rule == "per_halo":
        return res.any(axis=0), res, det
    if rule == "per_halo_frac":
        return weighted_fraction(res, w) >= opts["frac_min"] - 1e-12, res, det
    raise ValueError(rule)


def stack_setting(key: str, mass_bin: dict, opts: dict, n_boot: int = N_BOOT, return_boot: bool = True) -> dict:
    """stackcore.stack with stack_bin's conventions (seed, strata, weights) and this setting's mask."""
    grid = sc.variant(key)["grid"]
    rows = mass_bin["rows"]
    kept, hmask, det = kept_and_halo_mask(mass_bin, grid, opts)
    vals = np.asarray(sc.per_halo(key)[rows])
    seed = sc.STUDY_SEED + 1000 * int(mass_bin["index"])
    r = sc.stack(vals, mass_bin["weights"], "mean", halo_shell_mask=hmask, n_boot=n_boot, seed=seed,
                 strata=mass_bin["strata"], return_boot=return_boot)
    if hmask is not None:
        kept = kept & (r["n_used"] > 0)
    r["resolved"] = kept
    r["edges"] = sc.edges(grid)
    return r


# ============================================================================ geometry helpers
def x_volume(edges):
    a, b = np.asarray(edges[:-1], float), np.asarray(edges[1:], float)
    return 0.75 * (b**4 - a**4) / (b**3 - a**3)


def band_index(edges):
    xv = x_volume(edges)
    out = np.full(len(xv), -1)
    for k, (a, b) in enumerate(BANDS):
        out[(xv >= a) & (xv < b)] = k
    return out


def first_edge(kept, edges):
    idx = np.flatnonzero(kept)
    return float(edges[idx[0]]) if len(idx) else None


def contiguous(kept):
    idx = np.flatnonzero(kept)
    return bool(len(idx) == 0 or np.all(np.diff(idx) == 1))


def sig(x, n=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def plain(o, n=4):
    """Recursively convert to JSON-ready values with n significant digits (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): plain(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v, n) for v in o]
    if isinstance(o, np.ndarray):
        if o.dtype == bool:
            return [bool(v) for v in o.ravel()] if o.ndim <= 1 else [plain(v, n) for v in o]
        return [plain(v, n) for v in o] if o.ndim > 1 else [sig(v, n) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return sig(o, n)
    return o


def write_json(path, obj, n=4):
    Path(path).write_text(json.dumps(plain(obj, n), separators=(",", ":"), allow_nan=False))
