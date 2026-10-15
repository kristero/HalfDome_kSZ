"""Shared stacking core of stacking_choices_20261014 (import with sys.path.insert(0, OUT + '/core')).

Reads the per-halo cache written by core/build_cache.py (cache/*.npy, memory-mapped) and provides:

  load_index(), variant_keys(**tokens), variant(key), halo_table(), edges(grid), per_halo(key) -> mmap [4588, S],
  counts(grid), hsml_x(grid), shell_volumes(grid), y200(rows)
  mass_bins(scheme)                          bins of 'M200c_0.2' | 'M200c_0.1' | 'M200c_0.3' | 'M200c_0.5' |
                                             'M500c_0.2' | 'M500c_0.5' | 'Lee22_h'
  resolution_mask(rows, grid, rule, ...)     'strict' | 'diagnostic' -> [S]; 'per_halo' -> [n, S]; 'none' -> [S]
  stack(values, weights, statistic, ...)     'mean' | 'unweighted_mean' | 'median' | 'geomean' | 'y_weighted_mean' |
                                             'trimmed_mean'; bootstrap errors (optionally stratified, paired by seed)
  stack_bin(key, bin, ...), stack_bins(key, scheme, ...)    convenience drivers -> result dicts / records
  observable_impact(dlnF, field), impact_from_stacks(F_variant, F_reference, field)
  record(...)                                compact JSON-ready dict (4 significant digits) for the website

Conventions
-----------
* Rows: every per-halo array has 4588 rows in STUDY/selected_halos.npz order (= cache/halo_table.npz 'row').
* Missing values: NaN halo-shells (undefined ratios, excision retained < 50%) are left out of every statistic shell
  by shell (they are never set to zero).  Test F set NaN to 0 before stacking; its geometric P/E have no NaN.
* Inclusion weights: 1 / inclusion probability of the (0.2-dex M200c bin, census class) stratum
  (halo_table 'weight'); the 'mean' statistic is the Horvitz-Thompson (ratio) estimator of the population mean.
* Bootstrap: halo resampling with replacement, n_boot draws, numpy default_rng(seed); with ``strata`` the resampling
  is done within each stratum (the design strata are mass_bin x class_main).  The same (n, seed, strata) give the
  same resamples, so the replicates of two variants of one bin are paired draw by draw (return_boot=True).
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
STUDY = OUT.parent
CACHE = OUT / "cache"
IMPORTANCE_FILE = STUDY / "F_observables" / "results" / "importance_F.npz"
STUDY_SEED = 20261012
STATISTICS = ("mean", "unweighted_mean", "median", "geomean", "y_weighted_mean", "trimmed_mean")
RULES = ("strict", "diagnostic", "per_halo", "none")
X500_GRIDS = ("x500", "b24")
KEY_TOKENS = ("field", "estimator", "agn", "selection", "phase", "grid", "norm")


# ============================================================================ cache access
@lru_cache(maxsize=1)
def load_index() -> dict:
    """cache/index.json: {'conventions', 'variants': {key: metadata}, 'resolution_files', ...}."""
    return json.loads((CACHE / "index.json").read_text())


def variant(key: str) -> dict:
    return load_index()["variants"][key]


def variant_keys(**tokens) -> list:
    """Keys whose tokens match, e.g. variant_keys(field='P', grid='fit', agn='removed')."""
    bad = set(tokens) - set(KEY_TOKENS)
    if bad:
        raise ValueError(f"unknown tokens {bad}")
    out = []
    for key in load_index()["variants"]:
        parts = dict(zip(KEY_TOKENS, key.split("|")))
        if all(parts[k] == str(v) for k, v in tokens.items()):
            out.append(key)
    return out


def make_key(field, estimator="geometric", agn="removed", selection="all", phase="all", grid="fit", norm=None) -> str:
    """Build a key; norm defaults to the field's standard normalisation on the grid."""
    if norm is None:
        x500 = grid in X500_GRIDS
        norm = {"P": "B24" if x500 else "P200", "E": "B24" if x500 else "rhoc_fb", "G": "B24" if x500 else "rhoc_fb",
                "E_fullion": "B24" if x500 else "rhoc_fb", "T_M": "T500" if x500 else "T200",
                "T_V": "T500" if x500 else "T200", "T_X": "T500" if x500 else "T200", "T_e": "T500" if x500 else "T200",
                "K": "K500" if x500 else "K200", "C_rho": "none", "C_P": "none"}[field]
    return "|".join((field, estimator, agn, selection, phase, grid, norm))


@lru_cache(maxsize=1)
def halo_table() -> dict:
    """cache/halo_table.npz as a dict of arrays (rows = study order)."""
    with np.load(CACHE / "halo_table.npz") as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def edges(grid: str) -> np.ndarray:
    """Shell edges of a grid in units of its radius (R200c: x200/fit/lee22/old; R500c: x500/b24)."""
    return np.load(CACHE / f"edges_{grid}.npy")


def grid_radius(grid: str) -> str:
    return "R500c" if grid in X500_GRIDS else "R200c"


def per_halo(key: str) -> np.ndarray:
    """Read-only memory map [4588, S] of one variant."""
    return np.load(CACHE / variant(key)["file"], mmap_mode="r")


def counts(grid: str) -> np.ndarray:
    """[4588, S] gas-particle counts per shell (all owners and phases, recent AGN included)."""
    return np.load(CACHE / f"counts_{grid}.npy", mmap_mode="r")


def hsml_x(grid: str) -> np.ndarray:
    """[4588, S] upper-median smoothing length h a / R per shell (NaN in empty shells)."""
    return np.load(CACHE / f"hsml_{grid}.npy", mmap_mode="r")


def retained(selection: str, grid: str) -> np.ndarray:
    """Retained-volume fraction [4588, S] of an excision selection (exc_*)."""
    return np.load(CACHE / f"retained__{selection}__{grid}.npy", mmap_mode="r")


def shell_volumes(grid: str) -> np.ndarray:
    """Dimensionless shell volumes 4 pi/3 (x_hi^3 - x_lo^3) (multiply by R^3 for physical volumes)."""
    e = edges(grid)
    return 4 * np.pi / 3 * np.diff(e**3)


def integral(values, grid: str, upto: float) -> np.ndarray:
    """sum_s F_s 4pi/3 (x_hi^3 - x_lo^3) over shells with outer edge <= upto (NaN -> 0); per halo or per stack."""
    e = edges(grid)
    inside = e[1:] <= upto + 1e-12
    v = np.nan_to_num(np.asarray(values, float), nan=0.0)
    return v[..., inside] @ shell_volumes(grid)[inside]


def y200(rows=None) -> np.ndarray:
    """Own integrated Compton Y inside R200c [Mpc^2] (sum of particle y, all gas, recent AGN removed)."""
    y = halo_table()["Y200_own_mpc2"]
    return y if rows is None else y[np.asarray(rows)]


# ============================================================================ mass bins
H_DEFAULT = 0.681


def _hubble() -> float:
    try:
        return float(load_index()["hubble"])
    except Exception:       # noqa: BLE001
        plan = json.loads((STUDY / "metadata_plan.json").read_text())
        return float(np.atleast_1d(plan["cosmology"]["h"])[0])


def _edge_list(lo, hi, step):
    e = list(np.round(np.arange(lo, hi + 1e-9, step), 6))
    if e[-1] < hi - 1e-9:
        e.append(hi)                      # last partial bin
    return np.asarray(e, float)


def scheme_definition(scheme: str) -> dict:
    """Variable and edges of a binning scheme."""
    tab = halo_table()
    if scheme == "M200c_0.2":           # the stratification bins, exact stored edges
        return {"variable": "logm200c", "edges": np.asarray(tab["mass_edges"], float), "units": "log10 M200c [Msun]"}
    if scheme == "M200c_0.1":
        return {"variable": "logm200c", "edges": _edge_list(12.5, 15.5, 0.1), "units": "log10 M200c [Msun]"}
    if scheme == "M200c_0.3":
        return {"variable": "logm200c", "edges": _edge_list(12.5, 15.5, 0.3), "units": "log10 M200c [Msun]"}
    if scheme == "M200c_0.5":
        return {"variable": "logm200c", "edges": _edge_list(12.5, 15.5, 0.5), "units": "log10 M200c [Msun]"}
    if scheme == "M500c_0.2":
        # M500c >= 10^12.5 implies M200c >= 10^12.5, so these bins are complete in the parent population
        return {"variable": "logm500c", "edges": _edge_list(12.5, 15.5, 0.2), "units": "log10 M500c [Msun]"}
    if scheme == "M500c_0.5":           # B24-like
        return {"variable": "logm500c", "edges": _edge_list(13.0, 15.5, 0.5), "units": "log10 M500c [Msun]"}
    if scheme == "Lee22_h":
        return {"variable": "logm200c_h", "edges": _edge_list(13.0, 15.4, 0.2), "units": "log10 M200c [h^-1 Msun]",
                "h": _hubble()}
    raise ValueError(f"unknown scheme {scheme!r}")


SCHEMES = ("M200c_0.2", "M200c_0.1", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5", "Lee22_h")


@lru_cache(maxsize=None)
def _mass_bins(scheme: str) -> tuple:
    tab = halo_table()
    sd = scheme_definition(scheme)
    if sd["variable"] == "logm200c_h":
        x = tab["logm200c"] + np.log10(sd["h"])
    else:
        x = tab[sd["variable"]]
    e = sd["edges"]
    w_all = tab["weight"]
    strata_all = tab["mass_bin"] * 3 + tab["class_main"]
    out = []
    for k in range(len(e) - 1):
        rows = np.flatnonzero((x >= e[k]) & (x < e[k + 1]))
        if not len(rows):
            continue
        w = w_all[rows]
        label = f"{e[k]:.1f}-{e[k + 1]:.1f}"
        out.append({"scheme": scheme, "label": label, "index": k, "lo": float(e[k]), "hi": float(e[k + 1]),
                    "variable": sd["variable"], "units": sd["units"], "rows": rows, "weights": w,
                    "mean_log_mass": float(np.average(x[rows], weights=w)),
                    "mean_logm200c": float(np.average(tab["logm200c"][rows], weights=w)),
                    "mean_logm500c": float(np.average(tab["logm500c"][rows], weights=w)),
                    "N": int(len(rows)), "N_eff": float(w.sum()**2 / (w**2).sum()),
                    "sum_weight": float(w.sum()), "strata": strata_all[rows]})
    return tuple(out)


def mass_bins(scheme: str = "M200c_0.2") -> list:
    """Non-empty bins of a scheme: dicts with label, lo, hi, rows, weights, mean_log_mass (of the scheme variable,
    inclusion-weighted), mean_logm200c, mean_logm500c, N, N_eff, sum_weight, strata (mass_bin x class_main)."""
    return list(_mass_bins(scheme))


def find_bin(scheme: str, label: str) -> dict:
    for b in _mass_bins(scheme):
        if b["label"] == label:
            return b
    raise KeyError(f"{scheme}: no bin {label}")


# ============================================================================ resolution
RES_DEFAULTS = dict(min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04,
                    diagnostic_total_count=200, diagnostic_nonzero_fraction=0.5)


def resolution_detail(rows, grid: str, **options) -> dict:
    """analysis.masks.resolution_masks semantics on the cached counts / smoothing lengths.

    geometric[h, s]: inner edge >= max(min_edge, softening_factor * eps / R) (R = R200c or R500c of the grid);
    resolved[h, s]:  geometric & count >= min_count & inner edge >= hsml_factor * upper-median h a / R
                     (empty shells: NaN smoothing length -> unresolved);
    strict[s]:       every halo resolved;
    diagnostic[s]:   every halo geometric & sum of counts >= diagnostic_total_count &
                     fraction of haloes with particles >= diagnostic_nonzero_fraction.
    """
    o = dict(RES_DEFAULTS, **options)
    rows = np.asarray(rows)
    x = edges(grid)[:-1]
    tab = halo_table()
    eps_over_r = (tab["eps_over_r500c"] if grid in X500_GRIDS else tab["eps_over_r200c"])[rows]
    c = np.asarray(counts(grid)[rows])
    h = np.asarray(hsml_x(grid)[rows])
    geometric = x[None, :] >= np.maximum(o["min_edge"], o["softening_factor"] * eps_over_r[:, None])
    with np.errstate(invalid="ignore"):
        resolved = geometric & (c >= o["min_count"]) & (x[None, :] >= o["hsml_factor"] * h)
    strict = np.all(resolved, axis=0)
    diagnostic = (np.all(geometric, axis=0) & (c.sum(axis=0) >= o["diagnostic_total_count"])
                  & (np.mean(c > 0, axis=0) >= o["diagnostic_nonzero_fraction"]))
    return {"geometric": geometric, "resolved": resolved, "strict": strict, "diagnostic": diagnostic,
            "resolved_fraction": resolved.mean(axis=0), "options": o}


def resolution_mask(rows, grid: str, rule: str = "strict", min_count: int = 20, softening_factor: float = 4.0,
                    hsml_factor: float = 2.0, min_edge: float = 0.04, **extra) -> np.ndarray:
    """'strict' / 'diagnostic' -> shell mask [S]; 'per_halo' -> [n, S] (unresolved halo-shells dropped
    individually); 'none' -> all shells [S]."""
    if rule == "none":
        return np.ones(len(edges(grid)) - 1, bool)
    if rule not in RULES:
        raise ValueError(rule)
    det = resolution_detail(rows, grid, min_count=min_count, softening_factor=softening_factor,
                            hsml_factor=hsml_factor, min_edge=min_edge, **extra)
    return det["resolved"] if rule == "per_halo" else det[rule]


# ============================================================================ statistics
def _bootstrap_counts(n: int, n_boot: int, seed: int, strata=None) -> np.ndarray:
    """Multiplicity matrix [n_boot, n] of halo resamples (analysis.stacking conventions)."""
    if n_boot <= 0:
        return np.zeros((0, n))
    rng = np.random.default_rng(seed)
    if strata is None:
        pos = rng.integers(0, n, size=(n_boot, n))
    else:
        strata = np.asarray(strata)
        pos = np.empty((n_boot, n), dtype=np.int64)
        for s in np.unique(strata):
            where = np.flatnonzero(strata == s)
            pos[:, where] = where[rng.integers(0, len(where), size=(n_boot, len(where)))]
    flat = (np.arange(n_boot)[:, None] * n + pos).ravel()
    return np.bincount(flat, minlength=n_boot * n).reshape(n_boot, n).astype(np.float64)


def _weighted_quantile_rows(vs: np.ndarray, W: np.ndarray, q: float) -> np.ndarray:
    """Weighted quantile of sorted values vs [m] for each weight row W [r, m] (inverse of the weighted step CDF; at an
    exact tie of the cumulative weight with q * total, the mean of the two neighbouring values, so equal weights
    reproduce np.median).  Rows with zero total weight give NaN."""
    cw = np.cumsum(W, axis=1)
    tot = cw[:, -1:]
    target = q * tot
    tol = 1e-12 * np.maximum(tot, 1e-300)
    ge = cw >= target - tol
    gt = cw > target + tol
    k_lo = np.argmax(ge, axis=1)
    k_hi = np.where(gt.any(axis=1), np.argmax(gt, axis=1), W.shape[1] - 1)
    out = 0.5 * (vs[k_lo] + vs[k_hi])
    out[tot[:, 0] <= 0] = np.nan
    return out


def _trimmed_rows(vs: np.ndarray, W: np.ndarray, trim: float) -> np.ndarray:
    """Weighted trimmed mean per weight row: the lowest and highest ``trim`` fractions of the total weight are removed
    exactly (a straddling halo keeps the fraction of its weight inside [trim, 1 - trim])."""
    cw = np.cumsum(W, axis=1)
    tot = cw[:, -1:]
    lo = cw - W
    kept = np.clip(np.minimum(cw, (1 - trim) * tot) - np.maximum(lo, trim * tot), 0.0, None)
    s = kept.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, kept @ vs / np.where(s > 0, s, 1.0), np.nan)


def stack(values, weights=None, statistic: str = "mean", halo_shell_mask=None, n_boot: int = 200,
          seed: int = STUDY_SEED, strata=None, y200=None, trim: float = 0.05, return_boot: bool = False) -> dict:
    """Per-shell stack of per-halo values [n, S].

    statistic:
      'mean'            inclusion-weighted mean sum(w F)/sum(w) over haloes with finite (and masked-in) F
      'unweighted_mean' plain mean (inclusion weights ignored; biased when strata are sampled unequally)
      'median'          weighted median (inclusion weights; inverse weighted CDF, ties averaged)
      'geomean'         exp(sum(w ln F)/sum(w)) over haloes with F > 0 (F <= 0 haloes are left out; see n_used)
      'y_weighted_mean' sum(w Y200 F)/sum(w Y200): inclusion weight x the halo's own Y(<R200c) (B12-style
                        signal weighting; Y200 = sum of particle y inside R200c, all gas, recent AGN removed;
                        pass ``y200`` [n], e.g. stackcore.y200(rows))
      'trimmed_mean'    weighted mean after removing the lowest and highest ``trim`` (5%) of the weight per shell
    halo_shell_mask: None, [S] or [n, S] booleans (False = halo-shell left out).
    Returns dict:
      value [S]; err [S] = bootstrap SD (ddof 1) of the statistic over halo resamples (NaN if n_boot = 0);
      boot_q16 / boot_q84 [S] percentiles of the bootstrap replicates;
      sd [S] = reliability-weighted SD of the per-halo F about the weighted mean (analysis.stacking.weighted_stack
               maths, statistic weights), q16 / q84 [S] = weighted 16th / 84th percentiles of the per-halo F,
      sd_ln [S] = the same SD of ln F over F > 0 (lognormal scatter);
      n_used [S] haloes entering the statistic; n_eff [S] = (sum w)^2 / sum w^2 over them; n_total, statistic;
      boot [n_boot, S] if return_boot (paired across variants for equal n, seed and strata).
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic {statistic!r} not in {STATISTICS}")
    v = np.array(values, dtype=float)                     # private copy (values may be a memory map)
    if v.ndim != 2:
        raise ValueError("values must be [n, S]")
    n, S = v.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float).copy()
    if w.shape != (n,):
        raise ValueError("weights must be [n]")
    if statistic == "unweighted_mean":
        w = np.ones(n)
    elif statistic == "y_weighted_mean":
        if y200 is None:
            raise ValueError("y_weighted_mean needs y200 [n]")
        y = np.asarray(y200, float)
        if np.any(~np.isfinite(y)) or np.any(y < 0):
            raise ValueError("y200 must be finite and >= 0")
        w = w * y
    base_ok = np.isfinite(v)
    if halo_shell_mask is not None:
        m = np.asarray(halo_shell_mask, bool)
        base_ok &= np.broadcast_to(m if m.ndim == 2 else m[None, :], (n, S))
    ok = base_ok & (v > 0) if statistic == "geomean" else base_ok
    okf = ok.astype(float)
    C = _bootstrap_counts(n, n_boot, seed, strata)
    Wd = np.vstack([w[None, :], C * w[None, :]])        # row 0 = the sample itself
    # ---------------------------------------------------------------- the statistic (sample + replicates)
    if statistic in ("mean", "unweighted_mean", "y_weighted_mean", "geomean"):
        with np.errstate(divide="ignore", invalid="ignore"):
            work = np.where(ok, np.log(np.where(ok, v, 1.0)) if statistic == "geomean" else v, 0.0)
            num = Wd @ work
            den = Wd @ okf
            allv = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        if statistic == "geomean":
            allv = np.exp(allv)
    else:
        allv = np.full((Wd.shape[0], S), np.nan)
        for s in range(S):
            idx = np.flatnonzero(ok[:, s])
            if not len(idx):
                continue
            order = idx[np.argsort(v[idx, s], kind="stable")]
            vs = v[order, s]
            Ws = Wd[:, order]
            allv[:, s] = (_weighted_quantile_rows(vs, Ws, 0.5) if statistic == "median"
                          else _trimmed_rows(vs, Ws, trim))
    value = allv[0]
    boot = allv[1:]
    if n_boot >= 2:
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            nb = np.isfinite(boot).sum(axis=0)
            err = np.where(nb >= 2, np.nanstd(boot, axis=0, ddof=1), np.nan)
            bq = np.nanpercentile(boot, [16, 84], axis=0)
    else:
        err = np.full(S, np.nan)
        bq = np.full((2, S), np.nan)
    # ---------------------------------------------------------------- per-halo scatter (statistic weights)
    Wm = np.where(base_ok, w[:, None], 0.0)
    v0 = np.where(base_ok, v, 0.0)
    tot = Wm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_w = np.where(tot > 0, (Wm * v0).sum(axis=0) / tot, np.nan)
        var = (Wm * (v0 - mean_w[None, :])**2).sum(axis=0) / tot
        corr = 1 - (Wm**2).sum(axis=0) / tot**2
        sd = np.where(corr > 0, np.sqrt(var / corr), np.nan)
        pos = base_ok & (v > 0)
        Wl = np.where(pos, w[:, None], 0.0)
        lv = np.where(pos, np.log(np.where(pos, v, 1.0)), 0.0)
        tl = Wl.sum(axis=0)
        ml = np.where(tl > 0, (Wl * lv).sum(axis=0) / tl, np.nan)
        varl = (Wl * (lv - ml[None, :])**2).sum(axis=0) / tl
        corrl = 1 - (Wl**2).sum(axis=0) / tl**2
        sd_ln = np.where(corrl > 0, np.sqrt(varl / corrl), np.nan)
    q16 = np.full(S, np.nan)
    q84 = np.full(S, np.nan)
    for s in range(S):
        idx = np.flatnonzero(base_ok[:, s])
        if not len(idx):
            continue
        order = idx[np.argsort(v[idx, s], kind="stable")]
        q16[s] = _weighted_quantile_rows(v[order, s], w[order][None, :], 0.16)[0]
        q84[s] = _weighted_quantile_rows(v[order, s], w[order][None, :], 0.84)[0]
    Wu = np.where(ok, w[:, None], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(Wu.sum(axis=0) > 0, Wu.sum(axis=0)**2 / (Wu**2).sum(axis=0), 0.0)
    out = {"statistic": statistic, "value": value, "err": err, "boot_q16": bq[0], "boot_q84": bq[1], "sd": sd,
           "sd_ln": sd_ln, "q16": q16, "q84": q84, "n_used": ok.sum(axis=0), "n_eff": n_eff, "n_total": n,
           "n_boot": n_boot, "seed": seed, "stratified": strata is not None}
    if return_boot:
        out["boot"] = boot
    return out


# ============================================================================ drivers
def stack_bin(key: str, mass_bin: dict, statistic: str = "mean", rule: str = "strict", n_boot: int = 200,
              seed: int | None = None, stratified: bool = True, resolution_options: dict | None = None,
              return_boot: bool = False) -> dict:
    """Stack one cached variant in one mass bin (from mass_bins()).

    rule 'strict' / 'diagnostic' / 'none': every halo enters every shell, 'resolved' [S] flags the shells passing the
    rule; 'per_halo': unresolved halo-shells are dropped individually and 'resolved' = n_used > 0.
    seed defaults to STUDY_SEED + 1000 * bin index (same for every variant of the bin -> paired replicates);
    stratified: resample within the design strata (mass_bin x class_main) of the bin's haloes.
    """
    grid = variant(key)["grid"]
    rows = mass_bin["rows"]
    vals = np.asarray(per_halo(key)[rows])
    S = vals.shape[1]
    ro = resolution_options or {}
    if rule == "per_halo":
        hmask = resolution_mask(rows, grid, "per_halo", **ro)
    else:
        hmask = None
    seed = STUDY_SEED + 1000 * int(mass_bin["index"]) if seed is None else seed
    res = stack(vals, mass_bin["weights"], statistic, halo_shell_mask=hmask, n_boot=n_boot, seed=seed,
                strata=mass_bin["strata"] if stratified else None,
                y200=y200(rows) if statistic == "y_weighted_mean" else None, return_boot=return_boot)
    res["resolved"] = (res["n_used"] > 0) if rule == "per_halo" else (
        resolution_mask(rows, grid, rule, **ro) if rule != "none" else np.ones(S, bool))
    res.update(key=key, grid=grid, rule=rule, bin=mass_bin["label"], scheme=mass_bin["scheme"], N=mass_bin["N"],
               N_eff_bin=mass_bin["N_eff"], mean_log_mass=mass_bin["mean_log_mass"], edges=edges(grid))
    return res


def stack_bins(key: str, scheme: str = "M200c_0.2", statistic: str = "mean", rule: str = "strict",
               n_boot: int = 200, stratified: bool = True, labels=None, resolution_options: dict | None = None,
               sig: int = 4, as_records: bool = True):
    """stack_bin for every (or the listed) bin of a scheme; records (JSON-ready) or full result dicts."""
    out = []
    for b in mass_bins(scheme):
        if labels is not None and b["label"] not in labels:
            continue
        r = stack_bin(key, b, statistic, rule, n_boot=n_boot, stratified=stratified,
                      resolution_options=resolution_options)
        out.append(record(key, variant(key)["field"], scheme, b["label"], r["edges"], r, b["N"], b["N_eff"],
                          extra={"statistic": statistic, "rule": rule, "mean_log_mass": b["mean_log_mass"],
                                 "resolved": r["resolved"]}, sig=sig) if as_records else r)
    return out


# ============================================================================ observables
OBSERVABLES = {
    # name: (importance key, ell or None, k (1 linear, 2 quadratic), field, description)
    "tSZ_yy_total_ell100": ("imp__yytot", 100.0, 2, "P", "tSZ C_ell^yy total (1h + 2h), ell = 100"),
    "tSZ_yy_total_ell1000": ("imp__yytot", 1000.0, 2, "P", "tSZ C_ell^yy total, ell = 1000"),
    "tSZ_yy_total_ell3000": ("imp__yytot", 3000.0, 2, "P", "tSZ C_ell^yy total, ell = 3000"),
    "tSZ_yy_1h_ell100": ("imp__yy1h", 100.0, 2, "P", "tSZ C_ell^yy 1-halo, ell = 100"),
    "tSZ_yy_1h_ell1000": ("imp__yy1h", 1000.0, 2, "P", "tSZ C_ell^yy 1-halo, ell = 1000"),
    "tSZ_yy_1h_ell3000": ("imp__yy1h", 3000.0, 2, "P", "tSZ C_ell^yy 1-halo, ell = 3000"),
    "y_mean": ("imp__ymean", None, 1, "P", "mean Compton y"),
    "tSZxDM_total_ell1000_P": ("imp__ydmtot_P", 1000.0, 1, "P", "y x DM C_ell total, ell = 1000 (pressure side)"),
    "kSZ_1h_ell100": ("imp__ksz1h", 100.0, 2, "E", "kSZ C_ell 1-halo, ell = 100"),
    "kSZ_1h_ell1000": ("imp__ksz1h", 1000.0, 2, "E", "kSZ C_ell 1-halo, ell = 1000"),
    "kSZ_1h_ell3000": ("imp__ksz1h", 3000.0, 2, "E", "kSZ C_ell 1-halo, ell = 3000"),
    "DM_mean": ("imp__dm_mean", None, 1, "E", "mean halo DM"),
    "DM_var": ("imp__dm_var", None, 2, "E", "DM variance"),
    "DM_Cl_total_ell1000": ("imp__dmcltot", 1000.0, 2, "E", "DM-field C_ell total, ell = 1000"),
    "DM_q16": ("imp__dm_q16", None, 1, "E", "DM PDF 16th percentile"),
    "DM_q50": ("imp__dm_q50", None, 1, "E", "DM PDF median"),
    "DM_q84": ("imp__dm_q84", None, 1, "E", "DM PDF 84th percentile"),
    "tSZxDM_total_ell1000_E": ("imp__ydmtot_E", 1000.0, 1, "E", "y x DM C_ell total, ell = 1000 (electron side)"),
}
CAL_BINS = tuple(range(5, 15))          # engine bins 13.5-13.7 ... 15.3-15.5 (F_observables CAL_BINS)


@lru_cache(maxsize=1)
def _importance() -> dict:
    with np.load(IMPORTANCE_FILE) as z:
        out = {k: z[k] for k in z.files}
    if not np.array_equal(out["shell_edges"], edges("fit")):
        raise ValueError("importance_F shell edges differ from the cached fit grid")
    return out


def observable_impact(dlnF, field: str, observables=None) -> dict:
    """Propagate a change of the stacked mean profile to the observables (first order).

    dlnF: [10, 35] = ln(F_new / F_ref) of the stacked mean on the 10 calibrated 0.2-dex M200c bins 13.5-13.7 ...
    15.3-15.5 (F_observables CAL_BINS 5..14) x the 35 fit-grid shells (= the F grid); a [15, 35] or [16, 35] array
    (all engine bins) is accepted and only rows 5..14 are used.  NaN entries are treated as no change and reported
    as 'missing_share'.

    Uses F_observables/results/importance_F.npz 'imp__<obs>' [16 engine bins, 35 shells(, 106 ell)]:
    imp[b, s] = (dO / d ln F_{b,s}) / (k O_truth), k = 2 for quadratic observables (C_ell auto spectra, kSZ, DM
    variance, DM C_ell), k = 1 for means, quantiles and each side of the y x DM cross; these are the scatter-inclusive
    in-basis Jacobians at the measured FLAMINGO truth (every halo of bin b scaled in shell s), so a change of a bin's
    mean profile is propagated as the same fractional change of every halo of that bin (scatter kept).
    With delta = exp(dlnF) - 1 (fractional change):  dO/O = k * sum_{b in calibrated, s} imp[b, s] delta[b, s]
    and dlnO = ln(1 + dO/O).  This is exact for the linear observables (y_mean, DM_mean) and first order for the
    quadratic ones; tested against the exact F engine (test_core: A2-i / A2-ii / AGN-kept truths).
    Bins outside 13.5-15.5 use the published fallback profiles in the truth and do not respond.
    Returns {observable: {'dlnO', 'dO_over_O', 'k', 'per_bin' [10] (contributions to dO/O), 'calibrated_share'
    (sum of imp over the calibrated cells: dO/O = k * share * delta for a uniform delta), 'missing_share' (|imp|
    share of the NaN cells), 'description'}}.
    """
    imp = _importance()
    d = np.asarray(dlnF, float)
    if d.ndim == 2 and d.shape[0] in (15, 16):
        d = d[list(CAL_BINS)]
    if d.shape != (len(CAL_BINS), 35):
        raise ValueError(f"dlnF must be [10, 35] (or [15|16, 35]), got {np.shape(dlnF)}")
    miss = ~np.isfinite(d)
    delta = np.where(miss, 0.0, np.expm1(np.where(miss, 0.0, d)))
    ell = imp["ell"]
    out = {}
    for name, (ikey, at, k, fld, desc) in OBSERVABLES.items():
        if fld != field or (observables is not None and name not in observables):
            continue
        M = imp[ikey]
        if at is not None:
            M = M[..., int(np.flatnonzero(ell == at)[0])]
        Mc = M[list(CAL_BINS)]
        per_bin = k * (Mc * delta).sum(axis=1)
        frac = float(per_bin.sum())
        absM = np.abs(Mc)
        out[name] = {"dlnO": float(np.log1p(frac)) if frac > -1 else float("nan"), "dO_over_O": frac, "k": k,
                     "per_bin": per_bin, "calibrated_share": float(Mc.sum()),
                     "missing_share": float(absM[miss].sum() / absM.sum()) if absM.sum() > 0 else 0.0,
                     "description": desc}
    if not out:
        raise ValueError(f"no observables for field {field!r} (use 'P' or 'E')")
    return out


def impact_from_stacks(F_variant, F_reference, field: str) -> dict:
    """observable_impact of dlnF = ln(F_variant / F_reference) ([10, 35] stacked means on the fit grid; cells where
    either is non-finite or <= 0 are NaN = missing)."""
    a, b = np.asarray(F_variant, float), np.asarray(F_reference, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where((a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b), np.log(a / b), np.nan)
    return observable_impact(d, field)


def impact_cross(dlnP, dlnE) -> dict:
    """y x DM cross spectrum (ell = 1000): pressure side + electron side, first order."""
    p = observable_impact(dlnP, "P", ["tSZxDM_total_ell1000_P"])["tSZxDM_total_ell1000_P"]
    e = observable_impact(dlnE, "E", ["tSZxDM_total_ell1000_E"])["tSZxDM_total_ell1000_E"]
    frac = p["dO_over_O"] + e["dO_over_O"]
    return {"dlnO": float(np.log1p(frac)), "dO_over_O": frac, "P_side": p["dO_over_O"], "E_side": e["dO_over_O"]}


# ============================================================================ JSON records
def _sig(x, sig=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{sig}g}")


def _sig_list(a, sig=4):
    a = np.asarray(a)
    if a.dtype == bool:
        return [bool(v) for v in a.ravel()]
    if np.issubdtype(a.dtype, np.integer):
        return [int(v) for v in a.ravel()]
    return [_sig(v, sig) for v in a.ravel()]


def record(variant_id: str, field: str, scheme: str, bin_label: str, x_edges, result: dict, N: int, N_eff: float,
           extra: dict | None = None, sig: int = 4) -> dict:
    """Compact JSON-ready dict for the website (numbers rounded to ``sig`` significant digits, NaN -> null)."""
    rec = {"variant": variant_id, "field": field, "scheme": scheme, "bin": bin_label, "x_edges": _sig_list(x_edges, sig),
           "value": _sig_list(result["value"], sig), "err": _sig_list(result["err"], sig),
           "q16": _sig_list(result["q16"], sig), "q84": _sig_list(result["q84"], sig),
           "n_used": _sig_list(result["n_used"], sig), "N": int(N), "N_eff": _sig(N_eff, sig)}
    for k in ("sd", "sd_ln", "n_eff"):
        if k in result:
            rec[k] = _sig_list(result[k], sig)
    for k, v in (extra or {}).items():
        rec[k] = _sig_list(v, sig) if isinstance(v, (np.ndarray, list, tuple)) else (
            _sig(v, sig) if isinstance(v, (float, np.floating, int, np.integer)) else v)
    return rec
