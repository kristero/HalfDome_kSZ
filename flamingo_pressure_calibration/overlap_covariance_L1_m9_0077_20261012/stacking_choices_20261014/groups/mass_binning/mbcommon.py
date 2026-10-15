"""Shared helpers of the 'mass_binning' group (stacking_choices_20261014/groups/mass_binning).

Everything is computed from the validated stacking core (core/stackcore.py) and its cache; nothing outside this
group directory is written.

Conventions used throughout the group
-------------------------------------
* Reference: sc.make_key(field, grid='fit') (T_e / K: estimator 'weighted', the only one cached for them), statistic
  'mean' (inclusion-weighted), rule 'strict' (min_count 20, softening_factor 4, hsml_factor 2, min_edge 0.04),
  scheme 'M200c_0.2', all haloes of the bin.
* Web records of every variant use sc.stack_bin (core default seeds -> identical 'ref' records in every group).
* Comparisons across *different* bin sets (schemes, sub-bins, pooled bins) need replicates that are paired halo by
  halo across all bins.  They use ONE global bootstrap of the whole sample (200 draws, seed GLOBAL_SEED): a bin's
  replicate stack is the inclusion-weighted mean over the bin's haloes with the global multiplicities, so every bin
  of every scheme is resampled from the same draws.
* Bootstrap design ('census', the default here): in the randomly sampled 0.2-dex bins (12.5-14.5, inclusion weights
  != 1) resampling is stratified by the design strata mass_bin x class_main; in the complete-census bins (14.5-15.5,
  every weight = 1) it is plain resampling within the 0.2-dex bin.  The core's convention ('core', sc.stack_bin with
  stratified=True) also stratifies the census bins by class_main, whose strata hold 17/2/2 (15.1-15.3) and 2/3/1
  (15.3-15.5) haloes; a 1-halo stratum never varies, so that design understates the top bins' sampling error.  The
  web records keep the core convention (identical 'ref' records in every group); results.json errors use 'census'.
* Continuous painting (continuous_paint): every calibrated halo is painted with the scheme's node stacks
  interpolated (ln F linear in mean log M200c) to its OWN log M200c; the painted bin mean is compared with the true
  bin mean (the reference stack = mean of the haloes' own profiles).  No scheme is exact by construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/"
           "stacking_choices_20261014")
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

GROUP = OUT / "groups" / "mass_binning"
WORK = GROUP / "work"
FIG = GROUP / "figures"
WORK.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

N_BOOT = 200
GLOBAL_SEED = sc.STUDY_SEED + 4242
FIELDS_WEB = ("P", "E", "G", "T_e", "K")
OBS_FIELDS = ("P", "E")
REF_SCHEME = "M200c_0.2"
CAL_LABELS = ("13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9",
              "14.9-15.1", "15.1-15.3", "15.3-15.5")
BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))
BAND_LABELS = ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")
OBS_KEEP = {"P": ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "y_mean"),
            "E": ("kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var")}
RULE = dict(rule="strict", min_count=20, softening_factor=4.0, hsml_factor=2.0)
# Okabe-Ito-like categorical order of the dataviz reference palette (validated adjacent-pair CVD set)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def ref_key(field: str, grid: str = "fit") -> str:
    est = "weighted" if field in ("T_e", "K", "T_M", "T_V", "T_X", "C_rho", "C_P") else "geometric"
    return sc.make_key(field, estimator=est, grid=grid)


def xvol(e) -> np.ndarray:
    """Volume-weighted shell radius 0.75 (b^4 - a^4) / (b^3 - a^3)."""
    e = np.asarray(e, float)
    a, b = e[:-1], e[1:]
    return 0.75 * (b**4 - a**4) / (b**3 - a**3)


EDGES_FIT = np.asarray(sc.edges("fit"), float)
X_FIT = xvol(EDGES_FIT)
BAND_IDX = [np.flatnonzero((X_FIT >= lo) & (X_FIT < hi)) for lo, hi in BANDS]

TAB = sc.halo_table()
W = np.asarray(TAB["weight"], float)
NH = len(W)
STRATA_CORE = np.asarray(TAB["mass_bin"] * 3 + TAB["class_main"])
CENSUS_BINS = np.array(sorted({int(m) for m in np.unique(TAB["mass_bin"])
                               if np.all(W[TAB["mass_bin"] == m] == 1.0)}))
IS_CENSUS = np.isin(TAB["mass_bin"], CENSUS_BINS)
STRATA_CENSUS = np.where(IS_CENSUS, TAB["mass_bin"] * 3, STRATA_CORE)
STRATA = STRATA_CENSUS
DESIGNS = {"census": STRATA_CENSUS, "core": STRATA_CORE}
_C = {}


def global_counts(design: str = "census") -> np.ndarray:
    """[N_BOOT, 4588] multiplicities of the global bootstrap (paired across every bin and scheme).  design 'census'
    (default): mass_bin x class_main strata in the sampled bins, plain within-bin resampling in the census bins;
    'core': mass_bin x class_main everywhere (the core's stack_bin convention)."""
    if design not in _C:
        _C[design] = sc._bootstrap_counts(NH, N_BOOT, GLOBAL_SEED, strata=DESIGNS[design])
    return _C[design]


def load(key: str) -> np.ndarray:
    return np.asarray(sc.per_halo(key), float)


def wmean_boot(F: np.ndarray, rows, mask=None, weights=None, design: str = "census") -> tuple:
    """Inclusion-weighted mean of F[rows] (NaN skipped, optional [S] or [n,S] mask) and its global-bootstrap
    replicates.  Returns (value [S], boot [N_BOOT, S], n_used [S])."""
    rows = np.asarray(rows)
    v = F[rows]
    ok = np.isfinite(v)
    if mask is not None:
        m = np.asarray(mask, bool)
        ok &= np.broadcast_to(m if m.ndim == 2 else m[None, :], v.shape)
    work = np.where(ok, v, 0.0)
    okf = ok.astype(float)
    w = W[rows] if weights is None else np.asarray(weights, float)
    Wd = np.vstack([w[None, :], global_counts(design)[:, rows] * w[None, :]])
    with np.errstate(divide="ignore", invalid="ignore"):
        num = Wd @ work
        den = Wd @ okf
        allv = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return allv[0], allv[1:], ok.sum(axis=0)


def strict_mask(rows, grid="fit") -> np.ndarray:
    return np.asarray(sc.resolution_mask(rows, grid, "strict"), bool)


def interp_lnF(m_nodes, F_nodes, m_target, extrapolate=True):
    """Linear interpolation of ln F in log mass between nodes (sorted).  F_nodes [..., n_nodes, S] (leading axes =
    bootstrap).  Returns (F [..., S], segment index, t, extrapolated flag).  Non-positive / non-finite -> NaN."""
    m_nodes = np.asarray(m_nodes, float)
    order = np.argsort(m_nodes)
    m_nodes = m_nodes[order]
    F_nodes = np.asarray(F_nodes, float)[..., order, :]
    i = int(np.clip(np.searchsorted(m_nodes, m_target) - 1, 0, len(m_nodes) - 2))
    t = (m_target - m_nodes[i]) / (m_nodes[i + 1] - m_nodes[i])
    extr = bool(t < 0 or t > 1)
    if extr and not extrapolate:
        t = float(np.clip(t, 0, 1))
    a = F_nodes[..., i, :]
    b = F_nodes[..., i + 1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        good = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
        out = np.where(good, np.exp((1 - t) * np.log(np.where(good, a, 1.0)) + t * np.log(np.where(good, b, 1.0))),
                       np.nan)
    return out, int(order[i]), float(t), extr


def continuous_paint(m_nodes, F_nodes, B_nodes, cal_bins, design: str = "census", interp: str = "linear"):
    """Continuous painting through a set of node stacks.

    m_nodes [K] (mean log M200c of the nodes), F_nodes [K, S], B_nodes [K, N_BOOT, S] (paired replicates).
    cal_bins: list of mass_bins() dicts (the calibrated 0.2-dex M200c bins).  Every halo h of a calibrated bin gets
    F(m_h) = exp(ln F linear in log M200c between the two bracketing nodes; linear extrapolation beyond the end
    nodes); the painted bin profile is the inclusion-weighted mean over the bin's haloes (replicate r: weights x the
    global multiplicities of draw r, the same draws as the node replicates).
    interp 'pchip': ln F monotone-cubic (PCHIP) in log M200c through the nodes instead of linear (beyond the end
    nodes: linear extrapolation with the end-segment chord slope, as for 'linear').
    Returns (painted [n_cal, S], painted replicates [N_BOOT, n_cal, S], n_extrapolated_haloes [n_cal])."""
    m_nodes = np.asarray(m_nodes, float)
    order = np.argsort(m_nodes)
    m_nodes = m_nodes[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        LF = np.log(np.where(np.asarray(F_nodes, float)[order] > 0, np.asarray(F_nodes, float)[order], np.nan))
        LB = np.log(np.where(np.asarray(B_nodes, float)[order] > 0, np.asarray(B_nodes, float)[order], np.nan))
    C = global_counts(design)
    if interp == "pchip":
        return _continuous_paint_pchip(m_nodes, LF, LB, cal_bins, C)
    out, outb, n_ext = [], [], []
    for b in cal_bins:
        rows = np.asarray(b["rows"])
        m = np.asarray(TAB["logm200c"][rows], float)
        w = W[rows]
        k = np.clip(np.searchsorted(m_nodes, m) - 1, 0, len(m_nodes) - 2)
        t = (m - m_nodes[k]) / (m_nodes[k + 1] - m_nodes[k])
        n_ext.append(int(((t < 0) | (t > 1)).sum()))
        num = np.zeros(LF.shape[1])
        numb = np.zeros((C.shape[0], LF.shape[1]))
        wb = C[:, rows] * w[None, :]                                   # [B, n]
        for kk in np.unique(k):
            sel = np.flatnonzero(k == kk)
            tt = t[sel][:, None]
            prof = np.exp((1 - tt) * LF[kk][None, :] + tt * LF[kk + 1][None, :])          # [n_sel, S]
            num += w[sel] @ prof
            for r in range(C.shape[0]):
                pr = np.exp((1 - tt) * LB[kk, r][None, :] + tt * LB[kk + 1, r][None, :])
                numb[r] += wb[r, sel] @ pr
        out.append(num / w.sum())
        outb.append(numb / wb.sum(axis=1)[:, None])
    return np.array(out), np.moveaxis(np.array(outb), 0, 1), np.array(n_ext)


def _continuous_paint_pchip(m_nodes, LF, LB, cal_bins, C):
    """PCHIP version of continuous_paint (ln F in log M200c; shells with a non-finite node value fall back to linear
    interpolation through the finite nodes)."""
    from scipy.interpolate import PchipInterpolator

    def lnF_at(m_eval, Y):                       # Y [K, X] -> [n_eval, X]
        out = np.full((len(m_eval), Y.shape[1]), np.nan)
        fin = np.all(np.isfinite(Y), axis=0)
        inside = (m_eval >= m_nodes[0]) & (m_eval <= m_nodes[-1])
        if fin.any():
            f = PchipInterpolator(m_nodes, Y[:, fin], axis=0, extrapolate=False)
            out[np.ix_(inside, fin)] = f(m_eval[inside])
            lo, hi = m_eval < m_nodes[0], m_eval > m_nodes[-1]
            s0 = (Y[1, fin] - Y[0, fin]) / (m_nodes[1] - m_nodes[0])
            s1 = (Y[-1, fin] - Y[-2, fin]) / (m_nodes[-1] - m_nodes[-2])
            out[np.ix_(lo, fin)] = Y[0, fin][None] + (m_eval[lo] - m_nodes[0])[:, None] * s0[None]
            out[np.ix_(hi, fin)] = Y[-1, fin][None] + (m_eval[hi] - m_nodes[-1])[:, None] * s1[None]
        for j in np.flatnonzero(~fin):
            ok = np.isfinite(Y[:, j])
            if ok.sum() >= 2:
                mm, yy = m_nodes[ok], Y[ok, j]
                k = np.clip(np.searchsorted(mm, m_eval) - 1, 0, len(mm) - 2)
                t = (m_eval - mm[k]) / (mm[k + 1] - mm[k])
                out[:, j] = (1 - t) * yy[k] + t * yy[k + 1]
        return out

    Bn, S = LB.shape[1], LF.shape[1]
    out, outb, n_ext = [], [], []
    for b in cal_bins:
        rows = np.asarray(b["rows"])
        m = np.asarray(TAB["logm200c"][rows], float)
        w = W[rows]
        n_ext.append(int(((m < m_nodes[0]) | (m > m_nodes[-1])).sum()))
        prof = np.exp(lnF_at(m, LF))                                              # [n, S]
        out.append((w @ prof) / w.sum())
        allb = np.exp(lnF_at(m, LB.reshape(LB.shape[0], Bn * S))).reshape(len(m), Bn, S)    # [n, B, S]
        wb = C[:, rows] * w[None, :]
        outb.append(np.einsum("bn,nbs->bs", wb, allb) / wb.sum(axis=1)[:, None])
    return np.array(out), np.moveaxis(np.array(outb), 0, 1), np.array(n_ext)


def band_ratios(v_var, v_ref, b_var, b_ref, resolved) -> dict:
    """Per radial band: median over resolved shells of v_var/v_ref and the SD of the same statistic over the paired
    replicates.  Returns {band_label: {'ratio', 'err', 'q16', 'q84', 'n_shells'} or None}."""
    resolved = np.asarray(resolved, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        r0 = np.asarray(v_var, float) / np.asarray(v_ref, float)
        rb = np.asarray(b_var, float) / np.asarray(b_ref, float)
    out = {}
    for lab, idx in zip(BAND_LABELS, BAND_IDX):
        sel = idx[resolved[idx] & np.isfinite(r0[idx])]
        if not len(sel):
            out[lab] = None
            continue
        val = float(np.median(r0[sel]))
        bs = np.nanmedian(rb[:, sel], axis=1)
        bs = bs[np.isfinite(bs)]
        out[lab] = {"ratio": sig(val), "err": sig(np.std(bs, ddof=1)) if len(bs) > 2 else None,
                    "q16": sig(np.percentile(bs, 16)) if len(bs) > 2 else None,
                    "q84": sig(np.percentile(bs, 84)) if len(bs) > 2 else None, "n_shells": int(len(sel))}
    return out


def impacts(F_var, F_ref, B_var, B_ref, field) -> dict:
    """First-order observable impact (sc.impact_from_stacks) of [10, 35] stacks, with paired-bootstrap SD.
    Values in per cent (dO/O x 100)."""
    keep = OBS_KEEP[field]
    base = sc.impact_from_stacks(F_var, F_ref, field)
    reps = {k: [] for k in keep}
    for b in range(B_var.shape[0]):
        r = sc.impact_from_stacks(B_var[b], B_ref[b], field)
        for k in keep:
            reps[k].append(r[k]["dO_over_O"])
    out = {}
    for k in keep:
        arr = np.asarray(reps[k], float)
        arr = arr[np.isfinite(arr)]
        out[k] = {"pct": sig(100 * base[k]["dO_over_O"]), "err_pct": sig(100 * np.std(arr, ddof=1)),
                  "missing_share": sig(base[k]["missing_share"]),
                  "per_bin_pct": dict(zip(CAL_LABELS, [sig(100 * v) for v in base[k]["per_bin"]]))}
    return out


def sig(x, n=4):
    return sc._sig(x, n)


def sig_list(a, n=4):
    return sc._sig_list(a, n)


def web_record(variant_id, field, scheme, bin_label, logm, N, N_eff, edges, value, err, q16, q84, n_used, resolved,
               x_unit="R200c") -> dict:
    edges = np.asarray(edges, float)
    return {"variant": variant_id, "field": field, "scheme": scheme, "bin": bin_label, "logm": sig(logm),
            "N": int(N), "N_eff": sig(N_eff), "x_unit": x_unit, "edges": sig_list(edges), "x": sig_list(xvol(edges)),
            "value": sig_list(value), "err": sig_list(err), "q16": sig_list(q16), "q84": sig_list(q84),
            "n_used": [int(v) for v in np.asarray(n_used).ravel()],
            "resolved": [int(bool(v)) for v in np.asarray(resolved).ravel()]}


def record_from_stack(variant_id, field, scheme, b, res, logm=None) -> dict:
    """Web record from an sc.stack_bin / sc.stack result and a mass_bins() dict."""
    return web_record(variant_id, field, scheme, b["label"], b["mean_log_mass"] if logm is None else logm, b["N"],
                      b["N_eff"], res.get("edges", EDGES_FIT), res["value"], res["err"], res["q16"], res["q84"],
                      res["n_used"], res["resolved"])


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False, default=_default))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return sig(o)
    if isinstance(o, np.ndarray):
        return sig_list(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
