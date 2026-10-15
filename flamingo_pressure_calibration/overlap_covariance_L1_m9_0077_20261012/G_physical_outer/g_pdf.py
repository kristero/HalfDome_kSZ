"""DM PDF passes for Test G through Test F's f_pdf (compound-Poisson halo model, z_s = 1), unchanged except for
one wrapped function:

  f_pdf._deposit is replaced at run time (in this process and its forked workers; the file is not edited) by
  fast_deposit, which bins only up to the largest touched DM index instead of adding two full 65,536-point
  arrays per (halo cell, histogram group).  The arithmetic is the same element by element, so the histograms are
  bitwise identical (checked on random input by deposit_identity_check, and end to end by the parity stage).

Entries: every model variant is one histogram; per calibrated bin it deposits
  det : the bin's shell profile with weight 1;
  S1  : 9 Gauss-Hermite lognormal amplitudes of the whole profile (Test F's 'lognormal' recipe);
  S2  : GH2 x GH2 correlated lognormal amplitudes (A_in, A_out) of the bound and outer components;
  CV  : the above for every fold model, weighted by the fold's held-out inclusion-weight share.
"""
from __future__ import annotations

import g_common as GC
import g_physics as PH
import numpy as np

import f_pdf as FP  # noqa: E402  (Test F, read-only)

DM_POINTS, DX = FP.DM_POINTS, FP.DX
_ORIGINAL_DEPOSIT = FP._deposit


def fast_deposit(X, W, h, hist, overflow):
    """f_pdf._deposit maths without the full-length minlength bincount (identical results for finite input).
    Non-finite or astronomically large DM values (a fitted power law diverging in the unresolved core, e.g. E1
    with s ~ 9 at R_fit <= 1.5) go to the overflow bin, as any DM beyond the grid does; f_pdf's own cast to int64
    would wrap them to negative indices."""
    pos = X.ravel() * (1.0 / DX)
    w = W.ravel()
    over = ~(pos < DM_POINTS - 1)
    if over.any():
        overflow[h] += float(w[over].sum())
        w = np.where(over, 0.0, w)
        pos = np.where(over, 0.0, pos)
    idx = pos.astype(np.int64)
    frac = pos - idx
    if idx.size == 0:
        return
    wf = w * frac
    a = np.bincount(idx, weights=w - wf)
    b = np.bincount(idx, weights=wf)
    n = len(a)
    hist[h, :n] += a
    hist[h, 1:n + 1] += b


def install():
    FP._deposit = fast_deposit


def deposit_identity_check(seed=GC.SEED, trials=20):
    """Max |difference| between the original and the fast deposit on random input (must be exactly 0)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in range(trials):
        n = int(rng.integers(1, 4000))
        X = rng.exponential(scale=[5.0, 50.0, 500.0, 3000.0][t % 4], size=(2, n))
        if t % 5 == 0:
            X[0, :10] = 9000.0                       # overflow entries
        W = rng.random((2, n))
        h1, o1 = np.zeros((3, DM_POINTS)), np.zeros(3)
        h2, o2 = np.zeros((3, DM_POINTS)), np.zeros(3)
        _ORIGINAL_DEPOSIT(X, W, 1, h1, o1)
        fast_deposit(X, W, 1, h2, o2)
        worst = max(worst, float(np.max(np.abs(h1 - h2))), float(np.max(np.abs(o1 - o2))))
    return worst


def run_hists(specs, fallback, group=GC.PDF_GROUP, label="pdf"):
    """specs: list of (label, per_bin) with per_bin[c] = (rows [k, ns] or [ns], weights [k] or scalar).
    Runs f_pdf in groups of <= ``group`` histograms (memory: 8 chunks x group x 0.5 MB in the parent).
    Returns ({label: quantiles}, {label: (mean, variance)}, timings)."""
    install()
    quant, moments, timings = {}, {}, []
    B_ns = GC.N_B
    for start in range(0, len(specs), group):
        part = specs[start:start + group]
        ent = FP.Entries(B_ns)
        for lab, per_bin in part:
            h = ent.new_hist(lab)
            for c in range(GC.N_CAL):
                rows, wts = per_bin[c]
                rows = np.atleast_2d(np.asarray(rows, float))
                if rows.shape[1] < B_ns:
                    rows = np.pad(rows, ((0, 0), (0, B_ns - rows.shape[1])))
                ent.add(c, rows, h, wts)
        ent.freeze()
        res = FP.run(ent, "T", include_fallback=False, fallback=fallback)
        for lab, s in zip(res["labels"], res["summaries"]):
            quant[lab] = np.asarray(s["quantiles"], float)
            moments[lab] = (float(s["mean"]), float(s["variance"]))
        timings.append(res["timing"])
        GC.log(f"{label}: {min(start + group, len(specs))}/{len(specs)} histograms, "
               f"{res['timing']['wall_seconds']:.0f} s")
    return quant, moments, timings


def det_entries(E):
    """Deterministic: one profile per bin, weight 1.  E [nb, ns]."""
    return [(E[c], 1.0) for c in range(GC.N_CAL)]


def s1_entries(E, s1):
    """S1: 9-node lognormal amplitude mixture of the whole profile; s1 [nb, 3] (sigma^2_E in column 1)."""
    out = []
    for c in range(GC.N_CAL):
        amp, wq = PH.gh1(s1[c, 1])
        out.append((amp[:, None] * E[c][None, :], wq))
    return out


def s2_entries(Bc, Oc, M):
    """S2: correlated (A_in, A_out) lognormal mixture on the bound / outer components; M [nb, 4, 4] moment
    matrix (electrons: indices 2, 3)."""
    out = []
    for c in range(GC.N_CAL):
        C2 = np.log(M[c][np.ix_([2, 3], [2, 3])])
        A, wq = PH.gh2(C2)
        out.append((A[:, 0:1] * Bc[c][None, :] + A[:, 1:2] * Oc[c][None, :], wq))
    return out


def mixture_entries(per_fold, weights_by_fold):
    """CV out-of-fold mixtures: per_fold[k] = per-bin entries list; weights_by_fold [nb, K] (held-out shares)."""
    out = []
    for c in range(GC.N_CAL):
        rows, wts = [], []
        for k, entries in enumerate(per_fold):
            if weights_by_fold[c, k] <= 0:
                continue
            r, w = entries[c]
            r = np.atleast_2d(np.asarray(r, float))
            rows.append(r)
            wts.append(np.broadcast_to(np.asarray(w, float), (len(r),)) * weights_by_fold[c, k])
        out.append((np.concatenate(rows), np.concatenate(wts)))
    return out
