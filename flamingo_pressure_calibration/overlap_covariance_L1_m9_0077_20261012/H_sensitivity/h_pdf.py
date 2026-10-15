"""DM PDF passes of Test H through Test F's f_pdf (compound-Poisson halo model, z_s = 1), on the fine basis sets.

f_pdf is used unchanged except that its deposit function is replaced at run time (in this process and its forked
workers; no file is edited) by Test G's g_pdf.fast_deposit, which gives bitwise-identical histograms (checked by
g_pdf.deposit_identity_check in the 'truth' stage).  Entries: per calibrated bin c, rows F [k, 75] (fine shell
averages) with weights [k]; every label is one histogram.  The published-fallback haloes (identical in every
variant) deposit once into a slot that is added to every histogram before the FFT.
"""
from __future__ import annotations

import h_common as HC
import numpy as np

FC, FP = HC.FC, HC.FP
import g_pdf as GP  # noqa: E402  (Test G, read-only; configured through h_common's environment)

GROUP = 48 if not HC.SMOKE else 16      # histograms per f_pdf pass (parent memory: chunks x group x 0.5 MB)


def install():
    GP.install()


def identity_check():
    return GP.deposit_identity_check()


def run_specs(specs, which, fallback=None, include_fallback=False, keep_density=False, label="pdf"):
    """specs: list of (label, per_bin), per_bin[c] = (rows [k, ns] or [ns], weights [k] or scalar).
    Returns ({label: summary dict}, fallback (hist, overflow), timings)."""
    install()
    out, timings = {}, []
    fb = fallback
    for start in range(0, len(specs), GROUP):
        part = specs[start:start + GROUP]
        ent = FP.Entries(HC.N_FINE)
        for lab, per_bin in part:
            h = ent.new_hist(lab)
            for c in range(FC.N_CAL):
                rows, wts = per_bin[c]
                rows = np.atleast_2d(np.asarray(rows, float))
                ent.add(c, rows, h, wts)
        ent.freeze()
        first = fb is None
        HC.mem(f"before fork: {label} [{which}] group {start // GROUP + 1}")
        res = FP.run(ent, which, include_fallback=first or include_fallback, fallback=None if first else fb,
                     keep_density=keep_density)
        if first:
            fb = res["fallback"]
        for lab, s in zip(res["labels"], res["summaries"]):
            out[lab] = s
        timings.append(res["timing"])
        HC.log(f"{label} [{which}]: {min(start + GROUP, len(specs))}/{len(specs)} histograms, "
               f"{res['timing']['wall_seconds']:.0f} s, {res['timing']['entries']} entries")
    return out, fb, timings


def run_mecke(items, which, fallback, label="mecke"):
    """Mecke quantile Jacobians.  items: list of (label, ref_summary (with density), per_bin entries, include
    fallback pieces).  Returns {label: (J [5, N_CAL, 75], Jfb [5, 16, N_X])}, timing."""
    install()
    ent = FP.Entries(HC.N_FINE)
    mecke = []
    for lab, ref, per_bin, with_fb in items:
        h = ent.new_hist(lab)
        plan = {"fallback": bool(with_fb)}
        for c in range(FC.N_CAL):
            rows, wts = per_bin[c]
            plan[c] = ent.add(c, np.atleast_2d(np.asarray(rows, float)), h, wts)
        mecke.append((ref, plan))
    ent.freeze()
    # the fallback haloes' Mecke pieces are accumulated only while they deposit (f_pdf: include_fallback); the slot
    # they deposit is identical to ``fallback`` (it is overwritten by it when given)
    with_fallback = any(item[3] for item in items)
    HC.mem(f"before fork: {label} [{which}]")
    res = FP.run(ent, which, mecke=mecke, include_fallback=with_fallback, fallback=fallback)
    out = {}
    for (lab, ref, _, _), J, Jfb in zip(items, res["J"], res["Jfb"]):
        out[lab] = (J, Jfb)
    HC.log(f"{label} [{which}]: {len(items)} Mecke sets, {res['timing']['wall_seconds']:.0f} s")
    return out, res["timing"]


def det_entries(F):
    """One deterministic profile per calibrated bin (weight 1).  F [N_CAL, ns]."""
    return [(F[c], 1.0) for c in range(FC.N_CAL)]


def halo_entries(pr, key, rows_by_bin, weights_by_bin):
    """Per-halo entries: rows [k, ns] of pr[key] and weights normalised to unit sum per bin."""
    out = []
    for c in range(FC.N_CAL):
        r = rows_by_bin[c]
        w = np.asarray(weights_by_bin[c], float)
        out.append((np.nan_to_num(pr[key][r], nan=0.0), w / w.sum()))
    return out


def s1_entries(F, sigma2):
    """S1 lognormal amplitude mixture of the whole profile (9 Gauss-Hermite nodes); sigma2 [N_CAL]."""
    out = []
    for c in range(FC.N_CAL):
        t, w = np.polynomial.hermite.hermgauss(9)
        s = np.sqrt(max(float(sigma2[c]), 0.0))
        amp = np.exp(s * np.sqrt(2) * t - sigma2[c] / 2)
        out.append((amp[:, None] * np.asarray(F[c], float)[None, :], w / np.sqrt(np.pi)))
    return out


def pad(F):
    F = np.asarray(F, float)
    if F.shape[-1] == HC.N_FINE:
        return F
    return np.pad(F, [(0, 0)] * (F.ndim - 1) + [(0, HC.N_FINE - F.shape[-1])])
