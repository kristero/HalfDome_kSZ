"""DM-PDF passes of Test J through Test F's f_pdf (compound-Poisson halo model, z_s = 1, observer frame).

f_pdf is used unchanged except for two run-time settings made in this process (and inherited by its forked workers):
  * its deposit function is Test G's g_pdf.fast_deposit (bitwise-identical histograms; checked in 'truth');
  * j_common.fallback_cut(R): the published-fallback haloes deposit their projected columns cut in 3D at R R200c
    (1, 2 or 4) instead of always 4 R200c;
and with the calibrated support of the configuration (j_common.support: C14, C10 or C0).
The compound-Poisson model: a sightline to z_s = 1 crosses Poisson-many haloes; halo population (z, M, impact parameter b)
contributes an intensity lambda = N/(4 pi) (R200c/d_A)^2 area_b (expected crossings) at DM(b) = column_scale x Sigma(b),
Sigma(b) = 2 int_0^{sqrt(X^2 - b^2)} F(sqrt(b^2 + l^2)) dl the projected column of the (cut) profile F in R200c units,
column_scale = tau amplitude (d_A/R200c)^2 / (DM_FACTOR (1 + z)) [pc cm^-3].  The PDF of the summed DM is
p = IFFT(exp(FFT(Lambda) - Lambda_tot)) on a 0.125 pc cm^-3 grid to 8192 (Lambda: the deposited intensity histogram).
Entries per calibrated bin c: rows F [k, 75] (fine shell averages) with weights [k] summing to 1 per bin (one histogram
per label); the fallback haloes of the configuration deposit once into a slot shared by every histogram of the pass.
"""
from __future__ import annotations

import j_common as JC
import numpy as np

FC, FP, GP = JC.FC, JC.FP, JC.GP
GROUP = 48 if not JC.SMOKE else 16
KEEP = 2**14                      # density points kept (to 2048 pc cm^-3)
_FALLBACK: dict = {}


def _summ(s, keep_density):
    out = {k: s[k] for k in ("quantiles", "mean", "variance", "mass_on_grid", "negative_mass")}
    if keep_density or "density" in s:
        d = np.asarray(s["density"], float)
        cdf = np.cumsum(d) * FP.DX
        out["cdf_at_1pc"] = float(np.interp(1.0, np.arange(len(d)) * FP.DX + 0.5 * FP.DX, cdf))
        out["p_first_bin"] = float(d[0] * FP.DX)
    return out


def run_specs(specs, which, cut, config, keep_density=False, keep_full=(), label="pdf"):
    """specs: list of (label, per_bin), per_bin[c] = (rows [k, ns] or [ns], weights [k] or scalar) for the bins of
    ``config``.  Returns ({label: summary}, timings); summaries keep 'density' (float32, first KEEP points) if asked,
    and the full f_pdf summary under '_full' for the labels in ``keep_full`` (Mecke references)."""
    out, timings = {}, []
    with JC.support(config), JC.fallback_cut(cut) as n_cut:
        GP.install()
        key = (config, cut)
        fb = _FALLBACK.get(key)
        for start in range(0, len(specs), GROUP):
            part = specs[start:start + GROUP]
            ent = FP.Entries(JC.N_FINE)
            for lab, per_bin in part:
                h = ent.new_hist(lab)
                for c in range(FC.N_CAL):
                    rows, wts = per_bin[c]
                    ent.add(c, np.atleast_2d(np.asarray(rows, float)), h, wts)
            ent.freeze()
            first = fb is None
            JC.mem(f"before fork: {label} [{which}, {config}, cut {cut:g}] group {start // GROUP + 1}")
            res = FP.run(ent, which, include_fallback=first, fallback=None if first else fb, keep_density=True)
            if first:
                fb = res["fallback"]
                _FALLBACK[key] = fb
            for lab, s in zip(res["labels"], res["summaries"]):
                d = _summ(s, True)
                if keep_density:
                    d["density"] = np.asarray(s["density"][:KEEP], np.float32)
                if lab in keep_full:
                    d["_full"] = s
                out[lab] = d
            res["timing"]["n_seg_fallback"] = n_cut
            timings.append(res["timing"])
            JC.log(f"{label} [{which}, {config}, cut {cut:g}]: {min(start + GROUP, len(specs))}/{len(specs)} histograms, "
                   f"{res['timing']['wall_seconds']:.0f} s, {res['timing']['entries']} entries")
            del res, ent
    return out, timings


def run_mecke(items, which, cut, config, label="mecke"):
    """Mecke quantile Jacobians dQ_p/dln F[c, s] (f_pdf).  items: list of (label, reference summary WITH density
    (full length), per_bin entries).  The fallback haloes' pieces are always accumulated.
    Returns {label: (J [5, N_CAL, 75], Jfb [5, 16, 67])}, timing."""
    with JC.support(config), JC.fallback_cut(cut):
        GP.install()
        ent = FP.Entries(JC.N_FINE)
        mecke = []
        for lab, ref, per_bin in items:
            h = ent.new_hist(lab)
            plan = {"fallback": True}
            for c in range(FC.N_CAL):
                rows, wts = per_bin[c]
                plan[c] = ent.add(c, np.atleast_2d(np.asarray(rows, float)), h, wts)
            mecke.append((ref, plan))
        ent.freeze()
        JC.mem(f"before fork: {label} [{which}, {config}, cut {cut:g}]")
        res = FP.run(ent, which, mecke=mecke, include_fallback=True, fallback=_FALLBACK.get((config, cut)))
    out = {lab: (J, Jfb) for (lab, _, _), J, Jfb in zip(items, res["J"], res["Jfb"])}
    JC.log(f"{label} [{which}, {config}, cut {cut:g}]: {len(items)} Mecke sets, {res['timing']['wall_seconds']:.0f} s")
    return out, res["timing"]


def det_entries(F):
    """One deterministic profile per calibrated bin of the active configuration, weight 1.  F [n_bins, ns]."""
    return [(JC.pad75(F[c]), 1.0) for c in range(len(F))]


def halo_entries(rows_by_bin, weights_by_bin):
    out = []
    for r, w in zip(rows_by_bin, weights_by_bin):
        w = np.asarray(w, float)
        out.append((JC.pad75(np.nan_to_num(np.atleast_2d(r), nan=0.0)), w / w.sum()))
    return out

