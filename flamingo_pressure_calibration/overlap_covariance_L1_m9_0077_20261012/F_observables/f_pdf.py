"""F1b: compound-Poisson DM PDF of the halo-only model with per-halo and per-variant columns.

Mirrors engine.column_setup / column_terms / deposit / compound_poisson_pdf / pdf_summary (z_s = 1,
cells z <= 1, observer factor 1/(1+z), 0.125 pc cm^-3 CIC grid to 8192 with overflow) on the basis grid
of f_kernels.py, but each calibrated cell deposits the columns of many profiles at once:
  DM_e(b) = column_scale * sum_s F_e,s c_s(b),  c_s = projected basis function of shell s (x units),
for every *entry* e of the cell's mass bin: one entry per halo for the scatter-inclusive truth (its own
measured shell averages, weight = inclusion weight / bin total), one entry per deterministic model
variant, several entries (lognormal amplitude nodes, CV folds) for mixtures.  Entries deposit into
histograms (intensities); the published-fallback haloes deposit once into a shared slot that is added to
every histogram before the FFT.  Zero-DM deposits are skipped (they cancel exactly in the compound-Poisson
transform).  Mecke quantile Jacobians dQ/dln F[c, s] = sum lam w f(Q - x) g_s / f(Q) are accumulated for
requested (reference, entry) pairs (deterministic profiles or per-halo subsamples).
"""
from __future__ import annotations

import multiprocessing
import resource
import time

import f_common as FC
import f_kernels as K
import numpy as np

E = FC.E
A = E.A
DM_POINTS, DX = E.DM_POINTS, E.DM_DX
SUB = 96                      # entries per deposit sub-batch (memory bound)
_JOB: dict = {}


class Entries:
    """Per calibrated bin c: rows F [n_e, ns], histogram index h [n_e], weight w [n_e]."""

    def __init__(self, ns: int):
        self.ns = ns
        self._F = {c: [] for c in range(FC.N_CAL)}
        self._h = {c: [] for c in range(FC.N_CAL)}
        self._w = {c: [] for c in range(FC.N_CAL)}
        self.labels: list[str] = []
        self.frozen = False

    def new_hist(self, label: str) -> int:
        self.labels.append(label)
        return len(self.labels) - 1

    def add(self, c: int, F, hist: int, weights) -> np.ndarray:
        """Append rows for bin c; returns their entry indices."""
        F = np.atleast_2d(np.asarray(F, float))
        if F.shape[1] < self.ns:
            F = np.pad(F, ((0, 0), (0, self.ns - F.shape[1])))
        start = sum(len(x) for x in self._F[c])
        self._F[c].append(F)
        self._h[c].append(np.full(len(F), hist, int))
        self._w[c].append(np.broadcast_to(np.asarray(weights, float), (len(F),)).copy())
        return np.arange(start, start + len(F))

    def freeze(self):
        self.F, self.h, self.w, self.groups = {}, {}, {}, {}
        for c in range(FC.N_CAL):
            self.F[c] = np.concatenate(self._F[c]) if self._F[c] else np.zeros((0, self.ns))
            self.h[c] = np.concatenate(self._h[c]) if self._h[c] else np.zeros(0, int)
            self.w[c] = np.concatenate(self._w[c]) if self._w[c] else np.zeros(0)
            groups = []
            for hist in np.unique(self.h[c]):
                idx = np.flatnonzero(self.h[c] == hist)
                for s in range(0, len(idx), SUB):
                    groups.append((int(hist), idx[s:s + SUB]))
            self.groups[c] = groups
        self.frozen = True
        return self


def _deposit(X, W, h, hist, overflow):
    """Cloud-in-cell deposit (engine.deposit maths).  Zero-DM nodes deposit at DM = 0, which cancels
    exactly in the compound-Poisson transform (exp(FFT(L) - L_tot)); overflow beyond the grid is tracked."""
    pos = X.ravel() * (1.0 / DX)
    w = W.ravel()
    idx = pos.astype(np.int64)
    frac = pos - idx
    over = idx >= DM_POINTS - 1
    if over.any():
        overflow[h] += float(w[over].sum())
        w = np.where(over, 0.0, w)
        idx = np.where(over, 0, idx)
    wf = w * frac
    hist[h] += np.bincount(idx, weights=w - wf, minlength=DM_POINTS)
    hist[h, 1:] += np.bincount(idx, weights=wf, minlength=DM_POINTS)[:-1]


def _zshell(j, acc):
    job = _JOB
    sky, grid = job["sky"], job["grid"]
    x = grid.x
    index = np.flatnonzero(sky.counts[j] > 0)
    zz = sky.z[j]
    if not len(index) or zz > 1.:
        return
    mass = sky.mass[index]
    N = sky.counts[j, index].astype(float)
    da, radius, y_amp, tau_amp = sky.ingredients.geometry(zz, mass)
    dm_scale = 1. / (E.DM_FACTOR * (1 + zz))
    column_scale = tau_amp * (da / radius)**2 * dm_scale
    area_scale = N / (4 * np.pi) * (radius / da)**2
    safe, valid = job["profile"].bins(mass)
    cal, fb = np.flatnonzero(valid), np.flatnonzero(~valid)
    cidx = job["cal_index"][safe[cal]]
    evolution = A.published_profile(1, mass, zz, x) / A.published_profile(1, mass, 0., x)
    A_seg, area, seg_shell = job["A_seg"], job["area"], job["seg_shell"]
    ent = job["entries"]
    ns, n_seg = job["ns"], job["n_seg"]
    for c in np.unique(cidx):
        cells = cal[cidx == c]
        if not len(ent.F[c]):
            continue
        prof = job["g"][c] * evolution[cells]
        col = np.zeros((len(cells), ns, len(area)))
        for s_i, (a, b) in enumerate(grid.segments[:n_seg]):
            col[:, seg_shell[s_i]] += prof[:, a:b + 1] @ A_seg[s_i].T
        col *= (column_scale[cells][:, None] / job["gbar"][c][None, :])[:, :, None]   # DM per unit F_s
        lam = area_scale[cells][:, None] * area[None, :]
        for hist, idx in ent.groups[c]:
            X = np.tensordot(ent.F[c][idx], col, axes=([1], [1]))                 # [e, m, k]
            _deposit(X, ent.w[c][idx][:, None, None] * lam[None], hist, acc["hist"], acc["overflow"])
        for r, (ref, plan) in enumerate(job["mecke"]):
            rows = plan.get(c)
            if rows is None or not len(rows):
                continue
            for s0 in range(0, len(rows), 32):
                sub = rows[s0:s0 + 32]
                Fm = ent.F[c][sub]
                X = np.tensordot(Fm, col, axes=([1], [1]))                      # [e, m, k]
                T = np.stack([np.interp(q - X, job["dm_grid"], ref["density"], left=0., right=0.)
                              for q in ref["quantiles"]])                         # [p, e, m, k]
                T *= ent.w[c][sub][None, :, None, None] * lam[None, None]
                Vv = np.tensordot(T, Fm, axes=([1], [0]))                        # [p, m, k, s]
                acc["J"][r][:, c] += np.einsum("pmks,msk->ps", Vv, col, optimize=True) \
                    / ref["density_at_quantiles"][:, None]
    if job["include_fallback"] and len(fb):
        onehot = np.zeros((len(fb), 16))
        onehot[np.arange(len(fb)), sky.mass_bin[index][fb]] = 1.
        prof_all = A.published_profile(1, mass[fb], 0., x) * evolution[fb]
        for start in range(0, len(fb), K.BATCH):
            sl = slice(start, start + K.BATCH)
            prof = prof_all[sl]
            col = np.zeros((prof.shape[0], FC.N_F, len(area)))
            for s_i, (a, b) in enumerate(grid.segments[:job["n_seg_4"]]):
                col[:, seg_shell[s_i]] += prof[:, a:b + 1] @ A_seg[s_i].T
            col *= column_scale[fb][sl][:, None, None]
            X = col.sum(axis=1)
            lam = area_scale[fb][sl][:, None] * area[None, :]
            _deposit(X, lam, job["fallback_slot"], acc["hist"], acc["overflow"])
            for r, (ref, plan) in enumerate(job["mecke"]):
                if not plan.get("fallback", False):
                    continue
                dens = np.stack([np.interp(q - X, job["dm_grid"], ref["density"], left=0., right=0.)
                                 for q in ref["quantiles"]])                     # [p, m, k]
                acc["Jfb"][r] += np.einsum("pmk,mk,mb,msk->pbs", dens, lam, onehot[sl], col, optimize=True) \
                    / ref["density_at_quantiles"][:, None, None]


def _chunk(shells):
    job = _JOB
    nh = job["n_hist"] + 1
    acc = {"hist": np.zeros((nh, DM_POINTS)), "overflow": np.zeros(nh),
           "J": [np.zeros((len(FC.QUANTILES), FC.N_CAL, job["ns"])) for _ in job["mecke"]],
           "Jfb": [np.zeros((len(FC.QUANTILES), 16, FC.N_F)) for _ in job["mecke"]]}
    for j in shells:
        _zshell(j, acc)
    return acc


def summarise(intensity, overflow):
    pmf = E.compound_poisson_pdf(intensity, overflow)
    s = E.pdf_summary(pmf)
    return {"quantiles": s["quantiles"], "mean": s["mean"], "variance": s["variance"], "density": s["density"],
            "density_at_quantiles": s["density_at_quantiles"], "mass_on_grid": s["mass_on_grid"],
            "negative_mass": s["negative_mass"]}


def run(entries: Entries, which: str = "T", mecke=(), include_fallback: bool = True, fallback=None,
        workers: int = FC.WORKERS, chunks_per_worker: int = 2, keep_density: bool = False,
        return_raw: bool = False) -> dict:
    """Deposit every entry; returns per-histogram PDF summaries (+ Mecke Jacobians).

    mecke: sequence of (reference summary dict, plan) with plan = {c: entry indices of bin c, 'fallback': bool}.
    fallback: (hist, overflow) of the fallback slot from an earlier pass (then include_fallback=False).
    """
    assert entries.frozen
    started = time.perf_counter()
    sky = FC.make_sky()
    base, grid, seg_shell, n_seg_4 = K.basis_grid()
    support = FC.SUPPORT
    profile = E.Profile(1, "bins", support["mass_edges"], np.asarray(support["support"], bool), support["max_logmass"])
    cal_index = np.full(len(support["support"]), -1)
    cal_index[FC.CAL_BINS] = np.arange(FC.N_CAL)
    z = np.load(FC.WORK / f"basis_{which}.npz")
    ns = int(z["n_shell"])
    assert entries.ns == ns
    cs = E.column_setup(grid, FC.X_END, True)
    n_hist = len(entries.labels)
    _JOB.clear()
    _JOB.update(sky=sky, grid=grid, seg_shell=seg_shell, n_seg_4=n_seg_4, n_seg=int(z["n_seg"]), ns=ns,
                profile=profile, cal_index=cal_index, g=z["g_E"], gbar=z["gbar_E"], A_seg=cs["A_seg"],
                area=cs["area"], entries=entries, n_hist=n_hist, fallback_slot=n_hist,
                include_fallback=include_fallback, mecke=list(mecke), dm_grid=np.arange(DM_POINTS) * DX)
    shells = [j for j in range(len(sky.z)) if sky.z[j] <= 1.]
    sub = sky.counts.copy()
    sub[sky.z > 1.] = 0
    sky_cost = type("S", (), {"counts": sub, "z": sky.z})
    chunks = E._balanced_chunks(sky_cost, workers * chunks_per_worker)
    results = []
    for k, item in enumerate(FC.pool_map(_chunk, chunks, workers)):
        results.append(item)
        FC.log(f"pdf pass ({which}, {n_hist} hists): chunk {k + 1}/{len(chunks)}, {time.perf_counter() - started:.0f} s")
    _JOB.clear()
    hist = sum(r["hist"] for r in results)
    overflow = sum(r["overflow"] for r in results)
    J = [sum(r["J"][i] for r in results) for i in range(len(mecke))]
    Jfb = [sum(r["Jfb"][i] for r in results) for i in range(len(mecke))]
    if fallback is not None:
        hist[n_hist] = fallback[0]
        overflow[n_hist] = fallback[1]
    summaries = []
    for h in range(0 if return_raw else n_hist):
        s = summarise(hist[h] + hist[n_hist], overflow[h] + overflow[n_hist])
        if not keep_density:
            s = {k: v for k, v in s.items() if k != "density"}
        summaries.append(s)
    timing = {"wall_seconds": time.perf_counter() - started, "workers": workers, "histograms": n_hist,
              "entries": int(sum(len(entries.F[c]) for c in range(FC.N_CAL))),
              "peak_rss_parent_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "peak_rss_largest_child_GiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2}
    FC.log(f"pdf pass done: {timing}")
    out = {"labels": list(entries.labels), "summaries": summaries, "J": J, "Jfb": Jfb,
           "fallback": (hist[n_hist].copy(), float(overflow[n_hist])), "timing": timing,
           "hist_sums": hist.sum(axis=1), "overflow": overflow}
    if return_raw:
        out["raw_hist"], out["raw_overflow"] = hist[:n_hist], overflow[:n_hist]
    return out
