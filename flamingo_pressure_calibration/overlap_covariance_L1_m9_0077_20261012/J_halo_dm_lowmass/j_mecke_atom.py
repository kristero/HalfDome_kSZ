"""Test J1 revision (2026-10-16): Mecke quantile Jacobians that count the whole DM = 0 atom.

Why.  Every sightline DM of the compound-Poisson halo model is linear in the painted profile values, and a sightline
that crosses no halo keeps DM = 0 under any rescaling.  Scaling every profile by alpha therefore scales every quantile
outside the atom by alpha: Q_p(alpha F) = alpha Q_p(F), and by Euler's theorem the exact importance map of such a
quantile, I[b, s] = (dQ_p/d ln F_{b,s}) / Q_p, sums to 1 over bins and shells, atom or not.
Mecke (Slivnyak-Mecke) identity for the sum S of the haloes' contributions x (Poisson intensity lambda(dx)):
    E[sum_i g(x_i) delta(S - Q)] = int lambda(dx) g(x) f(Q - x),
where f is the FULL law of S, including the atom pi_0 delta(.) at S = 0 (pi_0 = P(no crossing) = exp(-Lambda_tot)).
Hence dQ_p/d eps = int lambda g f(Q_p - x) dx / f(Q_p) for x -> x + eps g, and with g = x (all profiles scaled)
int lambda x f(Q - x) = E[S delta(S - Q)] = Q f(Q): the sum rule holds exactly.  The atom contributes
    pi_0 * [lambda- and g-weighted density of single-halo DM values at x = Q] / f(Q),
i.e. the haloes whose own DM equals the quantile on sightlines that cross nothing else.
On the CIC grid (f_pdf / engine.pdf_summary) the PDF is the node pmf on DM_i = i DX (DX = 0.125 pc cm^-3) with the
piecewise-linear density f = pmf/DX between nodes; the atom is the node-0 mass pmf[0] (= pi_0 up to the CIC share of
DM < DX, J1 gate G9: 1.9e-4 relative for D1), a triangle of half-width DX centred on DM = 0.  f_pdf / kernels/engine.py
evaluate f(Q - x) as np.interp(Q - x, grid, density, left=0), which keeps only the half of that triangle at Q - x >= 0:
half the atom term is lost (J1 sum rules of D1: 0.59-0.92).  The whole-atom density is
    f_w(u) = interp(u, grid, density, left=0) + (pmf[0]/DX) max(0, 1 - |u|/DX) [u < 0].
It equals f_pdf's own expression with a shifted reference, which is how it is computed here (no code of f_pdf or
engine.py is changed or patched):
    f_w(u) = interp(u + DX, grid, density_shift, left=0, right=0),  density_shift[0] = 0, density_shift[i] = density[i-1]:
for u + DX in [0, DX] the interpolation runs from 0 to density[0] (the missing left half of the node-0 triangle); for
u + DX >= DX it is the original interpolation between the same two nodes; the dropped last node (8191.875 pc cm^-3)
lies far above every quantile.  f_pdf uses the reference's quantiles only as Q in Q - x and its density only in that
interpolation, and divides by reference['density_at_quantiles']: the whole-atom reference passes quantiles + DX, the
shifted density and the unchanged f(Q_p).  The fallback haloes' pieces (Jqfb) get the same density.

Stage 'mecke' of run_test_j2.py (maas; after 'j1base', which re-runs J1's prepare / basis / fallback into
<results>/j1base with the bases in the node-local scratch):
    J_RESULTS=<results>/j1base J_SCRATCH=<scratch> J_WORKERS=8 MECKE_OUT=<results> python -B j_mecke_atom.py
Per definition D in D0-D3 (configuration C14, treatment T1), in J1's truth-stage order:
  1. reference PDFs: the per-halo truth and the truth-mean PDF passes of C14|D|T1 re-run with J1's entries
     (j_truth.halo_entries / det_entries); parity: quantiles, mean and variance equal J1's results/truth_summary.json;
  2. one f_pdf pass with four Mecke sets on J1's entries (j_truth.subsample: <= 64 haloes per bin, and the truth mean):
     each with J1's reference ('half', J1's formula) and with the whole-atom reference ('whole'); parity: 'half'
     equals J1's results/truth.npz Jq|* and Jqfb|* (same entries, chunks and summation order, 8 workers).
Outputs (MECKE_OUT): mecke_atom.npz (Jq_<kind>|<label> [5, 14, 75], Jqfb_<kind>|<label> [5, 16, 67], quantiles and
pmf[0] of each reference), mecke_atom_summary.json (sum rules of both formulas, parities, timings), mecke_parity.json.
Smoke (J_MODE=smoke): J1's smoke truth re-run in <results>/j1base is the parity reference.

Mode --checks (stage 'mecke_checks' of run_test_j2.py; MECKE_OUT = its own directory, e.g. results_mecke_checks):
(1) finite-difference test of the radial distribution of the truth-mean maps: per definition, the bin-mean profiles of
every calibrated bin are scaled by (1 +- eps), eps = 0.05, in one radial band at a time (j_sens.BANDS by shell mid-point,
plus the outermost fine shell inside the boundary), each painted det through f_pdf; the central difference
(Q(+) - Q(-))/(2 eps) is the exact dQ_p/d ln F_band up to O(eps^2), to compare with Q_p times the band sum of the map
(calibrated bins); the unscaled pass must reproduce J1's truth-mean quantiles (mecke_fd.json);
(2) the sampling noise of the per-halo subsample maps ('imp': <= 64 haloes per bin, j_truth.subsample): per subsampled
halo h and quantile p the sum over shells of its Mecke Jacobian, S_h,p = sum_{cells, b} lambda w_h f_w(Q_p - X_h) X_h /
f(Q_p) (X_h: the halo's total column, the sum of its per-shell columns), with the whole-atom density f_w of J1's stored
truth PDF (results/truth.npz pdf|C14|D|T1|truth|density, float32 to 2048 pc cm^-3, zero beyond); S_fb,p the same for the
fallback haloes.  noise_pass() repeats f_pdf.run's set-up and the column construction of f_pdf._zshell line for line
and accumulates these scalars per entry (f_pdf's per-shell accumulation, run with one set per halo, costs ~30 times
more).  sum_h S_h,p + S_fb,p reproduces the whole-atom subsample map sum of the main pass (to the float32 precision of
the stored density).  Bootstrap (2000 draws, seed
20261016): within every bin whose subsample is smaller than the bin, resample its haloes with replacement and rescale
their weights to the bin total; sigma_p = SD of (sum_h S_h,p + S_fb,p)/Q_p (with-replacement resampling: conservative,
no finite-population correction).  Outputs: mecke_fd.json, mecke_noise.npz (per-halo sums, weights, bins),
mecke_noise.json, SHA256SUMS.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import j_common as JC
import j_obs as JO
import j_pdf as JP
import j_truth as JT
import numpy as np

FC, FP = JC.FC, JC.FP
OUT = Path(os.environ.get("MECKE_OUT", str(JC.RESULTS.parent))).resolve()
J1_TRUTH_DIR = JC.RESULTS if JC.SMOKE else JC.HERE / "results"
MECKE_DEFS = ("D0", "D1", "D2", "D3")


def whole_atom_reference(ref):
    """f_pdf reference whose Mecke density counts the whole node-0 atom (see the module docstring)."""
    d = np.asarray(ref["density"], float)
    shifted = np.zeros_like(d)
    shifted[1:] = d[:-1]
    return {"quantiles": np.asarray(ref["quantiles"], float) + FP.DX, "density": shifted,
            "density_at_quantiles": np.asarray(ref["density_at_quantiles"], float)}


def run_pairs(items, which, cut, config):
    """f_pdf pass with the Mecke sets (label, J1 reference, per-bin entries) x {half, whole} on shared entries."""
    with JC.support(config), JC.fallback_cut(cut):
        JP.GP.install()
        ent = FP.Entries(JC.N_FINE)
        mecke, keys = [], []
        for lab, ref, per_bin in items:
            h = ent.new_hist(lab)
            plan = {"fallback": True}
            for c in range(FC.N_CAL):
                rows, wts = per_bin[c]
                plan[c] = ent.add(c, np.atleast_2d(np.asarray(rows, float)), h, wts)
            for kind, r in (("half", ref), ("whole", whole_atom_reference(ref))):
                mecke.append((r, plan))
                keys.append((lab, kind))
        ent.freeze()
        JC.mem(f"before fork: mecke pairs [{which}, {config}, cut {cut:g}]")
        res = FP.run(ent, which, mecke=mecke, include_fallback=True, fallback=JP._FALLBACK.get((config, cut)))
    out = {key: (J, Jfb) for key, J, Jfb in zip(keys, res["J"], res["Jfb"])}
    return out, res["timing"]


def rel_max(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.max(np.abs(b)) if b.size else 0.0
    return float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a - b), initial=0.0))


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    pr = JC.load_npz(JC.WORK / "profiles_j.npz")
    ts1 = JC.read_json(J1_TRUTH_DIR / "truth_summary.json")
    tr1 = JC.load_npz(J1_TRUTH_DIR / "truth.npz", keys=lambda k: k.startswith(("Jq|", "Jqfb|")))
    store = {"dx": np.array(FP.DX), "quantile_levels": np.asarray(FC.QUANTILES, float)}
    summ = {"definitions": __doc__, "mode": JC.MODE, "workers": JC.WORKERS, "j1_truth_dir": str(J1_TRUTH_DIR),
            "per_definition": {}, "timing": {}}
    parity = {"reference_pdf": {}, "half_vs_J1": {}}
    for d in MECKE_DEFS:
        v = JC.DEFS[d]
        ek = JO.SEL_KEY[v["sel"]]
        tag = f"C14|{d}|T1"
        mom = JO.moments(pr, v["sel"], v["cut"], "T1", "C14")
        t1 = time.perf_counter()
        specs = [(f"{tag}|truth", JT.halo_entries(pr, ek, "T1", v["cut"], "C14")),
                 (f"{tag}|truthmean", JT.det_entries(mom, "C14"))]
        res, tim_ref = JP.run_specs(specs, v["basis"], v["cut"], "C14", keep_full={lab for lab, _ in specs},
                                    label=f"mecke reference {d}")
        refs = {lab: res[lab]["_full"] for lab, _ in specs}
        rec = {}
        for lab, s in refs.items():
            j1 = ts1["pdf"][lab]
            rec[lab] = {"quantiles_max_rel": rel_max(s["quantiles"], j1["quantiles"]),
                        "quantiles_max_abs": float(np.max(np.abs(np.asarray(s["quantiles"]) - j1["quantiles"]))),
                        "mean_rel": abs(s["mean"] / j1["mean"] - 1), "variance_rel": abs(s["variance"] / j1["variance"] - 1)}
            store[f"Q|{lab}"] = np.asarray(s["quantiles"], float)
            store[f"density_at_Q|{lab}"] = np.asarray(s["density_at_quantiles"], float)
            store[f"pmf0|{lab}"] = np.array(float(s["density"][0]) * FP.DX)
        parity["reference_pdf"][d] = rec
        t2 = time.perf_counter()
        items = [(f"{tag}|truth_subsample", refs[f"{tag}|truth"], JT.subsample(pr, ek, v["cut"])),
                 (f"{tag}|truthmean", refs[f"{tag}|truthmean"], JT.det_entries(mom, "C14"))]
        m, tim = run_pairs(items, v["basis"], v["cut"], "C14")
        sums, hv = {}, {}
        for (lab, kind), (J, Jfb) in m.items():
            store[f"Jq_{kind}|{lab}"] = J
            store[f"Jqfb_{kind}|{lab}"] = Jfb
            Q = np.asarray(refs[lab.replace("_subsample", "")]["quantiles"], float)
            sums[f"{kind}|{lab}"] = ((J.sum(axis=(1, 2)) + Jfb.sum(axis=(1, 2))) / Q - 1).tolist()
            if kind == "half":
                a, b = tr1.get(f"Jq|{lab}"), tr1.get(f"Jqfb|{lab}")
                hv[lab] = {"Jq_max_abs_diff_over_max": None if a is None else rel_max(J, a),
                           "Jqfb_max_abs_diff_over_max": None if b is None else rel_max(Jfb, b),
                           "Jq_bitwise_equal": None if a is None else bool(np.array_equal(J, a)),
                           "Jqfb_bitwise_equal": None if b is None else bool(np.array_equal(Jfb, b))}
        parity["half_vs_J1"][d] = hv
        summ["per_definition"][d] = {"sum_rule_minus_1": sums,
                                     "pmf0": {lab: float(store[f"pmf0|{lab}"]) for lab in refs},
                                     "quantiles": {lab: store[f"Q|{lab}"].tolist() for lab in refs}}
        summ["timing"][d] = {"reference_seconds": t2 - t1, "mecke_seconds": time.perf_counter() - t2,
                             "reference_passes": tim_ref, "mecke_pass": tim}
        JC.log(f"mecke {d}: reference parity {rec}; half vs J1 {hv}; sums {sums}")
        del res, refs, m
        JC.release()
    ref_ok = all(r["quantiles_max_rel"] <= 1e-12 and r["mean_rel"] <= 1e-12 and r["variance_rel"] <= 1e-12
                 for rec in parity["reference_pdf"].values() for r in rec.values())
    half_ok = all(x["Jq_max_abs_diff_over_max"] is not None and x["Jq_max_abs_diff_over_max"] <= 1e-12
                  and x["Jqfb_max_abs_diff_over_max"] <= 1e-12
                  for rec in parity["half_vs_J1"].values() for x in rec.values())
    parity.update(reference_pdf_equal_J1=ref_ok, half_formula_equals_J1=half_ok, passed=bool(ref_ok and half_ok),
                  rule="reference quantiles / mean / variance and the half-atom Jacobians equal J1's to <= 1e-12")
    summ["seconds"] = time.perf_counter() - t0
    np.savez_compressed(OUT / "mecke_atom.npz", **store)
    JC.write_json(OUT / "mecke_atom_summary.json", summ)
    JC.write_json(OUT / "mecke_parity.json", parity)
    JC.log(f"mecke done in {summ['seconds']:.0f} s; parity passed {parity['passed']}")
    return 0 if parity["passed"] else 2


_NJ: dict = {}


def _noise_chunk(shells):
    """f_pdf._zshell's column construction; per entry the scalar sum_{cells, b} lam w f_w(Q - X) X / f(Q)."""
    job, A, E = _NJ, JC.A, JC.E
    sky, grid, ref = job["sky"], job["grid"], job["ref"]
    x = grid.x
    qs, dq = np.asarray(ref["quantiles"], float), np.asarray(ref["density_at_quantiles"], float)
    S = np.zeros((len(job["F"]), len(qs)))
    S_fb = np.zeros(len(qs))
    for j in shells:
        index = np.flatnonzero(sky.counts[j] > 0)
        zz = sky.z[j]
        if not len(index) or zz > 1.:
            continue
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
        for c in np.unique(cidx):
            rows = np.flatnonzero(job["c"] == c)
            if not len(rows):
                continue
            cells = cal[cidx == c]
            prof = job["g"][c] * evolution[cells]
            col = np.zeros((len(cells), job["ns"], len(area)))
            for s_i, (a, b) in enumerate(grid.segments[:job["n_seg"]]):
                col[:, seg_shell[s_i]] += prof[:, a:b + 1] @ A_seg[s_i].T
            col *= (column_scale[cells][:, None] / job["gbar"][c][None, :])[:, :, None]
            lam = area_scale[cells][:, None] * area[None, :]
            for s0 in range(0, len(rows), 32):
                sub = rows[s0:s0 + 32]
                X = np.tensordot(job["F"][sub], col, axes=([1], [1]))                 # [e, m, k]
                T = np.stack([np.interp(q - X, job["dm_grid"], ref["density"], left=0., right=0.) for q in qs])
                S[sub] += np.einsum("pemk,emk,mk->ep", T, X, lam, optimize=True) * job["w"][sub][:, None] / dq[None, :]
        if len(fb):
            prof_all = A.published_profile(1, mass[fb], 0., x) * evolution[fb]
            for start in range(0, len(fb), JC.FK.BATCH):
                sl = slice(start, start + JC.FK.BATCH)
                prof = prof_all[sl]
                col = np.zeros((prof.shape[0], FC.N_F, len(area)))
                for s_i, (a, b) in enumerate(grid.segments[:job["n_seg_4"]]):
                    col[:, seg_shell[s_i]] += prof[:, a:b + 1] @ A_seg[s_i].T
                col *= column_scale[fb][sl][:, None, None]
                X = col.sum(axis=1)
                lam = area_scale[fb][sl][:, None] * area[None, :]
                dens = np.stack([np.interp(q - X, job["dm_grid"], ref["density"], left=0., right=0.) for q in qs])
                S_fb += np.einsum("pmk,mk,mk->p", dens, lam, X, optimize=True) / dq
    return S, S_fb


def noise_pass(per_bin, ref, which, cut):
    """f_pdf.run's set-up (C14 support, fallback cut at ``cut``) with _noise_chunk; per_bin[c] = (rows [n, 75], w [n])."""
    t0 = time.perf_counter()
    F, w, cb = [], [], []
    for c, (rows, wts) in enumerate(per_bin):
        rows = np.atleast_2d(np.asarray(rows, float))
        F.append(rows)
        w.append(np.broadcast_to(np.asarray(wts, float), (len(rows),)))
        cb.append(np.full(len(rows), c))
    F, w, cb = np.concatenate(F), np.concatenate(w), np.concatenate(cb)
    with JC.support("C14"), JC.fallback_cut(cut):
        E = JC.E
        sky = FC.make_sky()
        base, grid, seg_shell, n_seg_4 = JC.FK.basis_grid()
        support = FC.SUPPORT
        profile = E.Profile(1, "bins", support["mass_edges"], np.asarray(support["support"], bool), support["max_logmass"])
        cal_index = np.full(len(support["support"]), -1)
        cal_index[FC.CAL_BINS] = np.arange(FC.N_CAL)
        z = np.load(FC.WORK / f"basis_{which}.npz")
        ns = int(z["n_shell"])
        assert F.shape[1] == ns, (F.shape, ns)
        cs = E.column_setup(grid, FC.X_END, True)
        _NJ.clear()
        _NJ.update(sky=sky, grid=grid, seg_shell=seg_shell, n_seg_4=n_seg_4, n_seg=int(z["n_seg"]), ns=ns,
                   profile=profile, cal_index=cal_index, g=z["g_E"], gbar=z["gbar_E"], A_seg=cs["A_seg"],
                   area=cs["area"], F=F, w=w, c=cb, ref=ref, dm_grid=np.arange(FP.DM_POINTS) * FP.DX)
        sub = sky.counts.copy()
        sub[sky.z > 1.] = 0
        chunks = E._balanced_chunks(type("S", (), {"counts": sub, "z": sky.z}), JC.WORKERS * 2)
        JC.mem(f"before fork: mecke noise [{which}, cut {cut:g}] ({len(F)} entries)")
        S, S_fb = np.zeros((len(F), len(ref["quantiles"]))), np.zeros(len(ref["quantiles"]))
        for k, (s_, sfb_) in enumerate(FC.pool_map(_noise_chunk, chunks, JC.WORKERS)):
            S += s_
            S_fb += sfb_
            JC.log(f"mecke noise pass ({which}, {len(F)} entries): chunk {k + 1}/{len(chunks)}, "
                   f"{time.perf_counter() - t0:.0f} s")
        _NJ.clear()
    return S, S_fb, cb, w, {"wall_seconds": time.perf_counter() - t0, "workers": JC.WORKERS, "entries": int(len(F))}


FD_EPS = 0.05
BANDS = ((0.0, 0.2335), (0.2335, 0.5185), (0.5185, 1.0), (1.0, 2.0), (2.0, 4.0))       # j_sens.BANDS
BAND_LABELS = ("<0.233R", "0.233-0.519R", "0.519-1R", "1-2R", "2-4R")


def fd_check(pr, ts1):
    """Central finite differences of the truth-mean quantiles for (1 +- FD_EPS) scalings of radial bands."""
    X = JC.X200
    mid = np.sqrt(np.maximum(X[:-1], 1e-6) * X[1:])
    band_of = np.searchsorted(np.array([b[1] for b in BANDS]), mid, side="left")
    out = {"eps": FD_EPS, "bands": dict(zip(BAND_LABELS, [list(b) for b in BANDS])), "per_definition": {}}
    for d in MECKE_DEFS:
        t1 = time.perf_counter()
        v = JC.DEFS[d]
        kB = JC.K_CUT[v["cut"]]
        E = np.asarray(JO.moments(pr, v["sel"], v["cut"], "T1", "C14")["E"], float)
        sets = {"outermost_shell": [kB - 1]}
        for q, lab in enumerate(BAND_LABELS):
            sh = [i for i in range(kB) if band_of[i] == q]
            if sh:
                sets[lab] = sh
        specs = [("base", [(JC.pad75(E[c]), 1.0) for c in range(len(E))])]
        for name, sh in sets.items():
            for sgn in (1, -1):
                Ek = E.copy()
                Ek[:, sh] *= 1 + sgn * FD_EPS
                specs.append((f"{name}|{sgn:+d}", [(JC.pad75(Ek[c]), 1.0) for c in range(len(Ek))]))
        res, tim = JP.run_specs(specs, v["basis"], v["cut"], "C14", label=f"mecke fd {d}")
        Qb = np.asarray(res["base"]["quantiles"], float)
        Qj = np.asarray(ts1["pdf"][f"C14|{d}|T1|truthmean"]["quantiles"], float)
        rec = {"base_vs_J1_truthmean_max_abs": float(np.max(np.abs(Qb - Qj))), "quantiles": Qb.tolist(), "sets": {}}
        for name, sh in sets.items():
            qp, qm = (np.asarray(res[f"{name}|{x}"]["quantiles"], float) for x in ("+1", "-1"))
            rec["sets"][name] = {"shells": [int(sh[0]), int(sh[-1]) + 1],
                                 "x_range": [float(X[sh[0]]), float(X[sh[-1] + 1])],
                                 "dQ_dlnF_central": ((qp - qm) / (2 * FD_EPS)).tolist(),
                                 "Q_plus": qp.tolist(), "Q_minus": qm.tolist()}
        out["per_definition"][d] = rec
        out.setdefault("timing", {})[d] = {"seconds": time.perf_counter() - t1, "passes": tim}
        JC.log(f"mecke fd {d}: base vs J1 {rec['base_vs_J1_truthmean_max_abs']}")
        del res
        JC.release()
    return out


def checks_main():
    import hashlib
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    pr = JC.load_npz(JC.WORK / "profiles_j.npz")
    ts1 = JC.read_json(J1_TRUTH_DIR / "truth_summary.json")
    JC.write_json(OUT / "mecke_fd.json", fd_check(pr, ts1))
    tr1 = JC.load_npz(J1_TRUTH_DIR / "truth.npz", keys=lambda k: k.startswith("pdf|C14|") and k.endswith("|truth|density"))
    main_sums = {}
    mp = OUT.parent / "results_mecke" / "mecke_atom_summary.json"
    if not JC.SMOKE and mp.exists():
        main_sums = JC.read_json(mp)["per_definition"]
    grid = np.arange(FP.DM_POINTS) * FP.DX
    rng = np.random.default_rng(20261016)
    n_boot = 2000
    store, summ = {}, {"definitions": __doc__, "mode": JC.MODE, "workers": JC.WORKERS, "n_boot": n_boot,
                       "per_definition": {}, "timing": {}}
    for d in MECKE_DEFS:
        t1 = time.perf_counter()
        v = JC.DEFS[d]
        ek = JO.SEL_KEY[v["sel"]]
        tag = f"C14|{d}|T1"
        dens = np.zeros(FP.DM_POINTS)
        d32 = np.asarray(tr1[f"pdf|{tag}|truth|density"], float)
        dens[:len(d32)] = d32
        Q = np.asarray(ts1["pdf"][f"{tag}|truth"]["quantiles"], float)
        ref = whole_atom_reference({"quantiles": Q, "density": dens, "density_at_quantiles": np.interp(Q, grid, dens)})
        sub = JT.subsample(pr, ek, v["cut"])
        S, S_fb, cb, wb, tim = noise_pass(sub, ref, v["basis"], v["cut"])
        total = (S.sum(axis=0) + S_fb) / Q - 1
        n_pop = {c: len(JT.bin_rows(pr, b)) for c, b in enumerate(JC.J_BINS)}
        draws = np.zeros((n_boot, len(Q)))
        for c in range(FC.N_CAL):
            sel = np.flatnonzero(cb == c)
            if not len(sel):
                continue
            Wc = wb[sel].sum()
            if len(sel) >= n_pop[c]:
                draws += S[sel].sum(axis=0)[None, :]
                continue
            a = S[sel] / wb[sel][:, None]                                  # per unit weight
            k = rng.integers(0, len(sel), size=(n_boot, len(sel)))
            wk = wb[sel][k]                                                 # [n_boot, n]
            draws += (Wc / wk.sum(axis=1))[:, None] * np.einsum("bn,bnp->bp", wk, a[k])
        draws = (draws + S_fb[None, :]) / Q[None, :] - 1
        sigma = draws.std(axis=0, ddof=1)
        ms = main_sums.get(d, {}).get("sum_rule_minus_1", {}).get(f"whole|{tag}|truth_subsample")
        rec = {"quantiles": Q.tolist(), "n_haloes": int(len(cb)),
               "bins_resampled": [JC.J_LABELS[c] for c in range(FC.N_CAL)
                                  if 0 < np.sum(cb == c) < n_pop[c]],
               "sum_minus_1_per_halo_total": total.tolist(), "sigma_boot": sigma.tolist(),
               "main_pass_sum_minus_1": ms,
               "per_halo_total_vs_main_pass_max_abs": None if ms is None else float(np.max(np.abs(total - np.asarray(ms))))}
        summ["per_definition"][d] = rec
        summ["timing"][d] = {"seconds": time.perf_counter() - t1, "pass": tim}
        store[f"S|{d}"], store[f"S_fb|{d}"], store[f"bin|{d}"], store[f"w|{d}"] = S, S_fb, cb, wb
        JC.log(f"mecke noise {d}: {rec}")
        JC.release()
    summ["seconds"] = time.perf_counter() - t0
    np.savez_compressed(OUT / "mecke_noise.npz", **store)
    JC.write_json(OUT / "mecke_noise.json", summ)
    lines = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    JC.log(f"mecke checks done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(checks_main() if "--checks" in sys.argv[1:] else main())
