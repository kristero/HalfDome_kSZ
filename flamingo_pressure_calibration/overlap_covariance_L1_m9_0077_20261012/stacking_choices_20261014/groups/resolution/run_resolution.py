"""Group 'resolution': resolution criteria for 1D stacked profiles (FLAMINGO L1_m9 z = 0, 4,588 centrals).

Stages (all from the shared cache; nothing outside groups/resolution is written):
  1. masks: per setting x bin: kept shells, innermost kept edge, binding criterion, particles in the innermost kept
     shell, resolved fractions; grid dependence and halo-number dependence of the strict rule.
  2. stacks: reference and per-halo stacks (P, E, G, T_e, K) with 200 paired bootstrap replicates; per-band ratios
     to the reference; per-halo bias vs fraction of haloes kept; inner behaviour of unmasked shells.
  3. fits: B12 (P) / B16 (E) fitted (radii234 objective (a), analytic Jacobian) to each setting's kept shells with
     outer limit 2 and 4 R200c, plus the 200 paired replicates; parameter shifts, Y(<R200c), Y(<4R), N_e(<4R)
     against the reference fit and against the particle-sum stack integrals; observable impacts (first order).
Writes work/*.npz (arrays for figures / web) and results_main.json; assemble_results.py merges results.json.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import res_common as R  # noqa: E402

sc = R.sc
sys.path.insert(0, str(R.STUDY))
from analysis import fitting as FIT  # noqa: E402

WORK = R.GROUP / "work"
WORK.mkdir(exist_ok=True)
EDGES = sc.edges(R.GRID)
S = len(EDGES) - 1
XV = R.x_volume(EDGES)
BAND = R.band_index(EDGES)
VOL = sc.shell_volumes(R.GRID)
BINS = sc.mass_bins(R.SCHEME)
LABELS = [b["label"] for b in BINS]
CAL_IDX = [LABELS.index(l) for l in R.CAL_LABELS]
FAMILY = {"P": "b12", "E": "b16"}
OUTERS = (2.0, 4.0)
STACK_FIELDS = ("P", "E", "G", "T_e", "K")
N_BOOT = R.N_BOOT
MASK_RULES = ("strict", "diagnostic", "frac", "none")       # same stacked values as the reference


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =========================================================================================== 1. masks
def criterion_first_edges(b, opts):
    """First kept edge with each criterion alone (at this setting's threshold), under this setting's rule."""
    det = R.resolution_detail(b["rows"], R.GRID, opts)
    w = b["weights"]
    out = {}
    for name, key in (("count", "c_count"), ("softening", "c_soft"), ("smoothing_length", "c_hsml"),
                      ("min_edge", "c_min_edge")):
        m = det[key]
        if opts["rule"] == "strict":
            kept = m.all(axis=0)
        elif opts["rule"] == "per_halo":
            kept = m.any(axis=0)
        elif opts["rule"] in ("frac", "per_halo_frac"):
            kept = R.weighted_fraction(m, w) >= opts["frac_min"] - 1e-12
        else:
            kept = None
        out[name] = None if kept is None else R.first_edge(kept, EDGES)
    return out, det


def mask_stage():
    rows_out = {}
    resolved_fraction = {}
    for s in R.SETTINGS:
        o = s["opts"]
        res = {}
        for b in BINS:
            kept, hmask, det = R.kept_and_halo_mask(b, R.GRID, o)
            if hmask is not None:
                kept = kept & hmask.any(axis=0)
            fe = R.first_edge(kept, EDGES)
            crit, _ = criterion_first_edges(b, o)
            binding = []
            if fe is not None and o["rule"] not in ("none", "diagnostic"):
                binding = [k for k, v in crit.items() if v is not None and abs(v - fe) < 1e-9]
                if not binding:
                    binding = ["combination"]
            c = det["counts"]
            w = b["weights"]
            idx_k = np.flatnonzero(kept)
            info = {"first_edge": fe, "first_shell_index": int(idx_k[0]) if len(idx_k) else None,
                    "n_kept": int(kept.sum()),
                    "n_kept_to_2R": int(np.sum(kept & (EDGES[1:] <= 2 + 1e-12))),
                    "n_kept_to_1R": int(np.sum(kept & (EDGES[1:] <= 1 + 1e-12))),
                    "contiguous": R.contiguous(kept), "first_edge_by_criterion": crit, "binding": binding}
            if fe is not None:
                s0 = int(np.flatnonzero(kept)[0])
                cs = c[:, s0]
                info["innermost_counts"] = {"min": float(cs.min()), "q16": float(np.percentile(cs, 16)),
                                            "median": float(np.median(cs)), "q84": float(np.percentile(cs, 84)),
                                            "max": float(cs.max())}
                info["innermost_resolved_fraction"] = float(R.weighted_fraction(det["resolved"], w)[s0])
                if s0 > 0:      # failures in the shell just inside the first kept shell
                    s1 = s0 - 1
                    info["inside_shell_fail_fraction"] = {
                        "count": float(np.mean(~det["c_count"][:, s1])),
                        "softening": float(np.mean(~det["c_soft"][:, s1])),
                        "smoothing_length": float(np.mean(~det["c_hsml"][:, s1])),
                        "min_edge": float(np.mean(~det["c_min_edge"][:, s1])),
                        "any": float(np.mean(~det["resolved"][:, s1]))}
            info["kept"] = kept.astype(int)
            res[b["label"]] = info
            if s["short"] in ("ref", "mc0", "mc100", "mc500", "hf1", "hf3", "mconly20"):
                resolved_fraction.setdefault(s["short"], {})[b["label"]] = R.weighted_fraction(det["resolved"], w)
        rows_out[s["id"]] = res
    return rows_out, resolved_fraction


def counts_profiles():
    """Median and 16-84% of gas-particle counts per shell, and of x_inner / h, per bin (fit grid)."""
    c_all = np.asarray(sc.counts(R.GRID))
    h_all = np.asarray(sc.hsml_x(R.GRID))
    out = {}
    for b in BINS:
        c = c_all[b["rows"]]
        h = h_all[b["rows"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            xh = np.where(np.isfinite(h) & (h > 0), EDGES[:-1][None, :] / h, 0.0)
        out[b["label"]] = {
            "count_q16": np.percentile(c, 16, axis=0), "count_q50": np.median(c, axis=0),
            "count_q84": np.percentile(c, 84, axis=0), "count_min": c.min(axis=0),
            "count_zero_fraction": np.mean(c == 0, axis=0),
            "xh_q16": np.percentile(xh, 16, axis=0), "xh_q50": np.median(xh, axis=0),
            "xh_q84": np.percentile(xh, 84, axis=0), "xh_min": xh.min(axis=0),
            "h_q50": np.nanmedian(np.where(np.isfinite(h), h, np.nan), axis=0)}
    return out


def grid_dependence():
    """First strict edge with the reference criteria on every R200c grid (and R500c grids), per bin."""
    out = {}
    for grid in ("fit", "old", "lee22", "x200", "x500", "b24"):
        e = sc.edges(grid)
        res = {}
        for b in BINS:
            m = sc.resolution_mask(b["rows"], grid, "strict")
            idx = np.flatnonzero(m)
            res[b["label"]] = {"first_edge": float(e[idx[0]]) if len(idx) else None,
                               "contiguous": R.contiguous(m), "n_kept": int(m.sum())}
        out[grid] = {"radius": sc.grid_radius(grid), "dex_per_shell_median": float(np.median(np.diff(np.log10(e[e > 0])))),
                     "bins": res}
    return out


def n_dependence(n_draws=200, seed=20261014):
    """Strict rule vs number of haloes: first strict edge for random subsets (bins with >= 300 haloes)."""
    rng = np.random.default_rng(seed)
    out = {}
    for b in BINS:
        det = R.resolution_detail(b["rows"], R.GRID, dict(R.REF))
        res = det["resolved"]
        n = len(b["rows"])
        w = b["weights"]
        o = {"N": n, "full": R.first_edge(res.all(axis=0), EDGES),
             "full_frac95": R.first_edge(R.weighted_fraction(res, w) >= 0.95 - 1e-12, EDGES)}
        for m in (30, 100, 300):
            if m >= n:
                continue
            fes, fes95 = [], []
            for _ in range(n_draws):
                sub = rng.choice(n, m, replace=False)
                fes.append(R.first_edge(res[sub].all(axis=0), EDGES))
                fes95.append(R.first_edge(R.weighted_fraction(res[sub], w[sub]) >= 0.95 - 1e-12, EDGES))
            for tag, v in (("", fes), ("frac95_", fes95)):
                v = np.array([np.nan if u is None else u for u in v])
                o[f"{tag}subset_{m}"] = {"median": float(np.nanmedian(v)), "q16": float(np.nanpercentile(v, 16)),
                                         "q84": float(np.nanpercentile(v, 84)),
                                         "frac_none": float(np.mean(np.isnan(v)))}
        out[b["label"]] = o
    return out


# =========================================================================================== 2. stacks
def compute_stacks():
    """{(short, field, label): result}. Mask-only settings share the reference values (resolved differs)."""
    stacks = {}
    for f in STACK_FIELDS:
        key = R.field_key(f)
        for b in BINS:
            ref = R.stack_setting(key, b, dict(R.REF))
            core = sc.stack_bin(key, b, return_boot=True)
            assert np.array_equal(ref["value"], core["value"], equal_nan=True)
            assert np.array_equal(ref["boot"], core["boot"], equal_nan=True)
            for s in R.SETTINGS:
                o = s["opts"]
                if o["rule"] in MASK_RULES:
                    if f not in ("P", "E") and s["short"] not in ("ref", "none"):
                        continue
                    kept, _, _ = R.kept_and_halo_mask(b, R.GRID, o)
                    r = dict(ref)
                    r["resolved"] = kept
                else:
                    r = R.stack_setting(key, b, o)
                stacks[(s["short"], f, b["label"])] = r
    return stacks


def band_ratio(v, ref, vb, refb, kept):
    """Median over the kept shells of each band of v/ref, with the paired-bootstrap SD."""
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = v / ref
        ratb = vb / refb
    for k, lab in enumerate(R.BAND_LABELS):
        sel = (BAND == k) & kept & np.isfinite(rat) & (ref > 0)
        if not sel.any():
            out[lab] = {"ratio": None, "err": None, "n_shells": 0}
            continue
        med = float(np.median(rat[sel]))
        mb = np.nanmedian(ratb[:, sel], axis=1)
        mb = mb[np.isfinite(mb)]
        out[lab] = {"ratio": med, "err": float(np.std(mb, ddof=1)) if len(mb) > 2 else None,
                    "n_shells": int(sel.sum())}
    return out


def stack_band_ratios(stacks):
    out = {}
    for s in R.SETTINGS:
        for f in STACK_FIELDS:
            for lab in LABELS:
                if (s["short"], f, lab) not in stacks:
                    continue
                r = stacks[(s["short"], f, lab)]
                ref = stacks[("ref", f, lab)]
                out.setdefault(s["id"], {}).setdefault(f, {})[lab] = band_ratio(
                    r["value"], ref["value"], r["boot"], ref["boot"], r["resolved"])
    return out


def per_halo_bias(stacks):
    """Per shell: per_halo / reference (all haloes) ratio, its paired SD, and the weighted fraction of haloes kept."""
    out = {}
    for f in STACK_FIELDS:
        for b in BINS:
            lab = b["label"]
            ph = stacks[("per_halo", f, lab)]
            ref = stacks[("ref", f, lab)]
            _, hmask, _ = R.kept_and_halo_mask(b, R.GRID, dict(R.REF, rule="per_halo"))
            frac = R.weighted_fraction(hmask, b["weights"])
            with np.errstate(divide="ignore", invalid="ignore"):
                rat = ph["value"] / ref["value"]
                ratb = ph["boot"] / ref["boot"]
                err = np.nanstd(ratb, axis=0, ddof=1)
            out.setdefault(f, {})[lab] = {"ratio": rat, "err": err, "fraction_kept": frac,
                                          "strict_kept": ref["resolved"].astype(int)}
    return out


def none_inner(stacks):
    """Inner shells that the reference rejects: values, zero fractions, particle-sum shares of Y and N_e."""
    P = np.asarray(sc.per_halo(R.field_key("P")))
    E = np.asarray(sc.per_halo(R.field_key("E")))
    c_all = np.asarray(sc.counts(R.GRID))
    out = {}
    for b in BINS:
        lab = b["label"]
        rp, re_ = stacks[("ref", "P", lab)], stacks[("ref", "E", lab)]
        kept = rp["resolved"]
        unk = ~kept & (EDGES[1:] <= 1 + 1e-12)
        rows = b["rows"]
        inside1 = EDGES[1:] <= 1 + 1e-12
        inside4 = EDGES[1:] <= 4 + 1e-12
        yP1 = float(np.sum(rp["value"][inside1] * VOL[inside1]))
        yE4 = float(np.sum(re_["value"][inside4] * VOL[inside4]))
        yP4 = float(np.sum(rp["value"][inside4] * VOL[inside4]))
        out[lab] = {
            "unkept_inner_shells": int(unk.sum()),
            "share_Y200_in_unkept": float(np.sum(rp["value"][unk] * VOL[unk]) / yP1) if yP1 > 0 else None,
            "share_Y4R_in_unkept": float(np.sum(rp["value"][unk] * VOL[unk]) / yP4) if yP4 > 0 else None,
            "share_Ne4R_in_unkept": float(np.sum(re_["value"][unk] * VOL[unk]) / yE4) if yE4 > 0 else None,
            "P_zero_fraction": np.mean(P[rows] == 0, axis=0), "E_zero_fraction": np.mean(E[rows] == 0, axis=0),
            "count_median": np.median(c_all[rows], axis=0),
            "P_rel_err": rp["err"] / rp["value"], "E_rel_err": re_["err"] / re_["value"]}
    return out


def inner_decomposition(fits, stacks):
    """What the rejected inner shells contain, against the reference B12 / B16 fit to 2 R200c extrapolated inward:
    stack / extrapolation for P, n_e and their hot (xray) and cold / star-forming phases, the hot share of the n_e
    excess, P with recently AGN-heated gas kept / removed, empty and no-non-AGN-gas halo fractions, and the median
    particle count and x_in / h per shell (all per fit shell, all bins)."""
    P_all = np.asarray(sc.per_halo(R.field_key("P")))
    c_all = np.asarray(sc.counts(R.GRID))
    h_all = np.asarray(sc.hsml_x(R.GRID))
    keys = {"P_xray": sc.make_key("P", phase="xray"), "P_cold_sf": sc.make_key("P", phase="cold_sf"),
            "P_agn_kept": sc.make_key("P", agn="kept"), "E_xray": sc.make_key("E", phase="xray"),
            "E_cold_sf": sc.make_key("E", phase="cold_sf")}
    out, extra_arrays = {}, {}
    for b in BINS:
        lab = b["label"]
        rows = b["rows"]
        st = {k: sc.stack_bin(key, b, n_boot=0)["value"] for k, key in keys.items()}
        for k, v in st.items():
            extra_arrays[f"stack|ref_{k}|{lab}|value"] = v
        P = stacks[("ref", "P", lab)]["value"]
        E = stacks[("ref", "E", lab)]["value"]
        c = c_all[rows]
        h = h_all[rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            xh = np.where(np.isfinite(h) & (h > 0), EDGES[:-1][None, :] / h, 0.0)
            d = {"kept_reference": stacks[("ref", "P", lab)]["resolved"].astype(int),
                 "P_agn_kept_over_removed": st["P_agn_kept"] / P,
                 "E_cold_sf_share": st["E_cold_sf"] / E, "P_cold_sf_share": st["P_cold_sf"] / P,
                 "count_zero_fraction": np.mean(c == 0, axis=0),
                 "no_nonagn_gas_fraction": np.mean(P_all[rows] == 0, axis=0),
                 "count_median": np.median(c, axis=0), "xh_median": np.median(xh, axis=0)}
            for f, F in (("P", P), ("E", E)):
                fr = fits.get(("ref", f, 2.0, lab))
                if fr is None:
                    continue
                mod = np.asarray(fr["model"])
                d[f"{f}_over_ref2R_extrapolation"] = F / mod
                d[f"{f}_xray_over_ref2R_extrapolation"] = st[f"{f}_xray"] / mod
                d[f"{f}_cold_sf_over_ref2R_extrapolation"] = st[f"{f}_cold_sf"] / mod
                if f == "E":
                    exc = F - mod
                    d["E_excess_hot_share"] = np.where(exc > 0, (st["E_xray"] - mod) / exc, np.nan)
                    d["E_excess_cold_share"] = np.where(exc > 0, st["E_cold_sf"] / exc, np.nan)
        out[lab] = d
    return out, extra_arrays


# =========================================================================================== 3. fits
def _fit(family, mean, sd, mask, start=None):
    try:
        return FIT.objective_a(family, mean, sd, mask, start=start, edges=EDGES, jac="analytic")
    except Exception:       # noqa: BLE001
        return None


def fit_one(args):
    """Fit one (setting, field, outer, bin) with its 200 paired replicates."""
    short, field, outer, lab, mean, sd, boot, kept = args
    fam = FAMILY[field]
    mask = kept & (EDGES[1:] <= outer + 1e-12)
    # a shell with a single contributing halo (per-halo masks) has no scatter estimate (sd = NaN): give it a 100%
    # error instead of letting the NaN abort the fit
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.abs(mean))
    fit = _fit(fam, mean, sd, mask)
    if fit is None:
        return (short, field, outer, lab), None
    p0 = fit["parameters"]
    nb = boot.shape[0]
    bp = np.full((nb, 3), np.nan)
    bm = np.full((nb, S), np.nan)
    for k in range(nb):
        fk = _fit(fam, boot[k], sd, mask, start=p0)
        if fk is not None:
            bp[k] = fk["parameters"]
            bm[k] = fk["model"]
    return (short, field, outer, lab), {
        "params": np.asarray(p0), "accepted": bool(fit["accepted"]), "rms_percent": fit["rms_percent"],
        "chi2_per_dof": fit["chi2_per_dof"], "n_shells": fit["n_shells"], "model": np.asarray(fit["model"]),
        "boot_params": bp, "boot_model": bm, "inner_edge": R.first_edge(fit["mask"], EDGES)}


def run_fits(stacks, workers=8):
    jobs = []
    for s in R.SETTINGS:
        for f in ("P", "E"):
            for outer in OUTERS:
                for lab in LABELS:
                    r = stacks[(s["short"], f, lab)]
                    jobs.append((s["short"], f, outer, lab, r["value"], r["sd"], r["boot"], r["resolved"]))
    log(f"fits: {len(jobs)} jobs x {N_BOOT + 1} fits")
    with Pool(workers) as pool:
        res = dict(pool.map(fit_one, jobs, chunksize=4))
    return res


def integrals(model):
    m = np.nan_to_num(np.asarray(model, float), nan=0.0)
    in1 = EDGES[1:] <= 1 + 1e-12
    in4 = EDGES[1:] <= 4 + 1e-12
    return m[..., in1] @ VOL[in1], m[..., in4] @ VOL[in4]


def ratio_stats(a, b, ab, bb):
    """a/b with the paired-bootstrap SD of the replicate ratios."""
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return {"ratio": None, "err": None}
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = ab / bb
    rb = rb[np.isfinite(rb)]
    return {"ratio": float(a / b), "err": float(np.std(rb, ddof=1)) if len(rb) > 2 else None}


def fit_summaries(fits, stacks):
    out = {}
    for s in R.SETTINGS:
        for f in ("P", "E"):
            for outer in OUTERS:
                for lab in LABELS:
                    fv = fits.get((s["short"], f, outer, lab))
                    fr = fits.get(("ref", f, outer, lab))
                    st = stacks[("ref", f, lab)]
                    t1, t4 = integrals(st["value"])
                    t1b, t4b = integrals(st["boot"])
                    d = {"ok": fv is not None}
                    if fv is not None:
                        i1, i4 = integrals(fv["model"])
                        i1b, i4b = integrals(fv["boot_model"])
                        d.update(params=fv["params"], params_err_ln=np.nanstd(np.log(fv["boot_params"]), axis=0, ddof=1),
                                 accepted=fv["accepted"], rms_percent=fv["rms_percent"],
                                 chi2_per_dof=fv["chi2_per_dof"], n_shells=fv["n_shells"], inner_edge=fv["inner_edge"],
                                 int_lt1_vs_truth=ratio_stats(i1, t1, i1b, t1b),
                                 int_lt4_vs_truth=ratio_stats(i4, t4, i4b, t4b))
                        if fr is not None:
                            r1, r4 = integrals(fr["model"])
                            r1b, r4b = integrals(fr["boot_model"])
                            with np.errstate(divide="ignore", invalid="ignore"):
                                dlnp = np.log(fv["params"] / fr["params"])
                                dlnpb = np.log(fv["boot_params"] / fr["boot_params"])
                            d.update(dln_params_vs_ref=dlnp, dln_params_vs_ref_err=np.nanstd(dlnpb, axis=0, ddof=1),
                                     int_lt1_vs_ref=ratio_stats(i1, r1, i1b, r1b),
                                     int_lt4_vs_ref=ratio_stats(i4, r4, i4b, r4b),
                                     band_ratio_vs_ref=band_ratio(fv["model"], fr["model"], fv["boot_model"],
                                                                  fr["boot_model"], np.ones(S, bool)))
                    out.setdefault(s["id"], {}).setdefault(f, {}).setdefault(f"{outer:g}R", {})[lab] = d
    return out


# ------------------------------------------------------------------------------------------- observables
OBS = {"P": ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "tSZ_yy_1h_ell3000", "y_mean"),
       "E": ("kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var")}


def _impact(Fv, Fr, field, Fvb=None, Frb=None):
    imp = sc.impact_from_stacks(Fv, Fr, field)
    out = {}
    for name in OBS[field]:
        o = imp[name]
        out[name] = {"dO_over_O": o["dO_over_O"], "missing_share": o["missing_share"]}
    if Fvb is not None:
        reps = {name: [] for name in OBS[field]}
        for k in range(Fvb.shape[0]):
            ik = sc.impact_from_stacks(Fvb[k], Frb[k], field)
            for name in OBS[field]:
                reps[name].append(ik[name]["dO_over_O"])
        for name in OBS[field]:
            v = np.asarray(reps[name])
            v = v[np.isfinite(v)]
            out[name]["err"] = float(np.std(v, ddof=1)) if len(v) > 2 else None
    return out


def cal_stack(fits_or_stacks, key_fn, attr, boot_attr):
    """[10, 35] (and [n_boot, 10, 35]) arrays over the calibrated bins; NaN where missing."""
    F = np.full((10, S), np.nan)
    Fb = np.full((N_BOOT, 10, S), np.nan)
    for i, lab in enumerate(R.CAL_LABELS):
        r = fits_or_stacks.get(key_fn(lab))
        if r is None:
            continue
        F[i] = r[attr]
        Fb[:, i] = r[boot_attr]
    return F, Fb


def observable_impacts(fits, stacks):
    out = {}
    for s in R.SETTINGS:
        short = s["short"]
        d = {}
        for f in ("P", "E"):
            # stack level: the variant's stacked values on its kept shells vs the reference stack
            Fv, Fvb = cal_stack(stacks, lambda lab: (short, f, lab), "value", "boot")
            Fr, Frb = cal_stack(stacks, lambda lab: ("ref", f, lab), "value", "boot")
            keep = np.array([stacks[(short, f, lab)]["resolved"] for lab in R.CAL_LABELS])
            Fv = np.where(keep, Fv, np.nan)
            Fvb = np.where(keep[None], Fvb, np.nan)
            d.setdefault("stack", {})[f] = _impact(Fv, Fr, f, Fvb, Frb)
            for outer in OUTERS:
                tag = f"fit{outer:g}R"
                Mv, Mvb = cal_stack(fits, lambda lab: (short, f, outer, lab), "model", "boot_model")
                Mr, Mrb = cal_stack(fits, lambda lab: ("ref", f, outer, lab), "model", "boot_model")
                d.setdefault(f"{tag}_vs_ref_fit", {})[f] = _impact(Mv, Mr, f, Mvb, Mrb)
                d.setdefault(f"{tag}_vs_truth", {})[f] = _impact(Mv, Fr, f, Mvb, Frb)
                # calibrated bins without a fit (fewer than MIN_SHELLS kept shells): they enter as 'no change'
                miss_v = [lab for lab in R.CAL_LABELS if fits.get((short, f, outer, lab)) is None]
                miss_r = [lab for lab in R.CAL_LABELS if fits.get(("ref", f, outer, lab)) is None]
                n_av = {lab: int(np.sum(stacks[(short, f, lab)]["resolved"] & (EDGES[1:] <= outer + 1e-12)))
                        for lab in miss_v}
                d.setdefault("missing_fit_bins", {}).setdefault(tag, {})[f] = {
                    "variant": miss_v, "reference": miss_r, "kept_shells_within_outer": n_av,
                    "min_shells_required": int(FIT.MIN_SHELLS)}
        out[s["id"]] = d
    return out


# =========================================================================================== main
def main():
    t0 = time.time()
    log("stage 1: masks")
    masks, resolved_fraction = mask_stage()
    cprof = counts_profiles()
    griddep = grid_dependence()
    ndep = n_dependence()
    log("stage 2: stacks")
    stacks = compute_stacks()
    bands = stack_band_ratios(stacks)
    phb = per_halo_bias(stacks)
    inner = none_inner(stacks)
    log("stage 3: fits")
    fits = run_fits(stacks, workers=int(os.environ.get("RES_WORKERS", "8")))
    fsum = fit_summaries(fits, stacks)
    inner_dec, inner_arrays = inner_decomposition(fits, stacks)
    log("observable impacts")
    impacts = observable_impacts(fits, stacks)
    # ------------------------------------------------------------------ save arrays for figures / web
    arr = {}
    for (short, f, lab), r in stacks.items():
        p = f"stack|{short}|{f}|{lab}|"
        for k in ("value", "err", "q16", "q84", "n_used", "resolved"):
            arr[p + k] = np.asarray(r[k])
    for (short, f, outer, lab), r in fits.items():
        if r is None:
            continue
        p = f"fit|{short}|{f}|{outer:g}|{lab}|"
        arr[p + "model"] = r["model"]
        arr[p + "model_err"] = np.nanstd(r["boot_model"], axis=0, ddof=1)
        arr[p + "params"] = r["params"]
        arr[p + "mask_inner"] = np.array([np.nan if r["inner_edge"] is None else r["inner_edge"]])
    arr.update(inner_arrays)
    np.savez_compressed(WORK / "arrays.npz", **arr)
    R.write_json(WORK / "results_main.json", {
        "masks": masks, "resolved_fraction": resolved_fraction, "counts_profiles": cprof,
        "grid_dependence": griddep, "n_dependence": ndep, "stack_band_ratios": bands, "per_halo_bias": phb,
        "none_inner": inner, "inner_decomposition": inner_dec, "fits": fsum, "impacts": impacts, "wall_seconds": time.time() - t0}, n=5)
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
