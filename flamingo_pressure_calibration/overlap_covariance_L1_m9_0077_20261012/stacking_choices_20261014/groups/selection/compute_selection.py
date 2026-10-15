#!/usr/bin/env python
"""Group 'selection' of stacking_choices_20261014: which gas (and which haloes) enter the 1D stacked profiles.

Dimensions
  1. Recently AGN-heated gas: kept vs removed (reference); the AGN-only phase and its share of the total.
  2. Gas phase: X-ray phase only, cold/star-forming phase only (its share), central-bound X-ray gas.
  3. Electron counting: SWIFT ElectronNumberDensities (reference) vs fully ionised H+He vs gas density / mu_XGP.
  4. Ownership: 16 gas selections (bound-membership classes, overlapper removal, A2-i, A2-ii, excision).
  5. Halo sample: CLEAN / overlapped subsets (row subsets with their inclusion weights).

Everything uses the validated core (core/stackcore.py) with the REFERENCE settings: geometric estimator, statistic
'mean' (inclusion-weighted), rule 'strict' (min_count 20, softening x4, hsml x2, min_edge 0.04), scheme M200c_0.2,
fit grid, n_boot = 200 with the core's default per-bin seeds and design-stratified resampling, so bootstrap
replicates are paired across variants.  Row subsets (dimension 5) reuse the full bin's resample matrix restricted to
the subset rows, so they are paired with the reference as well.

Outputs (this directory): web_variants.json (website records), results.json (ratios, impacts, shares), log.
Run:  LD_LIBRARY_PATH=<env>/lib <env>/bin/python -B compute_selection.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

GROUP = "selection"
SCHEME = "M200c_0.2"
GRID = "fit"
NB = 200
STAT = "mean"
RULE = "strict"
RES = dict(min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04)
BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]
BAND_LABELS = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4"]
ANNULI = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (0.0, 2.0), (0.0, 4.0)]
ANNULUS_LABELS = ["0-1", "1-2", "2-4", "0-2", "0-4"]
LINEAR = ("P", "E", "G")
OBS = {"P": ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "tSZ_yy_1h_ell100",
             "tSZ_yy_1h_ell1000", "tSZ_yy_1h_ell3000", "y_mean", "tSZxDM_total_ell1000_P"],
       "E": ["kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var", "DM_q16", "DM_q50", "DM_q84",
             "DM_Cl_total_ell1000", "tSZxDM_total_ell1000_E"]}

EDGES = sc.edges(GRID)
LO, HI = EDGES[:-1], EDGES[1:]
X = 0.75 * (HI**4 - LO**4) / (HI**3 - LO**3)          # volume-weighted shell radius
S = len(X)
BAND_IDX = [np.flatnonzero((X >= lo) & (X < hi)) for lo, hi in BANDS]
ANN_MASK = [(LO >= lo - 1e-12) & (HI <= hi + 1e-12) for lo, hi in ANNULI]
VOL = sc.shell_volumes(GRID)
BINS = sc.mass_bins(SCHEME)
CAL = [b for b in BINS if b["lo"] >= 13.5 - 1e-9]
assert [b["index"] for b in CAL] == list(sc.CAL_BINS), "calibrated bins must be engine bins 5..14"
TAB = sc.halo_table()


def ref_key(field: str, **tok) -> str:
    est = "weighted" if field in ("T_e", "K", "T_M", "T_V", "T_X") else "geometric"
    return sc.make_key(field, estimator=tok.pop("estimator", est), grid=GRID, **tok)


# ============================================================================ variant definitions
SETTINGS_BASE = {"statistic": STAT, "rule": RULE, "min_count": RES["min_count"],
                 "softening_factor": RES["softening_factor"], "hsml_factor": RES["hsml_factor"],
                 "min_edge": RES["min_edge"], "scheme": SCHEME, "n_boot": NB}
REF_TEMPLATE = "{field}|geometric|removed|all|all|fit|{norm} (T_e, K: estimator 'weighted')"
ALL_SAMPLE = "all haloes of the bin (inclusion weights)"

SUBSETS = {
    "clean": ("class_main == 0 (no overlapper >= 0.1 M200c reaching inside 2 R200c)", TAB["class_main"] == 0),
    "overlapped": ("class_main > 0 (an overlapper >= 0.1 M200c reaches inside 2 R200c)", TAB["class_main"] > 0),
    "overlap_in1R": ("class_main == 1 (overlapper reaching inside 1 R200c)", TAB["class_main"] == 1),
    "overlap_annulus": ("class_main == 2 (overlapper reaching only the 1-2 R200c annulus)", TAB["class_main"] == 2),
    "clean_mu0.03": ("class_mu0p03 == 0 (no overlapper >= 0.03 M200c inside 2 R200c)", TAB["class_mu0p03"] == 0),
    "clean_mu0.3": ("class_mu0p3 == 0 (no overlapper >= 0.3 M200c inside 2 R200c)", TAB["class_mu0p3"] == 0),
}


def V(short, dimension, label, description, fields, template, sample=ALL_SAMPLE, subset=None, impact=True,
      web=True):
    """fields: {field shown: cache key}."""
    vid = "ref" if short == "ref" else f"{GROUP}:{short}"
    return {"id": vid, "short": short, "dimension": dimension, "label": label, "description": description,
            "fields": fields, "template": template, "sample": sample, "subset": subset, "impact": impact, "web": web}


def keys_for(fields, **tok):
    return {f: ref_key(f, **tok) for f in fields}


VARIANTS = [
    V("ref", "Reference", "Reference (all gas, AGN-heated removed)",
      "All gas in the sphere (own satellites and neighbours kept), recently AGN-heated gas removed, geometric "
      "estimator, inclusion-weighted mean, strict resolution mask.", keys_for(["P", "E", "G", "T_e", "K"]),
      REF_TEMPLATE),
    # ---------------------------------------------------------------- 1. recently AGN-heated gas
    V("agn_kept", "Recently AGN-heated gas", "AGN-heated gas kept",
      "Gas heated by AGN in the last ~15 Myr (SOAP filter a_last >= 0.998956, 8.8e6 <= T <= 1.76e8 K) is kept; "
      "the reference removes it.", keys_for(["P", "E", "G", "T_e", "K"], agn="kept"),
      "{field}|geometric|kept|all|all|fit|{norm} (T_e, K: estimator 'weighted')"),
    V("agn_only", "Recently AGN-heated gas", "AGN-heated gas only",
      "Only the recently AGN-heated particles, i.e. the gas the reference removes. Its share of the total profile "
      "is this / (reference + this).", keys_for(["P", "E", "G"], agn="only"),
      "{field}|geometric|only|all|all|fit|{norm}", impact=False),
    # ---------------------------------------------------------------- 2. gas phase
    V("xray", "Gas phase", "X-ray phase only",
      "Only gas with SFR = 0, T >= 1e5 K and not recently AGN-heated (the B24 X-ray selection); cold "
      "(T < 1e5 K) and star-forming gas are removed.", keys_for(["P", "E", "G", "T_e"], phase="xray"),
      "{field}|geometric|removed|all|xray|fit|{norm} (T_e: estimator 'weighted')"),
    V("cold_sf", "Gas phase", "Cold/star-forming gas only",
      "Only star-forming or T < 1e5 K gas (not AGN-heated): the part of the reference that the X-ray selection "
      "drops. Its ratio to the reference is its share of the total.",
      keys_for(["P", "E", "G", "T_e"], phase="cold_sf"),
      "{field}|geometric|removed|all|cold_sf|fit|{norm} (T_e: estimator 'weighted')", impact=False),
    V("cb_xray", "Gas phase", "Central-bound X-ray gas",
      "X-ray-phase gas bound to the central subhalo only (an X-ray-analysis-like selection): cold gas, "
      "satellites' gas, neighbours' gas and unbound gas are all removed.",
      keys_for(["P", "E", "G"], selection="central_bound", phase="xray"),
      "{field}|geometric|removed|central_bound|xray|fit|{norm}"),
    # ---------------------------------------------------------------- 3. electron counting
    V("e_fullion", "Electron counting", "Fully ionised H+He",
      "Electrons from each particle's H and He masses assuming full ionisation, n_e = (m/m_p)(X_H + X_He/2), "
      "instead of SWIFT's ElectronNumberDensities.", {"E": ref_key("E_fullion")},
      "E_fullion|geometric|removed|all|all|fit|rhoc_fb"),
    V("e_gas_muxgp", "Electron counting", "Gas density / mu_XGP",
      "Electrons from the gas density divided by XGPaint's constant mu_XGP = 1.263 m_p (in XGPaint units this is "
      "the gas-density stack itself): what painting a gas-density profile as electrons does.",
      {"E": ref_key("G")}, "G|geometric|removed|all|all|fit|rhoc_fb, shown as E"),
]

OWN = [
    ("central_bound", "Central-bound only",
     "Only gas bound to the central subhalo (HBT+ membership; SOAP ExclusiveSphere-like). Own satellites' gas, "
     "neighbours' gas and unbound diffuse gas are removed."),
    ("host_bound", "Host-bound (central + own satellites)",
     "Gas bound to the central or to its own satellites (FoF-like). Unbound diffuse gas and gas bound to other "
     "haloes are removed."),
    ("no_other", "No gas bound to other haloes",
     "Central, own satellites and unbound gas are kept; gas bound to any other distinct halo (any mass) is removed."),
    ("no_satellites_or_other", "Central-bound + unbound",
     "All bound substructure is removed (own satellites and other haloes); central-bound and unbound gas are kept."),
    ("rm_ov_mu0.03", "Remove overlappers mu >= 0.03",
     "Gas bound to any structure (own satellite or neighbour) of >= 3% of M200c that reaches inside 2 R200c is "
     "removed; everything else is kept."),
    ("rm_ov_mu0.1", "Remove overlappers mu >= 0.1",
     "Gas bound to any structure (own satellite or neighbour) of >= 10% of M200c that reaches inside 2 R200c is "
     "removed; everything else is kept."),
    ("rm_ov_mu0.3", "Remove overlappers mu >= 0.3",
     "Gas bound to any structure (own satellite or neighbour) of >= 30% of M200c that reaches inside 2 R200c is "
     "removed; everything else is kept."),
    ("rm_ov_mu0.1_keepsat", "Remove neighbour overlappers (keep satellites)",
     "Like 'remove overlappers mu >= 0.1', but own satellites are kept: only gas bound to neighbouring "
     "(non-satellite) overlappers is removed."),
    ("rm_ov_mu0.1_reach1R", "Remove overlappers reaching 1R",
     "Gas bound to overlappers of >= 10% of M200c is removed only when they reach inside 1 R200c."),
    ("A2i", "A2-i: no gas bound to painted haloes",
     "Gas bound to other distinct haloes that HalfDome paints (M200c >= 6.6e12 Msun) is removed; their unbound "
     "atmospheres and the gas of smaller haloes stay."),
    ("A2ii", "A2-ii: painting partition",
     "Each particle goes to the painted halo (M200c >= 6.6e12 Msun) in which it has the smallest r/R200c; gas "
     "closer in scaled distance to a painted neighbour is removed. Satellites and unbound gas stay with the host."),
    ("A2ii_A2i", "A2-ii and A2-i",
     "Partition-owned (A2-ii) and, in addition, not bound to another painted halo (A2-i)."),
    ("exc_all_k0.5", "Excision kappa = 0.5",
     "Spheres of 0.5 R200c,j around every overlapper (>= 10% of M200c, reaching inside 2R; satellites and "
     "neighbours) are cut out and each shell is divided by its retained volume; shells keeping < 50% are dropped."),
    ("exc_all_k1", "Excision kappa = 1",
     "Spheres of 1 R200c,j around every overlapper (>= 10% of M200c, reaching inside 2R; satellites and "
     "neighbours) are cut out and each shell is divided by its retained volume; shells keeping < 50% are dropped."),
    ("exc_all_k2", "Excision kappa = 2",
     "Spheres of 2 R200c,j around every overlapper (>= 10% of M200c, reaching inside 2R) are cut out, shells "
     "renormalised by retained volume; shells keeping < 50% are dropped."),
    ("exc_nbr_k1", "Excision of neighbours, kappa = 1",
     "Excision with kappa = 1 around neighbouring overlappers only; own satellites are kept."),
]
for sel, label, desc in OWN:
    VARIANTS.append(V(sel, "Ownership", label, desc, keys_for(["P", "E", "G"], selection=sel),
                      "{field}|geometric|removed|%s|all|fit|{norm}" % sel))

SAMPLE_TXT = {
    "clean": ("CLEAN haloes only", "Only haloes with no overlapping structure of >= 10% of their M200c reaching "
              "inside 2 R200c (class_main = 0), with their inclusion weights; all gas."),
    "overlapped": ("Overlapped haloes only", "Only haloes with an overlapper of >= 10% of M200c reaching inside "
                   "2 R200c (class_main > 0), with their inclusion weights; all gas."),
    "overlap_in1R": ("Overlapper inside 1R", "Only haloes whose overlapper (in practice an own satellite) reaches "
                     "inside 1 R200c (class_main = 1)."),
    "overlap_annulus": ("Overlapper in 1-2R only", "Only haloes whose overlapper reaches the 1-2 R200c annulus but "
                        "not inside 1 R200c (class_main = 2)."),
    "clean_mu0.03": ("CLEAN at mu >= 0.03", "Only haloes with no overlapper of >= 3% of M200c reaching inside "
                     "2 R200c (a stricter clean definition)."),
    "clean_mu0.3": ("CLEAN at mu >= 0.3", "Only haloes with no overlapper of >= 30% of M200c reaching inside "
                    "2 R200c (a looser clean definition)."),
}
for name, (label, desc) in SAMPLE_TXT.items():
    VARIANTS.append(V(name, "Halo sample (overlap class)", label, desc, keys_for(["P", "E", "G", "T_e", "K"]),
                      REF_TEMPLATE, sample=SUBSETS[name][0], subset=name,
                      web=True))


# ============================================================================ stacking helpers
_C = {}


def boot_counts(b):
    """The core's resample multiplicities for this bin (same seed/strata as stack_bin's defaults)."""
    k = b["index"]
    if k not in _C:
        _C[k] = sc._bootstrap_counts(b["N"], NB, sc.STUDY_SEED + 1000 * int(b["index"]), b["strata"])
    return _C[k]


def mean_boot(vals, w, C, sub=None):
    """Inclusion-weighted mean (NaN skipped) of vals [n, S] and its replicates under multiplicities C [nb, n];
    sub [n] bool restricts to a row subset (the replicates stay paired with the full bin's)."""
    s = np.ones(len(w)) if sub is None else sub.astype(float)
    ok = np.isfinite(vals)
    work = np.where(ok, vals, 0.0)
    Wd = np.vstack([(w * s)[None, :], C * (w * s)[None, :]])
    with np.errstate(divide="ignore", invalid="ignore"):
        num = Wd @ work
        den = Wd @ ok.astype(float)
        allv = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return allv[0], allv[1:]


MASK_STRICT = {b["label"]: sc.resolution_mask(b["rows"], GRID, "strict", **RES) for b in BINS}
MASK_DIAG = {b["label"]: sc.resolution_mask(b["rows"], GRID, "diagnostic", **RES) for b in BINS}

STACKS = {}       # (vid, field, bin) -> dict(value, boot, res (for record), resolved, N, N_eff, logm)
VERIFY = {"stack_bin_vs_local_max_rel_value": 0.0, "stack_bin_vs_local_max_rel_boot": 0.0, "checked": 0}


def compute_stack(v, field, key, b):
    rows = b["rows"]
    w = b["weights"]
    vals = np.asarray(sc.per_halo(key)[rows])
    C = boot_counts(b)
    if v["subset"] is None:
        res = sc.stack_bin(key, b, statistic=STAT, rule=RULE, n_boot=NB, resolution_options=RES, return_boot=True)
        value, boot = res["value"], res["boot"]
        # check that the local paired-bootstrap code reproduces the core exactly (used for the subsets)
        lv, lb = mean_boot(vals, w, C)
        with np.errstate(divide="ignore", invalid="ignore"):
            dv = np.nanmax(np.abs(lv - value) / np.maximum(np.abs(value), 1e-300)) if np.any(np.isfinite(value)) else 0
            db = np.nanmax(np.abs(lb - boot) / np.maximum(np.abs(boot), 1e-300)) if np.any(np.isfinite(boot)) else 0
        VERIFY["stack_bin_vs_local_max_rel_value"] = max(VERIFY["stack_bin_vs_local_max_rel_value"], float(dv))
        VERIFY["stack_bin_vs_local_max_rel_boot"] = max(VERIFY["stack_bin_vs_local_max_rel_boot"], float(db))
        VERIFY["checked"] += 1
        resolved = np.asarray(res["resolved"], bool)
        N, N_eff, logm = b["N"], b["N_eff"], b["mean_log_mass"]
        rec_res = res
    else:
        sub = SUBSETS[v["subset"]][1][rows]
        value, boot = mean_boot(vals, w, C, sub)
        n_sub = int(sub.sum())
        if n_sub:
            per = sc.stack(vals[sub], w[sub], STAT, n_boot=0)        # per-halo percentiles, n_used
            with np.errstate(invalid="ignore"):
                nb = np.isfinite(boot).sum(axis=0)
                err = np.where(nb >= 2, np.nanstd(np.where(np.isfinite(boot), boot, np.nan), axis=0, ddof=1), np.nan)
            rec_res = {"value": value, "err": err, "q16": per["q16"], "q84": per["q84"], "n_used": per["n_used"]}
            resolved = sc.resolution_mask(rows[sub], GRID, RULE, **RES)
            ws = w[sub]
            N, N_eff = n_sub, float(ws.sum()**2 / (ws**2).sum())
            logm = float(np.average(TAB["logm200c"][rows[sub]], weights=ws))
        else:
            rec_res, resolved, N, N_eff, logm = None, np.zeros(S, bool), 0, 0.0, float("nan")
    STACKS[(v["id"], field, b["label"])] = {"value": value, "boot": boot, "rec": rec_res, "resolved": resolved,
                                            "N": N, "N_eff": N_eff, "logm": logm}


# ============================================================================ statistics on ratios
def sig(x, n=4):
    return sc._sig(x, n)


def sig_list(a, n=4):
    return sc._sig_list(np.asarray(a, float), n)


def summarize(val, boot):
    with np.errstate(invalid="ignore"):
        bt = boot[np.isfinite(boot)]
    if not np.isfinite(val):
        return None
    # 'value' = the ratio (8 significant digits), 'dev' = value - 1 (4 digits, keeps tiny effects), 'err' = paired
    # bootstrap SD, q16/q84 = replicate percentiles of the ratio (8 digits)
    out = {"value": sig(val, 8), "dev": sig(val - 1.0)}
    if len(bt) >= 2:
        out.update(err=sig(np.std(bt, ddof=1)), q16=sig(np.percentile(bt, 16), 8), q84=sig(np.percentile(bt, 84), 8))
    else:
        out.update(err=None, q16=None, q84=None)
    return out


def band_ratios(rv, rb, mask):
    out = {}
    for lab, idx in zip(BAND_LABELS, BAND_IDX):
        sel = idx[mask[idx]]
        sel = sel[np.isfinite(rv[sel])]
        if not len(sel):
            out[lab] = None
            continue
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            val = float(np.nanmedian(rv[sel]))
            bt = np.nanmedian(rb[:, sel], axis=1)
        d = summarize(val, bt)
        if d is not None:
            d["n_shells"] = int(len(sel))
            d["x_range"] = [sig(LO[sel].min()), sig(HI[sel].max())]
        out[lab] = d
    return out


def integrals(value, boot):
    """Volume integrals of a stacked profile over the annuli (NaN -> 0): sample [A], replicates [nb, A]."""
    v = np.nan_to_num(value, nan=0.0)
    bt = np.nan_to_num(boot, nan=0.0)
    return (np.array([(v * VOL)[m].sum() for m in ANN_MASK]),
            np.stack([(bt * VOL[None, :])[:, m].sum(axis=1) for m in ANN_MASK], axis=1))


def ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((b != 0) & np.isfinite(a) & np.isfinite(b), a / b, np.nan)


# ============================================================================ main
def main():
    t0 = time.time()
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    # ---------------------------------------------------------------- stacks
    for v in VARIANTS:
        for field, key in v["fields"].items():
            sc.variant(key)          # raises if missing
            for b in BINS:
                compute_stack(v, field, key, b)
    say(f"stacks: {len(STACKS)} (variant, field, bin) in {time.time() - t0:.1f} s; core vs local bootstrap "
        f"max rel value {VERIFY['stack_bin_vs_local_max_rel_value']:.2e}, boot "
        f"{VERIFY['stack_bin_vs_local_max_rel_boot']:.2e} over {VERIFY['checked']}")
    assert VERIFY["stack_bin_vs_local_max_rel_value"] < 1e-10 and VERIFY["stack_bin_vs_local_max_rel_boot"] < 1e-10

    results = {
        "group": GROUP, "created": time.strftime("%Y-%m-%d %H:%M"),
        "reference": {"key_template": REF_TEMPLATE, **SETTINGS_BASE, "grid": GRID,
                      "seeds": "STUDY_SEED + 1000 x bin index, design-stratified (mass_bin x class_main)",
                      "subset_pairing": "row subsets reuse the full bin's resample multiplicities restricted to "
                                        "their rows (paired with the reference)"},
        "bands": dict(zip(BAND_LABELS, BANDS)), "annuli": dict(zip(ANNULUS_LABELS, ANNULI)),
        "x_fit": sig_list(X), "edges_fit": sig_list(EDGES),
        "bins": [{"label": b["label"], "N": b["N"], "N_eff": sig(b["N_eff"]), "logm": sig(b["mean_log_mass"], 5),
                  "calibrated": b["lo"] >= 13.5 - 1e-9,
                  "strict_first_edge": sig(LO[MASK_STRICT[b["label"]]].min()) if MASK_STRICT[b["label"]].any()
                  else None, "n_strict": int(MASK_STRICT[b["label"]].sum()),
                  "n_diagnostic": int(MASK_DIAG[b["label"]].sum())} for b in BINS],
        "calibrated_bins": [b["label"] for b in CAL],
        "notes": {
            "band_ratio": "median over the band's shells (assigned by volume-weighted radius x) that pass the "
                          "reference bin's strict mask of F_variant/F_ref per shell; err = SD of the same median "
                          "over 200 paired bootstrap replicates; q16/q84 = replicate percentiles. "
                          "'band_ratio_diagnostic' uses the diagnostic mask instead (exploratory bins).",
            "integral_ratio": "ratio of volume integrals sum_s F_s 4pi/3 dx^3 of the stacked profiles over the "
                              "annulus (all shells, NaN -> 0); for P this is Y_annulus/P200R200^3 averaged over "
                              "haloes, for E the electron number.",
            "impact": "sc.impact_from_stacks on the 10 calibrated bins x 35 shells (all shells, first order; "
                      "dO_over_O in %); err from the paired replicates; 'resolved_only' keeps the reference "
                      "outside the strict mask (what a profile fitted to resolved shells would see).",
        },
        "variants": {},
    }

    for v in VARIANTS:
        entry = {"dimension": v["dimension"], "label": v["label"], "description": v["description"],
                 "fields": list(v["fields"]), "cache_keys": v["fields"], "sample": v["sample"],
                 "band_ratio": {}, "band_ratio_diagnostic": {}, "shell_ratio": {}, "integral_ratio": {},
                 "impact": {}}
        for field in v["fields"]:
            entry["band_ratio"][field] = {}
            entry["band_ratio_diagnostic"][field] = {}
            entry["shell_ratio"][field] = {}
            entry["integral_ratio"][field] = {}
            for b in BINS:
                lab = b["label"]
                st = STACKS[(v["id"], field, lab)]
                rf = STACKS[("ref", field, lab)]
                rv = ratio(st["value"], rf["value"])
                rb = ratio(st["boot"], rf["boot"])
                entry["band_ratio"][field][lab] = band_ratios(rv, rb, MASK_STRICT[lab])
                if b["lo"] < 13.5:
                    entry["band_ratio_diagnostic"][field][lab] = band_ratios(rv, rb, MASK_DIAG[lab])
                with np.errstate(invalid="ignore"):
                    nb = np.isfinite(rb).sum(axis=0)
                    rerr = np.where(nb >= 2, np.nanstd(np.where(np.isfinite(rb), rb, np.nan), axis=0, ddof=1),
                                    np.nan)
                entry["shell_ratio"][field][lab] = {"ratio": sig_list(rv, 7), "err": sig_list(rerr),
                                                    "resolved": [int(x) for x in MASK_STRICT[lab]],
                                                    "N": st["N"]}
                if field in LINEAR:
                    iv, ivb = integrals(st["value"], st["boot"])
                    ir, irb = integrals(rf["value"], rf["boot"])
                    entry["integral_ratio"][field][lab] = {
                        a: summarize(ratio(iv[k], ir[k]), ratio(ivb[:, k], irb[:, k]))
                        for k, a in enumerate(ANNULUS_LABELS)}
            # ------------------------------------------------ observable impact
            if v["impact"] and v["id"] != "ref" and field in ("P", "E"):
                Fv = np.stack([STACKS[(v["id"], field, b["label"])]["value"] for b in CAL])
                Fr = np.stack([STACKS[("ref", field, b["label"])]["value"] for b in CAL])
                Fvb = np.stack([STACKS[(v["id"], field, b["label"])]["boot"] for b in CAL], axis=1)
                Frb = np.stack([STACKS[("ref", field, b["label"])]["boot"] for b in CAL], axis=1)
                Mres = np.stack([MASK_STRICT[b["label"]] for b in CAL])
                imp = sc.impact_from_stacks(Fv, Fr, field)
                imp_res = sc.impact_from_stacks(np.where(Mres, Fv, Fr), Fr, field)
                boots = [sc.impact_from_stacks(Fvb[k], Frb[k], field) for k in range(NB)]
                boots_res = [sc.impact_from_stacks(np.where(Mres, Fvb[k], Frb[k]), Frb[k], field) for k in range(NB)]
                out = {}
                for o in OBS[field]:
                    bt = np.array([bb[o]["dO_over_O"] for bb in boots]) * 100
                    btr = np.array([bb[o]["dO_over_O"] for bb in boots_res]) * 100
                    bt, btr = bt[np.isfinite(bt)], btr[np.isfinite(btr)]
                    out[o] = {"dO_over_O_pct": sig(imp[o]["dO_over_O"] * 100),
                              "err_pct": sig(np.std(bt, ddof=1)) if len(bt) > 1 else None,
                              "q16_pct": sig(np.percentile(bt, 16)) if len(bt) else None,
                              "q84_pct": sig(np.percentile(bt, 84)) if len(bt) else None,
                              "resolved_only_pct": sig(imp_res[o]["dO_over_O"] * 100),
                              "resolved_only_err_pct": sig(np.std(btr, ddof=1)) if len(btr) > 1 else None,
                              "missing_share": sig(imp[o]["missing_share"]),
                              "per_bin_pct": sig_list(np.asarray(imp[o]["per_bin"]) * 100),
                              "description": imp[o]["description"]}
                entry["impact"][field] = out
        results["variants"][v["id"]] = entry
    say(f"ratios and impacts done at {time.time() - t0:.1f} s")

    # ---------------------------------------------------------------- AGN-heated share of the total, cold/SF share
    def share(num_id, den_ids, fields, name):
        out = {}
        for field in fields:
            out[field] = {}
            for b in BINS:
                lab = b["label"]
                n = STACKS[(num_id, field, lab)]
                dv = sum(STACKS[(d, field, lab)]["value"] for d in den_ids)
                db = sum(STACKS[(d, field, lab)]["boot"] for d in den_ids)
                fv, fb = ratio(n["value"], dv), ratio(n["boot"], db)
                with np.errstate(invalid="ignore"):
                    nb = np.isfinite(fb).sum(axis=0)
                    ferr = np.where(nb >= 2, np.nanstd(np.where(np.isfinite(fb), fb, np.nan), axis=0, ddof=1), np.nan)
                iv, ivb = integrals(n["value"], n["boot"])
                idv, idb = integrals(dv, db)
                out[field][lab] = {
                    "shell": sig_list(fv), "shell_err": sig_list(ferr),
                    "resolved": [int(x) for x in MASK_STRICT[lab]],
                    "band_strict": band_ratios(fv, fb, MASK_STRICT[lab]),
                    "band_all_shells": band_ratios(fv, fb, np.ones(S, bool)),
                    "max_all_shells": {"value": sig(np.nanmax(fv)) if np.any(np.isfinite(fv)) else None,
                                       "x": sig(X[np.nanargmax(fv)]) if np.any(np.isfinite(fv)) else None},
                    "integral": {a: summarize(ratio(iv[k], idv[k]), ratio(ivb[:, k], idb[:, k]))
                                 for k, a in enumerate(ANNULUS_LABELS)}}
        results[name] = out

    share("selection:agn_only", ["selection:agn_only", "ref"], ["P", "E", "G"], "agn_share_of_total")
    share("selection:cold_sf", ["ref"], ["P", "E", "G"], "cold_sf_share_of_total")

    # ---------------------------------------------------------------- owner split of the stacked profile
    def comp(field, lab, which):
        g = lambda vid: STACKS[(vid, field, lab)]               # noqa: E731
        A, CB, HB, NO, NS = (g("ref"), g("selection:central_bound"), g("selection:host_bound"),
                             g("selection:no_other"), g("selection:no_satellites_or_other"))
        if which == "central_bound":
            return CB["value"], CB["boot"]
        if which == "own_satellites":
            return HB["value"] - CB["value"], HB["boot"] - CB["boot"]
        if which == "unbound":
            return NO["value"] - HB["value"], NO["boot"] - HB["boot"]
        if which == "bound_to_other_haloes":
            return A["value"] - NO["value"], A["boot"] - NO["boot"]
        raise ValueError(which)

    owner = {}
    closure = 0.0
    for field in ("P", "E", "G"):
        owner[field] = {}
        for b in BINS:
            lab = b["label"]
            A = STACKS[("ref", field, lab)]
            NS = STACKS[("selection:no_satellites_or_other", field, lab)]
            CB = STACKS[("selection:central_bound", field, lab)]
            un_v, _ = comp(field, lab, "unbound")
            closure = max(closure, float(np.nanmax(np.abs((NS["value"] - CB["value"]) - un_v)
                                                   / np.maximum(A["value"].max(), 1e-300))))
            ent = {}
            for which in ("central_bound", "own_satellites", "unbound", "bound_to_other_haloes"):
                cv, cb = comp(field, lab, which)
                fv, fb = ratio(cv, A["value"]), ratio(cb, A["boot"])
                iv, ivb = integrals(cv, cb)
                ia, iab = integrals(A["value"], A["boot"])
                ent[which] = {"shell": sig_list(fv),
                              "band_strict": band_ratios(fv, fb, MASK_STRICT[lab]),
                              "integral": {a: summarize(ratio(iv[k], ia[k]), ratio(ivb[:, k], iab[:, k]))
                                           for k, a in enumerate(ANNULUS_LABELS)}}
            owner[field][lab] = ent
    results["owner_split"] = owner
    results["owner_split_closure_max_rel"] = closure
    say(f"owner split closure (no_sat_or_other - central_bound vs no_other - host_bound): {closure:.2e}")

    # ---------------------------------------------------------------- subset composition
    comp_tab = {}
    for name, (desc, m) in SUBSETS.items():
        comp_tab[name] = {}
        for b in BINS:
            sub = m[b["rows"]]
            w = b["weights"]
            comp_tab[name][b["label"]] = {"N": int(sub.sum()), "weight_share": sig(w[sub].sum() / w.sum())}
    results["subset_composition"] = comp_tab

    # ---------------------------------------------------------------- per-halo electron ratio diagnostics
    E = np.asarray(sc.per_halo(ref_key("E")))
    EF = np.asarray(sc.per_halo(ref_key("E_fullion")))
    results["electron_zero_shells"] = {"E_zero_but_fullion_positive": int(((E == 0) & (EF > 0)).sum()),
                                       "E_positive_but_fullion_zero": int(((E > 0) & (EF == 0)).sum())}

    results["verification"] = {**VERIFY, "owner_split_closure_max_rel": closure,
                               "runtime_s": round(time.time() - t0, 1)}
    (HERE / "results.json").write_text(json.dumps(results, separators=(",", ":"), allow_nan=False))
    say(f"results.json written ({(HERE / 'results.json').stat().st_size / 1e6:.2f} MB)")

    # ---------------------------------------------------------------- website records
    web_variants, records = [], []
    for v in VARIANTS:
        if not v["web"]:
            continue
        web_variants.append({
            "id": v["id"], "dimension": v["dimension"], "label": v["label"], "description": v["description"],
            "fields": list(v["fields"]), "scheme": SCHEME, "grid": GRID,
            "settings": {"key_template": v["template"], **{k: SETTINGS_BASE[k] for k in
                         ("statistic", "rule", "min_count", "softening_factor", "hsml_factor")},
                         "min_edge": RES["min_edge"], "scheme": SCHEME, "sample": v["sample"], "n_boot": NB}})
        for field in v["fields"]:
            for b in BINS:
                st = STACKS[(v["id"], field, b["label"])]
                if st["rec"] is None:
                    continue
                r = sc.record(v["id"], field, SCHEME, b["label"], EDGES, st["rec"], st["N"], st["N_eff"])
                records.append({"variant": v["id"], "field": field, "scheme": SCHEME, "bin": b["label"],
                                "logm": sig(st["logm"]), "N": int(st["N"]), "N_eff": sig(st["N_eff"]),
                                "x_unit": "R200c", "edges": r["x_edges"], "x": sig_list(X),
                                "value": r["value"], "err": r["err"], "q16": r["q16"], "q84": r["q84"],
                                "n_used": r["n_used"], "resolved": [int(x) for x in st["resolved"]]})
    web = {"group": GROUP, "variants": web_variants, "records": records}
    txt = json.dumps(web, separators=(",", ":"), allow_nan=False)
    (HERE / "web_variants.json").write_text(txt)
    say(f"web_variants.json: {len(web_variants)} variants, {len(records)} records, {len(txt) / 1e6:.2f} MB")
    (HERE / "compute_selection.log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
