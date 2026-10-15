#!/usr/bin/env python
"""Group 'estimators' of stacking_choices_20261014.

How a shell value is computed (shell estimators, temperature definitions) and how haloes are combined (stacking
statistics, bootstrap errors), relative to the common reference configuration:
  key sc.make_key(field, grid='fit') (geometric, recent-AGN gas removed, all gas, all phases, P200 / rhoc_fb / T200 /
  K200), statistic 'mean' (inclusion-weighted), rule 'strict' (min_count 20, softening 4, hsml 2, min_edge 0.04),
  scheme M200c_0.2, all haloes, n_boot = 200 with the core's default per-bin seeds (paired replicates).

Writes (only in this directory):
  web_variants.json   website records (variants + records)
  results.json        band ratios with paired-bootstrap errors, observable impacts, diagnostics
  stacks.npz          per-shell arrays used by make_figures.py

Run:  LD_LIBRARY_PATH=<env>/lib <env>/bin/python -B compute_estimators.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

T0 = time.time()
GROUP = "estimators"
SCHEME = "M200c_0.2"
GRID = "fit"
NBOOT = 200
RULE = "strict"
RES_OPTS = dict(min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04)
BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]
BAND_LABELS = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4"]
P_OBS = ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "tSZ_yy_1h_ell100",
         "tSZ_yy_1h_ell1000", "tSZ_yy_1h_ell3000", "y_mean", "tSZxDM_total_ell1000_P"]
E_OBS = ["kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var", "DM_Cl_total_ell1000",
         "DM_q16", "DM_q50", "DM_q84", "tSZxDM_total_ell1000_E"]

EDGES = sc.edges(GRID)
A, B = EDGES[:-1], EDGES[1:]
XC = 0.75 * (B**4 - A**4) / (B**3 - A**3)          # volume-weighted shell radius
S = len(XC)
BINS = sc.mass_bins(SCHEME)
LABELS = [b["label"] for b in BINS]
CAL = [b for b in BINS if b["lo"] >= 13.5 - 1e-9]
CAL_LABELS = [b["label"] for b in CAL]
assert len(CAL) == 10
TAB = sc.halo_table()

# physical constants (SI) for the P = n_e k T_e identity in normalised units
M_P, M_E = 1.67262192369e-27, 9.1093837015e-31
MU_XGP = (M_E + 2 * M_P / (1 + 0.76)) / 0.9
C_PET = 0.59 * M_P / (200.0 * MU_XGP)          # P/P200 = C_PET * (E in rhoc_fb units) * (T_e/T200)


def K(field, est=None):
    est = est or ("weighted" if field in ("T_M", "T_V", "T_X", "T_e", "K", "C_rho", "C_P") else "geometric")
    return sc.make_key(field, est, grid=GRID)


def sig(v, n=4):
    return sc._sig(v, n)


def sig_list(a, n=4):
    return sc._sig_list(np.asarray(a, float), n)


# ------------------------------------------------------------------------------------------------ variant list
REF_FIELDS = ["P", "E", "G", "T_e", "K"]
VARIANTS = []          # dicts: id, dimension, label, description, comps: {record field: (key, statistic)}


def add(vid, dim, label, desc, comps, extra=None):
    VARIANTS.append(dict(id=vid, dimension=dim, label=label, description=desc, comps=comps, extra=extra or {}))


add("ref", "Reference", "Reference: geometric, inclusion-weighted mean",
    "Geometric shell estimator (sum over particles / geometric shell volume), recently AGN-heated gas removed, all gas, "
    "inclusion-weighted (Horvitz-Thompson) mean over all haloes of the 0.2-dex M200c bin; T_e and K are electron-"
    "weighted ratios of linear sums.",
    {f: (K(f), "mean") for f in REF_FIELDS})
# (1) shell estimators
add(f"{GROUP}:Q_V", "Shell estimator", "Q_V (SPH-volume weighted)",
    "Shell value = sum(V_i q_i)/sum(V_i) with SPH volumes V_i = m_i/rho_i instead of sum(q_i V_i)/V_shell; it equals the "
    "geometric value divided by the SPH filling factor sum(V_i)/V_shell (B24 eq. 8).",
    {"P": (K("P", "Q_V"), "mean"), "E": (K("E", "Q_V"), "mean"), "G": (K("G", "Q_V"), "mean")})
add(f"{GROUP}:Q_V_y", "Shell estimator", "Q_V_y (SPH volume, y-based P)",
    "Pressure from the particle Compton y (as the reference) but divided by the summed SPH volumes instead of the "
    "geometric shell volume; identical to the reference divided by the filling factor.",
    {"P": (K("P", "Q_V_y"), "mean")})
add(f"{GROUP}:Q_M", "Shell estimator", "Q_M (mass weighted)",
    "Mass-weighted shell mean sum(m_i q_i)/sum(m_i) (B24 eq. 10); for gas density this is sum(m rho)/sum(m), "
    "i.e. the volume mean times the clumping factor.",
    {"P": (K("P", "Q_M"), "mean"), "G": (K("G", "Q_M"), "mean")})
add(f"{GROUP}:Q_X", "Shell estimator", "Q_X (X-ray weighted)",
    "ROSAT 0.5-2 keV luminosity-weighted particle pressure of the X-ray phase (SFR <= 0, T >= 1e5 K, not AGN-heated), "
    "B24's fiducial estimator.", {"P": (K("P", "Q_X"), "mean")})
add(f"{GROUP}:nT_X", "Shell estimator", "n_e kT_X (T_X weighting)",
    "Geometric (mass-conserving) electron density times the ROSAT-luminosity-weighted temperature T_X of the X-ray "
    "phase; equals the reference times T_X/T_e. It isolates only the temperature-weighting part of the X-ray route: "
    "a real X-ray deprojection also uses an emission-weighted density (proportional to sqrt(<n_e^2>), biased high "
    "by about sqrt(C_rho) in clumpy shells), which this variant does not include.", {"P": (K("P", "nT_X"), "mean")})
# (2) temperature definitions, shown on the temperature (T_e) panel
add(f"{GROUP}:T_M", "Temperature definition", "T_M (mass weighted)",
    "Temperature panel shows the mass-weighted temperature sum(m T)/sum(m) instead of the electron-weighted T_e.",
    {"T_e": (K("T_M"), "mean")})
add(f"{GROUP}:T_V", "Temperature definition", "T_V (SPH-volume weighted)",
    "Temperature panel shows the SPH-volume-weighted temperature sum(V T)/sum(V) instead of T_e.",
    {"T_e": (K("T_V"), "mean")})
add(f"{GROUP}:T_X", "Temperature definition", "T_X (X-ray weighted)",
    "Temperature panel shows the ROSAT-luminosity-weighted temperature of the X-ray phase instead of T_e.",
    {"T_e": (K("T_X"), "mean")})
add(f"{GROUP}:T_PE", "Temperature definition", "<P>/(k<n_e>) from the stacks",
    "Temperature implied by the two stacked reference profiles, <P_e>/(k <n_e>) (the temperature a painter of the "
    "mean P and mean n_e profiles implicitly assumes), instead of the stacked per-halo T_e.",
    {"T_e": ("derived:P/E", "mean")})
# (3) stacking statistics
STAT_DESC = {
    "unweighted_mean": ("Unweighted mean", "Plain mean of the per-halo profiles, ignoring the inclusion weights of the "
                        "stratified sample. It over-represents the oversampled CONT-annulus class (15-16% of haloes "
                        "unweighted vs 10% weighted in 13.5-14.5); the CONT-1R class is not oversampled. Above "
                        "10^14.5 all haloes are taken, so it equals the reference there."),
    "median": ("Median", "Inclusion-weighted median of the per-halo profiles instead of the mean."),
    "geomean": ("Geometric mean (log-mean)", "exp of the inclusion-weighted mean of ln F over haloes with F > 0 "
                "(the mean-of-log statistic); haloes with an empty shell are left out of that shell."),
    "y_weighted_mean": ("Y200-weighted mean (B12)", "Mean weighted by inclusion weight x the halo's own Y(<R200c), "
                        "as in Battaglia+12."),
    "trimmed_mean": ("5% trimmed mean", "Inclusion-weighted mean after removing the lowest and highest 5% of the "
                     "weight in each shell."),
}
for st, (lab, desc) in STAT_DESC.items():
    add(f"{GROUP}:{st}", "Stacking statistic", lab, desc, {f: (K(f), st) for f in REF_FIELDS})

# ------------------------------------------------------------------------------------------------ stacking
print(f"{len(VARIANTS)} variants", flush=True)
STK = {}          # (vid, field, label) -> result dict (with boot)


def stack_one(key, b, statistic, stratified=True, n_boot=NBOOT):
    return sc.stack_bin(key, b, statistic, RULE, n_boot=n_boot, stratified=stratified, resolution_options=RES_OPTS,
                        return_boot=True)


for v in VARIANTS:
    for fld, (key, st) in v["comps"].items():
        if key.startswith("derived"):
            continue
        for b in BINS:
            STK[(v["id"], fld, b["label"])] = stack_one(key, b, st)
    print(f"  {v['id']}: {time.time() - T0:.0f} s", flush=True)

# derived temperature <P>/(c <E>) with paired replicates
for b in BINS:
    rp, re_ = STK[("ref", "P", b["label"])], STK[("ref", "E", b["label"])]
    rt = STK[("ref", "T_e", b["label"])]
    with np.errstate(divide="ignore", invalid="ignore"):
        val = rp["value"] / (C_PET * re_["value"])
        boot = rp["boot"] / (C_PET * re_["boot"])
    err = np.nanstd(boot, axis=0, ddof=1)
    STK[(f"{GROUP}:T_PE", "T_e", b["label"])] = dict(
        value=val, err=err, boot=boot, q16=np.full(S, np.nan), q84=np.full(S, np.nan),
        n_used=np.minimum(rp["n_used"], re_["n_used"]), resolved=rt["resolved"], sd=np.full(S, np.nan),
        sd_ln=np.full(S, np.nan), n_eff=rp["n_eff"])

# ------------------------------------------------------------------------------------------------ web records
def web_record(vid, fld, b, r):
    return {"variant": vid, "field": fld, "scheme": SCHEME, "bin": b["label"], "logm": sig(b["mean_log_mass"]),
            "N": int(b["N"]), "N_eff": sig(b["N_eff"]), "x_unit": "R200c", "edges": sig_list(EDGES),
            "x": sig_list(XC), "value": sig_list(r["value"]), "err": sig_list(r["err"]), "q16": sig_list(r["q16"]),
            "q84": sig_list(r["q84"]), "n_used": [int(n) for n in np.asarray(r["n_used"])],
            "resolved": [int(bool(z)) for z in np.asarray(r["resolved"])]}


web = {"group": GROUP, "variants": [], "records": []}
for v in VARIANTS:
    keys = {f: k for f, (k, st) in v["comps"].items()}
    stats = sorted({st for (k, st) in v["comps"].values()})
    first = next(iter(v["comps"].values()))[0]
    templ = first if len(v["comps"]) == 1 else "{field}|{geometric for P,E,G; weighted for T_e,K}|removed|all|all|fit|{default norm}"
    web["variants"].append({
        "id": v["id"], "dimension": v["dimension"], "label": v["label"], "description": v["description"],
        "fields": list(v["comps"]), "scheme": SCHEME, "grid": GRID,
        "settings": {"key_template": templ, "keys": keys, "statistic": stats[0] if len(stats) == 1 else stats,
                     "rule": RULE, "min_count": RES_OPTS["min_count"], "softening_factor": RES_OPTS["softening_factor"],
                     "hsml_factor": RES_OPTS["hsml_factor"], "scheme": SCHEME,
                     "sample": "all haloes of the bin, inclusion weights (1/inclusion probability)",
                     "n_boot": NBOOT, "bootstrap": "stratified by design strata (mass_bin x class_main), core default seeds"}})
    for fld in v["comps"]:
        for b in BINS:
            web["records"].append(web_record(v["id"], fld, b, STK[(v["id"], fld, b["label"])]))
(HERE / "web_variants.json").write_text(json.dumps(web, separators=(",", ":"), allow_nan=False))
print(f"web_variants.json: {(HERE / 'web_variants.json').stat().st_size / 1e6:.2f} MB, "
      f"{len(web['records'])} records", flush=True)

# ------------------------------------------------------------------------------------------------ band ratios
REF_OF = {"P": "P", "E": "E", "G": "G", "T_e": "T_e", "K": "K"}


def paired_ratio(r, rr):
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (rr["value"] > 0) & np.isfinite(rr["value"])
        val = np.where(ok, r["value"] / np.where(ok, rr["value"], 1), np.nan)
        okb = (rr["boot"] > 0) & np.isfinite(rr["boot"])
        boot = np.where(okb, r["boot"] / np.where(okb, rr["boot"], 1), np.nan)
    return val, boot


def band_summary(val, boot, resolved):
    out = []
    for (lo, hi), lab in zip(BANDS, BAND_LABELS):
        sel = (XC >= lo) & (XC < hi)
        use = sel & resolved
        ent = {"band": lab, "n_shells": int(sel.sum()), "n_resolved_shells": int(use.sum())}
        for tag, m in (("", use), ("_all", sel)):
            if m.sum() == 0 or not np.isfinite(val[m]).any():
                ent.update({f"ratio{tag}": None, f"err{tag}": None, f"q16{tag}": None, f"q84{tag}": None})
                continue
            med = np.nanmedian(val[m])
            with np.errstate(all="ignore"):
                mb = np.nanmedian(boot[:, m], axis=1) if boot is not None else np.array([np.nan])
            ok = np.isfinite(mb)
            ent.update({f"ratio{tag}": sig(med), f"err{tag}": sig(np.std(mb[ok], ddof=1)) if ok.sum() > 2 else None,
                        f"q16{tag}": sig(np.percentile(mb[ok], 16)) if ok.sum() > 2 else None,
                        f"q84{tag}": sig(np.percentile(mb[ok], 84)) if ok.sum() > 2 else None})
        out.append(ent)
    return out


results = {"group": GROUP, "created": time.strftime("%Y-%m-%d %H:%M"),
           "reference": {"key_template": "sc.make_key(field, grid='fit')", "statistic": "mean", "rule": RULE,
                         **RES_OPTS, "scheme": SCHEME, "n_boot": NBOOT, "seeds": "core default (20261012 + 1000 x bin index)",
                         "bootstrap": "stratified by design strata"},
           "bands_R200c": BAND_LABELS, "bins": LABELS, "calibrated_bins": CAL_LABELS,
           "bin_info": {b["label"]: {"N": b["N"], "N_eff": sig(b["N_eff"]), "mean_logm200c": sig(b["mean_log_mass"], 5)}
                        for b in BINS},
           "notes": {"ratio": "median over resolved (strict-rule) shells in the band of variant/reference; err = SD of "
                              "the same median over the 200 paired bootstrap replicates; q16/q84 their percentiles; "
                              "'_all' = the same over all shells of the band (unresolved shells included, exploratory)",
                     "impact": "sc.impact_from_stacks on the 10 calibrated bins x 35 fit shells (all shells, NaN = no "
                               "change); impact_resolved_only: shells failing the strict rule set to no change; err = "
                               "SD over the 200 paired bootstrap replicates; values are fractional changes dO/O. y_mean "
                               "and DM_mean are linear in the stacked profile (k = 1, exact); the C_ell, kSZ 1h and DM "
                               "variance are quadratic (k = 2) and depend on where in radius and mass the change sits, "
                               "so they are first-order only"}}

ratios = {}
SHELL = {}        # arrays for figures: (vid, fld, label) -> ratio, ratio_err
for v in VARIANTS:
    if v["id"] == "ref":
        continue
    ratios[v["id"]] = {}
    for fld in v["comps"]:
        ratios[v["id"]][fld] = {}
        for b in BINS:
            r = STK[(v["id"], fld, b["label"])]
            rr = STK[("ref", REF_OF[fld], b["label"])]
            val, boot = paired_ratio(r, rr)
            with np.errstate(all="ignore"):
                SHELL[(v["id"], fld, b["label"])] = (val, np.nanstd(boot, axis=0, ddof=1),
                                                    np.nanpercentile(boot, 16, axis=0), np.nanpercentile(boot, 84, axis=0))
            ratios[v["id"]][fld][b["label"]] = band_summary(val, boot, np.asarray(rr["resolved"], bool))
results["band_ratios"] = ratios
print(f"band ratios: {time.time() - T0:.0f} s", flush=True)

# ------------------------------------------------------------------------------------------------ observable impact
def cal_stack(vid, fld, what="value"):
    return np.array([STK[(vid, fld, lab)][what] for lab in CAL_LABELS])          # [10, S] or [10, nb, S]


RESOLVED_CAL = np.array([np.asarray(STK[("ref", "P", lab)]["resolved"], bool) for lab in CAL_LABELS])
impacts = {}
for v in VARIANTS:
    if v["id"] == "ref":
        continue
    for fld in ("P", "E"):
        if fld not in v["comps"]:
            continue
        obs = P_OBS if fld == "P" else E_OBS
        Fv, Fr = cal_stack(v["id"], fld), cal_stack("ref", fld)
        main = sc.impact_from_stacks(Fv, Fr, fld)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where((Fv > 0) & (Fr > 0), np.log(Fv / Fr), np.nan)
        d_res = np.where(RESOLVED_CAL, d, np.nan)
        res_only = sc.observable_impact(d_res, fld)
        bv = np.transpose(cal_stack(v["id"], fld, "boot"), (1, 0, 2))           # [nb, 10, S]
        br = np.transpose(cal_stack("ref", fld, "boot"), (1, 0, 2))
        bo = {o: [] for o in obs}
        for j in range(bv.shape[0]):
            ij = sc.impact_from_stacks(bv[j], br[j], fld)
            for o in obs:
                bo[o].append(ij[o]["dO_over_O"])
        ent = {}
        for o in obs:
            arr = np.asarray(bo[o])
            ent[o] = {"dO_over_O": sig(main[o]["dO_over_O"]), "dlnO": sig(main[o]["dlnO"]),
                      "err": sig(np.nanstd(arr, ddof=1)), "q16": sig(np.nanpercentile(arr, 16)),
                      "q84": sig(np.nanpercentile(arr, 84)),
                      "dO_over_O_resolved_only": sig(res_only[o]["dO_over_O"]),
                      "missing_share": sig(main[o]["missing_share"]),
                      "per_bin": dict(zip(CAL_LABELS, sig_list(main[o]["per_bin"])))}
        impacts.setdefault(v["id"], {})[fld] = ent
results["impact"] = impacts
print(f"impacts: {time.time() - T0:.0f} s", flush=True)

# ------------------------------------------------------------------------------------------------ diagnostics
diag = {}
arr = {}
with np.errstate(divide="ignore", invalid="ignore"):
    Pg = np.asarray(sc.per_halo(K("P")))
    Eg = np.asarray(sc.per_halo(K("E")))
    Gg = np.asarray(sc.per_halo(K("G")))
    PQV = np.asarray(sc.per_halo(K("P", "Q_V")))
    PQVy = np.asarray(sc.per_halo(K("P", "Q_V_y")))
    PQM = np.asarray(sc.per_halo(K("P", "Q_M")))
    PnTX = np.asarray(sc.per_halo(K("P", "nT_X")))
    EQV = np.asarray(sc.per_halo(K("E", "Q_V")))
    GQV = np.asarray(sc.per_halo(K("G", "Q_V")))
    GQM = np.asarray(sc.per_halo(K("G", "Q_M")))
    Te = np.asarray(sc.per_halo(K("T_e")))
    TX = np.asarray(sc.per_halo(K("T_X")))
    TM = np.asarray(sc.per_halo(K("T_M")))
    Crho = np.asarray(sc.per_halo(K("C_rho")))
    CP = np.asarray(sc.per_halo(K("C_P")))
    FILL = np.where(EQV > 0, Eg / EQV, np.nan)                  # summed SPH volume / geometric shell volume
    FILL_G = np.where(GQV > 0, Gg / GQV, np.nan)


def pct(a, q=(0.1, 1, 50, 99, 99.9)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return {f"p{p}": sig(np.percentile(a, p), 6) for p in q} | {"n": int(a.size)}


# resolved halo-shells (per halo) of the calibrated bins, to state identities where they matter
cal_rows = np.concatenate([b["rows"] for b in CAL])
res_ph = np.zeros_like(Pg, bool)
for b in BINS:
    res_ph[b["rows"]] = sc.resolution_mask(b["rows"], GRID, "per_halo", **RES_OPTS)
with np.errstate(divide="ignore", invalid="ignore"):
    ident = {
        "fill_from_E_over_fill_from_G": pct((FILL / FILL_G)[res_ph]),
        "P_QVy_over_P_geom_times_fill": pct((PQVy / Pg * FILL)[res_ph]),
        "P_QV_over_P_QVy": pct((PQV / PQVy)[res_ph]),
        "P_geom_over_cE_Te": pct((Pg / (C_PET * Eg * Te))[res_ph]),
        "P_nTX_over_P_geom_times_Te_over_TX": pct((PnTX / Pg * Te / TX)[res_ph]),
        "G_QM_over_G_QV_over_Crho": pct((GQM / GQV / Crho)[res_ph]),
        "C_PET": C_PET,
        "note": "percentiles over per-halo resolved halo-shells of all bins (per-halo rule). Ratios built from the "
                "same particle sums (f from E and G; P_QVy vs P_geom/f; G_QM/G_QV vs C_rho) hold to rounding. The three "
                "ratios that compare the stored ComptonY (y) with N_e k T built from the stored electron density, "
                "temperature, mass and density (P_QV/P_QVy, P_geom/(c E T_e), P_nTX identity) scatter by up to "
                "4e-4 (0.1-99.9 pct); see identity_noise for its N^-1/2 scaling (a per-particle mismatch, not float32 "
                "rounding)"}
diag["identities"] = ident
print("identities", json.dumps(ident), flush=True)

# how the y vs N_e kT mismatch scales with the number of (recent-AGN-removed) particles in the halo-shell
OWN = np.load(HERE / "owner_sums.npz") if (HERE / "owner_sums.npz").exists() else None
if OWN is None:
    import subprocess
    subprocess.run([sys.executable, "-B", str(HERE / "owner_sums.py")], check=True)
    OWN = np.load(HERE / "owner_sums.npz")
OS = OWN["sums"]                                   # [4588, owner class (smooth, sub), phase (hot, cold), Q, 35]
OQ = list(OWN["quantities"])


def osum(name, owner=None, phase=None):
    a = OS[:, :, :, OQ.index(name)]
    a = a.sum(axis=1) if owner is None else a[:, owner]
    return a.sum(axis=1) if phase is None else a[:, phase]


NPART = osum("count")                              # recent-AGN-removed gas particles per halo-shell
noise = {"definition": "|ratio - 1| per resolved halo-shell (per-halo rule), binned by the number N of recent-AGN-"
                       "removed gas particles in the halo-shell; a per-particle random mismatch sigma_1 gives "
                       "median|dev| ~ 0.674 sigma_1 / sqrt(N_eff), so median|dev| sqrt(N) is flat; float32 rounding "
                       "would give ~1e-7 (float32 eps = 1.19e-7)", "float32_eps": float(np.finfo(np.float32).eps)}
with np.errstate(divide="ignore", invalid="ignore"):
    for nm, rat in (("P_QV_over_P_QVy", PQV / PQVy), ("P_geom_over_cE_Te", Pg / (C_PET * Eg * Te))):
        d = np.abs(rat - 1)
        ent = {}
        for lo, hi in ((20, 100), (100, 1000), (1000, 10000), (10000, 1e9)):
            m = res_ph & np.isfinite(d) & (NPART >= lo) & (NPART < hi)
            ent[f"N_{lo}_{hi:g}"] = {"n": int(m.sum()), "median_absdev": sig(np.median(d[m])),
                                     "p99_absdev": sig(np.percentile(d[m], 99)),
                                     "median_absdev_x_sqrtN": sig(np.median(d[m] * np.sqrt(NPART[m]))),
                                     "implied_sigma1_estimate": sig(np.median(d[m] * np.sqrt(NPART[m])) / 0.6745)}
        noise[nm] = ent
diag["identity_noise"] = noise
print("identity noise", json.dumps(noise), flush=True)

# filling factor, clumping, and the clumping-correlation decomposition of Q_M/Q_V
with np.errstate(divide="ignore", invalid="ignore"):
    rV = (PQM / PQV - 1.0) / np.sqrt((Crho - 1.0) * (CP - 1.0))
fill_tab, clump_tab = {}, {}
for b in BINS:
    lab = b["label"]
    rows, w = b["rows"], b["weights"]
    seed = sc.STUDY_SEED + 1000 * int(b["index"])
    resolved = np.asarray(STK[("ref", "P", lab)]["resolved"], bool)
    rf = sc.stack(FILL[rows], w, "mean", n_boot=NBOOT, seed=seed, strata=b["strata"])
    rfm = sc.stack(FILL[rows], w, "median", n_boot=0)
    rc = sc.stack(Crho[rows], w, "mean", n_boot=0)
    rcm = sc.stack(Crho[rows], w, "median", n_boot=0)
    rcp = sc.stack(CP[rows], w, "median", n_boot=0)
    rr = sc.stack(np.where(np.isfinite(rV[rows]), rV[rows], np.nan), w, "median", n_boot=0)
    arr[("fill", lab)] = rf["value"]
    arr[("fill_err", lab)] = rf["err"]
    arr[("fill_med", lab)] = rfm["value"]
    arr[("fill_q16", lab)] = rf["q16"]
    arr[("fill_q84", lab)] = rf["q84"]
    arr[("crho_mean", lab)] = rc["value"]
    arr[("crho_med", lab)] = rcm["value"]
    arr[("cp_med", lab)] = rcp["value"]
    arr[("rV_med", lab)] = rr["value"]
    ent_f, ent_c = [], []
    for (lo, hi), bl in zip(BANDS, BAND_LABELS):
        m = (XC >= lo) & (XC < hi) & resolved
        if not m.any():
            ent_f.append({"band": bl, "fill_mean": None, "fill_median": None, "one_over_fill": None})
            ent_c.append({"band": bl, "C_rho_median": None, "C_P_median": None, "r_rhoP_median": None})
            continue
        fm = float(np.nanmedian(rf["value"][m]))
        ent_f.append({"band": bl, "fill_mean": sig(fm), "fill_err": sig(np.nanmedian(rf["err"][m])),
                      "fill_median": sig(np.nanmedian(rfm["value"][m])), "one_over_fill": sig(1 / fm),
                      "fill_q16": sig(np.nanmedian(rf["q16"][m])), "fill_q84": sig(np.nanmedian(rf["q84"][m]))})
        ent_c.append({"band": bl, "C_rho_median": sig(np.nanmedian(rcm["value"][m])),
                      "C_rho_mean": sig(np.nanmedian(rc["value"][m])),
                      "C_P_median": sig(np.nanmedian(rcp["value"][m])),
                      "r_rhoP_median": sig(np.nanmedian(rr["value"][m]))})
    fill_tab[lab] = ent_f
    clump_tab[lab] = ent_c
# correlation of the per-halo filling deficit with density clumping, per band, calibrated bins pooled
corr_fc = []
for (lo, hi), bl in zip(BANDS, BAND_LABELS):
    m = (XC >= lo) & (XC < hi)
    sel = res_ph[cal_rows][:, m]
    f_ = FILL[cal_rows][:, m][sel]
    c_ = Crho[cal_rows][:, m][sel]
    ok = np.isfinite(f_) & np.isfinite(c_)
    if ok.sum() > 50:
        from scipy.stats import spearmanr
        rho_s = spearmanr(f_[ok], np.log(c_[ok])).correlation
        corr_fc.append({"band": bl, "spearman_fill_vs_lnCrho": sig(rho_s), "n": int(ok.sum())})
    else:
        corr_fc.append({"band": bl, "spearman_fill_vs_lnCrho": None, "n": int(ok.sum())})
# filling factor split by the halo-shell's density clumping, and with the recently AGN-heated gas kept
split_fc = []
with np.errstate(divide="ignore", invalid="ignore"):
    EgK = np.asarray(sc.per_halo(sc.make_key("E", "geometric", "kept", grid=GRID)))
    EQVK = np.asarray(sc.per_halo(sc.make_key("E", "Q_V", "kept", grid=GRID)))
    FILL_K = np.where(EQVK > 0, EgK / EQVK, np.nan)
for (lo, hi), bl in zip(BANDS, BAND_LABELS):
    m = (XC >= lo) & (XC < hi)
    sel = res_ph[cal_rows][:, m]
    f_ = FILL[cal_rows][:, m][sel]
    fk = FILL_K[cal_rows][:, m][sel]
    c_ = Crho[cal_rows][:, m][sel]
    ent = {"band": bl, "n": int(np.isfinite(f_).sum())}
    if ent["n"] > 50:
        for tag, cm in (("Crho_lt_1.5", c_ < 1.5), ("Crho_1.5_100", (c_ >= 1.5) & (c_ < 100)), ("Crho_ge_100", c_ >= 100)):
            ok = cm & np.isfinite(f_)
            ent[f"median_fill_{tag}"] = sig(np.median(f_[ok])) if ok.sum() > 20 else None
            ent[f"n_{tag}"] = int(ok.sum())
        ent["median_fill_all"] = sig(np.nanmedian(f_))
        ent["median_fill_agn_kept"] = sig(np.nanmedian(fk))
    split_fc.append(ent)
# filling factor vs the smoothing length relative to the radius (a kernel-scale density-gradient bias would grow
# with h/r): per band, calibrated bins, per-halo resolved halo-shells
from scipy.stats import spearmanr  # noqa: E402
HS = np.asarray(sc.hsml_x(GRID))
fill_h = []
for (lo, hi), bl in zip(BANDS, BAND_LABELS):
    m = (XC >= lo) & (XC < hi)
    sel = res_ph[cal_rows][:, m]
    f_ = FILL[cal_rows][:, m][sel]
    hr = (HS[cal_rows][:, m] / XC[m][None, :])[sel]
    ok = np.isfinite(f_) & np.isfinite(hr)
    fill_h.append({"band": bl, "n": int(ok.sum()),
                   "median_h_over_r": sig(np.median(hr[ok])) if ok.sum() > 50 else None,
                   "median_fill": sig(np.median(f_[ok])) if ok.sum() > 50 else None,
                   "spearman_fill_vs_h_over_r": sig(spearmanr(f_[ok], hr[ok]).correlation) if ok.sum() > 50 else None})
# how well 1/<f> describes the stacked ratio <P_QV>/<P_geom> (resolved shells of the calibrated bins)
qv_vs_fill = {}
for lab in CAL_LABELS:
    resolved = np.asarray(STK[("ref", "P", lab)]["resolved"], bool)
    rat = STK[(f"{GROUP}:Q_V", "P", lab)]["value"] / STK[("ref", "P", lab)]["value"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = rat * arr[("fill", lab)] - 1
    qv_vs_fill[lab] = {"median_abs_dev": sig(np.nanmedian(np.abs(dev[resolved]))),
                       "max_abs_dev": sig(np.nanmax(np.abs(dev[resolved])))}
diag["filling_factor"] = {"definition": "f = sum of SPH volumes m_i/rho_i of the shell's particles (recent AGN removed) "
                                        "/ geometric shell volume = E_geom / E_QV per halo-shell",
                          "per_bin_band": fill_tab, "spearman_fill_vs_lnCrho_calibrated_resolved": corr_fc,
                          "median_fill_by_clumping_calibrated_resolved": split_fc,
                          "fill_vs_smoothing_length_calibrated_resolved": fill_h,
                          "stacked_QV_ratio_times_mean_fill_minus_1": qv_vs_fill}
diag["clumping"] = {"definition": "C_rho = <rho^2>_V/<rho>_V^2, C_P = <P^2>_V/<P>_V^2 (SPH-volume moments); "
                                  "r_rhoP = (P_QM/P_QV - 1)/sqrt((C_rho - 1)(C_P - 1)) is the SPH-volume correlation "
                                  "coefficient of rho and P (exact: P_QM/P_QV = <rho P>_V/(<rho>_V <P>_V))",
                    "per_bin_band": clump_tab}

# integrals: Y(<R) and N_e(<R) implied by each stacked profile, relative to the geometric stack
integ = {}
for vid, fld in [(f"{GROUP}:Q_V", "P"), (f"{GROUP}:Q_V_y", "P"), (f"{GROUP}:Q_M", "P"), (f"{GROUP}:Q_X", "P"),
                 (f"{GROUP}:nT_X", "P"), (f"{GROUP}:Q_V", "E"), (f"{GROUP}:Q_V", "G"), (f"{GROUP}:Q_M", "G")] + \
                [(f"{GROUP}:{st}", f) for st in STAT_DESC for f in ("P", "E")]:
    integ.setdefault(vid, {})[fld] = {}
    for b in BINS:
        lab = b["label"]
        rv, rr = STK[(vid, fld, lab)], STK[("ref", fld, lab)]
        integ[vid][fld][lab] = {f"<{u}R": sig(sc.integral(rv["value"], GRID, u) / sc.integral(rr["value"], GRID, u))
                                for u in (0.5, 1.0, 2.0, 4.0)}
diag["integral_ratios"] = {"definition": "sum_s F_s V_s over shells inside u R200c of the variant stack / the same for "
                                         "the reference stack (all shells, unresolved included)", "values": integ}
# check: geometric stack integral = weighted mean of the haloes' own Y(<R200c) in P200 R200c^3 units
SIGMA_T, MEC2, MPC = 6.6524587321e-29, 8.1871057769e-14, 3.0856775814913673e22
yown = TAB["Y200_own_mpc2"] * MPC**2 * MEC2 / SIGMA_T / (TAB["p200_pa"] * (TAB["r200c"] * MPC) ** 3)
yint = sc.integral(Pg, GRID, 1.0)
diag["check_Y200_own_vs_shell_integral"] = pct(yint / yown)


def wmean(v, w):
    """Inclusion-weighted mean over haloes, NaN halo-shells skipped (the core's 'mean' without bootstrap)."""
    ok = np.isfinite(v)
    W = np.where(ok, w[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * np.where(ok, v, 0.0)).sum(0) / W.sum(0)


def band_medians(a, resolved):
    out = {}
    for (lo, hi), bl in zip(BANDS, BAND_LABELS):
        m = (XC >= lo) & (XC < hi) & resolved
        out[bl] = sig(np.nanmedian(a[m])) if m.any() and np.isfinite(a[m]).any() else None
    return out


# ---- T_X vs T_e: X-ray phase cut x L_X weighting (and the role of substructure-bound gas)
# T_X/T_e = [T_e(X-ray phase)/T_e(all gas)] x [T_X/T_e(X-ray phase)]; stacks of T/T200, inclusion-weighted means.
KB_KEV = 8.617333262e-8                            # keV per K
J_PER_KEV = 1.602176634e-16
with np.errstate(divide="ignore", invalid="ignore"):
    Tex = np.asarray(sc.per_halo(sc.make_key("T_e", "weighted", phase="xray", grid=GRID)))
    T200 = TAB["T200_K"]
    # owner-split X-ray-phase temperatures, normalised by T200 (validated against the cache: T_X = lx_t/lx_rosat)
    TX_smooth = osum("lx_t", 0, 0) / osum("lx_rosat", 0, 0) / T200[:, None]
    Te_smooth_x = osum("electron_kT", 0, 0) / osum("electrons", 0, 0) / J_PER_KEV / KB_KEV / T200[:, None]
    LX_sub_share = osum("lx_rosat", 1, 0)
    LX_x = osum("lx_rosat", None, 0)
    kTe_keV = Te * T200[:, None] * KB_KEV
txd = {}
for b in BINS:
    lab, rows, w = b["label"], b["rows"], b["weights"]
    resolved = np.asarray(STK[("ref", "P", lab)]["resolved"], bool)
    te, tex, tx = wmean(Te[rows], w), wmean(Tex[rows], w), wmean(TX[rows], w)
    txs, tes = wmean(TX_smooth[rows], w), wmean(Te_smooth_x[rows], w)
    with np.errstate(divide="ignore", invalid="ignore"):
        parts = {"TX_over_Te": tx / te, "phase_cut_Te_xray_over_Te": tex / te, "LX_weighting_TX_over_Te_xray": tx / tex,
                 "smooth_gas_TX_over_Te_xray": txs / tes, "TX_all_over_TX_smooth": tx / txs,
                 "LX_share_substructure_bound": wmean(LX_sub_share[rows], w) / wmean(LX_x[rows], w),
                 "kTe_keV": wmean(kTe_keV[rows], w)}
    for nm, a in parts.items():
        arr[(f"txd|{nm}", lab)] = np.asarray(a, float)
    txd[lab] = {nm: band_medians(a, resolved) for nm, a in parts.items()}
diag["T_X_decomposition"] = {
    "definition": "inclusion-weighted mean stacks of T/T200 (T200 per halo); band = median over resolved shells. "
                  "phase_cut = T_e of the X-ray phase (SFR <= 0, T >= 1e5 K, not AGN) / T_e of all gas; LX_weighting = "
                  "T_X / T_e of the X-ray phase; product = T_X/T_e. smooth_gas_* use only central-bound + unbound gas "
                  "of the X-ray phase; LX_share_substructure_bound = share of the X-ray-phase ROSAT luminosity in gas "
                  "bound to other subhaloes; kTe_keV = stacked electron-weighted temperature in keV",
    "per_bin_band": txd}

# ---- what drives the mass-weighted estimators: cold/star-forming gas and gas bound to other subhaloes
# Shares are stack-level and additive: e.g. the cold share of the stacked G_QM is
# mean_h[sum(m rho)_cold / sum(m)_all] / mean_h[sum(m rho)_all / sum(m)_all] (each halo in G_QM units).
with np.errstate(divide="ignore", invalid="ignore"):
    m_all, mr_all, mp_all, y_all = osum("mass"), osum("mass_rho"), osum("mass_pe"), osum("y")
    nGQM = np.nanmedian(GQM / (mr_all / m_all), axis=1)[:, None]       # per-halo unit conversions (constant per halo)
    PQM_c = PQM
    nPQM = np.nanmedian(PQM_c / (mp_all / m_all), axis=1)[:, None]
    VS = sc.shell_volumes(GRID)
    nGg = np.nanmedian(Gg / (m_all / VS), axis=1)[:, None] / VS
    nPg = np.nanmedian(Pg / (y_all / VS), axis=1)[:, None] / VS
    conv_check = {"G_geom": float(np.nanmax(np.abs(Gg / (m_all * nGg) - 1))),
                  "P_geom": float(np.nanmax(np.abs(Pg / (y_all * nPg) - 1))),
                  "G_QM": float(np.nanmax(np.abs(GQM / (mr_all / m_all * nGQM) - 1))),
                  "P_QM": float(np.nanmax(np.abs(PQM_c / (mp_all / m_all * nPQM) - 1)))}
SUBSETS = {"cold": (None, 1), "substructure_bound": (1, None), "smooth_hot": (0, 0)}
mwd = {}
for b in BINS:
    lab, rows, w = b["label"], b["rows"], b["weights"]
    resolved = np.asarray(STK[("ref", "P", lab)]["resolved"], bool)
    ent = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        tot_m = wmean(m_all[rows] * nGg[rows], w)
        tot_y = wmean(y_all[rows] * nPg[rows], w)
        tot_mr = wmean(mr_all[rows] / m_all[rows] * nGQM[rows], w)
        tot_mp = wmean(mp_all[rows] / m_all[rows] * nPQM[rows], w)
        parts = {"G_QM_over_geom_all": tot_mr / tot_m, "P_QM_over_geom_all": tot_mp / tot_y}
        for sn, (oc, ph) in SUBSETS.items():
            ms, mrs, mps, ys = (osum(q, oc, ph)[rows] for q in ("mass", "mass_rho", "mass_pe", "y"))
            parts[f"mass_share_{sn}"] = wmean(ms * nGg[rows], w) / tot_m
            parts[f"y_share_{sn}"] = wmean(ys * nPg[rows], w) / tot_y
            parts[f"sum_m_rho_share_{sn}"] = wmean(mrs / m_all[rows] * nGQM[rows], w) / tot_mr
            parts[f"sum_m_pe_share_{sn}"] = wmean(mps / m_all[rows] * nPQM[rows], w) / tot_mp
        # Q_M / geometric of the gas left after removing a subset (both stacks over the same remaining gas)
        for tag, (keep_oc, keep_ph) in {"without_cold": (None, 0), "without_substructure": (0, None),
                                        "smooth_hot_only": (0, 0)}.items():
            ms, mrs, mps, ys = (osum(q, keep_oc, keep_ph)[rows] for q in ("mass", "mass_rho", "mass_pe", "y"))
            parts[f"G_QM_over_geom_{tag}"] = wmean(mrs / ms * nGQM[rows], w) / wmean(ms * nGg[rows], w)
            parts[f"P_QM_over_geom_{tag}"] = wmean(mps / ms * nPQM[rows], w) / wmean(ys * nPg[rows], w)
        vs_ = osum("volume", 0, 0)[rows]
        ms_, mrs_ = osum("mass", 0, 0)[rows], osum("mass_rho", 0, 0)[rows]
        parts["C_rho_median_smooth_hot"] = sc.stack(mrs_ * vs_ / ms_**2, w, "median", n_boot=0)["value"]
        parts["C_rho_median_all"] = sc.stack(Crho[rows], w, "median", n_boot=0)["value"]
    for nm, a in parts.items():
        arr[(f"mwd|{nm}", lab)] = np.asarray(a, float)
    mwd[lab] = {nm: band_medians(a, resolved) for nm, a in parts.items()}
diag["mass_weighting_decomposition"] = {
    "definition": "subsets of the recent-AGN-removed gas from STUDY/measurements_owner.npz (owner_sums.py): cold = "
                  "star-forming or T < 1e5 K; substructure_bound = bound to any subhalo other than the central "
                  "(own satellites and neighbours, owners 1-15); smooth_hot = central-bound or unbound gas of the X-ray "
                  "phase. mass_share / y_share = share of the geometric (volume-integrated) gas mass / Compton y; "
                  "sum_m_rho_share / sum_m_pe_share = additive share of the stacked G_QM / P_QM numerators. "
                  "*_QM_over_geom_<tag> = stacked Q_M / stacked geometric profile of the same remaining gas. Stack-level "
                  "inclusion-weighted means; band = median over resolved shells",
    "unit_conversion_check_max_rel_dev": conv_check, "per_bin_band": mwd}
print(f"decompositions: {time.time() - T0:.0f} s", flush=True)

# temperatures: band ratios already in band_ratios; add stacked-level T_e vs <P>/(c<E>) and per-band T ratios
# statistics: lognormal test, skewness, zero fraction, Y-weight correlation
logn = {}
for fld in ("P", "E", "G", "T_e", "K"):
    data = np.asarray(sc.per_halo(K(fld)))
    logn[fld] = {}
    for b in BINS:
        lab = b["label"]
        rows, w = b["rows"], b["weights"]
        v = data[rows]
        ref = STK[("ref", fld, lab)]
        gm = STK[(f"{GROUP}:geomean", fld, lab)]
        md = STK[(f"{GROUP}:median", fld, lab)]
        resolved = np.asarray(ref["resolved"], bool)
        pos = np.isfinite(v) & (v > 0)
        fin = np.isfinite(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            lv = np.where(pos, np.log(np.where(pos, v, 1)), 0.0)
            Wl = np.where(pos, w[:, None], 0.0)
            tl = Wl.sum(0)
            m1 = (Wl * lv).sum(0) / tl
            m2 = (Wl * (lv - m1) ** 2).sum(0) / tl
            m3 = (Wl * (lv - m1) ** 3).sum(0) / tl
            m4 = (Wl * (lv - m1) ** 4).sum(0) / tl
            skew = m3 / m2**1.5
            kurt = m4 / m2**2 - 3
            Wf = np.where(fin, w[:, None], 0.0)
            zero_frac = (Wf * (v <= 0)).sum(0) / Wf.sum(0)
            pred = np.exp(ref["sd_ln"] ** 2 / 2)
            # cumulant expansion of ln<F> - <ln F> to 4th order (raw weighted moments of ln F)
            pred4 = np.exp(m2 / 2 + m3 / 6 + (m4 - 3 * m2**2) / 24)
            mean_over_gm = ref["value"] / gm["value"]
            mean_over_md = ref["value"] / md["value"]
            # Y200 weighting: weighted correlation of Y200 with F (inclusion weights)
            Y = sc.y200(rows)
            Wy = np.where(fin, w[:, None], 0.0)
            vy = np.where(fin, v, 0.0)
            tw = Wy.sum(0)
            mf = (Wy * vy).sum(0) / tw
            my = (Wy * Y[:, None]).sum(0) / tw
            cov = (Wy * (vy - mf) * (Y[:, None] - my)).sum(0) / tw
            sf = np.sqrt((Wy * (vy - mf) ** 2).sum(0) / tw)
            sy = np.sqrt((Wy * (Y[:, None] - my) ** 2).sum(0) / tw)
            corrY = cov / (sf * sy)
            cvY = sy / my
            cvF = sf / mf
            pred_yw = 1 + corrY * cvY * cvF
        for nm, a in (("sd_ln", ref["sd_ln"]), ("skew_ln", skew), ("exkurt_ln", kurt), ("zero_frac", zero_frac),
                      ("pred_lognormal", pred), ("pred_cumulant4", pred4), ("mean_over_geomean", mean_over_gm), ("mean_over_median", mean_over_md),
                      ("corr_Y200_F", corrY), ("cv_Y200", cvY), ("cv_F", cvF), ("pred_yweight_ratio", pred_yw),
                      ("sd_over_mean", ref["sd"] / ref["value"])):
            arr[(f"{nm}|{fld}", lab)] = np.asarray(a, float)
        ent = []
        for (lo, hi), bl in zip(BANDS, BAND_LABELS):
            m = (XC >= lo) & (XC < hi) & resolved
            if not m.any():
                ent.append({"band": bl, "resolved": False})
                continue
            ent.append({"band": bl, "resolved": True, "sd_ln": sig(np.nanmedian(ref["sd_ln"][m])),
                        "skew_ln": sig(np.nanmedian(skew[m])), "exkurt_ln": sig(np.nanmedian(kurt[m])),
                        "zero_frac": sig(np.nanmedian(zero_frac[m])),
                        "exp_sd2_over_2": sig(np.nanmedian(pred[m])),
                        "exp_cumulants_to_4th": sig(np.nanmedian(pred4[m])),
                        "mean_over_geomean": sig(np.nanmedian(mean_over_gm[m])),
                        "mean_over_median": sig(np.nanmedian(mean_over_md[m])),
                        "corr_Y200_F": sig(np.nanmedian(corrY[m])), "cv_Y200": sig(np.nanmedian(cvY[m])),
                        "cv_F": sig(np.nanmedian(cvF[m])), "pred_yweight_ratio": sig(np.nanmedian(pred_yw[m]))})
        logn[fld][lab] = ent
diag["distribution"] = {"definition": "per-halo distribution of F per shell (inclusion weights): sd_ln, skewness and "
                                      "excess kurtosis of ln F (F > 0), weighted fraction of F <= 0, lognormal "
                                      "expectation exp(sd_ln^2/2) of mean/geomean (= mean/median for a lognormal), its "
                                      "4th-order cumulant extension exp(k2/2 + k3/6 + k4/24) of ln F, the "
                                      "measured ratios, and the Y200-weighting decomposition 1 + corr(Y,F) cv(Y) cv(F); "
                                      "band values = median over resolved shells", "per_field": logn}
print(f"distribution: {time.time() - T0:.0f} s", flush=True)

# class composition (unweighted vs weighted) per bin
comp = {}
for b in BINS:
    cls = TAB["class_main"][b["rows"]]
    w = b["weights"]
    comp[b["label"]] = {f"class{c}": {"unweighted_frac": sig(np.mean(cls == c)), "weighted_frac": sig(w[cls == c].sum() / w.sum())}
                        for c in (0, 1, 2)}
diag["class_composition"] = {"classes": "0 CLEAN, 1 CONT-1R (overlapper mu >= 0.1 reaches inside 1R), 2 CONT-annulus "
                                        "(reaches 1-2R only)", "per_bin": comp}

# unweighted mean in wide bins (illustration of the sampling design)
wide = {}
for b in sc.mass_bins("M200c_0.5"):
    ent = {}
    for fld in ("P", "E"):
        rw = sc.stack_bin(K(fld), b, "mean", RULE, n_boot=NBOOT, resolution_options=RES_OPTS, return_boot=True)
        ru = sc.stack_bin(K(fld), b, "unweighted_mean", RULE, n_boot=NBOOT, resolution_options=RES_OPTS, return_boot=True)
        val, boot = paired_ratio(ru, rw)
        ent[fld] = band_summary(val, boot, np.asarray(rw["resolved"], bool))
    wide[b["label"]] = {"N": b["N"], "N_eff": sig(b["N_eff"]), "mean_logm200c": sig(b["mean_log_mass"], 5),
                        "unweighted_logm200c": sig(np.mean(TAB["logm200c"][b["rows"]]), 5), "ratio": ent}
diag["unweighted_vs_weighted_M200c_0.5"] = wide

# ------------------------------------------------------------------------------------------------ bootstrap errors
boot_diag = {}
for fld in ("P", "E", "G", "T_e"):
    data = np.asarray(sc.per_halo(K(fld)))
    boot_diag[fld] = {}
    for b in BINS:
        lab = b["label"]
        ref = STK[("ref", fld, lab)]
        resolved = np.asarray(ref["resolved"], bool)
        analytic = ref["sd"] / np.sqrt(ref["n_eff"])
        unstr = stack_one(K(fld), b, "mean", stratified=False)
        # spatial jackknife over the 4^3 blocks
        rows, w = b["rows"], b["weights"]
        v = data[rows]
        blk = TAB["block"][rows]
        fin = np.isfinite(v)
        th = []
        for k in np.unique(blk):
            keep = blk != k
            Wk = np.where(fin[keep], w[keep][:, None], 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                th.append((Wk * np.where(fin[keep], v[keep], 0)).sum(0) / Wk.sum(0))
        th = np.asarray(th)
        nb = len(th)
        jk = np.sqrt((nb - 1) / nb * np.nansum((th - np.nanmean(th, 0)) ** 2, 0))
        rel = {st: STK[(f"{GROUP}:{st}" if st != "mean" else "ref", fld, lab)] for st in ("mean",) + tuple(STAT_DESC)}
        with np.errstate(invalid="ignore", divide="ignore"):
            arrs = {"boot_over_analytic": ref["err"] / analytic, "unstrat_over_strat": unstr["err"] / ref["err"],
                    "jack_over_boot": jk / ref["err"], "rel_err_mean": ref["err"] / ref["value"]}
            for st, r in rel.items():
                arrs[f"rel_err_{st}"] = r["err"] / r["value"]
                arrs[f"err_{st}_over_err_mean"] = r["err"] / ref["err"]
        for nm, a in arrs.items():
            arr[(f"boot|{nm}|{fld}", lab)] = np.asarray(a, float)
        ent = []
        for (lo, hi), bl in zip(BANDS, BAND_LABELS):
            m = (XC >= lo) & (XC < hi) & resolved
            if not m.any():
                ent.append({"band": bl, "resolved": False})
                continue
            e = {"band": bl, "resolved": True}
            for nm, a in arrs.items():
                e[nm] = sig(np.nanmedian(a[m]))
            relm = float(np.nanmedian(arrs["rel_err_mean"][m]))
            e["N_eff_for_1pct_mean"] = sig(b["N_eff"] * (relm / 0.01) ** 2, 3)
            e["N_eff_for_3pct_mean"] = sig(b["N_eff"] * (relm / 0.03) ** 2, 3)
            ent.append(e)
        boot_diag[fld][lab] = ent
results["bootstrap"] = {"definition": "boot_over_analytic = bootstrap SD of the mean / (sd/sqrt(n_eff)); "
                                      "unstrat_over_strat = unstratified / stratified bootstrap SD; jack_over_boot = 64-"
                                      "block (4^3) spatial jackknife SD / bootstrap SD; rel_err_<stat> = bootstrap SD / "
                                      "value; err_<stat>_over_err_mean; N_eff_for_1pct/3pct_mean = N_eff needed for a 1% / "
                                      "3% error of the mean (scaling the bin's relative error as N_eff^-1/2); band values = "
                                      "median over resolved shells", "per_field": boot_diag}
results["diagnostics"] = diag
print(f"bootstrap: {time.time() - T0:.0f} s", flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return sig_list(o)
    if isinstance(o, (np.floating, float)):
        return sig(o, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


(HERE / "results.json").write_text(json.dumps(clean(results), indent=1, allow_nan=False))

# ------------------------------------------------------------------------------------------------ arrays for figures
npz = {"x": XC, "edges": EDGES, "labels": np.array(LABELS), "cal_labels": np.array(CAL_LABELS),
       "logm": np.array([b["mean_log_mass"] for b in BINS]), "N_eff": np.array([b["N_eff"] for b in BINS])}
for (vid, fld, lab), r in STK.items():
    base = f"{vid}|{fld}|{lab}"
    npz[f"val|{base}"] = np.asarray(r["value"], float)
    npz[f"err|{base}"] = np.asarray(r["err"], float)
    npz[f"q16|{base}"] = np.asarray(r["q16"], float)
    npz[f"q84|{base}"] = np.asarray(r["q84"], float)
    npz[f"res|{base}"] = np.asarray(r["resolved"], bool)
for (vid, fld, lab), (val, err, q16, q84) in SHELL.items():
    base = f"{vid}|{fld}|{lab}"
    npz[f"ratio|{base}"] = val
    npz[f"ratio_err|{base}"] = err
    npz[f"ratio_q16|{base}"] = q16
    npz[f"ratio_q84|{base}"] = q84
for (nm, lab), a in arr.items():
    npz[f"diag|{nm}|{lab}"] = np.asarray(a, float)
np.savez_compressed(HERE / "stacks.npz", **npz)
print(f"done: {time.time() - T0:.0f} s", flush=True)
