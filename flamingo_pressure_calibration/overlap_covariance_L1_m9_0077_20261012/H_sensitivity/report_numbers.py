"""Report tables of Test H (local post-processing of the fetched results; no new physics).

Writes tables/H_report_*.csv and tables/H_key_numbers.json from results/{sens,beyond,binning,params,truth}.* .
Adds one derived quantity: the *converged* truncation radius R_X^conv = the smallest fine edge R such that the
calibrated-signal fraction C stays within 1 - X of 1 at every larger edge (|C(R') - 1| <= 1 - X for all R' >= R).
For monotone curves it equals the first crossing R_X; where interference terms make C overshoot 1 (high-ell 1-halo
spectra) it is the radius beyond which truncating changes the signal by less than 1 - X.
Also the *binning-only* error at every truncation radius (H5): for scheme s, mode m and cut R,
  d_O(s, m, R) = e_O(s, m, R) - e_O(M200c_0.2, piecewise, R),   e_O = O_painted / O_ref - 1 (binning_summary.json),
which removes the truncation loss common to every scheme (the reference M200c_0.2 piecewise painting equals the
engine-bin truth mean cut at R); family RMS of d over tSZ (yytot), kSZ (ksz1h), tSZxDM (ydmtot) at ell = 100 / 300 /
1000 / 3000 and DM moments (mean, variance; the DM quantiles are not evaluated at every cut) ->
tables/H_report_binning_only.csv (det vs the truth mean, and S1 vs the scatter-inclusive truth).
Usage: python -B report_numbers.py [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
CUTS = (0.5185, 1.0, 1.5, 2.0, 2.5, 3.0, 3.3019, 3.6342)
OBS = [("yytot", e) for e in (100, 300, 1000, 3000, 8000)] + [("yy1h", e) for e in (100, 1000, 3000)] + \
      [("yy2h", e) for e in (100, 1000)] + [("ymean", None)] + [("ksz1h", e) for e in (100, 300, 1000, 3000, 8000)] + \
      [("ksz2h", e) for e in (100, 1000)] + [("ksztot", e) for e in (100, 1000, 3000)] + \
      [("dm_mean", None), ("dm_var", None)] + [("dmcltot", e) for e in (100, 1000, 3000)] + \
      [("ydmtot", e) for e in (100, 1000, 3000)] + [("ydmtot_Pcut", 1000), ("ydmtot_Ecut", 1000)]


def crossing(x, C, X):
    k = np.flatnonzero(C >= X)
    if not len(k):
        return np.nan
    k = int(k[0])
    if k == 0:
        return float(x[0])
    a, b = C[k - 1], C[k]
    t = (X - a) / (b - a) if b != a else 1.0
    return float(np.exp(np.log(x[k - 1]) + t * (np.log(x[k]) - np.log(x[k - 1]))))


def converged(x, C, X):
    bad = np.flatnonzero(np.abs(C - 1) > 1 - X + 1e-12)
    if not len(bad):
        return float(x[0])
    k = int(bad[-1]) + 1
    return float(x[k]) if k < len(x) else np.nan


def curve_stats(x, c):
    C = (c - c[0]) / (c[-1] - c[0])
    T = c / c[-1]
    out = {}
    for X in LEVELS:
        out[f"R{int(round(100 * X))}"] = crossing(x, C, X)
        out[f"R{int(round(100 * X))}conv"] = converged(x, C, X)
    for r in CUTS:
        k = int(np.argmin(np.abs(x - r)))
        out[f"loss_total_{r:g}R"] = float(1 - T[k])
        out[f"lossC_{r:g}R"] = float(1 - C[k])
    out["C_max"] = float(np.max(C))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = HERE / ("results_smoke" if a.smoke else "results")
    tab = res / "tables"
    S = dict(np.load(res / "sens.npz"))
    ss = json.loads((res / "sens_summary.json").read_text())
    x = S["x_edges"]
    ell = S["ell"]
    key = {}
    rows = []
    for sel in ("a2ii", "main"):
        for name, e in OBS:
            k = f"trunc_all__{sel}__{name}"
            if k not in S:
                continue
            c = S[k] if e is None else S[k][:, int(np.flatnonzero(ell == e)[0])]
            st = curve_stats(x, c)
            rows.append(dict(selection=sel, observable=name, ell=e if e else "", source="moments (per-halo truth)", **st))
            key[f"{sel}|{name}|{e}"] = st
        q = S[f"trunc_all__{sel}__dmq_mean"]
        for p, qn in enumerate(("q16", "q50", "q84", "q975", "q999")):
            st = curve_stats(x, q[:, p])
            rows.append(dict(selection=sel, observable=f"dm_{qn}", ell="", source="DM-PDF passes (truth mean)", **st))
            key[f"{sel}|dm_{qn}|truthmean"] = st
        qh = S[f"trunc_halo__{sel}__dmq"]
        kk = S[f"trunc_halo__{sel}__k"]
        for p, qn in enumerate(("q16", "q50", "q84", "q975", "q999")):
            d = {f"loss_total_{x[k]:.4g}R": float(1 - qh[j, p] / qh[-1, p]) for j, k in enumerate(kk[:-1])}
            rows.append(dict(selection=sel, observable=f"dm_{qn}", ell="", source="DM-PDF passes (per-halo truth)", **d))
            key[f"{sel}|dm_{qn}|perhalo"] = d
    with open(tab / "H_report_radii.csv", "w", newline="") as f:
        cols = []
        for r in rows:
            for c in r:
                if c not in cols:
                    cols.append(c)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{v:.4g}" if isinstance(v, float) else v) for c, v in r.items()})
    # per-bin own radii (1-halo / linear)
    rows_b = []
    for sel in ("a2ii",):
        for name, e in (("yy1h", 100), ("yy1h", 1000), ("yy1h", 3000), ("ksz1h", 100), ("ksz1h", 1000),
                        ("ksz1h", 3000), ("dmcl1h", 1000), ("dm_mean", None), ("dm_var", None), ("ymean", None)):
            own = S[f"trunc_own__{sel}__{name}"]
            for c in range(own.shape[0]):
                cc = own[c] if e is None else own[c, :, int(np.flatnonzero(ell == e)[0])]
                st = curve_stats(x, cc)
                rows_b.append(dict(selection=sel, observable=name, ell=e if e else "", bin=c,
                                   **{k: v for k, v in st.items() if k.startswith("R") or k.startswith("lossC")}))
    with open(tab / "H_report_radii_by_bin.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_b[0]))
        w.writeheader()
        for r in rows_b:
            w.writerow({c: (f"{v:.4g}" if isinstance(v, float) else v) for c, v in r.items()})
    # importance carried by halo-shells that the resolution rule (strict / diagnostic base mask) marks unresolved
    h0 = np.load(HERE / "inputs" / "h0_profiles_x200.npz")
    base = h0["base_mask"]                                   # [15 engine bins, 67]
    cal = [int(b) for b in S["cal_bins"]]
    unres = {}
    for sel in ("a2ii", "main"):
        for k in S:
            if not k.startswith(f"imp__{sel}__"):
                continue
            M = S[k]
            name = k.split("__")[-1]
            ats = [(e, int(np.flatnonzero(ell == e)[0])) for e in (100, 1000, 3000)] if M.ndim == 3 else [(None, None)]
            for e, i in ats:
                m = M[..., i] if i is not None else M
                unres[f"{sel}|{name}|{e}"] = float(sum(m[b][~base[b]].sum() for b in cal))
    key["unresolved_calibrated_share"] = unres
    first = {S["engine_bin_labels"][b]: float(x[int(np.argmax(base[b]))]) for b in cal}
    key["first_resolved_inner_edge_by_bin"] = first
    key["bands"] = {k: v for k, v in ss["bands"].items() if any(k.startswith(f"{s}__") for s in ("a2ii", "main"))}
    key["peaks"] = ss["peaks"]
    key["self_fraction"] = ss["self_fraction"]
    key["calibrated_share"] = {k: v["calibrated"] for k, v in ss["shares"].items()}
    key["tolerance"] = ss["tolerance"]
    key["binning_only"] = binning_only(res, tab)
    key["H6_painting_family_rms_pct"] = painting_family_rms(res, tab)
    (tab / "H_key_numbers.json").write_text(json.dumps(key, indent=1, default=float) + "\n")
    print(len(rows), "radius rows,", len(rows_b), "per-bin rows")


BIN_FAMILIES = {"tSZ": [f"yytot_{e}" for e in (100, 300, 1000, 3000)],
                "kSZ": [f"ksz1h_{e}" for e in (100, 300, 1000, 3000)],
                "DM_moments": ["dm_mean", "dm_var"],
                "tSZxDM": [f"ydmtot_{e}" for e in (100, 300, 1000, 3000)]}


FIT_FAMILIES = {"tSZ": [f"yytot_{e}" for e in (100, 300, 1000, 3000)],
                "kSZ": [f"ksz1h_{e}" for e in (100, 300, 1000, 3000)],
                "DM": ["dm_mean", "dm_var", "dm_q16", "dm_q50", "dm_q84", "dm_q975", "dm_q999"],
                "tSZxDM": [f"ydmtot_{e}" for e in (100, 300, 1000, 3000)]}


def painting_family_rms(res, tab):
    """H6: family RMS (Test F's families, ell = 100-3000, DM incl. quantiles) of the B12/B16 painting errors
    (params_summary.json painting_errors: fitted profiles painted to R_cut = 4 vs the A2-ii truth)."""
    p = res / "params_summary.json"
    if not p.exists():
        return {}
    pe = json.loads(p.read_text())["painting_errors"]
    out, rows = {}, []
    for case, d in pe.items():
        for scat, e in d.items():
            row = {"case": case, "scatter": scat}
            for fam, keys in FIT_FAMILIES.items():
                v = [e[k] for k in keys if k in e]
                row[f"{fam}_rms_pct"] = float(100 * np.sqrt(np.mean(np.square(v))))
            rows.append(row)
            out[f"{case}|{scat}"] = {k: v for k, v in row.items() if k.endswith("_pct")}
    with open(tab / "H_report_fit_painting.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{v:.4g}" if isinstance(v, float) else v) for c, v in r.items()})
    return out


def binning_only(res, tab):
    """Family RMS of the binning-only error d_O = e_O(scheme) - e_O(reference) at each R_cut (see the docstring)."""
    p = res / "binning_summary.json"
    if not p.exists():
        return {}
    E = json.loads(p.read_text())["errors"]
    out, rows = {}, []
    for scat, ref_key in (("det", "vs_truth_mean"), ("S1", "vs_truth")):
        for name in sorted(E):
            sc, mode, rc, sv = name.split("|")
            if sv != scat:
                continue
            ref = E[f"M200c_0.2|piecewise|{rc}|{scat}"][ref_key]
            e = E[name][ref_key]
            row = dict(scheme=sc, mode=mode, R_cut=float(rc), scatter=scat, reference="M200c_0.2 piecewise",
                       compared_against=ref_key)
            for fam, keys in BIN_FAMILIES.items():
                d = [e[k] - ref[k] for k in keys if k in e and k in ref]
                row[f"{fam}_rms_pct"] = float(100 * np.sqrt(np.mean(np.square(d)))) if d else float("nan")
                row[f"{fam}_maxabs_pct"] = float(100 * np.max(np.abs(d))) if d else float("nan")
            rows.append(row)
            out[name] = {k: v for k, v in row.items() if k.endswith("_pct")}
    with open(tab / "H_report_binning_only.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{v:.4g}" if isinstance(v, float) else v) for c, v in r.items()})
    return out


if __name__ == "__main__":
    main()
