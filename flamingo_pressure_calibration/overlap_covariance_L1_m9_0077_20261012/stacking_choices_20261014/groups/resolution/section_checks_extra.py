"""Second set of checks behind the corrected SECTION.md of group 'resolution' (2026-10-14).

Exact 2-decimal values for the impact tables, per-shell per-halo ratios in bands >= 0.25 R200c, the L2p8 binding
count under the strict rule, frac95 / strict N-dependence for other seeds, grid-dependence differences in dex, and a
few single values. Writes section_checks_extra.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent
sys.path.insert(0, str(G))
import res_common as RC  # noqa: E402

sc = RC.sc
R = json.loads((G / "results.json").read_text())
PS = R["per_setting"]
E = sc.edges(RC.GRID)
BINS = sc.mass_bins(RC.SCHEME)
LABS = [b["label"] for b in BINS]
CAL = list(RC.CAL_LABELS)
A = np.load(G / "work" / "arrays.npz")
out = {}

# ---------------------------------------------------------------- exact table values (2 decimals, from raw)
TAB_OBS = ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "kSZ_1h_ell1000", "kSZ_1h_ell3000",
           "DM_mean", "DM_var"]
rows = {}
for s in ("mc50", "mc100", "mc200", "mc500", "hf1", "hf3", "me0", "me0.1", "frac95", "frac84", "diagnostic",
          "per_halo_f50", "per_halo", "none"):
    d = PS[f"resolution:{s}"]["summary"]["impact_fit4R_vs_ref_fit"]
    rows[s] = {o: f"{100 * d[o]['dO_over_O']:+.2f} +- {100 * d[o]['err']:.2f}" for o in TAB_OBS}
    rows[s]["missing"] = {o: f"{100 * d[o]['missing_share']:.2f}" for o in TAB_OBS}
out["table_4R_vs_ref_fit"] = rows
for s, r in rows.items():
    print(s, {k: v for k, v in r.items() if k != "missing"}, r["missing"] if s == "mc500" else "")
tr = {}
for s in ("ref", "resolution:frac95", "resolution:mc100", "resolution:hf3", "resolution:per_halo", "resolution:none"):
    d = PS[s]["summary"]["impact_fit4R_vs_truth"]
    tr[s] = {o: f"{100 * d[o]['dO_over_O']:+.2f} +- {100 * d[o]['err']:.2f}" for o in
             ("tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "kSZ_1h_ell1000", "DM_mean", "DM_var")}
    print("truth", s, tr[s])
out["table_4R_vs_truth"] = tr
for s in ("none", "per_halo"):
    d = PS[f"resolution:{s}"]["summary"]["impact_fit2R_vs_ref_fit"]
    print("2R", s, {o: f"{100 * d[o]['dO_over_O']:+.2f} +- {100 * d[o]['err']:.2f}" for o in TAB_OBS})
d = PS["resolution:per_halo"]["summary"]["impact_stack"]
print("stack per_halo", {o: f"{100 * d[o]['dO_over_O']:+.2f} +- {100 * d[o]['err']:.2f} (miss {100 * d[o]['missing_share']:.2f})"
                         for o in TAB_OBS})
d = PS["resolution:mc500"]["summary"]["impact_fit2R_vs_ref_fit"]
print("mc500 2R missing", {o: round(100 * d[o]["missing_share"], 2) for o in TAB_OBS})

# sorted distinct non-per-halo 4R errors (mc500 == mconly500, keep mc500 only)
lst = []
for sid, dd in PS.items():
    if "per_halo" in sid or sid == "ref" or "mconly" in sid:
        continue
    for o, v in dd["summary"]["impact_fit4R_vs_ref_fit"].items():
        lst.append((round(100 * v["err"], 3), sid.split(":")[-1], o))
lst.sort(reverse=True)
out["errors_4R_sorted"] = lst[:10]
print("largest 4R errors:", lst[:10])
ph = PS["resolution:per_halo"]["summary"]["impact_fit4R_vs_ref_fit"]
print("per_halo 4R errors (table obs):", {o: round(100 * ph[o]["err"], 3) for o in TAB_OBS})

# ---------------------------------------------------------------- per-halo: per-shell ratios in bands >= 0.25
xv = RC.x_volume(E)
rf = {}
for lab in LABS:
    b = next(bb for bb in BINS if bb["label"] == lab)
    det = RC.resolution_detail(b["rows"], RC.GRID, dict(RC.REF))
    frac = RC.weighted_fraction(det["resolved"], b["weights"])
    rf[lab] = frac
dev = {}
for lab in CAL:
    for f in ("P", "E"):
        v = A[f"stack|per_halo|{f}|{lab}|value"] / A[f"stack|ref|{f}|{lab}|value"]
        kept = A[f"stack|per_halo|{f}|{lab}|resolved"].astype(bool)
        sel = kept & (xv >= 0.25) & (np.abs(v - 1) > 1e-6)
        if sel.any():
            dev[f"{lab}|{f}"] = [(round(float(E[i]), 4), round(float(E[i + 1]), 4), round(float(v[i]), 4),
                                  round(float(rf[lab][i]), 3)) for i in np.flatnonzero(sel)]
out["per_halo_shells_ge025_not_one"] = dev
print("\nper-halo shells at x_vol >= 0.25 with ratio != 1 (x_in, x_out, ratio, resolved weight fraction):")
for k, v in dev.items():
    print(" ", k, v)

# ---------------------------------------------------------------- L2p8: strict-rule binding count
emp = R["redshift"]["L2p8_empirical"]
n_tot = n_h_strict = n_h_f95 = 0
exc_strict, exc_f95, floor_cases = [], [], []
for z, dz in emp.items():
    for b, d in dz["bins"].items():
        n_tot += 1
        bc = d["by_criterion"]
        for rule, key, exc in (("strict", "strict_first_edge", exc_strict), ("frac95", "frac95_first_edge", exc_f95)):
            fe = d[key]
            binds = [k for k in bc if bc[k][rule] is not None and fe is not None and abs(bc[k][rule] - fe) < 1e-6]
            if "smoothing_length" in binds:
                if rule == "strict":
                    n_h_strict += 1
                else:
                    n_h_f95 += 1
            else:
                exc.append((z, b, d["N"], fe, {k: bc[k][rule] for k in bc}, binds))
            if rule == "frac95" and "softening" in binds:
                floor_cases.append((z, b, d["N"], fe, binds))
out["L2p8"] = {"n": n_tot, "h_binds_strict": n_h_strict, "h_binds_frac95": n_h_f95, "exceptions_strict": exc_strict,
               "exceptions_frac95": exc_f95, "softening_or_floor_binds_frac95": floor_cases,
               "N_min_max": [min(d["N"] for dz in emp.values() for d in dz["bins"].values()),
                             max(d["N"] for dz in emp.values() for d in dz["bins"].values())]}
print("\nL2p8:", json.dumps(out["L2p8"], indent=0))

# ---------------------------------------------------------------- N dependence with other seeds
def n_dep(seed, n_draws=200):
    rng = np.random.default_rng(seed)
    ns = nf = nt = 0
    cases = []
    for b in BINS:
        if b["label"] not in CAL[:-2]:
            # keep the same rng stream as run_resolution (it loops over all bins, drawing only for m < n)
            pass
        det = RC.resolution_detail(b["rows"], RC.GRID, dict(RC.REF))
        res = det["resolved"]
        n = len(b["rows"])
        w = b["weights"]
        full = RC.first_edge(res.all(axis=0), E)
        full95 = RC.first_edge(RC.weighted_fraction(res, w) >= 0.95 - 1e-12, E)
        for m in (30, 100, 300):
            if m >= n:
                continue
            fes, fes95 = [], []
            for _ in range(n_draws):
                sub = rng.choice(n, m, replace=False)
                fes.append(RC.first_edge(res[sub].all(axis=0), E))
                fes95.append(RC.first_edge(RC.weighted_fraction(res[sub], w[sub]) >= 0.95 - 1e-12, E))
            if b["label"] in CAL[:-2]:
                f1 = np.nanmedian(np.array([np.nan if u is None else u for u in fes]))
                f2 = np.nanmedian(np.array([np.nan if u is None else u for u in fes95]))
                nt += 1
                ns += abs(f1 - full) > 1e-6
                nf += abs(f2 - full95) > 1e-6
                if abs(f2 - full95) > 1e-6:
                    cases.append((b["label"], m))
    return {"seed": seed, "n_tests": nt, "strict_diff": int(ns), "frac95_diff": int(nf), "frac95_cases": cases}


nd = [n_dep(s) for s in (20261014, 1, 2, 3, 4, 5)]
out["n_dependence_seeds"] = nd
print("\nN dependence by seed:", nd)

# ---------------------------------------------------------------- grid dependence in dex
gd = R["grid_dependence"]
gdx = {}
for lab in LABS:
    ref = gd["fit"]["bins"][lab]["first_edge"]
    row = {}
    for g in ("old", "lee22", "x200"):
        v = gd[g]["bins"][lab]["first_edge"]
        row[g] = None if (v is None or ref is None) else round(float(np.log10(v / ref)), 4)
        if (v is None) != (ref is None):
            row[g] = f"fit {ref} vs {g} {v}"
    gdx[lab] = row
out["grid_dex_vs_fit"] = gdx
print("\ngrid dex differences vs fit:", json.dumps(gdx, indent=0))
mx = {}
for grp, labs in (("calibrated", CAL), ("exploratory", LABS[:5])):
    vals = [abs(v) for lab in labs for v in gdx[lab].values() if isinstance(v, float)]
    mx[grp] = max(vals)
out["grid_dex_max"] = mx
print("max |dex|:", mx)

# ---------------------------------------------------------------- single values
ID = R["inner_decomposition"]
Ea = A["stack|ref|E|14.3-14.5|value"]
Ex = A["stack|ref_E_xray|14.3-14.5|value"]
m = A["fit|ref|E|2|14.3-14.5|model"]
out["hot_share_14.3-14.5_shell5"] = float((Ex[5] - m[5]) / (Ea[5] - m[5]))
print("\nhot share of excess 14.3-14.5 shell 5 (full precision):", out["hot_share_14.3-14.5_shell5"])
for lab in ("13.5-13.7", "13.9-14.1", "14.3-14.5", "14.9-15.1"):
    Ea = A[f"stack|ref|E|{lab}|value"]
    Ex = A[f"stack|ref_E_xray|{lab}|value"]
    m = A[f"fit|ref|E|2|{lab}|model"]
    print(f"  {lab}: hot share shells 1/5 = {(Ex[1] - m[1]) / (Ea[1] - m[1]):.4f} / {(Ex[5] - m[5]) / (Ea[5] - m[5]):.4f}")
print("ref 2R Y(<R200c) 13.5-13.7:", PS["ref"]["fits"]["P"]["2R"]["13.5-13.7"]["int_lt1_vs_truth"])
print("mc500 4R tSZ3000 missing share:", PS["resolution:mc500"]["summary"]["impact_fit4R_vs_ref_fit"]["tSZ_yy_total_ell3000"])
print("mconly first edges 15.1-15.5:", {s: [PS[f"resolution:{s}"]["summary"]["first_edge"][l] for l in CAL[-2:]]
                                       for s in ("mconly0", "mconly5", "mconly10", "mconly20", "mconly50",
                                                 "mconly100", "mconly200", "mconly500")})
print("mconly first edges 14.9-15.1:", {s: PS[f"resolution:{s}"]["summary"]["first_edge"]["14.9-15.1"]
                                       for s in ("mconly5", "mconly10", "mconly20", "mconly50")})
print("frac95 first edges:", PS["resolution:frac95"]["summary"]["first_edge"])
print("per_halo_f50 first edges:", PS["resolution:per_halo_f50"]["summary"]["first_edge"])
print("mc500 kept 13.5-13.7:", PS["resolution:mc500"]["masks"]["13.5-13.7"]["kept"])

# web variants: fields per variant
W = json.loads((G / "web_variants.json").read_text())
fields = {}
for rec in W["records"]:
    fields.setdefault(rec["variant"], set()).add(rec.get("field"))
fields = {k: sorted(v) for k, v in fields.items()}
out["web_fields"] = fields
print("\nweb variant fields:", json.dumps(fields))

json.dump(out, open(G / "section_checks_extra.json", "w"), indent=1, default=str)
print("wrote section_checks_extra.json")
