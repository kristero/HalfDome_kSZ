"""Read-only checks behind statements in SECTION.md that are not printed by report_tables.py (stdout only).

Reads results.json, the core's grid edges and bins, and Test F's importance_F.npz.  Writes nothing.
Run:  LD_LIBRARY_PATH=<halfdome env>/lib <halfdome env>/bin/python -B section_checks.py
"""
from __future__ import annotations

import json

import numpy as np

import gs_common as C

sc = C.sc
R = json.loads((C.HERE / "results.json").read_text())
D = R["derived_summaries"]
CAL = R["calibrated_bins"]


def h(t):
    print(f"\n==== {t}")


# ---------------------------------------------------------------- grids
h("grids: shell counts, ranges, widths")
for g in ("fit", "x200", "lee22", "old", "x500", "b24"):
    e = sc.edges(g)
    d = np.diff(np.log10(e[1:]))
    print(g, "shells", len(e) - 1, "range", float(e[0]), float(e[-1]), "min/max dex (excluding the first shell)",
          round(float(d.min()), 5), round(float(d.max()), 4))
fe = sc.edges("fit")
dfe = np.diff(np.log10(fe[1:]))
print("fit edges", np.round(fe, 4).tolist())
print("fit dex per shell (from the second shell)", np.round(dfe, 4).tolist())
xe = sc.edges("x200")
dx = np.r_[np.inf, np.diff(np.log10(xe[1:]))]          # dx[i] = width of shell i (shell 0 starts at 0)
k = int(np.argmin(dx))
print("x200 thinnest shell", xe[k], xe[k + 1], "dex", dx[k], "dlnx", np.log(xe[k + 1] / xe[k]))
thin = [(round(float(xe[i]), 4), round(float(xe[i + 1]), 4), round(float(dx[i]), 4)) for i in range(1, len(dx))
        if dx[i] < 0.02]
print("x200 edges beyond 0.95 R200c:", [round(float(v), 4) for v in xe if v > 0.95])
print("x200 shells thinner than 0.02 dex:", thin)
oe = sc.edges("old")
print("old edges near 1:", [round(float(x), 4) for x in oe if 0.9 < x < 1.1])
print("sliver 1.34-1.3508 dex:", np.log10(1.3507999847024064 / 1.34))

# ---------------------------------------------------------------- bins
h("bin sizes (calibrated)")
for b in C.calibrated(C.bins()):
    print(b["label"], "N", b["N"], "N_eff", round(float(b["N_eff"]), 1))

# ---------------------------------------------------------------- self-similarity: spread / noise
h("cross-bin spread / median stack noise at the tabulated radii")
X_TAB = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
for f in ("P", "E"):
    for name, s in R["self_similarity"][f].items():
        x = s["x_R200c_equivalent"]
        sp, nz = s["common_bins_spread_sd_lnF"], s["median_bootstrap_fractional_error_of_stack"]
        r = [sp[x.index(xx)] / nz[x.index(xx)] for xx in X_TAB if sp[x.index(xx)] is not None]
        print(f, name, "spread/noise", round(min(r), 1), "-", round(max(r), 1))
h("within-bin scatter: R500c/P_e,500 minus ref (P)")
ssP = R["self_similarity"]["P"]
a = ssP["R200c, P200 (ref)"]["common_bins_within_bin_sd_ln_median"]
b = ssP["R500c, P_e,500 (B24)"]["common_bins_within_bin_sd_ln_median"]
c = ssP["R500c, P200"]["common_bins_within_bin_sd_ln_median"]
x = ssP["R200c, P200 (ref)"]["x_R200c_equivalent"]
print("x", x)
print("R500c,P_e500 - ref", [None if u is None else round(v - u, 3) for u, v in zip(a, b)])
print("R500c,P200  - ref", [None if u is None else round(v - u, 3) for u, v in zip(a, c)])

# ---------------------------------------------------------------- cumulative budgets, all 10 calibrated bins
h("% of the value inside 4R lying beyond 1 R200c, all calibrated bins")
for f in ("P", "E"):
    cu = R["cumulative_fractions_fit_grid"][f]
    v = {l: cu[l]["beyond_1"]["value"] for l in CAL}
    print(f, {l: round(100 * x, 1) for l, x in v.items()}, "range", round(100 * min(v.values()), 1), "-",
          round(100 * max(v.values()), 1))
    g4 = {l: (round(cu[l]["dlnI_dlnx_3.634-4"]["value"], 3), round(cu[l]["dlnI_dlnx_3.634-4"]["err"], 3)) for l in CAL}
    print(f, "dlnI/dlnx 3.634-4 (value, err)", g4)

# ---------------------------------------------------------------- clumping, all calibrated bins with resolved bands
h("clumping band medians over the calibrated bins")
inner = ("0-0.1", "0.1-0.25", "0.25-0.5")
fv_in, fv_out, cp_out, crho_beyond, qv_diff, bins_used = [], [], [], [], [], []
for l in CAL:
    bm = R["clumping"][l]["band_medians_resolved"]
    fvi = [bm["f_V"][k] for k in inner if bm["f_V"][k] is not None]
    if fvi:
        fv_in += fvi
        bins_used.append(l)
    if bm["f_V"]["3-4"] is not None:
        fv_out.append(bm["f_V"]["3-4"])
        cp_out.append(bm["C_P"]["3-4"])
    crho_beyond += [bm["C_rho"][k] for k in ("1-1.5", "1.5-2", "2-3", "3-4") if bm["C_rho"][k] is not None]
    for k in bm["P_QV/P_geo (ratio of stacks)"]:
        u, v = bm["P_QV/P_geo (ratio of stacks)"][k], bm["P_QVy/P_geo (ratio of stacks)"][k]
        if u is not None and v is not None:
            qv_diff.append(abs(u / v - 1))
print("bins with resolved inner bands:", bins_used)
print("f_V inside 0.5R:", round(min(fv_in), 4), "-", round(max(fv_in), 4))
print("f_V at 3-4R:", round(min(fv_out), 4), "-", round(max(fv_out), 4))
print("C_P at 3-4R:", round(min(cp_out), 2), "-", round(max(cp_out), 2))
print("C_rho at 1-4R:", round(min(crho_beyond), 1), "-", round(max(crho_beyond), 1))
print("max |P_QV / P_QVy - 1| over band medians:", max(qv_diff))

# ---------------------------------------------------------------- Test F importance: where tSZ lives
h("Test F importance share of the calibrated bins by radius (fit grid)")
IMP = dict(np.load(C.STUDY / "F_observables" / "results" / "importance_F.npz"))
print("keys", sorted(IMP))
ell = IMP["ell"]
assert np.allclose(IMP["shell_edges"], fe)
RANGES = (("<0.2335", 0.0, 0.2335), ("0.2335-1", 0.2335, 1.0), ("0.2739-1", 0.2739, 1.0), ("1-2", 1.0, 2.0),
          ("2-4", 2.0, 4.0))
for key, lab in (("imp__yytot", "tSZ"), ("imp__ksz1h", "kSZ 1h")):
    M = IMP[key][5:15]                        # calibrated bins 13.5-15.5 of the 16-bin engine
    for L in (100.0, 1000.0, 3000.0):
        j = int(np.flatnonzero(ell == L)[0])
        m = M[..., j].sum(axis=0)
        tot = m.sum()
        sh = {}
        for name, lo, hi in RANGES:
            sel = (fe[:-1] >= lo - 1e-3) & (fe[1:] <= hi + 1e-3)
            sh[name] = round(float(m[sel].sum() / tot), 3)
        print(lab, int(L), sh)

# ---------------------------------------------------------------- objective (a) sigma of split shells
h("objective (a) sigma: x200 sub-shell / parent fit shell (strict shells to 4R, calibrated bins)")
xe = sc.edges("x200")
for field in ("P", "E"):
    rat = []
    for b in C.calibrated(C.bins()):
        sf = sc.stack_bin(sc.make_key(field, grid="fit"), b, "mean", "strict", n_boot=C.N_BOOT)
        sx = sc.stack_bin(sc.make_key(field, grid="x200"), b, "mean", "strict", n_boot=C.N_BOOT)
        gf = np.maximum(sf["sd"] / sf["value"], 0.05)
        gx = np.maximum(sx["sd"] / sx["value"], 0.05)
        for i in range(len(xe) - 1):
            if not sx["resolved"][i] or xe[i + 1] > 4 + 1e-9:
                continue
            j = int(np.searchsorted(fe, xe[i] + 1e-9) - 1)
            if fe[j + 1] < xe[i + 1] - 1e-9 or not sf["resolved"][j]:
                continue
            if abs(fe[j] - xe[i]) < 1e-9 and abs(fe[j + 1] - xe[i + 1]) < 1e-9:
                continue                                  # the same shell on both grids
            rat.append(gx[i] / gf[j])
    rat = np.array(rat)
    print(field, "n", len(rat), "min", round(float(rat.min()), 3), "median", round(float(np.median(rat)), 3),
          "max", round(float(rat.max()), 3))
