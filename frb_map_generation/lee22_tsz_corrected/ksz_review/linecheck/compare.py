#!/usr/bin/env python3
"""Compare paint_ksz_websky.jl maps (Julia 1.6.2, idark) with the independent py_painter.py maps and analytic per-halo integrals."""
import argparse
import json

import healpy as hp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--nside", type=int, default=1024)
a = ap.parse_args()
R = a.root; ns = a.nside; om = 4 * np.pi / hp.nside2npix(ns)
V5 = ("b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky")
WS_SCALE = 1.1116436255321183
res = {}


def jmap(run, v):
    return hp.read_map("%s/%s/maps/halfdome_ksz_%s_nside%d.fits" % (R, run, v, ns), dtype=np.float64, verbose=False)


def prov(run):
    d = {}
    for line in open("%s/%s/maps/provenance.txt" % (R, run)):
        k, _, val = line.strip().partition("=")
        d[k] = val
    return d


for run, tag in (("julia_subset", "subset"), ("julia_subset_nocomp", "subset_nocomp"), ("julia_first200k", "first200k")):
    print("\n==== %s  vs  py_%s" % (run, tag))
    H = np.load("%s/py/py_%s_halos.npz" % (R, tag))
    p = prov(run)
    print("halos: julia %s python %d | websky-selected: julia %s python %d | exact low-z: julia %s python %d | vlos rms julia %s python %.4f"
          % (p["halos_read"], H["z"].size, p["websky_selected_halos"], H["cut"].sum(), p["exact_lowz_halos"], np.sum(H["z"] < 0.1),
             p["vlos_rms_kms"], np.sqrt(np.mean((H["v"] * 299792.458)**2))))
    print("%-18s %13s %13s %13s | %9s %9s | %10s %10s %10s %10s" % ("variant", "sum J*Om", "sum P*Om", "analytic", "J/P-1", "J/A-1",
                                                                    "rms(J-P)/", "max|J-P|/", "sumJ2/", "noise(P-A)"))
    print("%-18s %13s %13s %13s | %9s %9s | %10s %10s %10s %10s" % ("", "", "", "", "", "", "rms(P)", "max|P|", "sumP2-1", "/|A|"))
    for v in V5:
        J = jmap(run, v); P = np.load("%s/py/py_%s_%s.npy" % (R, tag, v))
        SJ, SP, SA = J.sum() * om, P.sum() * om, H["an_" + v].sum()
        d = J - P
        r = dict(SJ=SJ, SP=SP, SA=SA, JP=SJ / SP - 1, JA=SJ / SA - 1, rms=np.sqrt(np.mean(d**2)) / np.sqrt(np.mean(P**2)),
                 mx=np.abs(d).max() / np.abs(P).max(), pw=np.sum(J**2) / np.sum(P**2) - 1,
                 noise=np.sqrt(np.sum((H["ps_" + v] - H["an_" + v])**2)) / abs(SA))
        res["%s:%s" % (run, v)] = r
        print("%-18s %13.6e %13.6e %13.6e | %9.2e %9.2e | %10.2e %10.2e %10.2e %10.2e" % (v, SJ, SP, SA, r["JP"], r["JA"], r["rms"], r["mx"], r["pw"], r["noise"]))
        # resolved halos: python pixel sums vs analytic (validates the python painter and the integrals)
    for v in V5:
        an, ps = H["an_" + v], H["ps_" + v]
        for lab, m in (("npix>=300 & thmax<0.03", (H["npix"] >= 300) & (H["thmax"] < 0.03) & (an != 0)),
                       ("npix>=300 & thmax>=0.03", (H["npix"] >= 300) & (H["thmax"] >= 0.03) & (an != 0))):
            if m.sum() > 0:
                q = ps[m] / an[m]
                print("   per-halo py pixel sum / analytic  %-17s %-24s n=%5d  median %.5f  16-84%% [%.5f, %.5f]" % (
                    v, lab, m.sum(), np.median(q), *np.percentile(q, [16, 84])))

print("\n==== test (ii): Julia b16_websky with --no-comp vs %.7f x independent B16 1.5 R200c map (WebSky selection/mass, no comp)" % WS_SCALE)
J = jmap("julia_subset_nocomp", "b16_websky"); P = np.load("%s/py/py_subset_nocomp_b16_ws_plain.npy" % R)
Pp = np.load("%s/py/py_subset_b16_ws_plain.npy" % R)
print("python plain maps identical between the two python runs:", np.array_equal(P, Pp))
big = np.abs(P) > 1e-3 * np.abs(P).max()
q = J[big] / P[big]
print("sum J / sum P_plain = %.7f   pixels |P|>1e-3 max: n=%d  median J/P %.7f  1-99%% [%.7f, %.7f]  rms(J-%.4f P)/rms(%.4f P) = %.2e"
      % (J.sum() / P.sum(), big.sum(), np.median(q), *np.percentile(q, [1, 99]), WS_SCALE, WS_SCALE,
         np.sqrt(np.mean((J - WS_SCALE * P)**2)) / np.sqrt(np.mean((WS_SCALE * P)**2))))
res["ii"] = dict(sum_ratio=J.sum() / P.sum(), median_ratio=float(np.median(q)))
print("\n==== compensation term alone: Julia (subset - subset_nocomp) vs python (subset - subset_nocomp)")
for v in ("b16_websky", "lee22corr_websky"):
    Jc = jmap("julia_subset", v) - jmap("julia_subset_nocomp", v)
    Pc = np.load("%s/py/py_subset_%s.npy" % (R, v)) - np.load("%s/py/py_subset_nocomp_%s.npy" % (R, v))
    H = np.load("%s/py/py_subset_halos.npz" % R)
    print("%-18s sum Jc*Om %.6e  sum Pc*Om %.6e  analytic comp %.6e   rms(Jc-Pc)/rms(Pc) %.2e   sign check: corr(Jc, +v-weighted) J.P>0: %s"
          % (v, Jc.sum() * om, Pc.sum() * om, H["an_comp_only"].sum(), np.sqrt(np.mean((Jc - Pc)**2)) / np.sqrt(np.mean(Pc**2)), np.dot(Jc, Pc) > 0))
for v in ("b16_phys", "lee22corr_phys", "lee22pipe_phys"):
    print("phys map unchanged by --no-comp (%s): max |diff| = %.3e" % (v, np.abs(jmap("julia_subset", v) - jmap("julia_subset_nocomp", v)).max()))
json.dump(res, open("%s/compare_summary.json" % R, "w"), indent=1, default=float)
