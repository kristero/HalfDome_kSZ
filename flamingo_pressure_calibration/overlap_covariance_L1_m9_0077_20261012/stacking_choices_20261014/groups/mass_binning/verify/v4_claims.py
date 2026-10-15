"""Round-4 verifier: the five previously wrong claims, recomputed from results.json only (own numpy)."""
import json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
R = json.loads((HERE / "results.json").read_text())
A = R["analyses"]
cp = A["scheme_impacts_continuous_painting"]
it = A["interpolant_test"]
pa = A["painting_assignment_estimate"]

print("==== Claim 0: painting estimate vs metric C / B")
for S in ("M200c_0.2", "M200c_0.3", "M200c_0.5"):
    for m in ("smooth", "step"):
        print(S, m, pa[f"{S}|{m}"])
base = pa["M200c_0.2|smooth"]["vs_smooth_M200c_0.1_pct"]
pairs = {"y_mean": ("P", "y_mean"), "DM_mean": ("E", "DM_mean"),
         "tSZ_1h_lowell": ("P", "tSZ_yy_total_ell100"), "kSZ_1h_lowell": ("E", "kSZ_1h_ell100")}
for S in ("M200c_0.3", "M200c_0.5"):
    a = pa[f"{S}|smooth"]["vs_smooth_M200c_0.1_pct"]
    for k, (f, o) in pairs.items():
        ratio = 100 * ((1 + a[k] / 100) / (1 + base[k] / 100) - 1)
        diff = a[k] - base[k]
        c = cp[S][f]["vs_reference_painted_same_way"][o]
        b = cp[S][f]["vs_truth"][o]
        t = pa[f"{S}|smooth"]["vs_truth_pct"][k]
        print(f"  {S} {k}: est(ratio) {ratio:+.3f} est(diff) {diff:+.3f} | C {c['pct']:+.3f}±{c['err_pct']:.3f} "
              f"pull {(ratio - c['pct']) / c['err_pct']:+.2f} | est vs truth {t:+.3f}  B {b['pct']:+.3f}±{b['err_pct']:.3f}")

print("==== Claim 1: mass trend slopes")
mt = A["mass_trend"]
bands = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4"]
def sl(f, bn, band, key="slope_central_per_0p1dex"):
    for e in mt[f]:
        if e["bin"] == bn:
            s = (e.get(key) or {}).get(band)
            if s is None:
                return None
            if isinstance(s, dict):
                return (100 * s["value"], 100 * s.get("err", np.nan))
            return (s, np.nan)
    return None
print("keys of a mass_trend entry:", list(mt["P"][3].keys()))
bins = [e["bin"] for e in mt["P"]]
for f in ("P", "E"):
    for bn in bins:
        row = [sl(f, bn, b) for b in bands]
        print(f, bn, " ".join("   --    " if r is None else f"{r[0]:+5.1f}±{r[1]:.1f}" for r in row))

print("==== Claim 1b: curvature per dex^2 at 0.25-0.5 and 0.5-1")
for f in ("P", "E"):
    for e in mt[f]:
        c = e.get("curvature_per_dex2") or {}
        if c:
            print(f, e["bin"], {b: c.get(b) for b in ("0.25-0.5", "0.5-1")})
lo, hi = [], []
for f in ("P", "E"):
    for e in mt[f]:
        c = e.get("curvature_per_dex2") or {}
        lm = float(e["bin"].split("-")[0])
        for b in ("0.25-0.5", "0.5-1"):
            if b in c and c[b] is not None:
                v = c[b]["value"] if isinstance(c[b], dict) else c[b]
                if 13.5 <= lm < 14.3: lo.append(v)
                elif 14.3 <= lm < 15.1: hi.append(v)
lo, hi = np.array(lo), np.array(hi)
print("13.5-14.3: n", lo.size, "range", lo.min(), lo.max(), "all neg", np.all(lo < 0))
print("14.3-15.1: n", hi.size, "range", hi.min(), hi.max(), "n pos", np.sum(hi > 0))

print("==== Claim 2: subsample mean vs median bias (P)")
pr = A["sample_size"]["precision"]
for band in ("2-3", "3-4"):
    for N in ("10", "25", "50"):
        rows = []
        for bn in pr:
            if N not in pr[bn]:
                continue
            r = pr[bn][N]["P"][band]
            se = r["sd"] / np.sqrt(400)
            rows.append((bn, 100 * r["mean_bias"], 100 * se, r["mean_bias"] / se, 100 * r["median_bias"]))
        mb = np.array([x[1] for x in rows]); se = np.array([x[2] for x in rows]); z = np.array([x[3] for x in rows]); md = np.array([x[4] for x in rows])
        print(f"P {band} N={N}: mean_bias {mb.min():+.2f}..{mb.max():+.2f}  se {se.min():.2f}..{se.max():.2f}  |z|max {np.abs(z).max():.2f}  median {md.min():+.2f}..{md.max():+.2f}")
        for x in rows:
            print(f"     {x[0]} mean {x[1]:+.2f} se {x[2]:.2f} z {x[3]:+.2f} median {x[4]:+.2f}")

print("==== Claim 3: per-bin attribution of 0.5-dex losses")
S = "M200c_0.5"
def pb(d):
    return {k: round(v, 3) for k, v in d["per_bin_pct"].items()}
for lab, src in (("C linear", cp[S]["E"]["vs_reference_painted_same_way"]), ("B linear", cp[S]["E"]["vs_truth"]),
                 ("B pchip", it["pchip"][S]["E"]["vs_truth"]), ("pchip vs 0.2 pchip", it["pchip"][S]["E"]["vs_0.2dex_same_interp"])):
    for o in ("kSZ_1h_ell3000", "DM_mean", "DM_var", "kSZ_1h_ell1000"):
        d = src[o]; p = pb(d)
        low3 = sum(d["per_bin_pct"][b] for b in ("13.5-13.7", "13.7-13.9", "13.9-14.1"))
        print(f"{lab} {o}: total {d['pct']:+.3f}±{d['err_pct']:.3f} sum(per_bin) {sum(d['per_bin_pct'].values()):+.3f} 13.5-14.1 sum {low3:+.3f}")
        print("     ", p)
for lab, src in (("C linear", cp[S]["P"]["vs_reference_painted_same_way"]), ("B linear", cp[S]["P"]["vs_truth"]),
                 ("B pchip", it["pchip"][S]["P"]["vs_truth"]), ("pchip vs 0.2 pchip", it["pchip"][S]["P"]["vs_0.2dex_same_interp"])):
    for o in ("tSZ_yy_total_ell1000",):
        d = src[o]; p = pb(d)
        s1 = d["per_bin_pct"]["14.9-15.1"] + d["per_bin_pct"]["15.1-15.3"]
        print(f"{lab} {o}: total {d['pct']:+.3f}±{d['err_pct']:.3f} 15.3-15.5 {d['per_bin_pct']['15.3-15.5']:+.3f} 14.9-15.3 {s1:+.3f}")
        print("     ", p)
# node mean mass of the 15.0-15.5 bin
print("n_extrapolated (0.5 dex):", cp[S]["P"]["n_extrapolated_haloes"])

print("==== 0.5-dex PCHIP tSZ ell=1000 vs truth: grouped per-bin contributions")
d = it["pchip"]["M200c_0.5"]["P"]["vs_truth"]["tSZ_yy_total_ell1000"]["per_bin_pct"]
g = {"13.5-14.9": sum(d[b] for b in ("13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9")),
     "14.9-15.3": d["14.9-15.1"] + d["15.1-15.3"], "15.3-15.5": d["15.3-15.5"]}
print({k: round(v, 3) for k, v in g.items()}, "total", round(sum(g.values()), 3))
print("==== curvature in 15.1-15.3 (0.25-0.5, 0.5-1)")
for f in ("P", "E"):
    for e in mt[f]:
        if e["bin"] == "15.1-15.3":
            print(f, {b: e["curvature_per_dex2"].get(b) for b in ("0.25-0.5", "0.5-1")})
