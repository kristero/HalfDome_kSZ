"""Round-4 verifier: spot checks of tables and other quantitative statements (results.json only)."""
import json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
R = json.loads((HERE / "results.json").read_text())
A = R["analyses"]
cp = A["scheme_impacts_continuous_painting"]
sp = A["scheme_impacts_smooth_painting"]
it = A["interpolant_test"]
obsP = ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "y_mean"]
obsE = ["kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var"]
def fmt(d):
    return f"{d['pct']:+.2f}±{d['err_pct']:.2f}"
print("keys smooth:", list(sp["M200c_0.3"].keys()), list(sp["M200c_0.3"]["P"].keys()))
print("==== metric C table")
for S in ("M200c_0.1", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5", "Lee22_h"):
    print(S, " ".join(fmt(cp[S]["P"]["vs_reference_painted_same_way"][o]) for o in obsP), "|",
          " ".join(fmt(cp[S]["E"]["vs_reference_painted_same_way"][o]) for o in obsE))
print("==== metric B (cp vs_truth, linear)")
for S in ("M200c_0.2", "M200c_0.1", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5", "Lee22_h"):
    print(S, " ".join(fmt(cp[S]["P"]["vs_truth"][o]) for o in obsP), "|", " ".join(fmt(cp[S]["E"]["vs_truth"][o]) for o in obsE))
print("==== interpolant_test vs_truth")
for interp in ("linear", "pchip"):
    for S in ("M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5"):
        print(interp, S, " ".join(fmt(it[interp][S]["P"]["vs_truth"][o]) for o in obsP), "|", " ".join(fmt(it[interp][S]["E"]["vs_truth"][o]) for o in obsE))
print("==== metric A (smooth painting)")
def get_a(S, f, o):
    d = sp[S][f]
    if o in d: return d[o]
    for k in d:
        if isinstance(d[k], dict) and o in d[k]:
            return d[k][o]
for S in ("M200c_0.2", "M200c_0.1", "M200c_0.3", "M200c_0.5", "Lee22_h"):
    try:
        print(S, " ".join(fmt(get_a(S, "P", o)) for o in obsP), "|", " ".join(fmt(get_a(S, "E", o)) for o in obsE))
    except Exception as ex:
        print(S, "ERR", ex, list(sp[S]["P"].keys()))

print("==== metric A / metric C factors at 0.3 dex")
a_k = get_a("M200c_0.3", "E", "kSZ_1h_ell3000")["pct"]; c_k = cp["M200c_0.3"]["E"]["vs_reference_painted_same_way"]["kSZ_1h_ell3000"]["pct"]
a_v = get_a("M200c_0.3", "E", "DM_var")["pct"]; c_v = cp["M200c_0.3"]["E"]["vs_reference_painted_same_way"]["DM_var"]["pct"]
print(f"kSZ3000 {a_k:.4f}/{c_k:.4f} = {a_k/c_k:.3f};  DMvar {a_v:.4f}/{c_v:.4f} = {a_v/c_v:.3f}")

print("==== top bins")
tb = A["top_bins"]
bs = tb["bin_shares"]
# shares as fractions of the calibrated-bin part
for f, obs in (("P", obsP), ("E", obsE)):
    for o in obs:
        tot = sum(bs[f][b][o] for b in bs[f])
        top = bs[f]["15.1-15.3"][o] + bs[f]["15.3-15.5"][o]
        samp = sum(bs[f][b][o] for b in ("13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5"))
        print(f"  {o}: sum shares {tot:.4f} top2 {100*top:.2f} top2/sum {100*top/tot:.2f}  sampled13.5-14.5 {100*samp:.2f}")
print(tb["notes"]["bin_shares"])
rs = tb["reference_sample_error_pct"]
print(tb["notes"]["reference_sample_error_pct"])
for f in ("P", "E"):
    for des in rs[f]:
        for sel in rs[f][des]:
            print(f, des, sel, {k: round(v, 3) for k, v in rs[f][des][sel].items()})
print("---- continuous painting treatments (vs reference painted same way)")
cpt = tb["continuous_painting"]
for t in cpt:
    P = cpt[t]["P"]["vs_reference_painted_same_way"]; E = cpt[t]["E"]["vs_reference_painted_same_way"]
    print(t, " ".join(fmt(P[o]) for o in obsP[:3]), "| kSZ100", fmt(E["kSZ_1h_ell100"]), "DMvar", fmt(E["DM_var"]))
    if t == "reference":
        Pt = cpt[t]["P"]["vs_truth"]; print("   ref vs truth", " ".join(fmt(Pt[o]) for o in obsP[:3]))
print("---- step impacts")
for t in tb["impacts"]:
    P = tb["impacts"][t]["P"]
    print(t, " ".join(fmt(P[o]) for o in obsP[:3]))
print("individual rel_err P 0.5-1:", {b: tb["individual"][b]["P"]["0.5-1"] for b in tb["individual"]})
print(tb["notes"]["individual"])

print("==== normalisation")
nb = A["normalisation_own_vs_binmean"]
imp = nb["impacts"]
for S in imp:
    P = imp[S]["P"]; E = imp[S]["E"]
    print(S, " ".join(f"{fmt(P[o])}({P[o]['pct']/P[o]['err_pct']:.2f}s)" for o in obsP), "|", " ".join(fmt(E[o]) for o in obsE))
print("band table")
first = nb["bins"]["M200c_0.2"][5]["fields"]["P"]
print("example band entry:", first.get("0.5-1"))
def band(entry, f, b):
    x = entry["fields"][f].get(b)
    if x is None: return "  --  "
    if isinstance(x, dict):
        v = x.get("value", x.get("ratio")); e = x.get("err")
        return f"{100*(v-1):+.1f}±{100*e:.1f}" if v is not None and v > 0.5 else str(x)
    return str(x)
for S, bl in (("M200c_0.2", ("13.5-13.7", "14.1-14.3", "14.5-14.7")), ("M200c_0.5", ("13.5-14.0", "14.0-14.5", "14.5-15.0"))):
    for e in nb["bins"][S]:
        if e["bin"] in bl:
            cells = [band(e, "P", b) for b in ("0.25-0.5", "0.5-1", "1-1.5", "2-3")] + [band(e, "E", b) for b in ("0.5-1", "1-1.5")] + \
                    [band(e, "T_e", "0.5-1")] + [band(e, "K", b) for b in ("0.25-0.5", "0.5-1")]
            print(S, e["bin"], f"sd_ln_s {e['sd_ln_s']:.3f}", " ".join(cells), "s_range", e["s_range"], "Pbar_arith_over_geo", e["Pbar_arith_over_geo"])
# s ranges and Pbar_arith_over_geo over calibrated bins
for S in ("M200c_0.2", "M200c_0.5"):
    sr = [e["s_range"] for e in nb["bins"][S]]
    pg = [e["Pbar_arith_over_geo"] for e in nb["bins"][S]]
    print(S, "s min", min(s[0] for s in sr), "s max", max(s[1] for s in sr), "Pbar range", min(pg), max(pg))
    for e in nb["bins"][S]:
        print("    ", e["bin"], e["N"], e["s_range"], e["Pbar_arith_over_geo"])

print("==== Jensen decomposition")
jd = nb["jensen_decomposition"]
rows = [("13.5-14.0", "P", "0.25-0.5"), ("13.5-14.0", "P", "0.5-1"), ("13.5-14.0", "E", "0.25-0.5"), ("13.5-14.0", "E", "0.5-1"),
        ("13.5-14.0", "T_e", "0.5-1"), ("13.5-14.0", "K", "0.25-0.5"), ("13.5-14.0", "K", "0.5-1"), ("14.5-15.0", "P", "0.5-1"), ("14.5-15.0", "E", "0.5-1")]
for b, f, band_ in rows:
    x = jd["M200c_0.5"][b]["fields"][f][band_]
    print(b, f, band_, f"{x['measured_pct']:+.1f}±{x['measured_err_pct']:.1f} pure {x['pure_jensen_pct']:+.1f} 2nd {x['second_order_pct']:+.1f} trend {x['trend_model_pct']:+.1f} a+alpha {x['a_plus_alpha']:.2f} beta {x['beta']:+.2f}  meas/pure {x['measured_pct']/x['pure_jensen_pct']:.2f}")
# pulls
def pulls(S, key):
    out = []
    for b in jd[S]:
        for f in jd[S][b]["fields"]:
            for band_, x in jd[S][b]["fields"][f].items():
                if x is None or x.get("measured_err_pct") in (None, 0) or x.get(key) is None: continue
                if not np.isfinite(x["measured_err_pct"]) or x["measured_err_pct"] <= 0: continue
                out.append((x["measured_pct"] - x[key]) / x["measured_err_pct"])
    return np.array(out)
for key in ("pure_jensen_pct", "second_order_pct", "trend_model_pct"):
    for S in ("M200c_0.2", "M200c_0.5"):
        p = pulls(S, key)
        print(key, S, "n", p.size, "rms", round(float(np.sqrt(np.mean(p**2))), 3), "max|p|", round(float(np.abs(p).max()), 2))
    p = np.concatenate([pulls("M200c_0.2", key), pulls("M200c_0.5", key)])
    print(key, "ALL n", p.size, "max|p|", round(float(np.abs(p).max()), 2))
# a+alpha ranges in 0.5-dex bins at 0.25-1
for f in ("P", "E", "T_e", "K"):
    v = [jd["M200c_0.5"][b]["fields"][f][bb]["a_plus_alpha"] for b in jd["M200c_0.5"] for bb in ("0.25-0.5", "0.5-1")
         if jd["M200c_0.5"][b]["fields"][f].get(bb) is not None]
    print("a+alpha", f, round(min(v), 2), round(max(v), 2), "n", len(v))
for f in ("P", "E", "T_e", "K"):
    for b in ("13.5-14.0",):
        print("beta", f, b, {bb: round(jd["M200c_0.5"][b]["fields"][f][bb]["beta"], 3) for bb in ("0.25-0.5", "0.5-1")})
bE = [jd["M200c_0.5"]["13.5-14.0"]["fields"]["E"][bb]["beta"] for bb in ("0.25-0.5", "0.5-1")]
bT = [jd["M200c_0.5"]["13.5-14.0"]["fields"]["T_e"][bb]["beta"] for bb in ("0.25-0.5", "0.5-1")]
print("beta_T - 2/3 beta_E:", [round(t - 2/3*e, 3) for t, e in zip(bT, bE)])

print("==== pull summaries 13.5-15.1")
ps = A["scheme_pull_summary_13.5_15.1"]
for S in ps:
    print(S, {f: (ps[S][f]["n"], round(ps[S][f]["rms_pull"], 3), ps[S][f]["n_abs_gt_3"]) for f in ("P", "E")})
print("==== wide vs fine: P 0.5 dex bins")
w = A["wide_vs_fine_summary"]["calibrated"]["M200c_0.5"]["P"]
print({k: v for k, v in w.items() if k != "bins"})
print(json.dumps(w["bins"])[:1500])
