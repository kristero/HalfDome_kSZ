"""v6b: extra spot checks (website resolution flags of subset records, zero-electron shells, v2 count, A2-ii 1.5-2R E band)."""
import json, os
import numpy as np
SEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(SEL))
CACHE = os.path.join(ROOT, "cache")
IDX = json.load(open(os.path.join(CACHE, "index.json")))["variants"]
TAB = dict(np.load(os.path.join(CACHE, "halo_table.npz")))
R = json.load(open(os.path.join(SEL, "results.json")))
W = json.load(open(os.path.join(SEL, "web_variants.json")))
E = np.load(os.path.join(CACHE, "edges_fit.npy")); LO = E[:-1]
me = TAB["mass_edges"]
out = {}
strict = {}
for k in range(len(me) - 1):
    rows = np.flatnonzero((TAB["logm200c"] >= me[k]) & (TAB["logm200c"] < me[k + 1]))
    lab = f"{me[k]:.1f}-{me[k+1]:.1f}"
    c = np.load(os.path.join(CACHE, "counts_fit.npy"), mmap_mode="r")[rows]
    h = np.load(os.path.join(CACHE, "hsml_fit.npy"), mmap_mode="r")[rows]
    eps = TAB["eps_over_r200c"][rows]
    with np.errstate(invalid="ignore"):
        ok = (LO[None, :] >= np.maximum(0.04, 4 * eps[:, None])) & (c >= 20) & (LO[None, :] >= 2 * h)
    strict[lab] = (rows, ok.all(axis=0))
# (1) website subset records flagged resolved although full-bin strict fails
subs = [v for v in W["variants"] if any(s in (v["id"] if isinstance(v, dict) else v) for s in ("clean", "overlap"))]
ids = [v["id"] if isinstance(v, dict) else v for v in W["variants"]]
cnt = {}
for rec in W["records"]:
    vid = rec["variant"]
    if not any(s in vid for s in ("clean", "overlap")):
        continue
    fs = strict[rec["bin"]][1]
    res = np.asarray(rec["resolved"], bool)
    n = int((res & ~fs).sum())
    if n:
        cnt.setdefault(rec["field"], 0)
        cnt[rec["field"]] += n
        if float(rec["bin"].split("-")[0]) >= 13.5:
            cnt.setdefault(rec["field"] + "_calibrated_13.5-15.5", 0)
            cnt[rec["field"] + "_calibrated_13.5-15.5"] += n
        if rec["field"] == "P":
            cnt.setdefault("P_by_variant", {}).setdefault(vid, 0)
            cnt["P_by_variant"][vid] += n
out["website_subset_resolved_but_fullbin_strict_fails"] = cnt
# (2) zero SWIFT electrons with positive full-ionisation electrons, and in strict shells of calibrated bins
E0 = np.load(os.path.join(CACHE, IDX["E|geometric|removed|all|all|fit|rhoc_fb"]["file"]))
EF = np.load(os.path.join(CACHE, IDX["E_fullion|geometric|removed|all|all|fit|rhoc_fb"]["file"]))
z = (E0 == 0) & (EF > 0)
nres = 0
for lab, (rows, st) in strict.items():
    if float(lab.split("-")[0]) >= 13.5:
        nres += int((z[rows] & st[None, :]).sum())
out["zero_E_halo_shells"] = int(z.sum())
out["zero_E_in_strict_cal"] = nres
# (3) A2-ii E 1.5-2 band, full precision from results.json
br = R["variants"]["selection:A2ii"]["band_ratio"]["E"]
out["A2ii_E_1.5-2"] = {l: br[l]["1.5-2"]["value"] for l in br if br[l].get("1.5-2")}
br = R["variants"]["selection:A2ii"]["band_ratio"]["P"]
out["A2ii_P_1.5-2"] = {l: br[l]["1.5-2"]["value"] for l in br if br[l].get("1.5-2")}
# (4) X = 0.76 N_e(<4R) from results.json integral ratios at full stored precision
mu = (1 / 1836.15267343 + 2 / 1.76) / 0.9
ir = R["variants"]["selection:e_gas_muxgp"]["integral_ratio"]["E"]
cal9 = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1", "15.1-15.3"]
out["X0.76_Ne4R"] = {l: (100 * (ir[l]["0-4"]["value"] * mu * 0.88 - 1), ir[l]["0-4"]["value"]) for l in cal9}
# (5) v2_independent comparison count
v2 = json.load(open(os.path.join(SEL, "verify", "v2_independent.json")))
def count(o):
    if isinstance(o, dict):
        if "match" in o or "ok" in o:
            return 1
        return sum(count(v) for v in o.values())
    if isinstance(o, list):
        return sum(count(v) for v in o)
    return 0
out["v2_independent_match_entries"] = count(v2)
out["v2_n_mismatch"] = v2.get("n_mismatch")
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(SEL, "verify", "v6b_extra.json"), "w"), indent=1)
