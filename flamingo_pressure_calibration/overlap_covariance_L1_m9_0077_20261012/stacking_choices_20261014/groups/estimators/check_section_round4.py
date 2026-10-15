"""Last-round text check of SECTION.md (2026-10-14).  Read-only.

Prints, from results.json in this folder, the numbers behind the statements
corrected in the last round: N and N_eff of the main bins, the bins in which
f is resolved at <0.1 and 0.1-0.25 R200c, the Q_M and Q_X mass trends per
band, the n_e kT_X values beyond 0.5 R200c, and n_e kT_X against the ratio of
the stacked temperatures T_X/T_e.  Each statement is also asserted.
Run: python -B check_section_round4.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
BANDS = R["bands_R200c"]
CAL = R["calibrated_bins"]
MAIN, TOP = CAL[:8], CAL[8:]
LOW = [b for b in MAIN if float(b.split("-")[1]) <= 14.5]


def br(variant, field, b, band, key="ratio"):
    for e in R["band_ratios"][variant][field][b]:
        if e["band"] == band:
            return e[key]
    raise KeyError(band)


def monotonic_up(v):
    return all(b > a for a, b in zip(v, v[1:]))


print("== N and N_eff")
bi = R["bin_info"]
for b in MAIN + TOP:
    print(f"  {b}: N = {bi[b]['N']}, N_eff = {bi[b]['N_eff']}")
nlow = [bi[b]["N"] for b in LOW]
neff = [bi[b]["N_eff"] for b in LOW]
print(f"  below 10^14.5 ({LOW[0]}..{LOW[-1]}): N {min(nlow)}-{max(nlow)}, "
      f"N_eff {min(neff):.1f}-{max(neff):.1f}")
assert (min(nlow), max(nlow)) == (317, 327)
assert (round(min(neff)), round(max(neff))) == (309, 319)

print("== f resolved bins at <0.1 and 0.1-0.25 R200c (10 calibrated bins)")
pb = R["diagnostics"]["filling_factor"]["per_bin_band"]
for band, nexp in (("<0.1", 5), ("0.1-0.25", 8)):
    vals = [(b, e["fill_mean"]) for b in CAL for e in pb[b]
            if e["band"] == band and e["fill_mean"] is not None]
    print(f"  {band}: {len(vals)} bins {vals[0][0]}..{vals[-1][0]}, "
          f"fill_mean {min(v for _, v in vals):.4f}-{max(v for _, v in vals):.4f}")
    assert len(vals) == nexp

print("== Q_M and Q_X per band (P, ratio +- paired err)")
trend = {}
for v in ("estimators:Q_M", "estimators:Q_X"):
    for band in BANDS:
        row = [(b, br(v, "P", b, band), br(v, "P", b, band, "err")) for b in MAIN]
        row = [x for x in row if x[1] is not None]
        print(f"  {v.split(':')[1]} {band:>8}: " +
              ", ".join(f"{b} {r:.4g}+-{e:.2g}" for b, r, e in row))
        trend[(v, band)] = [r for b, r, e in row]
        if band in ("0.25-0.5", "0.5-1"):
            sub = [r for b, r, e in row if b != "13.5-13.7"]
            print(f"      monotonic over 13.7-15.1: {monotonic_up(sub)}")
assert monotonic_up(trend[("estimators:Q_M", "0.25-0.5")][1:])
assert monotonic_up(trend[("estimators:Q_M", "0.5-1")])
assert monotonic_up(trend[("estimators:Q_X", "0.5-1")][1:])
assert not monotonic_up(trend[("estimators:Q_X", "0.25-0.5")][1:])
assert not monotonic_up(trend[("estimators:Q_M", "<0.1")])
assert not monotonic_up(trend[("estimators:Q_M", "0.1-0.25")])

print("== n_e kT_X (P) beyond 0.25 R200c, all calibrated bins")
for b in CAL:
    print(f"  {b}: " + ", ".join(
        f"{band} {br('estimators:nT_X', 'P', b, band):.4g}"
        for band in BANDS[2:]))
low = [br("estimators:nT_X", "P", b, band) for b in LOW for band in BANDS[2:]]
assert min(low) > 1
b1 = [br("estimators:nT_X", "P", "14.5-14.7", band) for band in BANDS[3:]]
assert min(b1) > 1
b2 = [br("estimators:nT_X", "P", "14.7-14.9", band) for band in BANDS[3:6]]
print(f"  14.5-14.7 beyond 0.5: {min(b1):.4g}-{max(b1):.4g}; "
      f"14.7-14.9 at 0.5-2: {min(b2):.4g}-{max(b2):.4g}")
below = [b for b in CAL if br("estimators:nT_X", "P", b, "2-3") < 1]
print(f"  < 1 at 2-3 R200c in: {below}")
assert below == ["15.1-15.3"]

print("== n_e kT_X (P) vs stacked T_X/T_e at 3-4 R200c, 8 main bins")
a = [br("estimators:nT_X", "P", b, "3-4") for b in MAIN]
t = [br("estimators:T_X", "T_e", b, "3-4") for b in MAIN]
print(f"  n_e kT_X {min(a):.4g}-{max(a):.4g};  T_X/T_e {min(t):.4g}-{max(t):.4g}")
print("all assertions passed")
