#!/usr/bin/env python
"""Add a compact 'summary' block to results.json: ranges over the calibrated bins (10^13.5-15.5) of the band ratios,
integral ratios, shares and impacts, so that every number quoted in the report can be cited as
results.json:summary.<...>.  Run after compute_selection.py.  Also prints the tables used in the report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
CAL = R["calibrated_bins"]
CAL_MAIN = [b for b in CAL if b != "15.3-15.5"]          # N = 6 bin reported separately
BANDS = list(R["bands"])
ANN = list(R["annuli"])


def rng(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return [round(min(v), 6), round(max(v), 6)] if v else None


def pct(x):
    return None if x is None else round(100 * x, 4)


summary = {"note": "ranges [min, max] over the calibrated bins 13.5-15.3 ('cal') and the value in 15.3-15.5 (N = 6, "
                   "'top'); band and integral numbers are (ratio - 1) in %, impacts in %", "variants": {}}
for vid, v in R["variants"].items():
    if vid == "ref":
        continue
    s = {"label": v["label"], "band_dev_pct": {}, "integral_dev_pct": {}, "impact_pct": {}}
    for f in v["fields"]:
        s["band_dev_pct"][f] = {}
        for bd in BANDS:
            vals = [pct(v["band_ratio"][f][b][bd]["dev"]) if v["band_ratio"][f][b][bd] else None for b in CAL_MAIN]
            top = v["band_ratio"][f]["15.3-15.5"][bd]
            s["band_dev_pct"][f][bd] = {"cal": rng(vals), "top": pct(top["dev"]) if top else None}
        if v["integral_ratio"].get(f):
            s["integral_dev_pct"][f] = {}
            for a in ANN:
                vals = [pct(v["integral_ratio"][f][b][a]["dev"]) for b in CAL_MAIN]
                s["integral_dev_pct"][f][a] = {"cal": rng(vals),
                                               "top": pct(v["integral_ratio"][f]["15.3-15.5"][a]["dev"])}
    for f, d in v["impact"].items():
        for o, x in d.items():
            s["impact_pct"][o] = {"value": x["dO_over_O_pct"], "err": x["err_pct"],
                                  "resolved_only": x["resolved_only_pct"]}
    summary["variants"][vid] = s

for name in ("agn_share_of_total", "cold_sf_share_of_total"):
    out = {}
    for f, d in R[name].items():
        out[f] = {"integral_pct": {a: {"cal": rng([pct(d[b]["integral"][a]["value"]) for b in CAL_MAIN]),
                                       "top": pct(d["15.3-15.5"]["integral"][a]["value"])} for a in ANN},
                  "band_strict_pct": {bd: {"cal": rng([pct(d[b]["band_strict"][bd]["value"])
                                                       if d[b]["band_strict"][bd] else None for b in CAL_MAIN])}
                                      for bd in BANDS},
                  "max_shell_pct": {b: [pct(d[b]["max_all_shells"]["value"]), d[b]["max_all_shells"]["x"]]
                                    for b in d}}
        out[f]["integral_pct_by_bin"] = {b: {a: pct(d[b]["integral"][a]["value"]) for a in ANN} for b in d}
    summary[name] = out

own = {}
for f, d in R["owner_split"].items():
    own[f] = {}
    for c in ("central_bound", "own_satellites", "unbound", "bound_to_other_haloes"):
        own[f][c] = {a: {"cal": rng([pct(d[b][c]["integral"][a]["value"]) for b in CAL_MAIN]),
                         "top": pct(d["15.3-15.5"][c]["integral"][a]["value"])} for a in ANN}
summary["owner_split_integral_share_pct"] = own
R["summary"] = summary
(HERE / "results.json").write_text(json.dumps(R, separators=(",", ":"), allow_nan=False))

# ------------------------------------------------------------------ printed tables
if __name__ == "__main__":
    S = summary["variants"]
    print("BAND DEV % (cal range 13.5-15.3 | top 15.3-15.5)")
    for vid, s in S.items():
        for f, d in s["band_dev_pct"].items():
            row = []
            for bd in BANDS:
                c = d[bd]["cal"]
                row.append("-" if c is None else f"{c[0]:+.2f}..{c[1]:+.2f}")
            print(f"{vid[10:]:24s} {f:4s}", " ".join(x.rjust(15) for x in row))
    print("\nINTEGRAL DEV % (cal range)")
    for vid, s in S.items():
        for f, d in s["integral_dev_pct"].items():
            print(f"{vid[10:]:24s} {f:3s}", " ".join(
                f"{a}:{d[a]['cal'][0]:+.2f}..{d[a]['cal'][1]:+.2f}|{d[a]['top']:+.2f}".rjust(28) for a in ANN))
    print("\nOWNER SPLIT integral share % (cal range | top)")
    for f, d in own.items():
        for c, e in d.items():
            print(f, c.ljust(22), " ".join(f"{a}:{e[a]['cal'][0]:.1f}-{e[a]['cal'][1]:.1f}|{e[a]['top']:.1f}"
                                         for a in ANN))
    for name in ("agn_share_of_total", "cold_sf_share_of_total"):
        print("\n", name)
        for f, d in summary[name].items():
            print(f, {a: d["integral_pct"][a] for a in ANN})
            print("   strict bands", {bd: d["band_strict_pct"][bd]["cal"] for bd in BANDS})
