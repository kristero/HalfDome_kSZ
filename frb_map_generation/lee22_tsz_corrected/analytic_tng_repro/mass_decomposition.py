#!/usr/bin/env python3
"""Mass-decade decomposition of Konietzka's TNG300 within-R200c halo DM against the analytic halo model (means and
hit fractions only; no FFT).

TNG decades are differences of nested windows, valid under both the nominal and the reinterpreted (upper edge
x C_EXT) reading because the lower edges are shared:
  [1e9,1e10] = 9to14 - 10to14, [1e10,1e11] = 10to14 - 11to14, [1e11,1e12] = 11to14 - 12to14,
  [1e12,1e13] = 12to14 - 13to14, [1e13, 1e14 or c 1e14] = 13to14, [1e14 or c 1e14, 1e15 or c 1e15] = 10to15 - 10to14.
For each decade: TNG mean DM over all rays and fraction of rays with a non-zero difference (> 0.01 pc cm^-3),
model mean over all rays and unclustered hit fraction 1 - exp(-Nbar) with Tinker08, and the effective abundance
A_eff = -ln(1 - hit_TNG) / Nbar_Tinker08 (the hit fraction is purely geometric: sum of n pi R200c^2).
Output: analysis/mass_decomposition.json, analysis/mass_decomposition.csv."""
import csv
import json

import numpy as np

import repro_lib as R

ZS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
DECADES = {
    "1e9-1e10": ((1e9, 1e10), ("m1e9_to_1e14", "m1e10_to_1e14")),
    "1e10-1e11": ((1e10, 1e11), ("m1e10_to_1e14", "m1e11_to_1e14")),
    "1e11-1e12": ((1e11, 1e12), ("m1e11_to_1e14", "m1e12_to_1e14")),
    "1e12-1e13": ((1e12, 1e13), ("m1e12_to_1e14", "m1e13_to_1e14")),
    "1e13-1e14 nominal": ((1e13, 1e14), ("m1e13_to_1e14", None)),
    "1e13-9e14 reinterpreted": ((1e13, R.C_EXT * 1e14), ("m1e13_to_1e14", None)),
    "1e14-1e15 nominal": ((1e14, 1e15), ("m1e10_to_1e15", "m1e10_to_1e14")),
    "9e14-9e15 reinterpreted": ((R.C_EXT * 1e14, R.C_EXT * 1e15), ("m1e10_to_1e15", "m1e10_to_1e14")),
    "all (>=1e9)": ((1e9, 1e16), ("all", None)),
}


def tng_decade(a, b, z_s):
    v = R.tng(a, z_s) - (R.tng(b, z_s) if b else 0.0)
    return float(v.mean()), float(np.mean(v > 0.01))


def main():
    tables = {m: R.load_table(m) for m in R.MODELS}
    out = {"z_sources": ZS, "decades": {}}
    rows = []
    for name, (win, (a, b)) in DECADES.items():
        d = {"window_msun": win, "tng": {}, "model": {}}
        for z_s in ZS:
            d["tng"][str(z_s)] = dict(zip(("mean_all", "hit"), tng_decade(a, b, z_s)))
        for abund in ("tinker08", "tng300"):
            for m, t in tables.items():
                mom = R.moments(t, win, ZS, abund)
                d["model"][f"{m}|{abund}"] = {str(z): dict(mean_all=mom[z][0], nbar=mom[z][1], hit=1 - np.exp(-mom[z][1])) for z in ZS}
        nb = {z: d["model"]["b16|tinker08"][str(z)]["nbar"] for z in ZS}
        d["A_eff_from_hits"] = {str(z): (float(-np.log(1 - d["tng"][str(z)]["hit"]) / nb[z]) if 0 < d["tng"][str(z)]["hit"] < 1 else None)
                                for z in ZS}
        out["decades"][name] = d
        for z_s in ZS:
            row = dict(decade=name, z_s=z_s, tng_mean=d["tng"][str(z_s)]["mean_all"], tng_hit=d["tng"][str(z_s)]["hit"],
                       tinker_hit=d["model"]["b16|tinker08"][str(z_s)]["hit"], A_eff=d["A_eff_from_hits"][str(z_s)])
            for m in R.MODELS:
                row[f"mean_{m}_tinker"] = d["model"][f"{m}|tinker08"][str(z_s)]["mean_all"]
                row[f"mean_{m}_tng300"] = d["model"][f"{m}|tng300"][str(z_s)]["mean_all"]
            rows.append(row)
    (R.OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (R.OUT / "analysis" / "mass_decomposition.json").write_text(json.dumps(out, indent=1))
    with open(R.OUT / "analysis" / "mass_decomposition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    for name, d in out["decades"].items():
        print(name)
        for z_s in ZS:
            k = str(z_s)
            ms = "  ".join(f"{m[:10]} {d['model'][m + '|tinker08'][k]['mean_all']:7.2f}" for m in R.MODELS)
            print(f"  z_s {z_s}: TNG mean {d['tng'][k]['mean_all']:7.2f} hit {d['tng'][k]['hit']:.4f} | Tinker hit "
                  f"{d['model']['b16|tinker08'][k]['hit']:.4f} A_eff {fmt(d['A_eff_from_hits'][k])} | {ms}")


if __name__ == "__main__":
    main()
