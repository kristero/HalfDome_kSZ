#!/usr/bin/env python3
"""verify_0 / step 1: parse Lee22 Table 8 (no-c density) and Table 3 (with c) directly from the arXiv TeX
and write a grid of per-halo parameters (n0, x'_c, beta') for the Julia XGPaint test.

Independent of the repository's lee2022_frb_dm_profile.jl: coefficients are read from the TeX with a regex,
not typed by hand. Masses in the grid are PHYSICAL Msun; the paper's h^-1 Msun pivot is converted with the
h of the cosmology under test.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/home/cbllover/HalfDome/frb_map_generation")
TEX = ROOT / "outputs/lee22_fig5_audit_20260922/reference/source/sample631.tex"
OUT = ROOT / "outputs/lee22_normalization_forensics_20260928/verify_0"
OUT.mkdir(parents=True, exist_ok=True)


def parse_table(label):
    tex = TEX.read_text()
    i = tex.index(r"\label{" + label + "}")
    j = tex.rfind(r"\begin{table", 0, i)
    block = tex[j:i]
    rows = {}
    for line in block.splitlines():
        m = re.match(r"\s*\$(n_0|x'_c|\\beta'|P_0|x_c|\\beta)\$\s*&(.*)", line)
        if not m:
            continue
        name = m.group(1).replace("\\", "")
        cells = m.group(2).split("&")
        vals = []
        for c in cells:
            mm = re.search(r"\$\s*(-?[0-9.]+)_", c)
            if mm:
                vals.append(float(mm.group(1)))
            elif "--" in c:
                vals.append(None)
        rows[name] = vals
    mc = re.search(r"multirow\{3\}\{\*\}\{\$([0-9.]+)_", block)
    rows["log10Mcut_hinv"] = float(mc.group(1))
    return rows


t8 = parse_table("table:8")
t3 = parse_table("table:3")
print("Table 8 (parsed from TeX):", t8)
print("Table 3 (parsed from TeX):", t3)


def bpl(r, a, ap):
    ap = a if ap is None else ap
    return np.where(r < 1, r ** a, r ** ap)


def noc_params(M_msun, z, h, pivot="mcut"):
    """Table 8 with eq. 11 (extended form, shared M_cut). pivot='1e14h' uses 1e14 h^-1 Msun for n0,
    pivot='1e14' uses 1e14 physical Msun for n0 (the repository's :legacy_1e14)."""
    A0, am, amp, az = t8["n_0"][:4]
    Mcut = 10 ** t8["log10Mcut_hinv"] / h  # physical Msun
    r = M_msun / Mcut
    if pivot == "mcut":
        rn = r
    elif pivot == "1e14h":
        rn = M_msun / (1e14 / h)
    elif pivot == "1e14":
        rn = M_msun / 1e14
    n0 = A0 * rn ** am * (1 + z) ** az
    A0x, amx, ampx, azx = t8["x'_c"]
    xc = A0x * bpl(r, amx, ampx) * (1 + z) ** azx
    A0b, amb, ampb, azb = t8["beta'"]
    be = A0b * bpl(r, amb, ampb) * (1 + z) ** azb
    return n0, xc, be


if __name__ == "__main__":
    cosmologies = {
        "halfdome": dict(Omega_c=0.261, Omega_b=0.049, h=0.68),
        "tng": dict(Omega_c=0.3089 - 0.0486, Omega_b=0.0486, h=0.6774),
    }
    masses = [3e12, 1e13, 3e13, 1e14, 3e14, 1e15, 3e15]
    zs = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for cname, c in cosmologies.items():
        for M in masses:
            for z in zs:
                n0, xc, be = noc_params(M, z, c["h"])
                rows.append((cname, c["Omega_c"], c["Omega_b"], c["h"], M, z, float(n0), float(xc), float(be)))
    with open(OUT / "step1_param_grid.csv", "w") as f:
        f.write("cosmo,Omega_c,Omega_b,h,M_msun,z,n0,xc,beta_prime\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    json.dump({"table8": t8, "table3": t3}, open(OUT / "step1_tables_parsed_from_tex.json", "w"), indent=1)
    print(f"wrote {len(rows)} rows")
