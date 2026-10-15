#!/usr/bin/env python
"""Periodic-boundary check of OUR FLAMINGO L2p8_m9 stacks (stack_products.py, fullsnap_products.py and the
Lee22-workstream density_products.py, all streaming particles cell by cell with minimum-image separations).

For every stored halo: does its particle sphere (R100c for the reduced snapshots; r_sphere for the full-snapshot
products; R200c for the y/N(<R200c) checks) cross a face of the 2.8 Gpc box, and do those halos pass the
end-to-end comparison with SOAP (which is periodic by construction)?  A missing wrap would show up as a particle
deficit up to ~50 % for such halos.  Also the fraction of halos that would have been affected at all.

Output: OUT/flamingo_face_check.json, OUT/flamingo_face_check.txt
"""
import glob
import json
import os

import numpy as np

REF = "/home/cbllover/HalfDome/flamingo_pressure_calibration/fullsnap_calibration/reference"
DENS = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/C_flamingo/idark_density/products"
SELS = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/C_flamingo/idark_input/selections"
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007"
BOX = 2800.0      # comoving Mpc (L2p8_m9; checked against the stored centres below)
lines, res = [], {}


def say(s=""):
    print(s, flush=True)
    lines.append(s)


for fn in sorted(glob.glob(os.path.join(REF, "L2p8_m9_*_products.npz"))):
    d = np.load(fn)
    a = float(d["scale_factor"])
    c = d["cat_centre_mpc"] / a                                       # comoving Mpc
    dmin = np.min(np.minimum(c, BOX - c), axis=1)
    full = "fullsnap" in os.path.basename(fn)
    if full:   # full-snapshot products: sphere of sphere_r200 x R200c (receipt), stored as cat_r_sphere_mpc when present
        sph = json.load(open(fn.replace(".npz", ".json")))["sphere_r200"]
        r_part = (d["cat_r_sphere_mpc"] if "cat_r_sphere_mpc" in d.files else sph * d["cat_r_200_crit_mpc"]) / a
    else:
        r_part = d["cat_r_100_crit_mpc"] / a
    r200 = d["cat_r_200_crit_mpc"] / a
    cross = dmin < r_part
    cross200 = dmin < r200
    chk = {k: d[k] for k in d.files if k.startswith("check_")}
    name = os.path.basename(fn).replace(".npz", "")
    rec = dict(z=float(d["redshift"]), n_haloes=int(len(c)), centre_range_cmpc=[float(c.min()), float(c.max())],
               n_sphere_crossing_face=int(cross.sum()), n_R200c_crossing_face=int(cross200.sum()),
               frac_sphere_crossing=float(cross.mean()))
    for k, v in chk.items():
        rec[k + "_max_abs_all"] = float(np.nanmax(np.abs(v)))
        rec[k + "_crossing_haloes"] = [float(x) for x in v[cross]]
    res[name] = rec
    say(f"{name}: z={rec['z']:.2f} {rec['n_haloes']} haloes, centres in [{c.min():.1f}, {c.max():.1f}] cMpc; "
        f"sphere crosses a face: {cross.sum()} ({100 * cross.mean():.2f} %), R200c crosses: {cross200.sum()}")
    if cross200.any():
        t = dmin[cross200] / r200[cross200]
        cap = (1 - t) ** 2 * (2 + t) / 4          # volume fraction of the R200c sphere beyond the nearest face
        rec["R200c_crossing_dmin_over_R200c"] = t.tolist()
        rec["R200c_crossing_cap_volume_fraction"] = cap.tolist()
        say("    R200c-crossing haloes: d_face/R200c = " + ", ".join(f"{x:.2f}" for x in t) +
            "; volume fraction of the R200c sphere beyond the face (a no-wrap code would lose this volume) = " +
            ", ".join(f"{x:.3f}" for x in cap))
    for k, v in chk.items():
        say(f"    {k}: max|.| all = {np.nanmax(np.abs(v)):.2e}; face-crossing haloes: " +
            (", ".join(f"{x:+.1e}" for x in v[cross]) if cross.any() else "none"))

say("\nLee22-workstream density products (density_products.py; selections = same haloes as the stacks):")
for fn in sorted(glob.glob(os.path.join(DENS, "L2p8_m9_*_density_products.npz"))):
    d = np.load(fn)
    snap = int(d["snapshot"])
    a = float(d["scale_factor"])
    sel = np.load(os.path.join(SELS, f"selection_{snap:04d}.npz"))
    idx = {int(i): k for k, i in enumerate(sel["halo_index"])}
    ks = np.array([idx[int(i)] for i in d["halo_index"]])
    c = sel["centre_mpc"][ks] / a
    dmin = np.min(np.minimum(c, BOX - c), axis=1)
    cross = dmin < d["r100_mpc"] / a
    # gas mass inside R200c vs SOAP (stored in the selection file when available)
    extra = ""
    for key in ("gas_mass_200_crit_msun", "m_gas_200_crit_msun", "gas_mass_r200_msun"):
        if key in sel.files:
            rel = d["m_r200"] / sel[key][ks] - 1
            extra = f"; M_gas(<R200c)/SOAP - 1 for crossing haloes: " + ", ".join(f"{x:+.1e}" for x in rel[cross])
            break
    res[f"density_{snap:04d}"] = dict(n_haloes=int(len(c)), n_sphere_crossing_face=int(cross.sum()))
    say(f"  snapshot {snap}: {len(c)} haloes, R100c sphere crosses a face: {cross.sum()}{extra}")

with open(os.path.join(OUT, "flamingo_face_check.json"), "w") as f:
    json.dump(res, f, indent=1)
with open(os.path.join(OUT, "flamingo_face_check.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
