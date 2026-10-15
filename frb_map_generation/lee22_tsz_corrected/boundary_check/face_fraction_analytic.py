#!/usr/bin/env python
"""Analytic fraction of halos whose x R200c sphere crosses a face of a periodic cube (uniform halo positions):
f = 1 - (1 - 2 x R200c,com / L)^3.  TNG300-1 (L = 205 Mpc/h, Lee22) versus CAMELS (L = 25 Mpc/h, our checks),
masses in h^-1 Msun as in Lee22 (10^13 - 10^14.2), z = 0, 1, 2.  Output: OUT/face_fraction_analytic.txt"""
import os

import numpy as np

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007"
RHO_C0 = 2.775e11          # h^2 Msun / Mpc^3


def r200c_com(m_hinv, z, om):
    rc = RHO_C0 * (om * (1 + z) ** 3 + 1 - om)
    return (3 * m_hinv / (4 * np.pi * 200 * rc)) ** (1 / 3) * (1 + z)     # comoving h^-1 Mpc


lines = ["f = fraction of halos whose x R200c sphere crosses a box face (uniform positions)",
         "box               logM[h^-1 Msun]  z   R200c,com[h^-1 Mpc]  f(x=1)  f(x=1.34)  f(x=2.5)"]
for name, L, om in (("TNG300-1 205 Mpc/h", 205.0, 0.3089), ("CAMELS 25 Mpc/h", 25.0, 0.3)):
    for lm in (13.0, 13.5, 14.0, 14.2):
        for z in (0.0, 1.0, 2.0):
            R = r200c_com(10 ** lm, z, om)
            f = [1 - max(1 - 2 * x * R / L, 0) ** 3 for x in (1.0, 1.34, 2.5)]
            lines.append(f"{name:18s} {lm:5.1f}           {z:3.1f}  {R:6.3f}               " +
                         "  ".join(f"{v:.4f}" for v in f))
print("\n".join(lines))
with open(os.path.join(OUT, "face_fraction_analytic.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
