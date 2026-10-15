#!/usr/bin/env python
"""Periodically translated copy of a single-file CAMELS snapshot + group catalogue, for group_particles runs.

Only what the Lee22 example programs read is materialised: the proxy snapshot has the original Header attributes,
PartType0/Coordinates = (x + shift) mod L rounded to float32 and STORED AS FLOAT64 (group_particles declares
IllustrisFields::Coordinates as double, as in TNG300 snapshots, and asserts the element size; CAMELS stores float32;
the code casts to coord_t = float on reading, so it sees exactly the float32 values), with shift = 0 the stored
float32 values are kept unchanged (no modulo), and HDF5 external links for PartType0/Masses,
InternalEnergy and ElectronAbundance; the proxy group file has the original Header, Group/GroupPos shifted the same
way and external links for Group_M_Crit200 and Group_R_Crit200.  A periodic translation is the same simulation
with the box faces at other places relative to the halos, so it samples other face placements for the
low-face pair-dropping bug of group_particles c3fe9b8.

Usage: python make_shift_proxy.py SNAP.hdf5 GROUPS.hdf5 OUTDIR SX SY SZ   (shift in ckpc/h)
"""
import os
import sys

import h5py
import numpy as np

snap, groups, outdir = (os.path.abspath(a) for a in sys.argv[1:4])
shift = np.array([float(v) for v in sys.argv[4:7]])
os.makedirs(outdir, exist_ok=True)


def shifted(x, L):
    """float32 positions after the periodic translation (unchanged when shift == 0)."""
    if not np.any(shift):
        return x.astype(np.float32)
    y = np.mod(x.astype(np.float64) + shift[None, :], L)
    return y.astype(np.float32)


with h5py.File(snap, "r") as f:
    L = float(f["Header"].attrs["BoxSize"])
    with h5py.File(os.path.join(outdir, "snapshot_proxy.hdf5"), "w") as o:
        h = o.create_group("Header")
        for k, v in f["Header"].attrs.items():
            h.attrs[k] = v
        h.attrs["ProxyShift"] = shift
        g = o.create_group("PartType0")
        g.create_dataset("Coordinates", data=shifted(f["PartType0/Coordinates"][:], L).astype(np.float64))
        for k in ("Masses", "InternalEnergy", "ElectronAbundance"):
            g[k] = h5py.ExternalLink(snap, "/PartType0/" + k)
with h5py.File(groups, "r") as f:
    with h5py.File(os.path.join(outdir, "groups_proxy.hdf5"), "w") as o:
        h = o.create_group("Header")
        for k, v in f["Header"].attrs.items():
            h.attrs[k] = v
        h.attrs["ProxyShift"] = shift
        g = o.create_group("Group")
        g.create_dataset("GroupPos", data=shifted(f["Group/GroupPos"][:], L))
        for k in ("Group_M_Crit200", "Group_R_Crit200"):
            g[k] = h5py.ExternalLink(groups, "/Group/" + k)
print(f"proxy written to {outdir}, shift {shift.tolist()}, L {L}")
