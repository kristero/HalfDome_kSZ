#!/usr/bin/env python
"""Raw per-halo shell sums split by ownership class and gas phase (group 'estimators').

Streams STUDY/measurements_owner.npz['owner200'] ([4588, 17 owners, 3 phases, 16 quantities, 67 x200 shells],
2.0 GB as float64) one block of haloes at a time straight from the zip member (peak RAM ~0.3 GB) and keeps, for the
recently-AGN-removed gas (phases xray_selected and other), the fit-grid (35-shell) sums of

    count, mass, volume, y, electrons, electron_kT, mass_pe, mass_rho, lx_rosat, lx_t

for 2 ownership classes x 2 phases:
    owner class 0 'smooth': gas bound to the central subhalo (owner 0) or unbound (owner 16)
    owner class 1 'sub'   : gas bound to any other subhalo (owners 1-15: own satellites and neighbours)
    phase 0 'hot'  : X-ray phase (SFR <= 0, T >= 1e5 K, not recently AGN-heated)
    phase 1 'cold' : star-forming or T < 1e5 K, not recently AGN-heated

These sums are only used to explain the mass-weighted (Q_M) and X-ray-weighted (T_X) estimators; the stacks
themselves come from the core cache.  Output: owner_sums.npz (float64, [4588, 2, 2, 10, 35]).

Run:  LD_LIBRARY_PATH=<env>/lib <env>/bin/python -B owner_sums.py
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

STUDY = OUT.parent
SRC = STUDY / "measurements_owner.npz"
QUANTITIES = ('count', 'mass', 'volume', 'y', 'electrons', 'electron_kT', 'gas_pressure_volume',
              'mass_pe', 'volume_pe2', 'mass_t', 'volume_t', 'mass_rho', 'lx_rosat', 'lx_pe', 'lx_t', 'y2')
KEEP = ('count', 'mass', 'volume', 'y', 'electrons', 'electron_kT', 'mass_pe', 'mass_rho', 'lx_rosat', 'lx_t')
QI = [QUANTITIES.index(q) for q in KEEP]
SMOOTH_OWNERS = [0, 16]
SUB_OWNERS = list(range(1, 16))
PHASES = (1, 2)                      # xray_selected, other  (0 = recent AGN, excluded)


def main():
    t0 = time.time()
    ex, ef = sc.edges("x200"), sc.edges("fit")
    idx = np.array([int(np.argmin(np.abs(ex - e))) for e in ef])
    assert np.max(np.abs(ex[idx] - ef)) == 0.0
    with zipfile.ZipFile(SRC) as z, z.open("owner200.npy") as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
        assert shape[1:] == (17, 3, 16, len(ex) - 1) and not fortran, shape
        n = shape[0]
        row = int(np.prod(shape[1:]))
        out = np.zeros((n, 2, 2, len(KEEP), len(ef) - 1))
        chunk = 128
        for start in range(0, n, chunk):
            k = min(chunk, n - start)
            buf = fh.read(k * row * dtype.itemsize)
            a = np.frombuffer(buf, dtype=dtype).reshape((k,) + shape[1:])
            a = a[:, :, :, QI, :]                                           # [k, 17, 3, Q, 67]
            for oc, owners in enumerate((SMOOTH_OWNERS, SUB_OWNERS)):
                for pc, ph in enumerate(PHASES):
                    s = a[:, owners, ph].sum(axis=1)                         # [k, Q, 67]
                    c = np.concatenate([np.zeros(s.shape[:-1] + (1,)), np.cumsum(s, axis=-1)], axis=-1)
                    out[start:start + k, oc, pc] = c[..., idx[1:]] - c[..., idx[:-1]]
    np.savez_compressed(HERE / "owner_sums.npz", sums=out, quantities=np.array(KEEP),
                        owner_classes=np.array(["smooth (central-bound + unbound)", "sub (bound to other subhaloes)"]),
                        phases=np.array(["hot (X-ray phase)", "cold (SF or T < 1e5 K)"]), edges=ef)
    print(f"owner_sums.npz written ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
