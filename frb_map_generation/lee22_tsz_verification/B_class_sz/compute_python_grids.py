#!/usr/bin/env python3
"""Tabulate y_ell(z, M) for one profile model and transform on either
   grid 'hm'  : the halo-model Gauss-Legendre nodes (halo_model.z_nodes / lnm_nodes over the catalogue range; node
                panels break exactly at z = 0.1, 0.25, 0.5, 1, 2 and M = 1e13, 10^13.6, 3e13, 1e14, 3e14, 10^14.8 Msun/h,
                so any of those sub-ranges is an exact subset), or
   grid 'cat' : the centres of the catalogue histogram bins (catalogue_hist.py) for the catalogue-weighted terms.
Usage: compute_python_grids.py MODEL KIND GRID   (MODEL b12|lee22|lee22_noclip, KIND sphere|aperture)
Output: OUT/halo_model/ygrid_<GRID>_<MODEL>_<KIND>.npz"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import halo_model as HM  # noqa: E402

if __name__ == "__main__":
    model, kind, grid = sys.argv[1:4]
    outd = Path(C.OUT) / "halo_model"; outd.mkdir(parents=True, exist_ok=True)
    ing = HM.Ingredients()
    if grid == "hm":
        z, wz = HM.z_nodes(C.CAT_Z_MIN, C.CAT_Z_MAX, 10)
        lnm, wm = HM.lnm_nodes(C.CAT_M_MIN, C.CAT_M_MAX, HM.BREAKS, 10)
    elif grid == "cat":
        d = np.load(Path(C.OUT) / "catalogue" / "catalogue_hist.npz")
        ze, me = d["z_edges"], d["lnm_edges"]
        z = 0.5 * (ze[1:] + ze[:-1]); lnm = 0.5 * (me[1:] + me[:-1])
        z = np.where(z < 0.1, np.sqrt(ze[1:] * ze[:-1]), z)       # geometric centres in the log-spaced part
        wz = wm = None
    else:
        raise ValueError(grid)
    t0 = time.time()
    y = HM.y_ell_grid(ing, model, z, lnm, ing.ell, kind=kind, X=4.0)
    out = dict(ell=ing.ell, z=z, lnm=lnm, y=y, model=model, kind=kind, grid=grid, seconds=time.time() - t0)
    if wz is not None:
        out.update(wz=wz, wm=wm)
    np.savez(outd / f"ygrid_{grid}_{model}_{kind}.npz", **out)
    print(f"{grid} {model} {kind}: {y.shape} in {time.time() - t0:.0f} s", flush=True)
