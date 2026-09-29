"""Write the per-snapshot halo selections (the exact haloes of the preliminary pressure stacks) for density_products.py.

Reads flamingo_pressure_calibration/notebook_cache/stacks/L2p8_m9_NNNN_stack_products.npz (read-only) and writes
<out>/selections/selection_NNNN.npz: halo_index, centre_mpc (physical Mpc), r200_mpc, r100_mpc (physical), m200c_msun,
edges (the stack shells in r/R200c), redshift, scale_factor.
"""
import sys
from pathlib import Path
import numpy as np

STACKS = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache/stacks")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/C_flamingo/idark_input")
(out / "selections").mkdir(parents=True, exist_ok=True)
for snap in [78, 68, 58, 48, 38, 18, 14]:
    d = np.load(STACKS / f"L2p8_m9_{snap:04d}_stack_products.npz")
    np.savez(out / "selections" / f"selection_{snap:04d}.npz", halo_index=d["cat_halo_index"], centre_mpc=d["cat_centre_mpc"],
             r200_mpc=d["cat_r_200_crit_mpc"], r100_mpc=d["cat_r_100_crit_mpc"], m200c_msun=d["cat_m200c_msun"],
             edges=d["edges"], redshift=float(d["redshift"]), scale_factor=float(d["scale_factor"]))
    print(snap, float(d["redshift"]), len(d["cat_halo_index"]))
