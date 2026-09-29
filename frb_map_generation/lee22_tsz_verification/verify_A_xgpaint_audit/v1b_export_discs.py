"""Verifier A: re-create the v1 discs (same seed) and export pixels + candidate halos for the Julia decomposition
(v1c_julia_decompose.jl), plus a per-halo breakdown of my own y in the two discrepant discs."""
import json, sys
import numpy as np, h5py, healpy as hp
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_A_xgpaint_audit")
import vlib
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
NSIDE = 4096; DISC = np.deg2rad(0.1)
with h5py.File("/home/cbllover/HalfDome/lightcone_100.hdf5", "r") as f:
    z = f["redshift"][:].astype(np.float64); m = f["halo_mass_m200c"][:] / 0.68; pos = f["Position"][:]
nrm = np.sqrt((pos.astype(np.float64) ** 2).sum(1))
cosL = vlib.Cos(neff=3.046)
thmax_all = np.arctan(4 * cosL.r200_m(m, z) / vlib.MPC / cosL.dA_mpc(z))
rng = np.random.default_rng(20260929)
def pick(lo, hi, zlo, zhi):
    idx = np.flatnonzero((m >= 10 ** lo) & (m < 10 ** hi) & (z >= zlo) & (z < zhi)); return int(rng.choice(idx))
centres = {}
for name, args in {"M14p5_z0p6": (14.4, 14.6, 0.55, 0.65), "M13p5_z1p8": (13.4, 13.6, 1.75, 1.85),
                   "M15_z0p15_shapeclip": (14.9, 15.1, 0.1, 0.2)}.items():
    i = pick(*args); centres[name] = pos[i].astype(np.float64) / nrm[i]
v = rng.normal(size=3); centres["random"] = v / np.linalg.norm(v)
with open(f"{O}/v1b_disc_pixels.csv", "w") as fp, open(f"{O}/v1b_disc_halos.csv", "w") as fh:
    fp.write("disc,pix_ring0\n"); fh.write("disc,halo_index0,m200c_msun,z,px,py,pz,thmax_sel\n")
    for cname, c in centres.items():
        pix = hp.query_disc(NSIDE, c, DISC)
        for p in pix: fp.write(f"{cname},{p}\n")
        cosd = (pos[:, 0] * c[0] + pos[:, 1] * c[1] + pos[:, 2] * c[2]) / nrm
        sep = np.arccos(np.clip(cosd, -1, 1))
        cand = np.flatnonzero(sep < DISC + thmax_all * (1 + 1e-4) + 1e-6)
        for i in cand:
            fh.write(f"{cname},{i},{m[i]!r},{z[i]!r},{float(pos[i,0])!r},{float(pos[i,1])!r},{float(pos[i,2])!r},{thmax_all[i]!r}\n")
        lowz = cand[z[cand] < 0.05]
        print(cname, "candidates", len(cand), "z<0.05:", [(int(i), round(float(np.log10(m[i])), 3), round(float(z[i]), 5),
              round(float(np.degrees(sep[i])), 3), round(float(np.degrees(thmax_all[i])), 3)) for i in lowz])
        big = cand[m[cand] > 10 ** 14.9]
        print("   M>10^14.9:", [(int(i), round(float(np.log10(m[i])), 4), round(float(z[i]), 4)) for i in big])
