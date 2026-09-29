"""Verifier A, check 1: end-to-end test of the three STORED production y maps against my own Python y(theta) (vlib.py),
with no Julia and no interpolator cache. Discs chosen independently of the workstream (different halos / directions).
Painter conventions re-implemented from paint_halfdome_battaglia12_tsz_map.jl + paint_batch_external_r200c!:
  M = halo_mass_m200c / 0.68 (physical Msun), z = catalogue redshift (float32 -> float64), direction = Position/|Position|,
  theta_max = atan(4 R200c / d_A), pixel centres (RING) with theta < theta_max, y(max(theta, theta_min~0)).
Output: v1_pixel_check.csv / .json in the verifier output dir."""
import json, sys, time
import numpy as np, h5py, healpy as hp
from astropy.io import fits
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_A_xgpaint_audit")
import vlib

O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
R = "/home/cbllover/HalfDome/frb_map_generation/outputs"
MAPS = {
    "b12_allz": (f"{R}/tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits", "b12", None),
    "lee22_allz": (f"{R}/tsz_dm_cross_updated_20260917/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits", "lee22", None),
    "lee22_calib": (f"{R}/tsz_dm_fullsky_20260918/maps/lee22_noconc_pressure_compton_y_calib_zmax2p0_nside4096_m200c_r200cx4.fits", "lee22",
                    (1.4705882352941176e13, 9.278784477649915e14, 2.0)),
}
NSIDE = 4096; DISC = np.deg2rad(0.1)
t0 = time.time()
with h5py.File("/home/cbllover/HalfDome/lightcone_100.hdf5", "r") as f:
    z = f["redshift"][:].astype(np.float64)
    m = f["halo_mass_m200c"][:] / 0.68
    pos = f["Position"][:]
print("loaded", time.time() - t0, pos.shape, pos.dtype)
pos64norm = np.sqrt((pos.astype(np.float64) ** 2).sum(1))
cosB = vlib.Cos(neff=3.04); cosL = vlib.Cos(neff=3.046); cos0 = vlib.Cos(rad=False)
thmax_all = np.arctan(4 * cosL.r200_m(m, z) / vlib.MPC / cosL.dA_mpc(z))   # selection only (margin added below)

rng = np.random.default_rng(20260929)
def pick(lo, hi, zlo, zhi):
    idx = np.flatnonzero((m >= 10 ** lo) & (m < 10 ** hi) & (z >= zlo) & (z < zhi))
    return int(rng.choice(idx))
centres = {}
for name, args in {"M14p5_z0p6": (14.4, 14.6, 0.55, 0.65), "M13p5_z1p8": (13.4, 13.6, 1.75, 1.85),
                   "M15_z0p15_shapeclip": (14.9, 15.1, 0.1, 0.2)}.items():
    i = pick(*args); centres[name] = (pos[i].astype(np.float64) / pos64norm[i], dict(halo=i, log10M=float(np.log10(m[i])), z=float(z[i])))
v = rng.normal(size=3); centres["random"] = (v / np.linalg.norm(v), {})

def fits_pixels(path, pix):
    with fits.open(path, memmap=True) as h:
        col = h[1].data["PIXELS"]
        return np.array(col[pix], dtype=np.float64)

rows, summ = [], {}
for cname, (cvec, info) in centres.items():
    pix = hp.query_disc(NSIDE, cvec, DISC)            # RING, pixel centres within DISC
    pvec = np.array(hp.pix2vec(NSIDE, pix)).T          # (npix, 3)
    cosd = (pos[:, 0] * cvec[0] + pos[:, 1] * cvec[1] + pos[:, 2] * cvec[2]) / pos64norm
    sep = np.arccos(np.clip(cosd, -1, 1))
    cand = np.flatnonzero(sep < DISC + thmax_all * (1 + 1e-4) + 1e-6)
    print(cname, info, "npix", len(pix), "candidates", len(cand))
    for mapname, (path, mod, sel) in MAPS.items():
        c = cosB if mod == "b12" else cosL
        keep = cand
        if sel is not None:
            keep = cand[(m[cand] >= sel[0]) & (m[cand] < sel[1]) & (z[cand] <= sel[2])]
        ytot = np.zeros(len(pix)); ytot0 = np.zeros(len(pix)); nh = 0; edge = np.zeros(len(pix), bool)
        for i in keep:
            u = pos[i].astype(np.float64) / pos64norm[i]
            th = np.arccos(np.clip(pvec @ u, -1, 1))
            r200 = c.r200_m(m[i], z[i]) / vlib.MPC; dA = c.dA_mpc(z[i])
            tmax = min(np.arctan(4 * r200 / dA), np.pi)
            edge |= np.abs(th / tmax - 1) < 1e-7
            sel_p = th < tmax
            if not sel_p.any():
                continue
            nh += 1
            th200 = np.arctan(r200 / dA)
            ytot[sel_p] += vlib.y_of_X(mod, m[i], z[i], th[sel_p] / th200, c)
            # literal (no radiation) variant for the rho_crit convention offset
            r0 = cos0.r200_m(m[i], z[i]) / vlib.MPC; d0 = cos0.dA_mpc(z[i])
            ytot0[sel_p] += vlib.y_of_X(mod, m[i], z[i], th[sel_p] / np.arctan(r0 / d0), cos0)
        ymap = fits_pixels(path, pix)
        rel = ymap / ytot - 1; rel0 = ymap / ytot0 - 1
        ok = ~edge
        s = dict(disc=cname, map=mapname, n_pix=int(len(pix)), n_candidates=int(len(keep)), n_contributing=int(nh),
                 n_edge_pixels=int(edge.sum()), median_y_map=float(np.median(ymap)),
                 max_abs_rel_vs_mine=float(np.max(np.abs(rel[ok]))), rms_rel_vs_mine=float(np.sqrt(np.mean(rel[ok] ** 2))),
                 median_rel_vs_mine=float(np.median(rel[ok])), median_rel_vs_mine_norad=float(np.median(rel0[ok])),
                 max_abs_rel_vs_mine_norad=float(np.max(np.abs(rel0[ok]))), **{f"centre_{k}": val for k, val in info.items()})
        print("  ", json.dumps(s))
        rows.append(s)
    # cross-check the calibration-range selection: Lee22 all-z minus calib should equal the out-of-range halos' y
import csv
with open(f"{O}/v1_pixel_check.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)
json.dump(dict(rows=rows, elapsed_s=time.time() - t0), open(f"{O}/v1_pixel_check.json", "w"), indent=1)
print("done", time.time() - t0)
