"""Verifier A: merge v1c (Julia itp / direct per pixel) with the stored maps and my own Python y (recomputed here)."""
import csv, json
import numpy as np
from astropy.io import fits
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
R = "/home/cbllover/HalfDome/frb_map_generation/outputs"
MAPS = {"b12_allz": f"{R}/tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits",
        "lee22_allz": f"{R}/tsz_dm_cross_updated_20260917/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits",
        "lee22_calib": f"{R}/tsz_dm_fullsky_20260918/maps/lee22_noconc_pressure_compton_y_calib_zmax2p0_nside4096_m200c_r200cx4.fits"}
rows = list(csv.DictReader(open(f"{O}/v1c_disc_itp_direct.csv")))
out = {}
for mname, path in MAPS.items():
    with fits.open(path, memmap=True) as h:
        col = h[1].data["PIXELS"]
        for disc in sorted({r["disc"] for r in rows}):
            R_ = [r for r in rows if r["map"] == mname and r["disc"] == disc]
            pix = np.array([int(r["pix_ring0"]) for r in R_])
            yi = np.array([float(r["y_itp"]) for r in R_]); yd = np.array([float(r["y_direct"]) for r in R_])
            yil = np.array([float(r["y_itp_lowz_part"]) for r in R_]); ydl = np.array([float(r["y_direct_lowz_part"]) for r in R_])
            ym = np.array(col[pix], dtype=np.float64)
            o = dict(map_vs_itp_maxabs=float(np.max(np.abs(ym / yi - 1))),
                     itp_vs_direct_maxabs=float(np.max(np.abs(yi / yd - 1))), itp_vs_direct_median=float(np.median(yi / yd - 1)),
                     lowz_share_of_y_max=float(np.max(ydl / yd)),
                     itp_vs_direct_excl_lowz_maxabs=float(np.max(np.abs((yi - yil) / (yd - ydl) - 1))),
                     lowz_itp_over_direct_range=[float(np.min(yil[ydl > 0] / ydl[ydl > 0])), float(np.max(yil[ydl > 0] / ydl[ydl > 0]))] if (ydl > 0).any() else None)
            out[f"{mname}/{disc}"] = o
            print(mname, disc, json.dumps(o))
json.dump(out, open(f"{O}/v1d_merge.json", "w"), indent=1)
