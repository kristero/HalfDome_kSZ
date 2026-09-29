"""Verifier A: independent (Python, no Julia) check of the stored production caches on native grid nodes:
prof_y[logM_k, z_j, logtheta_i] vs my direct LOS y (vlib) with XGPaint-like radiation cosmology. Only nodes with x = theta/theta200 <= 4."""
import json, sys
import numpy as np, h5py
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_A_xgpaint_audit")
import vlib
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
C = {"b12": ("/home/cbllover/HalfDome/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2", vlib.Cos(neff=3.04)),
     "lee22": ("/home/cbllover/HalfDome/frb_map_generation/outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2", vlib.Cos(neff=3.046))}
lin = lambda r: np.linspace(r["start"], r["stop"], int(r["len"]))
out = {}
rng = np.random.default_rng(7)
for mod, (path, cos) in C.items():
    with h5py.File(path, "r") as f:
        lt, zz, lm = lin(f["prof_logthetas"][()]), lin(f["prof_redshift"][()]), lin(f["prof_logMs"][()])
        Y = f["prof_y"]
        worst = 0.0; rows = []
        for j in list(rng.choice(len(zz), 12, replace=False)) + [0, 1, 5]:
            for k in list(rng.choice(len(lm), 4, replace=False)):
                m, z = 10 ** lm[k], zz[j]
                th = np.exp(lt)
                r200 = cos.r200_m(m, z) / vlib.MPC; th200 = np.arctan(r200 / cos.dA_mpc(z))
                X = th / th200
                s = (X <= 4) & (X >= 1e-4)
                if not s.any():
                    continue
                y = vlib.y_of_X(mod, m, z, X[s], cos)
                ys = np.asarray(Y[k, j, :])[s]
                rel = float(np.max(np.abs(ys / y - 1)))
                worst = max(worst, rel); rows.append((round(float(z), 4), round(float(lm[k]), 3), int(s.sum()), rel))
        out[mod] = dict(grid=dict(logtheta=[float(lt[0]), float(lt[-1]), len(lt)], z=[float(zz[0]), float(zz[-1]), len(zz)],
                                  log10M=[float(lm[0]), float(lm[-1]), len(lm)]), max_rel_node_vs_mine=worst, nodes=rows)
        print(mod, out[mod]["grid"], "max rel", worst)
json.dump(out, open(f"{O}/v7_cache_nodes_python.json", "w"), indent=1)
