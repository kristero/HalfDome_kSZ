#!/usr/bin/env python3
"""Validate models.py against the Julia/XGPaint tabulation halfdome_sphere_profiles.h5 (HalfDome cosmology),
and check that the enclosed-electron shortcut for the mean DM equals int DM dOmega from the chord table."""
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import interpolate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "outputs" / "analytic_halo_pdf_20260925" / "tables" / "halfdome_sphere_profiles.h5"
OUT = ROOT / "outputs" / "lee22_normalization_forensics_20260928" / "A_tng_redshift_test"
OUT.mkdir(parents=True, exist_ok=True)

cosmo = M.HALFDOME_XGP
rng = np.random.default_rng(1)
lines = []
with h5py.File(TABLE, "r") as f:
    lm = f["log10_m200c_msun"][:]; zs = f["redshift"][:]; xs = f["x_r200c"][:]
    for model, key in (("b16", "b16"), ("lee22_xgpnative", "lee22_noconc_xgpnative")):
        g = f[key + "/chord_mean_dm_pc_cm3"]        # h5py shape (nm, nz, nx)
        th = f[key + "/theta200c_rad"][:]           # (nm, nz)
        r2 = f[key + "/r200c_mpc"][:]
        worst_g = worst_t = worst_r = 0.0
        nodes = [(0, 0, 0), (699, 99, 199), (400, 50, 100), (460, 10, 5), (300, 90, 150)]
        nodes += [tuple(int(v) for v in (rng.integers(0, 700), rng.integers(0, 100), rng.integers(0, 200))) for _ in range(25)]
        for (j, i, k) in nodes:
            mass = 10 ** lm[j]; z = zs[i]; x = xs[k]
            ours = M.chord_mean_dm(model, cosmo, x, mass, z)
            ref = g[j, i, k]
            worst_g = max(worst_g, abs(ours / ref - 1))
            r200 = cosmo.r200c_m(mass, z) / M.MPC_M
            worst_r = max(worst_r, abs(r200 / r2[j, i] - 1))
            theta = np.arctan(r200 / (cosmo.chi_mpc(z) / (1 + z)))
            worst_t = max(worst_t, abs(theta / th[j, i] - 1))
        lines.append("{:16s} max |rel diff| over {} nodes: g {:.2e}  R200c {:.2e}  theta200c {:.2e}".format(
            model, len(nodes), worst_g, worst_r, worst_t))

        # enclosed-electron shortcut: pi theta^2 int_0^1 DM(x) 2x dx  ==  N_e (1+z) / chi^2  (in pc cm^-3 sr)
        worst_n = 0.0
        for (j, i) in ((300, 20), (450, 50), (500, 90), (600, 5), (380, 70)):
            mass = 10 ** lm[j]; z = zs[i]
            gg = g[j, i, :]
            spl = interpolate.CubicSpline(np.log(xs), np.log(gg))
            xx = np.concatenate([np.linspace(0, 1e-4, 50)[:-1], np.logspace(-4, np.log10(0.9995), 20000),
                                 1 - np.logspace(np.log10(5e-4), -12, 4000)])
            gx = np.exp(spl(np.log(np.clip(xx, xs[0], xs[-1]))))
            dm = 2 * np.sqrt(np.clip(1 - xx ** 2, 0, None)) * gx
            disc = np.trapz(dm * 2 * xx, xx) * np.pi * th[j, i] ** 2
            ne = M.electrons_in_r200(model, cosmo, mass, z)
            chi_m = cosmo.chi_mpc(z) * M.MPC_M
            shortcut = ne * (1 + z) / chi_m ** 2 * M.M2_TO_PC_CM3
            worst_n = max(worst_n, abs(disc / shortcut - 1))
        lines.append("{:16s} enclosed-electron shortcut vs chord-table disc integral: max |rel diff| {:.2e}".format(model, worst_n))

# literal / xgpnative factor
lines.append("xgpnative factor 0.9 X_H m_p / m_per_e = {:.13f} (repo: 0.6016316602183194)".format(M.XGP_NATIVE_FACTOR))
txt = "\n".join(lines)
print(txt)
(OUT / "validation_models_vs_julia_table.txt").write_text(txt + "\n")
