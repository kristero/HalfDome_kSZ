#!/usr/bin/env python
"""Direct (ground-truth) gas, electron-density and electron-pressure profiles of CAMELS IllustrisTNG halos
computed from a raw snapshot + SUBFIND group catalogue.  Runs on idark (python3.8, numpy, scipy, h5py).

Three profile families are computed from the SAME gas cells around each halo centre (GroupPos, periodic box):

 (A) 'ill'  : exact re-implementation of illstack_CAMELS (Moser+22; the CAMELS precomputed Profiles files):
              25 log bins between lims = [0.1, 1e5] ckpc/h (comoving, fixed, not scaled with R200 or 1+z),
              Profiles[0] = sum(Masses)/V_shell [1e10 Msun/h / (ckpc/h)^3],
              Profiles[1] = sum((gamma-1) Masses InternalEnergy)/V_shell [1e10 Msun/h (km/s)^2 / (ckpc/h)^3]
                            (THERMAL pressure of all gas, incl. star-forming cells),
              Profiles[3] = mass-weighted T/1e10 [K], mu = 4/(1+3X_H+4 X_H x_e) (m_p = 1.67e-24, k_B = 1.38e-16),
              cells within search_radius x max(Group_R_Crit200 over ALL groups) of the centre.
              Used only to validate the precomputed files against the raw data.

 (B) 'true' : ground truth in PHYSICAL units, bins in x = r/R200c (40 log bins 0.01-5 plus a centre bin r<0.01 R200c):
              per-bin sums over cells (volume-integrated quantities, so shell value = sum / V_shell,phys):
                Mgas [Msun], Ne = x_e X_H m/m_p [#], Pe*V = n_e k_B T V = 4 x_e X_H/(1+3X_H+4X_H x_e) (gamma-1) m u [erg],
                Pth*V = (gamma-1) m u [erg], cell volume V = m/rho [kpc^3 phys], each also for star-forming cells
                only (StarFormationRate > 0), plus counts.  x_e = ElectronAbundance (n_e/n_H), X_H = 0.76 (TNG/Lee22).
              Integrated inside x < 1 (sphere of R200c): Mgas, Ne, PeV (= Y_sph in erg), PthV, each all / SF.

 (C) 'lee'  : emulation of Lee22's group_particles bookkeeping (examples/ne_prof.cpp, y_prof.cpp, PR #1):
              128 log shells between 0.03 and 2.5 Group_R_Crit200 (ckpc/h), innermost shell from the origin,
              cells beyond 2.5 R200 dropped; sum_i x_e X_H m_i/m_p (m_p in 1e10 Msun/h units with the SIM h) and
              sum_i 4 x_e X_H/(1+3X_H+4X_H x_e)(gamma-1) m_i u_i, each divided by the COMOVING (ckpc/h)^3 shell volume.
              P200_code = 100 G M200 rho_crit(Time) f_b / R200 with G = 4.30091e4 kpc (km/s)^2/1e10 Msun,
              rho_crit(Time) = 2.775e-8 (Omega0/a^3 + OmegaLambda) [1e10 Msun/h / (kpc/h)^3] (physical), R200 comoving.
              n200_code = 200 rho_crit(Time) f_b / (X_H m_p) in the same (kpc/h)^-3 units (no a^-3), i.e. the
              M1 hypothesis for the (unpublished) density normalisation.

Usage: python camels_direct_profiles.py SNAPSHOT.hdf5 GROUPS.hdf5 OUT.npz [--search-radius 6] [--mmin 10]
"""
import argparse
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- constants (CGS)
G_CGS = 6.67430e-8
M_P = 1.67262192369e-24
MSUN = 1.98847e33
KPC = 3.0856775814913673e21
KB = 1.380649e-16
X_H = 0.76
GAMMA = 5.0 / 3.0
OMEGA_B = 0.049  # CAMELS: fixed for all simulations

p = argparse.ArgumentParser()
p.add_argument("snap")
p.add_argument("groups")
p.add_argument("out")
p.add_argument("--search-radius", type=float, default=6.0, help="illstack search radius in units of max R200 (all groups)")
p.add_argument("--mmin", type=float, default=10.0, help="minimum Group_M_Crit200 [1e10 Msun/h] (illstack: > 10)")
p.add_argument("--xmax", type=float, default=5.0, help="outer radius of the ground-truth profiles [R200c]")
args = p.parse_args()

t0 = time.time()
with h5py.File(args.groups, "r") as g:
    hdr = dict(g["Header"].attrs)
    gpos = g["Group/GroupPos"][:].astype(np.float64)
    m200 = g["Group/Group_M_Crit200"][:].astype(np.float64)
    r200 = g["Group/Group_R_Crit200"][:].astype(np.float64)
with h5py.File(args.snap, "r") as f:
    shdr = dict(f["Header"].attrs)
    pos = f["PartType0/Coordinates"][:].astype(np.float64)
    mass = f["PartType0/Masses"][:].astype(np.float64)
    dens = f["PartType0/Density"][:].astype(np.float64)
    u = f["PartType0/InternalEnergy"][:].astype(np.float64)
    xe = f["PartType0/ElectronAbundance"][:].astype(np.float64)
    sfr = f["PartType0/StarFormationRate"][:].astype(np.float64)
print(f"read {len(mass)} cells, {len(m200)} groups in {time.time() - t0:.0f} s", flush=True)

L = float(shdr["BoxSize"])
h = float(shdr["HubbleParam"])
a = float(shdr["Time"])
z = float(shdr["Redshift"])
Om = float(shdr["Omega0"])
OL = float(shdr["OmegaLambda"])
Ob = float(shdr.get("OmegaBaryon", OMEGA_B))
fb = Ob / Om
print(f"z={z:.5f} a={a:.5f} h={h} Om={Om} OL={OL} Ob={Ob} L={L}", flush=True)

pos = np.mod(pos, L)
tree = cKDTree(pos, boxsize=L)
print(f"tree built {time.time() - t0:.0f} s", flush=True)

# per-cell derived quantities (code units -> CGS where needed)
mu_cell = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
m_g = mass * 1e10 * MSUN / h                         # g
Ne_cell = xe * X_H * m_g / M_P                        # number of electrons
PthV_cell = (GAMMA - 1.0) * m_g * u * 1e10            # erg  (u in (km/s)^2)
PeV_cell = (xe * X_H * mu_cell) * PthV_cell           # n_e k T V = x_e X_H mu (gamma-1) m u  [erg]
V_cell = (mass / dens) * (a / h) ** 3                 # kpc^3 physical
is_sf = sfr > 0
# illstack per-cell quantities (code units)
ill_Tcell = u * (4.0 * 1.67e-24 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)) * (GAMMA - 1.0) / 1.38e-16  # K/1e10
ill_PV = mass * u * (GAMMA - 1.0)
# Lee22 per-cell quantities (code units: 1e10 Msun/h, (km/s)^2)
mprot_code = h * 1.6726219 / 1.98847 * 1e-67
lee_Ne = xe * X_H * mass / mprot_code
lee_PeV = 4.0 * xe * X_H / (1.0 + 3.0 * X_H + 4.0 * X_H * xe) * (GAMMA - 1.0) * mass * u

# ---------------------------------------------------------------- halo selection
sel = np.where((m200 > args.mmin) & (r200 > 0))[0]
rhmax = r200.max()                                    # illstack uses max over ALL groups
Rsearch = args.search_radius * rhmax
print(f"{len(sel)} halos with M200c > {args.mmin} 1e10 Msun/h; rhmax={rhmax:.1f} ckpc/h, search {Rsearch:.0f} ckpc/h", flush=True)

# (A) illstack bins
NB_ILL = 25
lo, hi = 0.1, 1e5
bs = (np.log(hi) - np.log(lo)) / NB_ILL
ill_edges = lo * np.exp(bs * np.arange(NB_ILL + 1))
ill_cen = lo * np.exp(bs * (np.arange(NB_ILL) + 0.5))
ill_vol = 4 * np.pi / 3 * (ill_edges[1:] ** 3 - ill_edges[:-1] ** 3)
# (B) truth bins in x (bin 0 = 0 < x < 0.01)
NB_T = 40
x_edges = np.concatenate([[0.0], np.logspace(-2, np.log10(args.xmax), NB_T + 1)])
# (C) Lee22 bins
NB_LEE = 128

nh = len(sel)
out = {k: np.zeros((nh, NB_ILL)) for k in ("ill_dens", "ill_pth", "ill_tmw", "ill_n")}
tkeys = ("Mgas", "Ne", "PeV", "PthV", "V", "n", "Mgas_sf", "Ne_sf", "PeV_sf", "PthV_sf", "V_sf", "n_sf")
tru = {k: np.zeros((nh, NB_T + 1)) for k in tkeys}
ikeys = ("Mgas", "Ne", "PeV", "PthV", "Mgas_sf", "Ne_sf", "PeV_sf", "PthV_sf", "V")
integ = {k: np.zeros(nh) for k in ikeys}
lee_ne = np.zeros((nh, NB_LEE))
lee_pe = np.zeros((nh, NB_LEE))
lee_npart = np.zeros((nh, NB_LEE))
lee_rlo = np.zeros((nh, NB_LEE))
lee_rhi = np.zeros((nh, NB_LEE))

for j, ig in enumerate(sel):
    c = gpos[ig]
    Rq = max(Rsearch, args.xmax * r200[ig], 2.5 * r200[ig])
    idx = np.asarray(tree.query_ball_point(c, Rq), dtype=np.int64)
    d = pos[idx] - c
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(1))                      # ckpc/h
    # (A) illstack
    s = r < Rsearch
    rr = r[s]
    ii = idx[s]
    w_m = mass[ii]
    out["ill_n"][j] = np.histogram(rr, ill_edges)[0]
    out["ill_dens"][j] = np.histogram(rr, ill_edges, weights=w_m)[0] / ill_vol
    out["ill_pth"][j] = np.histogram(rr, ill_edges, weights=ill_PV[ii])[0] / ill_vol
    sw = np.histogram(rr, ill_edges, weights=w_m)[0]
    out["ill_tmw"][j] = np.histogram(rr, ill_edges, weights=ill_Tcell[ii] * w_m)[0] / np.where(sw > 0, sw, 1)
    # (B) truth in x = r/R200c (ratio is the same in comoving and physical coordinates)
    x = r / r200[ig]
    s = x < args.xmax
    xx = x[s]
    ii = idx[s]
    sf = is_sf[ii]
    for key, arr in (("Mgas", m_g / MSUN), ("Ne", Ne_cell), ("PeV", PeV_cell), ("PthV", PthV_cell), ("V", V_cell)):
        vals = arr[ii]
        tru[key][j] = np.histogram(xx, x_edges, weights=vals)[0]
        tru[key + "_sf"][j] = np.histogram(xx[sf], x_edges, weights=vals[sf])[0]
    tru["n"][j] = np.histogram(xx, x_edges)[0]
    tru["n_sf"][j] = np.histogram(xx[sf], x_edges)[0]
    inn = xx < 1.0
    for key, arr in (("Mgas", m_g / MSUN), ("Ne", Ne_cell), ("PeV", PeV_cell), ("PthV", PthV_cell)):
        vals = arr[ii]
        integ[key][j] = vals[inn].sum()
        integ[key + "_sf"][j] = vals[inn & sf].sum()
    integ["V"][j] = V_cell[ii][inn].sum()
    # (C) Lee22 group_particles emulation (comoving ckpc/h shells)
    Rc = r200[ig]
    logRmin = np.log(0.03 * Rc)
    logRmax = np.log(2.5 * Rc)
    dlogR = (logRmax - logRmin) / (NB_LEE - 1)
    s = r < 2.5 * Rc
    rr = r[s]
    ii = idx[s]
    logR = np.log(np.maximum(rr, 1e-30))
    k = np.where(logR < logRmin, 0, 1 + np.floor((logR - logRmin) / dlogR)).astype(np.int64)
    ok = k < NB_LEE
    k = k[ok]
    ii = ii[ok]
    kk = np.arange(NB_LEE)
    r1 = np.where(kk == 0, 0.0, np.exp(logRmin + dlogR * (kk - 1)))
    r2 = np.where(kk == 0, np.exp(logRmin), np.exp(logRmin + dlogR * kk))
    shell = 4 * np.pi / 3 * (r2 ** 3 - r1 ** 3)       # (ckpc/h)^3 COMOVING
    lee_ne[j] = np.bincount(k, weights=lee_Ne[ii], minlength=NB_LEE) / shell
    lee_pe[j] = np.bincount(k, weights=lee_PeV[ii], minlength=NB_LEE) / shell
    lee_npart[j] = np.bincount(k, minlength=NB_LEE)
    lee_rlo[j] = r1 / Rc
    lee_rhi[j] = r2 / Rc
    if j % 50 == 0:
        print(f"halo {j}/{nh} M={m200[ig]:.1f} ncell={len(idx)} t={time.time() - t0:.0f} s", flush=True)

rho_crit_code = 2.775e-8 * (Om / a ** 3 + OL)          # 1e10 Msun/h / (kpc/h)^3, PHYSICAL rho_crit(z)
P200_code = 100.0 * 4.30091e4 * m200[sel] * rho_crit_code * fb / r200[sel]
n200_code = 200.0 * rho_crit_code * fb / (X_H * mprot_code)

np.savez_compressed(
    args.out,
    z=z, a=a, h=h, Om=Om, OL=OL, Ob=Ob, L=L, X_H=X_H, search_radius=args.search_radius, rhmax=rhmax,
    ID=sel, M200_code=m200[sel], R200_code=r200[sel], GroupPos=gpos[sel],
    ill_edges=ill_edges, ill_cen=ill_cen, **out,
    x_edges=x_edges, **{"true_" + k: v for k, v in tru.items()}, **{"in200_" + k: v for k, v in integ.items()},
    lee_ne=lee_ne, lee_pe=lee_pe, lee_npart=lee_npart, lee_xlo=lee_rlo, lee_xhi=lee_rhi,
    lee_P200_code=P200_code, lee_n200_code=n200_code, lee_mprot_code=mprot_code,
    snapfile=args.snap, groupsfile=args.groups,
)
print(f"wrote {args.out} in {time.time() - t0:.0f} s", flush=True)
