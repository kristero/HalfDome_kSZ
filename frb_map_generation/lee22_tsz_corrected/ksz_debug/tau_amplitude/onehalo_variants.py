"""Catalogue one-halo kSZ power of HalfDome for alternative tau conventions, computed independently of XGPaint (numpy):
  C_ell = (1/4pi) sum_cells sum_i (v_los,i/c)^2 |tau_ell(M, z)|^2,
  tau_ell = sigma_T/d_A^2 * 4 pi int_0^{X R200c} r^2 n_e(r) j0(ell r / d_A) dr   (flat sky; exact for a 3-D truncated profile,
  i.e. the painter's chord-limited 4 R200c sphere).
Uses the painter's (log10 M, z) histogram with summed (v_los/c)^2 (maps/halo_mass_redshift_histogram.h5) and cell centres, as
poisson_mass_split.jl does. Baseline must reproduce poisson_mass_split.jl (ksz_poisson_mass_split.h5).
Variants (tau amplitude / convention group):
  base        XGPaint B16 AGN: f_b rho_crit rho_fit, exponent -(beta+gamma)/alpha, n_e = 0.9 rho_gas/(m_e + 2 m_p/(1+X_H)), 4 R200c sphere
  ws_ne       WebSky eq. (3.15) electron convention n_e = (1 - Y_p/2) rho_gas / m_p (no 0.9)
  printed     B16 eq. (A1) as printed, exponent -(beta-gamma)/alpha
  stein34     Stein+20 eq. (3.4) literal, exponent -beta with B16 alpha, beta, gamma
  shock       B16 shock-heating parameter set (Table 2 right)
  proj4       projected (cylinder) truncation at 4 R200c instead of the sphere (XGPaint default LOS)
  sph1, sph2  sphere at 1 and 2 R200c
  wscut       base with WebSky halo selection: M200c > 1e13 Msun and theta200c > 0.5 arcmin
  comp_*      base minus WebSky-style compensation: uniform sphere of comoving radius (M_comp / (4 pi rho_m0))^(1/3)
              (overdensity 3) with N_e = 0.9 f_b M_comp / (mu_e m_p) (field convention), M_comp = k x M200c
"""
import sys, time
import numpy as np
import h5py
from scipy.integrate import quad
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/tau_amplitude")
import tau_b16_numpy as T

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/tau_amplitude"
BASE = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002"
TCMB_UK = 2.7255e6

with h5py.File(f"{BASE}/maps/halo_mass_redshift_histogram.h5", "r") as f:
    count = f["count"][:].T          # (im, iz) as in Julia
    sv2 = f["sum_vlos_over_c_squared"][:].T
    lme = f["logm_edges"][:]
    ze = f["z_edges"][:]
with h5py.File(f"{BASE}/ksz_poisson_mass_split.h5", "r") as f:
    ells = f["ell"][:]
    cl1h_julia = f["cl_1h_b16"][:].sum(0)

cells = np.argwhere(count > 0)
print(f"cells {len(cells)} halos {count.sum()}")

# distances on a fine z grid
zg = np.linspace(0, 4.1, 4101)
chi_g = np.concatenate([[0], np.cumsum(0.5 * (1 / T.E(zg[1:]) + 1 / T.E(zg[:-1])) * np.diff(zg))]) * T.C_KMS / (100 * T.h)  # Mpc
chi_of = lambda z: np.interp(z, zg, chi_g)
rho_m0 = T.Om * T.rho_crit(0.0)  # kg/m^3

NE_XGP = 0.9 / T.mass_per_electron()                # electrons per kg of gas
NE_WS = (1 - 0.24 / 2) / T.M_P
MUE_FIELD = T.mass_per_electron()                    # mu_e m_p for fully ionised He (z < 3)


def shape(u, al, be, ga, mode):
    if mode == "stein34":
        return u ** ga * (1 + u ** al) ** (-be)
    s = -1 if mode == "printed" else 1
    return u ** ga * (1 + u ** al) ** (-(be + s * ga) / al)


NR = 1200
x_grid = np.exp(np.linspace(np.log(1e-5), np.log(4.0), NR))   # r / R200c, sphere integrals


def tau_ell_cell(M, z, variant):
    """tau_ell on the global ells for one (M, z); also returns theta200. Composite variants join tokens with '+'."""
    if "+" in variant:
        toks = variant.split("+")
        tl, th = tau_ell_cell(M, z, toks[0])
        for t in toks[1:]:
            if t.startswith("comp_") or t == "wscut":
                continue
            raise ValueError(t)
        comp = [t for t in toks if t.startswith("comp_")]
        if comp:
            tl = tl - comp_term(M, z, float(comp[0].split("_")[1]))
        return tl, th
    if variant == "msun_noh":      # sensitivity: catalogue masses already in Msun (painter's /0.68 not applied)
        return tau_ell_cell(M * T.h, z, "base")
    model = "shock" if variant == "shock" else "AGN"
    rho0, xc, al, be, ga = T.b16_params(M, z, model)
    mode = variant if variant in ("printed", "stein34") else "plus"
    if variant in ("steintext_a", "steintext_b"):   # Stein+20 text: alpha = 1, gamma = -0.3 fixed (B16 rho0, beta, xc kept)
        al, ga = 1.0, -0.3
        mode = "stein34" if variant == "steintext_a" else "plus"
    rc = T.rho_crit(z)
    R = T.r200c(M, z) * T.MPC                 # m physical
    chi = chi_of(z) * T.MPC
    dA = chi / (1 + z)
    ne_per_kg = NE_WS if variant == "ws_ne" else NE_XGP
    amp = T.SIGMA_T * ne_per_kg * T.fb * rc * rho0          # sigma_T n_e / shape, m^-1
    Xmax = {"sph1": 1.0, "sph2": 2.0}.get(variant, 4.0)
    k = ells / dA                                              # 1/m
    if variant == "proj4":
        # cylinder of radius 4 R200c through the full LOS: 2-D Hankel of the full-LOS projected profile
        xs = np.exp(np.linspace(np.log(1e-5), np.log(4.0), 700))
        proj = np.array([T.los_integral(xx, (rho0, xc, al, be, ga), np.inf, +1) for xx in xs])   # R200c units
        from scipy.special import j0 as J0
        kr = np.outer(k, xs * R)
        integ = proj[None, :] * J0(kr) * (xs * R)[None, :] ** 2 * R     # Sigma(R) R dR with dlnR
        tl = amp * 2 * np.pi * np.trapz(integ, np.log(xs), axis=1) / dA ** 2
        return tl, np.arctan(R / dA)
    x = x_grid[x_grid <= Xmax] if Xmax < 4 else x_grid
    r = x * R
    f = shape(x / xc, al, be, ga, mode)
    kr = np.outer(k, r)
    j0 = np.sinc(kr / np.pi)
    integ = f[None, :] * j0 * r[None, :] ** 3                  # r^2 f dr = r^3 f dln r
    tl = amp * 4 * np.pi * np.trapz(integ, np.log(r), axis=1) / dA ** 2
    if variant.startswith("comp_"):
        tl = tl - comp_term(M, z, float(variant.split("_")[1]))
    return tl, np.arctan(R / dA)


def comp_term(M, z, kfac):
    """WebSky-style compensation: uniform sphere, mass kfac*M200c, comoving radius at overdensity 3 of the mean matter density,
    electrons in the field convention f_e = 0.9, mu_e(z<3)."""
    chi = chi_of(z) * T.MPC
    dA = chi / (1 + z)
    Mc = kfac * M * T.MSUN
    Rc_com = (Mc / (4 * np.pi * rho_m0)) ** (1 / 3)
    Ne = 0.9 * T.fb * Mc / MUE_FIELD
    y = ells * Rc_com / chi
    W = 3 * (np.sin(y) - y * np.cos(y)) / y ** 3
    return T.SIGMA_T * Ne / dA ** 2 * W


def run(variant):
    cl = np.zeros(len(ells))
    clm = np.zeros((6, len(ells)))
    edges = [12.0, 13.0, 13.5, 14.0, 14.5, 15.0, 15.7]
    for im, iz in cells:
        lm = 0.5 * (lme[im] + lme[im + 1]); z = 0.5 * (ze[iz] + ze[iz + 1]); M = 10 ** lm
        tl, th200 = tau_ell_cell(M, z, variant)
        if "wscut" in variant.split("+") and (M <= 1e13 or th200 <= 0.5 / 60 * np.pi / 180):
            continue
        c = sv2[im, iz] * tl ** 2 / (4 * np.pi)
        cl += c
        b = np.searchsorted(edges, lm, side="right") - 1
        clm[b] += c
    return cl, clm


def D(cl):
    return ells * (ells + 1) * cl / (2 * np.pi) * TCMB_UK ** 2


if __name__ == "__main__":
    variants = sys.argv[1:] or ["base", "ws_ne", "printed", "stein34", "shock", "sph1", "sph2", "wscut", "comp_1.0", "comp_1.3", "proj4"]
    ws = np.loadtxt(f"{BASE}/websky_halo_stein2020_fig6_digitized_binned.csv", delimiter=",")
    Lref = np.array([300, 520, 1000, 2000, 3000, 4700])
    res = {}
    for v in variants:
        t0 = time.time()
        cl, clm = run(v)
        res[v] = cl
        np.savez(f"{OUT}/onehalo_{v}.npz", ell=ells, cl=cl, cl_massbins=clm)
        Dv = D(cl)
        Di = np.exp(np.interp(np.log(Lref), np.log(ells), np.log(Dv)))
        wsi = np.interp(np.log(Lref), np.log(ws[:, 0]), ws[:, 1], left=np.nan)
        print(f"{v:10s} ({time.time()-t0:5.1f}s) D_ell[uK^2] at {list(Lref)}: " + " ".join(f"{d:.4f}" for d in Di)
              + " | /WebSky: " + " ".join(f"{d/w:.2f}" for d, w in zip(Di, wsi)), flush=True)
    if "base" in res:
        r = res["base"] / cl1h_julia
        print("base / poisson_mass_split.jl (Julia):", " ".join(f"l={ells[i]:.0f}:{r[i]:.4f}" for i in range(0, 80, 10)))
        for v in res:
            rr = res[v] / res["base"]
            print(f"  {v:10s}/base at ell {list(Lref)}: " + " ".join(
                f"{np.exp(np.interp(np.log(L), np.log(ells), np.log(np.abs(rr)))):.3f}" for L in Lref))
