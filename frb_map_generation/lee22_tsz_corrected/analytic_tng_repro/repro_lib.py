#!/usr/bin/env python3
"""Shared pieces of the analytic TNG300 reproduction (2026-09-30): profile tables, halo abundance (Tinker08 and
the TNG300 hydro abundance), slice generator for the Torkamani+26 compound-Poisson engine
(../../analytic_pdf_torkamani2026/halo_model_dm_pdf.py, imported unchanged), the BFC hot-gas table truncated at
5 R200c as in Torkamani+26, and the Konietzka TNG300 ray arrays.

Units: masses in physical Msun (M200c), distances in Mpc, DM in pc cm^-3 (observer frame)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate

ROOT = Path("/home/cbllover/HalfDome")
FRB = ROOT / "frb_map_generation"
ENGINE = FRB / "analytic_pdf_torkamani2026"
sys.path.insert(0, str(ENGINE))
import halo_model_dm_pdf as hm  # noqa: E402

OUT = FRB / "outputs" / "lee22_analytic_tng_repro_20260930"
TABLE = OUT / "tables" / "repro_sphere_profiles.h5"
BFC_R200_TABLE = FRB / "outputs" / "analytic_halo_pdf_20260925" / "tables" / "bfc_tng_fiducial_sphere_profiles_planck15.npz"
BFC_5R200_TABLE = OUT / "tables" / "bfc_tng_fiducial_5r200c_planck15.npz"
KON = ROOT / "frb_catalog_comparison_outputs" / "haloDM-20260819T104507Z-1-001" / "haloDM"
BFC_FIG6 = ROOT / "frb_catalog_comparison_outputs" / "analytical_bfc_torkamani2026" / "figure6_bfc_z0p7_digitized.csv"
COSMO = hm.PLANCK15_TNG
H_TNG = COSMO.h
PDF_EDGES = np.logspace(-1, np.log10(3e4), 300)        # the simulation histograms' edges (299 log bins)
CDF_GRID = np.logspace(-3, np.log10(3.3e4), 40001)
LIN_EDGES = np.arange(0.0, 4000.01, 10.0)                # total-LSS comparison (linear DM)

MODELS = ("b16", "lee22_prev_xgpnative", "lee22_corr_fn5p40", "lee22_corr_fn5p57", "bfc")
MODEL_LABEL = {"b16": "Battaglia16", "lee22_prev_xgpnative": "Lee22 previous (x0.6016)",
               "lee22_corr_fn5p40": "Lee22 corrected, F_n = 5.4", "lee22_corr_fn5p57": "Lee22 corrected, F_n = 5.57",
               "bfc": "BFC (Torkamani Table 1)"}

# Effective upper-edge factor of Konietzka's windows (window algebra, ../konietzka_check/konietzka_window_hmf.py)
C_EXT = 9.0
# TNG300-1 hydro halo counts, Pillepich et al. 2018b (arXiv:1707.03406) Sec. 2: M200c >= 1e13: 3733 (z = 0),
# 2333 (z = 1); >= 1e14: 280 (z = 0); >= 1e15: 3 (z = 0). Tinker08 expectation in the same comoving volume
# (205 Mpc/h)^3 (computed in ../konietzka_check/analyze_konietzka_check.py and re-derived in finite_volume.py):
TINKER_BOX = dict(z0_13_14=4648.4, z0_14_16=296.5, z1_13_14=2801.1, z1_14_16=52.1)
A_LO = {0.0: (3733 - 280) / TINKER_BOX["z0_13_14"], 1.0: (2333 - 0.945 * 52.1) / TINKER_BOX["z1_13_14"]}
A_HI = 280 / TINKER_BOX["z0_14_16"]


MASS_DECOMP = OUT / "analysis" / "mass_decomposition.json"
_A_LOW = None


def low_mass_abundance():
    """TNG300 abundance / Tinker08 below 1e13 Msun, per decade, measured geometrically from the ray hit fractions of
    the nested-window differences (mass_decomposition.py: A_eff = -ln(1 - hit_TNG) / Nbar_Tinker08), averaged over
    z_s = 0.1-1 (it varies by <= 5 % with z_s)."""
    global _A_LOW
    if _A_LOW is None:
        d = json.loads(MASS_DECOMP.read_text())["decades"]
        _A_LOW = {}
        for key, lo in (("1e9-1e10", 9), ("1e10-1e11", 10), ("1e11-1e12", 11), ("1e12-1e13", 12)):
            v = [a for a in d[key]["A_eff_from_hits"].values() if a is not None]
            _A_LOW[lo] = float(np.mean(v))
    return _A_LOW


def abundance_factor(kind, log10m, z):
    """Multiplicative factor on the Tinker08 M200c mass function.
      tinker08     1
      tng300       Pillepich+18b TNG300-1 counts for M >= 1e13 (a_lo(z) for 1e13-1e14, A_HI above), 1 below
      tng300_full  as tng300 for M >= 1e13, and the hit-calibrated per-decade factors below 1e13"""
    log10m = np.asarray(log10m, float)
    if kind == "tinker08":
        return np.ones_like(log10m)
    if kind in ("tng300", "tng300_full"):
        a_lo = np.interp(z, [0.0, 1.0], [A_LO[0.0], A_LO[1.0]])
        out = np.where(log10m >= 14.0, A_HI, np.where(log10m >= 13.0, a_lo, 1.0))
        if kind == "tng300_full":
            for lo, a in low_mass_abundance().items():
                out = np.where((log10m >= lo) & (log10m < lo + 1), a, out)
            out = np.where(log10m < 9.0, low_mass_abundance()[9], out)
        return out
    raise ValueError(kind)


class Table:
    """g(x; M, z) = DM_sphere/(2 sqrt(1-x^2)) on the engine's nodes, x in units of the truncation radius R_t,
    theta = angular R_t. For the R200c tables R_t = R200c."""

    def __init__(self, log10m, z, x, g, theta, name, rt_over_r200=1.0):
        self.log10m, self.z, self.x, self.g, self.theta200 = log10m, z, x, g, theta
        self.name, self.rt_over_r200 = name, rt_over_r200


def load_table(model):
    if model == "bfc":
        d = np.load(BFC_R200_TABLE)
        return Table(d["log10m"], d["z"], d["x"], d["g"], d["theta200"], "bfc")
    if model == "bfc_5r200":
        d = np.load(BFC_5R200_TABLE)
        return Table(d["log10m"], d["z"], d["x"], d["g"], d["theta_t"], "bfc_5r200", rt_over_r200=5.0)
    t = hm.HalfDomeSphereTable(TABLE, model)
    return Table(t.log10m, t.z, t.x, t.g, t.theta200, model)


def bfc_chord_mean_dm_rt(cosmo, log10m, z, xt, rt=5.0, par=hm.BFC_TNG_FIDUCIAL, n_gl=128):
    """BFC hot gas (engine's Table 1 implementation, hm.bfc_chord_mean_dm) integrated through a sphere of radius
    rt R200c instead of R200c: g = DM/(2 sqrt(1-xt^2)) with xt = b/(rt R200c)."""
    mass = 10 ** np.asarray(log10m, float)
    # hm.bfc_chord_mean_dm evaluates rho0 and the profile shape; reuse it at x = 0 to get the normalization factor
    c = hm.dutton_maccio14_c200c(cosmo, mass, z)
    nu = hm.DELTA_C / (cosmo.sigma_M(mass) * cosmo.growth(z)[0])
    eps = par["eps0"] + par["eps1"] * nu
    q = mass * cosmo.h / 10 ** par["log10Mc"]
    beta = 3 * q ** par["mu"] / (1 + q ** par["mu"])
    fhga = hm.bfc_fractions(cosmo, mass, par)[0]
    u = np.union1d(np.logspace(-6, 3.5, 6000), [1.0])
    lu = np.log(u)
    ynfw = 1 / ((u[None] * c[:, None]) * (1 + u[None] * c[:, None]) ** 2 * (1 + (u[None] / eps[:, None]) ** 2) ** 2)
    m_inf = hm._trapz(ynfw * u ** 3, lu, axis=1)
    m_200 = hm._trapz((ynfw * u ** 3)[:, u <= 1], lu[u <= 1], axis=1)

    def yhga(uu):
        return (1 + uu / par["theta_c"]) ** -beta[:, None] * (1 + (uu / eps[:, None]) ** par["gamma"]) ** (-par["delta"] / par["gamma"])

    norm = hm._trapz(yhga(u[None]) * u ** 3, lu, axis=1)
    r200 = hm.r200c_mpc(cosmo, mass, z)
    rho0 = fhga * (m_inf / m_200) * mass / (4 * np.pi * r200 ** 3 * norm)
    ne0 = rho0 * hm.MSUN_G / hm.MPC_CM ** 3 * hm.ELECTRONS_PER_MP / hm.M_P_G
    xt = np.asarray(xt, float)
    b = rt * xt                                                   # impact parameter / R200c
    L = rt * np.sqrt(np.clip(1 - xt ** 2, 0, None))               # half chord / R200c
    t, w = np.polynomial.legendre.leggauss(n_gl)
    t = 0.5 * (t + 1); w = 0.5 * w
    # substitute l = L s^2 to resolve the centre-crossing cusp (dl = 2 L s ds)
    s = t
    l = L[:, None] * s[None] ** 2
    r = np.sqrt(b[:, None] ** 2 + l ** 2)
    prof = yhga(r.reshape(1, -1)).reshape(len(mass), len(xt), n_gl)
    mean_col = (prof * (w * 2 * s)[None, None, :]).sum(-1)         # (1/L) int_0^L y dl
    fac = ne0 * r200 * hm.MPC_CM / hm.PC_CM / (1 + z)
    # DM = 2 L fac mean_col ; g = DM / (2 sqrt(1-xt^2)) = rt fac mean_col
    return rt * fac[:, None] * mean_col, dict(fhga=fhga, beta=beta, eps=eps)


def build_bfc_5r200_table(log10m=None, zs=None, xs=None, rt=5.0):
    if BFC_5R200_TABLE.exists():
        return
    if log10m is None:
        log10m = np.round(np.arange(8.005, 15.9951, 0.01), 3)
    if zs is None:
        zs = np.round(np.arange(0.005, 0.9951, 0.01), 3)
    if xs is None:
        xs = np.concatenate([10 ** np.linspace(-4, np.log10(0.3), 90), np.linspace(0.31, 0.9995, 110)])
    g = np.zeros((len(log10m), len(zs), len(xs)))
    theta = np.zeros((len(log10m), len(zs)))
    for i, z in enumerate(zs):
        g[:, i, :], _ = bfc_chord_mean_dm_rt(COSMO, log10m, z, xs, rt=rt)
        r200 = hm.r200c_mpc(COSMO, 10 ** log10m, z)
        theta[:, i] = np.arctan(rt * r200 / (COSMO.chi(z)[0] / (1 + z)))
    BFC_5R200_TABLE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(BFC_5R200_TABLE, g=g, theta_t=theta, x=xs, log10m=log10m, z=zs, rt_over_r200=rt,
             params=json.dumps(hm.BFC_TNG_FIDUCIAL), electrons_per_mp=hm.ELECTRONS_PER_MP, cosmology=COSMO.name,
             note="BFC hot gas (Torkamani+26 Table 1) inside a sphere of radius 5 R200c; x = b/(5 R200c)")


def slices(table, window, z_source, abundance="tinker08", cosmo=COSMO, x_edges=None, dlm=0.01, dz=0.01):
    """Per-z-slice inputs for hm.compound_poisson_pdf (same construction as run_analytic_halo_pdfs.slices_for, plus a
    source-redshift cut and an abundance factor on Tinker08)."""
    if x_edges is None:
        x_edges = hm.impact_samples()
    lo, hi = window
    lm = table.log10m
    frac = np.clip((np.minimum(lm + dlm / 2, np.log10(hi)) - np.maximum(lm - dlm / 2, np.log10(lo))) / dlm, 0, 1)
    sel = frac > 0
    logx = np.log(table.x)
    xe = np.clip(x_edges, table.x[0], table.x[-1])
    chord = 2 * np.sqrt(np.clip(1 - x_edges ** 2, 0, None))
    area_frac = np.diff(x_edges ** 2)
    mass = 10 ** lm[sel]
    for i, z in enumerate(table.z):
        if z >= z_source:
            break
        g = table.g[sel, i, :]
        theta = table.theta200[sel, i]
        spl = interpolate.CubicSpline(logx, np.log(np.maximum(g, 1e-300)), axis=1)
        dm_edges = np.exp(spl(np.log(xe))) * chord[None, :]
        dm_edges = np.minimum.accumulate(dm_edges, axis=1)
        dm_edges[:, -1] = 0.0
        w = (cosmo.chi(z)[0] ** 2 / cosmo.H_over_c(z) * dz * hm.tinker08_dndlnm(cosmo, mass, z) * dlm * np.log(10)
             * frac[sel] * abundance_factor(abundance, lm[sel], z))
        disc = np.pi * theta ** 2
        aw = (w * disc)[:, None] * area_frac[None, :]
        bias = hm.tinker10_bias(cosmo, mass, z)
        mid = 0.5 * (dm_edges[:, 1:] + dm_edges[:, :-1])
        yield dict(z=float(z), dz=dz, dm_edges=dm_edges, area_weights=aw, bias=bias,
                   nbar=float((w * disc).sum()), nbar_b=float((w * disc * bias).sum()), mean_dm=float((aw * mid).sum()))


def moments(table, window, z_sources, abundance="tinker08", cosmo=COSMO):
    """Mean DM over all rays and expected crossings Nbar (unclustered hit = 1 - exp(-Nbar)) to each z_s, without FFT."""
    out = {}
    cum_mean = cum_nbar = 0.0
    zs = sorted(z_sources)
    k = 0
    for s in slices(table, window, max(zs) + 1e-9, abundance, cosmo):
        while k < len(zs) and s["z"] >= zs[k]:
            out[zs[k]] = (cum_mean, cum_nbar); k += 1
        cum_mean += s["mean_dm"]; cum_nbar += s["nbar"]
    while k < len(zs):
        out[zs[k]] = (cum_mean, cum_nbar); k += 1
    return out


# ------------------------------------------------------------------ TNG300 (Konietzka) arrays
TNG_FILES = {"lss_total": "DMall_IGM_halo.npy", "all": "DMhalo_r200_all.npy", "m1e9_to_1e13": "DMhalo_r200_mass9to13.npy",
             "m1e9_to_1e14": "DMhalo_r200_mass9to14.npy", "m1e10_to_1e12": "DMhalo_r200_mass10to12.npy",
             "m1e10_to_1e13": "DMhalo_r200_mass10to13.npy", "m1e10_to_1e14": "DMhalo_r200_mass10to14.npy",
             "m1e10_to_1e15": "DMhalo_r200_mass10to15.npy", "m1e11_to_1e14": "DMhalo_r200_mass11to14.npy",
             "m1e12_to_1e14": "DMhalo_r200_mass12to14.npy", "m1e12_to_1e15": "DMhalo_r200_mass12to15.npy",
             "m1e13_to_1e14": "DMhalo_r200_mass13to14.npy"}
_TNG_Z = None
_TNG = {}


def tng(label, z_s):
    global _TNG_Z
    if _TNG_Z is None:
        _TNG_Z = np.load(KON / "redshifts_DMhalo.npy")
    if label not in _TNG:
        _TNG[label] = np.load(KON / TNG_FILES[label], mmap_mode="r")
    i = np.flatnonzero(np.isclose(_TNG_Z, z_s))
    assert len(i) == 1, (label, z_s)
    return np.asarray(_TNG[label][int(i[0])], float)


# ------------------------------------------------------------------ statistics
def ray_stats(values):
    v = np.asarray(values, float)
    h = v > 0
    return dict(hit_fraction=float(h.mean()), mean_all=float(v.mean()), mean_hit=float(v[h].mean()),
                median_hit=float(np.median(v[h])), std_all=float(v.std()), n=int(v.size))


def model_cdf_hit(npz, clustered=True):
    """(grid, CDF over hit rays) from a saved run."""
    c = npz["cdf_hit_clustered" if clustered else "cdf_hit_unclustered"]
    return npz["cdf_grid"], c / c[-1]


def ks_distance(sample, grid, cdf):
    """sup |F_model - F_sample| over the sample's order statistics (two-sided)."""
    x = np.sort(np.asarray(sample, float))
    n = x.size
    fm = np.interp(x, grid, cdf, left=0.0, right=1.0)
    return float(max(np.max(np.arange(1, n + 1) / n - fm), np.max(fm - np.arange(0, n) / n)))


def compound_poisson_pdf_tapered(lattice, slices_iter, cosmo=COSMO, clustering_dz=0.05, taper_lambda=None):
    """hm.compound_poisson_pdf (same arithmetic, copied) with an optional Gaussian taper exp[-(lambda/taper_lambda)^2] on
    the clustering exponent 1/2 <A^2>. Needed only when every ray crosses hundreds of halos (Torkamani's 5 R200c discs
    down to 1e8 Msun/h): then the Gaussian peak-background average (their eq. 46) gives Re 1/2 <A^2> -> +1/2 sum amp
    nbar_b^2 >> Nbar at lambda >~ 1 (DM scales <~ 5 pc cm^-3) and the characteristic function does not decay. The taper
    keeps the physical small-lambda part (the lambda^2 broadening by large-scale modulation of the halo abundance) and
    leaves clustering out of the sub-10 pc cm^-3 structure. taper_lambda=None reproduces the engine exactly."""
    n = lattice.n
    total = np.zeros(n + 2)
    nbar = mean_all = 0.0
    S = cosmo.projected_variance()
    log_cl = np.zeros(n // 2 + 1, complex)
    lam = 2 * np.pi * np.arange(n // 2 + 1) / (n * lattice.h)
    taper = np.ones_like(lam) if taper_lambda is None else np.exp(-(lam / taper_lambda) ** 2)
    group = None
    n_groups = 0

    def flush(g):
        zc = g["z_w"] / g["dz"]
        amp = cosmo.H_over_c(zc) * cosmo.growth(zc)[0] ** 2 * S / g["dz"]
        A = np.fft.rfft(hm.deposit_to_cells(lattice, g["dep_b"])) - g["nbar_b"]
        return 0.5 * amp * A * A

    for s in slices_iter:
        dd, dd_b = hm.deposit_difference(lattice, s["dm_edges"], [s["area_weights"], s["area_weights"] * s["bias"][:, None]])
        total += dd
        nbar += s["nbar"]
        mean_all += s["mean_dm"]
        gkey = int(np.floor(s["z"] / clustering_dz + 1e-9))
        if group is not None and group["key"] != gkey:
            log_cl += flush(group); n_groups += 1
            group = None
        if group is None:
            group = dict(key=gkey, dep_b=np.zeros(n + 2), nbar_b=0.0, z_w=0.0, dz=0.0)
        group["dep_b"] += dd_b
        group["nbar_b"] += s["nbar_b"]
        group["z_w"] += s["z"] * s["dz"]; group["dz"] += s["dz"]
    if group is not None:
        log_cl += flush(group); n_groups += 1
    total = hm.deposit_to_cells(lattice, total)
    log_u = np.fft.rfft(total) - nbar
    p_u = np.fft.irfft(np.exp(log_u), n)
    p_cl = np.fft.irfft(np.exp(log_u + log_cl * taper), n)
    info = dict(n_clustering_groups=n_groups, projected_variance_mpc=S, dropped_crossings_per_ray=lattice.dropped,
                taper_lambda=taper_lambda, max_re_log_cf_clustered=float(np.max((log_u + log_cl * taper).real)))
    zero_u = float(np.exp(-nbar)) if nbar < 700 else 0.0
    return hm.PDFResult(lattice, p_u, p_cl, zero_u, 0.0 if nbar > 50 else float("nan"), nbar, mean_all, info)
