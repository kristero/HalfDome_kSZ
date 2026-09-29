#!/usr/bin/env python3
"""Map-level CLASS-SZ verification of the painted Battaglia12 and Lee22 Compton-y maps, the Lee22/B12 mass and
redshift decomposition, the M1 (pressure x (1+z)^2) hypothesis, and the catalogue-vs-Tinker08 offset.

Inputs: OUT/class_sz/*.npz (run_class_sz.py), OUT/halo_model/ygrid_*.npz (compute_python_grids.py),
OUT/catalogue/catalogue_hist.npz (catalogue_hist.py), map spectra fullsky_spectra.npz (production maps) and the
verify_5 split-map spectra tsz_split_cl.npz.
Map spectra: map2alm lmax 8192, iter 3, pixel window NOT deconvolved. The maps are point samples of the profile at
pixel centres (XGPaint evaluates y at the pixel centre; no pixel-area average), so the like-for-like theory has NO
pixel window: 'raw' is the primary convention, 'pixwin-deconvolved' (C_ell/w_ell^2) is shown for comparison with
the 2026-09-25 truncation series, which used it.
Bands: mean over integer ell in [0.9 l, 1.1 l] (as verify_5), theory interpolated log-log onto the same integers.
Output: OUT/analysis/results.json, curves.npz.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import halo_model as HM  # noqa: E402

OUT = Path(C.OUT); CS = OUT / "class_sz"; HMD = OUT / "halo_model"
AND = OUT / "analysis"; AND.mkdir(parents=True, exist_ok=True)
BANDS = (30, 100, 300, 1000, 2000, 3000, 5000)
DECOMP_ELL = (100, 300, 1000, 3000)
CLIP = C.LEE_CLIP_HINV
M_EDGES = [C.CAT_M_MIN, 1e13, 3e13, 1e14, 3e14, CLIP, C.CAT_M_MAX]
Z_EDGES = [C.CAT_Z_MIN, 0.1, 0.25, 0.5, 1.0, 2.0, C.CAT_Z_MAX]
LMAP = np.arange(8193)


def band(cl_int, l):
    s = slice(int(l * 0.9), int(l * 1.1) + 1)
    return float(np.mean(cl_int[s]))


def to_int(ell, cl):
    """log-log interpolation of a theory C_ell onto integer ell 0..8192 (ell < 10 set to the ell = 10 value)."""
    out = np.exp(np.interp(np.log(np.maximum(LMAP, ell[0])), np.log(ell), np.log(np.maximum(cl, 1e-300))))
    out[0] = 0
    return out


GLITCHES = []


def deglitch(ell, c, name, term, thr=0.03):
    """Replace isolated single-multipole CLASS-SZ integration failures (a point deviating by > thr in log from the
    log-mean of its two neighbours, and by > 1.5x the neighbours' own deviations) by that log-mean; logged in GLITCHES."""
    c = np.array(c, float)
    lc = np.log(np.maximum(c, 1e-300))
    dev = np.zeros_like(lc); dev[1:-1] = lc[1:-1] - 0.5 * (lc[2:] + lc[:-2])
    for i in range(1, len(c) - 1):
        if abs(dev[i]) > thr and abs(dev[i]) > 1.5 * max(abs(dev[i - 1]), abs(dev[i + 1])) and c[i - 1] > 0 and c[i + 1] > 0:
            g = dict(run=name, term=term, ell=float(ell[i]), factor=float(np.exp(dev[i])))
            if g not in GLITCHES:
                GLITCHES.append(g)
            c[i] = np.sqrt(c[i - 1] * c[i + 1])
    return c


def cs(name):
    d = np.load(CS / f"{name}.npz")
    return d["ell"], deglitch(d["ell"], d["cl_1h"], name, "1h"), deglitch(d["ell"], d["cl_2h"], name, "2h")


def best(nm):
    """Most precise CLASS-SZ run on disk for a run name: HIPREC, else MEDPREC (run_class_sz_prec2.py), else default."""
    for suf in ("_hiprec", "_medprec", ""):
        if (CS / f"{nm}{suf}.npz").exists():
            return nm + suf
    return None


PREC_USED = {}


class Grid:
    """Python halo model from a tabulated y grid; sub-ranges are exact subsets of the GL nodes."""

    def __init__(self, ing, model, kind):
        d = np.load(HMD / f"ygrid_hm_{model}_{kind}.npz")
        self.z, self.wz, self.lnm, self.wm, self.y, self.ell = d["z"], d["wz"], d["lnm"], d["wm"], d["y"], d["ell"]
        self.n = ing.dndlnm(self.z, self.lnm); self.b = ing.bias(self.z, self.lnm)
        self.vol = ing.chi_h(self.z) ** 2 / ing.hub_h(self.z)
        self.pk = ing.pk(self.z, self.ell)

    def masks(self, m_lo, m_hi, z_lo, z_hi):
        M = np.exp(self.lnm)
        return (M > m_lo) & (M < m_hi), (self.z > z_lo) & (self.z < z_hi)

    def cl(self, m_lo=C.CAT_M_MIN, m_hi=C.CAT_M_MAX, z_lo=C.CAT_Z_MIN, z_hi=C.CAT_Z_MAX, y=None, keep=False):
        y = self.y if y is None else y
        mm, zm = self.masks(m_lo, m_hi, z_lo, z_hi)
        W = (self.wz * self.vol * zm)[:, None] * (self.wm * self.n * mm[None, :])     # (nz, nm)
        d1 = W[:, :, None] * y ** 2
        A = np.einsum("jm,jml->jl", self.wm * self.n * self.b * mm[None, :], y)
        d2z = (self.wz * self.vol * zm)[:, None] * A ** 2 * self.pk
        r = dict(cl_1h=d1.sum((0, 1)), cl_2h=d2z.sum(0))
        if keep:
            r["d1"] = d1
            r["d2"] = ((self.wz * self.vol * zm)[:, None, None] * (self.wm * self.n * self.b * mm[None, :])[:, :, None]
                       * y * (A * self.pk)[:, None, :])
        return r

    def counts(self, m_lo, m_hi, z_lo, z_hi):
        mm, zm = self.masks(m_lo, m_hi, z_lo, z_hi)
        return float(4 * np.pi * ((self.wz * self.vol * zm)[:, None] * (self.wm * self.n * mm[None, :])).sum())


class CatGrid:
    """Catalogue-weighted terms from the (z, lnM) histogram of the 85.2M painted halos (full sky)."""

    def __init__(self, ing, model, kind):
        h = np.load(OUT / "catalogue" / "catalogue_hist.npz")
        d = np.load(HMD / f"ygrid_cat_{model}_{kind}.npz")
        self.N, self.ze, self.me = h["N"], h["z_edges"], h["lnm_edges"]
        self.z, self.lnm, self.y, self.ell = d["z"], d["lnm"], d["y"], d["ell"]
        self.b = ing.bias(self.z, self.lnm)
        chi = ing.chi_h(self.ze)
        self.dV = 4 * np.pi / 3 * (chi[1:] ** 3 - chi[:-1] ** 3)            # (Mpc/h)^3 per z bin, full sky
        self.pk = ing.pk(self.z, self.ell)

    def masks(self, m_lo, m_hi, z_lo, z_hi):
        mc = np.exp(0.5 * (self.me[1:] + self.me[:-1])); zc = 0.5 * (self.ze[1:] + self.ze[:-1])
        return (mc > m_lo) & (mc < m_hi), (zc > z_lo) & (zc < z_hi)

    def cl(self, m_lo=C.CAT_M_MIN, m_hi=C.CAT_M_MAX, z_lo=0.0, z_hi=10.0, y=None, keep=False):
        y = self.y if y is None else y
        mm, zm = self.masks(m_lo, m_hi, z_lo, z_hi)
        Nm = self.N * mm[None, :] * zm[:, None]
        d1 = Nm[:, :, None] * y ** 2 / (4 * np.pi)
        A = np.einsum("jm,jml->jl", Nm * self.b, y) / self.dV[:, None]
        d2z = (self.dV / (4 * np.pi))[:, None] * A ** 2 * self.pk
        r = dict(cl_1h=d1.sum((0, 1)), cl_2h=d2z.sum(0))
        if keep:
            r["d1"] = d1
            r["d2"] = (self.dV / (4 * np.pi))[:, None, None] * (Nm * self.b / self.dV[:, None])[:, :, None] * y * (A * self.pk)[:, None, :]
        return r

    def counts(self, m_lo, m_hi, z_lo, z_hi):
        mm, zm = self.masks(m_lo, m_hi, z_lo, z_hi)
        return float((self.N * mm[None, :] * zm[:, None]).sum())


def zfac(g):
    return ((1 + g.z) ** 2)[:, None, None]


def hybrid_y(gl, gb):
    hi = np.exp(gl.lnm) >= CLIP
    return np.where(hi[None, :, None], gb.y, gl.y)


def main():
    ing = HM.Ingredients()
    fs = np.load(C.FULLSKY_SPECTRA, allow_pickle=True)
    sp = np.load(C.SPLIT_SPECTRA)
    import healpy as hp
    pw2 = hp.pixwin(4096, lmax=8192) ** 2
    maps = {"B12": fs["cl_y_b12"], "L22": fs["cl_y_lee22p"], "L22calib": fs["cl_y_lee22p_calib"],
            "L22low": sp["cl_Ll"], "L22high": sp["cl_Lh"], "B12low": sp["cl_Bl"], "B12high": sp["cl_Bh"],
            "hybrid": sp["cl_hyb"], "L22low_x_high": sp["cl_Ll_x_Lh"], "B12low_x_high": sp["cl_Bl_x_Bh"]}

    # ---------------------------------------------------------------- CLASS-SZ theory (1h, 2h) on integer ell
    T = {}
    def put(key, e, c1, c2):
        T[key + "_1h"], T[key + "_2h"] = to_int(e, c1), to_int(e, c2)
        T[key] = T[key + "_1h"] + T[key + "_2h"]
    for nm in ("b12_paint_all", "b12_paint_low", "b12_paint_high", "lee22_paint_low", "lee22_paint_high",
               "lee22_paint_calib", "lee22m1_paint_low", "lee22m1_paint_high", "lee22m1_paint_calib",
               "b12_published_cosmomodel_damp", "b12_published_massless_nodamp", "b12_paint_cosmomodel_nodamp",
               "b12_paint_massless_damp", "lee22_paint_high_noclip",
               # precision / low-z diagnostics (run_class_sz_extra.py)
               "b12_paint_all_hiprec", "lee22_paint_low_hiprec", "b12_paint_zmin0p02", "b12_paint_zmin0p02_hiprec"):
        b = best(nm)
        if b is not None:                    # CS_ = best available precision (see run_class_sz_prec2.py)
            put("CS_" + nm, *cs(b)); PREC_USED[nm] = b
        if (CS / f"{nm}.npz").exists():      # CSdef_ = default precision of the 2026-09-25 dictionary
            put("CSdef_" + nm, *cs(nm))
    old = np.load(C.TRUNC_CLASS_SZ + "/class_sz_tsz_xout4.npz")
    fac = 2 * np.pi / (old["ell"] * (old["ell"] + 1))
    put("CS_truncation_series_xout4", old["ell"], old["dl_1h"] * fac, old["dl_2h"] * fac)

    # ---------------------------------------------------------------- Python halo model (Tinker08, CLASS-SZ ingredients)
    G = {(m, k): Grid(ing, m, k) for m in ("b12", "lee22") for k in ("sphere", "aperture")}
    ell_g = G[("b12", "sphere")].ell
    P = {}
    for k in ("sphere", "aperture"):
        gb, gl = G[("b12", k)], G[("lee22", k)]
        yh = hybrid_y(gl, gb)
        runs = {"b12": (gb, {}, None), "b12_low": (gb, dict(m_hi=CLIP), None), "b12_high": (gb, dict(m_lo=CLIP), None),
                "lee22": (gl, {}, None), "lee22_low": (gl, dict(m_hi=CLIP), None), "lee22_high": (gl, dict(m_lo=CLIP), None),
                "lee22_calib": (gl, dict(m_lo=C.LEE_CALIB_MMIN_HINV, m_hi=CLIP, z_hi=C.LEE_CALIB_ZMAX), None),
                "lee22m1": (gl, {}, gl.y * zfac(gl)), "lee22m1_low": (gl, dict(m_hi=CLIP), gl.y * zfac(gl)),
                "lee22m1_calib": (gl, dict(m_lo=C.LEE_CALIB_MMIN_HINV, m_hi=CLIP, z_hi=C.LEE_CALIB_ZMAX), gl.y * zfac(gl)),
                "b12_zmin0p02": (gb, dict(z_lo=0.02), None),
                "hybrid": (gl, {}, yh), "lee22m1_hybrid": (gl, {}, np.where((np.exp(gl.lnm) >= CLIP)[None, :, None], gb.y, gl.y * zfac(gl)))}
        for nm, (g, kw, y) in runs.items():
            r = g.cl(y=y, **kw)
            put(f"PY_{k}_{nm}", ell_g, r["cl_1h"], r["cl_2h"])
    # CLASS-SZ-anchored Lee22 totals: exact CLASS-SZ 1-halo (sum of the two exact pieces) + Python 2-halo of the full
    # range (contains the low x high cross term that separate CLASS-SZ runs cannot give), sphere x_out = 4 like CLASS-SZ
    for nm, lo, hi_ in (("lee22", "lee22_paint_low", "lee22_paint_high"), ("lee22m1", "lee22m1_paint_low", "lee22m1_paint_high")):
        T[f"CSX_{nm}_1h"] = T[f"CS_{lo}_1h"] + T[f"CS_{hi_}_1h"]
        T[f"CSX_{nm}_2h"] = T[f"PY_sphere_{nm}_2h"]
        T[f"CSX_{nm}"] = T[f"CSX_{nm}_1h"] + T[f"CSX_{nm}_2h"]
    T["CSX_hybrid_1h"] = T["CS_lee22_paint_low_1h"] + T["CS_b12_paint_high_1h"]
    T["CSX_hybrid_2h"] = T["PY_sphere_hybrid_2h"]
    T["CSX_hybrid"] = T["CSX_hybrid_1h"] + T["CSX_hybrid_2h"]

    # ---------------------------------------------------------------- catalogue-weighted (aperture, as painted)
    CG = {m: CatGrid(ing, m, "aperture") for m in ("b12", "lee22")}
    cb, cl_ = CG["b12"], CG["lee22"]
    zf = ((1 + cl_.z) ** 2)[:, None, None]
    yh_cat = np.where((np.exp(0.5 * (cl_.me[1:] + cl_.me[:-1])) >= CLIP)[None, :, None], cb.y, cl_.y)
    catruns = {"b12": (cb, {}, None), "b12_low": (cb, dict(m_hi=CLIP), None), "b12_high": (cb, dict(m_lo=CLIP), None),
               "lee22": (cl_, {}, None), "lee22_low": (cl_, dict(m_hi=CLIP), None), "lee22_high": (cl_, dict(m_lo=CLIP), None),
               "lee22_calib": (cl_, dict(m_lo=C.LEE_CALIB_MMIN_HINV, m_hi=CLIP, z_hi=C.LEE_CALIB_ZMAX), None),
               "lee22m1": (cl_, {}, cl_.y * zf), "lee22m1_calib": (cl_, dict(m_lo=C.LEE_CALIB_MMIN_HINV, m_hi=CLIP, z_hi=C.LEE_CALIB_ZMAX), cl_.y * zf),
               "hybrid": (cl_, {}, yh_cat)}
    for nm, (g, kw, y) in catruns.items():
        r = g.cl(y=y, **kw)
        put(f"CAT_{nm}", ell_g, r["cl_1h"], r["cl_2h"])

    # ---------------------------------------------------------------- band tables
    def row(cl_int):
        return {str(l): band(cl_int, l) for l in BANDS}
    bands = {"maps_raw": {k: row(v) for k, v in maps.items()},
             "maps_pixwin_deconvolved": {k: row(np.where(pw2 > 0, v / pw2, 0)) for k, v in maps.items()},
             "theory": {k: row(v) for k, v in T.items()}}
    def R(num, den):
        try:
            return {str(l): bands_get(num, l) / bands_get(den, l) for l in BANDS}
        except KeyError:            # a diagnostic run that is not on disk
            return None
    def bands_get(key, l):
        if key.startswith("map:"):
            return bands["maps_raw"][key[4:]][str(l)]
        if key.startswith("mapdec:"):
            return bands["maps_pixwin_deconvolved"][key[7:]][str(l)]
        return bands["theory"][key][str(l)]
    ratios = {
        # ---- B12
        "B12 map / CLASS-SZ (paint cosmology, 1h+2h, x_out=4)": R("map:B12", "CS_b12_paint_all"),
        "B12 map (pixwin-deconvolved) / CLASS-SZ": R("mapdec:B12", "CS_b12_paint_all"),
        "B12 map / Python T08 aperture (as painted)": R("map:B12", "PY_aperture_b12"),
        "B12 map / catalogue-weighted aperture (1h cat + 2h cat)": R("map:B12", "CAT_b12"),
        "Python T08 sphere / CLASS-SZ (B12 validation)": R("PY_sphere_b12", "CS_b12_paint_all"),
        "Python T08 aperture / sphere (B12)": R("PY_aperture_b12", "PY_sphere_b12"),
        "catalogue / T08, aperture B12 1h": R("CAT_b12_1h", "PY_aperture_b12_1h"),
        "catalogue / T08, aperture B12 (1h+2h)": R("CAT_b12", "PY_aperture_b12"),
        "CLASS-SZ 2026-09-25 run (published cosmology, 0.06 eV nu, 1h damping) / this paint-cosmology run, both default precision": R("CS_truncation_series_xout4", "CSdef_b12_paint_all"),
        "B12 map / CLASS-SZ 2026-09-25 run": R("map:B12", "CS_truncation_series_xout4"),
        "reproduction: this run of the 2026-09-25 dictionary / stored 2026-09-25 result": R("CSdef_b12_published_cosmomodel_damp", "CS_truncation_series_xout4"),
        "CLASS-SZ published massless / paint massless (default precision)": R("CSdef_b12_published_massless_nodamp", "CSdef_b12_paint_all"),
        "CLASS-SZ paint with cosmo_model (0.06 eV nu) / paint massless (default precision)": R("CSdef_b12_paint_cosmomodel_nodamp", "CSdef_b12_paint_all"),
        "CLASS-SZ paint 1h damping on / off (default precision)": R("CSdef_b12_paint_massless_damp", "CSdef_b12_paint_all"),
        "B12 low map / CLASS-SZ low": R("map:B12low", "CS_b12_paint_low"),
        "B12 high map / CLASS-SZ high": R("map:B12high", "CS_b12_paint_high"),
        # ---- Lee22
        "Lee22 map / CLASS-SZ Lee22 as painted (CS 1h low+high + PY 2h)": R("map:L22", "CSX_lee22"),
        "Lee22 map / Python T08 aperture": R("map:L22", "PY_aperture_lee22"),
        "Lee22 map / catalogue-weighted": R("map:L22", "CAT_lee22"),
        "Lee22 calib map / CLASS-SZ calib (exact single run)": R("map:L22calib", "CS_lee22_paint_calib"),
        "Lee22 calib map / catalogue-weighted calib": R("map:L22calib", "CAT_lee22_calib"),
        "Lee22 low map / CLASS-SZ low": R("map:L22low", "CS_lee22_paint_low"),
        "Lee22 high map / CLASS-SZ high": R("map:L22high", "CS_lee22_paint_high"),
        "Lee22 low map / catalogue low": R("map:L22low", "CAT_lee22_low"),
        "Lee22 high map / catalogue high": R("map:L22high", "CAT_lee22_high"),
        "Python sphere / CLASS-SZ, Lee22 low": R("PY_sphere_lee22_low", "CS_lee22_paint_low"),
        "Python sphere / CLASS-SZ, Lee22 high": R("PY_sphere_lee22_high", "CS_lee22_paint_high"),
        "Python sphere / CLASS-SZ, Lee22 calib": R("PY_sphere_lee22_calib", "CS_lee22_paint_calib"),
        "hybrid map / CLASS-SZ hybrid": R("map:hybrid", "CSX_hybrid"),
        "catalogue / T08, aperture Lee22 1h": R("CAT_lee22_1h", "PY_aperture_lee22_1h"),
        # ---- Lee22 / B12
        "Lee22/B12: maps": R("map:L22", "map:B12"),
        "Lee22/B12: CLASS-SZ": R("CSX_lee22", "CS_b12_paint_all"),
        "Lee22/B12: Python T08 aperture": R("PY_aperture_lee22", "PY_aperture_b12"),
        "Lee22/B12: catalogue-weighted": R("CAT_lee22", "CAT_b12"),
        "Lee22 calib / B12 all: maps": R("map:L22calib", "map:B12"),
        "Lee22 calib / B12 all: CLASS-SZ": R("CS_lee22_paint_calib", "CS_b12_paint_all"),
        "hybrid/B12: maps": R("map:hybrid", "map:B12"),
        "hybrid/B12: CLASS-SZ": R("CSX_hybrid", "CS_b12_paint_all"),
        # ---- M1 hypothesis
        "M1/as painted, all masses: CLASS-SZ": R("CSX_lee22m1", "CSX_lee22"),
        "M1/B12, all masses: CLASS-SZ": R("CSX_lee22m1", "CS_b12_paint_all"),
        "M1/as painted, calib: CLASS-SZ": R("CS_lee22m1_paint_calib", "CS_lee22_paint_calib"),
        "M1 calib / B12 all: CLASS-SZ": R("CS_lee22m1_paint_calib", "CS_b12_paint_all"),
        "M1/as painted, all masses: catalogue": R("CAT_lee22m1", "CAT_lee22"),
        "M1/B12, all masses: catalogue": R("CAT_lee22m1", "CAT_b12"),
        "M1 hybrid (M1 below clip, B12 above) / B12: Python T08 aperture": R("PY_aperture_lee22m1_hybrid", "PY_aperture_b12"),
        "M1 (predicted map) = Lee22 map x CLASS-SZ M1/as-painted / B12 map": None,
        # ---- no-clip diagnostic
        "Lee22 high, shape NOT clipped / clipped: CLASS-SZ (both default precision)": R("CSdef_lee22_paint_high_noclip", "CSdef_lee22_paint_high"),
        "2h share of CLASS-SZ B12": R("CS_b12_paint_all_2h", "CS_b12_paint_all"),
        "2h share of CLASS-SZ Lee22": R("CSX_lee22_2h", "CSX_lee22"),
        "Lee22 2h cross term (low x high) / Lee22 2h total, Python sphere": {
            str(l): 1 - (bands_get("PY_sphere_lee22_low_2h", l) + bands_get("PY_sphere_lee22_high_2h", l)) / bands_get("PY_sphere_lee22_2h", l)
            for l in BANDS},
    }
    # CLASS-SZ precision / low-z diagnostics (continuation, run_class_sz_extra.py)
    ratios.update({
        "CLASS-SZ hiprec / default precision, B12 total": R("CS_b12_paint_all_hiprec", "CSdef_b12_paint_all"),
        "CLASS-SZ hiprec / default precision, B12 1h": R("CS_b12_paint_all_hiprec_1h", "CSdef_b12_paint_all_1h"),
        "CLASS-SZ hiprec / default precision, Lee22 low total": R("CS_lee22_paint_low_hiprec", "CSdef_lee22_paint_low"),
        "CLASS-SZ best / default precision, Lee22 high total": R("CS_lee22_paint_high", "CSdef_lee22_paint_high"),
        "CLASS-SZ best / default precision, Lee22 calib total": R("CS_lee22_paint_calib", "CSdef_lee22_paint_calib"),
        "B12 map / CLASS-SZ at default precision": R("map:B12", "CSdef_b12_paint_all"),
        "validation 1h: Python sphere / CLASS-SZ hiprec, b12_paint_all": R("PY_sphere_b12_1h", "CS_b12_paint_all_hiprec_1h"),
        "validation 2h: Python sphere / CLASS-SZ hiprec, b12_paint_all": R("PY_sphere_b12_2h", "CS_b12_paint_all_hiprec_2h"),
        "validation total: Python sphere / CLASS-SZ default precision, b12_paint_all": R("PY_sphere_b12", "CSdef_b12_paint_all"),
        "validation total: Python sphere / CLASS-SZ hiprec, b12_paint_all": R("PY_sphere_b12", "CS_b12_paint_all_hiprec"),
        "validation 1h: Python sphere / CLASS-SZ hiprec, lee22_paint_low": R("PY_sphere_lee22_low_1h", "CS_lee22_paint_low_hiprec_1h"),
        "validation 1h: Python sphere / CLASS-SZ default precision, B12 z > 0.02": R("PY_sphere_b12_zmin0p02_1h", "CSdef_b12_paint_zmin0p02_1h"),
        "validation 1h: Python sphere / CLASS-SZ hiprec, B12 z > 0.02": R("PY_sphere_b12_zmin0p02_1h", "CS_b12_paint_zmin0p02_hiprec_1h"),
        "validation total: Python sphere / CLASS-SZ hiprec, B12 z > 0.02": R("PY_sphere_b12_zmin0p02", "CS_b12_paint_zmin0p02_hiprec"),
        "share of B12 power from z < 0.02 (Python sphere)": {str(l): 1 - bands_get("PY_sphere_b12_zmin0p02", l) / bands_get("PY_sphere_b12", l) for l in BANDS},
    })
    # Python-vs-CLASS-SZ validation, 1h and 2h separately, for every exactly representable CLASS-SZ run
    for py, csn in (("b12", "b12_paint_all"), ("b12_low", "b12_paint_low"), ("b12_high", "b12_paint_high"),
                    ("lee22_low", "lee22_paint_low"), ("lee22_high", "lee22_paint_high"), ("lee22_calib", "lee22_paint_calib"),
                    ("lee22m1_low", "lee22m1_paint_low"), ("lee22m1_calib", "lee22m1_paint_calib")):
        for t in ("1h", "2h"):
            ratios[f"validation {t}: Python sphere / CLASS-SZ (best precision), {csn}"] = R(f"PY_sphere_{py}_{t}", f"CS_{csn}_{t}")
            ratios[f"validation {t}: Python sphere / CLASS-SZ default precision, {csn}"] = R(f"PY_sphere_{py}_{t}", f"CSdef_{csn}_{t}")
    ratios["M1 (predicted map) = Lee22 map x CLASS-SZ M1/as-painted / B12 map"] = {
        str(l): bands_get("map:L22", l) * bands_get("CSX_lee22m1", l) / bands_get("CSX_lee22", l) / bands_get("map:B12", l)
        for l in BANDS}

    # ---------------------------------------------------------------- mass function offset (counts)
    gb = G[("b12", "aperture")]
    counts = []
    for i in range(len(M_EDGES) - 1):
        for j in range(len(Z_EDGES) - 1):
            nt = gb.counts(M_EDGES[i], M_EDGES[i + 1], Z_EDGES[j], Z_EDGES[j + 1])
            nc = cb.counts(M_EDGES[i], M_EDGES[i + 1], Z_EDGES[j] if j else 0.0, Z_EDGES[j + 1] if j < len(Z_EDGES) - 2 else 10.0)
            counts.append(dict(m_lo=M_EDGES[i], m_hi=M_EDGES[i + 1], z_lo=Z_EDGES[j], z_hi=Z_EDGES[j + 1],
                               N_catalogue=nc, N_tinker08=nt, ratio=nc / nt if nt > 0 else None))
    tot_c, tot_t = cb.counts(C.CAT_M_MIN, C.CAT_M_MAX, 0, 10), gb.counts(C.CAT_M_MIN, C.CAT_M_MAX, C.CAT_Z_MIN, C.CAT_Z_MAX)
    above = dict(N_catalogue=cb.counts(CLIP, C.CAT_M_MAX, 0, 10), N_tinker08=gb.counts(CLIP, C.CAT_M_MAX, C.CAT_Z_MIN, C.CAT_Z_MAX))

    # ---------------------------------------------------------------- decomposition at DECOMP_ELL
    def decomp(grid_b, grid_l, which, zlo_default):
        rb = grid_b.cl(keep=True); rl = grid_l.cl(keep=True)
        if which == "cat":
            mc = np.exp(0.5 * (grid_b.me[1:] + grid_b.me[:-1])); zc = 0.5 * (grid_b.ze[1:] + grid_b.ze[:-1])
        else:
            mc, zc = np.exp(grid_b.lnm), grid_b.z
        out = {}
        for L in DECOMP_ELL:
            i = int(np.argmin(np.abs(grid_b.ell - L)))
            tb = rb["cl_1h"][i] + rb["cl_2h"][i]; tl = rl["cl_1h"][i] + rl["cl_2h"][i]
            rowsm, rowsz = [], []
            for k in range(len(M_EDGES) - 1):
                s = (mc > M_EDGES[k]) & (mc < M_EDGES[k + 1])
                b1, b2 = rb["d1"][:, s, i].sum(), rb["d2"][:, s, i].sum()
                l1, l2 = rl["d1"][:, s, i].sum(), rl["d2"][:, s, i].sum()
                rowsm.append(dict(m_lo=M_EDGES[k], m_hi=M_EDGES[k + 1], frac_B12=(b1 + b2) / tb, frac_L22=(l1 + l2) / tl,
                                  L22_over_B12_bin=(l1 + l2) / (b1 + b2), L22_over_B12_bin_1h=l1 / b1,
                                  excess_share=((l1 + l2) - (b1 + b2)) / (tl - tb), contrib_to_ratio=((l1 + l2) - (b1 + b2)) / tb))
            ze = list(Z_EDGES); ze[0] = 0.0 if which == "cat" else ze[0]; ze[-1] = 10.0 if which == "cat" else ze[-1]
            for k in range(len(ze) - 1):
                s = (zc > ze[k]) & (zc < ze[k + 1])
                b1, b2 = rb["d1"][s, :, i].sum(), rb["d2"][s, :, i].sum()
                l1, l2 = rl["d1"][s, :, i].sum(), rl["d2"][s, :, i].sum()
                rowsz.append(dict(z_lo=Z_EDGES[k], z_hi=Z_EDGES[k + 1], frac_B12=(b1 + b2) / tb, frac_L22=(l1 + l2) / tl,
                                  L22_over_B12_bin=(l1 + l2) / (b1 + b2),
                                  excess_share=((l1 + l2) - (b1 + b2)) / (tl - tb), contrib_to_ratio=((l1 + l2) - (b1 + b2)) / tb))
            # joint (M >= clip) x (z < 0.1) share
            sm = mc >= CLIP; sz = zc < 0.1
            l_hl = (rl["d1"][np.ix_(sz, sm)][..., i].sum() + rl["d2"][np.ix_(sz, sm)][..., i].sum()) / tl
            # joint (mass bin x redshift bin) contribution to Lee22/B12 - 1 (1h + 2h attribution)
            joint = np.zeros((len(M_EDGES) - 1, len(ze) - 1))
            for a in range(len(M_EDGES) - 1):
                sa = (mc > M_EDGES[a]) & (mc < M_EDGES[a + 1])
                for bz in range(len(ze) - 1):
                    sb = (zc > ze[bz]) & (zc < ze[bz + 1])
                    ix = np.ix_(sb, sa)
                    joint[a, bz] = ((rl["d1"][ix][..., i].sum() + rl["d2"][ix][..., i].sum())
                                    - (rb["d1"][ix][..., i].sum() + rb["d2"][ix][..., i].sum())) / tb
            out_joint = dict(axes="[mass bin][z bin], (C_L22(bin) - C_B12(bin)) / C_B12(total)", matrix=joint.tolist())
            out[str(L)] = dict(ell=float(grid_b.ell[i]), ratio_total=tl / tb, ratio_1h=rl["cl_1h"][i] / rb["cl_1h"][i],
                               joint_contrib_to_ratio=out_joint,
                               two_halo_share_B12=rb["cl_2h"][i] / tb, two_halo_share_L22=rl["cl_2h"][i] / tl,
                               L22_share_M_above_clip_and_z_below_0p1=l_hl, mass=rowsm, redshift=rowsz)
        return out
    dec = {"python_T08_aperture": decomp(G[("b12", "aperture")], G[("lee22", "aperture")], "hm", None),
           "catalogue_weighted_aperture": decomp(cb, cl_, "cat", None)}
    # CLASS-SZ cross-check of the decomposition: 1h per mass bin (exact), 1h+2h per z bin (exact)
    csdec = {}
    for L in DECOMP_ELL:
        rowm, rowz = [], []
        tb1 = band(T["CS_b12_paint_all_1h"], L); tl1 = band(T["CSX_lee22_1h"], L)
        for k in range(len(M_EDGES) - 1):
            b = band(to_int(*cs(best(f"b12_paint_mbin{k}"))[:2]), L); l = band(to_int(*cs(best(f"lee22_paint_mbin{k}"))[:2]), L)
            rowm.append(dict(m_lo=M_EDGES[k], m_hi=M_EDGES[k + 1], frac_B12_1h=b / tb1, frac_L22_1h=l / tl1, L22_over_B12_bin_1h=l / b))
        tb = band(T["CS_b12_paint_all"], L)
        tl = band(T["CS_lee22_paint_low"], L) + band(T["CS_lee22_paint_high"], L)   # no cross 2h (per-z bins lack it too)
        for k in range(len(Z_EDGES) - 1):
            eb, c1, c2 = cs(best(f"b12_paint_zbin{k}")); b = band(to_int(eb, c1 + c2), L)
            el, a1, a2 = cs(best(f"lee22_paint_low_zbin{k}")); eh, h1, h2 = cs(best(f"lee22_paint_high_zbin{k}"))
            l = band(to_int(el, a1 + a2), L) + band(to_int(eh, h1 + h2), L)
            rowz.append(dict(z_lo=Z_EDGES[k], z_hi=Z_EDGES[k + 1], frac_B12=b / tb, frac_L22_nocross=l / tl, L22_over_B12_bin=l / b))
        csdec[str(L)] = dict(mass_1h=rowm, redshift=rowz, sum_mass_bins_over_total_B12_1h=sum(r["frac_B12_1h"] for r in rowm),
                             sum_z_bins_over_total_B12=sum(r["frac_B12"] for r in rowz))
    dec["class_sz_bins"] = csdec

    res = dict(bands=bands, ratios=ratios, counts=dict(bins=counts, total_catalogue=tot_c, total_tinker08=tot_t,
                                                       total_ratio=tot_c / tot_t, above_clip=above),
               decomposition=dec, pixwin2={str(l): float(pw2[l]) for l in BANDS},
               class_sz_precision_used=PREC_USED, class_sz_glitches_replaced=GLITCHES,
               notes=dict(maps="raw = map2alm spectra as stored (pixel window not deconvolved); maps are point samples",
                          CSX="CLASS-SZ 1h (low + high exact pieces) + Python 2h (full range incl. cross term)",
                          PY="Python halo model with CLASS-SZ Tinker08/Tinker10/P_lin ingredients",
                          CAT="catalogue-weighted: 1h = sum over the 85.2M halos of |y_ell|^2/4pi; 2h = binned catalogue "
                              "density x Tinker10 bias x P_lin", mass_units="Msun/h, h = 0.68"))
    json.dump(res, open(AND / "results.json", "w"), indent=1, default=float)
    np.savez(AND / "curves.npz", ell=LMAP, pixwin2=pw2, **{"map_" + k: v for k, v in maps.items()}, **{"th_" + k: v for k, v in T.items()})
    # console summary
    for k, v in ratios.items():
        if v is None:
            print(f"{k:95s} (missing input)"); continue
        print(f"{k:95s}", "  ".join(f"{l}:{v[str(l)]:.3f}" for l in BANDS))
    print("counts total cat/T08", tot_c / tot_t, "above clip", above)


if __name__ == "__main__":
    main()
