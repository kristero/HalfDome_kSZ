"""Quantitative illustrations for the methods survey ('Ways to obtain 1D stacked profiles').

Reads only the existing particle-pass products of the study (read-only) and writes
methods_numbers.json / methods_numbers.npz next to this script.

Sections
  A  stacking statistic: inclusion-weighted mean of normalised profiles (reference) vs median, log-mean,
     M200-weighted ("aggregate particle stack in scaled units"), M^(5/3)-weighted (raw aggregate
     sum y / sum V P200) and measured-Y200-weighted (Battaglia+12) stacks; paired halo bootstrap.
  B  SPH-volume filling sum(m/rho)/V_shell (why the B24 Q_V estimator reads high).
  C  particle-centre binning vs SPH-kernel deposition onto the same shells (Wendland C2, H = 1.936492 h),
     estimated from the per-shell smoothing-length histograms.
  D  projection: projected (2D) profile vs line-of-sight depth; onion-peeling deprojection bias when the
     outer boundary is set inside the true gas extent.
  E  radius/normalisation: mass trend of the bin-mean profile at fixed scaled radius, R200c/P200 vs
     R500c/P_e,500 (B24), and within-bin scatter sd(ln f).

Run:  LD_LIBRARY_PATH=<halfdome env>/lib python -B methods_numbers.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

STUDY = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
from analysis import newdata, variants  # noqa: E402

T0 = time.time()
RNG = np.random.default_rng(20261014)
NBOOT = 300
GAMMA_K = 1.936492          # SWIFT Wendland C2 (3D): compact support H = gamma h

d = newdata.load()
G = d.mod.G
X200 = d.grid("x200")
X500 = d.grid("x500")
w_all = d.inclusion_weights()
logm = np.log10(d.mass("x200"))
m200 = d.mass("x200")
y200 = np.asarray(d.halos["y_noagn_200_mpc2"], float)


def idx_of(grid, values):
    return np.array([int(np.argmin(np.abs(grid - v))) for v in values])


COARSE = [0.025, 0.04, 0.0591, 0.0873, 0.1289, 0.1905, 0.2814, 0.4157, 0.6141, 1.0, 1.34, 1.75, 2.0, 2.5, 3.0,
          3.6342, 4.0]
CI = idx_of(X200, COARSE)
assert np.allclose(X200[CI], COARSE, atol=6e-5), X200[CI]
CEDGES = X200[CI]
COARSE_GRID = ("x200", CEDGES)
CMID = np.sqrt(CEDGES[:-1] * CEDGES[1:])

BINS = [(12.9, 13.1), (13.5, 13.7), (13.9, 14.1), (14.3, 14.5), (14.7, 14.9), (14.9, 15.5)]


def rows_in(lo, hi):
    return np.flatnonzero((logm >= lo - 1e-9) & (logm < hi - 1e-9))


def wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


# ------------------------------------------------------------------------------------------- profiles
t = time.time()
prof = {
    "pressure": variants.profile(d, "pressure", "geometric", agn="removed", grid=COARSE_GRID).values,
    "electrons": variants.profile(d, "electrons", "geometric", agn="removed", grid=COARSE_GRID).values,
}
pq = variants.profile(d, "pressure", "Q_V", agn="removed", grid=COARSE_GRID)
vol_geo = variants._volume(d, "x200", CEDGES, None)
filling = pq.sums["volume"] / vol_geo
count_c = d.rebin(d.array("owner200")[:, :, :, d.index("quantities", "count"), :].sum(axis=(1, 2)), CI)
print(f"profiles {time.time() - t:.1f}s", flush=True)

res_mask = {}
for lo, hi in BINS:
    r = rows_in(lo, hi)
    m = variants.resolution(d, r, grid=COARSE_GRID)
    res_mask[(lo, hi)] = {"strict": m["strict"], "base": m["base"], "supported": m["supported"],
                          "resolved_fraction": m["resolved_fraction"]}

# ------------------------------------------------------------------------------------------- A stacking statistic
STATS = ("median", "logmean", "m_weighted", "m53_weighted", "y200_weighted")


def stack_stats(f, w, m, y):
    out = {}
    mean = np.sum(w[:, None] * f, 0) / np.sum(w)
    out["mean"] = mean
    out["median"] = np.array([wmedian(f[:, s], w) for s in range(f.shape[1])])
    pos = f > 0
    lm = np.full(f.shape[1], np.nan)
    for s in range(f.shape[1]):
        k = pos[:, s]
        if k.sum() >= 3:
            lm[s] = np.exp(np.sum(w[k] * np.log(f[k, s])) / np.sum(w[k]))
    out["logmean"] = lm
    out["zero_fraction"] = np.sum(w[:, None] * (~pos), 0) / np.sum(w)
    out["m_weighted"] = np.sum((w * m)[:, None] * f, 0) / np.sum(w * m)
    out["m53_weighted"] = np.sum((w * m ** (5 / 3))[:, None] * f, 0) / np.sum(w * m ** (5 / 3))
    out["y200_weighted"] = np.sum((w * y)[:, None] * f, 0) / np.sum(w * y)
    return out


A = {}
for field in ("pressure", "electrons"):
    A[field] = {}
    for lo, hi in BINS:
        r = rows_in(lo, hi)
        f, w, m, y = prof[field][r], w_all[r], m200[r], y200[r]
        base = stack_stats(f, w, m, y)
        ratios = {s: base[s] / base["mean"] for s in STATS}
        boots = {s: [] for s in STATS}
        mean_boot = []
        for _ in range(NBOOT):
            b = RNG.integers(0, len(r), len(r))
            st = stack_stats(f[b], w[b], m[b], y[b])
            mean_boot.append(st["mean"] / base["mean"])
            for s in STATS:
                boots[s].append(st[s] / st["mean"])
        entry = {"n_haloes": int(len(r)), "edges": CEDGES.tolist(),
                 "resolved_strict": res_mask[(lo, hi)]["strict"].tolist(),
                 "resolved_fraction": np.round(res_mask[(lo, hi)]["resolved_fraction"], 4).tolist(),
                 "mean": base["mean"].tolist(), "zero_fraction": np.round(base["zero_fraction"], 4).tolist(),
                 "mean_rel_err_16_84": np.round(np.nanpercentile(np.array(mean_boot), [16, 84], 0) - 1, 4).tolist()}
        for s in STATS:
            arr = np.array(boots[s])
            entry[s] = {"ratio_to_mean": np.round(ratios[s], 4).tolist(),
                        "p16": np.round(np.nanpercentile(arr, 16, 0), 4).tolist(),
                        "p84": np.round(np.nanpercentile(arr, 84, 0), 4).tolist()}
        A[field][f"{lo}-{hi}"] = entry
print(f"A {time.time() - t:.1f}s", flush=True)

# ------------------------------------------------------------------------------------------- B SPH volume filling
B = {}
for lo, hi in BINS:
    r = rows_in(lo, hi)
    w = w_all[r]
    agg = np.sum(w[:, None] * pq.sums["volume"][r], 0) / np.sum(w[:, None] * vol_geo[r], 0)
    B[f"{lo}-{hi}"] = {"filling_aggregate": np.round(agg, 4).tolist(),
                       "filling_median": np.round([wmedian(filling[r, s], w) for s in range(len(CMID))], 4).tolist()}

# ------------------------------------------------------------------------------------------- C kernel deposition
t = time.time()
hs = d.array("hs200")                                   # [n, 67, 384] u4, h a / R200c
hs_edges = np.asarray(d.array("hs_edges"), float)       # 385
FAC = 4
hsc = hs.reshape(hs.shape[0], hs.shape[1], -1, FAC).sum(-1).astype(np.float64)   # [n, 67, 96]
hce = hs_edges[::FAC]
hcen = np.sqrt(hce[:-1] * hce[1:])
NB = len(hcen)
cnt_all = d.array("owner200")[:, :, :, d.index("quantities", "count"), :].sum(axis=(1, 2))   # [n, 67]
hist_tot = hsc.sum(-1)
mismatch = float(np.max(np.abs(hist_tot - cnt_all)))
print("hist vs count max abs diff", mismatch, flush=True)

# transfer matrix T[s, b, t]: kernel-mass fraction of a particle in source shell s (positions at J
# volume quantiles) with support H_b deposited into target shell t (X200 grid); last column = beyond 4 R200c
J = 6
un, uw = leggauss(64)
u = 0.5 * (un + 1.0)
uwt = 0.5 * uw * 4 * np.pi * u ** 2 * 21 / (2 * np.pi) * (1 - u) ** 4 * (1 + 4 * u)
assert abs(uwt.sum() - 1) < 1e-10
NS = len(X200) - 1
T = np.zeros((NS, NB, NS + 1))
for s in range(NS):
    a3, b3 = X200[s] ** 3, X200[s + 1] ** 3
    r0 = (a3 + (b3 - a3) * (np.arange(J) + 0.5) / J) ** (1 / 3)        # [J]
    H = GAMMA_K * hcen                                                  # [NB]
    dist = u[None, None, :] * H[None, :, None]                         # [1, NB, U]
    rr = r0[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        Pin = np.clip((X200[None, None, None, :] ** 2 - (rr - dist)[..., None] ** 2)
                      / (4 * rr[..., None] * dist[..., None]), 0, 1)   # [J, NB, U, 68]
    frac_lt = np.einsum("jbue,u->jbe", Pin, uwt)                        # [J, NB, 68]
    ft = np.diff(frac_lt, axis=-1)                                      # [J, NB, 67]
    out = 1 - frac_lt[..., -1]
    T[s, :, :NS] = ft.mean(0)
    T[s, :, NS] = out.mean(0)
cons = float(np.max(np.abs(T.sum(-1) - 1)))
print(f"transfer matrix {time.time() - t:.1f}s, conservation {cons:.2e}", flush=True)
Tm = T.reshape(NS * NB, NS + 1)

C = {}
qnames = {"pressure": "y", "electrons": "electrons"}
dep = {}
for field, qn in qnames.items():
    lin = variants.owner_sum(d, "x200", qn, variants.selection("all"), "removed")       # [n, 67]
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(hist_tot > 0, lin / np.where(hist_tot > 0, hist_tot, 1), 0.0)    # per particle
    D = np.zeros((d.n, NS + 1))
    for c0 in range(0, d.n, 400):
        sl = slice(c0, min(c0 + 400, d.n))
        Amat = (per[sl, :, None] * hsc[sl]).reshape(-1, NS * NB)
        D[sl] = Amat @ Tm
    dep[field] = D
    lost = D[:, NS].sum() / max(lin.sum(), 1e-300)
    # coarse shells: linear sums, then stack ratio (normalisation identical per halo)
    lin_c = d.rebin(lin, CI)
    dep_c = d.rebin(D[:, :NS], CI)
    # normalised profile f = lin / (V_shell N_h): the per-halo, per-shell scale is 1 / (V_shell N_h)
    norm_h = d.p200() if field == "pressure" else d.rho_c("x200")
    kscale = 1.0 / (vol_geo * norm_h[:, None])
    C[field] = {"fraction_deposited_beyond_4R": float(lost)}
    for lo, hi in BINS:
        r = rows_in(lo, hi)
        w = w_all[r]
        num_dep = np.sum(w[:, None] * kscale[r] * dep_c[r], 0)
        num_pc = np.sum(w[:, None] * kscale[r] * lin_c[r], 0)
        ratio = num_dep / num_pc
        # fraction of each shell's particles whose kernel support H exceeds the shell's inner radius
        hist_c = np.moveaxis(d.rebin(np.moveaxis(hsc[r], -1, -2), CI), -1, -2)          # [nr, 16, NB]
        big = (GAMMA_K * hcen[None, :] > CEDGES[:-1, None])                             # [16, NB]
        frac_big = np.sum(w[:, None] * (hist_c * big[None]).sum(-1), 0) / np.maximum(
            np.sum(w[:, None] * hist_c.sum(-1), 0), 1e-300)
        # weighted median h/R200c per coarse shell (pooled particles)
        pooled = np.sum(w[:, None, None] * hist_c, 0)                                    # [16, NB]
        cum = np.cumsum(pooled, -1)
        medh = np.array([hcen[np.searchsorted(cum[s], 0.5 * cum[s, -1])] if cum[s, -1] > 0 else np.nan
                         for s in range(len(CMID))])
        C[field][f"{lo}-{hi}"] = {"deposited_over_particle_centre": np.round(ratio, 4).tolist(),
                                  "fraction_particles_H_gt_inner_edge": np.round(frac_big, 4).tolist(),
                                  "median_h_over_R200c": np.round(medh, 5).tolist()}
del hs, hsc
print(f"C {time.time() - t:.1f}s", flush=True)


# ------------------------------------------------------------------------------------------- D projection
def chord(R, a, b, D):
    """LOS length within |l| < D of the shell a < r < b at projected radius R."""
    la = np.sqrt(np.maximum(a ** 2 - R ** 2, 0))
    lb = np.sqrt(np.maximum(b ** 2 - R ** 2, 0))
    return 2 * (np.minimum(lb, D) - np.minimum(la, D))


def vcb(Rc, r):
    """Volume of the ball of radius r inside the infinite cylinder of radius Rc."""
    return 4 * np.pi / 3 * (r ** 3 - np.maximum(r ** 2 - Rc ** 2, 0) ** 1.5)


Dres = {}
FULL = {}
for field in ("pressure", "electrons"):
    pf = variants.profile(d, field, "geometric", agn="removed", grid="x200").values      # [n, 67]
    FULL[field] = pf
    Dres[field] = {}
    for lo, hi in BINS[1:]:
        r = rows_in(lo, hi)
        w = w_all[r]
        f = np.sum(w[:, None] * pf[r], 0) / np.sum(w)                                    # [67]
        Rs = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0])
        a, b = X200[:-1], X200[1:]
        proj = {}
        for Dd in (1.0, 2.0, 3.0, 4.0):
            proj[Dd] = np.array([np.sum(f * chord(R, a, b, Dd)) for R in Rs])
        frac_out1 = np.array([np.sum(f * chord(R, a, b, 4.0) * (a >= 1.0)) for R in Rs]) / proj[4.0]
        frac_out2 = np.array([np.sum(f * chord(R, a, b, 4.0) * (a >= 2.0)) for R in Rs]) / proj[4.0]
        # onion peeling on the fine X200 grid with outer boundary X_out (annuli = shells up to X_out);
        # the "observed" annulus sums contain the true gas to 4 R200c; X_out = 4 is the exact baseline
        onion = {}
        vsh = 4 * np.pi / 3 * (b ** 3 - a ** 3)
        for xout in (1.34, 2.0, 3.0, 4.0):
            na = int(np.flatnonzero(np.isclose(X200, xout, atol=1e-4))[0])
            e = X200[:na + 1]
            obs = np.array([np.sum(((vcb(e[i + 1], b) - vcb(e[i], b)) - (vcb(e[i + 1], a) - vcb(e[i], a))) * f)
                            for i in range(na)])
            M = np.zeros((na, na))
            for i in range(na):
                M[i, :] = (vcb(e[i + 1], e[1:]) - vcb(e[i], e[1:])) - (vcb(e[i + 1], e[:-1]) - vcb(e[i], e[:-1]))
            frec = np.linalg.solve(M, obs)
            ce = CEDGES[CEDGES <= xout + 1e-9]
            rec_c, tru_c = [], []
            for j in range(len(ce) - 1):
                k = (a[:na] >= ce[j] - 1e-9) & (b[:na] <= ce[j + 1] + 1e-9)
                rec_c.append(np.sum(frec[k] * vsh[:na][k]) / np.sum(vsh[:na][k]))
                tru_c.append(np.sum(f[:na][k] * vsh[:na][k]) / np.sum(vsh[:na][k]))
            onion[f"Xout_{xout}"] = {"coarse_edges": ce.tolist(),
                                     "recovered_over_true": np.round(np.array(rec_c) / np.array(tru_c), 4).tolist()}
        # discretisation-only error: onion peeling on the 16 coarse annuli (piecewise-constant shells), no truncation
        ce = CEDGES
        nc = len(ce) - 1
        obs_c = np.array([np.sum(((vcb(ce[i + 1], b) - vcb(ce[i], b)) - (vcb(ce[i + 1], a) - vcb(ce[i], a))) * f)
                          for i in range(nc)])
        Mc = np.zeros((nc, nc))
        for i in range(nc):
            Mc[i, :] = (vcb(ce[i + 1], ce[1:]) - vcb(ce[i], ce[1:])) - (vcb(ce[i + 1], ce[:-1]) - vcb(ce[i], ce[:-1]))
        # coarse grid starts at 0.025: add the inner 0-0.025 gas to the first annulus model as part of shell 0
        frec_c = np.linalg.solve(Mc, obs_c)
        tru_c2 = np.array([np.sum((f * vsh)[(a >= ce[j] - 1e-9) & (b <= ce[j + 1] + 1e-9)])
                           / np.sum(vsh[(a >= ce[j] - 1e-9) & (b <= ce[j + 1] + 1e-9)]) for j in range(nc)])
        onion["coarse16_Xout_4.0_discretisation_only"] = {"coarse_edges": ce.tolist(),
                                                         "recovered_over_true": np.round(frec_c / tru_c2, 4).tolist()}
        # cylinder vs sphere integrals (LOS through the 4 R200c sphere)
        cyl_sph = {}
        for Rc in (0.65, 1.0, 2.0):
            ycyl = np.sum((vcb(Rc, b) - vcb(Rc, a)) * f)
            ysph = np.sum(4 * np.pi / 3 * (np.minimum(b, Rc) ** 3 - np.minimum(a, Rc) ** 3) * f)
            cyl_sph[f"R{Rc}"] = round(float(ycyl / ysph), 4)
        Dres[field][f"{lo}-{hi}"] = {
            "R": Rs.tolist(),
            "depth_over_full4R": {f"D{Dd}": np.round(proj[Dd] / proj[4.0], 4).tolist() for Dd in (1.0, 2.0, 3.0)},
            "projected_fraction_from_r_gt_1R": np.round(frac_out1, 4).tolist(),
            "projected_fraction_from_r_gt_2R": np.round(frac_out2, 4).tolist(),
            "onion_peeling": onion, "cylinder_over_sphere": cyl_sph}
print(f"D {time.time() - t:.1f}s", flush=True)

# ------------------------------------------------------------------------------------------- E radius scaling
E = {}
X500_PICK = [(0.3705, 0.4481), (0.7927, 0.9587), (1.4023, 1.6959), (2.4806, 3.0)]
X200_PICK = [(0.2335, 0.2814), (0.5185, 0.6141), (0.9071, 1.1516), (1.5845, 1.8587)]
fit_bins = [(lo, lo + 0.2) for lo in np.arange(13.5, 14.95, 0.2)] + [(14.9, 15.5)]
for field in ("pressure", "electrons"):
    E[field] = {}
    p500 = variants.profile(d, field, "geometric", agn="removed", grid=("x500", G.X500[idx_of(G.X500, [0] + sum(map(list, X500_PICK), []))]),
                            normalisation="B24").values
    p200 = variants.profile(d, field, "geometric", agn="removed", grid=("x200", G.X200[idx_of(G.X200, [0] + sum(map(list, X200_PICK), []))]),
                            normalisation="P200").values
    for name, pv, picks in (("R200c_P200", p200, X200_PICK), ("R500c_B24", p500, X500_PICK)):
        cols = [1 + 2 * i for i in range(len(picks))]      # shells between picked edge pairs
        means, lms, sds = [], [], []
        for lo, hi in fit_bins:
            r = rows_in(lo, hi)
            w = w_all[r]
            v = pv[r][:, cols]
            means.append(np.sum(w[:, None] * v, 0) / np.sum(w))
            lms.append(np.average(np.log10(m200[r]), weights=w))
            sd = []
            for c in range(v.shape[1]):
                k = v[:, c] > 0
                lv = np.log(v[k, c])
                mu = np.average(lv, weights=w[k])
                sd.append(float(np.sqrt(np.average((lv - mu) ** 2, weights=w[k]))))
            sds.append(sd)
        means, lms, sds = np.array(means), np.array(lms), np.array(sds)
        slopes = [float(np.polyfit(lms, np.log(means[:, c]), 1)[0]) for c in range(means.shape[1])]
        E[field][name] = {"shells": [list(p) for p in picks],
                          "dln_mean_dlog10M": np.round(slopes, 3).tolist(),
                          "max_over_min_bin_mean": np.round(means.max(0) / means.min(0), 3).tolist(),
                          "median_within_bin_sd_ln": np.round(np.median(sds, 0), 3).tolist()}
print(f"E {time.time() - t:.1f}s", flush=True)

# ------------------------------------------------------------------------------------------- F particle budget
Fb = {}
i1 = int(np.flatnonzero(np.isclose(X200, 1.0))[0])
i2 = int(np.flatnonzero(np.isclose(X200, 2.0))[0])
cnt_c = d.rebin(cnt_all, CI)
for lo, hi in BINS:
    r = rows_in(lo, hi)
    w = w_all[r]
    n1 = cnt_all[r, :i1].sum(1)
    n2 = cnt_all[r, :i2].sum(1)
    n4 = cnt_all[r].sum(1)
    Fb[f"{lo}-{hi}"] = {
        "median_Ngas_lt_R200c": float(wmedian(n1.astype(float), w)),
        "median_Ngas_lt_2R200c": float(wmedian(n2.astype(float), w)),
        "median_Ngas_lt_4R200c": float(wmedian(n4.astype(float), w)),
        "median_count_per_coarse_shell": [float(wmedian(cnt_c[r, s].astype(float), w)) for s in range(len(CMID))],
        "fraction_haloes_with_ge20_in_coarse_shell": np.round(np.sum(w[:, None] * (cnt_c[r] >= 20), 0) / np.sum(w), 3).tolist(),
        "fraction_haloes_with_ge20_in_coarse_shell_if_1pct_downsampled_expected": np.round(
            np.sum(w[:, None] * (cnt_c[r] / 100.0 >= 20), 0) / np.sum(w), 3).tolist(),
    }

# ------------------------------------------------------------------------------------------- G lightcone pixel scale (estimate)
om, h = 0.306, 0.681          # FLAMINGO D3A-like cosmology; estimate only
zz = np.linspace(0, 1, 20001)
Hz = 100 * h * np.sqrt(om * (1 + zz) ** 3 + 1 - om)
chi = np.concatenate([[0], np.cumsum(0.5 * (1 / Hz[1:] + 1 / Hz[:-1]) * np.diff(zz))]) * 299792.458
Gpix = {}
for nside in (4096, 8192, 16384):
    pix_arcmin = np.sqrt(4 * np.pi / (12 * nside ** 2)) * 180 / np.pi * 60
    row = {"pixel_arcmin": round(float(pix_arcmin), 3)}
    for z in (0.05, 0.1, 0.3, 0.5, 1.0):
        da = np.interp(z, zz, chi) / (1 + z)
        row[f"pixel_kpc_z{z}"] = round(float(da * pix_arcmin / 60 * np.pi / 180 * 1e3), 1)
    Gpix[f"nside_{nside}"] = row

# ------------------------------------------------------------------------------------------- write
result = {
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "inputs": "study measurements_owner.npz, selected_halos.npz (read-only); FLAMINGO L1_m9 HYDRO_FIDUCIAL z = 0",
    "conventions": {
        "reference": "inclusion-weighted mean of per-halo normalised profiles, geometric estimator, all gas, recent-AGN gas removed",
        "pressure_norm": "P200 = G M200c 200 rho_c f_b / (2 R200c)",
        "electron_norm": "n_e mu_XGP / (rho_c f_b) (XGPaint units)",
        "aggregate_note": "an aggregate particle stack in scaled units (sum of P/P200 over all haloes' particles / sum of scaled shell volumes) equals the M200c-weighted mean; the raw aggregate sum(y)/sum(V P200) equals the M^(5/3)-weighted mean",
        "kernel": f"Wendland C2, H = {GAMMA_K} h (SWIFT 3D), particles placed at {J} volume quantiles of their shell, per-particle value = shell mean, h from the stored per-shell histograms (coarsened x{FAC})",
        "bootstrap": f"{NBOOT} paired halo resamples per bin",
    },
    "coarse_edges_R200c": CEDGES.tolist(),
    "A_stacking_statistic": A,
    "B_sph_volume_filling": B,
    "C_kernel_deposition": C,
    "C_checks": {"histogram_vs_count_max_abs_diff": mismatch, "transfer_conservation_max_abs": cons},
    "D_projection": Dres,
    "E_radius_scaling": E,
    "F_particle_budget": Fb,
    "G_lightcone_pixel_scale_estimate": {"cosmology": "Omega_m 0.306, h 0.681 (flat; estimate)", "rows": Gpix},
    "runtime_s": round(time.time() - T0, 1),
}
(OUT / "methods_numbers.json").write_text(json.dumps(result, indent=1))
np.savez_compressed(OUT / "methods_numbers.npz", coarse_edges=CEDGES, transfer_matrix=T.astype(np.float32),
                    h_centres=hcen, x200=X200)
print(f"done {time.time() - T0:.1f}s", flush=True)
