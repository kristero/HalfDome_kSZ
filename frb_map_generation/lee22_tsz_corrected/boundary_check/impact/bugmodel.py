"""Shared pieces for the Lee22 boundary-bug impact test (2026-10-08).

The bug (group_particles c3fe9b8, detail/workspace_sorting.hpp PER() macro, signed/unsigned comparison) drops every
(group, particle) pair whose minimum image lies across a LOW face (x=0, y=0 or z=0) of the box; nothing else
changes.  Equivalent pair rule, used by the Python replica: drop iff for any axis x_prt - x_grp > L/2 (raw box
coordinates, a particle stored at exactly x = L counts as L, as in the C++ cell grid which clamps it into cell 63).

Lee22 binning (examples/y_prof.cpp, ne_prof.cpp): 128 shells, innermost from the origin to 0.03 R200c, then 127
log shells up to 2.5 R200c; logRmin, logRmax, dlogR and the bin index are float32 (coord_t), shell volumes double.
"""
import numpy as np

NB = 128
L_TNG300 = 205000.0          # ckpc/h
XH = 0.76
GAMMA = 5.0 / 3.0
MPROT_NE_PROF = 0.6774 * 1.6726219 / 1.98847 * 1e-67   # ne_prof.cpp (h hard-coded), 1e10 Msun/h


def shift_positions(x32, shift, L):
    """float32 positions after a periodic translation; identical to make_shift_proxy.shifted (unchanged if shift=0)."""
    x32 = np.asarray(x32, np.float32)
    shift = np.asarray(shift, float)
    if not np.any(shift):
        return x32
    return np.mod(x32.astype(np.float64) + shift, L).astype(np.float32)


def lee22_bins(R200_f32):
    """float32 logRmin, logRmax, dlogR (as in the YProfile/NeProfile constructors) and double shell volumes."""
    R = np.float32(R200_f32)
    logRmin = np.log(np.float32(0.03) * R)
    logRmax = np.log(np.float32(2.50) * R)
    dlogR = np.float32((logRmax - logRmin) / np.float32(NB - 1))
    k = np.arange(NB)
    lr1 = np.float64(logRmin) + np.float64(dlogR) * (k - 1.0)
    lr2 = np.float64(logRmin) + np.float64(dlogR) * k
    vol = 4.0 * np.pi / 3.0 * (np.exp(3.0 * lr2) - np.exp(3.0 * lr1))
    vol[0] = 4.0 * np.pi / 3.0 * np.exp(3.0 * np.float64(logRmin))
    r_lo = np.where(k == 0, 0.0, np.exp(lr1))
    r_hi = np.exp(lr2)
    r_hi[0] = np.exp(np.float64(logRmin))
    return logRmin, logRmax, dlogR, vol, r_lo, r_hi


def bin_index(Rsq32, logRmin, logRmax, dlogR):
    """bin index as in prt_insert (float32); -1 where the particle is not inserted."""
    logR = np.float32(0.5) * np.log(np.asarray(Rsq32, np.float32))
    k = np.where(logR < logRmin, 0, 1 + np.floor((logR - logRmin) / dlogR).astype(np.int64))
    k = np.where(logR > logRmax, -1, k)
    k = np.where(k >= NB, -1, k)
    return k


def x_edges_nominal():
    """bin edges in x = r/R200c (double, nominal; bin 0 = [0, 0.03])"""
    lr = np.log(0.03) + (np.log(2.5) - np.log(0.03)) / (NB - 1) * np.arange(NB)
    return np.concatenate([[0.0], np.exp(lr)])


def fibonacci_dirs(n=20000):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


DIRS = fibonacci_dirs()


def iso_expected_kept(r, L, dirs=DIRS):
    """E over uniformly random box translations of the kept fraction of an isotropic thin shell of radius r:
    E_dir[prod_i (1 - max(0, -r n_i)/L)] (exact for r < L/2)."""
    r = np.atleast_1d(np.asarray(r, float))
    neg = np.maximum(0.0, -dirs)                                    # (ndir, 3)
    out = np.empty(len(r))
    for j, rr in enumerate(r):
        out[j] = np.mean(np.prod(1.0 - rr * neg / L, axis=1))
    return out


def iso_cap_kept(c, r, L, dirs=DIRS):
    """fraction of the sphere of radius r around c (box coordinates) NOT across a low face (actual geometry)."""
    c = np.asarray(c, float)
    r = np.atleast_1d(np.asarray(r, float))
    out = np.ones(len(r))
    dlow = np.min(c)
    cut = r > dlow
    if cut.any():
        p = c[None, None, :] + r[cut, None, None] * dirs[None, :, :]
        out[cut] = np.all(p >= 0.0, axis=2).mean(1)
    return out


def shell_average(fun, r_lo, r_hi, nsub=6):
    """r^2 dr-weighted (volume) average of fun(r) over each shell [r_lo, r_hi] (log sub-sampling)."""
    lo = np.maximum(r_lo, 1e-3 * r_hi)
    t = (np.arange(nsub) + 0.5) / nsub
    rs = np.exp(np.log(lo)[:, None] + (np.log(r_hi) - np.log(lo))[:, None] * t[None, :])
    f = fun(rs.ravel()).reshape(rs.shape)
    w = rs ** 3
    return (f * w).sum(1) / w.sum(1)
