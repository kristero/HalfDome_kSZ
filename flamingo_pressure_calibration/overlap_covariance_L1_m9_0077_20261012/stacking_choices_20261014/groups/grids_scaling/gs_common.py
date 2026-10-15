"""Shared helpers of the 'grids_scaling' group (radial grid, radius / normalisation scaling, radial range).

Only reads the stacking core (core/stackcore.py) and the cache; writes nothing by itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GROUP = "grids_scaling"
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]                       # stacking_choices_20261014
STUDY = OUT.parent
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

N_BOOT = 200
SCHEME = "M200c_0.2"
BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))
BAND_LABELS = tuple(f"{a:g}-{b:g}" for a, b in BANDS)
OBS_P = ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "y_mean")
OBS_E = ("kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var")
RES = dict(rule="strict", min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04)

# physical constants (the study's pilot_physics values)
SIGMA_T = 6.6524587321e-29
M_E = 9.1093837015e-31
C_LIGHT = 299792458.0
M_P = 1.67262192369e-27
MPC_M = 3.08567758e22
MSUN_KG = 1.98841e30
PE_FACTOR = M_E * C_LIGHT**2 / SIGMA_T / MPC_M       # Pa per (Mpc^2 of y / Mpc^3)
MU_XGP_MSUN = (M_E + M_P * 2 / 1.76) / 0.9 / MSUN_KG


def xcentre(edges):
    """Volume-weighted shell radius 0.75 (b^4 - a^4) / (b^3 - a^3)."""
    e = np.asarray(edges, float)
    a, b = e[:-1], e[1:]
    return 0.75 * (b**4 - a**4) / (b**3 - a**3)


def bins(scheme=SCHEME):
    return sc.mass_bins(scheme)


def calibrated(bs):
    return [b for b in bs if b["lo"] >= 13.5 - 1e-9]


def seed_of(b):
    return sc.STUDY_SEED + 1000 * int(b["index"])


def stack_values(values, b, grid, statistic="mean", n_boot=N_BOOT):
    """sc.stack on derived per-halo values with the core's default seeds and strata (paired with stack_bin)."""
    res = sc.stack(np.asarray(values, float), b["weights"], statistic, n_boot=n_boot, seed=seed_of(b),
                   strata=b["strata"], return_boot=True)
    res["resolved"] = sc.resolution_mask(b["rows"], grid, "strict")
    res.update(grid=grid, bin=b["label"], scheme=b["scheme"], N=b["N"], N_eff_bin=b["N_eff"],
               mean_log_mass=b["mean_log_mass"], edges=sc.edges(grid))
    return res


def stack_key(key, b, n_boot=N_BOOT):
    return sc.stack_bin(key, b, "mean", "strict", n_boot=n_boot, return_boot=True)


def all_rows(res):
    """[1 + n_boot, S]: the sample value then the bootstrap replicates."""
    return np.vstack([res["value"][None, :], res["boot"]])


# ---------------------------------------------------------------------------- conservative remapping
def _cumulative(F, se):
    F = np.atleast_2d(np.asarray(F, float))
    bad = ~np.isfinite(F)
    Fz = np.where(bad, 0.0, F)
    cum = np.concatenate([np.zeros((F.shape[0], 1)), np.cumsum(Fz * np.diff(se**3)[None, :], axis=1)], axis=1)
    return F, bad, cum


def _flag(out, bad, se, te):
    """NaN where a target shell touches a NaN source shell or is not fully covered."""
    covered = (te[:-1] >= se[0] - 1e-12) & (te[1:] <= se[-1] + 1e-9)
    lo_idx = np.clip(np.searchsorted(se, te[:-1], side="right") - 1, 0, len(se) - 2)
    hi_idx = np.clip(np.searchsorted(se, te[1:], side="left") - 1, 0, len(se) - 2)
    for k in range(len(te) - 1):
        touched = bad[:, lo_idx[k]:hi_idx[k] + 1].any(axis=1)
        out[touched, k] = np.nan
    out[:, ~covered] = np.nan
    return out


def remap_rows_const(F, src_edges, tgt_edges, scale=1.0, factor=1.0):
    """Conservative remap assuming a constant density inside every source shell (C(x^3) piecewise linear).
    Kept for the accuracy comparison; remap_rows (PCHIP) is used everywhere else."""
    se = np.asarray(src_edges, float) * scale
    te = np.asarray(tgt_edges, float)
    F, bad, cum = _cumulative(F, se)
    us, ut = se**3, te**3
    out = np.empty((F.shape[0], len(te) - 1))
    for r in range(F.shape[0]):
        out[r] = np.diff(np.interp(ut, us, cum[r])) / np.diff(ut)
    return _flag(out, bad, se, te) * factor


def remap_rows(F, src_edges, tgt_edges, scale=1.0, factor=1.0):
    """Conservative remap of shell averages F [R, S] given on ``src_edges`` (units of R_src) onto ``tgt_edges``
    (units of R_tgt): the cumulative integral C(x) = sum F dV is known exactly at the source edges and is
    interpolated in ln x with a monotone PCHIP (so every source-shell integral is conserved and the density varies
    smoothly inside a shell); inside a first source shell that starts at x = 0 the density is taken constant.

    scale = R_src / R_tgt (source edges in target units = src_edges * scale); factor multiplies the values
    (normalisation change).  Target shells not fully covered by the source range, or touching a NaN source
    shell, are NaN.
    """
    from scipy.interpolate import PchipInterpolator
    se = np.asarray(src_edges, float) * scale
    te = np.asarray(tgt_edges, float)
    F, bad, cum = _cumulative(F, se)
    pos = se > 0
    x0 = se[pos][0]
    p = PchipInterpolator(np.log(se[pos]), cum[:, pos].T, axis=0)
    ct = np.zeros((len(te), F.shape[0]))
    for k, t in enumerate(te):
        if t <= 0:
            continue
        if t < x0:
            ct[k] = cum[:, pos][:, 0] * (t / x0)**3 if se[0] == 0 else 0.0
        else:
            ct[k] = p(np.log(min(t, se[-1])))
    out = (np.diff(ct, axis=0) / np.diff(te**3)[:, None]).T
    return _flag(out, bad, se, te) * factor


def remap_rows_powerlaw(F, src_edges, tgt_edges, scale=1.0, factor=1.0):
    """Conservative remap with a power-law cumulative inside every source shell (ln C linear in ln x, i.e. a
    power-law density per shell); constant density in a first shell starting at x = 0 or where C <= 0.
    Used only for the remap-method sensitivity test."""
    se = np.asarray(src_edges, float) * scale
    te = np.asarray(tgt_edges, float)
    F, bad, cum = _cumulative(F, se)
    ct = np.zeros((F.shape[0], len(te)))
    for k, t in enumerate(te):
        if t <= se[0]:
            continue
        t = min(t, se[-1])
        i = int(np.clip(np.searchsorted(se, t, side="right") - 1, 0, len(se) - 2))
        a, b = se[i], se[i + 1]
        ca, cb = cum[:, i], cum[:, i + 1]
        lin = ca + (cb - ca) * (t**3 - a**3) / (b**3 - a**3)          # constant density
        if a > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                pw = ca * np.exp(np.log(cb / ca) * np.log(t / a) / np.log(b / a))
            ok = (ca > 0) & (cb > 0)
            ct[:, k] = np.where(ok, pw, lin)
        else:
            ct[:, k] = lin
    out = np.diff(ct, axis=1) / np.diff(te**3)[None, :]
    return _flag(out, bad, se, te) * factor


def remap_rows_loglog(F, src_edges, tgt_edges, scale=1.0, factor=1.0, n_sub=64):
    """NON-conservative remap: ln F interpolated linearly in ln x between the source volume-weighted shell radii
    (flat beyond the outermost radii), averaged over each target shell by quadrature uniform in x^3 (volume).
    Used only for the remap-method sensitivity test."""
    se = np.asarray(src_edges, float) * scale
    te = np.asarray(tgt_edges, float)
    F = np.atleast_2d(np.asarray(F, float))
    bad = ~np.isfinite(F)
    xc = xcentre(se)
    u = (np.arange(n_sub) + 0.5) / n_sub
    xs = np.cbrt(te[:-1, None]**3 + u[None, :] * (te[1:, None]**3 - te[:-1, None]**3))     # [T, n_sub]
    lx = np.log(xs.ravel())
    out = np.empty((F.shape[0], len(te) - 1))
    for r in range(F.shape[0]):
        ok = np.isfinite(F[r]) & (F[r] > 0)
        if ok.sum() < 2:
            out[r] = np.nan
            continue
        v = np.exp(np.interp(lx, np.log(xc[ok]), np.log(F[r, ok])))
        out[r] = v.reshape(xs.shape).mean(axis=1)
    return _flag(out, bad, se, te) * factor


REMAP_METHODS = {"pchip": remap_rows, "constant_density": remap_rows_const, "powerlaw_cumulative": remap_rows_powerlaw,
                 "loglog_points": remap_rows_loglog}


def remap_mask(mask_src, src_edges, tgt_edges, scale=1.0):
    """Target shell flagged True when it is fully covered and every overlapping source shell is True."""
    se = np.asarray(src_edges, float) * scale
    te = np.asarray(tgt_edges, float)
    out = np.zeros(len(te) - 1, bool)
    for k in range(len(te) - 1):
        a, b = te[k], te[k + 1]
        if a < se[0] - 1e-12 or b > se[-1] + 1e-9:
            continue
        ov = (se[1:] > a + 1e-12) & (se[:-1] < b - 1e-12)
        out[k] = bool(np.all(np.asarray(mask_src)[ov])) if ov.any() else False
    return out


# ---------------------------------------------------------------------------- band ratios
def band_ratios(var_rows, ref_rows, ok, x):
    """Per band: median over the ``ok`` shells of var/ref (sample) and the paired-bootstrap SD of that median.

    var_rows, ref_rows: [1 + n_boot, S] on the same shells; ok [S] booleans; x [S] shell radii (R200c).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = var_rows / ref_rows
    out = {}
    for (a, b), lab in zip(BANDS, BAND_LABELS):
        sel = ok & (x >= a) & (x < b) & np.isfinite(ratio[0])
        if not sel.any():
            out[lab] = {"ratio": None, "err": None, "n_shells": 0}
            continue
        med = np.nanmedian(ratio[:, sel], axis=1)
        err = float(np.nanstd(med[1:], ddof=1)) if np.isfinite(med[1:]).sum() >= 2 else None
        out[lab] = {"ratio": float(med[0]), "err": err, "n_shells": int(sel.sum())}
    return out


# ---------------------------------------------------------------------------- impacts
def impact_rows(var_cal_rows, ref_cal_rows, field, obs, n_err=N_BOOT):
    """sc.impact_from_stacks on the sample (row 0) and the paired replicates (SD of dO/O)."""
    s = sc.impact_from_stacks(var_cal_rows[:, 0], ref_cal_rows[:, 0], field)
    out = {}
    reps = {o: [] for o in obs}
    for r in range(1, min(n_err, var_cal_rows.shape[1] - 1) + 1):
        ir = sc.impact_from_stacks(var_cal_rows[:, r], ref_cal_rows[:, r], field)
        for o in obs:
            reps[o].append(ir[o]["dO_over_O"])
    for o in obs:
        v = np.asarray(reps[o], float)
        out[o] = {"dO_over_O": s[o]["dO_over_O"], "dlnO": s[o]["dlnO"],
                  "err": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() >= 2 else None,
                  "missing_share": s[o]["missing_share"], "calibrated_share": s[o]["calibrated_share"]}
    return out


def impact_dln(dln, field, obs):
    r = sc.observable_impact(dln, field)
    return {o: {"dO_over_O": r[o]["dO_over_O"], "dlnO": r[o]["dlnO"], "missing_share": r[o]["missing_share"],
                "calibrated_share": r[o]["calibrated_share"]} for o in obs}


# ---------------------------------------------------------------------------- JSON helpers
def sig(x, n=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def siglist(a, n=4):
    a = np.asarray(a)
    if a.dtype == bool:
        return [int(v) for v in a.ravel()]
    if np.issubdtype(a.dtype, np.integer):
        return [int(v) for v in a.ravel()]
    return [sig(v, n) for v in a.ravel()]


def jsonify(o, n=6):
    """Recursively convert numpy scalars / arrays to JSON-ready values (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): jsonify(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonify(v, n) for v in o]
    if isinstance(o, np.ndarray):
        return jsonify(o.tolist(), n)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return sig(o, n)
    return o


def web_record(vid, field, b, edges, x_unit, res_like, resolved, sig_n=4):
    """Record in the website format (see the group brief)."""
    e = np.asarray(edges, float)
    q16 = res_like.get("q16")
    q84 = res_like.get("q84")
    S = len(e) - 1
    nan = np.full(S, np.nan)
    return {"variant": vid, "field": field, "scheme": b["scheme"], "bin": b["label"],
            "logm": sig(b["mean_log_mass"], sig_n), "N": int(b["N"]), "N_eff": sig(b["N_eff"], sig_n),
            "x_unit": x_unit, "edges": siglist(e, sig_n), "x": siglist(xcentre(e), sig_n),
            "value": siglist(res_like["value"], sig_n), "err": siglist(res_like["err"], sig_n),
            "q16": siglist(nan if q16 is None else q16, sig_n), "q84": siglist(nan if q84 is None else q84, sig_n),
            "n_used": siglist(np.asarray(res_like["n_used"]).astype(int)),
            "resolved": [int(bool(v)) for v in np.asarray(resolved)]}
