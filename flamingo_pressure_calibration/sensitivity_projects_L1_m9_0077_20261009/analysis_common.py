"""Shared physical contracts for the three isolated sensitivity projects.

This module reads compact products only. Kernels are physical z=0 volume
integrals; they are not sky spectra or a uniquely owned gas decomposition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "radii234_contamination_L1_m9_0077_20261008"
sys.path.insert(0, str(STUDY))
import pilot_physics as P
from fit_study import accepted, fit_one, resolution_masks
from models import shape, shell

RADII = np.array([2., 3., 4.])
K = np.r_[0., np.geomspace(1e-3, 12., 95)]
FIELDS = ("pressure", "electrons")
FIELD_KEY = {"pressure": "pressure_components", "electrons": "electron_components"}
FAMILY = {"pressure": "b12", "electrons": "b16"}
BOX_MPC = 1000.


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def load():
    plan = json.loads((STUDY / "metadata_plan.json").read_text())
    halos = dict(np.load(STUDY / "selected_halos.npz"))
    data = dict(np.load(STUDY / "measurements.npz"))
    cases = json.loads((STUDY / "products/study_summary.json").read_text())["bins"]
    count = np.bincount(halos["mass_bin"], minlength=15)
    inclusion = np.asarray(plan["population_per_bin"]) / count
    halos["inclusion_weight"] = inclusion[halos["mass_bin"]]
    halos["logmass"] = np.log10(halos["m200c_msun"])
    return plan, halos, data, cases


def weighted_stack(values, weights):
    weights = np.asarray(weights, float)
    weights /= weights.sum()
    mean = weights @ values
    variance = weights @ ((values - mean)**2)
    correction = 1 - np.square(weights).sum()
    std = np.sqrt(variance / correction) if correction > 0 else np.full_like(mean, np.nan)
    return mean, std, 1 / np.square(weights).sum()


def shell_transfer(k, radius, edges=P.EDGES):
    """Exact spherical transform of a unit-volume shell-constant field.

    Returns [halo, shell, k]. The small-kr expansion avoids cancellation;
    sinc(t)=sin(t)/t, while numpy.sinc uses a pi-scaled argument.
    """
    radius = np.atleast_1d(radius)
    k = np.atleast_1d(k)
    lo = radius[:, None, None] * edges[None, :-1, None]
    hi = radius[:, None, None] * edges[None, 1:, None]
    kl, kh = lo * k, hi * k
    denominator = (hi**3 - lo**3) * k**3
    value = np.divide(
        3 * (np.sin(kh) - kh*np.cos(kh) - np.sin(kl) + kl*np.cos(kl)),
        denominator, out=np.ones_like(kh), where=np.abs(denominator) > 0,
    )
    small = np.abs(kh) < .08
    k2 = k**2
    series = (1 - k2 * (hi**5-lo**5)/(10*(hi**3-lo**3))
              + k2**2 * (hi**7-lo**7)/(280*(hi**3-lo**3))
              - k2**3 * (hi**9-lo**9)/(15120*(hi**3-lo**3)))
    return np.where(small, series, value)


def measured_kernels(data, halos, rows, field, radius, components=(0, 1, 2, 3), k=K):
    """UY in physical Mpc^2 or UN as electron count; sharp finite sphere."""
    sum_name = "y" if field == "pressure" else "electrons"
    source = data["sums"][rows][:, components, P.SUM_NAMES.index(sum_name)].sum(axis=1)
    source = source * (P.EDGES[1:] <= radius + 1e-12)
    transfer = shell_transfer(k, halos["r_200_crit_mpc"][rows])
    return np.einsum("hs,hsk->hk", source, transfer, optimize=True)


def model_dimensionless_kernel(family, parameters, radius, q):
    """4pi integral x^2 F(x) sinc(qx) dx; q = physical k*R200c.

    Integrate separately over data shells to resolve the inner profile and
    preserve exact 2/3/4 boundaries; 32-point quadrature per shell.
    """
    q = np.atleast_1d(q)
    nodes, weights = leggauss(32)
    edges = P.EDGES[P.EDGES <= radius + 1e-12]
    lo, hi = edges[:-1, None], edges[1:, None]
    x = (lo + (hi-lo)*(nodes+1)/2).ravel()
    w = ((hi-lo)*weights/2).ravel() * x**2
    factor = 4*np.pi*shape(family, x, parameters)*w
    result = np.empty(len(q))
    # Bound temporary matrices even when callers pass many halo/radius pairs.
    for start in range(0, len(q), 2048):
        part = q[start:start+2048]
        result[start:start+2048] = np.sinc(part[:, None]*x[None, :]/np.pi) @ factor
    return result


def model_amplitude(halos, data, rows, field):
    radius = halos["r_200_crit_mpc"][rows]
    if field == "pressure":
        return P.SIGMA_T/(P.electron_mass*P.c**2) * data["p200_pa"][rows] * radius**3 * P.MPC_M
    return data["critical_density_msun_mpc3"][rows] * radius**3 / P.MU_XGP_MSUN


def model_kernels(parameters, halos, data, rows, field, radius, fb, k=K):
    """Apply a common bin shape to each halo's own physical normalization."""
    radius_mpc = halos["r_200_crit_mpc"][rows]
    q = (radius_mpc[:, None] * np.asarray(k)[None, :]).ravel()
    result = model_dimensionless_kernel(FAMILY[field], parameters, radius, q).reshape(len(rows), -1)
    amp = model_amplitude(halos, data, rows, field)
    if field != "pressure":
        amp = amp * fb
    return result * amp[:, None]


def relative_rms(prediction, target, mask):
    use = mask & (target > 0) & np.isfinite(target) & np.isfinite(prediction)
    return float(100*np.sqrt(np.mean((prediction[use]/target[use]-1)**2))) if use.any() else None


def weighted_rms(predictions, targets, mask, weights):
    use = mask[None, :] & (targets > 0) & np.isfinite(targets)
    values = np.where(use, (predictions/np.maximum(targets, 1e-300)-1)**2, 0)
    per_halo = np.divide(values.sum(axis=1), use.sum(axis=1),
                         out=np.full(len(targets), np.nan), where=use.sum(axis=1)>0)
    valid = np.isfinite(per_halo)
    return float(100*np.sqrt(np.average(per_halo[valid], weights=weights[valid])))


def pooled_fit(halos, data, rows, field, radius, softening, weights=None):
    values = data[FIELD_KEY[field]][rows].sum(axis=1)
    if weights is None:
        weights = halos["inclusion_weight"][rows]
    mean, std, neff = weighted_stack(values, weights)
    base, resolved, strict, supported = resolution_masks(halos, data, rows, softening)
    mask = base & (P.EDGES[1:] <= radius + 1e-12) & (mean > 0)
    fit = fit_one(FAMILY[field], mean, std, mask)
    return {"mean": mean, "std": std, "neff": neff, "base": base, "resolved": resolved,
            "strict": strict, "supported": supported, "mask": mask, "fit": fit}
