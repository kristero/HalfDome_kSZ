"""One definition of the joint generation and inference support.

Mass means physical M200c in Msun. beta is the raw Battaglia exponent:
P/P200 = P0 (x/xc)^(-0.3) (1+x/xc)^(-beta), x=r/R200c.
Keep changes to bounds and engineering cuts in prior.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import betaln, betainc
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
KEYS = ("battaglia_P0_amp", "battaglia_x_c_amp", "battaglia_beta_amp",
        "battaglia_P0_alpha_m", "battaglia_x_c_alpha_m", "battaglia_beta_alpha_m",
        "battaglia_P0_alpha_z", "battaglia_x_c_alpha_z", "battaglia_beta_alpha_z")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def mass_redshift(domain, count=2):
    m = np.logspace(*domain["log10_mass"], count)
    z = np.expm1(np.linspace(*np.log1p(domain["redshift"]), count))
    mass, redshift = np.meshgrid(m, z)
    return mass.ravel(), redshift.ravel()


def evolved(theta, mass, redshift):
    theta = np.asarray(theta, dtype=np.float64)
    return tuple(theta[..., k, None] * (mass/1e14)**theta[..., k+3, None]
                 * (1+redshift)**theta[..., k+6, None] for k in range(3))


def log_y200(p0, xc, beta):
    """Finite integral: P0 xc^3 B_t(2.7,beta-2.7), t=1/(1+xc).

    The dimensional factors and electron/thermal conversion cancel in the
    same-mass, same-redshift ratio to B12. Called only for beta > 2.7.
    """
    return (np.log(p0) + 3*np.log(xc) + betaln(2.7, beta-2.7)
            + np.log(betainc(2.7, beta-2.7, 1/(1+xc))))


class JointPrior:
    def __init__(self, config=None):
        self.config = config or json.loads((HERE / "prior.json").read_text())
        self.low = np.asarray(self.config["lower"])
        self.high = np.asarray(self.config["upper"])
        self.names = self.config["parameter_order"]
        self.interpolation_points = mass_redshift(self.config["interpolation_domain"])
        self.size_points = mass_redshift(self.config["catalogue_enclosing_domain"])
        self.y_points = mass_redshift(self.config["catalogue_enclosing_domain"],
                                     self.config["Y200_grid_size"])
        _, x, b = evolved(B12, *self.size_points)
        self.reference_size = x/b
        self.reference_log_y = log_y200(*evolved(B12, *self.y_points))

    def from_unit(self, unit):
        unit = np.asarray(unit, dtype=np.float64)
        theta = self.low + unit*(self.high-self.low)
        for k in self.config["log_uniform_indices"]:
            theta[..., k] = np.exp(np.log(self.low[k]) + unit[..., k]
                                   * np.log(self.high[k]/self.low[k]))
        return theta

    def to_unit(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        unit = (theta-self.low)/(self.high-self.low)
        for k in self.config["log_uniform_indices"]:
            unit[..., k] = np.log(theta[..., k]/self.low[k])/np.log(self.high[k]/self.low[k])
        return unit

    def contains(self, theta, return_metrics=False):
        """Vectorized, bounded-memory support evaluation; no silent clipping."""
        theta = np.asarray(theta, dtype=np.float64)
        scalar = theta.ndim == 1
        theta = theta.reshape(-1, 9)
        good = np.all(np.isfinite(theta) & (theta >= self.low) & (theta <= self.high), axis=1)
        metrics = {name: np.full(len(theta), np.nan) for name in
                   ("beta_min", "beta_max", "size_min", "size_max", "Y200_min", "Y200_max", "tail")}
        idx = np.flatnonzero(good)
        if len(idx):
            _, xc, beta = evolved(theta[idx], *self.interpolation_points)
            bmin, bmax = beta.min(1), beta.max(1)
            metrics["beta_min"][idx], metrics["beta_max"][idx] = bmin, bmax
            lo, hi = self.config["beta_range_on_interpolation_domain"]
            good[idx] &= (bmin >= lo) & (bmax <= hi)
            # Exact extremal powers; combining max xc and min beta is a
            # conservative bound on the missing central LOS column.
            valid = bmin > .7
            tail = np.ones(len(idx))
            endpoint = self.config["los_endpoint_R200"]
            xmax = xc[valid].max(1)
            tail[valid] = betainc(bmin[valid]-.7, .7, xmax/(endpoint+xmax))
            metrics["tail"][idx] = tail
            good[idx] &= tail <= self.config["maximum_central_column_tail"]
        idx = np.flatnonzero(good)
        if len(idx):
            _, xc, beta = evolved(theta[idx], *self.size_points)
            size = xc/beta/self.reference_size
            smin, smax = size.min(1), size.max(1)
            metrics["size_min"][idx], metrics["size_max"][idx] = smin, smax
            lo, hi = self.config["size_ratio_to_battaglia12"]
            good[idx] &= (smin >= lo) & (smax <= hi)
        idx = np.flatnonzero(good)
        for start in range(0, len(idx), 2048):
            part = idx[start:start+2048]
            ratio = np.exp(log_y200(*evolved(theta[part], *self.y_points)) - self.reference_log_y)
            ymin, ymax = ratio.min(1), ratio.max(1)
            metrics["Y200_min"][part], metrics["Y200_max"][part] = ymin, ymax
            lo, hi = self.config["Y200_ratio_to_battaglia12"]
            good[part] &= np.isfinite(ratio).all(1) & (ymin >= lo) & (ymax <= hi)
        if scalar:
            good = bool(good[0])
            metrics = {key: float(value[0]) for key, value in metrics.items()}
        return (good, metrics) if return_metrics else good

    def sample(self, count, rng=None):
        """IID exact rejection samples, for inference (not Sobol)."""
        rng = rng if rng is not None else np.random.default_rng()
        parts, remaining = [], int(count)
        while remaining:
            theta = self.from_unit(rng.random((min(65536, max(4096, remaining*64)), 9)))
            accepted = theta[self.contains(theta)][:remaining]
            parts.append(accepted)
            remaining -= len(accepted)
        return np.concatenate(parts) if parts else np.empty((0, 9))

    def sobol(self, count, seed=None):
        """Accepted prefix independent of requested count; retain proposal IDs.

        Rejection destroys the original Sobol digital-net balance. This is a
        deterministic space-filling design for the conditional density.
        """
        engine = qmc.Sobol(9, scramble=True, seed=self.config["seed"] if seed is None else seed)
        parts, indices, remaining, offset = [], [], int(count), 0
        while remaining:
            theta = self.from_unit(engine.random(65536))
            keep = np.flatnonzero(self.contains(theta))[:remaining]
            parts.append(theta[keep])
            indices.append(keep+offset)
            remaining -= len(keep)
            offset += len(theta)
        return np.concatenate(parts), np.concatenate(indices)
