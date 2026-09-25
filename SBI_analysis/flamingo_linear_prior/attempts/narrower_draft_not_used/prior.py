"""Independent uniform physical parameters, plus diagnostics of the whole box.

The numerical restrictions are checked on the entire rectangle BEFORE sampling.
No generated point is rejected, clipped, resampled, or assigned a new weight.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaln, hyp2f1
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
    mass = np.logspace(*domain["log10_mass"], count)
    redshift = np.expm1(np.linspace(*np.log1p(domain["redshift"]), count))
    m, z = np.meshgrid(mass, redshift)
    return m.ravel(), z.ravel()


def evolved(theta, mass, redshift):
    """Physical M200c in Msun; amplitude pivot 1e14 Msun at z=0."""
    theta = np.asarray(theta, dtype=float)
    return tuple(theta[..., k, None] * (mass / 1e14)**theta[..., k + 3, None]
                 * (1 + redshift)**theta[..., k + 6, None] for k in range(3))


def log_y200(p0, xc, beta):
    """Finite spherical pressure integral: valid on both sides of beta=2.7.

    Integral is P0*xc^0.3 * integral_0^1 r^1.7*(1+r/xc)^(-beta) dr.
    For beta>2.7 use the previous incomplete-beta expression. For shallower
    slopes, use the positive-term hypergeometric representation of the same
    FINITE integral. Infinite total thermal content is not assumed finite.
    This function is diagnostic only; it does not enter the map projection.
    """
    p0, xc, beta = np.broadcast_arrays(p0, xc, beta)
    if np.any((p0 <= 0) | (xc <= 0) | (beta <= 0)):
        raise ValueError("Pressure amplitudes, radii and slopes must be positive")
    t = 1 / (1 + xc)
    result = np.empty(p0.shape)
    steep = beta > 2.7
    result[steep] = (np.log(p0[steep]) + 3 * np.log(xc[steep])
                     + betaln(2.7, beta[steep] - 2.7)
                     + np.log(betainc(2.7, beta[steep] - 2.7, t[steep])))
    shallow = ~steep
    result[shallow] = (np.log(p0[shallow]) + 3 * np.log(xc[shallow])
                       + 2.7 * np.log(t[shallow]) - np.log(2.7)
                       + np.log(hyp2f1(2.7, 3.7 - beta[shallow], 3.7, t[shallow])))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite finite-pressure integral")
    return result


class UniformPrior:
    def __init__(self, config=None):
        self.config = config or json.loads((HERE / "prior.json").read_text())
        self.names = self.config["parameter_order"]
        self.low = np.asarray(self.config["lower"], dtype=float)
        self.high = np.asarray(self.config["upper"], dtype=float)
        if self.low.shape != (9,) or not np.all(self.high > self.low):
            raise ValueError("Expected nine ordered finite parameter intervals")
        if not np.isfinite([self.low, self.high]).all() or np.any(self.low[:3] <= 0):
            raise ValueError("Invalid physical parameter bounds")
        if self.config["log_uniform_indices"] or self.config["sampling_cuts"]:
            raise ValueError("This prior must be an unconditioned linear uniform box")
        self.interpolation_points = mass_redshift(self.config["interpolation_domain"])
        self.size_points = mass_redshift(self.config["catalogue_enclosing_domain"])
        self.y_points = mass_redshift(self.config["catalogue_enclosing_domain"],
                                     self.config["Y200_diagnostic_grid_size"])
        _, x, b = evolved(B12, *self.size_points)
        self.reference_size = x / b
        self.reference_log_y = log_y200(*evolved(B12, *self.y_points))

    def from_unit(self, unit):
        return self.low + np.asarray(unit, dtype=float) * (self.high - self.low)

    def to_unit(self, theta):
        return (np.asarray(theta, dtype=float) - self.low) / (self.high - self.low)

    def corners(self):
        return self.from_unit(np.array(list(itertools.product((0., 1.), repeat=9))))

    def contains(self, theta, return_metrics=False):
        theta = np.asarray(theta, dtype=float)
        good = np.all(np.isfinite(theta) & (theta >= self.low) & (theta <= self.high), axis=-1)
        if not return_metrics:
            return bool(good) if good.ndim == 0 else good
        metrics = self.metrics(theta)
        return (bool(good) if good.ndim == 0 else good), metrics

    def metrics(self, theta):
        scalar = np.asarray(theta).ndim == 1
        theta = np.asarray(theta).reshape(-1, 9)
        _, xc, beta = evolved(theta, *self.interpolation_points)
        bmin, bmax, xmax = beta.min(1), beta.max(1), xc.max(1)
        tail = np.ones(len(theta))
        valid = bmin > .7
        tail[valid] = betainc(bmin[valid] - .7, .7,
                              xmax[valid] / (self.config["los_endpoint_R200"] + xmax[valid]))
        _, x, b = evolved(theta, *self.size_points)
        size = x / b / self.reference_size
        ymin, ymax = np.empty(len(theta)), np.empty(len(theta))
        for start in range(0, len(theta), 1024):
            part = slice(start, start + 1024)
            ratio = np.exp(log_y200(*evolved(theta[part], *self.y_points)) - self.reference_log_y)
            ymin[part], ymax[part] = ratio.min(1), ratio.max(1)
        result = dict(beta_min=bmin, beta_max=bmax, xc_min=xc.min(1), xc_max=xmax,
                      size_min=size.min(1), size_max=size.max(1),
                      Y200_min=ymin, Y200_max=ymax, tail=tail)
        return {k: float(v[0]) for k, v in result.items()} if scalar else result

    def certify_box(self):
        """Exact amplitude extrema at parameter and M,z corners; conservative tail."""
        metrics = self.metrics(self.corners())
        bmin, bmax = metrics["beta_min"].min(), metrics["beta_max"].max()
        xmin, xmax = metrics["xc_min"].min(), metrics["xc_max"].max()
        if bmin <= self.config["whole_box_minimum_beta_required"]:
            raise ValueError("Box contains slopes unsupported by the preserved projector")
        tail = float(betainc(bmin - .7, .7, xmax / (self.config["los_endpoint_R200"] + xmax)))
        if tail > self.config["whole_box_maximum_central_column_tail"]:
            raise ValueError("Whole-box conservative missing-column bound exceeds 1 percent")
        return dict(beta_min=float(bmin), beta_max=float(bmax), xc_min=float(xmin),
                    xc_max=float(xmax), conservative_central_column_tail=tail,
                    all_parameter_corners=512, sampling_acceptance=1.0,
                    limitations="Analytic projection-domain checks; full-map extremes require preflight")

    def sobol(self, count, seed=None):
        if not 1 <= count <= 524288:
            raise ValueError("count must be in 1..524288")
        self.certify_box()
        engine = qmc.Sobol(9, scramble=True, seed=self.config["seed"] if seed is None else seed)
        power = int(np.ceil(np.log2(count)))
        unit = engine.random_base2(power)[:count]
        return self.from_unit(unit), np.arange(count, dtype=np.int64)

    def sample(self, count, rng=None):
        self.certify_box()
        rng = np.random.default_rng() if rng is None else rng
        return self.from_unit(rng.random((count, 9)))
