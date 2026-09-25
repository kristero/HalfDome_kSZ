"""Uniform P0/beta density conditioned on the unchanged production guardrails.

The other seven parameters are fixed to B12. Original extended ranges remain
proposal bounds. Sampling and log_prob use the same joint support. Its area is
a one-dimensional integral, using the linear dependence of Y200 on P0.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betainc
from scipy.stats import qmc
from guardrails import B12, JointPrior, evolved, log_y200

HERE = Path(__file__).resolve().parent


class FlatPrior:
    """Existing API name retained; flat means uniform joint physical density."""

    def __init__(self, config=None):
        self.config = config or json.loads((HERE / "prior.json").read_text())
        self.names = self.config["parameter_order"]
        expected = {"evolved_beta", "relative_size", "finite_Y200", "central_column_tail"}
        if self.names != ["P0", "beta"] or set(self.config["rejection_cuts"]) != expected:
            raise ValueError("Expected P0/beta with every original support guard enabled")
        if self.config["log_uniform_indices"]:
            raise ValueError("This prior is uniform in physical coordinates")
        self.low = np.asarray(self.config["lower"], dtype=float)
        self.high = np.asarray(self.config["upper"], dtype=float)
        if not np.all(np.isfinite([self.low, self.high])) or not np.all(self.low < self.high):
            raise ValueError("Bounds must be finite and strictly increasing")
        self.width = self.high - self.low
        self.proposal_area = float(np.prod(self.width))
        self.guards = JointPrior(json.loads((HERE / self.config["guardrail_config"]).read_text()))
        if self.config["full_parameter_order"] != self.guards.names:
            raise ValueError("Full parameter ordering differs from the guards")
        for name, value in self.config["fixed_parameters"].items():
            if value != B12[self.guards.names.index(name)]:
                raise ValueError("This validated slice fixes all seven other parameters to B12")
        if np.any(self.low < self.guards.low[[0,2]]) or np.any(self.high > self.guards.high[[0,2]]):
            raise ValueError("Proposals must remain inside the extended envelope")
        self.beta_limits = self._derive_beta_limits()
        self.area, self.area_integration_error = quad(
            lambda beta: float(np.diff(self.amplitude_bounds(beta))[0]),
            *self.beta_limits, epsabs=1e-9, epsrel=1e-11, limit=150)
        if self.area <= 0:
            raise ValueError("The requested slice has no permitted area")
        self.normalizing_mass = self.area / self.proposal_area

    @staticmethod
    def _parameters(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 0 or theta.shape[-1] != 2:
            raise ValueError("Expected a final dimension of two: P0, beta")
        return theta

    def expand(self, theta):
        """Insert seven fixed B12 values in the original parameter order."""
        theta = self._parameters(theta)
        order = self.config["full_parameter_order"]
        full = np.empty(theta.shape[:-1] + (len(order),), dtype=float)
        for index, name in enumerate(order):
            full[..., index] = (theta[..., self.names.index(name)] if name in self.names
                                else self.config["fixed_parameters"][name])
        return full

    def _derive_beta_limits(self):
        """Exact power-law corner bounds, with the original tail restriction."""
        unit = self.expand([1., 1.])
        _, xc, factor = evolved(unit, *self.guards.interpolation_points)
        lo, hi = self.guards.config["beta_range_on_interpolation_domain"]
        lower = max(self.low[1], float(np.max(lo / factor)))
        upper = min(self.high[1], float(np.min(hi / factor)))
        _, core, beta = evolved(unit, *self.guards.size_points)
        size_coefficient = core / beta / self.guards.reference_size
        slo, shi = self.guards.config["size_ratio_to_battaglia12"]
        lower = max(lower, float(np.max(size_coefficient / shi)))
        upper = min(upper, float(np.min(size_coefficient / slo)))
        endpoint = self.guards.config["los_endpoint_R200"]
        tail_limit = self.guards.config["maximum_central_column_tail"]
        xmax, bfactor = float(np.max(xc)), float(np.min(factor))
        tail = lambda b: float(betainc(b * bfactor - .7, .7, xmax / (endpoint + xmax)))
        if lower >= upper or tail(upper) > tail_limit:
            raise ValueError("No beta interval passes the shape/tail guards")
        if tail(lower) > tail_limit:
            lower = brentq(lambda b: tail(b) - tail_limit, lower, upper, xtol=1e-13)
        # Only roundoff-sized inward adjustments. Algebraic equality can round
        # outside an inequality when the original guard reevaluates the powers.
        for _ in range(16):
            if self._shape_passes(lower):
                break
            lower = np.nextafter(lower, np.inf)
        for _ in range(16):
            if self._shape_passes(upper):
                break
            upper = np.nextafter(upper, -np.inf)
        if not self._shape_passes(lower) or not self._shape_passes(upper):
            raise ValueError("Could not represent the derived shape boundaries")
        return np.array([lower, upper])

    def _shape_passes(self, beta):
        # Shape metrics are evaluated before integrated-pressure rejection.
        _, m = self.guards.contains(self.expand([18.1, beta]), return_metrics=True)
        c = self.guards.config
        return (c["beta_range_on_interpolation_domain"][0] <= m["beta_min"]
                and m["beta_max"] <= c["beta_range_on_interpolation_domain"][1]
                and c["size_ratio_to_battaglia12"][0] <= m["size_min"]
                and m["size_max"] <= c["size_ratio_to_battaglia12"][1]
                and m["tail"] <= c["maximum_central_column_tail"])

    def _unit_y_extrema(self, beta):
        """Y/B12 at P0=1 on the original nine-by-nine mass/redshift grid."""
        beta = np.asarray(beta, dtype=float)
        theta = np.stack((np.ones_like(beta), beta), axis=-1)
        ratios = np.exp(log_y200(*evolved(self.expand(theta), *self.guards.y_points))
                        - self.guards.reference_log_y)
        return ratios.min(axis=-1), ratios.max(axis=-1)

    def amplitude_bounds(self, beta):
        """Allowed P0 interval at beta; exact analytic boundary for the grid cut."""
        beta = np.asarray(beta, dtype=float)
        ymin, ymax = self._unit_y_extrema(beta)
        low_y, high_y = self.guards.config["Y200_ratio_to_battaglia12"]
        lower = np.maximum(self.low[0], low_y / ymin)
        upper = np.minimum(self.high[0], high_y / ymax)
        valid = (beta >= self.beta_limits[0]) & (beta <= self.beta_limits[1]) & (lower <= upper)
        return np.stack((lower, np.where(valid, upper, lower)), axis=-1)

    def beta_bounds(self, p0):
        """Allowed beta interval at P0; pressure decreases monotonically in beta."""
        p0 = float(p0)
        blo, bhi = self.beta_limits
        if not self.low[0] <= p0 <= self.high[0]:
            return np.array([blo, blo])
        ymin, ymax = self.guards.config["Y200_ratio_to_battaglia12"]
        low_pressure = lambda beta: p0 * float(self._unit_y_extrema(beta)[0]) - ymin
        high_pressure = lambda beta: p0 * float(self._unit_y_extrema(beta)[1]) - ymax
        if low_pressure(blo) < 0 or high_pressure(bhi) > 0:
            return np.array([blo, blo])
        if low_pressure(bhi) < 0:
            bhi = brentq(low_pressure, blo, bhi, xtol=1e-12)
        if high_pressure(blo) > 0:
            blo = brentq(high_pressure, blo, bhi, xtol=1e-12)
        return np.array([blo, bhi])

    def contains(self, theta):
        theta = self._parameters(theta)
        flat = theta.reshape(-1, 2)
        inside = np.all(np.isfinite(flat) & (flat >= self.low) & (flat <= self.high), axis=1)
        accepted = np.zeros(len(flat), dtype=bool)
        indices = np.flatnonzero(inside)
        if len(indices):
            accepted[indices] = self.guards.contains(self.expand(flat[indices]))
        return accepted.reshape(theta.shape[:-1])

    def log_prob(self, theta):
        """Normalized constant density inside every guard; negative infinity outside."""
        return np.where(self.contains(theta), -np.log(self.area), -np.inf)

    def sample(self, count, seed=None):
        """IID rejection samples from the linear extended proposal rectangle."""
        if count < 0 or int(count) != count:
            raise ValueError("count must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        parts, remaining = [], int(count)
        while remaining:
            proposed = rng.uniform(self.low, self.high, size=(min(16384, max(1024, remaining*2)), 2))
            accepted = proposed[self.contains(proposed)][:remaining]
            parts.append(accepted)
            remaining -= len(accepted)
        return np.concatenate(parts) if parts else np.empty((0,2))

    def sobol(self, count, seed=None, return_proposal_ids=False):
        """Stable accepted prefix. Rejection removes exact digital-net balance."""
        if count < 0 or int(count) != count:
            raise ValueError("count must be a nonnegative integer")
        seed = self.config["design_seed"] if seed is None else seed
        engine = qmc.Sobol(2, scramble=True, seed=seed)
        parts, ids, remaining, offset = [], [], int(count), 0
        while remaining:
            proposed = self.low + engine.random(4096) * self.width
            keep = np.flatnonzero(self.contains(proposed))[:remaining]
            parts.append(proposed[keep])
            ids.append(keep + offset)
            remaining -= len(keep)
            offset += len(proposed)
        theta = np.concatenate(parts) if parts else np.empty((0,2))
        indices = np.concatenate(ids) if ids else np.empty(0,dtype=np.int64)
        return (theta, indices) if return_proposal_ids else theta

    def as_torch_distribution(self, device="cpu", dtype=None):
        """The actual guarded distribution, never its BoxUniform envelope."""
        from torch_prior import GuardedTorchPrior
        return GuardedTorchPrior(self, device=device, dtype=dtype)
