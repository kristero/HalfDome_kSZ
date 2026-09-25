"""Physical-shape support, deliberately distinct from numerical certification.

The finite-energy condition assumes the positive gNFW profile describes an
isolated halo all the way to infinity. It is unnecessary for a truncated halo.
Every generated candidate still needs the separately defined fidelity gate.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])


def evolve(theta, mass, redshift):
    theta = np.asarray(theta)
    return tuple(theta[..., k, None] * (mass / 1e14)**theta[..., k+3, None]
                 * (1+redshift)**theta[..., k+6, None] for k in range(3))


class AnalyticPrior:
    """Uniform physical-volume proposal conditioned on analytic shape support.

    No Gaussian, log-normal, size-ratio or relative-Y weighting is applied.
    This object does not claim that historical pixel painting is accurate.
    """
    numerical_certification = False

    def __init__(self, config=None):
        self.config = config or json.loads((HERE / "protocol.json").read_text())
        self.low = np.asarray(self.config["exploration_lower"], float)
        self.high = np.asarray(self.config["exploration_upper"], float)
        mass, redshift = np.meshgrid(10.**np.array([12., 15.7]), [.001, 5.])
        self.points = mass.ravel(), redshift.ravel()
        if not self.config['finite_untruncated_energy_assumption']:
            raise ValueError('A truncated-halo prior needs its own explicit support definition')

    def contains(self, theta):
        theta = np.asarray(theta, float)
        scalar = theta.ndim == 1
        theta = theta.reshape(-1, 9)
        good = np.all(np.isfinite(theta) & (theta >= self.low) & (theta <= self.high), axis=1)
        indices = np.flatnonzero(good)
        if len(indices):
            p0, xc, beta = evolve(theta[indices], *self.points)
            good[indices] &= (np.isfinite(p0).all(1) & np.isfinite(xc).all(1)
                & np.isfinite(beta).all(1) & (p0 > 0).all(1) & (xc > 0).all(1)
                & (beta > 2.7).all(1))
        return bool(good[0]) if scalar else good

    def sample(self, count, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        result, remaining = [], int(count)
        while remaining:
            proposal = rng.uniform(self.low, self.high, size=(max(4096, remaining*4), 9))
            accepted = proposal[self.contains(proposal)][:remaining]
            result.append(accepted)
            remaining -= len(accepted)
        return np.concatenate(result) if result else np.empty((0, 9))

    def normalization(self):
        """Fraction of the rectangular volume satisfying the beta inequalities.

        Only beta0 and its two evolution exponents enter this condition.
        Integrate the permitted beta0 interval over the other two exponents.
        The other six physical-volume factors cancel from this fraction.
        """
        from scipy.integrate import quad
        if hasattr(self, '_normalization'):
            return self._normalization
        u=np.log(np.array([1e12,10**15.7])/1e14)
        v=np.log1p([.001,5.])
        lower,upper=self.low[2],self.high[2]
        def mass_integral(az):
            zterm=az*(v[0] if az>=0 else v[1])
            total=0.
            for a,b,um in ((self.low[5],min(0,self.high[5]),u[1]),
                           (max(0,self.low[5]),self.high[5],u[0])):
                if b<=a:continue
                crossings=[(np.log(2.7/beta)-zterm)/um for beta in (lower,upper)]
                points=[point for point in crossings if a<point<b]
                def width(am):
                    return max(0.,upper-max(lower,2.7*np.exp(-am*um-zterm)))
                total+=quad(width,a,b,points=points,epsabs=1e-10,epsrel=1e-10)[0]
            return total
        volume=quad(mass_integral,self.low[8],self.high[8],points=[0.] if self.low[8]<0<self.high[8] else [],
                    epsabs=1e-9,epsrel=1e-9,limit=150)[0]
        self._normalization=volume/((upper-lower)*(self.high[5]-self.low[5])*(self.high[8]-self.low[8]))
        return self._normalization

    def log_prob(self, theta):
        """Normalized analytic candidate density, excluding the numerical gate."""
        log_density=-np.log(self.high-self.low).sum()-np.log(self.normalization())
        return np.where(self.contains(theta),log_density,-np.inf)


def fidelity_distance(candidate, reference, covariance):
    """Covariance-whitened observable error; no inverse is explicitly formed."""
    difference = np.asarray(candidate)-np.asarray(reference)
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), difference)
    return float(np.linalg.norm(whitened))


def certify_pair(candidate, reference, reference_refined, covariance, budget=.1):
    """Transparent convergence check; two resolutions do not prove a limit.

    The allowance is divided between measured candidate error and measured
    reference change. A pass applies to this tested point and operator only.
    """
    error = fidelity_distance(candidate, reference_refined, covariance)
    reference_change = fidelity_distance(reference, reference_refined, covariance)
    return dict(error_sigma=error, reference_change_sigma=reference_change,
                budget_sigma=budget, passed=error+reference_change <= budget,
                scope="tested point and specified covariance; not a continuum certificate")
