"""Catalogue one-halo approximation for proposing full-map parameter fits.

This is a new numerical forward calculation, not an extrapolation of the
trained emulator. Halo-centre mask weights and a flat-sky Hankel transform
approximate the full map. Every advertised fit must be checked by repainting.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
from scipy.special import j0, logsumexp

from prior_model import (ELECTRON_FRACTION, FIDUCIAL, H, MEC2, MPC, OM, SIGMA_T,
    bin_cl, comparison_metrics, domain_metrics, from_unit, parameters,
    physical_scales, save_json, to_unit)


def projection_table(path):
    if path.exists():
        return dict(np.load(path))
    logq = np.linspace(np.log(1e-10), np.log(1e6), 641)
    logb = np.linspace(np.log(.75), np.log(2048), 401)
    nodes, weights = np.polynomial.legendre.leggauss(256)
    q = np.exp(logq)
    # LOS coordinate t=q*sinh(u). The core-scaled integration endpoint
    # 1e8 gives negligible omitted tails in the optimizer's beta>=1.3 domain.
    upper = np.arcsinh(1e8/q)
    u = (nodes[None, :]+1)*upper[:, None]/2
    logr = logq[:, None]+np.log(np.cosh(u))
    logjac = logr+np.log(weights[None, :]*upper[:, None]/2)
    values = np.empty((len(logb), len(logq)))
    for start in range(0, len(logb), 8):
        beta = np.exp(logb[start:start+8])
        log_integrand = -.3*logr[None, :, :]-beta[:, None, None]*np.logaddexp(0, logr)[None, :, :]
        values[start:start+8] = np.log(2)+logsumexp(log_integrand+logjac[None, :, :], axis=2)
    np.savez(path, logq=logq, logbeta=logb, logcolumn=values)
    return dict(logq=logq, logbeta=logb, logcolumn=values)


class CatalogueForward:
    def __init__(self, root, campaign, radial_order=256):
        self.root = Path(root)
        self.campaign = Path(campaign)
        table = projection_table(self.root / "audit/projection_table.npz")
        self.column = RectBivariateSpline(table["logbeta"], table["logq"], table["logcolumn"])
        cat = np.load(self.root / "audit/catalogue_quadrature.npz")
        self.mass, self.z, self.weight = cat["mass"], cat["z"], cat["weight"]
        # Keep the proposal approximation used in the first trial. The
        # independent pressure validation includes the small radiation term;
        # every final map always uses the exact archived cosmology.
        r200, p200 = physical_scales(self.mass, self.z, include_radiation=False)
        zgrid = np.linspace(0, 5, 20001)
        chi = cumulative_trapezoid(299792.458/(H*100*np.sqrt(OM*(1+zgrid)**3+1-OM)), zgrid, initial=0)
        distance = np.interp(self.z, zgrid, chi)*MPC/(1+self.z)
        self.theta200 = r200/distance
        self.amplitude = SIGMA_T/MEC2*ELECTRON_FRACTION*p200*r200
        nodes, weights = np.polynomial.legendre.leggauss(radial_order)
        self.x = (nodes+1)*2
        self.radial_weight = weights*2*self.x
        enodes, eweights = np.polynomial.legendre.leggauss(3)
        edges = np.append(np.arange(80, 7881, 200), 7980)
        centres = (edges[:-1]+edges[1:]-1)/2
        widths = edges[1:]-edges[:-1]
        ell = centres[:, None]+enodes[None, :]*widths[:, None]/2
        self.bin_weights = eweights[None, :]*(2*ell+1)
        self.bin_weights /= self.bin_weights.sum(axis=1)[:, None]
        self.ell_nodes = ell.ravel()
        # The full harmonic kernel is a few hundred MB and reused in every fit.
        self.kernel = j0(self.theta200[:, None, None]*self.ell_nodes[None, :, None]*self.x[None, None, :])
        self.kernel *= self.radial_weight[None, None, :]
        sigma = (2/60*np.pi/180)/np.sqrt(8*np.log(2))
        self.dl_factor = self.ell_nodes*(self.ell_nodes+1)/(2*np.pi)*np.exp(-self.ell_nodes*(self.ell_nodes+1)*sigma*sigma)
        self.fiducial_raw = self.raw(FIDUCIAL)
        self.ell, self.fiducial_full = bin_cl(np.load(self.campaign / "results/HalfDome/masked_clean_cl.npy"))
        self.correction = self.fiducial_full/self.fiducial_raw

    def raw(self, theta):
        p0, xc, beta = parameters(theta, self.mass, self.z)
        logq = np.log(self.x[None, :]/xc[:, None])
        logb = np.broadcast_to(np.log(beta)[:, None], logq.shape)
        column = np.exp(self.column.ev(logb.ravel(), logq.ravel()).reshape(logq.shape))
        radial = np.einsum("hlr,hr->hl", self.kernel, column, optimize=False)
        yl = radial*(2*np.pi*self.theta200*self.theta200*self.amplitude*p0*xc)[:, None]
        cl = np.sum(self.weight[:, None]*yl*yl, axis=0)/(4*np.pi)
        values = (cl*self.dl_factor).reshape(40, 3)
        return np.sum(values*self.bin_weights, axis=1)

    def predict(self, theta, correction=None):
        return self.raw(theta)*(self.correction if correction is None else correction)

    def fit(self, target, name, start=FIDUCIAL, correction=None, multistart=True, trust_radius=None):
        correction = self.correction if correction is None else correction
        starts = [np.asarray(start)]
        if multistart:
            for xc, beta in ((1.2, 7.), (2., 11.)):
                theta = np.asarray(start).copy()
                theta[1:3] = [xc, beta]
                starts.append(theta)
        best = None
        runs = []
        # Small regularization chooses a representative among degenerate
        # spectra; it is reported and is not a posterior or measured prior.
        anchor = to_unit(FIDUCIAL)
        lower, upper = np.zeros(9)+1e-7, np.ones(9)-1e-7
        if trust_radius is not None:
            lower = np.maximum(lower, to_unit(start)-trust_radius)
            upper = np.minimum(upper, to_unit(start)+trust_radius)
        def residual(unit):
            theta = from_unit(unit)
            metrics = domain_metrics(theta)
            minimum_beta = float(metrics["min_beta"])
            if minimum_beta < 1.3:
                return np.full(49, 10+10*(1.3-minimum_beta))
            maximum_beta = float(metrics["max_beta"])
            if maximum_beta > 40:
                return np.full(49, 10+(maximum_beta-40))
            prediction = self.predict(theta, correction)
            if not np.isfinite(prediction).all() or np.any(prediction <= 0):
                return np.full(49, 1e6)
            return np.r_[np.log(prediction/target), .005*(unit-anchor)]
        def jacobian(unit):
            # SciPy 1.10 has no workers argument for least_squares. Independent
            # finite differences use the CPU allocation without changing the
            # forward model, objective, or bounded optimization variables.
            central = residual(unit)
            steps = 2e-4*np.maximum(abs(unit), .01)
            steps = np.where(unit+steps >= 1-1e-7, -steps, steps)
            trial = np.tile(unit, (9, 1))
            trial[np.arange(9), np.arange(9)] += steps
            with ThreadPoolExecutor(max_workers=9) as executor:
                shifted = np.array(list(executor.map(residual, trial)))
            return ((shifted-central)/steps[:, None]).T
        for index, initial in enumerate(starts):
            began = time.monotonic()
            initial_unit = np.clip(to_unit(initial), lower+1e-12, upper-1e-12)
            result = least_squares(residual, initial_unit, jac=jacobian,
                bounds=(lower, upper),
                max_nfev=100, diff_step=2e-4, ftol=2e-7, xtol=2e-7, gtol=2e-7)
            theta = from_unit(result.x)
            prediction = self.predict(theta, correction)
            metrics = comparison_metrics(prediction, target)
            record = dict(theta=theta.tolist(), **metrics, elapsed_seconds=time.monotonic()-began,
                          nfev=result.nfev, success=bool(result.success), message=result.message)
            runs.append(record)
            save_json(self.root / "proposals" / (name+"_searches.json"), runs)
            print("Proposal {} start {}: RMS {:.3%}".format(name, index, metrics["rms_fractional"]), flush=True)
            if best is None or record["rms_log"] < best["rms_log"]:
                best = record.copy()
            if record["rms_fractional"] < .01:
                break
        best.update(searches=runs, fit_type="regularized catalogue-forward proposal; requires full-map validation",
                    regularization_coefficient=.005, minimum_grid_beta_for_proposal=1.3,
                    maximum_grid_beta_for_proposal=40., local_unit_cube_trust_radius=trust_radius)
        save_json(self.root / "proposals" / (name+".json"), best)
        return np.array(best["theta"]), best

    def validate_projection(self):
        rng = np.random.default_rng(20260914)
        errors = []
        for _ in range(30):
            q = np.exp(rng.uniform(np.log(1e-4), np.log(100)))
            beta = np.exp(rng.uniform(np.log(1.3), np.log(60)))
            upper = np.arcsinh(1e8/q)
            base = -.3*np.log(q)-beta*np.log1p(q)+np.log(q)
            def integrand(u):
                logr = np.log(q)+np.log(np.cosh(u))
                return np.exp(.7*logr-beta*np.logaddexp(0, logr)-base)
            direct = 2*np.exp(base)*quad(integrand, 0, upper, epsabs=0, epsrel=1e-9, limit=300)[0]
            value = np.exp(self.column.ev(np.log(beta), np.log(q)))
            errors.append(abs(value/direct-1))
        record = dict(max_relative_error=float(max(errors)), cases=len(errors),
                      fiducial_response_correction=self.correction.tolist(),
                      approximation="One-halo, flat-sky, centre-mask; missing cross-halo terms calibrated only at fiducial")
        save_json(self.root / "audit/forward_validation.json", record)
        if max(errors) > .005:
            raise ValueError("Projection interpolation error exceeds 0.5 percent")
