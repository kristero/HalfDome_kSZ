"""Profile-space propagation engine (overlap/covariance study, Step 0.1).

Any dimensionless z=0 profile F0(M,x), x=r/R200c, is propagated through the
frozen conditional lightcone halo model of the 2026-10-09 studies:

  u_ell(M,z) = amp(M,z) 4pi int x^2 F(M,z,x) W(M,x) sinc(q x) dx,  q=(ell+1/2)R200c/D_A
  C_ell^1h   = sum_cells N |u_ell|^2/(4pi)
  C_ell^2h   = sum_z P_lin(ell,z)/V_z [sum_M N b u_ell/(4pi)]^2       (linear bias)
  mean       = sum_cells N u_0/(4pi)                                   (mean y or tau)
  kSZ        = sum_cells sum(v_LOS^2)/c^2 |tau_ell|^2/(4pi) (catalogue velocities; also
               with N sigma_v,lin(z)^2) and an approximate linear 2h term
               sigma_v,lin^2/c^2 P_lin [sum N b tau_ell/(4pi)]^2/V_z
  DM         = halo-only observer DM for z_source=1 (z<=1 cells, factor 1/(1+z)):
               mean, compound-Poisson (Campbell) variance, DM-field C_ell, y x DM
               cross C_ell, and the exact compound-Poisson per-sightline PDF (FFT)
  Jacobians  = dO/dln F per radial segment and 0.2-dex mass bin for all of these
  Y(<x), N_e(<x) = 4pi int_0^x s^2 F ds (dimensionless, per catalogue mass)

with F(M,z,x) = F0(M,x) Fpub(M,z,x)/Fpub(M,0,x), the declared redshift response,
and amp the pressure (Compton-y) or electron (tau) normalisation of
lightcone_analytic.Ingredients.geometry.

Frozen modules are imported by exact path and never edited (run python -B):
  H = tsz_fit_truncation_.../halo_model/compute_halo_terms.py
  A = H.A = sensitivity_projects_.../analytic_observables/lightcone_analytic.py
Radial transforms use A.linear_radial_kernel, the exact Fourier integral of a
piecewise-linear F. A grid is split into segments at break radii: a hard cut at
a break is exact, while a window may extend the profile beyond it.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import hashlib
import importlib.util
import json
import multiprocessing
import resource
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import interpolate
from scipy.special import expit

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
CALIBRATION = STUDY.parent
TRUNCATION = CALIBRATION/"tsz_fit_truncation_L1_m9_0077_20261009"
SENSITIVITY = CALIBRATION/"sensitivity_projects_L1_m9_0077_20261009"
RADII234 = CALIBRATION/"radii234_contamination_L1_m9_0077_20261008"


def load_module(name, path):
    """Import a frozen source file by its exact path; the file is never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_module("frozen_compute_halo_terms", TRUNCATION/"halo_model/compute_halo_terms.py")
A = H.A          # frozen lightcone_analytic: transform, Ingredients, COUNTS
C = A.C          # frozen analysis_common
P = C.P          # frozen pilot_physics constants
VELOCITY = load_module("kernel_velocity", HERE/"velocity.py")

ELL = H.ELL.copy()                     # the 83 original multipoles, 20-6000
ELL_EXTENDED = np.unique(np.r_[ELL, 300., 8000., np.geomspace(6000., 2e4, 22)[1:]])
SHELL_EDGES = np.array([0., .1, .25, .5, 1., 1.5, 2., 3., 4.])
SHELLS = list(zip(SHELL_EDGES[:-1], SHELL_EDGES[1:]))
X_MIN = 1e-5
LOG_STEP = np.log(4./X_MIN)/1023       # 1024 log nodes on 1e-5..4R (original: 256)
DM_FACTOR = P.SIGMA_T*A.M2_PER_PC_CM3  # tau -> DM [pc cm^-3], before the 1/(1+z)
FIELDS = ("pressure", "electrons")
SOURCES = dict(
    compute_halo_terms=TRUNCATION/"halo_model/compute_halo_terms.py",
    lightcone_analytic=SENSITIVITY/"analytic_observables/lightcone_analytic.py",
    analysis_common=SENSITIVITY/"analysis_common.py",
    pilot_physics=RADII234/"pilot_physics.py",
    fit_parameters=TRUNCATION/"fit_inputs/fit_parameters.npz",
    profile_shapes=TRUNCATION/"fit_inputs/profile_shapes.npz",
    study_summary=RADII234/"products/study_summary.json",
    counts=Path(A.COUNTS), ingredients=Path(A.INGREDIENTS),
    velocities=SENSITIVITY/"lightcone_maps/fine_velocity_histogram.npz")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {key: sha(path) for key, path in SOURCES.items()}


def write_json(path, value):
    C.write_json(Path(path), value)


# ----------------------------------------------------------------------------
# Radial grids and exact transforms
# ----------------------------------------------------------------------------
class Grid:
    """Sorted radial nodes split into segments at exact break radii.

    Nodes closer than 1e-9 (relative) are merged, keeping the break value, so
    the piecewise-linear transform never sees a degenerate interval.
    """

    def __init__(self, x, breaks=(), exact=False):
        x = np.asarray(x, float)
        breaks = np.asarray(breaks, float)
        if exact:
            nodes = x
        else:
            candidates = np.unique(np.r_[x, breaks])
            is_break = np.isin(candidates, breaks)
            nodes, flags = [candidates[0]], [is_break[0]]
            for value, flag in zip(candidates[1:], is_break[1:]):
                if value-nodes[-1] > 1e-9*value:
                    nodes.append(value); flags.append(flag)
                elif flag and not flags[-1]:
                    nodes[-1], flags[-1] = value, True
            nodes = np.asarray(nodes)
        assert np.all(np.diff(nodes) > 0) and nodes[0] > 0
        breaks = np.unique(np.r_[nodes[0], breaks, nodes[-1]])
        breaks = breaks[(breaks >= nodes[0]) & (breaks <= nodes[-1])]
        index = np.searchsorted(nodes, breaks)
        assert np.array_equal(nodes[index], breaks), "every break must be a node"
        self.x = nodes
        self.breaks = breaks
        self.index = index
        self.segments = list(zip(index[:-1], index[1:]))

    def segments_upto(self, extent):
        """Number of leading segments that end at or before extent (a break)."""
        k = int(np.argmin(np.abs(self.breaks-extent)))
        assert abs(self.breaks[k]-extent) <= 1e-12*max(extent, 1.), (extent, self.breaks)
        return k

    def __len__(self):
        return len(self.x)


def make_grid(end, breaks=(), linear=(), log_step=LOG_STEP, refine=1, start=X_MIN):
    """Log nodes at fixed spacing plus break radii and linear edge zones.

    linear: (lo, hi, count) zones of evenly spaced nodes. refine multiplies the
    log density and the number of linear intervals (convergence doubling).
    """
    n = int(np.ceil(np.log(end/start)/(log_step/refine)))+1
    parts = [np.geomspace(start, end, n)]
    for lo, hi, count in linear:
        lo, hi = max(lo, start), min(hi, end)
        if hi > lo or (hi == lo and int(count) == 1):      # count 1 adds the single node lo
            parts.append(np.linspace(lo, hi, (int(count)-1)*refine+1))
    breaks = np.asarray([b for b in np.r_[breaks, end] if start < b <= end])
    return Grid(np.concatenate(parts), breaks)


def original_grid(cut):
    """The 256-node grid of compute_halo_terms.py line 173, as one segment."""
    return Grid(np.unique(np.r_[np.geomspace(1e-5, cut, 256), .9*cut]), exact=True)


def oscillatory_weights(x, q):
    """Closed-form branch of A.linear_radial_kernel (its lines 124-130), any q>0."""
    lo, hi = x[:-1], x[1:]
    dx = hi-lo
    ql = np.asarray(q, float).ravel()[:, None]
    sl, sh = np.sin(ql*lo), np.sin(ql*hi)
    cl, ch = np.cos(ql*lo), np.cos(ql*hi)
    i1 = (sh-sl)/ql**2-(hi*ch-lo*cl)/ql
    i2 = -(hi**2*ch-lo**2*cl)/ql+2*(hi*sh-lo*sl)/ql**2+2*(ch-cl)/ql**3
    weights = np.zeros((len(ql), len(x)))
    weights[:, :-1] += 4*np.pi/ql*(hi*i1-i2)/dx
    weights[:, 1:] += 4*np.pi/ql*(i2-lo*i1)/dx
    return weights


def radial_weights(x, q):
    """A.linear_radial_kernel with one guard for extended grids.

    The frozen kernel uses an 8-term sinc moment series for |q|<0.2. It is
    exact to ~1e-12 while q*x<=1.5, always true for x<=4R, but not on windows
    extended to ~20R. There the series is kept for nodes with q*x<=1.5 and the
    frozen closed form is used beyond. Grids ending below 7.5R are untouched.
    """
    weights = A.linear_radial_kernel(x, q)
    flat_q = np.asarray(q, float).ravel()
    flat = weights.reshape(len(flat_q), len(x))
    for i in np.flatnonzero((np.abs(flat_q) < .2) & (np.abs(flat_q)*x[-1] > 1.5)):
        k = max(int(np.searchsorted(x, 1.5/abs(flat_q[i]), side="right"))-1, 1)
        flat[i] = 0.
        flat[i, :k+1] += A.linear_radial_kernel(x[:k+1], flat_q[i:i+1])[0]
        flat[i, k:] += oscillatory_weights(x[k:], flat_q[i:i+1])[0]
    return weights


def transform(grid, profile, q, extent=None):
    """4pi int_0^extent x^2 F(x) sinc(q x) dx; profile [..., x] -> [..., q]."""
    count = len(grid.segments) if extent is None else grid.segments_upto(extent)
    q = np.atleast_1d(np.asarray(q, float))
    total = 0.
    for a, b in grid.segments[:count]:
        total = total+profile[..., a:b+1]@radial_weights(grid.x[a:b+1], q).T
    return total


def volume_integral(grid, profile, extent=None, start=None):
    """4pi int x^2 F dx over [start, extent] (breaks); the q=0 transform."""
    stop = len(grid.segments) if extent is None else grid.segments_upto(extent)
    first = 0 if start is None or start <= grid.x[0] else grid.segments_upto(start)
    total = 0.
    for a, b in grid.segments[first:stop]:
        total = total+profile[..., a:b+1]@A.linear_radial_kernel(grid.x[a:b+1], np.zeros(1))[0]
    return total


# ----------------------------------------------------------------------------
# Projection (column) operator for y(b), DM(b) and the DM variance
# ----------------------------------------------------------------------------
def b_quadrature(x, order=4):
    """Nodes and weights for int_0^{x[-1]} db, one panel per node interval.

    On [x_j, x_j+1] the substitution b = x_j+1-(x_j+1-x_j) u^2 removes the
    sqrt(x_j+1-b) behaviour of the column of a piecewise-linear profile, so
    the integrand is smooth on every panel.
    """
    t, w = leggauss(order)
    u, wu = (t+1)/2, w/2
    lo, hi = x[:-1], x[1:]
    b = hi[:, None]-(hi-lo)[:, None]*u**2
    wb = (hi-lo)[:, None]*2*u*wu
    return np.r_[x[0]*u, b.ravel()], np.r_[x[0]*wu, wb.ravel()]


def projection_operator(x, b, intervals=None):
    """M[b, j] for the column 2 int F(r) r dr/sqrt(r^2-b^2) of a piecewise-linear F.

    Exact for F linear between nodes, using int r/S dr = S and
    int r^2/S dr = (r S + b^2 ln(r+S))/2 with S = sqrt(r^2-b^2). intervals
    selects the node intervals that contribute (a hard cut or one shell).
    """
    b = np.asarray(b, float)
    if intervals is None:
        intervals = np.ones(len(x)-1, bool)
    operator = np.zeros((len(b), len(x)))
    for i in np.flatnonzero(intervals):
        r0, r1 = x[i], x[i+1]
        use = b < r1
        if not use.any():
            continue
        bb = b[use]
        lo = np.maximum(r0, bb)
        s1 = np.sqrt(r1*r1-bb*bb)
        slo = np.sqrt(np.maximum(lo*lo-bb*bb, 0.))
        ds = s1-slo
        dg = .5*(r1*s1-lo*slo)+.5*bb*bb*np.log((r1+s1)/(lo+slo))
        operator[use, i] += 2*(r1*ds-dg)/(r1-r0)
        operator[use, i+1] += 2*(dg-r0*ds)/(r1-r0)
    return operator


def interval_mask(grid, lo, hi):
    """Node intervals lying inside [lo, hi] (lo and hi are grid breaks)."""
    mid = .5*(grid.x[:-1]+grid.x[1:])
    return (mid > lo) & (mid < hi)


def column(grid, profile, b, extent=None):
    """Projected column Sigma(b) = 2 int F dl, x units, for F [..., x]."""
    extent = grid.x[-1] if extent is None else extent
    return profile@projection_operator(grid.x, b, interval_mask(grid, 0., extent)).T


def variance_forms(grid, extent, shells=(), order=4):
    """Quadratic forms Q with int d^2b Sigma(b)^2 = F^T Q F (x units).

    'total' is for the profile truncated at extent. For each shell k the
    bilinear form Q_k = M^T w M_k gives d/d(ln F in shell k) = 2 F^T Q_k F.
    """
    stop = grid.index[grid.segments_upto(extent)]
    b, wb = b_quadrature(grid.x[:stop+1], order)
    weight = 2*np.pi*b*wb
    full = projection_operator(grid.x, b, interval_mask(grid, 0., extent))
    weighted = full*weight[:, None]
    forms = {"total": full.T@weighted}
    for k, (lo, hi) in enumerate(shells):
        forms[k] = weighted.T@projection_operator(grid.x, b, interval_mask(grid, lo, hi))
    return forms


# ----------------------------------------------------------------------------
# Profile families, fits and fallbacks
# ----------------------------------------------------------------------------
def b12(parameters, x):
    """F = Pe/P200 = .5176 P0 (x/xc)^-.3 (1+x/xc)^-beta (H.pressure_shape)."""
    return H.pressure_shape(parameters, x)


def b16(parameters, x):
    """F = rho0 (x/.5)^-.2 (1+(x/.5)^alpha)^(-(beta-.2)/alpha) (models.shape b16)."""
    rho0, alpha, beta = np.moveaxis(np.asarray(parameters), -1, 0)
    q = x/.5
    return rho0[..., None]*q**(-.2)*(1+q**alpha[..., None])**(-(beta[..., None]-.2)/alpha[..., None])


SHAPES = {0: b12, 1: b16}


def support_definition():
    """The common support of compute_halo_terms: bins 13.5-15.5 and logM<=15.388."""
    shapes = np.load(SOURCES["profile_shapes"])
    fit = np.load(SOURCES["fit_parameters"])
    return dict(mass_edges=np.asarray(shapes["mass_edges"]),
                support=np.asarray(shapes["common_support"]).astype(bool),
                max_logmass=float(shapes["calibrated_logmass_max"]),
                pressure_parameters=np.asarray(fit["parameters"]))


def study_fits(field, radius_index):
    """radii234 'all'-gas fits; radius_index 0,1,2 = fit through 2,3,4R."""
    bins = json.loads(SOURCES["study_summary"].read_text())["bins"]
    return np.array([case["selections"]["all"][FIELDS[field]][radius_index]["fit"]["parameters"]
                     for case in bins])


class Profile:
    """z=0 dimensionless shape F0(M,x) with the published fallback.

    Calibrated masses (supported 0.2-dex bin and logM <= max_logmass) use the
    bin's own shape, constant within the bin. Every other mass uses the
    published B12/B16 profile at its actual mass, exactly as
    compute_halo_terms.evaluate_shapes. No interpolation across bin edges.
    """

    def __init__(self, field, label, mass_edges, support, max_logmass):
        self.field = field
        self.label = label
        self.mass_edges = np.asarray(mass_edges)
        self.support = np.asarray(support, bool)
        self.max_logmass = float(max_logmass)

    def bins(self, mass):
        logmass = np.log10(mass)
        bins = np.searchsorted(self.mass_edges, logmass, side="right")-1
        valid = (bins >= 0) & (bins < len(self.support))
        safe = np.clip(bins, 0, len(self.support)-1)
        valid &= self.support[safe] & (logmass <= self.max_logmass)
        return safe, valid

    def calibrated(self, safe_bins, x):
        raise NotImplementedError

    def base(self, mass, x):
        mass = np.atleast_1d(np.asarray(mass, float))
        safe, valid = self.bins(mass)
        profiles = self.calibrated(safe, x)
        fallback = A.published_profile(self.field, mass, 0., x)
        profiles[~valid] = fallback[~valid]
        return profiles


class FitProfile(Profile):
    """Per-bin parametric fit: parameters[bin] = (P0,xc,beta) or (rho0,alpha,beta)."""

    def __init__(self, field, parameters, label, mass_edges, support, max_logmass):
        super().__init__(field, label, mass_edges, support, max_logmass)
        self.parameters = np.asarray(parameters, float)

    def calibrated(self, safe_bins, x):
        return SHAPES[self.field](self.parameters[safe_bins], x)


class TableProfile(Profile):
    """Tabulated F(bin, x_table); log-log interpolation in x, power-law ends."""

    def __init__(self, field, x_table, table, label, mass_edges, support, max_logmass):
        super().__init__(field, label, mass_edges, support, max_logmass)
        self.x_table = np.asarray(x_table, float)
        self.table = np.asarray(table, float)

    def calibrated(self, safe_bins, x):
        spline = interpolate.interp1d(np.log(self.x_table), np.log(np.maximum(self.table, 1e-300)),
                                      axis=-1, bounds_error=False, fill_value="extrapolate")
        return np.exp(spline(np.log(x)))[safe_bins]


class ScenarioProfile:
    """The 0.02-dex scenario table of sensitivity_projects (A.profile_mass_grid).

    Used to reproduce the earlier analytic DM moments; the table already holds
    its own published fallback and 0.02-dex mass ramps.
    """

    def __init__(self, scenarios, case, field, radius_index, label):
        self.field = field
        self.label = label
        self.radius_index = radius_index
        self.subset = dict(scenarios)
        self.subset["profiles"] = scenarios["profiles"][[case]]

    def base(self, mass, x):
        return A.profile_mass_grid(self.subset, self.radius_index, np.atleast_1d(mass), x)[0, self.field]


def pressure_fit(fit_radius, support=None):
    """B12 pressure fit through fit_radius in {1,2,3,4} R200c (fit_parameters.npz)."""
    s = support or support_definition()
    return FitProfile(0, s["pressure_parameters"][:, int(fit_radius)-1], f"P{int(fit_radius)}",
                      s["mass_edges"], s["support"], s["max_logmass"])


def electron_fit(fit_radius, support=None):
    """B16 electron-density fit through fit_radius in {2,3,4} R200c (radii234)."""
    s = support or support_definition()
    return FitProfile(1, study_fits(1, int(fit_radius)-2), f"E{int(fit_radius)}",
                      s["mass_edges"], s["support"], s["max_logmass"])


# ----------------------------------------------------------------------------
# Windows (arguments in units of R200c)
# ----------------------------------------------------------------------------
def sigmoid_window(x, edge, width_dex):
    """W = [1+(x/x_t)^(1/w_ln)]^-1, a logistic in ln x (DK14 form), w_ln = w_dex ln 10.

    edge may be an array over masses, shape [m, 1], for matched variants.
    """
    return expit(-np.log(x/edge)/(width_dex*np.log(10.)))


def cosine_window(x, edge):
    """The existing taper H.edge_window: cosine over 0.9-1.0 x_t, zero beyond."""
    return H.edge_window(x, edge)


def gaussian_window(x, edge, width):
    """B12 Sect. 5.3 outer taper: 1 inside x_t, exp[-((x-x_t)/s)^2] beyond."""
    return np.where(x <= edge, 1., np.exp(-np.square((x-edge)/width)))


# ----------------------------------------------------------------------------
# Lightcone engine
# ----------------------------------------------------------------------------
C_KMS = A.C_KMS
DM_DX = .125                  # pc cm^-3, compound-Poisson PDF grid spacing
DM_POINTS = 2**16             # grid to 8192 pc cm^-3; larger single-halo DM -> overflow
QUANTILES = np.array([.16, .5, .84, .975, .999])
DM_ORDER = 3                  # b-quadrature order for columns (variance 2e-8 relative)


class Variant:
    """One profile x window x extent on one grid, or an amplitude-matched copy.

    window: None, an array over grid nodes [x], or over catalogue masses [238, x].
    extent: integration end (a grid break); None = grid end.
    jacobian: also accumulate dO/dln F per grid segment and 0.2-dex mass bin
      (alias importance=True).
    derived=(source, reference): u_ell(source)*u_0(reference)/u_0(source) per
      catalogue cell, as the existing taper_Ymatched and shape_matched controls.
    """

    def __init__(self, label, profile=None, grid=None, extent=None, window=None,
                 jacobian=False, derived=None, importance=False):
        self.label = label
        self.profile = profile
        self.grid = grid
        self.window = window
        self.jacobian = bool(jacobian or importance)
        self.derived = derived
        if derived is None:
            self.field = profile.field
            self.extent = float(grid.x[-1] if extent is None else extent)
            self.segments = grid.segments_upto(self.extent)
        else:
            self.field, self.extent, self.segments = None, None, None


class Lightcone:
    """Catalogue counts and velocities, CLASS ingredients, shell volumes, mass bins."""

    def __init__(self):
        hist = dict(np.load(A.COUNTS))
        self.counts = hist["N"]
        self.z_edges = hist["z_edges"]
        self.z = .5*(hist["z_edges"][1:]+hist["z_edges"][:-1])
        self.mass = np.exp(.5*(hist["lnm_edges"][1:]+hist["lnm_edges"][:-1]))/A.H
        self.ingredients = A.Ingredients()
        chi = self.ingredients.da_spline(hist["z_edges"])*(1+hist["z_edges"])
        self.volume = np.maximum(np.diff(chi**3)/3, 1e-30)  # comoving Mpc^3 per sr
        occupied = np.argwhere(self.counts > 0)
        self.cell_z, self.cell_mass = occupied[:, 0], occupied[:, 1]
        self.lookup = np.full(self.counts.shape, -1, int)
        self.lookup[tuple(occupied.T)] = np.arange(len(occupied))
        # Matched catalogue sum(v_LOS^2) per cell (the existing kSZ self-term input).
        fine = dict(np.load(SOURCES["velocities"]))
        assert all(np.array_equal(fine[k], hist[k]) for k in ("N", "z_edges", "lnm_edges"))
        self.v2_sum = fine["sum_vlos2_km2s2"]
        sigma, diagnostics = VELOCITY.sigma_v_los(self.z, self.ingredients.arrays)
        self.sigma_v_lin = sigma
        self.velocity_diagnostics = diagnostics
        support = support_definition()
        self.mass_edges = support["mass_edges"]
        self.mass_bin = np.clip(np.searchsorted(self.mass_edges, np.log10(self.mass), side="right")-1, 0, len(self.mass_edges)-1)
        assert np.log10(self.mass).min() >= self.mass_edges[0]
        self.n_mass_bins = len(self.mass_edges)      # 15 bins + one overflow bin (>= 15.5)


_JOB = {}


def deposit(values, weights, out):
    """Cloud-in-cell deposit of weights at DM values onto the PDF grid; returns overflow weight."""
    position = values/DM_DX
    index = np.floor(position).astype(np.int64)
    fraction = position-index
    inside = index < DM_POINTS-1
    overflow = float(weights[~inside].sum())
    i, f, w = index[inside], fraction[inside], weights[inside]
    out += np.bincount(i, weights=w*(1-f), minlength=DM_POINTS)
    out += np.bincount(i+1, weights=w*f, minlength=DM_POINTS)
    return overflow


def _evaluate_shell(j, chunk):
    """All variants for redshift shell j. Small per-shell results are returned;
    large sums (DM histograms, Jacobians) are added into the chunk accumulators."""
    job = _JOB
    sky, ell, variants = job["sky"], job["ell"], job["variants"]
    index = np.flatnonzero(sky.counts[j] > 0)
    if not len(index):
        return None
    zz = sky.z[j]
    mass = sky.mass[index]
    N = sky.counts[j, index].astype(float)
    ing = sky.ingredients
    da, radius, y_amp, tau_amp = ing.geometry(zz, mass)
    bias = ing.bias(zz, mass)
    pk = ing.pk(zz, ell)
    vol = sky.volume[j]
    q_all = np.c_[np.zeros(len(mass)), radius[:, None]/da*(ell[None]+.5)]
    nv, nl, nm = len(variants), len(ell), len(index)
    use_dm = zz <= 1.
    dm_scale = 1./(DM_FACTOR*(1+zz))
    v2 = sky.v2_sum[j, index]/C_KMS**2                    # sum (v_LOS/c)^2 per cell
    s2_lin = (sky.sigma_v_lin[j]/C_KMS)**2
    column_scale = tau_amp*(da/radius)**2*dm_scale       # DM(b) = column_scale * Sigma_x(b)
    area_scale = N/(4*np.pi)*(radius/da)**2              # halos per unit b-area (x units)
    onehot = np.zeros((nm, sky.n_mass_bins))
    onehot[np.arange(nm), sky.mass_bin[index]] = 1.
    quantiles_only = job["quantiles_only"]
    out = dict(index=index, one=np.zeros((nv, nl)), two=np.zeros((nv, nl)), mean=np.zeros(nv),
               cells=np.zeros((nv, nm)), dm_mean=np.zeros(nv), dm_var=np.zeros(nv), dm_mean_columns=np.zeros(nv),
               dm_one=np.zeros((nv, nl)), dm_two=np.zeros((nv, nl)), lam=np.zeros(nv), lam_flat=np.zeros(nv),
               ksz_one=np.zeros((nv, nl)), ksz_one_lin=np.zeros((nv, nl)), ksz_two=np.zeros((nv, nl)),
               cross_one=np.zeros((len(job["pairs"]), nl)), cross_two=np.zeros((len(job["pairs"]), nl)))
    transforms, columns, pieces_by_v = [None]*nv, [None]*nv, [None]*nv
    W_store = {}

    def tau_terms(v, y):
        out["ksz_one"][v] = np.sum(v2[:, None]*y[:, 1:]**2, axis=0)/(4*np.pi)
        out["ksz_one_lin"][v] = s2_lin*np.sum(N[:, None]*y[:, 1:]**2, axis=0)/(4*np.pi)
        Wb = np.sum(N[:, None]*bias[:, None]*y[:, 1:], axis=0)/(4*np.pi)
        out["ksz_two"][v] = s2_lin*pk*Wb*Wb/vol
        if use_dm:
            dm = y*dm_scale
            out["dm_mean"][v] = N@dm[:, 0]/(4*np.pi)
            out["dm_one"][v] = np.sum(N[:, None]*dm[:, 1:]**2, axis=0)/(4*np.pi)
            Wd = np.sum(N[:, None]*bias[:, None]*dm[:, 1:], axis=0)/(4*np.pi)
            out["dm_two"][v] = pk*Wd*Wd/vol
            W_store[("dm", v)] = Wd
        W_store[("tau_b", v)] = Wb

    def column_terms(v, sigma, setup):
        """DM variance, column mean and the PDF intensity from projected columns."""
        x_b = column_scale[:, None]*sigma                      # DM per b node [pc cm^-3]
        lam_b = area_scale[:, None]*setup["area"][None, :]      # expected hits per b node
        out["dm_var"][v] = np.sum(lam_b*x_b*x_b)
        out["dm_mean_columns"][v] = np.sum(lam_b*x_b)
        out["lam_flat"][v] = np.sum(lam_b)
        row = job["pdf_rows"][v]
        chunk["dm_overflow"][row] += deposit(x_b.ravel(), lam_b.ravel(), chunk["dm_hist"][row])
        return x_b, lam_b

    for grid in job["grids"]:
        members = [v for v in range(nv) if variants[v].grid is grid and variants[v].derived is None]
        x = grid.x
        evolution = {f: A.published_profile(f, mass, zz, x)/A.published_profile(f, mass, 0., x)
                     for f in {variants[v].field for v in members}}
        profiles = []
        for v in members:
            profile = job["base"][v][index]*evolution[variants[v].field]
            window = variants[v].window
            if window is not None:
                profile = profile*(window[index] if window.ndim == 2 else window)
            profiles.append(profile)
        if quantiles_only:
            for k, v in enumerate(members):
                if (variants[v].field == 1 and variants[v].jacobian and use_dm
                        and variants[v].label in job["pdf_reference"]):
                    _quantile_jacobian(v, profiles[k], grid, column_scale, area_scale, onehot, chunk)
            continue
        stacked = np.stack(profiles, axis=-1)                       # [m, x, v]
        partial = np.stack([np.matmul(radial_weights(x[a:b+1], q_all), stacked[:, a:b+1])
                            for a, b in grid.segments])               # [segment, m, ell+1, v]
        cumulative = np.cumsum(partial, axis=0)
        for k, v in enumerate(members):
            var = variants[v]
            amp = y_amp if var.field == 0 else tau_amp
            y = cumulative[var.segments-1, :, :, k]*amp[:, None]
            transforms[v] = y
            out["one"][v] = np.sum(N[:, None]*y[:, 1:]**2, axis=0)/(4*np.pi)
            W = np.sum(N[:, None]*bias[:, None]*y[:, 1:], axis=0)/(4*np.pi)
            W_store[("field", v)] = W
            out["two"][v] = pk*W*W/vol
            out["mean"][v] = N@y[:, 0]/(4*np.pi)
            out["cells"][v] = y[:, 0]
            if job["by_mass"]:
                chunk["by_mass"][v, index] += N[:, None]*y[:, 1:]**2/(4*np.pi)
            if var.field == 1:
                tau_terms(v, y)
                if use_dm:
                    angle = np.minimum(np.pi, var.extent*radius/da)
                    out["lam"][v] = np.sum(N*(1-np.cos(angle))/2)
                    setup = job["columns"][(id(grid), var.extent)]
                    sigma = profiles[k]@setup["A"].T
                    if var.label in job["column_sources"]:
                        columns[v] = sigma
                    x_b, lam_b = column_terms(v, sigma, setup)
            if var.jacobian:
                pieces = partial[:var.segments, :, :, k]*amp[None, :, None]   # [segment, m, ell+1]
                pieces_by_v[v] = pieces
                _jacobian_terms(v, var, y, pieces, W, N, bias, pk, vol, onehot, chunk, W_store, v2, s2_lin, dm_scale, use_dm)
                if var.field == 1 and use_dm:
                    setup = job["columns"][(id(grid), var.extent)]
                    J = chunk["J"][v]
                    for s, (a, b) in enumerate(grid.segments[:var.segments]):
                        g = column_scale[:, None]*(profiles[k][:, a:b+1]@setup["A_seg"][s].T)
                        J["dm_var"][s] += 2*np.sum(lam_b*x_b*g, axis=1)@onehot
                        J["dm_mean_columns"][s] += np.sum(lam_b*g, axis=1)@onehot
    if quantiles_only:
        return out
    for v, var in enumerate(variants):
        if var.derived is None:
            continue
        s, r = (job["labels"].index(name) for name in var.derived)
        scale = np.divide(transforms[r][:, 0], transforms[s][:, 0], out=np.ones(nm), where=transforms[s][:, 0] != 0)
        y = transforms[s]*scale[:, None]
        transforms[v] = y
        out["one"][v] = np.sum(N[:, None]*y[:, 1:]**2, axis=0)/(4*np.pi)
        W = np.sum(N[:, None]*bias[:, None]*y[:, 1:], axis=0)/(4*np.pi)
        W_store[("field", v)] = W
        out["two"][v] = pk*W*W/vol
        out["mean"][v] = N@y[:, 0]/(4*np.pi)
        out["cells"][v] = y[:, 0]
        if variants[s].field == 1:
            tau_terms(v, y)
            if use_dm and columns[s] is not None:
                source = variants[s]
                setup = job["columns"][(id(source.grid), source.extent)]
                column_terms(v, columns[s]*scale[:, None], setup)
                out["lam"][v] = out["lam"][s]
    for p, (yl, el) in enumerate(job["pairs"]):
        if not use_dm:
            continue
        a, e = job["labels"].index(yl), job["labels"].index(el)
        y, dm = transforms[a], transforms[e]*dm_scale
        out["cross_one"][p] = np.sum(N[:, None]*y[:, 1:]*dm[:, 1:], axis=0)/(4*np.pi)
        out["cross_two"][p] = pk*W_store[("field", a)]*W_store[("dm", e)]/vol
        if variants[a].jacobian and variants[e].jacobian:
            J = chunk["J_pair"][p]
            ys, dms = pieces_by_v[a][:, :, 1:], pieces_by_v[e][:, :, 1:]*dm_scale
            J["cross_1h_P"] += np.einsum("sml,ml,mb->sbl", ys, N[:, None]*dm[:, 1:], onehot, optimize=True)/(4*np.pi)
            J["cross_1h_E"] += np.einsum("sml,ml,mb->sbl", dms, N[:, None]*y[:, 1:], onehot, optimize=True)/(4*np.pi)
            Wys = np.einsum("sml,m,mb->sbl", ys, N*bias, onehot, optimize=True)/(4*np.pi)
            Wds = np.einsum("sml,m,mb->sbl", dms, N*bias, onehot, optimize=True)/(4*np.pi)
            J["cross_2h_P"] += pk[None, None]*Wys*W_store[("dm", e)][None, None]/vol
            J["cross_2h_E"] += pk[None, None]*Wds*W_store[("field", a)][None, None]/vol
    return out


def _jacobian_terms(v, var, y, pieces, W, N, bias, pk, vol, onehot, chunk, W_store, v2, s2_lin, dm_scale, use_dm):
    """dO/dln F per (segment, mass bin): F -> F(1+eps) inside one segment and bin."""
    J = chunk["J"][v]
    ys = pieces[:, :, 1:]
    J["one"] += 2*np.einsum("sml,ml,mb->sbl", ys, N[:, None]*y[:, 1:], onehot, optimize=True)/(4*np.pi)
    Ws = np.einsum("sml,m,mb->sbl", ys, N*bias, onehot, optimize=True)/(4*np.pi)
    J["two"] += 2*pk[None, None]*W[None, None]*Ws/vol
    J["mean"] += np.einsum("sm,m,mb->sb", pieces[:, :, 0], N, onehot, optimize=True)/(4*np.pi)
    if var.field != 1:
        return
    J["ksz_one"] += 2*np.einsum("sml,ml,mb->sbl", ys, v2[:, None]*y[:, 1:], onehot, optimize=True)/(4*np.pi)
    J["ksz_one_lin"] += 2*s2_lin*np.einsum("sml,ml,mb->sbl", ys, N[:, None]*y[:, 1:], onehot, optimize=True)/(4*np.pi)
    J["ksz_two"] += 2*s2_lin*pk[None, None]*W_store[("tau_b", v)][None, None]*Ws/vol
    if use_dm:
        dms = ys*dm_scale
        dm = y[:, 1:]*dm_scale
        J["dm_one"] += 2*np.einsum("sml,ml,mb->sbl", dms, N[:, None]*dm, onehot, optimize=True)/(4*np.pi)
        Wds = Ws*dm_scale
        J["dm_two"] += 2*pk[None, None]*W_store[("dm", v)][None, None]*Wds/vol
        J["dm_mean"] += np.einsum("sm,m,mb->sb", pieces[:, :, 0]*dm_scale, N, onehot, optimize=True)/(4*np.pi)


def _quantile_jacobian(v, profile, grid, column_scale, area_scale, onehot, chunk):
    """Pass 2: dQ_p/dln F = sum lambda g f(Q_p - x)/f(Q_p) (Mecke formula for the
    compound-Poisson sum); accumulates the numerator per (quantile, segment, bin)."""
    job = _JOB
    var = job["variants"][v]
    reference = job["pdf_reference"][var.label]
    setup = job["columns"][(id(grid), var.extent)]
    x_b = column_scale[:, None]*(profile@setup["A"].T)
    lam_b = area_scale[:, None]*setup["area"][None, :]
    grid_dm = np.arange(DM_POINTS)*DM_DX
    weights = np.stack([lam_b*np.interp(q-x_b, grid_dm, reference["density"], left=0., right=0.)
                        for q in reference["quantiles"]])                 # [p, m, b]
    J = chunk["J"][v]
    for s, (a, b) in enumerate(grid.segments[:var.segments]):
        g = column_scale[:, None]*(profile[:, a:b+1]@setup["A_seg"][s].T)
        J["dm_quantile_numerator"][:, s] += np.einsum("pmb,mb,mk->pk", weights, g, onehot, optimize=True)


def _evaluate_chunk(shells):
    job = _JOB
    nv, nl = len(job["variants"]), len(job["ell"])
    sky = job["sky"]
    nb = sky.n_mass_bins
    chunk = dict(dm_hist=np.zeros((len(job["pdf_rows_list"]), DM_POINTS)),
                 dm_overflow=np.zeros(len(job["pdf_rows_list"])), J={}, J_pair={})
    if job["by_mass"]:
        chunk["by_mass"] = np.zeros((nv, len(sky.mass), nl))
    for v, var in enumerate(job["variants"]):
        if not var.jacobian or var.derived is not None:
            continue
        ns = var.segments
        J = dict(one=np.zeros((ns, nb, nl)), two=np.zeros((ns, nb, nl)), mean=np.zeros((ns, nb)))
        if var.field == 1:
            for key in ("ksz_one", "ksz_one_lin", "ksz_two", "dm_one", "dm_two"):
                J[key] = np.zeros((ns, nb, nl))
            for key in ("dm_mean", "dm_var", "dm_mean_columns"):
                J[key] = np.zeros((ns, nb))
            J["dm_quantile_numerator"] = np.zeros((len(QUANTILES), ns, nb))
        chunk["J"][v] = J
    for p, (yl, el) in enumerate(job["pairs"]):
        a, e = job["labels"].index(yl), job["labels"].index(el)
        va, ve = job["variants"][a], job["variants"][e]
        if va.jacobian and ve.jacobian:
            assert va.grid is ve.grid and va.segments == ve.segments, "Jacobian pairs share grid and extent"
            chunk["J_pair"][p] = {k: np.zeros((va.segments, nb, nl))
                                  for k in ("cross_1h_P", "cross_1h_E", "cross_2h_P", "cross_2h_E")}
    shell_outputs = {j: _evaluate_shell(j, chunk) for j in shells}
    return shell_outputs, chunk


def _balanced_chunks(sky, count):
    """Deterministic cost-balanced shell chunks (greedy, largest first)."""
    cost = np.array([np.count_nonzero(sky.counts[j]) for j in range(len(sky.z))])
    order = np.argsort(-cost, kind="stable")
    loads = np.zeros(count)
    chunks = [[] for _ in range(count)]
    for j in order:
        if cost[j] == 0:
            continue
        k = int(np.argmin(loads))
        chunks[k].append(int(j))
        loads[k] += cost[j]
    return [sorted(c) for c in chunks if c]


def column_setup(grid, extent, jacobian, order=DM_ORDER):
    """b nodes, area weights and the column operator (and per-segment parts)."""
    stop = grid.index[grid.segments_upto(extent)]
    b, wb = b_quadrature(grid.x[:stop+1], order)
    setup = dict(b=b, area=2*np.pi*b*wb, A=projection_operator(grid.x, b, interval_mask(grid, 0., extent)))
    if jacobian:
        count = grid.segments_upto(extent)
        setup["A_seg"] = [projection_operator(grid.x[a:c+1], b) for a, c in grid.segments[:count]]
    return setup


def compound_poisson_pdf(intensity, overflow=0.):
    """Probability mass on the DM grid of a compound-Poisson sum (exact up to the
    CIC grid): pmf = IFFT(exp(FFT(Lambda) - Lambda_total)), zero-padded x2.
    Events above the grid (overflow) count towards the zero-hit normalisation
    only through exp(-overflow): such rays lie above every reported quantile."""
    total = intensity.sum()+overflow
    spectrum = np.fft.rfft(intensity, 2*DM_POINTS)
    pmf = np.fft.irfft(np.exp(spectrum-total), 2*DM_POINTS)[:DM_POINTS]
    return pmf


def pdf_summary(pmf):
    """Quantiles, mean, variance and density at the quantiles from a pmf."""
    grid = np.arange(DM_POINTS)*DM_DX
    clean = np.maximum(pmf, 0.)
    cdf = np.cumsum(clean)
    # The CIC pmf is a piecewise-linear density; its CDF at x_i+dx/2 is cdf[i].
    half = grid+.5*DM_DX
    quantiles = np.interp(QUANTILES, cdf, half)
    density = clean/DM_DX
    return dict(quantiles=quantiles, mean=float(np.sum(clean*grid)), variance=float(np.sum(clean*grid**2)-np.sum(clean*grid)**2),
                mass_on_grid=float(cdf[-1]), density=density, density_at_quantiles=np.interp(quantiles, grid, density),
                negative_mass=float(-np.sum(pmf[pmf < 0])))


class Engine:
    """Propagate many profile variants through the lightcone in one shell pass."""

    def __init__(self, sky=None):
        self.sky = sky or Lightcone()

    def run(self, variants, ell=ELL, workers=8, by_mass=False, verbose=True, pairs=(),
            pdf_reference=None, chunks_per_worker=2, dm_order=DM_ORDER):
        """Evaluate every variant (each on its own grid) for every redshift shell.

        pairs: (pressure_label, electron_label) for the y x DM cross spectra.
        pdf_reference: {electron label: pdf_summary(...)} switches to pass 2,
          the DM-quantile Jacobians only (needs jacobian variants).
        Shell results are combined in redshift order and chunk sums in chunk
        order, so the outputs do not depend on worker scheduling.
        """
        started, cpu = time.perf_counter(), time.process_time()
        sky = self.sky
        ell = np.asarray(ell, float)
        labels = [v.label for v in variants]
        assert len(set(labels)) == len(labels), "variant labels must be unique"
        grids = []
        for var in variants:
            if var.derived is None and all(var.grid is not g for g in grids):
                grids.append(var.grid)
        base = [None if v.derived else v.profile.base(sky.mass, v.grid.x) for v in variants]
        columns = {}
        for var in variants:
            if var.derived is None and var.field == 1:
                key = (id(var.grid), var.extent)
                needs = any(w.jacobian and w.grid is var.grid and w.extent == var.extent for w in variants)
                if key not in columns or (needs and "A_seg" not in columns[key]):
                    columns[key] = column_setup(var.grid, var.extent, needs, dm_order)
        fields = [v.field if v.derived is None else variants[labels.index(v.derived[0])].field for v in variants]
        pdf_rows_list = [v for v in range(len(variants)) if fields[v] == 1]
        column_sources = {v.derived[0] for v in variants if v.derived is not None}
        pdf_rows = {v: i for i, v in enumerate(pdf_rows_list)}
        _JOB.clear()
        _JOB.update(sky=sky, ell=ell, variants=variants, labels=labels, grids=grids, base=base,
                    columns=columns, column_sources=column_sources, pairs=list(pairs), by_mass=by_mass, pdf_rows=pdf_rows,
                    pdf_rows_list=pdf_rows_list, quantiles_only=pdf_reference is not None,
                    pdf_reference=pdf_reference or {})
        setup = time.perf_counter()-started
        chunks = _balanced_chunks(sky, max(1, workers*chunks_per_worker))
        results = []
        if workers > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                for done, item in enumerate(pool.imap(_evaluate_chunk, chunks)):
                    results.append(item)
                    if verbose:
                        print(f"engine: chunk {done+1}/{len(chunks)}, {time.perf_counter()-started:.0f} s", flush=True)
        else:
            for done, shells in enumerate(chunks):
                results.append(_evaluate_chunk(shells))
                if verbose:
                    print(f"engine: chunk {done+1}/{len(chunks)}, {time.perf_counter()-started:.0f} s", flush=True)
        _JOB.clear()
        timing = dict(setup_seconds=setup, wall_seconds=time.perf_counter()-started,
                      parent_cpu_seconds=time.process_time()-cpu, workers=workers, chunks=len(chunks),
                      variants=len(variants), multipoles=len(ell), radial_nodes=[len(g) for g in grids],
                      peak_RSS_GiB_parent=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                      peak_RSS_GiB_largest_child=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024**2)
        return self._assemble(results, variants, fields, ell, by_mass, timing, pdf_rows_list, pairs, pdf_reference)

    def _assemble(self, results, variants, fields, ell, by_mass, timing, pdf_rows_list, pairs, pdf_reference):
        sky = self.sky
        nz, nl, nv = len(sky.z), len(ell), len(variants)
        shells = {}
        for shell_outputs, _ in results:
            shells.update(shell_outputs)
        one_z, two_z = np.zeros((nv, nz, nl)), np.zeros((nv, nz, nl))
        scalars = {k: np.zeros(nv) for k in ("mean", "dm_mean", "dm_var", "dm_mean_columns", "lam", "lam_flat")}
        spectra = {k: np.zeros((nv, nl)) for k in ("dm_one", "dm_two", "ksz_one", "ksz_one_lin", "ksz_two")}
        cross = {k: np.zeros((len(pairs), nl)) for k in ("cross_one", "cross_two")}
        cells = np.zeros((nv, len(sky.cell_z)))
        for j in range(nz):                      # fixed redshift order: deterministic sums
            out = shells.get(j)
            if out is None:
                continue
            one_z[:, j], two_z[:, j] = out["one"], out["two"]
            for key in scalars:
                scalars[key] += out[key]
            for key in spectra:
                spectra[key] += out[key]
            for key in cross:
                cross[key] += out[key]
            cells[:, sky.lookup[j, out["index"]]] = out["cells"]
        result = dict(labels=np.array([v.label for v in variants]), fields=np.array(fields),
                      extents=np.array([np.nan if v.extent is None else v.extent for v in variants]),
                      ell=ell, cl_1h=one_z.sum(axis=1), cl_2h=two_z.sum(axis=1),
                      cl_1h_by_z=one_z, cl_2h_by_z=two_z, mean=scalars["mean"],
                      dm_mean=scalars["dm_mean"], dm_var=scalars["dm_var"], dm_mean_columns=scalars["dm_mean_columns"],
                      dm_cl_1h=spectra["dm_one"], dm_cl_2h=spectra["dm_two"],
                      ksz_cl_1h=spectra["ksz_one"], ksz_cl_1h_linear_sigma_v=spectra["ksz_one_lin"],
                      ksz_cl_2h_approx=spectra["ksz_two"], lambda_hit=scalars["lam"], lambda_flat=scalars["lam_flat"],
                      cell_zero_mode=cells, z=sky.z, mass=sky.mass, cell_z=sky.cell_z, cell_mass=sky.cell_mass,
                      sigma_v_linear_kms=sky.sigma_v_lin, timing=timing)
        result["cl_total"] = result["cl_1h"]+result["cl_2h"]
        if len(pairs):
            result.update(pair_labels=np.array([f"{a}x{b}" for a, b in pairs]),
                          cross_cl_1h=cross["cross_one"], cross_cl_2h=cross["cross_two"])
        hist = sum(chunk["dm_hist"] for _, chunk in results)
        overflow = sum(chunk["dm_overflow"] for _, chunk in results)
        if by_mass:
            result["cl_1h_by_mass"] = sum(chunk["by_mass"] for _, chunk in results)
        if pdf_rows_list and pdf_reference is None:
            summaries = [pdf_summary(compound_poisson_pdf(hist[r], overflow[r])) for r in range(len(pdf_rows_list))]
            quant = np.full((nv, len(QUANTILES)), np.nan)
            pdf_mean, pdf_var = np.full(nv, np.nan), np.full(nv, np.nan)
            for r, v in enumerate(pdf_rows_list):
                quant[v] = summaries[r]["quantiles"]
                pdf_mean[v], pdf_var[v] = summaries[r]["mean"], summaries[r]["variance"]
            result.update(dm_quantile_levels=QUANTILES, dm_quantiles=quant, dm_pdf_mean=pdf_mean,
                          dm_pdf_variance=pdf_var, dm_intensity=hist, dm_overflow_intensity=overflow,
                          dm_pdf_rows=np.array(pdf_rows_list), dm_grid_dx=DM_DX, dm_grid_points=DM_POINTS,
                          dm_pdf_summaries={variants[v].label: summaries[r] for r, v in enumerate(pdf_rows_list)})
        # Jacobians: sum chunk accumulators in chunk order.
        jac = {}
        for v, var in enumerate(variants):
            if not var.jacobian or var.derived is not None:
                continue
            total = None
            for _, chunk in results:
                part = chunk["J"][v]
                total = {k: a.copy() for k, a in part.items()} if total is None else {k: total[k]+part[k] for k in total}
            if pdf_reference is not None and var.label in pdf_reference:
                reference = pdf_reference[var.label]
                total["dm_quantile"] = total["dm_quantile_numerator"]/reference["density_at_quantiles"][:, None, None]
            jac[var.label] = total
        pair_jac = {}
        for p, (a, e) in enumerate(pairs):
            parts = [chunk["J_pair"][p] for _, chunk in results if p in chunk["J_pair"]]
            if parts:
                pair_jac[f"{a}x{e}"] = {k: sum(part[k] for part in parts) for k in parts[0]}
        result["jacobians"] = jac
        result["pair_jacobians"] = pair_jac
        return result


def save_result(path, result):
    """Write an engine result (timing/PDF summaries as JSON, Jacobians flattened) to npz."""
    skip = ("jacobians", "pair_jacobians", "dm_pdf_summaries")
    arrays = {k: v for k, v in result.items() if k not in skip and not isinstance(v, dict)}
    for k, v in result.items():
        if isinstance(v, dict) and k not in skip:
            arrays[f"{k}_json"] = np.array(json.dumps(C.plain(v)))
    for label, block in result.get("jacobians", {}).items():
        for key, value in block.items():
            arrays[f"J__{label}__{key}"] = value
    for label, block in result.get("pair_jacobians", {}).items():
        for key, value in block.items():
            arrays[f"Jpair__{label}__{key}"] = value
    np.savez_compressed(path, **arrays)


def match_edges(grid, base, target, width_dex, extent=None, iterations=80):
    """Edge x_t' per row so that 4pi int x^2 F0 W(x; x_t', w) dx = target.

    Bisection in ln x_t' (the windowed integral increases with x_t').
    base: [m, x] z=0 profiles; target: [m]; integration to extent (a break).
    """
    lo = np.full(len(base), np.log(grid.x[0]*10))
    hi = np.full(len(base), np.log(grid.x[-1]))
    for _ in range(iterations):
        mid = .5*(lo+hi)
        value = volume_integral(grid, base*sigmoid_window(grid.x, np.exp(mid)[:, None], width_dex), extent)
        above = value > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.exp(.5*(lo+hi))
