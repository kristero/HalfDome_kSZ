"""Sparse HEALPix phase tests: flux and Fourier power, not full-map validation.

Each halo is placed at reproducible random full-sky positions. No all-sky
pixel array is allocated. Power is measured before averaging positions so
unbiased mean flux cannot conceal aliasing variance.
"""
import argparse
import json
from pathlib import Path
import sys
import time

PILOT = Path('/lustre/work/kristero10/flamingo_prior_pilot_20260914')
sys.path.insert(0, str(PILOT/'runtime/python'))
sys.path.insert(0, str(PILOT/'code'))
import healpy as hp
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from scipy.special import j0, betaln
from prior_model import physical_scales, expansion_e2, H, MPC
from analytic_prior import evolve
from audit import log_column, save


def direction_samples(count, seed):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2*np.pi, count)
    costheta = rng.uniform(-1, 1, count)
    sintheta = np.sqrt(1-costheta**2)
    direction = np.column_stack([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])
    east = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(count)])
    north = np.cross(direction, east)
    return direction, east, north


def experiment(root, positions):
    table = np.load(PILOT/'audit/projection_table.npz')
    spline = RectBivariateSpline(table['logbeta'], table['logq'], table['logcolumn'])
    cases = json.loads((root/'inputs/cases.json').read_text())
    ell = np.array([0., 200., 1000., 3000., 6000., 7900.])
    direction, east, north = direction_samples(positions, 271828)
    phases = np.arange(8)*np.pi/4
    records = []
    for name, theta in cases.items():
        if name in ('P0_tiny', 'P0_huge'):
            continue  # Exactly the same normalized profile as B12.
        for mass, redshift in ((1e14, .5), (1e15, 2.)):
            p0, xc, beta = [float(value[0]) for value in evolve(theta, np.array([mass]), np.array([redshift]))]
            if beta <= .7:
                continue
            r200, _ = physical_scales(np.array([mass]), np.array([redshift]), include_radiation=True)
            distance = quad(lambda z: 299792.458/(100*H*np.sqrt(expansion_e2(z))), 0, redshift,
                            epsabs=0, epsrel=1e-11)[0]*MPC/(1+redshift)
            theta200 = float(r200[0]/distance)
            def column(radius):
                logq = np.log(np.maximum(radius, 1e-15)/xc)
                if np.any(logq<table['logq'][0]) or np.any(logq>table['logq'][-1]):
                    raise ValueError('Projection interpolation outside measured support')
                return xc*np.exp(spline.ev(np.full(np.shape(logq), np.log(beta)), logq))
            # The lower omitted radial area is bounded by the finite central column.
            reference = []
            inner_radius = max(1e-10, xc*np.exp(table['logq'][0])*1.01)
            for order in (256, 512):
                nodes, weights = np.polynomial.legendre.leggauss(order)
                logr = (nodes+1)/2*np.log(4/inner_radius)+np.log(inner_radius)
                radius = np.exp(logr)
                radial_weight = weights/2*np.log(4/inner_radius)*radius
                solid_angle = 2*np.pi*np.sin(theta200*radius)*theta200*radial_weight
                reference.append(np.sum(j0(ell[:, None]*theta200*radius[None, :])
                    *column(radius)[None, :]*solid_angle[None, :], axis=1))
            true = reference[-1]
            integrated_square = np.sum(column(radius)**2*solid_angle)
            effective_area = true[0]**2/integrated_square
            missing_inner_bound = (np.pi*(theta200*inner_radius)**2
                *2*xc*np.exp(betaln(.7, beta-.7))/true[0])
            assert missing_inner_bound < 1e-12
            direct_errors = []
            for radius in (.003, .1, 1., 4.):
                value, _, _ = log_column(radius, xc, beta)
                direct_errors.append(abs(float(column(radius))/np.exp(value)-1))
            for nside in (4096, 8192):
                started = time.monotonic()
                powers = np.zeros((positions, len(ell)))
                flux = np.zeros(positions)
                count_pixels = np.zeros(positions, int)
                area = hp.nside2pixarea(nside)
                path = root/'audit'/('pixel_'+name+'_'+str(int(mass))+'_'+str(nside)+'.npz')
                reused = path.exists()
                if reused:
                    previous = np.load(path)
                    np.testing.assert_array_equal(previous['theta'], theta)
                    assert previous['flux_ratio'].shape == (positions,)
                    flux, powers, count_pixels = (previous[key] for key in ('flux_ratio','power_ratio','pixel_count'))
                for i, center in enumerate(direction if not reused else []):
                    pixels = hp.query_disc(nside, center, 4*theta200, inclusive=False)
                    count_pixels[i] = len(pixels)
                    vectors = np.array(hp.pix2vec(nside, pixels)).T
                    chord = np.linalg.norm(vectors-center, axis=1)
                    separation = 2*np.arcsin(np.minimum(chord/2, 1.))
                    values = column(separation/theta200)*area
                    flux[i] = values.sum()/true[0]
                    dx, dy = vectors@east[i], vectors@north[i]
                    plane_coordinate = dx[:, None]*np.cos(phases)+dy[:, None]*np.sin(phases)
                    for k, multipole in enumerate(ell):
                        amplitude = np.sum(values[:, None]*np.exp(-1j*multipole*plane_coordinate), axis=0)
                        powers[i, k] = np.mean(abs(amplitude)**2)/true[k]**2
                np.savez_compressed(path, flux_ratio=flux, power_ratio=powers, ell=ell,
                    pixel_count=count_pixels, theta=theta, mass=mass, redshift=redshift)
                record = dict(case=name, mass=mass, redshift=redshift, nside=nside,
                    theta200_arcmin=theta200*180/np.pi*60, xc=xc, beta=beta,
                    mean_flux=float(flux.mean()), median_flux=float(np.median(flux)),
                    flux_quantiles=np.percentile(flux, [1, 16, 84, 99]).tolist(),
                    mean_power=powers.mean(0).tolist(),
                    power_mc_standard_error=(powers.std(0, ddof=1)/np.sqrt(positions)).tolist(),
                    maximum_flux=float(flux.max()), positions=positions,
                    reference_quadrature_change=float(np.max(abs(reference[0]/true-1))),
                    projection_vs_finite_LOS_max_relative_error=max(direct_errors),
                    omitted_inner_flux_upper_bound=float(missing_inner_bound),
                    effective_area_steradian=float(effective_area),
                    effective_pixel_count=float(effective_area/area),
                    exact_random_position_flux_power_lower_bound=float(max(1.,area/effective_area)),
                    pixel_samples_reused=reused,
                    seconds=time.monotonic()-started)
                records.append(record)
                save(root/'audit/pixel_summary.json', dict(rows=records, ell=ell.tolist(),
                    seed=271828, healpy_version=hp.__version__,
                    scope='Single-halo HEALPix sampling with flat-sky Fourier diagnostic; includes no interpolation, beam, mask coupling or halo correlations. Ratios independent of P0. Not a 40-bin full-map certificate.',
                    uncertainty='Finite random-position means can miss rare central hits. Standard errors are diagnostics, not rigorous tail bounds.'))
                print(name, mass, nside, record['mean_flux'], record['mean_power'][0], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=256)
    args = parser.parse_args()
    experiment(args.root, args.positions)
