"""Validate the released mass mapping, stream z=0 halos, and refit cached stacks."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import time
from pathlib import Path
import h5py
import numpy as np
from numpy.polynomial.chebyshev import chebvander2d, chebval2d
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from bigfile import BigFile
import physics as P

ROOT = Path(__file__).resolve().parent
DATA = Path('/lustre/work/Globus-lt/halfdome/full_res')


def polynomial(logmass, z, degree):
    return chebvander2d((logmass - 14.2) / 1.6, 2 * z / .5 - 1, degree)


def predict_mapping(logmass, z, coefficients):
    logmass, z = np.broadcast_arrays(logmass, z)
    return logmass + chebval2d((logmass - 14.2) / 1.6, 2 * z / .5 - 1, coefficients)


def reconstruct_mapping():
    """Recover a smooth released mapping, not a new SO mass measurement."""
    rng = np.random.default_rng(42)
    with h5py.File(DATA/'halos/lightcone_100.hdf5', 'r') as f:
        n = len(f['redshift'])
        start = max(0, n - 5_000_000)
        z = f['redshift'][start:]
        x = np.asarray(f['log10_m_rfof'][start:], float)
        y = np.log10(f['halo_mass_m200c'][start:])
        length = f['Length'][start:]
    valid = (length >= 320) & (z <= .5) & np.isfinite(y)
    # Hold out every fifth row before constructing balanced training cells.
    holdout = valid & (np.arange(len(z)) % 5 == 0)
    train = valid & ~holdout
    chosen = []
    rows = np.flatnonzero(train)
    key = np.floor((x[rows]-12.75)/.1).astype(int)*20 + np.minimum((z[rows]/.025).astype(int),19)
    order = np.argsort(key)
    boundaries = np.r_[0, np.flatnonzero(np.diff(key[order]))+1, len(order)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        ii = rows[order[lo:hi]]
        chosen.extend(rng.choice(ii, min(len(ii),120),replace=False))
    chosen = np.asarray(chosen, int)
    degree = (5, 3)
    matrix = polynomial(x[chosen], z[chosen], degree).reshape(len(chosen), -1)
    coef = np.linalg.lstsq(matrix, y[chosen]-x[chosen], rcond=None)[0].reshape(6, 4)
    test = np.flatnonzero(holdout)
    pred = predict_mapping(x[test], z[test], coef)
    err = 100 * np.expm1(np.log(10) * (pred-y[test]))
    near = z[test] < .1
    summary = dict(training_rows=len(chosen), heldout_rows=len(test),
        heldout_rms_percent=float(np.sqrt(np.mean(err**2))),
        heldout_p95_absolute_percent=float(np.percentile(np.abs(err), 95)),
        near_z0_rows=int(near.sum()),
        near_z0_rms_percent=float(np.sqrt(np.mean(err[near]**2))),
        raw_logmass_training_range=[float(x[chosen].min()), float(x[chosen].max())],
        redshift_training_range=[float(z[chosen].min()), float(z[chosen].max())],
        note='Accuracy against released assigned masses; no validation against direct SO masses.')
    # A separate basis degree and a narrower redshift interval expose mapping sensitivity.
    alternatives = []
    for deg, zmax in [((4, 3), .5), ((5, 2), .3), ((4, 2), .2)]:
        ids = chosen[z[chosen] <= zmax]
        a = polynomial(x[ids], z[ids], deg).reshape(len(ids), -1)
        coeff = np.linalg.lstsq(a, y[ids]-x[ids], rcond=None)[0].reshape(deg[0]+1, deg[1]+1)
        alternatives.append(dict(degree=list(deg), zmax=zmax, coefficients=coeff.tolist()))
    mass_grid = np.linspace(x[chosen].min(), x[chosen].max(), 160)
    main = predict_mapping(mass_grid, np.zeros_like(mass_grid), coef)
    spread = np.stack([100*np.expm1(np.log(10)*(predict_mapping(mass_grid, 0., np.asarray(a['coefficients']))-main)) for a in alternatives])
    summary['z0_alternative_max_abs_percent'] = float(np.max(np.abs(spread)))
    summary['passed'] = bool(summary['heldout_p95_absolute_percent'] < 2. and summary['near_z0_rms_percent'] < 2.)
    P.save_json(ROOT/'mass_mapping.json', dict(summary=summary, coefficients=coef.tolist(), alternatives=alternatives,
        source=str(DATA/'halos/lightcone_100.hdf5'), mass_unit='Msun/h', physical_mass_divide_h=P.TARGET['h']))
    np.savez_compressed(ROOT/'mass_mapping_validation.npz', raw_logmass=x[test[::max(1,len(test)//100000)]],
        redshift=z[test[::max(1,len(test)//100000)]], residual_percent=err[::max(1,len(test)//100000)],
        grid_raw_logmass=mass_grid, z0_logm200c=main, alternatives_percent=spread)
    if not summary['passed']:
        raise RuntimeError('Released mass-mapping validation failed; painting is gated.')
    return coef, summary


def select_snapshot(coef, mapping_summary):
    bf = BigFile(str(DATA/'nbody/seed_100/data/rfof_1.0000'))
    attrs = bf['Header'].attrs
    assert float(attrs['Time'][0]) == 1.
    assert abs(float(attrs['HubbleParam'][0])-P.TARGET['h']) < 1e-10
    m0 = float(attrs['MassTable'][1]) * 1e10
    h = P.TARGET['h']
    count = bf['RFOF/Length'].size
    output = ROOT/'halos_halfdome.h5'
    raw_hist = np.zeros(40, np.int64)
    edges = np.linspace(12.5, 16.5, 41)
    rejected_low = unsupported = 0
    lo_supported, hi_supported = mapping_summary['raw_logmass_training_range']
    with h5py.File(output, 'w') as f:
        datasets = {k:f.create_dataset(k, (0,3) if k in ('position','velocity') else (0,),
            maxshape=(None,3) if k in ('position','velocity') else (None,),
            dtype='f8' if k == 'mass' else 'i8' if k == 'row' else 'f4', chunks=True)
            for k in ['mass','position','velocity','row']}
        saved = 0
        for start in range(0, count, 2_000_000):
            stop = min(start+2_000_000, count)
            length = bf['RFOF/Length'][start:stop]
            ok = length >= 320
            rejected_low += int((~ok).sum())
            ii = np.flatnonzero(ok)
            if not len(ii):
                continue
            x = np.log10(length[ii].astype(float) * m0)
            support = (x >= lo_supported-.005) & (x <= hi_supported)
            unsupported += int((~support).sum())
            mass = 10**predict_mapping(x, 0., coef) / h
            raw_hist += np.histogram(np.log10(mass[support]), edges)[0]
            keep = support & (mass >= 1e13) & (mass < 10**15.5)
            ii = ii[keep]
            if len(ii):
                arrays = dict(mass=mass[keep], position=bf['RFOF/Position'][start:stop][ii]/h,
                    velocity=bf['RFOF/Velocity'][start:stop][ii], row=start+ii)
                for k, d in datasets.items():
                    d.resize(saved+len(ii), axis=0)
                    d[saved:] = arrays[k]
                saved += len(ii)
            P.save_json(ROOT/'selection_progress.json', dict(raw_rows_done=stop,total_raw_rows=count,saved=saved))
        f.attrs.update(box_mpc=float(attrs['BoxSize'][0])/h, redshift=0., mass_definition='released-calibrated M200c',
            mass_unit='physical Msun', position_unit='physical Mpc', velocity_unit='peculiar km/s',
            raw_minimum_particles=320, min_mass_msun=1e13, max_mass_msun=10**15.5)
    P.save_json(ROOT/'selection_summary.json', dict(raw_rows=count, selected_halos=saved,
        rejected_below_320_particles=rejected_low, outside_mapping_support=unsupported,
        calibration_support_raw_logmass=[lo_supported,hi_supported], box_mpc=float(attrs['BoxSize'][0])/h,
        histogram_edges_logmass=edges.tolist(), histogram=raw_hist.tolist(),
        note='Outside-support objects are excluded and counted, not silently extrapolated.'))


def fit_one(task):
    """Fit a geometric shell mean and a paired halo bootstrap."""
    field, component, b, cut, edges, profiles, mask, masses, resolved = task
    nodes,w = leggauss(32)
    lo,hi = edges[:-1,None], edges[1:,None]
    x = lo + (hi-lo)*(nodes+1)/2
    weight = 3*x*x*w*(hi-lo)/(2*(hi**3-lo**3))
    def model(pars):
        return (P.shape(field,x,pars)*weight).sum(axis=1)
    mean = profiles.mean(axis=0)
    sd = profiles.std(axis=0,ddof=1) if len(profiles)>1 else np.zeros_like(mean)
    use = mask & (edges[1:] <= cut+1e-12) & (mean > 0)
    if use.sum() < 6:
        return dict(field=field,component=component,bin=b,cut=cut,valid=False,reason='fewer than six resolved shells')
    # Match fit_study.fit_one: intrinsic fractional SD weights, not SD/sqrt(N).
    error = np.maximum(sd / np.maximum(mean,1e-300), .05)
    bounds = np.log(P.BOUNDS[field])
    initial = np.log(P.battaglia(field, np.exp(np.mean(np.log(masses)))))
    def fit(values, start, errors=error, fit_mask=use):
        active = fit_mask & (values > 0)
        return least_squares(lambda t:(np.log(model(np.exp(t))[active])-np.log(values[active]))/errors[active],
            np.clip(start,bounds[0]+1e-6,bounds[1]-1e-6),bounds=bounds,max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    result = fit(mean,initial)
    rng = np.random.default_rng(20261008+b)
    bootstrap = []
    for _ in range(128):
        rows = rng.integers(0,len(profiles),len(profiles))
        values=profiles[rows]
        mu=values.mean(axis=0)
        sigma=values.std(axis=0,ddof=1)
        r = fit(mu,result.x,np.maximum(sigma/np.maximum(mu,1e-300),.05))
        if r.success:
            bootstrap.append(np.exp(r.x).tolist())
    pred = model(np.exp(result.x))
    # An empirical halo-parameter distribution: this is distinct from mean-fit bootstrap error.
    individual = []
    individual_rows = []
    for row,values in enumerate(profiles):
        own_mask=resolved[row] & (edges[1:]<=cut+1e-12) & (values>0)
        if np.count_nonzero(own_mask) >= 6:
            r = fit(values,result.x,np.full(len(values),.05),own_mask)
            if r.success and np.min(np.minimum(r.x-bounds[0],bounds[1]-r.x)) > .02:
                individual.append(np.exp(r.x).tolist())
                individual_rows.append(row)
    return dict(field=field,component=component,bin=b,cut=cut,valid=bool(result.success),
        parameters=np.exp(result.x).tolist(),mean=mean.tolist(),sd=sd.tolist(),prediction=pred.tolist(),
        resolved_mask=use.tolist(),mass_pivot_msun=float(np.exp(np.mean(np.log(masses)))),n=len(profiles),
        rms_fit_percent=float(100*np.sqrt(np.mean((pred[use]/mean[use]-1)**2))),
        rms_inner_percent=float(100*np.sqrt(np.mean((pred[use & (edges[1:]<=.75)]/mean[use & (edges[1:]<=.75)]-1)**2)))
            if np.any(use & (edges[1:]<=.75)) else None,
        bootstrap_parameters=bootstrap,individual_parameters=individual,individual_rows=individual_rows,
        bound_distance=float(np.min(np.minimum(result.x-bounds[0],bounds[1]-result.x))),
        uncertainty_scope='Bootstrap: finite-halo uncertainty of empirical mean with diagonal weights. Individual fits: intrinsic profile diversity plus fit noise/degeneracy; neither is full inference covariance.')


def refit_cached(workers):
    with np.load(ROOT/'inputs/selected_halos.npz') as d:
        halos = dict(d)
    with np.load(ROOT/'inputs/measurements.npz') as d:
        data = dict(d)
    edges = data['edges']
    tasks = []
    for b in range(15):
        rows = np.flatnonzero(halos['mass_bin']==b)
        r = halos['r_200_crit_mpc'][rows]
        count = data['sums'][rows,:,0,:].sum(axis=1)
        resolved = (edges[:-1][None,:] >= np.maximum(.04,4*.0057/r)[:,None]) & (count >= 20)
        resolved &= edges[:-1][None,:] >= 2*data['hsml_median_upper_x'][rows]
        strict = resolved.all(axis=0)
        fallback = (count.sum(axis=0)>=200) & (np.mean(count>0,axis=0)>=.5)
        fallback &= edges[:-1] >= max(.04,float(np.max(4*.0057/r)))
        mask = strict if np.count_nonzero(strict & (edges[1:]<=2))>=6 else fallback
        for field,key in [('pressure','pressure_components'),('electrons','electron_components')]:
            for component,ids in [('all',[0,1,2,3]),('no_other',[0,1,3])]:
                profiles = data[key][rows][:,ids].sum(axis=1)
                for cut in [1,2,3,4]:
                    tasks.append((field,component,b,cut,edges,profiles,mask,halos['m200c_msun'][rows],resolved))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_one,tasks))
    P.save_json(ROOT/'fits.json', dict(edges=edges.tolist(), mass_edges=halos['mass_edges'].tolist(),
        source_hashes={name:hashlib.sha256((ROOT/'inputs'/name).read_bytes()).hexdigest()
            for name in ['selected_halos.npz','measurements.npz']},results=results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=26)
    args = parser.parse_args()
    start = time.time()
    P.save_json(ROOT/'cosmology_transfer.json',dict(source=P.SOURCE,target=P.TARGET,ratios=P.cosmology_ratios()))
    coef,summary = reconstruct_mapping()
    select_snapshot(coef,summary)
    refit_cached(args.workers)
    P.save_json(ROOT/'prepare_complete.json',dict(completed=True,wall_seconds=time.time()-start))


if __name__ == '__main__':
    main()
