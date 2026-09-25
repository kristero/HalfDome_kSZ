"""Separate LOS sensitivity on occupied catalogue support from cache corners."""
import argparse
import json
from pathlib import Path
import numpy as np
from analytic_prior import evolve
from audit import log_column, save


def run(root):
    cases=json.loads((root/'inputs/cases.json').read_text())
    cases.update(json.loads((root/'inputs/stress_cases.json').read_text()))
    hull=np.load(root/'inputs/catalogue_logmass_logredshift_hull.npy')
    catalogue=np.load(root/'inputs/mean_catalogue_160_192.npz')
    rng=np.random.default_rng(20260920)
    chosen=rng.choice(len(catalogue['mass']),64,replace=False)
    mass=np.r_[1e14*np.exp(hull[:,0]),catalogue['mass'][chosen]]
    redshift=np.r_[np.expm1(hull[:,1]),catalogue['z'][chosen]]
    rows=[]
    for name,theta in cases.items():
        _,xc,beta=evolve(theta,mass,redshift)
        for i in range(len(mass)):
            for radius in (.003,4.):
                values=[log_column(radius,xc[i],beta[i],endpoint=end)[0]
                        for end in (1e5,2e5,4e5)]
                rows.append(dict(case=name,mass=float(mass[i]),redshift=float(redshift[i]),
                    radius=radius,source='catalogue hull vertex' if i<len(hull) else 'occupied-bin mean',
                    doubled_endpoint_relative_change=float(abs(np.expm1(values[1]-values[0]))),
                    second_doubling_relative_change=float(abs(np.expm1(values[2]-values[1])))))
    worst=max(rows,key=lambda d:d['doubled_endpoint_relative_change'])
    save(root/'audit/los_catalogue_support.json',dict(tests=len(rows),seed=20260920,
        hull_vertices=len(hull),occupied_bin_samples=64,worst=worst,
        cases={name:dict(max_first_doubling=max(d['doubled_endpoint_relative_change'] for d in rows if d['case']==name),
                        max_second_doubling=max(d['second_doubling_relative_change'] for d in rows if d['case']==name))
               for name in cases},rows=rows,
        interpretation='Empirical occupied-support probes; the relative endpoint error is nonlinear in evolving shape, so hull vertices alone are not a mathematical maximum. This is not a full-map error budget.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
