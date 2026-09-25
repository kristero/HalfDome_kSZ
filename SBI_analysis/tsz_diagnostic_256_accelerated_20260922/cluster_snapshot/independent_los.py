"""Independent direct-LOS SciPy check of the 256 prior rows.

Julia uses a normalized sinh/log-radius projector. This check integrates along
physical LOS distance with geometric subintervals and a cusp-removing central
substitution. Ratios to the central column test shape/projection; they do not
independently calibrate the absolute electron-pressure amplitude.
"""
import json
from pathlib import Path
import time
import numpy as np
import toml
from scipy.integrate import quad
from launch import ROOT,atomic_json


def direct_column(x,xc,beta,outer=4.):
    end=np.sqrt((outer-x)*(outer+x))
    if x==0:
        scale=2*xc**.3*end**.7/.7
        core=min(1.,(xc/max(beta,1.)/end)**.7)
        breaks=np.unique(np.r_[0.,core*np.geomspace(1e-8,1,10),np.geomspace(core,1.,20),1.])
        f=lambda t:np.exp(-beta*np.log1p(end*t**(1/.7)/xc))
        return scale*sum(quad(f,a,b,epsabs=1e-13,epsrel=1e-10)[0] for a,b in zip(breaks[:-1],breaks[1:]))
    peak=-.3*np.log(x/xc)-beta*np.log1p(x/xc)
    core=min(end,np.sqrt(x*(x+xc)/max(beta,1.)))
    breaks=np.unique(np.r_[0.,np.geomspace(max(core*1e-5,1e-200),end,32)])
    def integrand(distance):
        radius=np.hypot(x,distance)
        return np.exp(-.3*np.log(radius/xc)-beta*np.log1p(radius/xc)-peak)
    value=sum(quad(integrand,a,b,epsabs=1e-13,epsrel=1e-10)[0] for a,b in zip(breaks[:-1],breaks[1:]))
    return np.exp(np.log(2*value)+peak) if value>0 else 0.


def main():
    started=time.monotonic();theta=np.load(ROOT/'diagnostic_256/theta_design.npy')
    records=[]
    for row,p in enumerate(theta):
        folder=ROOT/f'tests/audit{row//64}/{row:05d}'
        audit=toml.load(folder/'audit.toml');nodes=audit['attempts'][-1]['nodes']
        profiles=np.loadtxt(folder/f'probes_{nodes[0]}.csv',delimiter=',').reshape(64,39,6)
        for index in [0,21,42,63]:
            halo=profiles[index];mass=10.**halo[0,0];z=halo[0,1]
            xc=p[1]*(mass/1e14)**p[4]*(1+z)**p[7]
            beta=p[2]*(mass/1e14)**p[5]*(1+z)**p[8]
            central=direct_column(0,xc,beta)
            assert np.isfinite(central) and central>0
            # Radii above 1e-4 R200 avoid the inherited angular cache floor.
            for sample in [14,24,35,38]:
                x=halo[sample,2]
                independent=direct_column(x,xc,beta)/central
                julia=halo[sample,3]/halo[sample,5]
                relative=abs(independent/julia-1) if julia>1e-10 else None
                records.append(dict(row=row,log10_mass=float(halo[0,0]),z=float(z),x=float(x),
                    scipy_ratio=float(independent),julia_ratio=float(julia),
                    absolute_ratio_error=float(abs(independent-julia)),relative_error=relative))
    max_relative=max(r['relative_error'] for r in records if r['relative_error'] is not None)
    max_absolute=max(r['absolute_ratio_error'] for r in records)
    result=dict(passed=max_relative<1e-7 and max_absolute<1e-8,rows=len(theta),points=len(records),
        max_relative_visible=max_relative,max_absolute_ratio_error=max_absolute,
        seconds=time.monotonic()-started,
        scope='Normalized spherical column shapes; independent direct-distance integration',records=records)
    atomic_json(ROOT/'results/independent_los.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    assert result['passed'],'Independent projection disagreement requires investigation'


if __name__=='__main__':main()
