"""Broad-prior scalar references, volume identity, and uniform-sphere control."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc
from projection import column, direct_column, volume

ROOT=Path(__file__).resolve().parent
LOW=np.array([1,.025,2.8,-.6,-1,-.2,-6,-1.5,-.5])
HIGH=np.array([60,4,16,1.5,.4,.4,.5,3,2])


def main():
    for name in ['inputs','results','plots','logs']:(ROOT/name).mkdir(exist_ok=True)
    started=time.monotonic()
    u=qmc.Sobol(12,scramble=True,seed=20260920).random_base2(14)[:10000]
    theta=LOW+(HIGH-LOW)*u[:,:9]
    mass=10**(12+3.7*u[:,9]);z=np.expm1(np.log(6)*u[:,10])
    xc=theta[:,1]*(mass/1e14)**theta[:,4]*(1+z)**theta[:,7]
    beta=theta[:,2]*(mass/1e14)**theta[:,5]*(1+z)**theta[:,8]
    x=np.exp(np.log(1e-8)+u[:,11]*np.log(4/1e-8))
    # Exact centre and grazing chords, including beta below 0.7, are mandatory.
    extra=np.array([(a,c,b) for a in [0.,1e-10,.1,3.9999999999,4.,4.1]
                    for c in [.00001,.025,.5,4.,70000.] for b in [.05,.644,1.,2.7,16.,1000.]])
    points=np.vstack([np.c_[x,xc,beta],extra])
    values=[];direct_errors=[]
    for i,(a,c,b) in enumerate(points):
        value=column(a,c,b);values.append(value)
        if i%100==0 or i>=10000:
            other=direct_column(a,c,b)
            if max(value,other)>1e-280:
                direct_errors.append(abs(value-other)/max(value,other))
    np.savetxt(ROOT/'inputs/columns.csv',np.c_[points,values],delimiter=',',
               header='x,xc,beta_raw,python_column',comments='')
    identities=[]
    for c,b in [(.5,4.35),(1.13,.644),(.025,16.),(4.,2.8),(70000.,.05),(.001,100.)]:
        # Log impact parameter quadrature resolves compact cores independently.
        scale=volume(c,b)
        projected=quad(lambda t:2*np.pi*np.exp(2*t)*column(np.exp(t),c,b)/scale,
                       -45.,np.log(4.),epsabs=2e-9,epsrel=2e-9,limit=180)[0]
        identities.append(dict(xc=c,beta_raw=b,projected_over_volume=projected))
    uniform=quad(lambda x:2*np.pi*x*2*np.sqrt(16-x*x),0,4,epsabs=1e-11)[0]
    result=dict(columns=len(points),direct_reference_checks=len(direct_errors),
        maximum_direct_relative_error=max(direct_errors),
        finite=bool(np.isfinite(values).all()),nonnegative=bool(np.all(np.array(values)>=0)),
        underflowed_positive_columns=int(np.sum((np.array(values)==0)&(points[:,0]<4))),
        integrated_y_identities=identities,uniform_sphere_ratio=uniform/(4*np.pi*4**3/3),
        seconds=time.monotonic()-started,
        scope='Projection tests, not full-map interpolation or rendering certification')
    (ROOT/'results/profile_python.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['finite'] and result['nonnegative']
    assert max(direct_errors)<1e-7
    assert max(abs(r['projected_over_volume']-1) for r in identities)<1e-7


if __name__=='__main__':main()
