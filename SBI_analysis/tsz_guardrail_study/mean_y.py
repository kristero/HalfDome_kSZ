"""Independent FIRAS consistency diagnostic; does not alter the sampling prior.

The catalogue contribution is positive but incomplete (missing diffuse gas,
small halos and exterior pressure). A conservative inner-sphere contribution
can therefore flag excessive thermal energy, without a universal Y200/B12 cut.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import h5py
from scipy.integrate import cumulative_trapezoid
from scipy.special import betaln, betainc, logsumexp
sys.path.insert(0,'/lustre/work/kristero10/flamingo_prior_pilot_20260914/code')
from prior_model import physical_scales, expansion_e2, H, MPC, SIGMA_T, MEC2, ELECTRON_FRACTION
from analytic_prior import evolve
from audit import save


def histogram(root, nm, nz):
    path=root/'inputs'/('mean_catalogue_'+str(nm)+'_'+str(nz)+'.npz')
    if path.exists():return np.load(path)
    me=np.linspace(12,15.7,nm+1); ze=np.linspace(np.log(.001),np.log(5),nz+1)
    count=np.zeros((nm,nz)); sm=count.copy(); sz=count.copy()
    with h5py.File('/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5','r') as handle:
        for start in range(0,len(handle['redshift']),1000000):
            m=np.asarray(handle['halo_mass_m200c'][start:start+1000000],dtype=float)/H
            z=np.asarray(handle['redshift'][start:start+1000000],dtype=float)
            good=np.isfinite(m)&np.isfinite(z)&(m>=1e12)&(z>=.001)&(z<=5)&(m<=10**15.7)
            lm,lz=np.log10(m[good]),np.log(z[good])
            for target,weight in ((count,None),(sm,lm),(sz,lz)):
                target+=np.histogram2d(lm,lz,bins=(me,ze),weights=weight)[0]
    keep=count>0
    np.savez(path,mass=10**(sm[keep]/count[keep]),z=np.exp(sz[keep]/count[keep]),count=count[keep])
    assert int(count.sum())==85224251
    return np.load(path)


class MeanY:
    def __init__(self, catalogue):
        self.mass,self.z,self.count=(catalogue[key] for key in ('mass','z','count'))
        r200,p200=physical_scales(self.mass,self.z,include_radiation=True)
        zg=np.r_[0.,np.geomspace(1e-8,5.,32769)]
        chi=cumulative_trapezoid(299792.458/(100*H*np.sqrt(expansion_e2(zg))),zg,initial=0)
        distance=np.interp(self.z,zg,chi)*MPC/(1+self.z)
        self.theta200=r200/distance
        self.factor=self.count*self.theta200**2*SIGMA_T/MEC2*ELECTRON_FRACTION*p200*r200
        assert np.max(4*self.theta200)<np.pi
        self.spherical_lower_factor=np.sinc(4*self.theta200/np.pi)

    def evaluate(self, theta, order=96, cylinder=True):
        p0,xc,beta=evolve(theta,self.mass,self.z)
        assert np.all(beta>2.7)
        inner=np.exp(np.log(p0)+3*np.log(xc)+betaln(2.7,beta-2.7)
                     +np.log(betainc(2.7,beta-2.7,4/(4+xc))))
        lower=np.sum(self.factor*inner*self.spherical_lower_factor)
        result=dict(inner_sphere_lower=float(lower))
        if cylinder:
            nodes,weights=np.polynomial.legendre.leggauss(order)
            upper=np.sqrt(np.maximum(np.log(xc/4),0)+24)
            u=(nodes[None,:]+1)*upper[:,None]/2
            t=u*u; logq=np.log(4/xc)[:,None]+t
            loggeometry=-2*t-np.log1p(np.sqrt(-np.expm1(-2*t)))
            integrand=(3*np.log(4)+3*t-.3*logq-beta[:,None]*np.logaddexp(0,logq)
                +loggeometry+np.log(2*u)+np.log(weights[None,:]*upper[:,None]/2))
            outer=p0*np.exp(logsumexp(integrand,axis=1))
            flux=self.factor*(inner+outer)
            result.update(cylinder_flat_sky=float(np.sum(flux)),
                cylinder_spherical_lower=float(np.sum(flux*self.spherical_lower_factor)))
        return result


def run(root):
    cases=json.loads((root/'inputs/cases.json').read_text())
    cases.update(json.loads((root/'inputs/stress_cases.json').read_text()))
    coarse=MeanY(histogram(root,80,96));fine=MeanY(histogram(root,160,192))
    rows={}
    for name,theta in cases.items():
        # Some deliberately unphysical stress controls can violate beta support.
        if np.min(evolve(theta,fine.mass,fine.z)[2])<=2.7:continue
        a=coarse.evaluate(theta);b=fine.evaluate(theta);c=fine.evaluate(theta,192)
        rows[name]=dict(theta=theta,coarse=a,fine=b,refined_radial=c,
            histogram_relative_change=abs(a['cylinder_flat_sky']/c['cylinder_flat_sky']-1),
            radial_relative_change=abs(b['cylinder_flat_sky']/c['cylinder_flat_sky']-1))
        print('Mean y',name,c['cylinder_flat_sky'],flush=True)
    save(root/'audit/mean_y_cases.json',dict(cases=rows,
        limits={'Fixsen1996_95percent':15e-6,'Sabyr2025_pixel_95percent':8.3e-6,'Fabbian2025_95percent':5.2e-6},
        used_as_prior=False,
        interpretation='Independent observational diagnostic; histogram quadrature is approximate. Inner sphere with spherical-area lower factor is a conservative catalogue contribution, subject to catalogue discretization error. Cylinder adds infinite LOS pressure and assumes locally constant angular diameter distance.',
        caveats='This is not the total cosmic y: excluded halos, diffuse gas and pressure beyond the painting radius add positive contributions. Data confidence limits are not mathematical singularities.'))
    pool=np.load(root/'audit/prior_samples.npz')['analytic_extended'][:8192]
    production=np.load(root/'inputs/production_theta.npy')
    samples={}
    for label,theta in [('analytic_candidate',pool),('production',production)]:
        lower=[]
        for values in theta:
            lower.append(fine.evaluate(values,cylinder=False)['inner_sphere_lower'])
        samples[label+'_theta']=theta;samples[label+'_mean_y_lower']=np.array(lower)
    np.savez_compressed(root/'audit/mean_y_prior_samples.npz',**samples)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.root)
