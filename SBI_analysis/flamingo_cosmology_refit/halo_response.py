"""One- plus two-halo cosmology response with the preserved pressure geometry.

Tinker08 multiplicity and Tinker10 bias follow the equations implemented in
Colossus (https://bdiemer.bitbucket.io/colossus/). With massive neutrinos we use
the cold+baryon variance/density prescription (arXiv:1311.1514). This is a model
response, not an exact conversion of the HalfDome halo realization.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from scipy.special import j0, roots_legendre

from forward_proposals import projection_table
from prior_model import (G, MSUN, MPC, SIGMA_T, MEC2, ELECTRON_FRACTION,
                         parameters, physical_scales, expansion_e2, H)

DELTA_C = 1.68647


def tinker08(sigma,z,delta_mean):
    # Colossus uses the nearest integer overdensity for this interpolation.
    delta = np.rint(delta_mean)
    grid=np.array([200,300,400,600,800,1200,1600,2400,3200])
    assert np.all((delta>=200)&(delta<=3200))
    coefficients=(
        [.186,.200,.212,.218,.248,.255,.260,.260,.260],
        [1.47,1.52,1.56,1.61,1.87,2.13,2.30,2.53,2.66],
        [2.57,2.25,2.05,1.87,1.59,1.51,1.46,1.44,1.41],
        [1.19,1.27,1.34,1.45,1.58,1.80,1.97,2.24,2.44])
    a0,a,b,c=[np.interp(delta,grid,v) for v in coefficients]
    exponent=10**(-(0.75/np.log10(delta/75))**1.2)
    amplitude=a0*(1+z)**-.14
    a=a*(1+z)**-.06
    b=b*(1+z)**(-exponent)
    return amplitude*((sigma/b)**(-a)+1)*np.exp(-c/sigma**2)


def tinker10(sigma,delta_mean):
    nu=DELTA_C/sigma
    y=np.log10(delta_mean)
    e=np.exp(-(4/y)**4)
    a=0.44*y-.88
    return 1-(1+.24*y*e)*nu**a/(nu**a+DELTA_C**a)+.183*nu**1.5+(.019+.107*y+.19*e)*nu**2.4


def top_hat(x):
    small=abs(x)<1e-3
    result=np.empty_like(x)
    result[small]=1-x[small]**2/10+x[small]**4/280
    q=x[~small]
    result[~small]=3*(np.sin(q)-q*np.cos(q))/q**3
    return result


class HaloTheory:
    def __init__(self,root,name,legacy_paint=False,nmass=32,nz=24,nradial=128):
        root=Path(root)
        self.name=name
        data=np.load(root/"audit"/(name+".npz"))
        metadata=json.loads((root/"audit"/(name+".json")).read_text())
        self.metadata=metadata
        c=metadata["input"]
        support=json.loads((root/"audit/catalogue_support.json").read_text())
        # The old catalogue summary divided by 0.68. Restore native physical M.
        mlo=support["mass_min"]*.68/.6774
        mhi=support["mass_max"]*.68/.6774
        lm,wm=np.polynomial.legendre.leggauss(nmass)
        lz,wz=np.polynomial.legendre.leggauss(nz)
        lower,upper=np.log(mlo),np.log(mhi)
        mass=np.exp(lower+(lm+1)*(upper-lower)/2)
        self.wm=wm*(upper-lower)/2
        lower,upper=np.log1p(support["z_min"]),np.log1p(support["z_max"])
        self.z=np.expm1(lower+(lz+1)*(upper-lower)/2)
        self.wz=wz*(upper-lower)/2*(1+self.z)
        self.nmass,self.nz=nmass,nz
        hz=PchipInterpolator(data["z"],data["H_km_s_Mpc"])(self.z)
        chi=PchipInterpolator(data["z"],data["chi_Mpc"])(self.z)
        self.volume=299792.458/hz*chi**2
        rho0=3*(100*c["h"]*1000/MPC)**2/(8*np.pi*G)*MPC**3/MSUN*metadata["Omega_cb"]
        radius=(3*mass/(4*np.pi*rho0))**(1/3)
        power=RectBivariateSpline(data["z"],np.log(data["k"]),np.log(data["P_cb_Mpc3"]))
        pk=np.exp(power(self.z,np.log(data["k"])))
        # Use finite differences in mass at fixed power, not the quadrature
        # node spacing, to obtain an accurate logarithmic variance derivative.
        def variance(mass_scale):
            window=top_hat(radius[:,None]*mass_scale**(1/3)*data["k"][None,:])**2
            return np.trapz(pk[:,None,:]*window[None,:,:]*data["k"][None,None,:]**3,
                            x=np.log(data["k"]),axis=-1)/(2*np.pi**2)
        sigma=np.sqrt(variance(1))
        step=.001
        derivative=-(np.log(variance(np.exp(step)))-np.log(variance(np.exp(-step))))/(4*step)
        omega_cb_z=metadata["Omega_cb"]*(1+self.z)**3/(hz/(100*c["h"]))**2
        delta=200/omega_cb_z
        self.density=tinker08(sigma,self.z[:,None],delta[:,None])*rho0/mass[None,:]*derivative
        self.bias=tinker10(sigma,delta[:,None])
        assert np.all(self.density>0) and np.all(self.bias>0)
        self.mass=np.broadcast_to(mass,(nz,nmass)).ravel().copy()
        self.redshift=np.broadcast_to(self.z[:,None],(nz,nmass)).ravel()
        en,ew=np.polynomial.legendre.leggauss(3)
        edges=np.append(np.arange(80,7881,200),7980)
        ell=(edges[:-1,None]+edges[1:,None]-1)/2+en*(edges[1:]-edges[:-1])[:,None]/2
        self.bin_weights=ew*(2*ell+1)
        self.bin_weights/=self.bin_weights.sum(axis=1)[:,None]
        self.ell=ell.ravel()
        wave=(self.ell[None,:]+.5)/chi[:,None]
        # Rare nearby halos place the high-ell two-halo kernel beyond kmax.
        # Continue the measured high-k logarithmic slope, never silently use
        # RectBivariateSpline's constant extrapolation outside the table.
        clipped=np.clip(wave,data["k"][0],data["k"][-1])
        self.power_limber=np.exp(power.ev(np.broadcast_to(self.z[:,None],wave.shape).ravel(),np.log(clipped).ravel()).reshape(wave.shape))
        slopes=(np.log(pk[:,-1])-np.log(pk[:,-9]))/np.log(data["k"][-1]/data["k"][-9])
        self.power_limber*=np.where(wave>data["k"][-1],(wave/data["k"][-1])**slopes[:,None],1)
        if legacy_paint:
            self.mass*=.6774/.68
            r200,p200=physical_scales(self.mass,self.redshift,include_radiation=True)
            zg=np.linspace(0,4,20001)
            cg=cumulative_trapezoid(299792.458/(100*H*np.sqrt(expansion_e2(zg))),zg,initial=0)
            distance=np.interp(self.redshift,zg,cg)/(1+self.redshift)
        else:
            rho=3*(np.repeat(hz,nmass)*1000/MPC)**2/(8*np.pi*G)
            r200=(3*self.mass*MSUN/(4*np.pi*200*rho))**(1/3)
            p200=G*self.mass*MSUN*200*rho*(c["Omega_b"]/c["Omega_m"])/(2*r200)
            distance=np.repeat(chi/(1+self.z),nmass)
        angle=r200/(distance*MPC)
        self.amplitude=SIGMA_T/MEC2*ELECTRON_FRACTION*p200*r200*2*np.pi*angle**2
        # Nearby large halos have highly oscillatory Hankel kernels. Increase
        # radial order for those halos instead of aliasing their high-ell power.
        orders=2**np.ceil(np.log2(np.maximum(nradial,8*self.ell.max()*angle+32))).astype(int)
        self.groups=[]
        for order in np.unique(orders):
            indices=np.flatnonzero(orders==order)
            nodes,weights=roots_legendre(int(order))
            radius=(nodes+1)*2
            kernel=j0(angle[indices,None,None]*self.ell[None,:,None]*radius[None,None,:])*(weights*2*radius)
            self.groups.append((indices,radius,kernel))
        table=projection_table(root/"audit/projection_table.npz")
        self.column=RectBivariateSpline(table["logbeta"],table["logq"],table["logcolumn"])
        beam_sigma=(2/60*np.pi/180)/np.sqrt(8*np.log(2))
        self.dl_factor=self.ell*(self.ell+1)/(2*np.pi)*np.exp(-self.ell*(self.ell+1)*beam_sigma**2)

    def components(self,theta):
        p,x,b=parameters(theta,self.mass,self.redshift)
        transform=np.empty((len(self.mass),len(self.ell)))
        for indices,radius,kernel in self.groups:
            logq=np.log(radius[None,:]/x[indices,None])
            logb=np.broadcast_to(np.log(b[indices,None]),logq.shape)
            column=np.exp(self.column.ev(logb.ravel(),logq.ravel()).reshape(logq.shape))
            transform[indices]=np.einsum("hlr,hr->hl",kernel,column,optimize=False)
        yl=(transform*(self.amplitude*p*x)[:,None]).reshape(self.nz,self.nmass,-1)
        weights=self.density*self.wm[None,:]
        one=np.sum(self.wz[:,None]*self.volume[:,None]*np.sum(weights[:,:,None]*yl**2,axis=1),axis=0)
        biased=np.sum((weights*self.bias)[:,:,None]*yl,axis=1)
        two=np.sum(self.wz[:,None]*self.volume[:,None]*self.power_limber*biased**2,axis=0)
        def bins(cl):
            return np.sum((cl*self.dl_factor).reshape(40,3)*self.bin_weights,axis=1)
        return bins(one),bins(two)

    def total(self,theta):
        one,two=self.components(theta)
        return one+two


class CosmologyResponse:
    def __init__(self,root,**resolution):
        self.denominator=HaloTheory(root,"halfdome_native",legacy_paint=True,**resolution)
        self.numerator=HaloTheory(root,"flamingo_D3A",**resolution)

    def __call__(self,theta):
        return self.numerator.total(theta)/self.denominator.total(theta)
