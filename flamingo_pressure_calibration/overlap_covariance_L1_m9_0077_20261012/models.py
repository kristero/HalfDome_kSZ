"""Shell-averaged Battaglia shapes and pixel-averaged finite-sphere projections."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from functools import lru_cache
import pilot_physics as P

PARAM_NAMES={"b12":["P0","xc","beta"],"b16":["rho0","alpha","beta"]}
BOUNDS={"b12":([.001,.01,.5],[1.e4,10.,20.]),"b16":([.1,.1,.5],[1.e7,10.,20.])}

def shape(family,x,parameters):
    p,a,b=parameters
    if family=="b12":
        q=np.asarray(x)/a
        return .5176*p*q**(-.3)*(1+q)**(-b)
    q=np.asarray(x)/.5
    return p*q**(-.2)*(1+q**a)**(-(b-.2)/a)

@lru_cache(maxsize=8)
def shell_grid(edges_tuple):
    edges=np.asarray(edges_tuple)
    nodes,w=leggauss(32)
    lo,hi=edges[:-1,None],edges[1:,None]
    x=lo+(hi-lo)*(nodes+1)/2
    weight=3*x*x*w*(hi-lo)/(2*(hi**3-lo**3))
    return x,weight

def shell(family,parameters,edges=P.EDGES):
    x,w=shell_grid(tuple(edges))
    return np.sum(shape(family,x,parameters)*w,axis=1)

def fit_profile(family,profile,mask,error=None,start=None,model=None):
    lo,hi=np.log(np.asarray(BOUNDS[family]))
    initial=np.log(np.asarray(start if start is not None else (P.B12 if family=="b12" else P.B16)[:3]))
    error=np.ones_like(profile)*.05 if error is None else np.maximum(error,.05)
    evaluate=(lambda p:shell(family,p)) if model is None else model
    def residual(theta):
        prediction=evaluate(np.exp(theta))
        return (np.log(prediction[mask])-np.log(profile[mask]))/error[mask]
    fitted=least_squares(residual,np.clip(initial,lo+1e-7,hi-1e-7),bounds=(lo,hi),
        max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    p=np.exp(fitted.x)
    singular=np.linalg.svd(fitted.jac,compute_uv=False)
    bound=float(np.min(np.minimum(fitted.x-lo,hi-fitted.x)))
    rms=float(np.sqrt(np.mean((evaluate(p)[mask]/profile[mask]-1)**2))*100)
    return {"parameters":p.tolist(),"success":bool(fitted.success),"bound_distance":bound,
        "jacobian_condition":float(singular[0]/max(singular[-1],1e-30)),
        "rms_percent":rms,"nfev":fitted.nfev}

def fit_mask(halos,profiles,indices,softening):
    edge=P.EDGES[:-1]
    common=edge>=max(.04,float(np.max(4*softening/halos["r_200_crit_mpc"][indices])))
    common &= np.all(edge[None,:]>=2*profiles["hsml_median_x"][indices],axis=0)
    common &= np.all(profiles["count"][indices]>=20,axis=0)
    assert common.sum()>=6,common
    return common

@lru_cache(maxsize=8)
def projection_grid(rmax=4.,pixels=64,subpixels=3,n_los=128):
    # Average the model over square pixels, matching the particle-centre maps.
    width=2*rmax/pixels
    nodes,w=leggauss(n_los)
    base=-rmax+(np.arange(pixels)+.5)*width
    offsets=((np.arange(subpixels)+.5)/subpixels-.5)*width
    x=base[:,None]+offsets[None,:]
    xx,yy=np.meshgrid(x.ravel(),x.ravel(),indexing="ij")
    b2=xx*xx+yy*yy
    chord=np.sqrt(np.maximum(rmax*rmax-b2,0))
    radius=np.sqrt(b2[...,None]+(chord[...,None]*nodes)**2)
    radius=np.maximum(radius,1e-15)
    return radius,chord,w

def project(family,parameters,rmax=4.,pixels=64,subpixels=3):
    radius,chord,w=projection_grid(rmax,pixels,subpixels)
    value=np.sum(shape(family,radius,parameters)*w,axis=-1)*chord
    return value.reshape(pixels,subpixels,pixels,subpixels).mean(axis=(1,3))

def annular_mean(image,rmax=4.,edges=P.EDGES):
    n=image.shape[0]
    coordinate=-rmax+(np.arange(n)+.5)*2*rmax/n
    xx,yy=np.meshgrid(coordinate,coordinate,indexing="ij")
    radius=np.hypot(xx,yy)
    means=np.full(len(edges)-1,np.nan)
    counts=np.histogram(radius,bins=edges)[0]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        take=(radius>=lo)&(radius<hi)
        if take.any():means[j]=np.mean(image[take])
    return means,counts
