"""Pixel-averaged, finite-sphere projections on a shared 0.125R pixel grid."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
import measure as M
from models import shape


@lru_cache(maxsize=3)
def quadrature(radius):
    n=M.NPIX;sub=3;dx=2*M.EXTENT/n
    coordinate=-M.EXTENT+(np.arange(n)+.5)*dx
    offset=((np.arange(sub)+.5)/sub-.5)*dx
    xx,yy=np.meshgrid((coordinate[:,None]+offset).ravel(),(coordinate[:,None]+offset).ravel(),indexing='ij')
    # Square-grid symmetry removes duplicate LOS integrals without approximation.
    b2,inverse=np.unique(np.round((xx*xx+yy*yy).ravel(),14),return_inverse=True)
    chord=np.sqrt(np.maximum(radius*radius-b2,0))
    z,w=leggauss(128)
    r=np.maximum(np.sqrt(b2[:,None]+(chord[:,None]*z)**2),1e-15)
    return r,chord,w,inverse


def project_map(family,parameters,radius):
    r,chord,w,inverse=quadrature(float(radius))
    integrated=(shape(family,r,parameters)*w).sum(axis=1)*chord
    return integrated[inverse].reshape(M.NPIX,3,M.NPIX,3).mean(axis=(1,3))


def projected_profile(family,parameters,radius):
    return M.annular_means(project_map(family,parameters,radius))


def relative_rms(predicted,measured,mask):
    take=mask&np.isfinite(predicted)&np.isfinite(measured)&(measured>0)
    if not take.any():return None
    return float(100*np.sqrt(np.mean((predicted[take]/measured[take]-1)**2)))


def map_metrics(model,truth):
    residual=model-truth
    return {'relative_L2_percent':float(100*np.linalg.norm(residual)/max(np.linalg.norm(truth),1e-100)),
            'integrated_bias_percent':float(100*(model.sum()/max(truth.sum(),1e-100)-1))}
