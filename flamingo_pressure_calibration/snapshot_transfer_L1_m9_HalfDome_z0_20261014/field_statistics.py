"""Common probability and flat-map power definitions for both simulations."""
import numpy as np

EDGES = dict(pressure=np.linspace(-23,-9,281),electrons=np.linspace(-12,3,301),
             y=np.linspace(-14,-3,221),tau=np.linspace(-12,-1,221),dm=np.linspace(-7,5,241))
EDGES['deltaT'] = np.r_[-np.geomspace(1000,1e-6,181),0.,np.geomspace(1e-6,1000,181)]


def probability(values,name):
    """Probability per bin, with a separately reported exact-zero atom."""
    values = np.asarray(values).ravel()
    if not np.isfinite(values).all():
        raise ValueError('Non-finite field in probability distribution')
    if name != 'deltaT' and np.any(values<0):
        raise ValueError('Negative pressure/density/positive column')
    zero = values == 0
    nonzero = values[~zero]
    x = nonzero if name=='deltaT' else np.log10(nonzero)
    edges = EDGES[name]
    hist = np.histogram(x,bins=edges)[0].astype(float)/len(values)
    atoms = np.array([zero.mean(),np.count_nonzero(x<edges[0])/len(values),
        np.count_nonzero(x>edges[-1])/len(values)])
    assert abs(hist.sum()+atoms.sum()-1)<1e-10
    return hist,atoms


def power2d(image,side):
    """Mean-subtracted P_2D(k)=area*<|FFT(image)/N^2|^2>, no window deconvolution."""
    n = image.shape[0]
    assert image.shape==(n,n)
    delta = image.astype(float)-np.mean(image,dtype=float)
    spectrum = np.abs(np.fft.rfft2(delta)/n**2)**2*side**2
    kx = np.fft.fftfreq(n,d=side/n)*2*np.pi
    ky = np.fft.rfftfreq(n,d=side/n)*2*np.pi
    k = np.hypot(kx[:,None],ky[None,:])
    edges = np.geomspace(2*np.pi/side, np.pi*n/side, 41)
    sums = np.histogram(k,bins=edges,weights=spectrum)[0]
    counts = np.histogram(k,bins=edges)[0]
    result = np.divide(sums,counts,out=np.full(len(sums),np.nan),where=counts>0)
    return np.sqrt(edges[1:]*edges[:-1]),result,counts


def rebin_cube(cube,factor=2):
    n = cube.shape[0]
    assert cube.shape==(n,n,n) and n%factor==0
    return cube.reshape(n//factor,factor,n//factor,factor,n//factor,factor).mean(axis=(1,3,5))
