"""Finite spherical B12 projection in dimensionless R200 units.

beta is always the raw Battaglia exponent here. XGPaint's low-level gNFW
instead receives beta_internal = alpha*beta_raw-gamma. Never interchange them.
"""
import numpy as np
from scipy.integrate import quad


def shape(x, xc, beta):
    return np.exp(-.3*np.log(np.asarray(x)/xc)-beta*np.log1p(np.asarray(x)/xc))


def column(x, xc, beta, outer=4., rtol=2e-11):
    if x >= outer:
        return 0.
    end=np.sqrt((outer-x)*(outer+x))
    def logf(logr):
        return .7*logr+.3*np.log(xc)-beta*np.logaddexp(0.,logr-np.log(xc))
    lo=-np.inf if x==0 else np.log(x)
    peak_r=.7*xc/(beta-.7) if beta>.7 else outer
    logpeak=logf(np.clip(np.log(peak_r),lo,np.log(outer)))
    if x==0:
        value=quad(lambda t: np.exp(logf(t)-logpeak),-np.inf,np.log(outer),
                   epsabs=1e-12,epsrel=rtol,limit=150)[0]
    else:
        value=quad(lambda u:np.exp(logf(np.log(x)+np.log(np.cosh(u)))-logpeak),
                   0.,np.arcsinh(end/x),epsabs=1e-12,epsrel=rtol,limit=150)[0]
    return np.exp(np.log(2*value)+logpeak) if value>0 else 0.


def volume(xc,beta,outer=4.):
    # r^3 p(r) is the log-radius integrand for the pressure volume.
    peak_r=2.7*xc/(beta-2.7) if beta>2.7 else outer
    peak_log=np.log(min(outer,peak_r))
    def logf(t):return 2.7*t+.3*np.log(xc)-beta*np.logaddexp(0,t-np.log(xc))
    peak=logf(peak_log)
    val=quad(lambda t:np.exp(logf(t)-peak),-np.inf,np.log(outer),
             epsabs=1e-12,epsrel=2e-11,limit=150)[0]
    return 4*np.pi*np.exp(np.log(val)+peak)


def direct_column(x,xc,beta,outer=4.):
    """Independent direct-l quadrature with geometric breakpoints near the core."""
    if x>=outer:return 0.
    end=np.sqrt((outer-x)*(outer+x))
    if x==0:
        # l=t^(1/0.7) cancels the central power-law cusp exactly.
        scale=2*xc**.3*end**.7/.7
        f=lambda t:np.exp(-beta*np.log1p(end*t**(1/.7)/xc))
        core=min(1.,(xc/max(beta,1.)/end)**.7)
        breaks=np.unique(np.r_[0.,core*np.geomspace(1e-8,1,10),np.geomspace(core,1.,20),1.])
        return scale*sum(quad(f,a,b,epsabs=1e-13,epsrel=1e-10)[0] for a,b in zip(breaks[:-1],breaks[1:]))
    logp=-.3*np.log(x/xc)-beta*np.log1p(x/xc)
    core=min(end,np.sqrt(x*(x+xc)/max(beta,1.)))
    breaks=np.unique(np.r_[0.,np.geomspace(max(core*1e-5,1e-200),end,32)])
    def f(l):
        r=np.hypot(x,l)
        return np.exp(-.3*np.log(r/xc)-beta*np.log1p(r/xc)-logp)
    integral=sum(quad(f,a,b,epsabs=1e-13,epsrel=1e-10)[0]
                 for a,b in zip(breaks[:-1],breaks[1:]))
    return np.exp(np.log(2*integral)+logp)
