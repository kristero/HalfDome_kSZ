# Hydrostatic P_e from measured halo-component n_e + NFW (Duffy+08) mass, P=0 at 4 R200c,
# non-thermal fraction fitted on bins >= 10^13.5 only, then a cool-core / multiphase factor for all bins.
import os, numpy as np
from scipy.optimize import least_squares
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'tests.py')).read().split('# ---------- Test 1')[0])
def m_nfw(xx,c):
    mu=lambda y: np.log1p(y)-y/(1+y); return mu(c*xx)/mu(c)
from scipy.integrate import cumulative_trapezoid
Ph=np.zeros_like(tau)
for i in range(nz):
    for j in range(nm):
        if not mask[i,j].any(): continue
        M=10**node[i,j]*h; c=5.71*(M/2e12)**-0.084*(1+z[i])**-0.47
        xx=np.linspace(0.01,4.0,4000); E=np.exp(np.interp(np.log(xx),np.log(x),np.log(np.maximum(Ep[i,j],1e-30))))
        cum=cumulative_trapezoid(0.009*E*m_nfw(xx,c)/xx**2,xx,initial=0); Ph[i,j]=0.5176*np.interp(x,xx,cum[-1]-cum)
lr=np.where(mask,np.log(np.where(Ph>0,Pp/np.where(Ph>0,Ph,1),1)),0)   # ln(P_stack/P_HSE)
cl=mask&(LM>=13.5)
def fnth(p): return np.clip(p[0]*(1+Z)**p[1]*X**p[2],0,0.95)
r=least_squares(lambda p:(np.log(1-fnth(p))-lr)[cl],[0.2,0.5,0.8],bounds=([0,-3,0],[2,3,4]))
report('HSE + f_nth=a(1+z)^b x^n fitted on >=10^13.5',np.where(mask,np.log(1-fnth(r.x))-lr,0),3); print('    a,b,n =',np.round(r.x,3))
p3=r.x
def cc(q): xc=np.exp(q[0]+q[1]*lm+q[2]*lz); return np.log((X/xc)**2/(1+(X/xc)**2))
r2=least_squares(lambda q:(np.log(1-fnth(p3))+cc(q)-lr)[mask],[np.log(0.05),-1.5,0.0])
report('  + cool-core factor (x/xc)^2/(1+(x/xc)^2), xc(M,z)',np.where(mask,np.log(1-fnth(p3))+cc(r2.x)-lr,0),6); print('    ln xc0, b_m, b_z =',np.round(r2.x,3))
def joint(p): return (np.log(1-fnth(p[:3]))+cc(p[3:])-lr)[mask]
r3=least_squares(joint,np.r_[p3,r2.x],bounds=([0,-3,0,-12,-8,-8],[2,3,4,3,8,8]))
report('  joint refit of both (6 parameters)',np.where(mask,np.log(1-fnth(r3.x[:3]))+cc(r3.x[3:])-lr,0),6); print('    params',np.round(r3.x,3))
