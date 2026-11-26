import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid
d=np.load(''+__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/tau.npz')
tau,x,z,node,n,b32,lo,Pp,Ep=[d[k] for k in ['tau','x','z','node','n','b32','lo','Pp','Ep']]
A='/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/BC_power_split_cuts_models_20261111/step2_cuts_atlas/remote/out/'
Eall=np.load(A+'atlas_E.npz')['prof_all'][:,:16]
h=0.681
nz,nm,nx=tau.shape
mask=np.zeros_like(tau,bool)
for i in range(nz):
    for j in range(nm):
        if n[i,j]>=50 and np.isfinite(b32[i,j]):
            mask[i,j]=(x>=lo[i,j])&(x<=b32[i,j])&np.isfinite(tau[i,j])&(tau[i,j]>0)
grp=np.where(node<13.5,0,np.where(node<14.3,1,2))
names=['groups <13.5','13.5-14.3','>=14.3']
X=np.broadcast_to(x,tau.shape); Z=np.broadcast_to(z[:,None,None],tau.shape); LM=np.broadcast_to(node[:,:,None],tau.shape)
lm=LM-14; lz=np.log1p(Z)
def report(tag,res,k):
    out=[f'{tag:52s} k={k:2d}']
    for g in range(3):
        s=mask&(np.broadcast_to(grp[:,:,None],tau.shape)==g)
        out.append(f'{names[g]}: {np.sqrt(np.mean(res[s]**2)):.3f}')
    s=mask&(X>=0.3); out.append(f'all x>=0.3: {np.sqrt(np.mean(res[s]**2)):.3f}')
    print(' | '.join(out))
# ---------- Test 1: empirical temperature laws, residual in ln tau = residual in ln P given exact density
lt=np.log(np.where(tau>0,tau,1))
def m_models(p,kind):
    if kind=='universal':      # tau0 * outer decline, no M/z
        t0,xt,c=p; return np.log(t0)-c*np.log1p(X/xt)
    if kind=='selfsim_scaled': # + mass,z power laws on tau0
        t0,am,az,xt,c=p; return np.log(t0)+am*lm+az*lz-c*np.log1p(X/xt)
    if kind=='coolcore6':      # + inner cool-core dip with fixed shape, mass-dependent core radius
        t0,am,az,xt,c,xc0=p
        xc=xc0*np.exp(-1.0*lm)       # placeholder, replaced below
        return None
    if kind=='coolcore8':
        t0,am,az,xt,c,xc0,bm,bz=p
        xc=np.exp(xc0+bm*lm+bz*lz); a=2.0
        return np.log(t0)+am*lm+az*lz-c*np.log1p(X/xt)+np.log((X/xc)**a/(1+(X/xc)**a))
    if kind=='coolcore7':
        t0,am,az,xt,c,xc0,bm=p
        xc=np.exp(xc0+bm*lm); a=2.0
        return np.log(t0)+am*lm+az*lz-c*np.log1p(X/xt)+np.log((X/xc)**a/(1+(X/xc)**a))
    if kind=='coolcore9':
        t0,am,az,xt,c,xc0,bm,bz,a=p
        xc=np.exp(xc0+bm*lm+bz*lz)
        return np.log(t0)+am*lm+az*lz-c*np.log1p(X/xt)+np.log((X/xc)**a/(1+(X/xc)**a))
starts={'universal':[1.5,0.5,1.0],'selfsim_scaled':[1.5,-0.1,-0.3,0.5,1.0],
        'coolcore7':[1.5,-0.1,-0.3,0.5,1.0,np.log(0.05),-1.0],
        'coolcore8':[1.5,-0.1,-0.3,0.5,1.0,np.log(0.05),-1.0,0.0],
        'coolcore9':[1.5,-0.1,-0.3,0.5,1.0,np.log(0.05),-1.0,0.0,2.0]}
lb={'universal':[0,1e-3,0],'selfsim_scaled':[0,-2,-3,1e-3,0],'coolcore7':[0,-2,-3,1e-3,0,-12,-6],
    'coolcore8':[0,-2,-3,1e-3,0,-12,-6,-6],'coolcore9':[0,-2,-3,1e-3,0,-12,-6,-6,0.3]}
ub={'universal':[50,50,10],'selfsim_scaled':[50,2,3,50,10],'coolcore7':[50,2,3,50,10,2,6],
    'coolcore8':[50,2,3,50,10,2,6,6],'coolcore9':[50,2,3,50,10,2,6,6,8]}
print('=== Test 1: P = n_e(measured) x T(model); RMS of ln(P_pred/P_stack) inside [resolution edge, B32], bins n>=50')
best={}
for kind in starts:
    f=lambda p: (m_models(p,kind)-lt)[mask]
    r=least_squares(f,starts[kind],bounds=(lb[kind],ub[kind]),x_scale='jac')
    res=np.where(mask,m_models(r.x,kind)-lt,0); best[kind]=r.x
    report('T law: '+kind,res,len(r.x)); print('    params',np.round(r.x,3))
# ---------- Test 2: hydrostatic equilibrium with NFW total mass (Duffy+08 c200c), as Schneider+25 Sec. 6
def m_nfw(xx,c):
    mu=lambda y: np.log1p(y)-y/(1+y); return mu(c*xx)/mu(c)
def P_hse(i,j,Eprof,xout):
    M=10**node[i,j]*h; c=5.71*(M/2e12)**-0.084*(1+z[i])**-0.47
    xx=np.linspace(0.01,xout,4000); E=np.exp(np.interp(np.log(xx),np.log(x),np.log(np.maximum(Eprof,1e-30))))
    integrand=0.009*E*m_nfw(xx,c)/xx**2
    cum=cumulative_trapezoid(integrand,xx,initial=0); Ptot=cum[-1]-cum
    return 0.5176*np.interp(x,xx,Ptot)          # electron P / P200, thermal only, no non-thermal support
print('\n=== Test 2: hydrostatic P_e from measured n_e and NFW mass (no fitted parameters unless stated)')
for lab,Eprof,xout in [('halo-component n_e, P=0 at 4 R200c',Ep,4.0),('all-gas n_e, P=0 at 6 R200c',Eall,6.0)]:
    Ph=np.zeros_like(tau)
    for i in range(nz):
        for j in range(nm):
            if mask[i,j].any(): Ph[i,j]=P_hse(i,j,Eprof[i,j],xout)
    ratio=np.where(mask,np.log(np.where(Ph>0,Pp/np.where(Ph>0,Ph,1),1)),0)   # ln(P_stack / P_HSE)
    report('HSE, '+lab+', f_nth=0',-ratio,0)
    fz=np.minimum((1+Z)**0.5,(4**-0.8/0.18-1)*np.tanh(0.5*Z)+1)
    fn=np.clip(0.18*fz*X**0.8,0,0.99)
    res=np.where(mask,np.log(1-fn)-ratio,0); report('HSE, '+lab+', Shaw+10 f_nth (fixed)',res,0)
    # implied non-thermal fraction 1 - P_stack/P_HSE at a few radii
    for xs in [0.3,0.5,1.0]:
        jx=np.argmin(abs(x-xs)); row=[]
        for i in [0,6,9]:
            v=[1-np.exp(ratio[i,j,jx]) for j in range(nm) if mask[i,j,jx]]
            row.append(f'z={z[i]:.0f}: '+(f'{np.median(v):+.2f} [{np.min(v):+.2f},{np.max(v):+.2f}]' if v else '-'))
        print(f'    implied 1-P/P_HSE at x={x[jx]:.2f}  ',' ; '.join(row))
    # 2-parameter non-thermal law fitted: f = a (1+z)^b x^0.8 ; plus 3-par with free exponent
    def f2(p): fn=np.clip(p[0]*(1+Z)**p[1]*X**p[2],0,0.99); return (np.log(1-fn)-ratio)[mask]
    r=least_squares(f2,[0.2,0.5,0.8],bounds=([-2,-3,0],[2,3,4]))
    fn=np.clip(r.x[0]*(1+Z)**r.x[1]*X**r.x[2],0,0.99)
    report('HSE, '+lab+', fitted f_nth = a(1+z)^b x^n',np.where(mask,np.log(1-fn)-ratio,0),3); print('    a,b,n =',np.round(r.x,3))
