import numpy as np
from scipy.constants import electron_mass, proton_mass
A='/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/BC_power_split_cuts_models_20261111/step2_cuts_atlas/remote/out/'
P=np.load(A+'atlas_P.npz'); E=np.load(A+'atlas_E.npz')
mu_xgp=(electron_mass+proton_mass*2/1.76)/.9/proton_mass
conv=200*mu_xgp/0.59   # T_e / T200 with T200 = 0.59 m_p G M200c / (2 R200c)
x=P['x70']; z=P['redshift']; lab=P['entry_labels'][:16]; node=P['node'][:,:16]; n=P['n'][:,:16]
ic=list(P['cut_ids']).index('C5_signal32')
b32=np.minimum(P['cut_first'][:,:16,ic],E['cut_first'][:,:16,ic])
lo=np.maximum(P['first_lo_d3'][:,:16],E['first_lo_d3'][:,:16])
tau=conv*P['prof_a2ii'][:,:16]/E['prof_a2ii'][:,:16]
print('mu_xgp/m_p',mu_xgp,'conv',conv)
xs=[0.1,0.2,0.3,0.5,0.7,1.0,1.3]
ix=[np.argmin(abs(x-v)) for v in xs]
print('x nodes',np.round(x[ix],3))
for iz in [0,4,6,8,9]:
    print(f'--- z={z[iz]}  T_e/T200 at x (only inside [lo,B32]; n>=50)')
    for im in range(16):
        if n[iz,im]<50: continue
        row=[]
        for j in ix:
            ok = lo[iz,im]<=x[j]<=b32[iz,im]
            row.append(f'{tau[iz,im,j]:6.2f}' if ok else '   -  ')
        print(f'{lab[im]:>10} n={int(n[iz,im]):7d} B32={b32[iz,im]:.2f} lo={lo[iz,im]:.2f} |',' '.join(row))
np.savez(''+__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/tau.npz',tau=tau,x=x,z=z,node=node,n=n,b32=b32,lo=lo,
         Pp=P['prof_a2ii'][:,:16],Ep=E['prof_a2ii'][:,:16],Pse=P['prof_a2ii_se'][:,:16],Ese=E['prof_a2ii_se'][:,:16],edges=P['edges70'])
