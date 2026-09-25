"""Finite outer radius is a physical model choice, not a convergence parameter."""
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from projection import volume,shape
ROOT=Path(__file__).resolve().parent
cases=[('B12',.497,4.35),('Small core',.025,4.35),('Steep beta',.497,16.),
       ('Shallow beta',1.13,.644),('Extended core',4.,2.8)]
ell=np.geomspace(80,7980,300);theta200=np.pi/10800
sigma=2*theta200/np.sqrt(8*np.log(2))
nodes,w=roots_legendre(1024)
records=[]
plt.rcParams.update({'font.size':14,'axes.labelsize':16,'legend.fontsize':12})
fig,axes=plt.subplots(1,2,figsize=(12,5))
for label,xc,beta in cases:
    transforms=[]
    for outer in [4.,8.]:
        lo,hi=np.log(1e-11),np.log(outer)
        r=np.exp(lo+(nodes+1)*(hi-lo)/2)
        weights=4*np.pi*r**3*shape(r,xc,beta)*w*(hi-lo)/2
        transforms.append(np.sinc((ell[:,None]+.5)*theta200*r/np.pi)@weights)
    ratio=volume(xc,beta,8)/volume(xc,beta,4)
    records.append(dict(case=label,xc=xc,beta=beta,Y8_over_Y4=ratio,
        interpretation='Same 3D amplitude; added outer gas, not renormalized to fixed Y'))
    color=axes[0].plot(ell,transforms[0]/volume(xc,beta,4),label=label)[0].get_color()
    axes[1].semilogy(ell,np.maximum(transforms[1]**2/transforms[0]**2,1e-8),color=color)
axes[0].set(xlabel=r'$\ell$',ylabel=r'$\widetilde y_\ell/Y_4$',title=r'Sphere $4R_{200c}$')
axes[1].set(xlabel=r'$\ell$',ylabel=r'$C_\ell(X=8)/C_\ell(X=4)$',title=r'$\theta_{200c}=1$ arcmin')
axes[0].legend()
for ax in axes:ax.grid(alpha=.15)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(ROOT/'plots'/('outer_radius_sensitivity.'+ext),dpi=180,bbox_inches='tight')
(ROOT/'results/radius_sensitivity.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
