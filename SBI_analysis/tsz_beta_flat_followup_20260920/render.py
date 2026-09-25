"""Render saved beta support results without repeating the experiments."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from study import B12, BETA_LOW, BETA_HIGH, savefig, evolved, pressure, column

root = Path(__file__).resolve().parent
plt.rcParams.update({'font.size':15, 'axes.labelsize':17, 'xtick.labelsize':13,
                     'ytick.labelsize':13, 'pdf.fonttype':42})
labels = [r'$\beta_0$', r'$\alpha_{m,\beta}$', r'$\alpha_{z,\beta}$']
data = np.load(root/'results/beta_support.npz')
samples = data['beta']
old = np.load(root/'inputs/production_theta.npy')[:,[2,5,8]]
fig, axes = plt.subplots(1,3,figsize=(15,4.8))
for i,ax in enumerate(axes):
    bins = np.linspace(BETA_LOW[i],BETA_HIGH[i],37)
    for values,name,color in ((old,'Existing 8k','#777777'),
        (samples[data['minimum_beta_cache']>2.7],'Untruncated: cache','#D55E00'),
        (samples[data['minimum_beta_catalogue']>2.7],'Untruncated: catalogue','#0072B2')):
        ax.hist(values[:,i],bins=bins,density=True,histtype='step',linewidth=2,label=name,color=color)
    ax.hlines(1/(BETA_HIGH[i]-BETA_LOW[i]),BETA_LOW[i],BETA_HIGH[i],color='#009E73',
              linewidth=2.5,label='Flat: finite radius')
    ax.axvline(B12[[2,5,8]][i],color='black',linestyle=':',label='Battaglia12')
    ax.set(xlabel=labels[i],xlim=(BETA_LOW[i],BETA_HIGH[i]),ylim=(0,None))
axes[0].set_ylabel('Probability density')
axes[1].legend(fontsize=10)
fig.tight_layout()
savefig(root,'beta_priors',fig)

fit = np.load(root/'results/correlated_flat_catalogue.npz')
weights = fit['weights']
edges = [np.linspace(lo,hi,weights.shape[0]+1) for lo,hi in zip(BETA_LOW,BETA_HIGH)]
fig,axes = plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
pairs = [(0,1),(0,2),(1,2)]
probabilities = [weights.sum(axis=({0,1,2}-{i,j}).pop()) for i,j in pairs]
norm = LogNorm(vmin=1e-6,vmax=max(v.max() for v in probabilities))
for ax,(i,j),probability in zip(axes,pairs,probabilities):
    mesh=ax.pcolormesh(edges[i],edges[j],np.ma.masked_equal(probability.T,0),
                       cmap='viridis',norm=norm,shading='flat')
    ax.set(xlabel=labels[i],ylabel=labels[j])
fig.colorbar(mesh,ax=axes,label='Probability per cell',shrink=.86,pad=.02)
savefig(root,'correlated_flat_beta',fig)

# The upper beta tail is finite but compact; the former size-ratio cut is not
# an energy-divergence theorem. Keep all eight other coefficients at B12.
radius=np.geomspace(.001,3.999,220)
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
arrays=dict(radius=radius, mass_Msun=1e14, redshift=.5)
for beta0,color in zip((2.8,4.35,8.,16.),('#0072B2','black','#D55E00','#009E73')):
    theta=B12.copy()
    theta[2]=beta0
    p0,xc,beta=evolved(theta,mass=1e14,redshift=.5)
    profile=pressure(radius,p0,xc,beta)
    projected=np.array([column(r,p0,xc,beta) for r in radius])
    arrays['pressure_'+str(beta0)]=profile
    arrays['column_'+str(beta0)]=projected
    axes[0].loglog(radius,profile,color=color,label=r'$\beta_0=%g$'%beta0)
    axes[1].loglog(radius,projected,color=color)
axes[0].set(xlabel=r'$r/R_{200}$',ylabel=r'$P_e/P_{200}$')
axes[1].set(xlabel=r'$R_\perp/R_{200}$',ylabel=r'$y/[\sigma_TP_{200}R_{200}/(m_ec^2)]$')
axes[0].legend(fontsize=13)
for ax in axes:ax.grid(alpha=.18)
fig.tight_layout()
savefig(root,'high_beta_profiles',fig)
np.savez_compressed(root/'results/high_beta_profiles.npz',**arrays)
