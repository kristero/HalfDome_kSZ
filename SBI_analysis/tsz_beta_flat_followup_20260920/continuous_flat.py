"""Construct flat physical beta marginals on rigorously allowed whole cells.

This is a separate correlated three-parameter prior, not a production switch.
The condition is beta(M,z)>2.7 on the archived catalogue convex hull only.
It does not require that extrapolation through unused cache corners converge.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from study import BETA_LOW, BETA_HIGH, minimum_beta, savefig


class CorrelatedFlatBetaPrior:
    """Piecewise constant joint density, uniform within each allowed cell.

    Parameter order: beta0, alpha_m_beta, alpha_z_beta. Marginals are flat
    within the saved proportional-fitting tolerance; parameters are correlated.
    No assertion about map accuracy or the full nine-dimensional prior is made.
    """
    numerical_certification = False

    def __init__(self, filename):
        data = np.load(filename)
        self.weights = data['weights']
        self.edges = data['edges']
        self.nbin = self.weights.shape[0]
        self.low, self.high = self.edges[:, 0], self.edges[:, -1]
        self.width = (self.high-self.low)/self.nbin

    def sample(self, count, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        index = rng.choice(self.weights.size, size=count, p=self.weights.ravel())
        cells = np.array(np.unravel_index(index, self.weights.shape)).T
        return self.low + (cells+rng.random((count,3)))*self.width

    def log_prob(self, beta):
        beta = np.asarray(beta, float)
        scalar = beta.ndim == 1
        beta = beta.reshape(-1,3)
        valid = (np.isfinite(beta) & (beta>=self.low) & (beta<=self.high)).all(axis=1)
        result = np.full(len(beta), -np.inf)
        cells = np.minimum(((beta[valid]-self.low)/self.width).astype(int),self.nbin-1)
        probability = self.weights[tuple(cells.T)]
        with np.errstate(divide='ignore'):
            result[valid] = np.log(probability)-np.log(self.width).sum()
        return float(result[0]) if scalar else result


def construct(root):
    hull = np.load(root/'inputs/catalogue_logmass_logredshift_hull.npy')
    nbin = 48
    edges = np.array([np.linspace(lo,hi,nbin+1) for lo,hi in zip(BETA_LOW,BETA_HIGH)])
    lower = np.stack(np.meshgrid(*edges[:,:-1], indexing='ij'),axis=-1).reshape(-1,3)
    upper = lower+(BETA_HIGH-BETA_LOW)/nbin
    minima = np.empty(len(lower))
    for start in range(0,len(lower),512):
        lo, hi = lower[start:start+512], upper[start:start+512]
        # v=ln(1+z)>0, so the smallest redshift exponent always minimizes beta.
        # The minimizing mass exponent depends on the sign of ln(M/Mpivot).
        mterm = np.minimum(lo[:,1:2]*hull[None,:,0], hi[:,1:2]*hull[None,:,0])
        minima[start:start+len(lo)] = np.exp(np.log(lo[:,:1])+mterm+lo[:,2:3]*hull[None,:,1]).min(axis=1)
    support = (minima>2.7*(1+1e-12)).reshape((nbin,)*3)
    assert all(support.any(axis=tuple(j for j in range(3) if j!=i)).all() for i in range(3))
    weights = support.astype(float)
    weights /= weights.sum()
    for iteration in range(20000):
        for axis in range(3):
            marginal = weights.sum(axis=tuple(j for j in range(3) if j!=axis))
            shape = [1,1,1]
            shape[axis] = nbin
            weights *= (1/(nbin*marginal)).reshape(shape)
        error = max(np.max(np.abs(weights.sum(axis=tuple(j for j in range(3) if j!=i))*nbin-1)) for i in range(3))
        if error<1e-10:
            break
    assert error<1e-10
    weights /= weights.sum()
    filename = root/'results/continuous_flat_beta_prior.npz'
    np.savez_compressed(filename,weights=weights,edges=edges,support=support,cell_minimum_beta=minima.reshape(weights.shape))
    prior = CorrelatedFlatBetaPrior(filename)
    draws = prior.sample(131072,np.random.default_rng(20260922))
    actual_minimum = minimum_beta(draws,hull)
    cache=np.array([[np.log(m/1e14),np.log1p(z)] for m in (1e12,10**15.7) for z in (.001,5.)])
    cache_minimum=minimum_beta(draws,cache)
    assert (actual_minimum>2.7).all() and np.isfinite(prior.log_prob(draws)).all()
    assert np.isneginf(prior.log_prob([2.,0.,0.]))
    integral = float(np.exp(prior.log_prob((lower+upper)/2)).sum()*np.prod(prior.width))
    assert abs(integral-1)<1e-12
    record = dict(bins_per_axis=nbin, iterations=iteration+1, marginal_max_relative_error=float(error),
        all_cell_points_certified=True, hull_vertices=len(hull),
        analytic_minimum_beta_on_allowed_cells=float(minima[support.ravel()].min()),
        sampled_count=len(draws), sampled_minimum_beta=float(actual_minimum.min()),
        sampled_violations=int(np.sum(actual_minimum<=2.7)), density_integral=integral,
        sampled_cache_minimum_beta=float(cache_minimum.min()),
        sampled_cache_below_energy_fraction=float(np.mean(cache_minimum<=2.7)),
        sampled_cache_below_los_fraction=float(np.mean(cache_minimum<=.7)),
        support_volume_fraction=float(support.mean()),
        scope='Continuous correlated beta prior on whole allowed cells; catalogue-only energy condition; no pixel, cache or full nine-parameter production certification')
    (root/'results/continuous_flat_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    labels=[r'$\beta_0$',r'$\alpha_{m,\beta}$',r'$\alpha_{z,\beta}$']
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for i,ax in enumerate(axes):
        ax.hist(draws[:,i],bins=edges[i],density=True,histtype='step',color='#0072B2',linewidth=2,label='Sampled prior')
        ax.hlines(1/(BETA_HIGH[i]-BETA_LOW[i]),BETA_LOW[i],BETA_HIGH[i],color='black',linestyle='--',label='Uniform target')
        ax.set(xlabel=labels[i],ylim=(0,None),xlim=(BETA_LOW[i],BETA_HIGH[i]))
    axes[0].set_ylabel('Probability density')
    axes[1].legend(fontsize=12)
    fig.tight_layout()
    savefig(root,'continuous_flat_marginals',fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    pairs=[(0,1),(0,2),(1,2)]
    joint=[weights.sum(axis=({0,1,2}-{i,j}).pop()) for i,j in pairs]
    norm=LogNorm(vmin=1e-6,vmax=max(v.max() for v in joint))
    for ax,(i,j),probability in zip(axes,pairs,joint):
        mesh=ax.pcolormesh(edges[i],edges[j],np.ma.masked_equal(probability.T,0),cmap='viridis',norm=norm,shading='flat')
        ax.set(xlabel=labels[i],ylabel=labels[j])
    fig.colorbar(mesh,ax=axes,label='Probability per cell',shrink=.86,pad=.02)
    savefig(root,'continuous_flat_correlations',fig)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    construct(Path(__file__).resolve().parent)
