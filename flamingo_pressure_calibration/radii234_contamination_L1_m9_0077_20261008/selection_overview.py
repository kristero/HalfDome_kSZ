"""A metadata-only visual of the selected sample and its numerical challenges."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    h=dict(np.load(HERE/'selected_halos.npz'))
    plan=json.loads((HERE/'metadata_plan.json').read_text())
    centre=(h['mass_edges'][:-1]+h['mass_edges'][1:])/2
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13})
    axes[0].plot(centre,plan['population_per_bin'],'o-',label='available centrals',color='0.4')
    axes[0].plot(centre,plan['sample_per_bin'],'o-',label='selected centrals',color='#0072B2')
    axes[0].set_yscale('log');axes[0].set_ylabel('halo count');axes[0].legend(fontsize=12)
    axes[1].scatter(np.log10(h['m200c_msun']),h['ngas_200_crit'],s=3,alpha=.13,color='#0072B2',rasterized=True)
    median=[np.median(h['ngas_200_crit'][h['mass_bin']==b]) for b in range(15)]
    axes[1].plot(centre,median,'o-',color='black',label='selected median')
    axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylabel(r'$N_{\rm gas}(<R_{200c})$');axes[1].legend(fontsize=12)
    fraction=[h['disturbed_proxy'][h['mass_bin']==b].mean() for b in range(15)]
    axes[2].plot(centre,fraction,'o-',color='#D55E00',label='combined flag')
    axes[2].plot(centre,[(h['max_satellite_bound_mass_ratio_R2'][h['mass_bin']==b]>=.25).mean() for b in range(15)],'s--',color='#CC79A7',label='major companion')
    axes[2].plot(centre,[(h['centre_offset_R200c'][h['mass_bin']==b]>.07).mean() for b in range(15)],':',color='0.3',label='bound COM offset')
    axes[2].set_ylim(0,1);axes[2].legend(fontsize=11)
    axes[2].set_ylabel('disturbed-proxy fraction')
    for ax in axes:
        ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$');ax.spines[['top','right']].set_visible(False)
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(folder/('sample_resolution_overview.'+suffix),dpi=200,bbox_inches='tight')
    plt.close(fig)
    result={'selected':len(h['halo_index']),'median_Ngas_per_bin':median,'disturbed_fraction_per_bin':fraction,
            'scope':'SOAP metadata; not a particle profile measurement or resolution-convergence test'}
    (HERE/'selection_overview.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
