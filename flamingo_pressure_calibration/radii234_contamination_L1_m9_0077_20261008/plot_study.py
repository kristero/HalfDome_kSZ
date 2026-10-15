"""Publication figures for radius, membership, morphology and scatter controls.

All maps have the same pixels and field of view. A model is integrated through
the same finite sphere as its measured target. Subtraction is measured - model.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
import pilot_physics as P
import measure as M
from models import shell
from projection_model import project_map,projected_profile

HERE=Path(__file__).resolve().parent
OUT=HERE/'products'
FIG=HERE/'figures'
COLORS=['#0072B2','#D55E00','#009E73']
COMP_COLORS=['#0072B2','#E69F00','#CC79A7','#009E73']
MAP_LIMITS={}
plt.rcParams.update({'font.size':15,'axes.labelsize':17,'axes.titlesize':17,
    'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':11,
    'figure.dpi':120,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def mass_title(case):
    lo,hi=case['mass_edges']
    status='' if case['strict_resolution_supported'] else ' | diagnostic'
    return rf'${lo:.1f}\leq\log_{{10}}(M_{{200c}}/M_\odot)<{hi:.1f}$ | $N={case["halos"]}${status}'


def profile_figure(case):
    stats=dict(np.load(OUT/(case['label']+'_profiles.npz')))
    x=(P.EDGES[:-1]+P.EDGES[1:])/2
    fig,axes=plt.subplots(2,3,figsize=(16,8.2),layout='constrained')
    for row,(field,family,ylabel) in enumerate([
            ('pressure','b12',r'$P_e/P_{200}$'),
            ('electrons','b16',r'$\rho_\tau/(f_b\rho_c)$')]):
        mean=stats[f'all_{field}_mean'];std=stats[f'all_{field}_std']
        percent=stats[f'all_{field}_percentiles'];ref=stats[f'{field}_published_reference']
        a,ratio,component=axes[row]
        a.fill_between(x,percent[0],percent[2],color='0.8',label='halo 16-84%')
        a.errorbar(x,mean,yerr=std,fmt='.',color='black',ms=4,lw=.65,capsize=1,label='stack + halo SD')
        a.plot(x,ref,':',color='0.25',lw=2,label='published B12' if row==0 else 'published B16')
        ratio.plot(x,ref/mean,':',color='0.25',lw=2)
        ratios=[ref/mean]
        for j,entry in enumerate(case['selections']['all'][field]):
            if entry['fit'] is None:continue
            curve=shell(family,entry['fit']['parameters'])
            mask=np.asarray(entry['fit_mask'],dtype=bool)
            a.plot(x,curve,color=COLORS[j],lw=1,ls='--',alpha=.65)
            a.plot(x,np.where(mask,curve,np.nan),color=COLORS[j],lw=2,label=rf'{j+2}$R_{{200c}}$ fit')
            ratio.plot(x,curve/mean,color=COLORS[j],lw=1,ls='--',alpha=.65)
            ratio.plot(x,np.where(mask,curve/mean,np.nan),color=COLORS[j],lw=2)
            ratios.append(curve/mean)
        ratio.axhline(1,color='0.4',lw=1)
        ratio.set_ylabel('model / stack')
        for ci,name in enumerate(M.COMPONENTS):
            value=stats[f'{field}_{name}_mean']
            component.plot(x,np.divide(value,mean,out=np.zeros_like(mean),where=mean>0),
                color=COMP_COLORS[ci],lw=2,label=['central','own satellites','other halos','unbound'][ci])
        component.set_ylabel('fraction of shell signal');component.set_ylim(-.015,1.04)
        a.set_yscale('log');a.set_ylabel(ylabel)
        # Log axes cannot show negative lower error ends; retain every upper end.
        all_curves=np.r_[mean+std,ref,*[v*mean for v in ratios[1:]]]
        positive=np.r_[percent[0],all_curves];positive=positive[np.isfinite(positive)&(positive>0)]
        lower=float(positive.min())*.6 if len(positive) else 1e-8
        a.set_ylim(lower,float(np.nanmax(all_curves))*2)
        ratio_values=np.concatenate(ratios)
        ratio_values=ratio_values[np.isfinite(ratio_values)&(ratio_values>0)]
        low=min(.7,float(ratio_values.min())*.8);high=max(1.5,float(ratio_values.max())*1.2)
        if high>10 or low<.05:
            ratio.set_yscale('log');ratio.set_ylim(low,high)
        else:ratio.set_ylim(0,high)
        for ax in (a,ratio,component):
            ax.set_xscale('log');ax.set_xlim(.04,4.1);ax.set_xlabel(r'$r/R_{200c}$')
            for cut in (2,3):ax.axvline(cut,color='0.7',lw=.7,ls='--')
        first_edge=float(P.EDGES[:-1][case['fit_base_mask']].min())
        for ax in (a,ratio):
            ax.axvspan(.04,first_edge,facecolor='0.9',edgecolor='0.7',hatch='///',alpha=.3,zorder=-2)
        if row==0:a.legend(ncol=2,loc='lower left');component.legend(loc='upper left')
    fig.suptitle(mass_title(case),fontsize=19)
    save(fig,case['label']+'_profiles')


def triptych(case,maps,field,ri,selection='all',projected_refit=False):
    family='b12' if field=='pressure' else 'b16'
    components={'all':[0,1,2,3],'no_other':[0,1,3]}[selection]
    q={'pressure':0,'electrons':1,'gas':2}[field]
    entry=case['selections'][selection][field][ri]
    chosen=entry.get('projected_refit') if projected_refit else entry['fit']
    if chosen is None:return
    b=case['bin_index'];n=case['halos']
    target=maps['map_sums'][b,0,1,ri,components,q].sum(axis=0)/n
    pars=np.array(chosen['parameters']);model=project_map(family,pars,ri+2)
    residual=target-model
    cache_key=(case['label'],field,projected_refit)
    if cache_key not in MAP_LIMITS:
        positives=[];differences=[]
        # Share color scales across 2/3/4R and all/no-other controls. Every
        # maximum is retained; the residual scale never clips a large feature.
        selections=('all','no_other') if field=='pressure' else ('all',)
        for control in selections:
            comp=[0,1,2,3] if control=='all' else [0,1,3]
            for index,fit_entry in enumerate(case['selections'][control][field]):
                fit=fit_entry.get('projected_refit') if projected_refit else fit_entry['fit']
                if fit is None:continue
                image=maps['map_sums'][b,0,1,index,comp,q].sum(axis=0)/n
                prediction=project_map(family,fit['parameters'],index+2)
                positives.extend([image[image>0],prediction[prediction>0]])
                differences.append(float(np.abs(image-prediction).max()))
        positive=np.concatenate(positives)
        vmax=float(positive.max());vmin=max(float(np.percentile(positive,2)),vmax*1e-6)
        MAP_LIMITS[cache_key]=(vmin,vmax,max(max(differences),vmax*1e-8))
    vmin,vmax,limit=MAP_LIMITS[cache_key]
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    images=[]
    model_title='Projected refit' if projected_refit else '3D fit projected'
    for ax,image,title in zip(axes,(target,model,residual),('Stack (10 views)',model_title,'Stack - model')):
        is_residual=title=='Stack - model'
        im=ax.imshow(image.T,origin='lower',extent=(-4,4,-4,4),
            cmap='RdBu_r' if is_residual else 'magma',
            norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit) if is_residual else LogNorm(vmin,vmax))
        ax.add_patch(plt.Circle((0,0),ri+2,fill=False,color='0.7',lw=1,ls='--'))
        ax.set_title(title);ax.set_xlabel(r'$x/R_{200c}$');ax.set_ylabel(r'$y/R_{200c}$')
        images.append(im)
    label={'pressure':r'$y/y_{200}$','electrons':r'$\Sigma_\tau/(f_b\rho_c R_{200c})$',
        'gas':r'$\Sigma_{\rm gas}/(f_b\rho_c R_{200c})$'}[field]
    fig.colorbar(images[0],ax=axes[:2],shrink=.82,label=label)
    fig.colorbar(images[2],ax=axes[2],shrink=.82,label='residual '+label)
    symbols=('P_0','x_c','\\beta') if family=='b12' else ('\\rho_0','\\alpha','\\beta')
    ptext=', '.join(rf'${s}={v:.3g}$' for s,v in zip(symbols,pars))
    fig.suptitle(mass_title(case)+f' | {field} | {ri+2}'+r'$R_{200c}$'+'\n'+ptext,fontsize=16)
    suffix='_projected_refit' if projected_refit else ''
    save(fig,case['label']+f'_{selection}_{field}_R{ri+2}_maps'+suffix)


def projected_comparison(case):
    stats=dict(np.load(OUT/(case['label']+'_profiles.npz')))
    x=(P.EDGES[:-1]+P.EDGES[1:])/2
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for ri in range(3):
        entry=case['selections']['all']['pressure'][ri]
        if entry['fit'] is None:continue
        mean=stats[f'all_projected_pressure_mean_R{ri+2}'];std=stats[f'all_projected_pressure_std_R{ri+2}']
        valid=(M.ANNULUS_COUNT>0)&(mean>0)&(P.EDGES[1:]<=ri+2)
        axes[0,ri].errorbar(x[valid],mean[valid],yerr=std[valid],fmt='.',color='black',capsize=1,lw=.7,label='projected stack + halo SD')
        for fit,color,label in [(entry['fit'],COLORS[ri],'3D fit projected'),
                                (entry.get('projected_refit'),'#CC79A7','projected refit')]:
            if fit is None:continue
            prediction=projected_profile('b12',fit['parameters'],ri+2)
            axes[0,ri].plot(x[valid],prediction[valid],color=color,lw=2,label=label)
            axes[1,ri].plot(x[valid],prediction[valid]/mean[valid],color=color,lw=2)
        axes[0,ri].set_yscale('log');axes[0,ri].set_ylabel(r'$y/y_{200}$');axes[0,ri].set_title(f'{ri+2}R sphere')
        axes[1,ri].axhline(1,color='0.6');axes[1,ri].set_ylabel('model / projected stack')
        for ax in axes[:,ri]:ax.set_xscale('log');ax.set_xlim(.2,4.1);ax.set_xlabel(r'$b/R_{200c}$')
    axes[0,0].legend(fontsize=10);fig.suptitle(mass_title(case),fontsize=19)
    save(fig,case['label']+'_projected_profiles')


def contamination_figure(case,maps):
    b=case['bin_index'];x=(P.EDGES[:-1]+P.EDGES[1:])/2
    stats=dict(np.load(OUT/(case['label']+'_profiles.npz')))
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for column,(field,ax) in enumerate(zip(('pressure','electrons','gas'),axes[0])):
        mean=stats[f'all_{field}_mean']
        for key,color,label in [('no_other','#0072B2','exclude other halos'),
                               ('no_satellites_or_other','#D55E00','exclude satellites + other'),
                               ('host_bound','#009E73','central + own satellites')]:
            ax.plot(x,stats[f'{key}_{field}_mean']/mean,color=color,lw=2,label=label)
        if f'disturbed_{field}_mean' in stats and f'less_disturbed_{field}_mean' in stats:
            ax.plot(x,stats[f'disturbed_{field}_mean']/stats[f'less_disturbed_{field}_mean'],
                color='0.3',ls=':',lw=2,label='disturbed / less disturbed')
        if f'major_companion_{field}_mean' in stats and f'no_major_companion_{field}_mean' in stats:
            ax.plot(x,stats[f'major_companion_{field}_mean']/stats[f'no_major_companion_{field}_mean'],
                color='#CC79A7',ls='--',lw=2,label='major companion / no major')
        ax.axhline(1,color='0.65',lw=1);ax.set_xscale('log');ax.set_xlim(.04,4.1)
        ax.set_xlabel(r'$r/R_{200c}$');ax.set_ylabel('profile ratio');ax.set_title(field)
        # Preserve outlying ratios: no hidden hard upper truncation.
        ax.set_ylim(bottom=0);ax.margins(y=.1)
        scatter=axes[1,column]
        for key,color,label in [('all','black','all gas'),('no_other','#0072B2','exclude other halos'),
                               ('no_satellites_or_other','#D55E00','exclude satellites + other'),
                               ('host_bound','#009E73','host bound')]:
            profile=stats[f'{key}_{field}_mean'];sd=stats[f'{key}_{field}_std']
            valid=profile>0
            scatter.plot(x[valid],sd[valid]/profile[valid],color=color,lw=2,label=label)
        scatter.set_xscale('log');scatter.set_yscale('log');scatter.set_xlim(.04,4.1)
        scatter.set_xlabel(r'$r/R_{200c}$');scatter.set_ylabel('halo SD / mean')
    axes[0,0].legend(loc='best',fontsize=9);axes[1,0].legend(loc='best',fontsize=10)
    fig.suptitle(mass_title(case)+f' | disturbed proxy: {case["disturbed_count"]}/{case["halos"]}',fontsize=17)
    save(fig,case['label']+'_contamination')


def overview(cases):
    x=np.array([np.mean(c['mass_edges']) for c in cases])
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for row,field in enumerate(('pressure','electrons')):
        for ri,color in enumerate(COLORS):
            entries=[c['selections']['all'][field][ri] for c in cases]
            value=lambda key: [e.get(key,np.nan) if e.get(key) is not None else np.nan for e in entries]
            axes[row,0].plot(x,value('common_inner_RMS_percent'),'o-',color=color,label=f'{ri+2}R fit')
            axes[row,1].plot(x,value('all_4R_RMS_percent'),'o-',color=color)
            held=[e.get('heldout',{}).get('all_4R_RMS_percent',np.nan) for e in entries]
            axes[row,2].plot(x,held,'o-',color=color)
        for ax,title in zip(axes[row],('Common inner 2R','Common full 4R','Held-out full 4R')):
            ax.set_title(field+': '+title);ax.set_ylabel('fractional RMS (%)')
            ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$');ax.set_yscale('log')
            for c,m in zip(cases,x):
                if not c['strict_resolution_supported']:ax.axvspan(m-.1,m+.1,color='0.8',alpha=.25,zorder=0)
        axes[row,0].legend()
    save(fig,'overview_RMS')
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for j,field in enumerate(('pressure','electrons','gas')):
        for ri,color in enumerate(COLORS):
            fractions=[c['integrated_fractions'][field] for c in cases]
            axes[0,j].plot(x,[f['retained_fraction_of_4R'][ri] for f in fractions],'o-',color=color,label=f'{ri+2}R')
            axes[1,j].plot(x,[f['component_fraction'][ri][2] for f in fractions],'o-',color=color)
            axes[1,j].plot(x,[f['component_fraction'][ri][1] for f in fractions],':',color=color)
        axes[0,j].set_title(field);axes[0,j].set_ylabel('fraction of signal inside 4R');axes[0,j].set_ylim(0,1.04)
        axes[1,j].set_ylabel('bound component fraction');axes[1,j].set_title('Other halos (solid); satellites (dotted)')
        for ax in axes[:,j]:ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$')
    axes[0,0].legend();save(fig,'overview_signal_contamination')
    # Paired mean-fit bootstrap is distinct from the individual-halo spread.
    for field,family in [('pressure','b12'),('electrons','b16')]:
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        labels=[r'$P_0$',r'$x_c$',r'$\beta$'] if family=='b12' else [r'$\rho_0$',r'$\alpha$',r'$\beta$']
        for ri,color in enumerate(COLORS):
            entries=[c['selections']['all'][field][ri] for c in cases]
            for k in range(3):
                for row,key in [(0,'bootstrap_parameter_uncertainty'),(1,'individual_parameter_scatter')]:
                    points=[];low=[];high=[]
                    for e in entries:
                        summary=e.get(key,{})
                        quant=summary.get('percentiles')
                        if quant is None:points.append(np.nan);low.append(np.nan);high.append(np.nan)
                        else:points.append(quant[1][k]);low.append(quant[1][k]-quant[0][k]);high.append(quant[2][k]-quant[1][k])
                    axes[row,k].errorbar(x+(ri-1)*.025,points,yerr=[low,high],fmt='o-',ms=4,capsize=2,color=color,label=f'{ri+2}R')
                    axes[row,k].set_yscale('log');axes[row,k].set_ylabel(labels[k]);axes[row,k].set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$')
        axes[0,0].legend();axes[0,1].set_title('Mean-fit bootstrap: 16-84%')
        axes[1,1].set_title('Accepted individual fits: 16-84%')
        fig.suptitle(field+': correlated parameter distributions',fontsize=19)
        save(fig,'overview_'+field+'_parameters')


def membership_fit_overview(cases):
    x=np.array([np.mean(c['mass_edges']) for c in cases])
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    controls=[('all','black','all gas'),('no_other','#0072B2','exclude other halos'),
              ('no_satellites_or_other','#D55E00','exclude satellites + other'),('host_bound','#009E73','host bound')]
    for row,field in enumerate(('pressure','electrons')):
        for ri,ax in enumerate(axes[row]):
            for selection,color,label in controls:
                rms=[c['selections'][selection][field][ri]['fit']['rms_percent']
                    if c['selections'][selection][field][ri]['fit'] else np.nan for c in cases]
                ax.plot(x,rms,'o-',color=color,label=label,ms=4)
            ax.set_title(field+f': {ri+2}R fit');ax.set_yscale('log');ax.set_ylabel('native-domain RMS (%)')
            ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$')
    axes[0,0].legend(fontsize=10);save(fig,'overview_membership_RMS')
    for selection in ('no_other','no_satellites_or_other'):
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for row,field in enumerate(('pressure','electrons')):
            labels=[r'$P_0$',r'$x_c$',r'$\beta$'] if field=='pressure' else [r'$\rho_0$',r'$\alpha$',r'$\beta$']
            for ri,color in enumerate(COLORS):
                for k,ax in enumerate(axes[row]):
                    mean=[];low=[];high=[]
                    for c in cases:
                        p=c['paired_bootstrap_membership_shifts'][selection][field][ri]['parameter_percent_shift_16_50_84']
                        if p is None:mean.append(np.nan);low.append(np.nan);high.append(np.nan)
                        else:mean.append(p[1][k]);low.append(p[1][k]-p[0][k]);high.append(p[2][k]-p[1][k])
                    ax.errorbar(x+(ri-1)*.025,mean,yerr=[low,high],fmt='o-',color=color,ms=4,capsize=2,label=f'{ri+2}R')
                    ax.axhline(0,color='0.6',lw=1);ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$');ax.set_ylabel(labels[k]+' change (%)')
            axes[row,1].set_title(field)
        axes[0,0].legend();fig.suptitle(selection.replace('_',' ')+' relative to all gas',fontsize=18)
        save(fig,'overview_'+selection+'_parameter_shifts')


def projection_accuracy(cases):
    x=np.array([np.mean(c['mass_edges']) for c in cases])
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for ri in range(3):
        entries=[c['selections']['all']['pressure'][ri] for c in cases]
        for key,color,label in [('matched_sphere_map','#0072B2','3D fit, 10 views'),
                                 ('original_z_view_map','0.5','3D fit, original z'),
                                 ('projected_refit_map','#CC79A7','projected refit, 10 views')]:
            error=[e.get(key,{}).get('relative_L2_percent',np.nan) for e in entries]
            bias=[e.get(key,{}).get('integrated_bias_percent',np.nan) for e in entries]
            axes[0,ri].plot(x,error,'o-',color=color,label=label,ms=4)
            axes[1,ri].plot(x,bias,'o-',color=color,ms=4)
        axes[0,ri].set_title(f'{ri+2}R sphere');axes[0,ri].set_ylabel('whole-map relative L2 (%)')
        axes[0,ri].set_yscale('log');axes[1,ri].axhline(0,color='0.7');axes[1,ri].set_ylabel('integrated model bias (%)')
        for ax in axes[:,ri]:ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$')
    axes[0,0].legend(fontsize=10);save(fig,'overview_projection_accuracy')
    # Compare the same resolved, core-excluded 4R image for every fit radius.
    # Pixel centers are padded by half a pixel diagonal beyond the shell floor.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ri,color in enumerate(COLORS):
        entries=[c['selections']['all']['pressure'][ri] for c in cases]
        for ax,key,ylabel in zip(axes,('relative_L2_percent','integrated_bias_percent'),
                ('core-excluded 4R map L2 (%)','core-excluded 4R bias (%)')):
            values=[e.get('resolved_extrapolated_4R_map',{}).get(key,np.nan) for e in entries]
            ax.plot(x,values,'o-',color=color,label=f'{ri+2}R fit',ms=4)
            ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$');ax.set_ylabel(ylabel)
    axes[0].set_yscale('log');axes[0].legend();axes[1].axhline(0,color='0.7')
    save(fig,'overview_resolved_projection_accuracy')
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for row,field in enumerate(('pressure','electrons')):
        for col,ring in enumerate(('1to2','2to3','3to4')):
            ax=axes[row,col]
            for ri,color in enumerate(COLORS):
                values=[c['selections']['all'][field][ri].get('outer_map_L2_percent',{}).get(ring,np.nan) for c in cases]
                ax.plot(x,values,'o-',color=color,label=f'{ri+2}R fit',ms=4)
            ax.set_title(field+': '+ring.replace('to','-')+'R ring')
            ax.set_ylabel('ring map relative L2 (%)');ax.set_yscale('log')
            ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$')
    axes[0,0].legend();save(fig,'overview_outer_map_accuracy')


def morphology_fit_overview(cases):
    x=np.array([np.mean(c['mass_edges']) for c in cases])
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for row,field in enumerate(('pressure','electrons')):
        for ri,ax in enumerate(axes[row]):
            all_rms=[c['selections']['all'][field][ri]['fit']['rms_percent']
                if c['selections']['all'][field][ri]['fit'] else np.nan for c in cases]
            ax.plot(x,all_rms,'o-',color='black',label='all gas',ms=4)
            for group,color,label in [('major_companion','#CC79A7','major companion'),
                                      ('no_major_companion','#0072B2','no major companion'),
                                      ('large_bound_offset','#D55E00','large bound COM offset')]:
                values=[]
                for c in cases:
                    fits=c['groups'][group].get(field)
                    values.append(fits[ri]['rms_percent'] if fits and fits[ri] else np.nan)
                ax.plot(x,values,'o-',color=color,label=label,ms=4)
            ax.set_title(field+f': {ri+2}R');ax.set_yscale('log');ax.set_ylabel('native-domain RMS (%)')
            ax.set_xlabel(r'$\log_{10}(M_{200c}/M_\odot)$')
    axes[0,0].legend(fontsize=10);save(fig,'overview_morphology_RMS')


def main():
    FIG.mkdir(exist_ok=True)
    from swift_comparison import main as compare_SWIFT
    compare_SWIFT()
    summary=json.loads((OUT/'study_summary.json').read_text());cases=summary['bins']
    maps=dict(np.load(HERE/'projection_stacks.npz'))
    overview(cases);membership_fit_overview(cases);projection_accuracy(cases);morphology_fit_overview(cases)
    for b,case in enumerate(cases):
        case['bin_index']=b
        profile_figure(case);contamination_figure(case,maps)
        projected_comparison(case)
        for ri in range(3):
            for field in ('pressure','electrons','gas'):triptych(case,maps,field,ri)
            triptych(case,maps,'pressure',ri,selection='no_other')
        triptych(case,maps,'pressure',2,projected_refit=True)
        print(case['label'],flush=True)
    (FIG/'figure_inventory.json').write_text(json.dumps(sorted(p.name for p in FIG.glob('*.png')),indent=2)+'\n')


if __name__=='__main__':main()
