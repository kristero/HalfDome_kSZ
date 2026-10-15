"""Verify the 57 comparisons and render the scientific comparison figures."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
import physics as P
import field_statistics as S
from run_comparisons import cases

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'
LABELS=dict(pressure=r'$P_e\ [\mathrm{Pa}]$',electrons=r'$n_e\ [\mathrm{cm}^{-3}]$',
    y=r'$y$',tau=r'$\tau$',dm=r'$\mathrm{DM}\ [\mathrm{pc\ cm}^{-3}]$',deltaT=r'$\Delta T_{\rm kSZ}\ [\mu\mathrm{K}]$')
plt.rcParams.update({'font.size':15,'axes.labelsize':17,'axes.titlesize':17,'xtick.labelsize':13,
    'ytick.labelsize':13,'legend.fontsize':11,'figure.dpi':130,'savefig.dpi':210,
    'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none'})


def save(fig,name,vector=True):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    if vector:
        fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def read_stats(path):
    with h5py.File(path,'r') as f:
        return {k:f['statistics'][k][:] for k in f['statistics']}


def read_maps(path,region=0):
    with h5py.File(path,'r') as f:
        side=float(f.attrs.get('side_mpc',f.attrs.get('region_side_mpc',250.)))
        n=int(f.attrs.get('map_n',512))
        return P.map_observables(f['region_sums'][region],(side/n)**2)


def verified_inputs():
    receipts=[]
    for pop,case in cases():
        p=ROOT/'cases'/(pop+'_'+case['name']+'.h5')
        r=json.loads(p.with_suffix('.json').read_text())
        assert r['completed'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        for name,value in r['source_hashes'].items():
            choices=[ROOT/name,ROOT/'executed_code_v1'/name]
            assert any(q.exists() and hashlib.sha256(q.read_bytes()).hexdigest()==value for q in choices),name
        receipts.append(r)
    measured=ROOT/'measured_flamingo_compact.h5'
    receipt=json.loads((ROOT/'compact_complete.json').read_text())
    assert hashlib.sha256(measured.read_bytes()).hexdigest()==receipt['sha256']
    # The cache-lock correction must not change numerical science functions.
    old=ast.parse((ROOT/'executed_code_v1/run_comparisons.py').read_text())
    new=ast.parse((ROOT/'run_comparisons.py').read_text())
    names=['cases','load_regions','mapping_alternative','region_task','save_case','whole_task','whole_overviews']
    for name in names:
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    return receipts


def derive_common_support():
    """Combine already computed fields: Pe fit2/cut4 and ne fit3/cut4."""
    receipts=[]
    for pop in ['halfdome','flamingo']:
        p=ROOT/'cases'/(pop+'_fit2_cut4.h5');e=ROOT/'cases'/(pop+'_fit3_cut4.h5')
        output=ROOT/'cases'/(pop+'_joint_support_hybrid.h5')
        with h5py.File(p,'r') as pressure,h5py.File(e,'r') as electrons,h5py.File(output,'w') as out:
            sums=electrons['region_sums'][:];sums[:,0]=pressure['region_sums'][:,0]
            out.create_dataset('region_sums',data=sums,compression='gzip',compression_opts=1)
            out.create_dataset('origins',data=pressure['origins'][:])
            group=out.create_group('statistics')
            for key in electrons['statistics']:
                source=pressure if key.startswith('pressure_') or key.startswith('y_') else electrons
                group.create_dataset(key,data=source['statistics'][key][:])
            for key,value in pressure.attrs.items():out.attrs[key]=value
            out.attrs['case']='Pe fit2/cut4; ne fit3/cut4; common spherical support'
        receipts.append(dict(population=pop,file=output.name,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest(),e.name:hashlib.sha256(e.read_bytes()).hexdigest()}))
    P.save_json(ROOT/'derived_joint_support.json',receipts)


def distribution_figures(stats,measured):
    chosen=[('halfdome','battaglia_cut4','HalfDome B12/B16','#777777'),
            ('halfdome','main_hybrid','HalfDome updated','#d95f02'),
            ('flamingo','main_hybrid','FLAMINGO painted','#1b9e77')]
    for name in LABELS:
        fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
        edges=S.EDGES[name];x=(edges[:-1]+edges[1:])/2;dx=np.diff(edges)
        lines=[]
        for pop,case,label,color in chosen:
            d=stats[(pop,case)];h=d[name+'_pdf']/dx
            lo,hi=np.percentile(h,[16,84],axis=0)
            axes[0].plot(x,h.mean(axis=0),color=color,lw=2,label=label)
            axes[0].fill_between(x,lo,hi,color=color,alpha=.13)
            lines.append((label,float(np.mean(d[name+'_atoms'][:,0]))))
        h=measured[name+'_pdf']/dx
        axes[0].plot(x,h.mean(axis=0),color='black',lw=2,label='FLAMINGO gas')
        lines.append(('FLAMINGO gas',float(measured[name+'_atoms'][:,0].mean())))
        axes[0].set(yscale='log',ylabel='Probability density',xlabel=LABELS[name] if name=='deltaT' else 'log$_{10}$ '+LABELS[name])
        if name=='deltaT':
            axes[0].set_xscale('symlog',linthresh=.01)
        axes[0].legend()
        if name in ['pressure','electrons']:
            for pop,case,label,color in chosen[1:]:
                d=stats[(pop,case)]
                fine=d[name+'_pdf'].mean(axis=0)/dx;coarse=d[name+'_coarse_pdf'].mean(axis=0)/dx
                axes[1].plot(x,fine,color=color,lw=2,label=label+' 0.98 Mpc')
                axes[1].plot(x,coarse,color=color,lw=2,ls='--',label=label+' 1.95 Mpc')
            axes[1].plot(x,measured[name+'_coarse_pdf'].mean(axis=0)/dx,color='black',ls='--',label='Gas 1.95 Mpc')
            axes[1].set(yscale='log',xlabel='log$_{10}$ '+LABELS[name],ylabel='Probability density')
            axes[1].legend()
        else:
            for pop,case,label,color in chosen:
                d=stats[(pop,case)]
                ref=stats[(pop,'battaglia_cut4')][name+'_power'].mean(axis=0)
                k=d['k_per_mpc'];power=d[name+'_power'].mean(axis=0)
                good=np.isfinite(ref)&(ref>0)&np.isfinite(power)
                axes[1].plot(k[good],power[good]/ref[good],lw=2,color=color,label=label)
            ref=stats[('flamingo','battaglia_cut4')][name+'_power'].mean(axis=0)
            power=measured[name+'_power'].mean(axis=0);good=np.isfinite(ref)&(ref>0)&np.isfinite(power)
            axes[1].plot(k[good],power[good]/ref[good],color='black',lw=2,label='FL gas / FL B12/B16')
            axes[1].axhline(1,color='black',lw=1)
            axes[1].set(xscale='log',xlabel=r'$k_\perp\ [\mathrm{Mpc}^{-1}]$',ylabel='Power / same-population B12/B16')
            axes[1].legend()
        save(fig,'pdf_'+name)
        P.save_json(ROOT/('zero_atoms_'+name+'.json'),dict(values=lines,
            bands='16-84 percent spatial-region variation; not independent-mock covariance'))
        if name not in ['pressure','electrons']:
            fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
            for pop,case,label,color in chosen:
                d=stats[(pop,case)];k=d['k_per_mpc'];v=d[name+'_power']*k[None,:]**2/(2*np.pi)
                good=np.isfinite(v).all(axis=0)&(v.mean(axis=0)>0)
                ax.plot(k[good],v.mean(axis=0)[good],color=color,lw=2,label=label)
                lo,hi=np.percentile(v[:,good],[16,84],axis=0);ax.fill_between(k[good],lo,hi,color=color,alpha=.12)
            v=measured[name+'_power']*k[None,:]**2/(2*np.pi);ax.plot(k,np.nanmean(v,axis=0),color='black',lw=2,label='FLAMINGO gas')
            ax.set(xscale='log',yscale='log',xlabel=r'$k_\perp\ [\mathrm{Mpc}^{-1}]$',ylabel=r'$k^2P_{2D}/(2\pi)$')
            ax.axvline(np.pi/(2*250/512),color='gray',ls='--',lw=1);ax.legend();save(fig,'power_'+name)


def map_figures():
    gas=read_maps(ROOT/'measured_flamingo_compact.h5')
    fl=read_maps(ROOT/'cases/flamingo_main_hybrid.h5')
    hd=read_maps(ROOT/'cases/halfdome_main_hybrid.h5')
    flbase=read_maps(ROOT/'cases/flamingo_battaglia_cut4.h5')
    hdbase=read_maps(ROOT/'cases/halfdome_battaglia_cut4.h5')
    for q,name in [(0,'y'),(2,'dm'),(3,'deltaT'),(1,'tau')]:
        signals=[gas[q],fl[q],hd[q]]
        residual=[fl[q]-gas[q],fl[q]-flbase[q],hd[q]-hdbase[q]]
        maximum=max(float(np.max(np.abs(v))) for v in signals)
        if name=='deltaT':
            norm=SymLogNorm(linthresh=max(maximum*1e-3,1e-6),vmin=-maximum,vmax=maximum)
            cmap='RdBu_r'
        else:
            positive=np.concatenate([v[v>0] for v in signals])
            norm=LogNorm(vmin=max(float(np.percentile(positive,.5)),1e-30),vmax=maximum)
            cmap='viridis'
        limit=max(float(np.max(np.abs(v))) for v in residual)
        rnorm=SymLogNorm(linthresh=max(limit*1e-3,1e-20),vmin=-limit,vmax=limit)
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for j,image in enumerate(signals):
            v=np.ma.masked_less_equal(image,0) if name!='deltaT' else image
            im=axes[0,j].imshow(v.T,origin='lower',extent=[0,250,0,250],norm=norm,cmap=cmap)
            axes[0,j].set_title(['FLAMINGO gas','FLAMINGO painted','HalfDome painted'][j])
        for j,image in enumerate(residual):
            rim=axes[1,j].imshow(image.T,origin='lower',extent=[0,250,0,250],norm=rnorm,cmap='RdBu_r')
            axes[1,j].set_title(['Painted - gas (FL)','New - B12/B16 (FL)','New - B12/B16 (HD)'][j])
        for ax in axes.ravel():
            ax.set(xlabel='x [Mpc]',ylabel='y [Mpc]');ax.grid(False)
        fig.colorbar(im,ax=list(axes[0]),label=LABELS[name],shrink=.85,extend='min' if name!='deltaT' else 'neither')
        fig.colorbar(rim,ax=list(axes[1]),label='Absolute difference',shrink=.85)
        save(fig,'maps_'+name,vector=False)


def matrix_figures(stats):
    tables={}
    for name in ['y','tau','dm','deltaT']:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        values=np.empty((4,4))
        for i in range(4):
            for j in range(4):
                d=stats[('halfdome','fit%d_cut%d'%(i+1,j+1))]
                k=d['k_per_mpc'];use=(k>=.05)&(k<=.5)&np.isfinite(d[name+'_power']).all(axis=0)
                values[i,j]=np.nanmean(d[name+'_power'][:,use])
        versions=[values/values[3,3],values/np.diag(values)[None,:],values/np.diag(values)[:,None]]
        for ax,array,title in zip(axes,versions,['Reference: fit 4R, cut 4R','Reference: same-cut diagonal','Reference: same-fit diagonal']):
            im=ax.imshow(array,origin='lower',extent=[.5,4.5,.5,4.5],aspect='equal',cmap='viridis')
            for i in range(4):
                for j in range(4):ax.text(j+1,i+1,'%.2f'%array[i,j],ha='center',va='center',fontsize=14,color='white' if array[i,j]<np.median(array) else 'black')
            ax.set(xlabel=r'$R_{\rm cut}/R_{200c}$',ylabel=r'$R_{\rm fit}/R_{200c}$',xticks=[1,2,3,4],yticks=[1,2,3,4],title=title);ax.grid(False)
            fig.colorbar(im,ax=ax,shrink=.8)
        tables[name]=dict(power=values.tolist(),relative_to_44=versions[0].tolist(),relative_to_column_diagonal=versions[1].tolist(),relative_to_row_diagonal=versions[2].tolist(),k_band_mpc_inverse=[.05,.5])
        save(fig,'matrix_'+name)
    P.save_json(ROOT/'fit_cut_matrices.json',tables)


def profile_figures(fits):
    source=np.load(ROOT/'inputs/measurements.npz');halos=np.load(ROOT/'inputs/selected_halos.npz')
    edges=np.asarray(fits['edges']);x=(.5*(edges[1:]**3+edges[:-1]**3))**(1/3)
    for b in range(15):
        rows=np.flatnonzero(halos['mass_bin']==b)
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained',gridspec_kw={'height_ratios':[2,1]})
        for j,(field,key,label) in enumerate([('pressure','pressure_components',r'$P_e/P_{200}$'),('electrons','electron_components',r'$n_e\mu_e/(\rho_c f_b)$')]):
            values=source[key][rows].sum(axis=1);mean=values.mean(axis=0)
            low,high=np.percentile(values,[16,84],axis=0)
            axes[0,j].fill_between(x,low,high,color='gray',alpha=.22,label='Halo 16-84%')
            support=next(r for r in fits['results'] if r['field']==field and r['component']=='all' and r['bin']==b and r['cut']==4)
            resolved=np.asarray(support.get('resolved_mask',np.ones(len(x))),bool)
            axes[0,j].plot(x[resolved],mean[resolved],'k.',ms=6,label='Measured mean (fit bins)')
            axes[0,j].plot(x[~resolved],mean[~resolved],'o',mfc='none',mec='gray',ms=4,label='Excluded bins')
            median=np.median(values,axis=0);ix=np.flatnonzero(resolved & (median>0))[::3]
            axes[0,j].errorbar(x[ix],median[ix],yerr=[median[ix]-low[ix],high[ix]-median[ix]],
                fmt='none',ecolor='gray',capsize=2,lw=.8,label='Median; halo 16-84%')
            for cut,color in zip([1,2,3,4],['#0072B2','#009E73','#E69F00','#D55E00']):
                r=next(r for r in fits['results'] if r['field']==field and r['component']=='all' and r['bin']==b and r['cut']==cut)
                if not r['valid']:continue
                prediction=np.asarray(r['prediction']);pars=r['parameters']
                inside=x<=cut
                axes[0,j].plot(x[inside],prediction[inside],color=color,lw=2,label='%dR: %.3g, %.3g, %.3g'%(cut,*pars))
                axes[0,j].plot(x[~inside],prediction[~inside],color=color,lw=2,ls='--')
                axes[1,j].plot(x,prediction-mean,color=color,lw=2)
            axes[0,j].set(xscale='log',yscale='log',ylabel=label,title=['Pressure','Electron density'][j],xlim=[.035,4.05])
            axes[0,j].legend(fontsize=10)
            axes[1,j].axhline(0,color='black',lw=1)
            axes[1,j].set(xscale='log',xlabel=r'$r/R_{200c}$',ylabel='Model - measured',xlim=[.035,4.05])
        fig.suptitle('%.1f-%.1f | N=%d'%(halos['mass_edges'][b],halos['mass_edges'][b+1],len(rows)),fontsize=18)
        save(fig,'profiles_bin_%02d'%b)
    # Parameters: mean-fit bootstrap and intrinsic individual-fit diversity are separate.
    summary=[]
    for field,names in [('pressure',['P0','xc','beta']),('electrons',['rho0','alpha','beta'])]:
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for cut,color in zip([1,2,3,4],['#0072B2','#009E73','#E69F00','#D55E00']):
            rows=[r for r in fits['results'] if r['field']==field and r['component']=='all' and r['cut']==cut and r['valid']]
            mass=np.log10([r['mass_pivot_msun'] for r in rows])
            for j,param in enumerate(names):
                best=np.array([r['parameters'][j] for r in rows]);axes[0,j].plot(mass,best,color=color,lw=1)
                boot=np.array([np.percentile(np.asarray(r['bootstrap_parameters'])[:,j],[16,50,84]) for r in rows])
                axes[0,j].errorbar(mass,boot[:,1],yerr=[boot[:,1]-boot[:,0],boot[:,2]-boot[:,1]],color=color,fmt='o',ms=3,capsize=2,label='%dR'%cut)
                q=[]
                for r in rows:
                    a=np.asarray(r['individual_parameters'])
                    q.append(np.percentile(a[:,j],[16,50,84]) if len(a)>1 else [np.nan]*3)
                q=np.asarray(q);axes[1,j].plot(mass,q[:,1],color=color,lw=2)
                axes[1,j].fill_between(mass,q[:,0],q[:,2],color=color,alpha=.15)
                axes[0,j].set(ylabel=param,xlabel=r'$\log_{10}(M_{200c}/M_\odot)$',title=param+' mean-fit bootstrap')
                axes[1,j].set(ylabel=param,xlabel=r'$\log_{10}(M_{200c}/M_\odot)$',title=param+' individual-fit scatter')
                if j==0:
                    axes[0,j].set_yscale('log');axes[1,j].set_yscale('log')
        axes[0,0].legend(ncol=2)
        save(fig,'parameters_'+field)
    for r in fits['results']:
        if not r['valid']:continue
        a=np.asarray(r['bootstrap_parameters']);individual=np.asarray(r['individual_parameters'])
        summary.append(dict(field=r['field'],component=r['component'],mass_bin=r['bin'],fit_radius=r['cut'],
            mass_pivot_msun=r['mass_pivot_msun'],halo_count=r['n'],parameters=r['parameters'],
            bootstrap_16_50_84=np.percentile(a,[16,50,84],axis=0).tolist(),
            bootstrap_log_covariance=np.cov(np.log(a),rowvar=False).tolist(),
            individual_count=len(individual),individual_16_50_84=np.percentile(individual,[16,50,84],axis=0).tolist() if len(individual)>1 else None,
            individual_log_covariance=np.cov(np.log(individual),rowvar=False).tolist() if len(individual)>1 else None,
            rms_fit_percent=r['rms_fit_percent'],rms_inner_percent=r['rms_inner_percent']))
    P.save_json(ROOT/'parameter_distributions.json',summary)


def overview_figures():
    with h5py.File(ROOT/'measured_flamingo_compact.h5','r') as f:
        gas=P.map_observables(f['whole_sums'][:],(1000/1024)**2)
    for q,name in [(0,'y'),(2,'dm'),(3,'deltaT')]:
        images=[gas[q]];sides=[1000.]
        for pop in ['flamingo','halfdome']:
            with h5py.File(ROOT/('overview_'+pop+'_main_hybrid.h5'),'r') as f:
                side=float(f.attrs['box_mpc']);images.append(P.map_observables(f['sums'][:],(side/1024)**2)[q]);sides.append(side)
        limit=max(np.max(np.abs(a)) for a in images)
        norm=SymLogNorm(max(limit*1e-3,1e-9),vmin=-limit,vmax=limit) if name=='deltaT' else LogNorm(max(min(np.min(a[a>0]) for a in images),1e-30),limit)
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax,image,side,title in zip(axes,images,sides,['FLAMINGO gas','FLAMINGO painted','HalfDome painted']):
            im=ax.imshow((np.ma.masked_less_equal(image,0) if name!='deltaT' else image).T,
                origin='lower',extent=[0,side,0,side],norm=norm,cmap='RdBu_r' if name=='deltaT' else 'viridis')
            ax.set(title=title,xlabel='x [Mpc]',ylabel='y [Mpc]');ax.grid(False)
        fig.colorbar(im,ax=list(axes),shrink=.7,label=LABELS[name]);save(fig,'whole_box_'+name,vector=False)


def resource_history():
    jobs={}
    for job in ['602967.idark','602969.idark','602974[0].idark','602974[1].idark','602974[2].idark','602974[3].idark','602974[4].idark','602976.idark',os.environ.get('PBS_JOBID','')]:
        if not job:continue
        r=subprocess.run(['qstat','-x','-f','-F','json',job],capture_output=True,text=True)
        try:jobs[job]=json.loads(r.stdout).get('Jobs',{})
        except ValueError:jobs[job]={'query_error':r.stderr[:300]}
    P.save_json(ROOT/'resource_history.json',jobs)


def closure_tables(fits):
    """Physical integrated signals, averaging each halo at its own mass."""
    import painting as M
    from numpy.polynomial.legendre import leggauss
    source=np.load(ROOT/'inputs/measurements.npz');halos=np.load(ROOT/'inputs/selected_halos.npz')
    edges=source['edges'];nodes,w=leggauss(48)
    lo,hi=edges[:-1,None],edges[1:,None]
    x=lo+(hi-lo)*(nodes+1)/2;weights=3*x*x*w*(hi-lo)/(2*(hi**3-lo**3))
    rows_out=[];scatter=[]
    for b in range(15):
        rows=np.flatnonzero(halos['mass_bin']==b);mass=halos['m200c_msun'][rows];radius=halos['r_200_crit_mpc'][rows]
        for field,q,key,cut_ref in [('pressure',2,'pressure_components',2),('electrons',4,'electron_components',3)]:
            if field=='pressure':scale=source['p200_pa'][rows]*radius**3/P.Y_TO_P_VOLUME
            else:scale=source['critical_density_msun_mpc3'][rows]*(P.SOURCE['omega_b']/P.SOURCE['omega_m'])*P.MSUN_KG/P.MU_E_KG*radius**3
            for fit in [1,2,3,4]:
                pars=M.parameters(fits,field,fit,'all',mass)
                predicted=(P.shape(field,x[None,:,:],pars.T[:,:,None,None])*weights).sum(axis=2)
                for cut in [1,2,3,4]:
                    use=edges[1:]<=cut+1e-12
                    integral=4*np.pi/3*np.sum(predicted[:,use]*np.diff(edges**3)[use],axis=1)
                    true=source['sums'][rows,:,q,:][:,:,use].sum(axis=(1,2))
                    rows_out.append(dict(bin=b,field=field,fit_radius=fit,cut_radius=cut,
                        measured_mean=float(true.mean()),model_mean=float(np.mean(scale*integral)),
                        relative_difference_percent=float(100*(np.mean(scale*integral)/true.mean()-1)),
                        halo_count=len(rows)))
            r=next(r for r in fits['results'] if r['field']==field and r['component']=='all' and r['bin']==b and r['cut']==cut_ref)
            a=np.asarray(r.get('individual_parameters',[]))
            if len(a)>1:
                values=source[key][rows].sum(axis=1)
                model=(P.shape(field,x[None,:,:],a.T[:,:,None,None])*weights).sum(axis=2)
                scatter.append(dict(bin=b,field=field,individual_count=len(a),
                    accepted_fraction=len(a)/len(rows),measured_mean=values.mean(axis=0).tolist(),
                    measured_sd=values.std(axis=0,ddof=1).tolist(),draw_mean=model.mean(axis=0).tolist(),
                    draw_sd=model.std(axis=0,ddof=1).tolist(),edges=edges.tolist()))
    P.save_json(ROOT/'integrated_profile_checks.json',rows_out)
    P.save_json(ROOT/'scatter_closure.json',scatter)
    for b in [3,7,10,12,14]:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
        xx=(.5*(edges[1:]**3+edges[:-1]**3))**(1/3)
        for ax,field in zip(axes,['pressure','electrons']):
            r=next((r for r in scatter if r['bin']==b and r['field']==field),None)
            if r is None:continue
            ax.plot(xx,r['measured_sd'],color='black',lw=2,label='Measured halo SD')
            ax.plot(xx,r['draw_sd'],color='#d95f02',lw=2,label='Individual-fit profile SD')
            ax.set(xscale='log',yscale='log',xlabel=r'$r/R_{200c}$',ylabel='Profile scatter',title=field.capitalize());ax.legend()
        save(fig,'scatter_closure_bin_%02d'%b)


def halo_union_figures(stats):
    path=ROOT/'measured_halo_union_compact.h5'
    receipt=json.loads((ROOT/'halo_union_complete.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    d=read_stats(path)
    metrics=[]
    with h5py.File(path,'r') as f:union=f['region_sums'][:]
    with h5py.File(ROOT/'cases/flamingo_main_hybrid.h5','r') as f:
        raw=f['region_sums'][:]
    hybrid_coarse=raw.reshape(8,3,256,2,256,2).sum(axis=(3,5))
    for name in LABELS:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        edges=S.EDGES[name];xx=(edges[1:]+edges[:-1])/2;width=np.diff(edges)
        for cut,color in zip([1,2,3,4],['#0072B2','#009E73','#E69F00','#D55E00']):
            h=d[name+'_pdf'][:,cut-1].mean(axis=0)/width
            ax.plot(xx,h,color=color,lw=2,label='Measured union %dR'%cut)
        if name in ['pressure','electrons']:
            r=stats[('flamingo','main_hybrid')][name+'_pdf'].mean(axis=0)/width
        else:
            q=['y','tau','dm','deltaT'].index(name)
            r=np.mean([S.probability(P.map_observables(v,(250/256)**2)[q],name)[0] for v in hybrid_coarse],axis=0)/width
        ax.plot(xx,r,'k--',lw=2,label='Painted hybrid')
        ax.set(yscale='log',xlabel=LABELS[name] if name=='deltaT' else 'log$_{10}$ '+LABELS[name],ylabel='Probability density')
        if name=='deltaT':ax.set_xscale('symlog',linthresh=.01)
        ax.legend();save(fig,'halo_union_pdf_'+name)
    for fit in [1,2,3,4]:
        for cut in [1,2,3,4]:
            with h5py.File(ROOT/'cases'/('flamingo_fit%d_cut%d.h5'%(fit,cut)),'r') as f:
                raw=f['region_sums'][:]
            # Raw deposits add under pixel rebinning; map intensities then use the larger pixel area.
            coarse=raw.reshape(8,3,256,2,256,2).sum(axis=(3,5))
            for region in range(8):
                model=P.map_observables(coarse[region],(250/256)**2)
                true=P.map_observables(union[region,cut-1],(250/256)**2)
                for q,name in enumerate(['y','tau','dm','deltaT']):
                    diff=model[q]-true[q]
                    metrics.append(dict(fit_radius=fit,cut_radius=cut,region=region,observable=name,
                        rms_absolute=float(np.sqrt(np.mean(diff*diff))),
                        measured_rms=float(np.sqrt(np.mean(true[q]**2))),
                        relative_L2_percent=float(100*np.linalg.norm(diff)/max(np.linalg.norm(true[q]),1e-300)),
                        mean_model=float(model[q].mean()),mean_measured=float(true[q].mean())))
    P.save_json(ROOT/'halo_union_residual_metrics.json',metrics)


def quantitative_findings(stats,measured):
    result={}
    for name in LABELS:
        gas=measured[name+'_mean'].mean();grms=np.sqrt(np.mean(measured[name+'_rms']**2))
        fl=stats[('flamingo','main_hybrid')];hd=stats[('halfdome','main_hybrid')]
        result[name]=dict(measured_fl_mean=float(gas),measured_fl_rms=float(grms),
            fl_painted_mean=float(fl[name+'_mean'].mean()),hd_painted_mean=float(hd[name+'_mean'].mean()),
            fl_painted_rms=float(np.sqrt(np.mean(fl[name+'_rms']**2))),hd_painted_rms=float(np.sqrt(np.mean(hd[name+'_rms']**2))))
        if name!='deltaT':result[name]['fl_painted_mean_over_full_gas']=float(fl[name+'_mean'].mean()/gas)
        result[name]['fl_painted_rms_over_full_gas']=float(np.sqrt(np.mean(fl[name+'_rms']**2))/grms)
    comparisons={}
    for case in ['normalization_control','sem_weight_control','paired_scatter_42','paired_scatter_43',
                 'high_mass_cut_15p1','joint_support_hybrid','mass_mapping_alternative_0',
                 'mass_mapping_alternative_1','mass_mapping_alternative_2']:
        comparisons[case]={}
        for name in LABELS:
            d=stats[('halfdome',case)];ref=stats[('halfdome','main_hybrid')]
            row=dict(rms_ratio=float(np.sqrt(np.mean(d[name+'_rms']**2)/np.mean(ref[name+'_rms']**2))))
            if name!='deltaT':row['mean_ratio']=float(d[name+'_mean'].mean()/ref[name+'_mean'].mean())
            if name+'_power' in d:
                k=d['k_per_mpc'];use=(k>=.05)&(k<=1)&np.isfinite(d[name+'_power']).all(axis=0)
                row['power_ratio_k005_1']=float(d[name+'_power'][:,use].mean()/ref[name+'_power'][:,use].mean())
            comparisons[case][name]=row
    result['controls']=comparisons
    residual=json.loads((ROOT/'halo_union_residual_metrics.json').read_text())
    scores=[]
    for name in ['y','tau','dm','deltaT']:
        for cut in [1,2,3,4]:
            for fit in [1,2,3,4]:
                rows=[r for r in residual if r['observable']==name and r['cut_radius']==cut and r['fit_radius']==fit]
                score=100*np.sqrt(np.mean([r['rms_absolute']**2 for r in rows])/np.mean([r['measured_rms']**2 for r in rows]))
                scores.append(dict(observable=name,fit=fit,cut=cut,relative_L2_percent=float(score)))
    result['matched_halo_union_scores']=scores
    result['scope']='Finite z=0 boxes, selected mass interval, NGP pixels/cells. Neither lightcone predictions nor independent HalfDome gas validation.'
    P.save_json(ROOT/'quantitative_findings.json',result)


def main():
    start=time.time()
    receipts=verified_inputs()
    derive_common_support()
    stats={(pop,case['name']):read_stats(ROOT/'cases'/(pop+'_'+case['name']+'.h5')) for pop,case in cases()}
    for pop in ['halfdome','flamingo']:
        stats[(pop,'joint_support_hybrid')]=read_stats(ROOT/'cases'/(pop+'_joint_support_hybrid.h5'))
    measured=read_stats(ROOT/'measured_flamingo_compact.h5')
    fits=json.loads((ROOT/'fits.json').read_text())
    with ProcessPoolExecutor(max_workers=5) as pool:
        work=[pool.submit(distribution_figures,stats,measured),pool.submit(map_figures),
              pool.submit(matrix_figures,stats),pool.submit(profile_figures,fits),pool.submit(overview_figures)]
        for result in work:result.result()
    closure_tables(fits);halo_union_figures(stats);quantitative_findings(stats,measured);resource_history()
    summary=[]
    for (pop,case),d in stats.items():
        reference=stats[(pop,'main_hybrid')]
        for name in LABELS:
            h=d[name+'_pdf'].mean(axis=0)
            row=dict(population=pop,case=case,observable=name,zero_atom=float(d[name+'_atoms'][:,0].mean()),
                probability_total=float(h.sum()+d[name+'_atoms'].mean(axis=0).sum()))
            if name+'_mean' in d:
                row.update(mean=float(np.mean(d[name+'_mean'])),rms=float(np.sqrt(np.mean(d[name+'_rms']**2))),
                    rms_ratio_to_main=float(np.sqrt(np.mean(d[name+'_rms']**2)/np.mean(reference[name+'_rms']**2))))
            if name+'_power' in d:
                use=np.isfinite(d[name+'_power']).all(axis=0)&np.isfinite(reference[name+'_power']).all(axis=0)
                row['mean_power_ratio_to_main']=float(np.nanmean(d[name+'_power'][:,use])/np.nanmean(reference[name+'_power'][:,use]))
            summary.append(row)
    P.save_json(ROOT/'comparison_summary.json',dict(rows=summary,case_count=len(receipts),
        measured_data='full FLAMINGO gas; native electron field and actual gas peculiar velocities',
        painted_data='halo-only; bulk halo velocities; common mass interval 1e13-10^15.5 physical Msun',
        regions=8,region_side_mpc=250.,map_pixel_mpc=250/512,voxel_mpc=250/256,
        uncertainty='Bands show spatial-region variation, not independent-mock covariance.',
        wall_seconds=time.time()-start))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in FIG.glob('*')}
    P.save_json(ROOT/'cluster_analysis_complete.json',dict(completed=True,figures=manifest,case_count=len(receipts),wall_seconds=time.time()-start))


if __name__=='__main__':main()
