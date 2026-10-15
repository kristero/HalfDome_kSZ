"""Paired 2/3/4R fits, contamination controls, held-out checks and scatter.

Strict resolution gates are retained. If the 2R fit has fewer than six common
resolved shells, diagnostic stack fits are shown separately and never certified
for stochastic painting. Zero-pressure shells are excluded, never log-filled.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import csv
import json
from pathlib import Path
import resource
import time
import numpy as np
import pilot_physics as P
import measure as M
from models import fit_profile,shell
from projection_model import project_map,projected_profile,relative_rms,map_metrics

HERE=Path(__file__).resolve().parent
OUT=HERE/'products'
SELECTIONS={'all':[0,1,2,3],'no_other':[0,1,3],
            'no_satellites_or_other':[0,3],'host_bound':[0,1]}
FAMILIES={'pressure':('b12','pressure_components'),
          'electrons':('b16','electron_components'),
          'gas':('b16','gas_components')}
BOOTSTRAPS=128


def plain(value):
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def dump(path,value):path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def summarize(parameters):
    parameters=np.asarray(parameters)
    parameters=parameters[np.all(np.isfinite(parameters),axis=1)]
    if len(parameters)<2:return {'count':len(parameters),'percentiles':None,'log_covariance':None}
    covariance=np.cov(np.log(parameters),rowvar=False)
    rank=int(np.linalg.matrix_rank(covariance))
    return {'count':len(parameters),'percentiles':np.percentile(parameters,[16,50,84],axis=0),
            'standard_deviation':parameters.std(axis=0,ddof=1),
            'log_covariance':covariance,'log_covariance_rank':rank,
            'log_covariance_full_rank':rank==parameters.shape[1],
            'log_correlation':np.corrcoef(np.log(parameters),rowvar=False)}


def accepted(fit):
    return bool(fit['success'] and fit['bound_distance']>1e-5 and fit['jacobian_condition']<1e6)


def resolution_masks(halos,data,rows,softening):
    x=P.EDGES[:-1]
    count=data['sums'][rows,:,0].sum(axis=1)
    geometric=x[None,:]>=np.maximum(.04,4*softening/halos['r_200_crit_mpc'][rows,None])
    resolved=geometric & (count>=20)&(x[None,:]>=2*data['hsml_median_upper_x'][rows])
    strict=np.all(resolved,axis=0)
    # Diagnostic fallback keeps all selected halos; it relaxes numerical gates,
    # not population membership. These coefficients are explicitly unsupported.
    diagnostic=np.all(geometric,axis=0)&(count.sum(axis=0)>=200)&(np.mean(count>0,axis=0)>=.5)
    supported=int(np.sum(strict&(P.EDGES[1:]<=2)))>=6
    return strict if supported else diagnostic,resolved,strict,supported


def fit_one(family,mean,std,mask,start=None,model=None):
    valid=mask&np.isfinite(mean)&(mean>0)
    if valid.sum()<6:return None
    error=np.maximum(np.divide(std,mean,out=np.ones_like(std),where=mean>0),.05)
    return fit_profile(family,mean,valid,error,start=start,model=model)


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True)
    halos=dict(np.load(HERE/'selected_halos.npz'))
    data=dict(np.load(HERE/'measurements.npz'))
    maps=dict(np.load(HERE/'projection_stacks.npz'))
    plan=json.loads((HERE/'metadata_plan.json').read_text())
    cases=[];table=[];recommendation_metrics=[]
    for b,(lo,hi) in enumerate(zip(halos['mass_edges'][:-1],halos['mass_edges'][1:])):
        rows=np.flatnonzero(halos['mass_bin']==b);n=len(rows)
        base,resolved,strict,supported=resolution_masks(halos,data,rows,plan['softening_physical_mpc'])
        label=f'bin_{lo:.1f}_{hi:.1f}'.replace('.','p')
        count=data['sums'][rows,:,0].sum(axis=1)
        stats={'edges':P.EDGES,'halo_rows':rows,'fit_base_mask':base,'strict_mask':strict,
               'resolved_fraction':resolved.mean(axis=0),'count_mean':count.mean(axis=0)}
        case={'label':label,'mass_edges':[float(lo),float(hi)],'halos':n,
              'population':plan['population_per_bin'][b], 'strict_resolution_supported':supported,
              'fit_status':'common resolved shells' if supported else 'diagnostic only; strict 2R support absent',
              'strict_shell_count_to_2R':int(np.sum(strict&(P.EDGES[1:]<=2))),
              'fit_base_mask':base,'resolved_fraction':resolved.mean(axis=0),'selections':{},
              'disturbed_count':int(halos['disturbed_proxy'][rows].sum()),'groups':{}}
        rng=np.random.default_rng(20261008+b)
        bootstrap_rows=rng.integers(0,n,size=(BOOTSTRAPS,n))
        permutation=rng.permutation(n)
        test=permutation[:max(1,n//5)];train=permutation[max(1,n//5):]
        boot=np.full((3,2,BOOTSTRAPS,3),np.nan)
        control_boot=np.full((2,3,2,BOOTSTRAPS,3),np.nan)
        individual=np.full((n,3,2,3),np.nan)
        individual_good=np.zeros((n,3,2),bool)
        individual_rms=np.full((n,3,2),np.nan)
        individual_status=np.full((n,3,2),-1,dtype=np.int8)
        for selection,components in SELECTIONS.items():
            result={}
            for field,(family,key) in FAMILIES.items():
                values=data[key][rows][:,components].sum(axis=1)
                mean=values.mean(axis=0);std=values.std(axis=0,ddof=1)
                stats[f'{selection}_{field}_mean']=mean
                stats[f'{selection}_{field}_std']=std
                stats[f'{selection}_{field}_percentiles']=np.percentile(values,[16,50,84],axis=0)
                if selection=='all':
                    stats[f'{field}_bootstrap_mean_percentiles']=np.percentile(values[bootstrap_rows].mean(axis=1),[16,50,84],axis=0)
                if selection=='all':
                    reference=P.shell_model(family,P.B12 if family=='b12' else P.B16,halos['m200c_msun'][rows]).mean(axis=0)
                    stats[f'{field}_published_reference']=reference
                fits=[]
                for ri,radius in enumerate(M.RADII):
                    mask=base&(P.EDGES[1:]<=radius+1e-12)&(mean>0)
                    fit=fit_one(family,mean,std,mask)
                    entry={'radius_R200c':int(radius),'fit_mask':mask,'fit':fit,'resolution_supported':supported}
                    if fit is not None:
                        p=np.array(fit['parameters']);prediction=shell(family,p)
                        entry.update(accepted=accepted(fit),
                            common_inner_RMS_percent=relative_rms(prediction,mean,base&(P.EDGES[1:]<=2)),
                            all_4R_RMS_percent=relative_rms(prediction,mean,base),
                            outer_2to3_RMS_percent=relative_rms(prediction,mean,base&(P.EDGES[:-1]>=2)&(P.EDGES[1:]<=3)),
                            outer_3to4_RMS_percent=relative_rms(prediction,mean,base&(P.EDGES[:-1]>=3)))
                        if selection=='all':
                            entry['published_reference_RMS_percent']=relative_rms(reference,mean,mask)
                        for ci,component_name in enumerate(M.COMPONENTS):
                            stats[f'{field}_{component_name}_mean']=data[key][rows,ci].mean(axis=0)
                        if selection=='all' and field!='gas':
                            fi=0 if field=='pressure' else 1
                            for draw,chosen in enumerate(bootstrap_rows):
                                sample=values[chosen]
                                trial=fit_one(family,sample.mean(axis=0),sample.std(axis=0,ddof=1),mask,start=p)
                                if trial is not None and accepted(trial):boot[ri,fi,draw]=trial['parameters']
                            entry['bootstrap_parameter_uncertainty']=summarize(boot[ri,fi])
                            for j,row in enumerate(values):
                                own_mask=resolved[j]&(P.EDGES[1:]<=radius)&(row>0)
                                if own_mask.sum()<6:continue
                                trial=fit_profile(family,row,own_mask,start=p)
                                individual[j,ri,fi]=trial['parameters'];individual_rms[j,ri,fi]=trial['rms_percent']
                                good=accepted(trial)
                                individual_good[j,ri,fi]=good;individual_status[j,ri,fi]=int(good)
                            entry['individual_parameter_scatter']=summarize(individual[:,ri,fi][individual_good[:,ri,fi]])
                            entry['individual_fit_attempts']=int(np.sum(individual_status[:,ri,fi]>=0))
                            entry['individual_fit_accepted']=int(individual_good[:,ri,fi].sum())
                            entry['scatter_selection_warning']='Resolved and stable individual fits only; missing/unresolved halos can bias this distribution. It is not deconvolved intrinsic scatter.'
                            if n>=20:
                                training=values[train];heldout=values[test].mean(axis=0)
                                trial=fit_one(family,training.mean(axis=0),training.std(axis=0,ddof=1),mask,start=p)
                                if trial:
                                    entry['heldout']={'train_halos':len(train),'test_halos':len(test),'fit':trial,
                                        'inner_2R_RMS_percent':relative_rms(shell(family,trial['parameters']),heldout,base&(P.EDGES[1:]<=2)),
                                        'all_4R_RMS_percent':relative_rms(shell(family,trial['parameters']),heldout,base)}
                        elif selection in ('no_other','no_satellites_or_other') and field!='gas':
                            ci=0 if selection=='no_other' else 1
                            fi=0 if field=='pressure' else 1
                            for draw,chosen in enumerate(bootstrap_rows):
                                sample=values[chosen]
                                trial=fit_one(family,sample.mean(axis=0),sample.std(axis=0,ddof=1),mask,start=p)
                                if trial is not None and accepted(trial):control_boot[ci,ri,fi,draw]=trial['parameters']
                            entry['bootstrap_parameter_uncertainty']=summarize(control_boot[ci,ri,fi])
                        # Native coefficients evaluated against the same projected targets.
                        q={'pressure':0,'electrons':1,'gas':2}[field]
                        truth=maps['map_sums'][b,0,1,ri,components,q].sum(axis=0)/n
                        model=project_map(family,p,radius)
                        entry['matched_sphere_map']=map_metrics(model,truth)
                        original=maps['map_sums'][b,0,0,ri,components,q].sum(axis=0)/n
                        entry['original_z_view_map']=map_metrics(model,original)
                        camera=maps['view_profile_sums'][b,1:,ri][:,components,q].sum(axis=1)/n
                        camera_mean=camera.mean(axis=0)
                        camera_mask=(M.ANNULUS_COUNT>0)&(P.EDGES[:-1]>=.25)&(P.EDGES[1:]<=radius)&(camera_mean>0)
                        entry['camera_profile_RMS_scatter_percent']=float(100*np.sqrt(np.mean((camera.std(axis=0,ddof=1)[camera_mask]/camera_mean[camera_mask])**2))) if camera_mask.any() else None
                        truth4=maps['map_sums'][b,0,1,2,components,q].sum(axis=0)/n
                        entry['extrapolated_4R_map']=map_metrics(project_map(family,p,4),truth4)
                        truth2=maps['map_sums'][b,0,1,0,components,q].sum(axis=0)/n
                        entry['common_2R_map']=map_metrics(project_map(family,p,2),truth2)
                        coordinate=-M.EXTENT+(np.arange(M.NPIX)+.5)*2*M.EXTENT/M.NPIX
                        xx,yy=np.meshgrid(coordinate,coordinate,indexing='ij');projected_r=np.hypot(xx,yy)
                        first_edge=float(P.EDGES[:-1][base].min())
                        min_b=max(.25,first_edge+np.sqrt(2)*2*M.EXTENT/M.NPIX/2)
                        resolved_pixels=(projected_r>=min_b)&(projected_r<radius)
                        common4_pixels=(projected_r>=min_b)&(projected_r<4)
                        entry['resolved_projection_min_b_R200c']=min_b
                        entry['resolved_projection_map']=map_metrics(model[resolved_pixels],truth[resolved_pixels])
                        model4=project_map(family,p,4)
                        entry['resolved_extrapolated_4R_map']=map_metrics(model4[common4_pixels],truth4[common4_pixels])
                        entry['outer_map_L2_percent']={}
                        for lower,upper in [(1.,2.),(2.,3.),(3.,4.)]:
                            ring=(projected_r>=lower)&(projected_r<upper)
                            entry['outer_map_L2_percent'][f'{int(lower)}to{int(upper)}']=map_metrics(model4[ring],truth4[ring])['relative_L2_percent']
                        if field=='pressure' and selection in ('all','no_other'):
                            observed=data['projected_profiles'][rows,1,ri][:,components,0].sum(axis=1)
                            projected_mean=observed.mean(axis=0)
                            projected_std=observed.std(axis=0,ddof=1)
                            pmask=base&(P.EDGES[:-1]>=.25)&(P.EDGES[1:]<=radius)&(M.ANNULUS_COUNT>0)&(projected_mean>0)
                            pfit=fit_one(family,projected_mean,projected_std,pmask,start=p,
                                model=lambda par:projected_profile(family,par,radius))
                            entry['projected_refit']=pfit
                            if pfit:
                                pmodel=project_map(family,pfit['parameters'],radius)
                                entry['projected_refit_map']=map_metrics(pmodel,truth)
                                entry['resolved_projected_refit_map']=map_metrics(pmodel[resolved_pixels],truth[resolved_pixels])
                            stats[f'{selection}_projected_pressure_mean_R{int(radius)}']=projected_mean
                            stats[f'{selection}_projected_pressure_std_R{int(radius)}']=projected_std
                        table.append({'bin_low':lo,'bin_high':hi,'halos':n,'selection':selection,'field':field,
                            'radius_R200c':radius,'strict_resolution_supported':int(supported),'P0_or_rho0':p[0],'xc_or_alpha':p[1],'beta':p[2],
                            'fit_RMS_percent':fit['rms_percent'],'common_inner_RMS_percent':entry['common_inner_RMS_percent'],
                            'all_4R_RMS_percent':entry['all_4R_RMS_percent'],
                            'map_L2_percent':entry['matched_sphere_map']['relative_L2_percent'],
                            'extrapolated_4R_map_L2_percent':entry['extrapolated_4R_map']['relative_L2_percent'],
                            'resolved_extrapolated_4R_map_L2_percent':entry['resolved_extrapolated_4R_map']['relative_L2_percent'],
                            'heldout_inner_RMS_percent':entry.get('heldout',{}).get('inner_2R_RMS_percent'),
                                'heldout_4R_RMS_percent':entry.get('heldout',{}).get('all_4R_RMS_percent')})
                    fits.append(entry)
                result[field]=fits
            case['selections'][selection]=result
        # Disturbed/less-disturbed contrasts are snapshot morphology controls,
        # never labelled measured historical merger rates.
        major=halos['max_satellite_bound_mass_ratio_R2'][rows]>=.25
        offset=halos['centre_offset_R200c'][rows]>.07
        groups=[('disturbed',halos['disturbed_proxy'][rows]),('less_disturbed',~halos['disturbed_proxy'][rows]),
                ('major_companion',major),('no_major_companion',~major),
                ('large_bound_offset',offset),('small_bound_offset',~offset)]
        for group_name,group_mask in groups:
            nr=int(group_mask.sum());case['groups'][group_name]={'count':nr}
            for field,(family,key) in FAMILIES.items():
                values=data[key][rows[group_mask]].sum(axis=1)
                if nr==0:continue
                mean=values.mean(axis=0);stats[f'{group_name}_{field}_mean']=mean
                if nr>=6:
                    std=values.std(axis=0,ddof=1)
                    case['groups'][group_name][field]=[fit_one(family,mean,std,base&(P.EDGES[1:]<=r)) for r in M.RADII]
        case['paired_bootstrap_radius_shifts']={}
        for fi,field in enumerate(('pressure','electrons')):
            shifts={}
            for earlier,later in [(0,1),(1,2),(0,2)]:
                valid=np.all(np.isfinite(boot[[earlier,later],fi]),axis=(0,2))
                delta=100*(boot[later,fi,valid]/boot[earlier,fi,valid]-1)
                shifts[f'R{earlier+2}_to_R{later+2}']={'paired_draws':int(valid.sum()),
                    'parameter_percent_shift_16_50_84':np.percentile(delta,[16,50,84],axis=0) if valid.any() else None}
            case['paired_bootstrap_radius_shifts'][field]=shifts
        case['joint_individual_pressure_electron_scatter']={}
        for ri,radius in enumerate(M.RADII):
            valid=np.all(individual_good[:,ri],axis=1)
            joint=individual[valid,ri].reshape(-1,6)
            case['joint_individual_pressure_electron_scatter'][str(int(radius))]=summarize(joint)
        case['paired_bootstrap_membership_shifts']={}
        for ci,selection in enumerate(('no_other','no_satellites_or_other')):
            controls={}
            for fi,field in enumerate(('pressure','electrons')):
                changes=[]
                for ri in range(3):
                    valid=np.all(np.isfinite(boot[ri,fi])&np.isfinite(control_boot[ci,ri,fi]),axis=1)
                    delta=100*(control_boot[ci,ri,fi,valid]/boot[ri,fi,valid]-1)
                    changes.append({'paired_draws':int(valid.sum()),
                        'parameter_percent_shift_16_50_84':np.percentile(delta,[16,50,84],axis=0) if valid.any() else None})
                controls[field]=changes
            case['paired_bootstrap_membership_shifts'][selection]=controls
        # Integrated component fractions use linear shell sums, avoiding profile
        # ratios in low-signal shells. All intervals have exact 2/3/4 boundaries.
        fractions={}
        for quantity,sname in [('pressure','y'),('electrons','electrons'),('gas','mass')]:
            totals=[]
            for radius in M.RADII:
                inside=P.EDGES[1:]<=radius+1e-12
                comp=data['sums'][rows,:,P.SUM_NAMES.index(sname)][:,:,inside].sum(axis=2)
                # Equal normalized-halo weights, matching fitted profile stacks.
                normalizer={'pressure':data['p200_pa'][rows]*halos['r_200_crit_mpc'][rows]**3,
                            'electrons':data['critical_density_msun_mpc3'][rows]*halos['r_200_crit_mpc'][rows]**3,
                            'gas':data['critical_density_msun_mpc3'][rows]*halos['r_200_crit_mpc'][rows]**3}[quantity]
                totals.append((comp/normalizer[:,None]).mean(axis=0))
            totals=np.array(totals);fractions[quantity]={'component_integrals_relative_units':totals,
                'component_fraction':totals/totals.sum(axis=1)[:,None],
                'retained_fraction_of_4R':totals.sum(axis=1)/totals[2].sum()}
        case['integrated_fractions']=fractions
        np.savez_compressed(OUT/(label+'_profiles.npz'),**stats)
        np.savez_compressed(OUT/(label+'_parameters.npz'),halo_ids=halos['halo_index'][rows],
            mass_msun=halos['m200c_msun'][rows],parameters=individual,accepted=individual_good,
            status=individual_status,rms_percent=individual_rms,bootstrap=boot,
            control_bootstrap=control_boot,control_selections=np.array(['no_other','no_satellites_or_other']),
            strict_resolution_supported=supported)
        dump(OUT/(label+'_fits.json'),case);cases.append(case)
        print(json.dumps({'bin':label,'halos':n,'strict_supported':supported,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    with (OUT/'fit_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    dump(OUT/'study_summary.json',{'bins':cases,'bootstrap_draws':BOOTSTRAPS,
        'bootstrap_scope':'Paired halo resamples for mean-fit sensitivity, diagonal empirical objective; no spatial/cosmic-variance inference.',
        'methodology':'same halos and same base mask for 2/3/4R; original all-gas targets and membership controls; 20% halo holdout when N>=20',
        'wall_seconds':time.perf_counter()-started,'CPU_seconds':time.process_time(),
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


if __name__=='__main__':main()
