"""Small physical closure gates, run inside each allocated cluster worker."""
import csv
from pathlib import Path
import numpy as np
import physics as P
import painting as M
import field_statistics as S


def run(root,fits):
    catalogue=dict(mass=np.array([1e14]),position=np.array([[10.13,9.87,10.07]]),velocity=np.array([[0.,0.,300.]]))
    case=dict(p_fit=2,e_fit=3,p_cut=4,e_cut=3,baseline=True)
    sums,cube,info=M.region_fields(catalogue,fits,case,np.zeros(3),side=20,map_n=24,grid_n=20)
    conservation=np.array([sums[0].sum(),sums[1].sum()])/info['integrated_full_spheres']-1
    assert np.max(np.abs(conservation))<1e-12
    obs=P.map_observables(sums,(20/24)**2)
    expected=-P.T_CMB*1e6*300/(P.c/1000)*obs[1]
    assert np.max(np.abs(obs[3]-expected))<1e-12*max(np.max(np.abs(expected)),1e-100)
    for name,values in [('pressure',cube[0]*P.Y_TO_P_VOLUME/(20/20)**3),
                        ('electrons',cube[1]/(20/20)**3),('deltaT',obs[3])]:
        h,a=S.probability(values,name)
        assert abs(h.sum()+a.sum()-1)<1e-12
    for field in ['pressure','electrons']:
        _,small,_=M.kernels(fits,field,2,2,baseline=True)
        _,large,_=M.kernels(fits,field,2,4,baseline=True)
        assert np.all(large[:,0]>small[:,0])
    catalogue['position']=np.array([[10.13,9.87,.01]])
    clipped,_,_=M.region_fields(catalogue,fits,case,np.zeros(3),side=20,map_n=24,grid_n=20)
    assert 0<clipped[0].sum()<sums[0].sum()
    legacy=root/'inputs/legacy_fit_comparison.csv'
    differences=[]
    if legacy.exists():
        with legacy.open() as stream:
            for row in csv.DictReader(stream):
                if row['field'] not in ['pressure','electrons'] or row['selection'] not in ['all','no_other']:
                    continue
                b=int(round((float(row['bin_low'])-12.5)/.2))
                fit=next((r for r in fits['results'] if r['bin']==b and r['field']==row['field']
                    and r['component']==row['selection'] and r['cut']==int(float(row['radius_R200c'])) and r['valid']),None)
                if fit is None:
                    continue
                old=[float(row[k]) for k in ['P0_or_rho0','xc_or_alpha','beta']]
                x=np.geomspace(.1,4,100)
                differences.append(float(np.max(np.abs(P.shape(row['field'],x,fit['parameters'])/P.shape(row['field'],x,old)-1))))
        assert max(differences)<.002, 'Default fit differs materially from the established SD-weighted profile.'
    result=dict(passed=True,spherical_integral_relative_error=conservation.tolist(),
        slab_boundary_y_fraction=float(clipped[0].sum()/sums[0].sum()),
        legacy_profiles_checked=len(differences),legacy_profile_max_relative_difference=max(differences) if differences else None,
        caveat='These are numerical/contract gates, not independent scientific validation.')
    return result
