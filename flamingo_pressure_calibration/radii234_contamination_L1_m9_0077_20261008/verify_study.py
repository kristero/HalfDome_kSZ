"""Scientific consistency checks; these do not establish low-mass convergence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pilot_physics as P
from models import shell,fit_profile
from projection_model import project_map
from numpy.polynomial.legendre import leggauss

HERE=Path(__file__).resolve().parent


def main():
    h=dict(np.load(HERE/'selected_halos.npz'))
    d=dict(np.load(HERE/'measurements.npz'))
    maps=dict(np.load(HERE/'projection_stacks.npz'))
    checks=json.loads((HERE/'swift_checks.json').read_text())
    assert checks['all_particle_hashes_match'] and checks['checked_halos']==45
    assert len(np.unique(h['halo_index']))==4375
    assert np.all(h['bound_mass_msun']>0)
    assert np.all(np.isfinite(h['centre_offset_R200c']))
    assert np.all(np.isfinite(h['max_satellite_bound_mass_ratio_R2']))
    assert len(np.unique(np.column_stack((h['pair_halo'],h['pair_cell'])),axis=0))==len(h['pair_halo'])
    assert np.array_equal(np.bincount(h['mass_bin']),[300]*10+[895,326,127,21,6])
    assert np.all(d['sums']>=0)
    for key in ('pressure_components','electron_components','gas_components'):
        assert np.all(np.isfinite(d[key])) and np.all(d[key]>=0)
    for component,target in [('pressure_components','pressure_ratio'),('electron_components','tau_density_ratio'),('gas_components','gas_density_ratio')]:
        assert np.allclose(d[component].sum(axis=1),d[target],rtol=3e-12,atol=1e-20)
    # An earlier extraction used identical high-mass halos, but lacked the exact
    # 2R shell boundary. Recombine the two split shells before comparing sums.
    prior=HERE.parent/'scatter_r4_bins02_L1_m9_0077_20261008'
    ph=dict(np.load(prior/'selected_halos.npz'));pd=dict(np.load(prior/'full_profiles.npz'))
    lookup={int(x):i for i,x in enumerate(h['halo_index'])}
    rows=np.array([lookup[int(x)] for x in ph['halo_index']])
    rebin=np.array([(P.EDGES[:-1]>=lo-1e-12)&(P.EDGES[1:]<=hi+1e-12) for lo,hi in zip(pd['edges'][:-1],pd['edges'][1:])])
    old_errors={}
    for k,name in enumerate(P.SUM_NAMES):
        measured=d['sums'][rows,:,k].sum(axis=1)@rebin.T
        previous=pd[name]
        assert np.array_equal(measured,previous) if name=='count' else np.allclose(measured,previous,rtol=2e-9,atol=1e-22),name
        old_errors[name]=float(np.max(np.abs(measured-previous)/np.maximum(np.abs(previous),1e-100)))
    # Every rotation conserves each integrated field. The ten-view mean has the
    # same integral as the original z view; radial structure may differ.
    flux=maps['map_sums'].sum(axis=(-1,-2))
    assert np.allclose(flux[:,:,0],flux[:,:,1],rtol=3e-12,atol=1e-15)
    shell_volume=4*np.pi/3*np.diff(P.EDGES**3)
    for b in range(15):
        rows=h['mass_bin']==b
        for ri,radius in enumerate((2,3,4)):
            inside=P.EDGES[1:]<=radius+1e-12
            for q,key in enumerate(('pressure_components','electron_components','gas_components')):
                integrated=(d[key][rows][:,:,inside]*shell_volume[inside]).sum(axis=(0,2))
                assert np.allclose(flux[b,0,1,ri,:,q]*(8/64)**2,integrated,rtol=5e-10,atol=1e-15),(b,ri,key)
    # Projection normalization / quadrature and exact noiseless coefficient
    # recovery are numerical gates, not fits to simulation measurements.
    nodes,w=leggauss(512);quad={};recovery={}
    from models import shape
    for family,params in [('b12',np.array([12.,.6,4.8])),('b16',np.array([3500.,.8,4.]))]:
        recovery[family]={};quad[family]={}
        profile=shell(family,params)
        for radius in (2,3,4):
            mask=(P.EDGES[:-1]>.04)&(P.EDGES[1:]<=radius)
            fit=fit_profile(family,profile,mask)
            error=float(np.max(np.abs(np.array(fit['parameters'])/params-1)))
            assert error<1e-5
            x=(nodes+1)*radius/2
            volume=4*np.pi*np.sum(shape(family,x,params)*x*x*w)*radius/2
            projected=project_map(family,params,radius).sum()*(8/64)**2
            difference=float(abs(projected/volume-1));assert difference<.015
            recovery[family][str(radius)]=error;quad[family][str(radius)]=difference
    # SOAP is an independently measured aperture check, with known differences
    # in scalar CGS rounding. It is not a spatial-cutout completeness proof.
    inner=P.EDGES[1:]<=1+1e-12
    gas=d['sums'][:,:,P.SUM_NAMES.index('mass')][:,:,inner].sum(axis=(1,2))
    y=d['sums'][:,:,P.SUM_NAMES.index('y')][:,:,inner].sum(axis=(1,2))
    def aperture(value,reference):
        valid=reference>0
        fractional=value[valid]/reference[valid]-1
        return {'positive_reference_halos':int(valid.sum()),'zero_reference_halos':int((~valid).sum()),
                'fractional_difference_16_50_84':np.percentile(fractional,[16,50,84]).tolist(),
                'max_absolute_fractional_difference':float(np.abs(fractional).max())}
    result={'passed':True,'high_mass_old_shell_sums_max_relative_errors':old_errors,
        'components_sum_to_all_gas':True,'orientation_integrals_conserved':True,'map_shell_integrals_match':True,
        'noiseless_parameter_recovery_max_relative_error':recovery,
        'projection_integral_fractional_error':quad,'SWIFT_checks':checks,
        'SOAP_gas_R200c':aperture(gas,h['gasmass_200_crit_msun']),
        'SOAP_Y_R200c':aperture(y,h['y_200_crit_mpc2']),
        'hydrogen_mass_weighted_over_4R_16_50_84':np.nanpercentile(d['xh_mass_weighted'],[16,50,84]).tolist(),
        'native_field_convention':'Cooling-table Compton-Y and electron number density; both are zero for star-forming particles according to actual dataset headers.',
        'pressure_neKT_to_stored_Y':{},
        'membership_limit':'Original-index and original-particle-order contract plus equal chunk row counts; no independent ParticleID membership join is available.',
        'low_mass_resolution_convergence_established':False}
    for b in range(15):
        take=(h['mass_bin']==b)[:,None]&(d['pressure_pa']>0)&(d['pressure_nekt_pa']>0)&(P.EDGES[:-1][None]>=.04)
        ratio=d['pressure_nekt_pa'][take]/d['pressure_pa'][take]
        result['pressure_neKT_to_stored_Y'][str(b)]={'shells':len(ratio),
            'ratio_16_50_84':np.percentile(ratio,[16,50,84]).tolist() if len(ratio) else None}
    (HERE/'scientific_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='SWIFT_checks'},indent=2))


if __name__=='__main__':main()
