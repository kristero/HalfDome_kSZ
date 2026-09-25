"""Independent CLASS-SZ B12 profile check and matched-cosmology HMF reference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from classy_sz import Class
from projection import shape

ROOT=Path(__file__).resolve().parent
H=.68


def main():
    c=Class()
    for k in ['T_ncdm','deg_ncdm','m_ncdm','omega_ncdm','Omega_ncdm']:c.pars.pop(k,None)
    c.set(dict(h=H,omega_b=.049*H**2,omega_cdm=.261*H**2,n_s=.9667,sigma8=.8159,
        N_ncdm=0,N_ur=3.046,output='tSZ_1h,tSZ_2h',pressure_profile='B12',x_outSZ=4.,
        z_min=.002141595,z_max=3.8554277,M_min=4458834966504.787,M_max=2573960654045765.,
        ell_min=10.,ell_max=8192.,dell=0.,dlogell=.05,mass_function='T08M200c',
        concentration_parameter='D08',ndim_masses=200,ndim_redshifts=100,
        n_m_pressure_profile=100,n_z_pressure_profile=100,hm_consistency=0,
        use_fft_for_profiles_transform=1,N_samp_fftw=4096))
    start=time.monotonic();c.compute()
    cases=json.loads((ROOT/'inputs/cases.json').read_text())
    cases={k:cases[k] for k in ['Battaglia12','FL_L1_m9']}
    cases['shallow']=[18.1,.497,2.8,.154,-.00865,-.2,-.758,.731,-.5]
    records=[];profiles=[]
    for label,p in cases.items():
        for mass in [1e13,1e14,1e15]:
            for z in [.1,.5,2.]:
                xc=p[1]*(mass/1e14)**p[4]*(1+z)**p[7]
                beta=p[2]*(mass/1e14)**p[5]*(1+z)**p[8]
                amp=p[0]*(mass/1e14)**p[3]*(1+z)**p[6]
                radii=np.geomspace(1e-5,4,100)
                # CLASS-SZ halo masses are Msun/h. Its B12 mass pivot is
                # internally converted to physical Msun by the background h.
                actual=np.array([c.get_pressure_P_over_P_delta_at_x_M_z_b12_200c(
                    x,mass*H,z,A_P0=p[0],A_xc=p[1],A_beta=p[2],
                    alpha_m_P0=p[3],alpha_m_xc=p[4],alpha_m_beta=p[5],
                    alpha_z_P0=p[6],alpha_z_xc=p[7],alpha_z_beta=p[8],
                    alphap_m_P0=p[3],alphap_m_xc=p[4],alphap_m_beta=p[5]) for x in radii])
                expected=amp*shape(radii,xc,beta)
                err=float(np.max(np.abs(actual/expected-1)))
                records.append(dict(case=label,mass_msun=mass,z=z,max_relative_error=err))
                profiles.append(np.c_[radii,actual,expected])
    cl=c.cl_sz()
    np.savez(ROOT/'results/classsz_sphere4_h068.npz',ell=cl['ell'],
        dl_1h=np.array(cl['1h'])/1e12,dl_2h=np.array(cl['2h'])/1e12)
    np.savez(ROOT/'results/classsz_pressure_profiles.npz',profiles=profiles,
        metadata_json=json.dumps(records))
    result=dict(cases=records,maximum_relative_error=max(r['max_relative_error'] for r in records),
        seconds=time.monotonic()-start,h=H,Omega_b=.049,Omega_c=.261,
        units='dimensionless Compton-y D_ell',
        cosmology_scope='Preserved painter geometry and baryon fraction; sigma8/ns inherited from native HalfDome',
        scope='3-D profile identity; halo-model spectrum is not a fixed-catalogue painting truth')
    (ROOT/'results/classsz.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    c.struct_cleanup();c.empty()
    assert result['maximum_relative_error']<1e-7


if __name__=='__main__':main()
