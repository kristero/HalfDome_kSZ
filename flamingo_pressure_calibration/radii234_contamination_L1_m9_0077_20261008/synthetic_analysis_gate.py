"""Small control-flow gate for fitting, scatter and rendering; not science data.

The temporary synthetic population is removed after a successful check. It has
no relation to the measured simulation coefficients or resolution conclusions.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
import fit_study as F
import measure as M
import pilot_physics as P
import plot_study as V
from models import shell
from projection_model import project_map


def main():
    started=__import__('time').perf_counter()
    with tempfile.TemporaryDirectory(prefix='HalfDome_radius_fit_gate_') as directory:
        root=Path(directory);n=30;rng=np.random.default_rng(91)
        fractions=np.array([.6,.1,.1,.2]);normal=rng.normal(0,.06,(n,3))
        pressure=np.array([shell('b12',np.array([12.,.6,4.8])*np.exp(v)) for v in normal])
        electrons=np.array([shell('b16',np.array([3500.,.8,4.])*np.exp(v)) for v in normal])
        pc=pressure[:,None]*fractions[None,:,None];ec=electrons[:,None]*fractions[None,:,None]
        projected=np.zeros((n,2,3,4,3,M.NB));maps=np.zeros((1,2,2,3,4,3,64,64))
        for ri,radius in enumerate(M.RADII):
            for q,(family,params) in enumerate([('b12',[12.,.6,4.8]),('b16',[3500.,.8,4.]),('b16',[3500.,.8,4.])]):
                image=project_map(family,params,radius)
                projected[:,:,ri,:,q]=M.annular_means(image)[None,None,None]*fractions[None,None,:,None]
                maps[0,0,:,ri,:,q]=n*fractions[None,:,None,None]*image[None,None]
                maps[0,1,:,ri,:,q]=n/2*fractions[None,:,None,None]*image[None,None]
        sums=np.ones((n,4,11,M.NB));sums[:,:,0]=300*fractions[None,:,None]
        np.savez_compressed(root/'selected_halos.npz',halo_index=np.arange(n),m200c_msun=np.full(n,4e14),
            r_200_crit_mpc=np.ones(n),mass_bin=np.zeros(n,dtype=int),mass_edges=[14.5,14.7],disturbed_proxy=np.arange(n)<n/2,
            max_satellite_bound_mass_ratio_R2=np.where(np.arange(n)<n/3,.3,.1),centre_offset_R200c=np.where(np.arange(n)<n/2,.1,.02))
        np.savez_compressed(root/'measurements.npz',sums=sums,hsml_median_upper_x=np.full((n,M.NB),.001),
            pressure_components=pc,electron_components=ec,gas_components=ec,projected_profiles=projected,
            p200_pa=np.ones(n),critical_density_msun_mpc3=np.ones(n))
        np.savez_compressed(root/'projection_stacks.npz',map_sums=maps,
            view_profile_sums=np.broadcast_to(projected[:,1].sum(axis=0),(11,3,4,3,M.NB))[None])
        (root/'metadata_plan.json').write_text(json.dumps({'softening_physical_mpc':.001,'population_per_bin':[100]}))
        F.HERE=root;F.OUT=root/'products';F.BOOTSTRAPS=16;F.main()
        summary=json.loads((root/'products/study_summary.json').read_text());case=summary['bins'][0]
        assert case['strict_resolution_supported']
        assert all(e['individual_fit_accepted']==n for e in case['selections']['all']['pressure'])
        assert all(e['heldout']['test_halos']==6 for e in case['selections']['all']['pressure'])
        assert all(v['paired_draws']==16 for v in case['paired_bootstrap_radius_shifts']['pressure'].values())
        assert all(v['count']==n for v in case['joint_individual_pressure_electron_scatter'].values())
        V.OUT=root/'products';V.FIG=root/'figures';V.FIG.mkdir();case['bin_index']=0
        V.profile_figure(case);V.overview([case]);V.membership_fit_overview([case]);V.projection_accuracy([case])
        V.projected_comparison(case);V.triptych(case,{'map_sums':maps},'pressure',2)
        # Keep just one small rendering as gate evidence; raw synthetic arrays
        # and coefficient products have no future scientific use.
        target=Path(__file__).resolve().parent/'synthetic_gate_render.png'
        target.write_bytes((V.FIG/(case['label']+'_all_pressure_R4_maps.png')).read_bytes())
    result={'passed':True,'scope':'Synthetic control flow only; not simulation validation',
            'halos':n,'bootstrap_draws':16,'paired_radius_and_joint_scatter_checks':True,
            'wall_seconds':__import__('time').perf_counter()-started}
    (Path(__file__).resolve().parent/'synthetic_analysis_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
