"""Reduce completed measured grids to compact comparable statistics; preserve originals."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
import physics as P
import field_statistics as S

ROOT = Path(__file__).resolve().parent


def main():
    assert (ROOT/'extraction_complete.json').exists()
    source = ROOT/'measured_flamingo.h5'
    output = ROOT/'measured_flamingo_compact.h5'
    with h5py.File(source,'r') as f,h5py.File(output,'w') as out:
        side = float(f.attrs['region_side_mpc'])
        n = int(f.attrs['grid_n'])
        for key in ['whole_sums','region_sums','region_origins_mpc','totals','particle_volume_hist',
                    'particle_volume_atoms','pressure_log10_edges_pa','electrons_log10_edges_cm3']:
            out.create_dataset(key,data=f[key][:],compression='gzip',compression_opts=1)
        for key,value in f.attrs.items():
            out.attrs[key]=value
        group = out.create_group('statistics')
        for name in ['pressure','electrons']:
            hist = []; atoms=[]; coarse_hist=[];coarse_atoms=[];means=[];rms=[]
            for r in range(8):
                q = 0 if name=='pressure' else 1
                cube = f['grid_sums'][r,q].astype(float)/(side/n)**3
                if name=='pressure':
                    cube *= P.Y_TO_P_VOLUME
                h,a = S.probability(cube,name);hist.append(h);atoms.append(a)
                means.append(float(cube.mean()));rms.append(float(np.sqrt(np.mean(cube*cube))))
                h,a = S.probability(S.rebin_cube(cube),name);coarse_hist.append(h);coarse_atoms.append(a)
            for key,value in dict(pdf=hist,atoms=atoms,coarse_pdf=coarse_hist,coarse_atoms=coarse_atoms,mean=means,rms=rms).items():
                group.create_dataset(name+'_'+key,data=value)
        maps = f['region_sums'][:]
        pixel = (side/int(f.attrs['map_n']))**2
        for q,name in enumerate(['y','tau','dm','deltaT']):
            hist=[];atoms=[];powers=[];means=[];rms=[]
            for r in range(8):
                image = P.map_observables(maps[r],pixel)[q]
                h,a = S.probability(image,name);hist.append(h);atoms.append(a)
                means.append(float(image.mean()));rms.append(float(np.sqrt(np.mean(image*image))))
                k,power,counts = S.power2d(image,side);powers.append(power)
            group.create_dataset(name+'_pdf',data=hist)
            group.create_dataset(name+'_atoms',data=atoms)
            group.create_dataset(name+'_power',data=powers)
            group.create_dataset(name+'_mean',data=means)
            group.create_dataset(name+'_rms',data=rms)
        group.create_dataset('k_per_mpc',data=k)
        group.create_dataset('power_mode_counts',data=counts)
    P.save_json(ROOT/'compact_complete.json',dict(completed=True,
        file=output.name,bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        grid_source_preserved=str(source)))


if __name__=='__main__':
    main()
