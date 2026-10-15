"""Measured halo-only controls from cached voxels; no particle reread.

Halo gas means the unique geometric union of selected central-halo spheres.
It is not a particle binding classification. Voxel-centre masks inherit the
0.98 Mpc resolution limit, especially for small halos and the 1R cut.
"""
from pathlib import Path
import hashlib
import time
import h5py
import numpy as np
from numba import njit
import physics as P
import field_statistics as S

ROOT=Path(__file__).resolve().parent


@njit(cache=True)
def union_layers(position,radius,origin,side,n):
    layer=np.full((n,n,n),5,np.uint8)
    dx=side/n
    for h in range(len(radius)):
        p=position[h]-origin;r=radius[h]
        lo=np.floor((p-4*r)/dx).astype(np.int64)
        hi=np.floor((p+4*r)/dx).astype(np.int64)
        for i in range(max(0,lo[0]),min(n-1,hi[0])+1):
            for j in range(max(0,lo[1]),min(n-1,hi[1])+1):
                for k in range(max(0,lo[2]),min(n-1,hi[2])+1):
                    x=np.sqrt(((i+.5)*dx-p[0])**2+((j+.5)*dx-p[1])**2+((k+.5)*dx-p[2])**2)/r
                    if x<=4:
                        label=max(1,int(np.ceil(x)))
                        layer[i,j,k]=min(layer[i,j,k],label)
    return layer


def main():
    start=time.time()
    with h5py.File(ROOT/'halos_flamingo.h5','r') as f:
        mass=f['mass'][:];pos=f['position'][:]
    radius=P.scales(mass,P.SOURCE)[0]
    output=ROOT/'measured_halo_union_compact.h5'
    with h5py.File(ROOT/'measured_flamingo.h5','r') as source,h5py.File(output,'w') as out:
        origins=source['region_origins_mpc'][:]
        side=float(source.attrs['region_side_mpc']);n=int(source.attrs['grid_n']);volume=(side/n)**3
        maps=np.zeros((8,4,3,n,n))
        group=out.create_group('statistics')
        values={}
        for name in ['pressure','electrons','y','tau','dm','deltaT']:
            values[name]={key:[] for key in ['pdf','atoms','mean','rms']}
            if name not in ['pressure','electrons']:values[name]['power']=[]
        filling=[]
        for r,origin in enumerate(origins):
            take=np.all((pos+4*radius[:,None]>=origin)&(pos-4*radius[:,None]<origin+side),axis=1)
            layer=union_layers(pos[take],radius[take],origin,side,n)
            cube=source['grid_sums'][r][:]
            filling.append([float(np.mean(layer<=cut)) for cut in [1,2,3,4]])
            local={name:{key:[] for key in values[name]} for name in values}
            for cut in [1,2,3,4]:
                mask=layer<=cut
                for q,name in [(0,'pressure'),(1,'electrons')]:
                    v=np.where(mask,cube[q],0).astype(float)/volume
                    if q==0:v*=P.Y_TO_P_VOLUME
                    h,a=S.probability(v,name)
                    for key,data in dict(pdf=h,atoms=a,mean=float(v.mean()),rms=float(np.sqrt(np.mean(v*v)))).items():local[name][key].append(data)
                sums=np.stack([np.sum(np.where(mask,cube[q],0),axis=2,dtype=float) for q in range(3)])
                maps[r,cut-1]=sums
                obs=P.map_observables(sums,(side/n)**2)
                for q,name in enumerate(['y','tau','dm','deltaT']):
                    v=obs[q];h,a=S.probability(v,name);k,power,counts=S.power2d(v,side)
                    for key,data in dict(pdf=h,atoms=a,power=power,mean=float(v.mean()),rms=float(np.sqrt(np.mean(v*v)))).items():local[name][key].append(data)
            for name in values:
                for key in values[name]:values[name][key].append(local[name][key])
        for name in values:
            for key,v in values[name].items():group.create_dataset(name+'_'+key,data=v)
        group.create_dataset('k_per_mpc',data=k)
        group.create_dataset('power_mode_counts',data=counts)
        out.create_dataset('region_sums',data=maps,compression='gzip',compression_opts=1)
        out.create_dataset('filling_fraction',data=filling)
        out.create_dataset('region_origins_mpc',data=origins)
        out.attrs.update(region_side_mpc=side,map_n=n,grid_n=n,
            selection='unique geometric union of central-halo spheres; all gas; 1e13 <= M200c < 10^15.5 Msun')
        assert np.all(np.diff(maps[:,:,:2],axis=1)>=-1e-12)
    P.save_json(ROOT/'halo_union_complete.json',dict(completed=True,wall_seconds=time.time()-start,
        bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        nested_positive_projections_monotone=True,voxel_size_mpc=side/n,
        limitation='Voxel-centre mask, not exact particle-sphere membership; no gas binding removal.'))


if __name__=='__main__':main()
