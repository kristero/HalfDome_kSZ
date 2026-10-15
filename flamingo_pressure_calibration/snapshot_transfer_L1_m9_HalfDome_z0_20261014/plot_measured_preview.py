"""Low-memory direct-FLAMINGO figures while idark is unavailable.

The input is a 34 MB compact export, not the particle snapshot or 3D grids.
Outputs have a distinct preview prefix so final comparison figures survive.
"""
from pathlib import Path
import hashlib
import json
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import physics as P

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'


def main():
    (ROOT/'preview.pid').write_text(str(os.getpid()))
    # WSL available memory; the Windows wrapper also checks the host.
    available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines()
                       if v.startswith('MemAvailable:')))*1024
    assert available >= 5*1024**3, 'Preserve at least 5 GiB available memory.'
    metadata=json.loads((ROOT/'strw_execution_metadata.json').read_text())
    receipt=metadata['measured_flamingo_compact_export.json']
    source=ROOT/'measured_flamingo_compact.h5'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['sha256']
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'axes.titlesize':18,
                         'xtick.labelsize':14,'ytick.labelsize':14,'savefig.dpi':200})
    FIG.mkdir(exist_ok=True)
    manifest={}
    with h5py.File(source,'r') as f:
        whole=P.map_observables(f['whole_sums'][:],(1000/1024)**2)
        region=P.map_observables(f['region_sums'][0],(250/512)**2)
    for index,name,label in [(0,'y',r'$y$'),(2,'dm',r'DM [pc cm$^{-3}$]'),
                              (3,'deltaT',r'$\Delta T_{\rm kSZ}\ [\mu{\rm K}]$')]:
        images=[whole[index],region[index]]
        limit=max(float(np.max(np.abs(v))) for v in images)
        if name=='deltaT':
            norm=SymLogNorm(linthresh=max(limit*1e-3,1e-7),vmin=-limit,vmax=limit,base=10)
            cmap='RdBu_r'
        else:
            positives=np.concatenate([v[v>0] for v in images])
            norm=LogNorm(vmin=max(float(np.percentile(positives,.5)),1e-30),vmax=limit)
            cmap='viridis'
        fig,axes=plt.subplots(1,2,figsize=(13,5.7),constrained_layout=True)
        for ax,image,side,title in zip(axes,images,[1000,250],['Whole box','250 Mpc region']):
            displayed=np.ma.masked_less_equal(image,0) if name!='deltaT' else image
            im=ax.imshow(displayed.T,origin='lower',extent=[0,side,0,side],cmap=cmap,norm=norm)
            ax.set(xlabel='x [Mpc]',ylabel='y [Mpc]',title=title)
        fig.colorbar(im,ax=list(axes),label=label,shrink=.85)
        path=FIG/('preview_flamingo_'+name+'.png')
        fig.savefig(path,bbox_inches='tight');plt.close(fig)
        manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    P.save_json(ROOT/'preview_complete.json',dict(completed=True,input_sha256=receipt['sha256'],
        figures=manifest,scope='Direct FLAMINGO gas only; actual gas velocities; NGP particle-centre maps.'))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
