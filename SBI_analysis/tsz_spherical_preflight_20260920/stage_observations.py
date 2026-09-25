"""Reuse previously processed FLAMINGO observations; no simulation downloads."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'flamingo_tsz/cluster_results/results'
ell=np.arange(7980);edges=np.r_[np.arange(80,7881,200),7980]
def bins(cl):
    dl=cl[:7980]*ell*(ell+1)/(2*np.pi)
    return np.array([np.average(dl[a:b],weights=2*ell[a:b]+1) for a,b in zip(edges[:-1],edges[1:])])
out=ROOT/'diagnostic_256/observations';out.mkdir(exist_ok=True)
manifest={}
for variant in ['L1_m9','fgas-8sigma','Mstar-1sigma']:
    clean=OLD/variant/'masked_clean_cl.npy';noisy=OLD/variant/'masked_noisy_cross_cl.npy'
    metadata=OLD/variant/'complete.toml'
    clean_cl=np.load(clean);noisy_cl=np.load(noisy)
    keep=ell[80:];factor=keep*(keep+1)/(2*np.pi)
    np.savez(out/(variant+'.npz'),clean_dl=bins(clean_cl),noisy_dl=bins(noisy_cl),
        ell_unbinned=keep,clean_dl_unbinned=clean_cl[keep]*factor,
        noisy_dl_unbinned=noisy_cl[keep]*factor)
    manifest[variant]=dict(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [clean,noisy,metadata]},
        convention='Existing 2 arcmin beam, fsky .4 mask and fixed observation noise; direct FL cosmology',
        moped_input='7900 individual D_ell values, ell=80..7979; no binning',
        interpretation='Out-of-family observation, not a gNFW parameter truth')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Staged three existing FLAMINGO observations; no maps downloaded.')
