#!/usr/bin/env python3
"""Record immutable input identities; only stream bytes, with no inference or fitting."""
import json
from pathlib import Path
import tomllib

from diagnose import PARENT, OLD, SEEDS, digest, write_json


def main():
    root=Path(__file__).resolve().parent
    inputs=[PARENT/'comparison/COMPLETE.json',PARENT/'comparison/VERIFIED.json',
            PARENT/'plan.json',PARENT/'data/grid.npz',PARENT/'data/local_model.npz',
            PARENT/'data/band_operator.npy',PARENT/'data/original_merlin_observations.npy',
            PARENT/'candidates/original_merlin/evaluation_final.json',
            PARENT/'candidates/merlin_band50_full/evaluation_final.json',
            PARENT/'candidates/original_merlin/posterior_final.npy',
            OLD/'reference/whitening_pca.npz']
    for directory in [OLD/'models/linear',OLD/'models/logit',
                      PARENT/'candidates/merlin_band50_full',
                      PARENT/'candidates/merlin_band50_full/bounded']:
        inputs.extend(directory.glob('merlin_seed*.pt'))
        inputs.extend(directory.glob('merlin_seed*.json'))
        inputs.append(directory/'features_merlin.npz')
    inputs.append(PARENT/'candidates/merlin_band50_full/compression.npz')
    inputs.extend((PARENT/'snapshot/inference').glob('*.py'))
    plan=json.loads((PARENT/'plan.json').read_text())
    for case in plan['cases']:
        directory=PARENT/case['path']
        inputs.extend(directory/name for name in ['clean_cl.npy','beam_corrected_cl.npy','noise_cl.npy','complete.toml'])
    catalogues=[]
    for seed in SEEDS:
        directory=PARENT/f'results/seed_{seed}'
        inputs.extend(directory/name for name in ['complete.toml','VALIDATED.json','beam_bl.npy','unmasked_clean_cl.npy'])
        meta=tomllib.loads((directory/'complete.toml').read_text())
        catalogue=Path(meta['catalogue'])
        assert catalogue.stat().st_size==meta['catalogue_bytes']
        catalogues.append(dict(path=str(catalogue),bytes=catalogue.stat().st_size,
                               mtime_ns=catalogue.stat().st_mtime_ns,
                               selected_halos=meta['selected_halos'],hash_not_recomputed=True))
    for name in ['code/paint_seed.jl',
                 'snapshot/tSZ_64k_flat_prior_SO_baseline_deproj0/engine/benchmark.jl',
                 'snapshot/tSZ_64k_flat_prior_SO_baseline_deproj0/halfdome_sources/code/halfdome/tSZ_visuals/config.jl']:
        inputs.append(PARENT/name)
    paths=sorted(set(inputs))
    hashes={str(p):digest(p) for p in paths}
    assert hashes[str(PARENT/'comparison/COMPLETE.json')]==json.loads((root/'COMPLETE.json').read_text())['parent_complete_sha256']
    write_json(root/'input_sha256.json',dict(sha256=hashes,catalogue_metadata=catalogues,
        scope='Consumed spectra, model inputs, trained weights and sources. Catalogues were reused through saved spectra.'))
    print('INPUTS VERIFIED',len(paths),'files;',sum(p.stat().st_size for p in paths),'bytes streamed')


if __name__=='__main__':
    main()
