"""Read-only metadata listing (directory names, HDF5 group keys, dataset shapes/attrs) of selected COSMA FLAMINGO products."""
import json, time
import hdfstream
t0 = time.time()
root = hdfstream.open('cosma', '/')
out = {}
def ls(path):
    try:
        return sorted(list(root[path].keys()))
    except Exception as e:
        return 'ERR ' + repr(e)[:300]
L = 'FLAMINGO/L1_m9'
variants = ls(L)
out['variants'] = {}
for v in variants:
    out['variants'][v] = ls(f'{L}/{v}')
out['L1_m9_SOAP_tail'] = ls(f'{L}/L1_m9/SOAP-HBT')[-5:]
out['L1_m9_snap0077_dir'] = ls(f'{L}/L1_m9/snapshots/flamingo_0077')
out['L1_m9_snap0057_dir'] = ls(f'{L}/L1_m9/snapshots/flamingo_0057')
out['L1_m9_power_spectra'] = ls(f'{L}/L1_m9/power_spectra')[:20]
out['L1_m9_healpix'] = ls(f'{L}/L1_m9/healpix_maps')
out['L1_m9_halo_lightcone'] = ls(f'{L}/L1_m9/halo_lightcone')
out['L1_m9_integrated_maps'] = ls(f'{L}/L1_m9/integrated_maps')
out['L1_m9_particle_lightcones'] = ls(f'{L}/L1_m9/particle_lightcones')[:20]
out['L1_m9_HBT'] = ls(f'{L}/L1_m9/HBT-HERONS')[:10]
out['L2p8_downsampled'] = ls('FLAMINGO/L2p8_m9/L2p8_m9/snapshots_downsampled')[:100]
out['L1_m8_snapshots'] = ls('FLAMINGO/L1_m8/L1_m8/snapshots')
out['L1_m8_reduced'] = ls('FLAMINGO/L1_m8/L1_m8/snapshots_reduced')
out['L2p8_snapshots'] = ls('FLAMINGO/L2p8_m9/L2p8_m9/snapshots')
out['elapsed_s'] = time.time() - t0
json.dump(out, open('cosma_listing2.json', 'w'), indent=1)
for k, v in out.items():
    if k == 'variants':
        for vv, c in v.items():
            print('variant', vv, json.dumps(c)[:300])
    else:
        print(k, len(v) if isinstance(v, list) else '', json.dumps(v)[:500])
