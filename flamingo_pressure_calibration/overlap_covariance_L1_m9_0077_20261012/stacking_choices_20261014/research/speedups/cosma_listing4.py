"""Read-only COSMA metadata: variant snapshot redshifts, reduced/downsampled snapshot contents for a variant, SOAP keys."""
import json, time
import numpy as np
import hdfstream
t0 = time.time()
root = hdfstream.open('cosma', '/')
out = {}
L = 'FLAMINGO/L1_m9'
v = 'fgas-8sigma'
snaps = sorted(root[f'{L}/{v}/snapshots'].keys())
out['variant_snapshots'] = {}
for s in snaps:
    n = s.split('_')[1]
    try:
        d = sorted(root[f'{L}/{v}/snapshots/{s}'].keys())
        f = root[f'{L}/{v}/snapshots/{s}/{s}.hdf5']
        h = f['Header'].attrs
        npt = np.asarray(h['NumPart_Total']).astype(np.int64); hw = np.asarray(h['NumPart_Total_HighWord']).astype(np.int64)
        out['variant_snapshots'][n] = {'z': round(float(np.asarray(h['Redshift']).ravel()[0]), 4), 'gas': int(npt[0] + (hw[0] << 32)), 'dir': d,
                                       'has_membership_field': 'GroupNr_bound' in f['PartType0'].keys()}
    except Exception as e:
        out['variant_snapshots'][n] = 'ERR ' + repr(e)[:200]
# reduced + downsampled for the variant at z=0
for kind in ('snapshots_reduced', 'snapshots_downsampled'):
    try:
        f = root[f'{L}/{v}/{kind}/flamingo_0077.hdf5']
        h = f['Header'].attrs
        npt = np.asarray(h['NumPart_Total']).astype(np.int64); hw = np.asarray(h['NumPart_Total_HighWord']).astype(np.int64)
        out[kind] = {'npart': (npt + (hw << 32)).tolist(), 'PartType0': sorted(list(f['PartType0'].keys())) if 'PartType0' in f else None,
                     'groups': sorted(list(f.keys()))}
    except Exception as e:
        out[kind] = 'ERR ' + repr(e)[:300]
# SOAP top-level groups and reduced flag for the variant
try:
    f = root[f'{L}/{v}/SOAP-HBT/halo_properties_0077.hdf5']
    out['soap_groups'] = sorted(list(f.keys()))
    out['soap_SOAP_keys'] = sorted(list(f['SOAP'].keys()))
    out['soap_SO_apertures'] = sorted(list(f['SO'].keys()))
    out['soap_other_apertures'] = {g: sorted(list(f[g].keys())) for g in ('ExclusiveSphere', 'InclusiveSphere', 'ProjectedAperture') if g in f}
    out['soap_rows'] = int(f['InputHalos/HaloCatalogueIndex'].shape[0])
except Exception as e:
    out['soap'] = 'ERR ' + repr(e)[:300]
# L1_m9 fiducial power spectra / healpix shells / halo lightcone counts
try:
    out['L1_m9_power_spectra_types'] = sorted(set(x.rsplit('_', 1)[0] for x in root[f'{L}/L1_m9/power_spectra'].keys()))
except Exception as e:
    out['L1_m9_power_spectra_types'] = 'ERR ' + repr(e)[:200]
try:
    out['L1_m9_nside4096'] = sorted(root[f'{L}/L1_m9/healpix_maps/nside_4096'].keys())
    out['L1_m9_halo_lightcone0'] = sorted(root[f'{L}/L1_m9/halo_lightcone/lightcone0'].keys())[:6]
    out['L1_m9_integrated_maps'] = {k: sorted(root[f'{L}/L1_m9/integrated_maps/{k}'].keys()) for k in ('broxterman24', 'mcdonald26', 'yang26')}
except Exception as e:
    out['L1_m9_maps'] = 'ERR ' + repr(e)[:200]
out['elapsed_s'] = time.time() - t0
json.dump(out, open('cosma_listing4.json', 'w'), indent=1)
print(json.dumps(out, indent=0)[:9000])
