"""Read-only COSMA metadata: snapshot outputs per L1_m9 hydro variant, PartType0 field list, headers, downsampled/reduced contents."""
import json, time
import numpy as np
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
hydro = ['L1_m9', 'Jet', 'Jet_fgas-4sigma', 'LS8', 'LS8_fgas-8sigma', 'Mstar-1sigma', 'Mstar-1sigma_fgas-4sigma', 'NoCooling', 'Planck',
         'PlanckNu0p24Fix', 'PlanckNu0p24Var', 'PlanckNu0p48Fix', 'fgas+2sigma', 'fgas-2sigma', 'fgas-4sigma', 'fgas-8sigma', 'L1_m9_extraDM', 'PlanckDCDM12', 'PlanckDCDM24']
out['snapshots'] = {}
out['soap'] = {}
out['other'] = {}
for v in hydro:
    s = ls(f'{L}/{v}/snapshots')
    out['snapshots'][v] = s
    so = ls(f'{L}/{v}/SOAP-HBT')
    out['soap'][v] = so if isinstance(so, str) else [x for x in so if x.startswith('halo_properties')]
    top = ls(f'{L}/{v}')
    for extra in ('snapshots_downsampled', 'snapshots_reduced'):
        if isinstance(top, list) and extra in top:
            out['other'][f'{v}/{extra}'] = ls(f'{L}/{v}/{extra}')
# header + field list of L1_m9 0077 and 0040
for snap in ('0077', '0040', '0010'):
    try:
        f = root[f'{L}/L1_m9/snapshots/flamingo_{snap}/flamingo_{snap}.hdf5']
        h = f['Header'].attrs
        npt = np.asarray(h['NumPart_Total']).tolist(); hw = np.asarray(h['NumPart_Total_HighWord']).tolist()
        out[f'hdr_{snap}'] = {'NumPart_Total': npt, 'HighWord': hw, 'z': float(np.asarray(h['Redshift']).ravel()[0]),
                             'NumFilesPerSnapshot': np.asarray(h['NumFilesPerSnapshot']).tolist(),
                             'PartType0_fields': sorted(list(f['PartType0'].keys()))}
    except Exception as e:
        out[f'hdr_{snap}'] = 'ERR ' + repr(e)[:300]
out['swift_snapshot_0077_n'] = len(ls(f'{L}/L1_m9/snapshots/flamingo_0077/swift_snapshot_0077')) if isinstance(ls(f'{L}/L1_m9/snapshots/flamingo_0077/swift_snapshot_0077'), list) else ls(f'{L}/L1_m9/snapshots/flamingo_0077/swift_snapshot_0077')
out['xray_0077'] = ls(f'{L}/L1_m9/snapshots/flamingo_0077/xray_0077')[:5]
out['membership_0077_n'] = len(ls(f'{L}/L1_m9/snapshots/flamingo_0077/membership_0077'))
# L2p8 downsampled file fields
try:
    f = root['FLAMINGO/L2p8_m9/L2p8_m9/snapshots_downsampled/flamingo_0078.hdf5']
    out['L2p8_ds_0078'] = {k: sorted(list(f[k].keys())) for k in f.keys() if k.startswith('PartType')}
    h = f['Header'].attrs
    out['L2p8_ds_0078_npart'] = np.asarray(h['NumPart_Total']).tolist()
except Exception as e:
    out['L2p8_ds_0078'] = 'ERR ' + repr(e)[:300]
out['elapsed_s'] = time.time() - t0
json.dump(out, open('cosma_listing3.json', 'w'), indent=1)
for v, s in out['snapshots'].items():
    print('snap', v, (len(s), s[:3], s[-3:]) if isinstance(s, list) else s)
for v, s in out['soap'].items():
    print('soap', v, (len(s), s[:2], s[-2:]) if isinstance(s, list) else s)
for k, s in out['other'].items():
    print('other', k, (len(s), s[:2], s[-2:]) if isinstance(s, list) else s)
for k in out:
    if k not in ('snapshots', 'soap', 'other'):
        print(k, json.dumps(out[k])[:900])
