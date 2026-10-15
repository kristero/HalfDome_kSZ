"""Read (on maas, read-only, nice 19) the SOAP columns of `columns.COLUMNS` at the study haloes' rows of
HYDRO_FIDUCIAL SOAP-HBT 0077, plus two full-column timing reads and a z = 2 aperture-unit check.
Writes ../soap_study_rows.npz (raw values in SOAP internal units, physical-CGS factors, a-exponents) and
../soap_read_timing.json."""
import base64, json, numpy as np
import rc, columns
OUT = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'
STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/'
rows = np.sort(np.load(STUDY + 'selected_halos.npz')['soap_row']).astype(np.int64)
SRC = r'''import os; os.nice(19)
import h5py, json, numpy as np, time, base64
rows=np.frombuffer(base64.b64decode(%r),dtype=np.int64)
cols=%r
base='/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/'
res={'cols':{}, 'timing':{}}
f=h5py.File(base+'halo_properties_0077.hdf5','r')
t_all=time.time()
for c in cols:
    d=f[c]; t0=time.time()
    v=d[rows]
    dt=time.time()-t0
    a=d.attrs
    res['cols'][c]={'data':base64.b64encode(np.ascontiguousarray(v,dtype=np.float64).tobytes()).decode(),
        'cgs':float(np.ravel(a['Conversion factor to physical CGS (including cosmological corrections)'])[0]),
        'a_exp':float(np.ravel(a['a-scale exponent'])[0]),'physical':float(np.ravel(a['Value stored as physical'])[0]),
        'dtype':str(d.dtype),'seconds':dt}
res['timing']['study_rows_total_s']=time.time()-t_all
for c in ('SO/200_crit/ComptonY','SO/200_crit/GasMass'):
    d=f[c]; t0=time.time(); v=d[...]; dt=time.time()-t0
    res['timing']['full_'+c]={'seconds':dt,'n':int(v.size),'logical_bytes':int(v.nbytes),'stored_bytes':int(d.id.get_storage_size()),'nonzero':int((v>0).sum())}
f.close()
g=h5py.File(base+'halo_properties_0037.hdf5','r')
z2={}
for c in ('ExclusiveSphere/10kpc/HalfMassRadiusGas','InclusiveSphere/10kpc/HalfMassRadiusGas'):
    d=g[c]; t0=time.time(); v=d[...]; dt=time.time()-t0
    a=d.attrs
    z2[c]={'seconds':dt,'max':float(v.max()),'p999':float(np.percentile(v[v>0],99.9)),'n_pos':int((v>0).sum()),
           'cgs':float(np.ravel(a['Conversion factor to physical CGS (including cosmological corrections)'])[0]),
           'cgs_nocosmo':float(np.ravel(a['Conversion factor to CGS (not including cosmological corrections)'])[0]),
           'a_exp':float(np.ravel(a['a-scale exponent'])[0]),'physical':float(np.ravel(a['Value stored as physical'])[0])}
z2['scale_factor']=float(np.ravel(g['Header'].attrs['Scale-factor'])[0])
res['z2_aperture_check']=z2
g.close()
print(json.dumps(res))
'''
src = SRC % (base64.b64encode(rows.tobytes()).decode(), columns.COLUMNS)
obj, dt, raw = rc.run(src, timeout=1800)
data = {'soap_row_sorted': rows}
meta = {}
for c, v in obj['cols'].items():
    data[c.replace('/', '__')] = np.frombuffer(base64.b64decode(v['data']), dtype=np.float64)
    meta[c] = {k: v[k] for k in ('cgs', 'a_exp', 'physical', 'dtype', 'seconds')}
np.savez_compressed(OUT + 'soap_study_rows.npz', **data)
timing = {'wall_including_ssh_s': dt, 'columns': meta, 'timing': obj['timing'], 'z2_aperture_check': obj['z2_aperture_check']}
json.dump(timing, open(OUT + 'soap_read_timing.json', 'w'), indent=1)
print(json.dumps(obj['timing'], indent=1)); print(json.dumps(obj['z2_aperture_check'], indent=1)); print('wall', dt)
