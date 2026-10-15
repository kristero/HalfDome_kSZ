"""Summarise inventory_raw_<tag>.json into a compact gas-dataset table (local)."""
import json, re, sys
OUT = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'
GAS = re.compile(r'Gas|ComptonY|XRay|Xray|Electron|Temperature|DopplerB|SORadius|SpectroscopicLike|ThermalEnergy|StarFormationRate|NumberOfGasParticles|TotalMass|MassFraction(External|Satellites)')


def units(a):
    ex = {k: a.get(f'U_{k} exponent', [0])[0] for k in 'MLTt'}
    s = ' '.join(f'{n}^{e:g}' for n, e in zip(('M', 'L', 'T', 't'), (ex['M'], ex['L'], ex['T'], ex['t'])) if e)
    return s or '1'


def family(group):
    parts = group.split('/')
    if parts[0] == 'ProjectedAperture':
        return 'ProjectedAperture/*/proj?'
    if parts[0] in ('SO', 'InclusiveSphere', 'ExclusiveSphere'):
        return parts[0] + '/*'
    return group


def summarise(tag):
    d = json.load(open(OUT + f'inventory_raw_{tag}.json'))
    ds = d['datasets']
    groups = {}
    for k, v in ds.items():
        g, n = k.rsplit('/', 1) if '/' in k else ('', k)
        groups.setdefault(g, {})[n] = v
    fam = {}
    for g, items in groups.items():
        f = family(g)
        for n, v in items.items():
            if not GAS.search(n):
                continue
            a = v['attrs']
            e = fam.setdefault(f, {}).setdefault(n, {'groups': [], 'dtype': v['dtype'], 'shape1': v['shape'][1:],
                                                       'units': units(a), 'a_exp': a.get('a-scale exponent', [None])[0],
                                                       'cgs': a.get('Conversion factor to physical CGS (including cosmological corrections)', [None])[0],
                                                       'description': a.get('Description', ''),
                                                       'lossy': a.get('Lossy compression filter', ''),
                                                       'stored_bytes': []})
            e['groups'].append(g.split('/', 1)[-1] if '/' in g else g)
            e['stored_bytes'].append(v['storage_bytes'])
    apertures = {k: sorted(set(x.rsplit('/', 1)[0] for x in ds if x.startswith(k + '/'))) for k in ('SO', 'InclusiveSphere', 'ExclusiveSphere', 'ProjectedAperture')}
    nrows = ds['InputHalos/HaloCatalogueIndex']['shape'][0]
    chunks = {k: v['chunks'] for k, v in ds.items() if k.endswith('/ComptonY') or k.endswith('/GasMass')}
    res = {'tag': tag, 'path': d['path'], 'file_bytes': d['file_bytes'], 'n_rows': nrows, 'n_datasets': len(ds),
           'profile_like_datasets': [k for k in ds if re.search(r'[Pp]rofile|[Rr]adial|[Bb]ins?$|Shell', k)],
           'top_groups': sorted(set(k.split('/')[0] for k in list(ds) + list(d['groups']))),
           'apertures': apertures, 'gas_datasets_by_family': fam, 'chunks_example': dict(list(chunks.items())[:6]),
           'meta': d['meta'], 'inventory_remote_seconds': d['elapsed']}
    json.dump(res, open(OUT + f'inventory_gas_{tag}.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    r = summarise(sys.argv[1])
    print(r['file_bytes'], r['n_rows'], r['n_datasets'], r['profile_like_datasets'])
    print(r['top_groups'])
    print(r['chunks_example'])
    for f, items in r['gas_datasets_by_family'].items():
        print('##', f)
        for n, e in sorted(items.items()):
            sb = e['stored_bytes']
            print(f"  {n:60s} {e['dtype']:8s} {str(e['shape1']):6s} [{e['units']}] a^{e['a_exp']} cgs={e['cgs']:.4g} n_ap={len(e['groups'])} lossy={e['lossy']} MB/col={sum(sb)/len(sb)/1e6:.1f} | {e['description'][:90]}")
