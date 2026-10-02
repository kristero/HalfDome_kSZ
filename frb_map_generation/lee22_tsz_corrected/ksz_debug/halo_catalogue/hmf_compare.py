"""HalfDome M200c / M200m halo mass functions per redshift shell vs Tinker08 (HalfDome and WebSky cosmologies), from the
full-catalogue histograms of halfdome_catalogue_pass.py. Writes expected Tinker counts per (logM [Msun], z) cell."""
from common import *
from tinker08 import Tinker
d = np.load(OUTD + 'halfdome_catalogue_pass.npz')
le = d['logm_edges']; ze = d['z_edges']; lmc = 0.5 * (le[1:] + le[:-1]); zcen = 0.5 * (ze[1:] + ze[:-1])
dlog = le[1] - le[0]
res = {}
for cname in ['halfdome', 'websky']:
    t = Tinker(cname, zmax=4.7, nz=48)
    h = t.p['h']
    chi_e = np.array([t.chi(z) for z in ze])            # Mpc/h
    dV = 4 * np.pi / 3 * np.diff(chi_e ** 3)            # (Mpc/h)^3 full sky
    for tag, delta, ref in (('c', 200, 'crit'), ('m', 200, 'mean')):
        NT = np.zeros((len(lmc), len(zcen))); NT_cap = np.zeros_like(NT)
        # catalogue masses are M[Msun/h]/0.68; Tinker in Msun/h -> M_h = M_msun * 0.68 (painter convention) 
        Mh = 10 ** lmc * HPAINT
        for j, z in enumerate(zcen):
            n = t.dndlnm(Mh, z, delta, ref) * np.log(10) * dlog
            NT[:, j] = n * dV[j]
            NT_cap[:, j] = t.dndlnm(Mh, z, delta, ref, zcap=2.5) * np.log(10) * dlog * dV[j]
        res[f'NT_{tag}_{cname}'] = NT; res[f'NTcap_{tag}_{cname}'] = NT_cap
    print('done', cname, flush=True)
np.savez(OUTD + 'tinker08_expected_counts.npz', logm_edges=le, z_edges=ze, **res)
shells = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]
mb = [12.8, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.5]
lines = []
for tag in ['c', 'm']:
    N = d['hist_n_' + tag]
    lines.append(f'\nM200{tag}: HalfDome / Tinker08 counts (HalfDome cosmology) [and / Tinker08 WebSky cosmology]; mass bins log10 M[Msun]')
    lines.append('shell      ' + ' '.join(f'{a:.2f}-{b:.2f}' for a, b in zip(mb[:-1], mb[1:])))
    for z1, z2 in shells:
        jz = (zcen > z1) & (zcen < z2)
        row = []; row2 = []
        for a, b in zip(mb[:-1], mb[1:]):
            im = (lmc > a) & (lmc < b)
            nh = N[np.ix_(im, jz)].sum(); nt = res[f'NT_{tag}_halfdome'][np.ix_(im, jz)].sum(); nw = res[f'NT_{tag}_websky'][np.ix_(im, jz)].sum()
            row.append(f'{nh / nt:5.3f}' if nh > 100 else '  -  '); row2.append(f'{nh / nw:5.3f}' if nh > 100 else '  -  ')
        lines.append(f'{z1:.1f}-{z2:.1f}    ' + '     '.join(row))
        lines.append(f'  [WS cosmo] ' + '     '.join(row2))
txt = '\n'.join(lines); print(txt)
open(OUTD + 'hmf_compare.txt', 'w').write(txt + '\n')
