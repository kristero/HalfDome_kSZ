"""BE: electron 2-halo templates on every row of the BD residual set (regions 0, 1, 2), exclusion factor 0.75, without
and with the uncorrelated mean term (be_twohalo_e.py). Check first: field='P' must reproduce the BD pressure template
hm_ex0.75 (bd_mass.Data.template) on a sample of rows.
Output: out/etemplate_ex0p75.npy, out/etemplate_ex0p75+mean.npy, out/etemplate_check.json.
"""
import json
import time

import numpy as np

import be_fit as BF
import be_twohalo_e as TE

BM = BF.BM
R1 = BM.R1


def main():
    t0 = time.time()
    B = BF.BEData()
    D = B.D
    rng_m = R1.MASS_RANGE['flamingo']
    # ---- check against the BD pressure template
    TP = D.template('hm_ex0.75')
    tp = TE.TwoHaloField('flamingo', rng_m, field='P')
    rng = np.random.default_rng(1)
    sample = np.sort(rng.choice(len(D.d), 60, replace=False))
    vals = np.array([tp.template_excl(D.M[i], D.z[i], D.lo[[i]], D.hi[[i]], np.array([not D.shell[i]]), rex_factor=.75)[0] for i in sample])
    check = dict(rows=sample.tolist(), max_rel_diff_vs_BD_pressure_template=float(np.max(abs(vals/TP[sample]-1))))
    print('pressure template check', check['max_rel_diff_vs_BD_pressure_template'], f'{time.time()-t0:.0f}s', flush=True)
    if check['max_rel_diff_vs_BD_pressure_template'] > 1e-8:
        raise SystemExit('reimplemented template does not reproduce the BD pressure template')
    te = TE.TwoHaloField('flamingo', rng_m, field='E')
    for mean, tag in ((False, 'ex0p75'), (True, 'ex0p75+mean')):
        t = np.zeros(len(D.d))
        for b in np.unique(D.block):
            rows = np.flatnonzero(D.block == b)
            t[rows] = te.template_excl(D.M[rows[0]], D.z[rows[0]], D.lo[rows], D.hi[rows], ~D.shell[rows], rex_factor=.75, mean_term=mean)
        np.save(BF.OUT/f'etemplate_{tag}.npy', t)
        r1 = D.region >= 1
        check[tag] = dict(median_template_over_data_beyond_B32=float(np.median(t[r1]/B.T['E_all'][r1])),
                          median_template_over_data_inside=float(np.median(t[D.region == 0]/B.T['E_all'][D.region == 0])), seconds=time.time()-t0)
        print(tag, check[tag], flush=True)
    (BF.OUT/'etemplate_check.json').write_text(json.dumps(check, indent=1)+'\n')


if __name__ == '__main__':
    main()
