"""BE step 3: the EXISTING BD pressure fits with a free 2-halo amplitude against pressure from the electron law x a
temperature law (and the hydrostatic model) with the same 2-halo term.

User, 26 Nov 2026: "Have to be compared to best pressure fit (with A_2h free amplitude). Also include 2-halo for
temperature. Use the existing fit for pressure."

Pressure references: existing BD fits evaluated with their SAVED parameters (no refit), paths relative to BD_ROOT.
  B6   constant alpha, A_2h(z) free, amplitude from shells beyond B32 to 6 R200c
       (explore2h/corner_1h_vs_1h2h_20261120/runs/B_2h_rout6.json: law_best, amp_best); the plotting fiducial (user, 20 Nov)
  B4   the same with amplitude shells to 4 R200c (explore2h/final/final_curved075_ivar.json: law_best, amp_best)
  A6   alpha(z) = 0.9394 + 0.2948 z held, amplitude to 6 R200c (explore2h/corner_1h_vs_1h2h_20261120/runs/A_2h_rout6.json)
  A4   the same to 4 R200c (explore2h/final/rec_held_curved.json)
  MB4  smooth mass-break law, 16 law parameters, amplitude to 4 R200c (explore2h/mass_weights/final/best_sbplzz_held.json:
       law, amp)
  Each: model = law + A_2h(z) t,  ln A_2h = a + b lz + c lz^2 (3 parameters),  t = Vikram+17 halo-model template with
  exclusion 0.75 (R200c + R200c'), bd_mass.Data.template('hm_ex0.75').
Electron-law models with the same 2-halo term (be_fit.py definitions), split procedure as BD: the law (and for joint fits
the electron law with the electron loss) from region 0 with the 2-halo term added; the amplitude from region 1 (to
4 R200c) or regions 1 + 2 (to 6 R200c) with weights v = 1/(0.1^2 + e^2); alternated until the largest step < 1e-5.
  joint / sep T7, joint / sep T7x, joint H6 with P_tot = 0 at 6 R200c.
Scores, model m = law + A_2h t, data d = Y200-weighted all-gas stack:
  L_P    = sum_{region 0} w_i ln^2(m_i/d_i), record weights (sum 2701)
  L_out4 = sum_{region 1} v_i ln^2(m_i/d_i);  L_out6 = sum_{regions 1, 2} v_i ln^2(m_i/d_i)
  RMS of ln(m/d) inside B32 by mass group and beyond B32 by radius band (bins with >= 100 haloes),
  A_2h(z), law / halo-component and law / host-bound of the 1-halo part beyond B32, electron loss L_E of the electron law used,
  parameter counts (electron law 9; pressure law 9, 10 or 16; temperature / hydrostatic 6-7; amplitude 3).
Usage: be_compare.py      (env BD_ROOT, BD_CODE, BE_SUMS, BE_OUT as be_fit.py)
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import be_fit as BF

BM = BF.BM
OUT = BF.OUT
BD_ROOT = Path(os.environ['BD_ROOT'])
EXIST = {
    'B6': ('explore2h/corner_1h_vs_1h2h_20261120/runs/B_2h_rout6.json', 'pl', 'const', 6),
    'B4': ('explore2h/final/final_curved075_ivar.json', 'pl', 'const', 4),
    'A6': ('explore2h/corner_1h_vs_1h2h_20261120/runs/A_2h_rout6.json', 'pl', 'held', 6),
    'A4': ('explore2h/final/rec_held_curved.json', 'pl', 'held', 4),
    'MB4': ('explore2h/mass_weights/final/best_sbplzz_held.json', 'sbplzz', 'held', 4),
}


def band_all(B, model, select, band_set):
    D = B.D
    r = np.log(model/D.d)
    out = {}
    for lo, hi in band_set:
        s = select & D.shell & (D.hi > lo) & (D.hi <= hi) & (D.nhalo >= 100)
        if s.sum() >= 3:
            out[f'{lo:g}-{hi:g}'] = dict(rms=float(np.sqrt(np.mean(r[s]**2))), mean=float(np.mean(r[s])), n=int(s.sum()))
    return out


def score(B, one, A, T, Em, counts):
    D = B.D
    full = one+A*T
    r = np.log(full/D.d)
    rE = np.log(Em/B.T['E_all'])
    i0, o4, o6 = D.region == 0, D.region == 1, D.region >= 1
    out = dict(L_P=float(np.sum(B.wP[i0]*r[i0]**2)), L_out4=float(np.sum(D.v[o4]*r[o4]**2)), L_out6=float(np.sum(D.v[o6]*r[o6]**2)),
               L_E=float(np.sum(B.wE[i0]*rE[i0]**2)),
               rms_inside_all=float(np.sqrt(np.mean(r[i0 & (D.nhalo >= 100)]**2))),
               bands_inside=BF.bands(B, full, D.d, i0, BF.BANDS_IN), bands_outer=BF.bands(B, full, D.d, o6, BF.BANDS_OUT),
               outer_all_masses=band_all(B, full, o6, BF.BANDS_OUT), inside_all_masses=band_all(B, full, i0, BF.BANDS_IN),
               amp_by_z={f'{z:g}': float(np.median(A[D.zi == k])) for k, z in enumerate(D.zs)},
               extension_one_halo=BF.extension(B, one), counts=counts)
    out['counts']['total_with_electrons'] = int(sum(counts.values()))
    return out, full


def main():
    t0 = time.time()
    B = BF.BEData()
    D = B.D
    F = BF.Fitter(B)
    T = D.template('hm_ex0.75')
    na, sa, la, ua, amp = BM.amp_model(D, 'A2zq')
    la, ua = np.array(la, float), np.array(ua, float)
    fits = json.loads((OUT/'be_fits.json').read_text())['results']
    fits.update({k: v for k, v in json.loads((OUT/'be_fits2.json').read_text())['results'].items() if 'params' in v})
    pE = np.array(fits['electron_law']['params'])
    E_alone = BF.Composite(B, 'T5').electrons(pE)
    res, arrays = {}, dict(template=T)
    # ---------- existing pressure fits, saved parameters
    for key, (path, mass, alpha, rout) in EXIST.items():
        d = json.loads((BD_ROOT/path).read_text())
        p = np.array(d['law_best'] if 'law_best' in d else d['law'], float)
        pa = np.array(d['amp_best'] if 'amp_best' in d else d['amp'], float)
        law = BM.Law(D, mass, alpha, 'fixed')
        assert len(p) == len(law.names), (key, len(p), law.names)
        one = law(p)
        A = amp(pa)
        sc, full = score(B, one, A, T, E_alone, dict(electron_law=9, pressure_law=len(p), amplitude=3))
        saved = d['L_in'] if 'L_in' in d else d['losses']['best']
        sc.update(source=path, saved_L_in=float(saved), recomputed_minus_saved=float(sc['L_P']-saved), rout=rout,
                  law_names=list(law.names), law=p.tolist(), amp=pa.tolist(), physicality=BF.physicality(BF.point_fns(B, 'plaw', law, p), B))
        res[f'existing_{key}'] = sc
        arrays[f'one__existing_{key}'], arrays[f'full__existing_{key}'] = one, full
        print(key, sc['L_P'], 'saved', saved, f'{time.time()-t0:.0f}s', flush=True)
    # ---------- electron-law models + 2-halo
    i0 = B.inner
    starts = {}
    comp, p, Ls = F.fit_comp('T7x', 'sep', i0, i0, pE_fixed=pE, nstart=6)
    starts['sep_T7x'] = p
    res['onehalo_sep_T7x'] = dict(L_P=Ls[0], params=p.tolist(), start_losses=Ls)
    comp, p, Ls = F.fit_comp('T7x', 'joint', i0, i0, pE_fixed=pE, start=p, nstart=3)
    starts['joint_T7x'] = p
    res['onehalo_joint_T7x'] = dict(L_P_plus_L_E=Ls[0], params=p.tolist(), start_losses=Ls,
                                    L_P=float(np.sum(B.wP[i0]*np.log(comp(p)/D.d)[i0]**2)))
    print('T7x 1-halo', res['onehalo_sep_T7x']['L_P'], res['onehalo_joint_T7x']['L_P'], f'{time.time()-t0:.0f}s', flush=True)
    for name in ('sep_T7', 'joint_T7', 'joint_H6_xout6'):
        starts[name] = np.array(fits[name]['params'])
    models = [('joint_T7x', 'T7x', 'joint', 4.), ('sep_T7x', 'T7x', 'sep', 4.), ('joint_T7', 'T7', 'joint', 4.), ('sep_T7', 'T7', 'sep', 4.),
              ('joint_H6_xout6', 'H6', 'joint', 6.)]
    for name, kind, mode, xout in models:
        for rout in (6, 4):
            arows = np.flatnonzero((D.region == 1 if rout == 4 else D.region >= 1) & (D.d > 0))
            sv = np.sqrt(D.v[arows])
            comp = BF.Composite(B, kind, xout)
            p = starts[name].copy()
            pa = np.array(sa, float)
            for it in range(15):
                one = comp(p)
                oa = BF.lsq(lambda e: sv*(np.log(np.maximum(one[arows]+amp(e)[arows]*T[arows], 1e-300))-D.lnd[arows]), pa, la, ua)
                two = amp(oa.x)*T
                _, pn, _ = F.fit_comp(kind, mode, i0, i0, pE_fixed=pE if mode == 'sep' else p[:9], two=two, start=p, xout=xout, nstart=1)
                step = max(np.max(abs(pn-p)), np.max(abs(oa.x-pa)))
                p, pa = pn, oa.x
                if step < 1e-5:
                    break
            one = comp(p)
            A = amp(pa)
            Em = comp.electrons(p[:9]) if mode == 'joint' else E_alone
            sc, full = score(B, one, A, T, Em, dict(electron_law=9, temperature_or_hydrostatic=len(comp.tn), amplitude=3))
            key = f'{name}_rout{rout}'
            sc.update(kind=kind, mode=mode, xout=xout, rout=rout, rounds=it+1, names=list(comp.names), params=p.tolist(), amp=pa.tolist(),
                      physicality=BF.physicality(BF.point_fns(B, 'comp', comp, p), B), Y200=BF.y200_points(B, BF.point_fns(B, 'comp', comp, p)))
            res[key] = sc
            arrays[f'one__{key}'], arrays[f'full__{key}'], arrays[f'E__{key}'] = one, full, Em
            print(key, 'L_P', round(sc['L_P'], 3), 'L_out6', round(sc['L_out6'], 1), 'A0', round(sc['amp_by_z']['0'], 2), f'{time.time()-t0:.0f}s', flush=True)
    (OUT/'be_compare.json').write_text(json.dumps(dict(task='compare', checks=B.check, results=res, seconds=time.time()-t0), indent=1)+'\n')
    np.savez_compressed(OUT/'be_compare_rows.npz', **arrays)


if __name__ == '__main__':
    main()
