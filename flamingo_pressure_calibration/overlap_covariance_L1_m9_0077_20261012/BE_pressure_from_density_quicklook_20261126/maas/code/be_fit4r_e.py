"""BE step 5: pressure AND electrons fitted to 4 R200c, each with its own 2-halo term (user, 26 Nov 2026: "Yes, fit
electrons" - the electron tail beyond B32 is then constrained by electron data; this changes decision D4 for fitting only).

Rows (bd_mass.Data): regions 0 (B32 residuals) and 1 (shells beyond B32 to 4 R200c) are fitted; region 2 (4-6 R200c) is a
prediction check. Data: Y200-weighted all-gas stacks d^P (pressure) and d^E (electrons; be_sums.npz).
Models:  P_i = P1h_i + A_P(z) t^P_i,   E_i = E1h_i + A_E(z) t^E_i,   ln A_F = a_F + b_F lz + c_F lz^2 (3 parameters each);
  t^P = BD Vikram+17 pressure template, exclusion 0.75 (bd_mass.Data.template('hm_ex0.75'));
  t^E = the same halo-model form for electrons with Battaglia+16 neighbours (be_twohalo_e.py, be_etemplate.py), exclusion
        0.75, without ('ex0p75') or with ('ex0p75+mean') the uncorrelated mean term; the better one (lower L_E,4R of the
        electron law alone) is used for every model.
  P1h, E1h: the 9-parameter electron law E1h and the pressure law, or P1h = E1h x temperature law (T7x, T9x) or hydrostatic
        (H6, x_out = 6) from be_fit.py.
Weights (parity rule, user decision 26 Nov): pressure: region 0 = record weights w (sum 2701), region 1 =
  (W0/N0) v / <v>_region1, v = 1/(0.1^2 + e_P^2); electrons: region 0 = w^E (be_fit.py, sum 2701), region 1 =
  (W0/N0) v_E / <v_E>_region1, v_E = 1/(0.1^2 + e_E^2); e_F = relative fold error of the all-gas stack of field F.
Losses: L_4R,P = sum_{0,1} w ln^2(P/d^P);  L_4R,E = sum_{0,1} w^E ln^2(E/d^E);  total = L_4R,P + L_4R,E.
Fits (least squares, all parameters of a model together, several starts):
  elaw_2h_<t>   electron law + A_E alone (both templates t)
  split_const / split_massbreak   = elaw_2h (best template) + the existing parity 4 R200c pressure fits (be_fit4r_parity.json),
                pressure law + A_P fitted alone; parameters 9 + 3 + 10 (16) + 3
  sep_T7x       electron law + A_E from elaw_2h held fixed; T7x + A_P from L_4R,P
  joint_<T7x|T9x|H6>   electron law + temperature (hydrostatic) law + A_P + A_E from L_4R,P + L_4R,E
Scores: both 4 R200c losses and their sum, the inside-B32 losses with the step-2 weights (L_P, L_E), RMS of ln(model/data)
inside B32 by mass group and beyond B32 by radius band for both fields (4-6 R200c = not fitted), A_P(z), A_E(z), counts.
Usage: be_fit4r_e.py   ->  out/be_fit4r_e.json, out/be_fit4r_e_rows.npz
"""
import json
import time

import numpy as np

import be_fit as BF

BM = BF.BM
OUT = BF.OUT


def band_all(D, model, data, select, band_set):
    r = np.log(model/data)
    out = {}
    for lo, hi in band_set:
        s = select & D.shell & (D.hi > lo) & (D.hi <= hi) & (D.nhalo >= 100)
        if s.sum() >= 3:
            out[f'{lo:g}-{hi:g}'] = dict(rms=float(np.sqrt(np.mean(r[s]**2))), mean=float(np.mean(r[s])), n=int(s.sum()))
    return out


def main():
    t0 = time.time()
    B = BF.BEData()
    D = B.D
    TP = D.template('hm_ex0.75')
    TE = {t: np.load(OUT/f'etemplate_{t}.npy') for t in ('ex0p75', 'ex0p75+mean')}
    _, sa, la, ua, amp = BM.amp_model(D, 'A2zq')
    la, ua = np.array(la, float), np.array(ua, float)
    fits = json.loads((OUT/'be_fits.json').read_text())['results']
    pE9 = np.array(fits['electron_law']['params'])
    P4 = json.loads((OUT/'be_fit4r_parity.json').read_text())['results']
    P4f = json.loads((OUT/'be_fit4r_parity_flex.json').read_text())['results']
    r0, r1 = D.region == 0, D.region == 1
    rows = np.flatnonzero(D.region <= 1)
    N0 = int(r0.sum())
    wP = D.w_best.copy()
    wP[r1] = wP[r0].sum()/N0*D.v[r1]/D.v[r1].mean()
    eE = B.T['E_err']/B.T['E_all']
    vE = 1/(.1**2+eE**2)
    wE = B.wE.copy()
    wE[r1] = wE[r0].sum()/N0*vE[r1]/vE[r1].mean()
    swP, swE = np.sqrt(wP[rows]), np.sqrt(wE[rows])
    dE, lndE = B.T['E_all'], B.lndE
    rng = np.random.default_rng(4)
    res = dict(weights=dict(P_region0=float(wP[r0].sum()), P_region1=float(wP[r1].sum()), E_region0=float(wE[r0].sum()),
                            E_region1=float(wE[r1].sum()), n_region1=int(r1.sum())))
    arrays = dict(template_P=TP, **{f'template_E_{t}': v for t, v in TE.items()})
    print('weights', res['weights'], flush=True)
    E5 = BF.Composite(B, 'T5')

    def score(name, oneP, aP, oneE, aE, tE, counts, extra):
        AP, AE = amp(aP), amp(aE)
        fP, fE = oneP+AP*TP, oneE+AE*tE
        rP, rE = np.log(fP/D.d), np.log(fE/dE)
        sc = dict(L_4R_P=float(np.sum(wP[rows]*rP[rows]**2)), L_4R_E=float(np.sum(wE[rows]*rE[rows]**2)),
                  L_P=float(np.sum(B.wP[r0]*rP[r0]**2)), L_E=float(np.sum(B.wE[r0]*rE[r0]**2)),
                  L_out6=float(np.sum(D.v[D.region >= 1]*rP[D.region >= 1]**2)), L_out6_E=float(np.sum(vE[D.region >= 1]*rE[D.region >= 1]**2)),
                  bands_inside=BF.bands(B, fP, D.d, r0, BF.BANDS_IN), bands_inside_E=BF.bands(B, fE, dE, r0, BF.BANDS_IN),
                  outer_all_masses=band_all(D, fP, D.d, D.region >= 1, BF.BANDS_OUT),
                  outer_all_masses_E=band_all(D, fE, dE, D.region >= 1, BF.BANDS_OUT),
                  amp_by_z={f'{z:g}': float(np.median(AP[D.zi == k])) for k, z in enumerate(D.zs)},
                  amp_by_z_E={f'{z:g}': float(np.median(AE[D.zi == k])) for k, z in enumerate(D.zs)},
                  amp=list(map(float, aP)), amp_E=list(map(float, aE)), counts=counts, **extra)
        sc['L_4R_total'] = sc['L_4R_P']+sc['L_4R_E']
        sc['counts']['total'] = int(sum(counts.values()))
        res[name] = sc
        arrays[f'full__{name}'], arrays[f'one__{name}'], arrays[f'fullE__{name}'], arrays[f'oneE__{name}'] = fP, oneP, fE, oneE
        print(name, 'L4R_P', round(sc['L_4R_P'], 3), 'L4R_E', round(sc['L_4R_E'], 3), 'total', round(sc['L_4R_total'], 3),
              'L_E(in)', round(sc['L_E'], 3), 'A_E(0)', round(sc['amp_by_z_E']['0'], 3), f'{time.time()-t0:.0f}s', flush=True)
        return sc

    def solve(resid, starts, LO, HI):
        out = []
        for s in starts:
            try:
                o = BF.lsq(resid, np.clip(s, LO+1e-7, HI-1e-7), LO, HI, nfev=4000)
                out.append((2*o.cost, o.x))
            except Exception:                                      # noqa: BLE001
                out.append((np.inf, np.asarray(s)))
        out.sort(key=lambda t: t[0])
        return out[0][1], [float(o[0]) for o in out]

    # ---------- (1) electron law + its 2-halo term alone
    best = None
    for t, tE in TE.items():
        E1 = E5.electrons(pE9)
        oa = BF.lsq(lambda e: swE*(np.log(np.maximum(E1[rows]+amp(e)[rows]*tE[rows], 1e-300))-lndE[rows]), np.array(sa, float), la, ua)
        LO, HI = np.r_[BF.ELaw.lo, la], np.r_[BF.ELaw.hi, ua]
        s0 = np.r_[pE9, oa.x]
        starts = [s0]+[s0+np.r_[.1*rng.standard_normal(9), np.zeros(3)] for _ in range(4)]

        def resid(q, tE=tE):
            return BF.safe_log_res(E5.electrons(q[:9])[rows]+amp(q[9:])[rows]*tE[rows], lndE[rows], swE)
        q, Ls = solve(resid, starts, LO, HI)
        oneE = E5.electrons(q[:9])
        LE4 = float(np.sum(wE[rows]*np.log((oneE+amp(q[9:])*tE)/dE)[rows]**2))
        res[f'elaw_2h_{t}'] = dict(params=q[:9].tolist(), amp_E=q[9:].tolist(), L_4R_E=LE4, start_losses=Ls,
                                   L_E=float(np.sum(B.wE[r0]*np.log((oneE+amp(q[9:])*tE)/dE)[r0]**2)),
                                   outer_all_masses_E=band_all(D, oneE+amp(q[9:])*tE, dE, D.region >= 1, BF.BANDS_OUT),
                                   bands_inside_E=BF.bands(B, oneE+amp(q[9:])*tE, dE, r0, BF.BANDS_IN),
                                   amp_by_z_E={f'{z:g}': float(np.median(amp(q[9:])[D.zi == k])) for k, z in enumerate(D.zs)},
                                   physicality_E=BF.physicality(BF.e_point_fn(B, q[:9]), B))
        arrays[f'oneE__elaw_2h_{t}'], arrays[f'fullE__elaw_2h_{t}'] = oneE, oneE+amp(q[9:])*tE
        print('electron law + 2-halo', t, LE4, Ls[:3], f'{time.time()-t0:.0f}s', flush=True)
        if best is None or LE4 < best[0]:
            best = (LE4, t, q)
    _, tbest, qE = best
    tE = TE[tbest]
    res['electron_template_used'] = tbest
    pEb, aEb = qE[:9], qE[9:]
    oneE_b = E5.electrons(pEb)
    # ---------- (2) split: existing parity 4 R200c pressure fits + the electron law with its 2-halo term
    for name, src, mass, alpha in (('split_const', 'plaw_const_4R', 'pl', 'const'), ('split_massbreak', 'plaw_massbreak_4R', 'sbplzz', 'held')):
        law = BM.Law(D, mass, alpha, 'fixed')
        pl = np.array(P4[src]['law'])
        score(name, law(pl), np.array(P4[src]['amp']), oneE_b, aEb, tE, dict(electron_law=9, amplitude_E=3, pressure_law=len(pl), amplitude_P=3),
              dict(source=f'be_fit4r_parity.json:{src} + elaw_2h_{tbest}', law=pl.tolist(), law_E=pEb.tolist()))
    # ---------- (3) temperature law with the electron law (+ 2-halo) held fixed
    comp = BF.Composite(B, 'T7x', 4.)
    src = P4['sep_T7x_4R']
    s0 = np.r_[src['law'], src['amp']]
    LO, HI = np.r_[comp.lo[9:], la], np.r_[comp.hi[9:], ua]

    def resid_sep(q):
        return BF.safe_log_res(comp(np.r_[pEb, q[:-3]])[rows]+amp(q[-3:])[rows]*TP[rows], D.lnd[rows], swP)
    q, Ls = solve(resid_sep, [s0]+[s0+np.r_[.15*rng.standard_normal(7), np.zeros(3)] for _ in range(3)], LO, HI)
    score('sep_T7x', comp(np.r_[pEb, q[:-3]]), q[-3:], oneE_b, aEb, tE, dict(electron_law=9, temperature=7, amplitude_P=3, amplitude_E=3),
          dict(law=np.r_[pEb, q[:-3]].tolist(), names=comp.names, start_losses=Ls))
    # ---------- (4) joint: electron law + temperature (hydrostatic) law + both amplitudes
    for name, kind, xout, src in (('joint_T7x', 'T7x', 4., P4['joint_T7x_4R']), ('joint_T9x', 'T9x', 4., P4f['joint_T9x_4R']),
                                  ('joint_H6', 'H6', 6., P4['joint_H6_xout6_4R'])):
        comp = BF.Composite(B, kind, xout)
        k = len(comp.tn)
        lawp = np.array(src['law'])
        s_a = np.r_[pEb, lawp[9:], src['amp'], aEb]                  # electron law from the electron-only 4 R200c fit
        s_b = np.r_[lawp, src['amp'], aEb]                           # electron law from the pressure-driven joint fit
        LO, HI = np.r_[comp.lo, la, la], np.r_[comp.hi, ua, ua]
        starts = [s_a, s_b]+[s_a+np.r_[np.zeros(9), .15*rng.standard_normal(k), np.zeros(6)] for _ in range(2)]

        def resid_joint(q, comp=comp, k=k):
            pl, aP, aE = q[:9+k], q[9+k:12+k], q[12+k:]
            return np.r_[BF.safe_log_res(comp(pl)[rows]+amp(aP)[rows]*TP[rows], D.lnd[rows], swP),
                         BF.safe_log_res(comp.electrons(pl[:9])[rows]+amp(aE)[rows]*tE[rows], lndE[rows], swE)]
        q, Ls = solve(resid_joint, starts, LO, HI)
        pl = q[:9+k]
        sc = score(name, comp(pl), q[9+k:12+k], comp.electrons(pl[:9]), q[12+k:], tE, dict(electron_law=9, temperature_or_hydrostatic=k,
                   amplitude_P=3, amplitude_E=3), dict(law=pl.tolist(), names=comp.names, start_losses=Ls,
                   at_bounds=[n for n, v, a, b in zip(comp.names, pl, comp.lo, comp.hi) if min(v-a, b-v) < 1e-4*(b-a)]))
        sc['physicality'] = BF.physicality(BF.point_fns(B, 'comp', comp, pl), B)
        sc['physicality_E'] = BF.physicality(BF.e_point_fn(B, pl[:9]), B)
    (OUT/'be_fit4r_e.json').write_text(json.dumps(dict(task='fit4r_e', checks=B.check, results=res, seconds=time.time()-t0), indent=1)+'\n')
    np.savez_compressed(OUT/'be_fit4r_e_rows.npz', **arrays)


if __name__ == '__main__':
    main()
