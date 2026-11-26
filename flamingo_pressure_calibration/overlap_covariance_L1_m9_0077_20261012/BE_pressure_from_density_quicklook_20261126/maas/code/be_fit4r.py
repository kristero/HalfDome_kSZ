"""BE step 4: 1-halo law + 2-halo term fitted in ONE loss out to 4 R200c (pressure laws and electron-law models).

User, 26 Nov 2026: "But with 2-halo you can fit further than B32, it should around 4r200c." This replaces, for the
1-halo + 2-halo fits, the split procedure of step 3 (law inside B32, amplitude beyond it) and the 16 Nov rule of judging
fits inside B32. Electrons still stop at B32 (user decision D4, 12 Nov): the electron loss stays on the B32 set.

Rows (bd_mass.Data): region 0 = the 1,494 B32 residuals; region 1 = the 1,025 shells of the 4 R200c set lying entirely
beyond B32; region 2 = the 4-6 R200c cells (NOT fitted here: prediction check).
Model of every row: m_i = law_i + A_2h(z_i) t_i, ln A_2h = a + b lz + c lz^2, t = Vikram+17 template, exclusion 0.75
(bd_mass.Data.template('hm_ex0.75')), law = shell mean (or integral) of the 1-halo law.
Loss L_4R = sum_{regions 0, 1} w_i ln^2(m_i/d_i),  w = bd_mass.Data.w_best: the record weights of the B32 set in region 0
and the record weight formula of the 4 R200c set in region 1 (same B32-set constants W0, S_ref, S_sens; bd_export_flamingo).
Joint electron-law models add L_E = sum_{region 0} w^E_i ln^2(E_i/d^E_i) (be_fit.py). All law and amplitude parameters
are fitted together by least squares (several starts: the step-3 split solution and perturbations).
Models: pressure law with one constant alpha (10 + 3), alpha(z) held (9 + 3), smooth mass break (16 + 3; bd_mass 'sbplzz');
electron law x temperature law T7x / T7 (joint and separate), hydrostatic + core term H6 (x_out = 6), all + 3 amplitude.
Scores (be_compare.score): L_P inside B32, L_out4 / L_out6 with the outer weights v (comparable with step 3), the region-1
part of L_4R, RMS by mass group inside B32 and by radius band beyond B32 (4-6 R200c = not fitted), A_2h(z), electron loss,
1-halo extension ratios, physicality, parameter counts; plus the weight share of region 1 in L_4R.
Outer weights (option --outer): 'record' = the record formula above (region 1 then carries 45 of 2,746 = 1.6 % of the
weight); 'parity' = every region-1 shell gets the mean region-0 weight times its fold-error shape,
w_i = (W0/N0) v_i / <v>_region1, v = 1/(0.1^2 + e^2), W0 = 2701, N0 = 1,494 (region 1 then carries about 41 %).
--set flex (after the user chose parity weights): joint/sep fits of the temperature laws with a mass- and/or
redshift-dependent outer decline radius (T8xm, T8xz, T9x; be_fit.py), started from the parity T7x fit.
--set eflex: the electron law with one more free constant (x_c, or the inner slope; be_fit.ELaw), jointly with T7x / T8xm.
Usage: be_fit4r.py [--outer record|parity] [--set main|flex|eflex]   (env as be_fit.py)   ->  out/be_fit4r[_parity].json, out/be_fit4r[_parity]_rows.npz
"""
import argparse
import json
import time

import numpy as np

import be_compare as BC
import be_fit as BF

BM = BF.BM
OUT = BF.OUT
ELAW_START = {'c': np.log(.5), 'g': -.2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outer', default='record', choices=('record', 'parity'))
    ap.add_argument('--set', default='main', choices=('main', 'flex', 'eflex'))
    args = ap.parse_args()
    outer = args.outer
    suffix = ('' if outer == 'record' else '_parity')+{'main': '', 'flex': '_flex', 'eflex': '_eflex'}[args.set]
    t0 = time.time()
    B = BF.BEData()
    D = B.D
    F = BF.Fitter(B)
    T = D.template('hm_ex0.75')
    _, sa, la, ua, amp = BM.amp_model(D, 'A2zq')
    la, ua = np.array(la, float), np.array(ua, float)
    C3 = json.loads((OUT/'be_compare.json').read_text())['results']
    fits = json.loads((OUT/'be_fits.json').read_text())['results']
    pE = np.array(fits['electron_law']['params'])
    E_alone = BF.Composite(B, 'T5').electrons(pE)
    rows = np.flatnonzero(D.region <= 1)
    w = D.w_best.copy()
    if outer == 'parity':
        r1 = D.region == 1
        w[r1] = w[D.region == 0].sum()/np.sum(D.region == 0)*D.v[r1]/D.v[r1].mean()
    sw = np.sqrt(w[rows])
    i0 = B.inner
    swE = np.sqrt(B.wE[i0])
    share = dict(outer_weights=outer, sum_w_region0=float(w[D.region == 0].sum()), sum_w_region1=float(w[D.region == 1].sum()),
                 n_region1=int((D.region == 1).sum()))
    rng = np.random.default_rng(3)
    res, arrays = dict(weight_share=share), dict(template=T)
    print('weights', share, flush=True)

    def run(name, law_eval, e_eval, lo, hi, starts, counts, kind, law_obj, ptype):
        def resid(q):
            pl, pa = q[:-3], q[-3:]
            m = law_eval(pl)+amp(pa)*T
            r = BF.safe_log_res(m[rows], D.lnd[rows], sw)
            if e_eval is not None:
                r = np.r_[r, BF.safe_log_res(e_eval(pl)[i0], B.lndE[i0], swE)]
            return r
        LO, HI = np.r_[lo, la], np.r_[hi, ua]
        out = []
        for s in starts:
            try:
                o = BF.lsq(resid, s, LO, HI, nfev=4000)
                out.append((2*o.cost, o.x))
            except Exception:                                      # noqa: BLE001
                out.append((np.inf, np.asarray(s)))
        out.sort(key=lambda t: t[0])
        q = out[0][1]
        pl, pa = q[:-3], q[-3:]
        one = law_eval(pl)
        A = amp(pa)
        Em = e_eval(pl) if e_eval is not None else E_alone
        sc, full = BC.score(B, one, A, T, Em, counts)
        r = np.log(full/D.d)
        sc.update(L_4R=float(np.sum(w[rows]*r[rows]**2)), L_4R_region1=float(np.sum(w[D.region == 1]*r[D.region == 1]**2)),
                  start_losses=[float(o[0]) for o in out], law=pl.tolist(), amp=pa.tolist(), kind=kind,
                  at_bounds=[n for n, v, a, b in zip(law_obj.names, pl, lo, hi) if min(v-a, b-v) < 1e-4*(b-a)])
        sc['names'] = list(law_obj.names)
        sc['physicality'] = BF.physicality(BF.point_fns(B, ptype, law_obj, pl), B)
        sc['Y200'] = BF.y200_points(B, BF.point_fns(B, ptype, law_obj, pl))
        res[name] = sc
        arrays[f'one__{name}'], arrays[f'full__{name}'], arrays[f'E__{name}'] = one, full, Em
        print(name, 'L_4R', round(sc['L_4R'], 3), 'L_P', round(sc['L_P'], 3), 'L_out6', round(sc['L_out6'], 1), 'A0', round(sc['amp_by_z']['0'], 2),
              'bounds', sc['at_bounds'], f'{time.time()-t0:.0f}s', flush=True)

    if args.set == 'eflex':
        # electron law with one more free constant (x_c or inner slope; be_fit.ELaw ev), joint with the temperature law, from the
        # parity T7x fit; plus the electron law alone with the extra constant (inside B32, electron loss only)
        base = json.loads((OUT/'be_fit4r_parity.json').read_text())['results']['joint_T7x_4R']
        law0 = np.array(base['law'])
        for ev in ('c', 'g'):
            extra = ELAW_START[ev]
            comp_e = BF.Composite(B, 'T5', 4., ev)
            sE = np.r_[pE, extra]
            o = BF.lsq(lambda q, c=comp_e: BF.safe_log_res(c.electrons(q, i0), B.lndE[i0], swE), sE, comp_e.lo[:comp_e.ne], comp_e.hi[:comp_e.ne])
            res[f'electron_law_{ev}'] = dict(names=comp_e.names[:comp_e.ne], params=o.x.tolist(), L_E=float(2*o.cost),
                                             L_E_9par=float(np.sum(B.wE[i0]*np.log(E_alone[i0]/B.T['E_all'][i0])**2)))
            print(f'electron law + {ev}', res[f'electron_law_{ev}']['L_E'], flush=True)
            for kind in ('T7x', 'T8xm'):
                comp = BF.Composite(B, kind, 4., ev)
                nx = len(comp.tn)-7
                s0 = np.r_[law0[:9], extra, law0[9:], np.zeros(nx), base['amp']]
                LO, HI = np.r_[comp.lo, la]+1e-6, np.r_[comp.hi, ua]-1e-6
                pert = lambda nT=len(comp.tn), ne=comp.ne: np.r_[np.zeros(ne), .15*rng.standard_normal(nT), np.zeros(3)]  # noqa: E731
                starts = [np.clip(s0, LO, HI)]+[np.clip(s0+pert(), LO, HI) for _ in range(3)]
                run(f'joint_{kind}_e{ev}_4R', comp, (lambda pl, comp=comp: comp.electrons(pl[:comp.ne])), comp.lo, comp.hi, starts,
                    dict(electron_law=comp.ne, temperature_or_hydrostatic=len(comp.tn), amplitude=3), kind, comp, 'comp')
        (OUT/f'be_fit4r{suffix}.json').write_text(json.dumps(dict(task='fit4r', outer=outer, set=args.set, checks=B.check, results=res,
                                                                   seconds=time.time()-t0), indent=1)+'\n')
        np.savez_compressed(OUT/f'be_fit4r{suffix}_rows.npz', **arrays)
        return
    if args.set == 'flex':
        # flexible outer temperature decline (added 26 Nov, after the user chose parity weights); starts from the parity T7x fit
        base = json.loads((OUT/'be_fit4r_parity.json').read_text())['results']
        for name, kind, mode in (('joint_T7x_4R', 'T7x', 'joint'), ('joint_T8xm_4R', 'T8xm', 'joint'), ('joint_T8xz_4R', 'T8xz', 'joint'),
                                 ('joint_T9x_4R', 'T9x', 'joint'), ('sep_T8xm_4R', 'T8xm', 'sep'), ('sep_T9x_4R', 'T9x', 'sep')):
            comp = BF.Composite(B, kind, 4.)
            nx = len(comp.tn)-7
            src = base['joint_T7x_4R'] if mode == 'joint' else base['sep_T7x_4R']
            law0 = np.array(src['law'])
            if mode == 'joint':
                s0 = np.r_[law0, np.zeros(nx), src['amp']]
                law_eval, e_eval = comp, (lambda pl, comp=comp: comp.electrons(pl[:9]))
                lo, hi = comp.lo, comp.hi
                pert = lambda nT=len(comp.tn): np.r_[np.zeros(9), .15*rng.standard_normal(nT), np.zeros(3)]  # noqa: E731
                obj = comp
            else:
                s0 = np.r_[law0, np.zeros(nx), src['amp']]
                law_eval, e_eval = (lambda pt, comp=comp: comp(np.r_[pE, pt])), None
                lo, hi = comp.lo[9:], comp.hi[9:]
                pert = lambda nT=len(comp.tn): np.r_[.15*rng.standard_normal(nT), np.zeros(3)]  # noqa: E731

                class SepObjF:
                    names = comp.tn
                    bpos = comp.bpos

                    @staticmethod
                    def pressure_nodes(pt, x=None, bpos=None, comp=comp):
                        return comp.pressure_nodes(np.r_[pE, pt], x=x, bpos=bpos)
                obj = SepObjF
            LO, HI = np.r_[lo, la]+1e-6, np.r_[hi, ua]-1e-6
            starts = [np.clip(s0, LO, HI)]+[np.clip(s0+pert(), LO, HI) for _ in range(4)]
            run(name, law_eval, e_eval, lo, hi, starts, dict(electron_law=9, temperature_or_hydrostatic=len(comp.tn), amplitude=3), kind, obj, 'comp')
        (OUT/f'be_fit4r{suffix}.json').write_text(json.dumps(dict(task='fit4r', outer=outer, set=args.set, checks=B.check, results=res,
                                                                   seconds=time.time()-t0), indent=1)+'\n')
        np.savez_compressed(OUT/f'be_fit4r{suffix}_rows.npz', **arrays)
        return
    # ---------- pressure laws (start: the existing split fits)
    for name, mass, alpha, ref in (('plaw_const_4R', 'pl', 'const', 'existing_B6'), ('plaw_held_4R', 'pl', 'held', 'existing_A6'),
                                   ('plaw_massbreak_4R', 'sbplzz', 'held', 'existing_MB4')):
        law = BM.Law(D, mass, alpha, 'fixed')
        s0 = np.r_[C3[ref]['law'], C3[ref]['amp']]
        starts = [s0]+[np.clip(s0+.05*rng.standard_normal(len(s0)), np.r_[law.lo, la]+1e-6, np.r_[law.hi, ua]-1e-6) for _ in range(3)]
        starts += [np.r_[st, C3[ref]['amp']] for st in BM.starts_for(law, 3)]
        run(name, lambda pl, law=law: law(pl), None, law.lo, law.hi, starts,
            dict(electron_law=9, pressure_law=len(law.names), amplitude=3), 'plaw', law, 'plaw')
    # ---------- electron-law models
    for name, kind, mode, xout, ref in (('joint_T7x_4R', 'T7x', 'joint', 4., 'joint_T7x_rout6'), ('joint_T7_4R', 'T7', 'joint', 4., 'joint_T7_rout6'),
                                        ('sep_T7x_4R', 'T7x', 'sep', 4., 'sep_T7x_rout6'), ('joint_H6_xout6_4R', 'H6', 'joint', 6., 'joint_H6_xout6_rout6')):
        comp = BF.Composite(B, kind, xout)
        p_ref = np.array(C3[ref]['params'])
        if mode == 'joint':
            law_eval = comp
            e_eval = (lambda pl, comp=comp: comp.electrons(pl[:9]))
            lo, hi = comp.lo, comp.hi
            s0 = np.r_[p_ref, C3[ref]['amp']]
            nT = len(comp.tn)
            pert = lambda: np.r_[np.zeros(9), .2*rng.standard_normal(nT), np.zeros(3)]  # noqa: E731
        else:
            law_eval = (lambda pt, comp=comp: comp(np.r_[pE, pt]))
            e_eval = None
            lo, hi = comp.lo[9:], comp.hi[9:]
            s0 = np.r_[p_ref[9:], C3[ref]['amp']]
            pert = lambda: np.r_[.2*rng.standard_normal(len(comp.tn)), np.zeros(3)]  # noqa: E731
        LO, HI = np.r_[lo, la]+1e-6, np.r_[hi, ua]-1e-6
        starts = [s0]+[np.clip(s0+pert(), LO, HI) for _ in range(3)]

        class SepObj:                                            # point evaluation with the fixed electron law
            names = comp.tn
            bpos = comp.bpos

            @staticmethod
            def pressure_nodes(pt, x=None, bpos=None, comp=comp):
                return comp.pressure_nodes(np.r_[pE, pt], x=x, bpos=bpos)
        obj = comp if mode == 'joint' else SepObj
        run(name, law_eval, e_eval, lo, hi, starts, dict(electron_law=9, temperature_or_hydrostatic=len(comp.tn), amplitude=3), kind, obj, 'comp')
    (OUT/f'be_fit4r{suffix}.json').write_text(json.dumps(dict(task='fit4r', outer=outer, checks=B.check, results=res, seconds=time.time()-t0), indent=1)+'\n')
    np.savez_compressed(OUT/f'be_fit4r{suffix}_rows.npz', **arrays)


if __name__ == '__main__':
    main()
