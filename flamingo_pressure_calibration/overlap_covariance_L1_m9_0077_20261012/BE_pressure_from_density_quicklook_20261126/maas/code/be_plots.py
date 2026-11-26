"""BE figures (STRW maas): pressure from the electron law with a temperature model vs the pressure law, FLAMINGO L1_m9.

Reads out/be_fits.json, out/be_fits_rows.npz, out/be_heldout.json, out/be_twohalo.json; evaluates the laws with be_fit.
Writes figures/*.png and *.pdf. Labels in plain words (no internal codes); at most three model curves per panel.
Models shown are chosen in SHOW (names of be_fits.json results).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import be_fit as BF  # noqa: E402

OUT = Path(os.environ['BE_OUT'])
FIG = OUT.parent/'figures'
FIG.mkdir(exist_ok=True)
SHOW = json.loads(os.environ.get('BE_SHOW', '{"ref": "plaw_held", "temp": "sep_T7", "hse": "sep_H6"}'))
LABEL = {'plaw_held': 'pressure law of record (9 parameters)', 'plaw_const': 'pressure law, one constant alpha (10 parameters)',
         'sep_T5': 'electron law x temperature, no core term (+5)', 'sep_T7': 'electron law x temperature law (+7)',
         'sep_T8': 'electron law x temperature law (+8)', 'joint_T7': 'electron law x temperature law, fitted together (+7)',
         'joint_T8': 'electron law x temperature law, fitted together (+8)',
         'sep_H3': 'hydrostatic from electron law (+3)', 'sep_H3c': 'hydrostatic, free concentration (+4)',
         'sep_H6': 'hydrostatic from electron law + core term (+6)', 'joint_H3': 'hydrostatic, fitted together (+3)',
         'joint_H6': 'hydrostatic + core term, fitted together (+6)',
         'joint_H6_xout6': 'hydrostatic + core term, fitted together (+6)', 'sep_H6_xout6': 'hydrostatic from electron law + core term (+6)'}
COL = {'ref': '#4a4a4a', 'temp': '#1f77b4', 'hse': '#d62728'}
GROUP_LABEL = {'small': '10^12.7-13.1', 'groups': '10^13.1-13.5', 'mid': '10^13.5-14.3', 'clusters': '>= 10^14.3'}


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    fits = json.loads((OUT/'be_fits.json').read_text())['results']
    f2 = OUT/'be_fits2.json'
    if f2.exists():
        fits.update({k: v for k, v in json.loads(f2.read_text())['results'].items() if 'params' in v})
    R = np.load(OUT/'be_fits_rows.npz')
    B = BF.BEData()
    D = B.D
    objs = {}
    for key, name in SHOW.items():
        r = fits[name]
        p = np.array(r['params'])
        if name.startswith('plaw'):
            law = BF.BM.Law(D, 'pl', name.split('_')[1], 'fixed')
            objs[key] = (name, BF.point_fns(B, 'plaw', law, p), None)
        else:
            comp = BF.Composite(B, name.split('_')[1], r['info'].get('xout', 4.))
            objs[key] = (name, BF.point_fns(B, 'comp', comp, p), BF.e_point_fn(B, p[:9]))
    pE = np.array(fits['electron_law']['params'])
    efn = BF.e_point_fn(B, pE)

    # ---------------- figure 1: pressure profiles to 4 R200c, six mass x redshift points
    picks = ((13., 0.), (14., 0.), (14.9, 0.), (13., 1.), (14., 1.), (13.5, 2.))
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.4), gridspec_kw=dict(height_ratios=[3, 1.3], hspace=.05, wspace=.28))
    x = np.geomspace(.02, 4., 300)
    for c, (lm10, z) in enumerate(picks):
        i = int(np.argmin(abs(np.log10(D.M)-lm10)+10*abs(D.z-z)))
        b = int(D.block[i])
        s = (D.block == b) & D.shell & (D.hi <= 4.01)
        xm = np.sqrt(np.maximum(D.lo[s], 1e-3)*D.hi[s])
        ax, axr = axes[0, c], axes[1, c]
        inside = D.region[s] == 0
        ax.fill_between(xm[~inside], (xm**3*B.T['P_hb'][s])[~inside], (xm**3*B.T['P_a2ii'][s])[~inside], color='0.85', lw=0,
                        label='host-bound to halo-component stack')
        ax.plot(xm[inside], (xm**3*D.d[s])[inside], 'ko', ms=3.5, label='FLAMINGO stack, fitted shells')
        ax.plot(xm[~inside], (xm**3*D.d[s])[~inside], 'o', mfc='none', mec='k', ms=3.5, label='FLAMINGO stack beyond the cut')
        for key, (name, fn, _) in objs.items():
            P = fn(b, x)
            ax.plot(x, x**3*P, color=COL[key], lw=1.6, label=LABEL.get(name, name))
            Pm = fn(b, xm)
            axr.plot(xm, Pm/D.d[s], color=COL[key], lw=1.3)
        for a in (ax, axr):
            a.axvline(D.b32[i], color='0.4', ls=':', lw=1)
            a.set_xscale('log')
            a.set_xlim(.02, 4.2)
        ax.set_yscale('log')
        ax.set_title(f'M200c = 10^{np.log10(D.M[i]):.1f} Msun, z = {D.z[i]:.1f}', fontsize=9)
        ax.set_xticklabels([])
        axr.axhline(1, color='k', lw=.7)
        axr.set_ylim(.5, 1.6)
        axr.set_xlabel('r / R200c')
        if c == 0:
            ax.set_ylabel('x^3 P_e / P200c')
            axr.set_ylabel('model / stack')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(.5, -.07))
    fig.suptitle('Electron pressure: the pressure law against pressure predicted from the electron-density law '
                 '(dotted line: fitted cut B32; models extended beyond it)', fontsize=10)
    save(fig, 'pressure_profiles')

    # ---------------- figure 2: RMS of ln(model/stack) inside the cut, by mass group
    groups = [g for g, _, _ in BF.GROUPS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = [SHOW['ref'], SHOW['temp'], SHOW['hse']]
    wbar = .26
    for k, name in enumerate(names):
        vals = [fits[name]['bands_P'].get(f'{g}|all', {}).get('rms', np.nan) for g in groups]
        axes[0].bar(np.arange(4)+(k-1)*wbar, vals, wbar, color=list(COL.values())[k], label=LABEL.get(name, name))
    axes[0].set_title('Pressure inside the cut (all fitted residuals of bins with >= 100 haloes)', fontsize=9)
    e_names = [('electron_law', 'electron law alone (9 parameters)', '#4a4a4a')]
    for nm in (SHOW['temp'], SHOW['hse']):
        if nm in fits:
            e_names.append((nm, f'electron law when fitted together with {"temperature law" if "T" in nm else "hydrostatic model"}',
                            '#1f77b4' if 'T' in nm else '#d62728'))
    for k, (nm, lab, col) in enumerate(e_names):
        vals = [fits[nm]['bands_E'].get(f'{g}|all', {}).get('rms', np.nan) for g in groups]
        axes[1].bar(np.arange(4)+(k-1)*wbar, vals, wbar, color=col, label=lab)
    axes[1].set_title('Electron density inside the cut', fontsize=9)
    for a in axes:
        a.set_xticks(range(4))
        a.set_xticklabels([GROUP_LABEL[g] for g in groups], fontsize=8)
        a.set_xlabel('M200c [Msun]')
        a.set_ylabel('RMS of ln(model / stack)')
        a.set_ylim(0, a.get_ylim()[1]*1.45)
        a.legend(fontsize=8, frameon=False, loc='upper right')
    save(fig, 'residual_rms')

    # ---------------- figure 3: stack temperature T_e/T200 = C_T P/E against the temperature law
    name_t = SHOW['temp']
    _, fnP, fnE = objs['temp']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    cmap = plt.get_cmap('viridis')
    for c, zt in enumerate((0., 1., 2.)):
        ax = axes[c]
        bl = sorted({int(b) for b in np.unique(D.block[(abs(D.z-zt) < .01) & (D.nhalo >= 100)])}, key=lambda b: D.M[D.block == b][0])
        bl = bl[::max(1, len(bl)//5)]
        for b in bl:
            lmv = np.log10(D.M[D.block == b][0])
            col = cmap((lmv-12.7)/(15.1-12.7))
            s = (D.block == b) & D.shell & (D.region == 0)
            xm = np.sqrt(np.maximum(D.lo[s], 1e-3)*D.hi[s])
            ax.plot(xm, R['tau'][s], 'o', color=col, ms=3)
            xx = np.geomspace(xm.min(), 4., 200)
            ax.plot(xx, BF.C_T*fnP(b, xx)/fnE(b, xx), color=col, lw=1.4, label=f'10^{lmv:.1f}')
            ax.axvline(D.b32[D.block == b][0], color=col, ls=':', lw=.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'z = {zt:g}', fontsize=9)
        ax.set_xlabel('r / R200c')
        ax.legend(title='M200c [Msun]', fontsize=7, title_fontsize=7, frameon=False)
    axes[0].set_ylabel('T_e / T200 = 428 P / n_e of the stacks')
    fig.suptitle(f'Stack temperature (points) and the fitted temperature law (lines; {LABEL.get(name_t, name_t)})', fontsize=10)
    save(fig, 'temperature_profiles')

    # ---------------- figure 4: implied non-thermal fraction for haloes >= 10^13.5
    name_h = SHOW['hse']
    rh = fits[name_h]
    ph = np.array(rh['params'])
    comp = BF.Composite(B, name_h.split('_')[1], rh['info'].get('xout', 4.))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for c, zt in enumerate((0., 1., 2.)):
        ax = axes[c]
        bl = sorted({int(b) for b in np.unique(D.block[(abs(D.z-zt) < .01) & (D.nhalo >= 100) & (np.log10(D.M) >= 13.5)])}, key=lambda b: D.M[D.block == b][0])
        for b in bl:
            i = int(np.flatnonzero(D.block == b)[0])
            lmv = np.log10(D.M[i])
            col = cmap((lmv-12.7)/(15.1-12.7))
            s = (D.block == b) & D.shell & (D.region == 0)
            xm = np.sqrt(np.maximum(D.lo[s], 1e-3)*D.hi[s])
            bpos = np.full(s.sum(), comp.bpos[i])
            Pt = np.sum(D.wq[s]*comp.p_tot_nodes(ph[:9], D.xq[s], bpos, np.exp(ph[12]) if comp.kind == 'H3c' else 1.), axis=1)
            ax.plot(xm, 1-D.d[s]/(BF.PE_FRAC*Pt), 'o-', color=col, ms=3, lw=.8, label=f'10^{lmv:.1f}')
        xx = np.geomspace(.05, 1.6, 100)
        ax.plot(xx, np.clip(ph[9]*(1+zt)**ph[10]*xx**ph[11], 0, .95), 'k-', lw=2, label='fitted non-thermal fraction')
        fz = min((1+zt)**.5, (4**-.8/.18-1)*np.tanh(.5*zt)+1)
        ax.plot(xx, .18*fz*xx**.8, 'k--', lw=1.5, label='Shaw+10 form used by Schneider+25')
        ax.axhline(0, color='0.5', lw=.7)
        ax.set_xscale('log')
        ax.set_ylim(-.5, 1.)
        ax.set_title(f'z = {zt:g}', fontsize=9)
        ax.set_xlabel('r / R200c')
        ax.legend(title='M200c [Msun]', fontsize=7, title_fontsize=7, frameon=False, ncol=2)
    axes[0].set_ylabel('1 - P_stack / P_hydrostatic(electron law)')
    fig.suptitle('Implied non-thermal pressure fraction (points; includes the cool-core / multiphase deficit at small radius)', fontsize=10)
    save(fig, 'nonthermal_fraction')

    # ---------------- figure 5: leave-one-output-out loss
    hp = OUT/'be_heldout.json'
    if hp.exists():
        H = json.loads(hp.read_text())['results']
        zs = [k for k in H[SHOW['ref']] if not k.startswith('total')]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, name in SHOW.items():
            name = name if name in H else name.replace('_xout6', '')
            if name in H:
                v = [H[name][zk]['L_P_test']/H[name][zk]['w_test'] for zk in zs]
                ax.plot([float(zk) for zk in zs], np.sqrt(v), 'o-', color=COL[key], label=f'{LABEL.get(name, name)}: total {H[name]["total_L_P_test"]:.1f}')
        ax.set_xlabel('held-out output redshift')
        ax.set_ylabel('weighted RMS of ln(model / stack), held out')
        ax.legend(fontsize=8, frameon=False)
        ax.set_title('Fit on nine outputs, predict the tenth (pressure, inside the cut)', fontsize=9)
        save(fig, 'heldout')
    print('figures written', sorted(p.name for p in FIG.glob('*.png')))


if __name__ == '__main__':
    main()
