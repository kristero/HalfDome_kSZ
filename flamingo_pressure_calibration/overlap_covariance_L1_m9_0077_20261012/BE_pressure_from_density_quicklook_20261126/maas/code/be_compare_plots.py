"""BE step 3 figures (STRW maas): existing pressure fit with a free 2-halo amplitude against electron law x temperature law
+ 2-halo. Reads out/be_compare.json, out/be_compare_rows.npz, out/be_fits_rows.npz (row metadata). Plain labels; models
chosen with env BE_CMP (keys ref, alt, temp; names of be_compare.json results).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(os.environ['BE_OUT'])
FIG = OUT.parent/'figures'
FIG.mkdir(exist_ok=True)
CMP = json.loads(os.environ.get('BE_CMP', '{"ref": "existing_B6", "alt": "existing_A6", "temp": "joint_T7x_rout6"}'))
LABEL = {'existing_B6': 'pressure fit, free 2-halo amplitude (constant alpha; existing BD fit)',
         'existing_B4': 'pressure fit, free 2-halo amplitude to 4 R200c (constant alpha; existing)',
         'existing_A6': 'pressure fit, free 2-halo amplitude (alpha(z) of record; existing BD fit)',
         'existing_A4': 'pressure fit, free 2-halo amplitude to 4 R200c (alpha(z) of record; existing)',
         'existing_MB4': 'pressure fit with mass break (16 parameters) + 2-halo (existing)',
         'joint_T7x_rout6': 'electron law x temperature law + 2-halo (fitted together)',
         'joint_T7_rout6': 'electron law x temperature law + 2-halo (fitted together)',
         'sep_T7x_rout6': 'electron law x temperature law + 2-halo (separate)',
         'joint_H6_xout6_rout6': 'hydrostatic + core term + 2-halo (fitted together)'}
COL = {'ref': '#222222', 'alt': '#9a9a9a', 'temp': '#1f77b4'}
GROUPS = (('small', '10^12.7-13.1'), ('groups', '10^13.1-13.5'), ('mid', '10^13.5-14.3'), ('clusters', '>= 10^14.3'))


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    C = json.loads((OUT/'be_compare.json').read_text())['results']
    R = np.load(OUT/'be_compare_rows.npz')
    M = np.load(OUT/'be_fits_rows.npz')
    block, lo, hi, region, shell, Mh, z, b32, d = (M[k] for k in ('block', 'lo', 'hi', 'region', 'shell', 'M', 'z', 'b32', 'd_P'))
    Pa, Ph = M['P_a2ii'], M['P_hb']

    # ---------------- profiles to 6 R200c, total and 1-halo part
    picks = ((13., 0.), (14., 0.), (14.9, 0.), (13., 1.), (14., 1.), (13.5, 2.))
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.6), gridspec_kw=dict(height_ratios=[3, 1.3], hspace=.05, wspace=.28))
    for c, (lm10, zz) in enumerate(picks):
        i = int(np.argmin(abs(np.log10(Mh)-lm10)+10*abs(z-zz)))
        s = np.flatnonzero((block == block[i]) & shell)
        xm = np.sqrt(np.maximum(lo[s], 1e-3)*hi[s])
        o = np.argsort(xm)
        s, xm = s[o], xm[o]
        inside = region[s] == 0
        ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(xm[~inside], (xm**3*Ph[s])[~inside], (xm**3*Pa[s])[~inside], color='0.88', lw=0, label='host-bound to halo-component stack')
        ax.plot(xm[inside], (xm**3*d[s])[inside], 'ko', ms=3.2, label='FLAMINGO all-gas stack, fitted shells')
        ax.plot(xm[~inside], (xm**3*d[s])[~inside], 'o', mfc='none', mec='k', ms=3.2, label='FLAMINGO all-gas stack beyond the cut')
        for key in ('ref', 'temp'):
            name = CMP[key]
            full, one = R[f'full__{name}'][s], R[f'one__{name}'][s]
            ax.plot(xm, xm**3*full, color=COL[key], lw=1.7, label=LABEL.get(name, name))
            ax.plot(xm, xm**3*one, color=COL[key], lw=1.1, ls='--', label='its 1-halo part')
            axr.plot(xm, full/d[s], color=COL[key], lw=1.3)
        for a in (ax, axr):
            a.axvline(b32[i], color='0.4', ls=':', lw=1)
            a.set_xscale('log')
            a.set_xlim(.02, 6.3)
        ax.set_yscale('log')
        ymax = np.max(xm**3*d[s])
        ax.set_ylim(ymax*1e-4, ymax*4)
        ax.set_title(f'M200c = 10^{np.log10(Mh[i]):.1f} Msun, z = {z[i]:.1f}', fontsize=9)
        ax.set_xticklabels([])
        axr.axhline(1, color='k', lw=.7)
        axr.set_ylim(.5, 1.6)
        axr.set_xlabel('r / R200c')
        if c == 0:
            ax.set_ylabel('x^3 P_e / P200c')
            axr.set_ylabel('model / stack')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(.5, -.08))
    fig.suptitle('Both with the same 2-halo term, amplitude A_2h(z) free, fitted on the shells beyond the cut to 6 R200c '
                 '(dotted line: cut B32)', fontsize=10)
    save(fig, 'compare_profiles_2h')

    # ---------------- RMS inside (by mass) and beyond the cut (by radius)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    keys = ['ref', 'alt', 'temp']
    w = .26
    for k, key in enumerate(keys):
        name = CMP[key]
        r = C[name]
        vin = [r['bands_inside'].get(f'{g}|all', {}).get('rms', np.nan) for g, _ in GROUPS]
        axes[0].bar(np.arange(4)+(k-1)*w, vin, w, color=COL[key], label=f"{LABEL.get(name, name)}: L = {r['L_P']:.2f}")
        bands = list(r['outer_all_masses'])
        vout = [r['outer_all_masses'][b]['rms'] for b in bands]
        axes[1].bar(np.arange(len(bands))+(k-1)*w, vout, w, color=COL[key], label=f"{LABEL.get(name, name)}: L_out = {r['L_out6']:.0f}")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([g for _, g in GROUPS], fontsize=8)
    axes[0].set_xlabel('M200c [Msun]')
    axes[0].set_title('Inside the cut (all fitted residuals, bins with >= 100 haloes)', fontsize=9)
    axes[1].set_xticks(range(len(bands)))
    axes[1].set_xticklabels([b.replace('4.01', '4').replace('6.01', '6')+' R200c' for b in bands], fontsize=8)
    axes[1].set_title('Beyond the cut (shells of all bins with >= 100 haloes)', fontsize=9)
    for a in axes:
        a.set_ylabel('RMS of ln(model / stack)')
        a.set_ylim(0, a.get_ylim()[1]*1.55)
        a.legend(fontsize=7.5, frameon=False, loc='upper left')
    save(fig, 'compare_rms_2h')

    # ---------------- amplitude
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in keys:
        name = CMP[key]
        a = C[name]['amp_by_z']
        zz = np.array([float(k) for k in a])
        ax.plot(zz, [a[k] for k in a], 'o-', color=COL[key], label=LABEL.get(name, name))
    ax.set_xlabel('z')
    ax.set_ylabel('2-halo amplitude A_2h(z)')
    ax.set_yscale('log')
    ax.legend(fontsize=7.5, frameon=False)
    save(fig, 'compare_amplitude')
    print('figures written', sorted(p.name for p in FIG.glob('compare_*.png')))


if __name__ == '__main__':
    main()
