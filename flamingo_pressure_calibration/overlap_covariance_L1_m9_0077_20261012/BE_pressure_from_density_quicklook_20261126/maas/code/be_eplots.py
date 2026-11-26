"""BE step 5 figures (STRW maas): electron density to 6 R200c (fitted to 4 R200c) for the electron law + 2-halo fitted alone
and for the joint electron law x temperature law model; RMS by band for both fields. Reads out/be_fit4r_e.json,
out/be_fit4r_e_rows.npz, out/be_fits_rows.npz. Plain labels.
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
LABEL = {'split_const': 'separate laws: electron law + 2-halo, pressure law (constant alpha) + 2-halo',
         'split_massbreak': 'separate laws, pressure law with mass break',
         'joint_T7x': 'electron law x temperature law, both + 2-halo, fitted together',
         'joint_T9x': 'electron law x temperature law (outer radius with M, z), fitted together',
         'joint_H6': 'hydrostatic + core term, fitted together',
         'sep_T7x': 'electron law (+ 2-halo) fixed, temperature law fitted'}
COL = {'split_const': '#222222', 'joint_T7x': '#1f77b4', 'split_massbreak': '#9a9a9a', 'joint_T9x': '#ff7f0e'}


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    C = json.loads((OUT/'be_fit4r_e.json').read_text())['results']
    R = np.load(OUT/'be_fit4r_e_rows.npz')
    M = np.load(OUT/'be_fits_rows.npz')
    block, lo, hi, region, shell, Mh, z, b32 = (M[k] for k in ('block', 'lo', 'hi', 'region', 'shell', 'M', 'z', 'b32'))
    dE, Ea, Eh = M['d_E'], M['E_a2ii'], M['E_hb']
    show = ['split_const', 'joint_T7x']
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
        ax.fill_between(xm[~inside], (xm**3*Eh[s])[~inside], (xm**3*Ea[s])[~inside], color='0.88', lw=0, label='host-bound to halo-component stack')
        ax.plot(xm[inside], (xm**3*dE[s])[inside], 'ko', ms=3.2, label='FLAMINGO all-gas electron stack, inside B32')
        ax.plot(xm[~inside], (xm**3*dE[s])[~inside], 'o', mfc='none', mec='k', ms=3.2, label='same, beyond B32 (fitted to 4 R200c)')
        for name in show:
            f, one = R[f'fullE__{name}'][s], R[f'oneE__{name}'][s]
            ax.plot(xm, xm**3*f, color=COL[name], lw=1.7, label=LABEL[name].split(':')[0] if name == 'split_const' else LABEL[name])
            ax.plot(xm, xm**3*one, color=COL[name], lw=1.1, ls='--', label='its 1-halo part')
            axr.plot(xm, f/dE[s], color=COL[name], lw=1.3)
        for a in (ax, axr):
            a.axvline(b32[i], color='0.4', ls=':', lw=1)
            a.axvline(4., color='0.7', ls='-', lw=.6)
            a.set_xscale('log')
            a.set_xlim(.02, 6.3)
        ax.set_yscale('log')
        ax.set_title(f'M200c = 10^{np.log10(Mh[i]):.1f} Msun, z = {z[i]:.1f}', fontsize=9)
        ax.set_xticklabels([])
        axr.axhline(1, color='k', lw=.7)
        axr.set_ylim(.5, 1.6)
        axr.set_xlabel('r / R200c')
        if c == 0:
            ax.set_ylabel('x^3 n_e mu / (rho_crit f_b)')
            axr.set_ylabel('model / stack')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(.5, -.08))
    fig.suptitle('Electron density, each model with its own 2-halo term, fitted to 4 R200c (dotted: B32; grey line: 4 R200c; '
                 '4-6 R200c not fitted)', fontsize=10)
    save(fig, 'electrons_profiles_4R')

    # RMS beyond B32 by band, both fields
    names = ['split_const', 'split_massbreak', 'joint_T7x', 'joint_T9x']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    w = .2
    for k, name in enumerate(names):
        r = C[name]
        for a, key, Lk in ((axes[0], 'outer_all_masses', 'L_4R_P'), (axes[1], 'outer_all_masses_E', 'L_4R_E')):
            bands = list(r[key])
            a.bar(np.arange(len(bands))+(k-1.5)*w, [r[key][b]['rms'] for b in bands], w, color=COL[name],
                  label=f"{LABEL[name]}: L = {r[Lk]:.1f}")
            a.set_xticks(range(len(bands)))
            a.set_xticklabels([b.replace('4.01', '4').replace('6.01', '6')+' R200c' for b in bands], fontsize=8)
    axes[0].set_title('Pressure beyond B32 (4-6 R200c not fitted)', fontsize=9)
    axes[1].set_title('Electron density beyond B32 (4-6 R200c not fitted)', fontsize=9)
    for a in axes:
        a.set_ylabel('RMS of ln(model / stack)')
        a.set_ylim(0, a.get_ylim()[1]*1.7)
        a.legend(fontsize=6.8, frameon=False, loc='upper left')
    save(fig, 'both_fields_outer_rms_4R')
    print('figures written')


if __name__ == '__main__':
    main()
