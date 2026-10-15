"""PNG: SOAP-only coarse stacks vs the particle stack (same haloes, same weights), z = 0, L1_m9."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'
A = json.load(open(OUT + 'soap_audit_part2.json'))['stacks']
BINS = ['0', '3', '6', '10']
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.linewidth': 0.8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(4, 4, figsize=(13, 10.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.3, 3, 1.3], 'hspace': 0.08, 'wspace': 0.25})
rows = [('P', r'$x^3\,P_e/P_{200}$', 3), ('rho_SO_plus_kpc', r'$x^2\,\rho_{\rm gas}/(f_b\,200\rho_c)$', 2)]
for c, b in enumerate(BINS):
    for r, (field, ylabel, power) in enumerate(rows):
        s = A[field][b]
        ax, axr = axes[2 * r, c], axes[2 * r + 1, c]
        fx, fp = np.array(s['fine_x']), np.array(s['fine_profile'])
        ok = fp > 0
        ax.plot(fx[ok], fx[ok]**power * fp[ok], color=BLUE, lw=2, solid_capstyle='round', label='particles, 67 shells (all haloes in bin)')
        e = np.array(s['x_edges']); va = np.array(s['soap_A']); vb = np.array(s['soap_B']); use = np.array(s['usable_shell'])
        for j in range(len(va)):
            lo, hi = max(e[j], 0.02), e[j + 1]
            xm = np.sqrt(lo * hi) if e[j] > 0 else hi / 2
            # plot shell average at the volume-weighted mean radius of the shell
            xv = (0.75 * (e[j + 1]**4 - e[j]**4) / (e[j + 1]**3 - e[j]**3))
            alpha = 1.0 if use[j] else 0.35
            ax.plot([lo, hi], [xv**power * va[j]] * 2, color=ORANGE, lw=2, alpha=alpha, solid_capstyle='round',
                    label='SOAP only: difference of stacked cumulatives' if j == 0 else None)
            if field == 'P':
                ax.plot([xv], [xv**power * vb[j]], 'o', ms=5, color=AQUA, mec='#fcfcfb', mew=1.5, alpha=alpha,
                        label='SOAP only: per-halo shells' if j == 0 else None)
            ra, rb = s['ratio_A_to_fine'][j], s['ratio_B_to_fine'][j]
            axr.plot([lo, hi], [ra, ra], color=ORANGE, lw=2, alpha=alpha, solid_capstyle='round')
            if field == 'P':
                axr.plot([xv], [rb], 'o', ms=5, color=AQUA, mec='#fcfcfb', mew=1.5, alpha=alpha)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.grid(True, color=GRID, lw=0.6); axr.grid(True, color=GRID, lw=0.6)
        axr.axhline(1, color=INK2, lw=0.8)
        axr.set_ylim(0.85, 1.15)
        if c == 0:
            ax.set_ylabel(ylabel); axr.set_ylabel('SOAP / particles\n(same interval)', fontsize=8)
        if r == 0:
            ax.set_title(f"log M200c = {s['bin'][0]:.1f}-{s['bin'][1]:.1f}  ({s['n_halos']} haloes)", fontsize=9, color=INK)
        ax.set_xlim(0.02, 4)
for ax in axes[-1]:
    ax.set_xlabel(r'$x = r/R_{200c}$')
from matplotlib.lines import Line2D
h = [Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=ORANGE, lw=2),
     Line2D([], [], marker='o', ls='none', ms=6, color=AQUA, mec='#fcfcfb', mew=1.5)]
l = ['particles, 67 shells (same haloes and weights)', 'SOAP only: difference of stacked cumulatives at mean aperture radius',
     'SOAP only: mean of per-halo shells (pressure)']
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
fig.text(0.5, 0.04, 'FLAMINGO L1_m9 z = 0: pressure from SOAP SO ComptonY (no recent AGN); gas density from SOAP SO + InclusiveSphere gas masses. '
         '\nFaded = shell with < 50 gas particles per halo (median) or outer edge below 0.025 R200c. The innermost bar is the average inside the smallest aperture.', ha='center', fontsize=8, color=INK2)
fig.savefig(OUT + 'soap_vs_particle_stacks.png', dpi=130, bbox_inches='tight')
print('saved')
