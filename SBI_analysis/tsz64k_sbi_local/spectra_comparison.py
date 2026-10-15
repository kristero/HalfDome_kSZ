"""tSZ auto-spectra of the HalfDome models against the FLAMINGO fiducial map, raw and as the SBI observes them.

Maps
  FLAMINGO L1_m9           the fiducial FLAMINGO tSZ map (the SBI's FLAMINGO observation; flamingo_tsz results)
  HalfDome + B12           frozen 64k engine (paint_mocks.jl: raw NSIDE 8192 point samples, 4 R200c spheres)
  HalfDome + FLAMINGO fit  frozen engine at the FLAMINGO L2p8 pressure-profile fit (flamingo_joint_fit.toml)
  HalfDome + Lee22         the full corrected Lee22 model map (frb_map_generation lee22p_m1: Table 7 pressure x (1+z)^2,
                           raw NSIDE 4096 point samples, projected 4 R200c aperture)
Figures (results_spectra_comparison/)
  fig_raw_autospectra.png             full-sky D_ell, no beam / mask / noise (log-ell display bins)
  fig_raw_autospectra_unbinned.png    the same at every multipole
  fig_sbi_observations.png            after the SBI operator (2 arcmin beam, f_sky 0.4 apodized mask, two SO baseline
                                      Deproj-0 noise splits, signed cross spectrum): the 40 band powers
  fig_sbi_observations_unbinned.png   the same at every multipole ell = 80..7979 (the inputs of Merlin, PCA and the
                                      no-binning model), one panel per map
Raw spectra: stored beamed spectra are divided by B_ell^2; the FLAMINGO map is a pixel average, so its pixel window is
divided out there (the HalfDome maps are point samples). In the SBI view nothing is divided out: FLAMINGO keeps its
pixel window, as in the SBI. Noise seeds: the engine mocks use the B12 observation's (39001001/39001002), the FLAMINGO
and full Lee22 maps the FLAMINGO observations' (22446/22447). Scatter: 21 noise draws of the B12 mock (the sky is the
same in every SBI simulation, so this is the whole scatter of a band or multipole).

    python spectra_comparison.py
Writes the figures and spectra_summary.json. Missing inputs are skipped (reported).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                                  # /home/cbllover/HalfDome
OUT = HERE / 'results_spectra_comparison'
LMAX = 7979
ELL = np.arange(LMAX + 1)
DL = ELL * (ELL + 1) / (2 * np.pi)
SIGMA = np.deg2rad(2 / 60) / np.sqrt(8 * np.log(2))
B2 = np.exp(-ELL * (ELL + 1) * SIGMA ** 2)                 # the engine's Healpix.gaussbeam(2 arcmin), squared
EDGES = np.r_[np.arange(80, 7881, 200), 7980]              # the SBI's 40 bands
SBI_ELL = slice(80, LMAX + 1)                              # the multipoles the SBI uses

FLAMINGO_TSZ = ROOT / 'SBI_analysis' / 'flamingo_tsz' / 'cluster_results' / 'results' / 'L1_m9'
PRODUCTS = ROOT / 'flamingo_pressure_calibration' / 'halfdome_tsz_comparison' / 'products' / 'comparison_products.npz'
LEE22_FULL_RAW = ROOT / 'frb_map_generation' / 'outputs' / 'lee22_corrected_20260930' / 'spectra' / 'corrected_spectra.npz'
MOCKS = HERE / 'mocks'
ENGINE = dict(B12=MOCKS / 'pass1_B12' / 'B12', FL_fit=MOCKS / 'idark_flamingo_fit' / 'out' / 'FLAMINGO_L2p8_fit')
LEE22_FULL_OBS = MOCKS / 'idark_lee22_full' / 'out' / 'Lee22_corrected_full'
ORDER = ('FL_L1', 'B12', 'FL_fit', 'Lee22_full')

# entity -> (label, colour, marker); FLAMINGO data in ink, models in the validated hues
INK, INK2, GRID, SURFACE = '#222222', '#6b6b6b', '#e6e6e3', '#fcfcfb'
STYLE = dict(FL_L1=('FLAMINGO L1_m9 map (fiducial; the SBI observation)', INK, 'o'),
             B12=('HalfDome + Battaglia+12', '#eda100', 's'),
             FL_fit=('HalfDome + FLAMINGO pressure-profile fit', '#2a78d6', 'D'),
             Lee22_full=('HalfDome + Lee22 (full corrected model)', '#e34948', '^'))


def bin40(cl):
    """The SBI's 40 band powers (D_ell, (2 ell + 1) weights) of a C_ell array starting at ell = 0."""
    dl = np.asarray(cl)[:LMAX + 1] * DL
    w = 2 * ELL + 1
    return np.array([np.sum(dl[a:b] * w[a:b]) / np.sum(w[a:b]) for a, b in zip(EDGES[:-1], EDGES[1:])])


def log_bins(cl, lo=20, n=56):
    """D_ell in logarithmic ell bins (display smoothing): (centres, values)."""
    edges = np.unique(np.round(np.geomspace(lo, LMAX + 1, n + 1)).astype(int))
    dl = np.asarray(cl)[:LMAX + 1] * DL
    w = 2 * ELL + 1
    centres = np.array([np.sum(ELL[a:b] * w[a:b]) / np.sum(w[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    values = np.array([np.sum(dl[a:b] * w[a:b]) / np.sum(w[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return centres, values


def load_raw(missing):
    """Full-sky raw C_ell (ell 0..7979) per map, and the NSIDE 4096 pixel window squared."""
    raw = {}
    pw2 = np.load(PRODUCTS)['pixwin'][:LMAX + 1] ** 2
    p = FLAMINGO_TSZ / 'unmasked_clean_cl.npy'
    if p.exists():
        raw['FL_L1'] = np.load(p)[:LMAX + 1] / B2 / pw2      # stored beamed (process_maps.jl), pixel-averaged map
    else:
        missing.append(str(p))
    for key, folder in ENGINE.items():
        p = folder / 'unmasked_clean_cl.npy'
        if p.exists():
            raw[key] = np.load(p)[:LMAX + 1] / B2            # engine output is beamed; point samples (no pixel window)
        else:
            missing.append(str(p))
    raw['Lee22_full'] = np.load(LEE22_FULL_RAW)['cl_y_lee22p_m1'][:LMAX + 1]
    return raw, pw2


def load_observed(missing):
    """Per map: (clean masked beamed C_ell, noisy masked cross C_ell) as the SBI sees them; the B12 noise draws."""
    obs = {}
    sources = dict(FL_L1=(FLAMINGO_TSZ / 'masked_clean_cl.npy', FLAMINGO_TSZ / 'masked_noisy_cross_cl.npy'),
                   Lee22_full=(LEE22_FULL_OBS / 'masked_clean_cl.npy', LEE22_FULL_OBS / 'masked_noisy_cross_cl.npy'),
                   **{k: (f / 'masked_clean_cl.npy', f / 'noisy_cross_39001001_39001002.npy') for k, f in ENGINE.items()})
    for key, (clean, noisy) in sources.items():
        if clean.exists() and noisy.exists():
            obs[key] = (np.load(clean)[:LMAX + 1], np.load(noisy)[:LMAX + 1])
        else:
            missing.append(str(noisy))
    draws = np.array([np.load(p)[:LMAX + 1] for p in sorted(ENGINE['B12'].glob('noisy_cross_*.npy'))])
    return obs, draws


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)


def log_ell_axis(ax, lo, ticks):
    ax.set_xscale('log')
    ax.set_xlim(lo, 8100)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


def ratio_axis(ax, label, top=3.2):
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_yscale('log')
    ax.set_ylim(0.3, top)
    ticks = [t for t in (0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4) if t < top]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.set_ylabel(label, color=INK, fontsize=10)


def figure_raw(raw, plt, unbinned=False):
    """Full-sky D_ell of each map and its ratio to FLAMINGO L1_m9: log-ell display bins, or every multipole."""
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(10, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.06))
    fig.patch.set_facecolor(SURFACE)
    lo = 2 if unbinned else 20
    for key in [k for k in ORDER if k in raw]:
        label, colour, _ = STYLE[key]
        if unbinned:
            c, v, ref = ELL[lo:], (raw[key] * DL)[lo:], (raw['FL_L1'] * DL)[lo:]
            lw = 0.9
        else:
            (c, v), (_, ref) = log_bins(raw[key]), log_bins(raw['FL_L1'])
            lw = 2.2 if key == 'FL_L1' else 2.0
        z = 3 if key == 'FL_L1' else 2
        ax.plot(c, 1e12 * v, color=colour, lw=lw, label=label, zorder=z)
        rx.plot(c, v / ref, color=colour, lw=lw, zorder=z)
    for a in (ax, rx):
        a.axvspan(lo, 80, color='#efefec', zorder=0)
        style_axes(a)
    log_ell_axis(rx, lo, [2, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000] if unbinned else
                 [20, 50, 100, 200, 500, 1000, 2000, 5000])
    ax.set_yscale('log')
    ax.set_ylim(0.004 if unbinned else 0.03, 6)
    ax.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell/2\pi$  [$10^{-12}$, Compton $y^2$]', color=INK, fontsize=10)
    what = 'every multipole (no binning)' if unbinned else 'log-spaced display bins'
    ax.set_title(f'tSZ auto-spectra, full sky, no beam / mask / noise, {what}: HalfDome models vs FLAMINGO',
                 color=INK, fontsize=11, loc='left')
    ratio_axis(rx, 'ratio to FLAMINGO L1_m9', top=4.2 if unbinned else 3.2)
    rx.text(lo * 1.1, 0.34, 'ell < 80: not used by the SBI', color=INK2, fontsize=8)
    rx.set_xlabel(r'multipole $\ell$', color=INK, fontsize=10)
    ax.legend(loc='lower right', fontsize=8.5, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005, 'FLAMINGO L1_m9: pixel window and 2′ beam divided out; FLAMINGO D3A cosmology.  HalfDome: point '
             'samples in HalfDome cosmology (no cosmology shift applied).\nFrozen 64k engine for B12 and the FLAMINGO fit '
             '(beam divided out); the full Lee22 map from the FRB pipeline (raw NSIDE 4096, projected 4 R200c).',
             color=INK2, fontsize=7.5)
    name = 'fig_raw_autospectra_unbinned.png' if unbinned else 'fig_raw_autospectra.png'
    fig.savefig(OUT / name, dpi=140, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def figure_observed(obs, draws, plt):
    """The 40 band powers the SBI sees (noisy markers, clean lines) and their ratio to FLAMINGO L1_m9 without noise."""
    order = [k for k in ORDER if k in obs]
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    sigma = np.array([bin40(d) for d in draws]).std(0, ddof=1)
    ref = bin40(obs['FL_L1'][0])
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(10, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.06))
    fig.patch.set_facecolor(SURFACE)
    floor = 4e-4                                   # in the plotted units (1e-12)
    offsets = dict(B12=-1.5, FL_L1=-0.5, FL_fit=0.5, Lee22_full=1.5)    # steps of min(2 % of ell, 30): inside the band
    for key in order:
        label, colour, marker = STYLE[key]
        clean, noisy = bin40(obs[key][0]), bin40(obs[key][1])
        x = centres + offsets[key] * np.minimum(0.02 * centres, 30.)
        ax.plot(centres, 1e12 * clean, color=colour, lw=1.6, zorder=2)
        pos = noisy > 0
        ax.plot(x[pos], 1e12 * noisy[pos], ls='none', marker=marker, ms=5.5, mfc=colour, mec=colour, mew=1.2,
                label=label, zorder=3)
        if (~pos).any():                           # negative band powers: an arrow at the floor
            ax.plot(x[~pos], np.full((~pos).sum(), floor * 1.25), ls='none', marker='$↓$', ms=9, color=colour,
                    zorder=3)
        rx.plot(centres, clean / ref, color=colour, lw=1.4, zorder=2)
        rx.errorbar(x, noisy / ref, yerr=sigma / ref, ls='none', marker=marker, ms=4.5, mfc=colour, mec=colour,
                    ecolor=colour, elinewidth=1, capsize=0, zorder=3)
    ax.plot(centres, 1e12 * sigma, color=INK2, ls=(0, (1, 1.5)), lw=1.6, label='noise scatter of one band (21 noise draws)')
    for a in (ax, rx):
        style_axes(a)
    log_ell_axis(rx, 75, [100, 200, 500, 1000, 2000, 5000])
    ax.set_yscale('log')
    ax.set_ylim(floor, 2)
    ax.set_ylabel(r'band power $D_b$  [$10^{-12}$]: beamed, masked, noisy', color=INK, fontsize=10)
    ax.set_title('What the SBI sees: 40 band powers after the 2′ beam, f_sky 0.4 mask and SO noise (split cross '
                 'spectrum)', color=INK, fontsize=11, loc='left')
    handles, labels = ax.get_legend_handles_labels()       # in the empty lower part of the ratio panel
    rx.legend(handles, labels, loc='lower left', ncol=2, fontsize=8, frameon=False, labelcolor=INK,
              columnspacing=1.2, handletextpad=0.4)
    ax.text(8000, floor * 2.2, '↓ at the bottom: negative band power', color=INK2, fontsize=7.5, ha='right')
    ratio_axis(rx, 'ratio to FLAMINGO L1_m9 (no noise)', top=4.2)
    rx.set_xlabel(r'multipole $\ell$ (band centre)', color=INK, fontsize=10)
    fig.text(0.01, 0.005, 'Markers: the noisy band powers (the 40-band SBI inputs); lines: the same maps without noise; '
             'error bars: band scatter from 21 noise draws (B12 mock).\nNoise seeds: HalfDome engine mocks 39001001/39001002 '
             '(the B12 observation), FLAMINGO and the full Lee22 map 22446/22447. FLAMINGO keeps its pixel window, as in the SBI.',
             color=INK2, fontsize=7.5)
    fig.savefig(OUT / 'fig_sbi_observations.png', dpi=140, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def figure_observed_unbinned(obs, draws, plt):
    """Every multipole the SBI uses, one panel per map: the noisy cross spectrum (thin), the same map without noise
    (dark line) with the 1 sigma noise scatter of one multipole (band), and the 40 noisy band powers (markers)."""
    order = [k for k in ORDER if k in obs]
    sigma_ell = (draws * DL).std(0, ddof=1)[SBI_ELL]
    ell = ELL[SBI_ELL]
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    fig, axes = plt.subplots(len(order), 1, figsize=(10, 2.9 * len(order) + 0.8), sharex=True,
                             gridspec_kw=dict(hspace=0.12))
    fig.patch.set_facecolor(SURFACE)
    for ax, key in zip(axes, order):
        label, colour, marker = STYLE[key]
        clean, noisy = (obs[key][0] * DL)[SBI_ELL], (obs[key][1] * DL)[SBI_ELL]
        trace = '#8a8a86' if key == 'FL_L1' else colour            # FLAMINGO's ink would hide the no-noise line
        ax.plot(ell, 1e12 * noisy, color=trace, lw=0.45, alpha=0.8, zorder=2, label='noisy, every multipole')
        ax.fill_between(ell, 1e12 * (clean - sigma_ell), 1e12 * (clean + sigma_ell), color='#8f8f8a', alpha=0.35, lw=0,
                        zorder=1, label='no noise ± 1σ noise scatter of one multipole')
        ax.plot(ell, 1e12 * clean, color=INK, lw=0.7, zorder=3, label='no noise')
        ax.plot(centres, 1e12 * bin40(obs[key][1]), ls='none', marker=marker, ms=5, mfc='white', mec=INK, mew=1.1,
                zorder=4, label='noisy 40 band powers')
        peak = 1e12 * clean.max()
        ax.set_ylim(-0.45 * peak, 1.3 * peak)
        ax.axhline(0, color=INK2, lw=0.6)
        style_axes(ax)
        ax.text(0.01, 0.95, label, transform=ax.transAxes, color=INK, fontsize=10, va='top', fontweight='bold')
        ax.set_ylabel(r'$D_\ell$  [$10^{-12}$]', color=INK, fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()           # a coloured trace, not FLAMINGO's grey
    fig.subplots_adjust(top=0.925)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.958), fontsize=8.5, frameon=False,
               labelcolor=INK, ncol=4)
    log_ell_axis(axes[-1], 75, [100, 200, 500, 1000, 2000, 5000])
    axes[-1].set_xlabel(r'multipole $\ell$', color=INK, fontsize=10)
    fig.suptitle('What the SBI sees at every multipole (ℓ = 80–7979, the Merlin / PCA / no-binning inputs): '
                 'beamed, masked, noisy', color=INK, fontsize=11, x=0.08, ha='left', y=0.975)
    fig.text(0.01, 0.003, 'Each panel on its own scale. Scatter band: 21 noise draws of the B12 mock (the same sky; at low '
             'ℓ it grows with the signal, so it is a slight underestimate for the brighter maps).\nNoise seeds: engine mocks '
             '39001001/39001002, FLAMINGO and the full Lee22 map 22446/22447.', color=INK2, fontsize=7.5)
    fig.savefig(OUT / 'fig_sbi_observations_unbinned.png', dpi=140, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    missing = []
    raw, pw2 = load_raw(missing)
    obs, draws = load_observed(missing)
    figure_raw(raw, plt)
    figure_raw(raw, plt, unbinned=True)
    figure_observed(obs, draws, plt)
    figure_observed_unbinned(obs, draws, plt)
    # numbers behind the figures: D_ell ratios to FLAMINGO L1_m9 at a few multipoles and band S/N
    pick = [100, 300, 1000, 2000, 3000, 5000, 7000]
    ref = raw['FL_L1']
    sigma_b = np.array([bin40(d) for d in draws]).std(0, ddof=1)
    summary = dict(missing=missing, ratio_to_FL_L1_raw={k: {str(l): float(v[l] / ref[l]) for l in pick} for k, v in raw.items()},
                   pixwin2_nside4096={str(l): float(pw2[l]) for l in pick},
                   band_noise_sigma_over_B12_clean=(sigma_b / bin40(obs['B12'][0])).tolist() if 'B12' in obs else None)
    if 'FL_L1' in obs:
        ref_b = bin40(obs['FL_L1'][0])
        summary['band_ratio_to_FL_L1_clean'] = {k: (bin40(v[0]) / ref_b).tolist() for k, v in obs.items()}
        summary['band_offset_from_FL_L1_in_sigma'] = {k: ((bin40(v[0]) - ref_b) / sigma_b).tolist() for k, v in obs.items()}
    (OUT / 'spectra_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('missing:', missing or 'none')
    print('written:', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
