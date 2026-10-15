"""SOAP audit, local analysis: aperture radii in R200c, SOAP-vs-particle validation, SOAP-only stacks.

Inputs (read-only): study selected_halos.npz, soap_extra_columns.npz, apertures.npz, metadata_plan.json;
this directory's particle_cumulative.npz (owner sums), soap_study_rows.npz, soap_read_timing.json,
inventory_gas_fid0077.json, inventory_raw_quick.json.
Outputs: ../soap_audit.json, ../soap_stacks.npz, ../soap_vs_particle_stacks.png
"""
import json
import numpy as np
from scipy.constants import physical_constants, electron_mass, c as C_LIGHT, Boltzmann, proton_mass

STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/'
OUT = STUDY + 'stacking_choices_20261014/research/soap/'
MPC_CM = 3.08567758e24
MPC_M = MPC_CM / 100
MSUN_G = 1.98841e33
MSUN_KG = MSUN_G / 1e3
SIGMA_T = physical_constants['Thomson cross section'][0]
G_NEWTON = physical_constants['Newtonian constant of gravitation'][0]
YCONV = electron_mass * C_LIGHT**2 / SIGMA_T          # Y [m^2] -> integral P_e dV [J = Pa m^3]

H = dict(np.load(STUDY + 'selected_halos.npz'))
EX = dict(np.load(STUDY + 'soap_extra_columns.npz'))
APX = np.load(STUDY + 'apertures.npz')['apertures']     # [n, (R200c, R500c, 5R500c, sphere), (all, bound), 7]
PLAN = json.load(open(STUDY + 'metadata_plan.json'))
FB = float(PLAN['baryon_fraction'])
PC = np.load(OUT + 'particle_cumulative.npz')
SR = np.load(OUT + 'soap_study_rows.npz')
TIM = json.load(open(OUT + 'soap_read_timing.json'))
N = len(H['m200c_msun'])
assert np.array_equal(EX['soap_row'], H['soap_row'])
order = np.searchsorted(SR['soap_row_sorted'], H['soap_row'])
assert np.array_equal(SR['soap_row_sorted'][order], H['soap_row'])

M200 = H['m200c_msun']; R200 = H['r_200_crit_mpc']; LOGM = np.log10(M200)
W = H['weight']; BIN = H['mass_bin']; EDGES = H['mass_edges']
P200 = G_NEWTON * (M200 * MSUN_KG) * (M200 * MSUN_KG / (4 * np.pi / 3 * (R200 * MPC_M)**3)) * FB / (2 * R200 * MPC_M)


def soap(col):
    """SOAP column at the study rows (halo order), converted to physical Msun / Mpc / Mpc^2 / K."""
    v = SR[col.replace('/', '__')][order]
    meta = TIM['columns'][col]
    f = meta['cgs']
    if col.endswith('SORadius'):
        return v * f / MPC_CM
    if 'ComptonY' in col:
        return v * f / MPC_CM**2
    if col.endswith('GasMass'):
        return v * f / MSUN_G
    return v * f


# ------------------------------------------------------------------ apertures
SO_ORDER = ['2500_crit', '1000_crit', '500_crit', '200_crit', '100_crit', 'BN98', '200_mean', '50_crit', '5xR_500_crit']
KPC = [10, 30, 50, 100, 300, 500, 1000, 3000]
AP = {}
for ap in SO_ORDER:
    d = {'kind': 'SO'}
    if ap == '200_crit':
        d.update(r=R200, Y=H['y_200_mpc2'], Yn=H['y_noagn_200_mpc2'], M=H['gasmass_200_msun'], T=EX['SO__200_crit__GasTemperature'])
    elif ap == '500_crit':
        d.update(r=H['r_500_crit_mpc'], Y=H['y_500_mpc2'], Yn=H['y_noagn_500_mpc2'], M=H['gasmass_500_msun'], T=EX['SO__500_crit__GasTemperature'])
    elif ap == '5xR_500_crit':
        d.update(r=H['r_5xr500_mpc'], Y=H['y_5x500_mpc2'], Yn=H['y_noagn_5x500_mpc2'], M=H['gasmass_5x500_msun'], T=soap('SO/5xR_500_crit/GasTemperature'))
    elif ap == '2500_crit':
        d.update(r=soap('SO/2500_crit/SORadius'), Y=soap('SO/2500_crit/ComptonY'), Yn=soap('SO/2500_crit/ComptonYWithoutRecentAGNHeating'),
                 M=EX['SO__2500_crit__GasMass'] * 1e10, T=EX['SO__2500_crit__GasTemperature'])
    else:
        d.update(r=soap(f'SO/{ap}/SORadius'), Y=soap(f'SO/{ap}/ComptonY'), Yn=soap(f'SO/{ap}/ComptonYWithoutRecentAGNHeating'),
                 M=soap(f'SO/{ap}/GasMass'), T=soap(f'SO/{ap}/GasTemperature'))
    AP['SO/' + ap] = d
for k in KPC:
    AP[f'InclusiveSphere/{k}kpc'] = {'kind': 'incl', 'r': np.full(N, k / 1e3), 'M': soap(f'InclusiveSphere/{k}kpc/GasMass'),
                                     'T': soap(f'InclusiveSphere/{k}kpc/GasTemperature')}
    AP[f'ExclusiveSphere/{k}kpc'] = {'kind': 'excl', 'r': np.full(N, k / 1e3), 'M': soap(f'ExclusiveSphere/{k}kpc/GasMass')}
for d in AP.values():
    for q in ('r', 'Y', 'Yn', 'M', 'T'):
        if q in d:
            d[q] = np.asarray(d[q], np.float64)
    d['x'] = d['r'] / R200

# ------------------------------------------------------------------ particle cumulative at arbitrary x
X200 = PC['x200']                        # 68 edges
Q = list(PC['quantities'])               # count, mass, y, electrons
LX = np.log(X200[1:])


def cum_edges(shells):                   # [n, ..., S] -> [n, ..., S+1]
    z = np.zeros(shells.shape[:-1] + (1,))
    return np.concatenate([z, np.cumsum(shells, axis=-1)], axis=-1)


CUM_INC = cum_edges(PC['shells200_inc'])  # [n, 3 phases, 4 q, 68]
CUM_EXC = cum_edges(PC['shells200_exc'])


def interp_cum_many(cum, xs):
    """Particle cumulative at per-halo radii xs [n, A] (units of R200c) -> [n, A, 3, 4].

    Monotone PCHIP in ln x through the cumulative sums at the 67 X200 edges (exact at every edge);
    below the first edge (0.025 R200c) the first shell is assumed uniform (C ~ x^3); NaN beyond 4 R200c."""
    from scipy.interpolate import PchipInterpolator
    n, A = xs.shape
    out = np.full((n, A) + cum.shape[1:-1], np.nan)
    for i in range(n):
        x = xs[i]
        inside = (x >= X200[1]) & (x <= X200[-1] * (1 + 1e-9))
        if inside.any():
            P = PchipInterpolator(LX, cum[i, ..., 1:], axis=-1)
            out[i, inside] = np.moveaxis(P(np.log(np.minimum(x[inside], X200[-1]))), -1, 0)
        core = (x > 0) & (x < X200[1])
        if core.any():
            out[i, core] = cum[i, ..., 1][None] * (x[core] / X200[1]).reshape(-1, *(1,) * (cum.ndim - 2))**3
    return out


def unpack(v):
    """[.., 3 phases, 4 q] -> dict Y (all gas), Yn (no recent AGN), M, Mn, E (electrons), N (particle count)."""
    return {'Y': v[..., :, 2].sum(-1), 'Yn': v[..., 1:, 2].sum(-1), 'M': v[..., :, 1].sum(-1), 'Mn': v[..., 1:, 1].sum(-1),
            'E': v[..., :, 3].sum(-1), 'N': v[..., :, 0].sum(-1)}


def ratio_stats(a, b, mask=None):
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    if mask is not None:
        ok &= mask
    r = a[ok] / b[ok]
    if r.size == 0:
        return None
    p16, p50, p84 = np.percentile(r, [16, 50, 84])
    return {'n': int(r.size), 'median': float(p50), 'p16': float(p16), 'p84': float(p84),
            'frac_within_1pct': float(np.mean(np.abs(r - 1) < 0.01)), 'frac_within_0p3pct': float(np.mean(np.abs(r - 1) < 0.003))}


MASS_GROUPS = {'12.5-13.5': (LOGM < 13.5), '13.5-14.5': (LOGM >= 13.5) & (LOGM < 14.5), '>=14.5': LOGM >= 14.5, 'all': np.ones(N, bool)}
audit = {'n_halos': N}

# ---- interpolation error of the particle cumulative (exact apertures available at R500c, 5R500c)
interp_check = {}
XS_CHECK = np.stack([H['r_500_crit_mpc'] / R200, H['r_5xr500_mpc'] / R200], axis=1)
PV_CHECK = unpack(interp_cum_many(CUM_INC, XS_CHECK))
for j, name in enumerate(['R500c', '5R500c'], start=1):
    pv = {k: v[:, j - 1] for k, v in PV_CHECK.items()}
    interp_check[name] = {'Y': ratio_stats(pv['Y'], APX[:, j, 0, 2]), 'M': ratio_stats(pv['M'], APX[:, j, 0, 1])}
audit['particle_interpolation_check'] = interp_check

# ---- aperture radii in R200c per 0.2-dex bin
radii = {}
for name, d in AP.items():
    if d['kind'] == 'excl':
        continue
    row = []
    for b in range(len(EDGES) - 1):
        m = BIN == b
        if m.sum() == 0:
            continue
        x = d['x'][m]
        row.append({'bin': [float(EDGES[b]), float(EDGES[b + 1])], 'n': int(m.sum()), 'median': float(np.median(x)),
                    'p16': float(np.percentile(x, 16)), 'p84': float(np.percentile(x, 84))})
    radii[name] = row
audit['aperture_x_by_bin'] = radii

# ---- particle values at every SOAP aperture radius of every halo
AP_NAMES = list(AP)
XS = np.stack([AP[k]['x'] for k in AP_NAMES], axis=1)
PV_INC = unpack(interp_cum_many(CUM_INC, XS))
PV_EXC = unpack(interp_cum_many(CUM_EXC, XS))
for j, k in enumerate(AP_NAMES):
    src = PV_EXC if AP[k]['kind'] == 'excl' else PV_INC
    AP[k]['particle'] = {q: v[:, j] for q, v in src.items()}

# ---- validation SOAP vs particles (only where the particle grid resolves the radius: 0.025 <= x <= 4)
val = {}
for name, d in AP.items():
    pv = d['particle']
    ingrid = (d['x'] >= X200[1]) & (d['x'] <= X200[-1])
    entry = {}
    for g, gm in MASS_GROUPS.items():
        gm = gm & ingrid
        e = {'M': ratio_stats(d['M'], pv['M'], gm), 'median_particles_in_aperture': float(np.median(pv['N'][gm])) if gm.any() else None}
        if 'Y' in d:
            e['Y'] = ratio_stats(d['Y'], pv['Y'], gm)
            e['Ynoagn'] = ratio_stats(d['Yn'], pv['Yn'], gm)
        if 'T' in d:
            # mass-weighted-temperature Y proxy, calibrated nowhere: sigma_T k / (m_e c^2) * M T / (mu_e m_p), mu_e = 2/(1+X), X = 0.76
            mu_e = 2 / 1.76
            yproxy = SIGMA_T * Boltzmann / (electron_mass * C_LIGHT**2) * d['M'] * MSUN_KG * d['T'] / (mu_e * proton_mass) / MPC_M**2
            e['Yproxy_MT'] = ratio_stats(yproxy, pv['Yn'], gm)
            # fully ionised primordial electrons from SOAP gas mass vs particle electrons
            ne_fi = d['M'] * MSUN_KG / (mu_e * proton_mass)
            e['electrons_from_Mgas'] = ratio_stats(ne_fi, pv['E'], gm)
        entry[g] = e
    entry['x_range_median'] = [float(np.nanmin(d['x'])), float(np.nanmedian(d['x'])), float(np.nanmax(d['x']))]
    entry['frac_beyond_4R200c'] = float(np.mean(d['x'] > 4))
    entry['frac_inside_first_edge_0p025'] = float(np.mean(d['x'] < X200[1]))
    val[name] = entry
audit['validation'] = val

# ---- recent-AGN share of Y per aperture (SOAP only)
agn = {}
for name, d in AP.items():
    if 'Y' not in d:
        continue
    row = {}
    for g, gm in MASS_GROUPS.items():
        ok = gm & (d['Y'] > 0)
        row[g] = {'median_1_minus_Ynoagn_over_Y': float(np.median(1 - d['Yn'][ok] / d['Y'][ok])),
                  'weighted_mean_fraction': float(np.sum(W[ok] * (d['Y'][ok] - d['Yn'][ok])) / np.sum(W[ok] * d['Y'][ok]))}
    agn[name] = row
audit['soap_recent_agn_fraction'] = agn
json.dump(audit, open(OUT + 'soap_audit_part1.json', 'w'), indent=1)
print('part 1 done')


# =================================================================== part 2: SOAP-only stacks
YNORM = MPC_M**2 * YCONV / (P200 * (R200 * MPC_M)**3)     # Y [Mpc^2] -> integral P_e dV / (P200 R200c^3)
MNORM = 1.0 / (FB * M200)                                  # M_gas -> M_gas / (f_b M200c)
SO_KEYS = ['SO/' + a for a in SO_ORDER]
INC_KEYS = [f'InclusiveSphere/{k}kpc' for k in KPC]


def wmean(v, w):
    ok = np.isfinite(v)
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def fine_stack(rows, qidx, norm, phases):
    """Weighted-mean normalised cumulative on the 68 X200 edges (particle 'truth')."""
    c = CUM_INC[rows][:, phases, qidx, :].sum(1) * norm[rows, None]
    return np.einsum('i,ij->j', W[rows], c) / W[rows].sum()


def eval_fine(cbar, x):
    from scipy.interpolate import PchipInterpolator
    x = np.atleast_1d(np.asarray(x, float))
    out = np.where(x < X200[1], cbar[1] * (x / X200[1])**3, np.nan)
    ins = (x >= X200[1]) & (x <= X200[-1])
    out[ins] = PchipInterpolator(LX, cbar[1:])(np.log(x[ins]))
    return out


def merge_points(keys, xbar, min_sep):
    """Order apertures by mean x and drop any within a fractional separation `min_sep` of the previous kept one."""
    o = np.argsort(xbar)
    kept = []
    for j in o:
        if kept and xbar[j] < xbar[kept[-1]] * (1 + min_sep):
            continue
        kept.append(j)
    return [keys[j] for j in kept], [j for j in kept]


def soap_stack(b, field, keys, min_sep=0.05, xmax=4.0):
    """Coarse SOAP-only shells for mass bin b. field 'P' (ComptonY without recent AGN, SO only) or 'rho' (gas mass)."""
    rows = np.where(BIN == b)[0]
    w = W[rows]
    if field == 'P':
        norm, qidx, phases, val, pkey, shell_norm = YNORM, 2, [1, 2], 'Yn', 'Yn', 4 * np.pi / 3
    else:
        norm, qidx, phases, val, pkey, shell_norm = MNORM, 1, [0, 1, 2], 'M', 'M', 1.0
    xbar = np.array([wmean(AP[k]['x'][rows], w) for k in keys])
    use = xbar <= xmax
    keys_u = [k for k, u in zip(keys, use) if u]
    kept, idx = merge_points(keys_u, xbar[use], min_sep)
    xb = xbar[use][idx]
    cs = np.array([wmean(AP[k][val][rows] * norm[rows], w) for k in kept])                 # SOAP stacked cumulative
    cp = np.array([wmean(AP[k]['particle'][pkey][rows] * norm[rows], w) for k in kept])     # particles at the same per-halo radii
    cbar = fine_stack(rows, qidx, norm, phases)
    cf = eval_fine(cbar, xb)                                                               # fine particle stack at the mean radii
    # per-halo shells (estimator B), SO apertures only and only where each halo's radii are increasing
    xi = np.stack([AP[k]['x'][rows] for k in kept], 1)
    vi = np.stack([AP[k][val][rows] * norm[rows] for k in kept], 1)
    x0 = np.concatenate([np.zeros((len(rows), 1)), xi], 1)
    v0 = np.concatenate([np.zeros((len(rows), 1)), vi], 1)
    dv3 = np.diff(x0**3, axis=1)
    per = np.where(dv3 > 0, np.diff(v0, axis=1) / (shell_norm * np.where(dv3 > 0, dv3, 1)), np.nan)
    estB = np.array([wmean(per[:, j], w) for j in range(per.shape[1])])
    e = np.concatenate([[0.0], xb])
    d3 = np.diff(e**3) * shell_norm
    estA = np.diff(np.concatenate([[0.0], cs])) / d3
    estA_p = np.diff(np.concatenate([[0.0], cp])) / d3
    truth = np.diff(np.concatenate([[0.0], cf])) / d3
    # fine-grid differential profile for plotting
    fine_x = np.sqrt(X200[1:] * X200[:-1]); fine_x[0] = X200[1] / 2
    fine = np.diff(cbar) / (np.diff(X200**3) * shell_norm)
    npart = np.array([np.median(AP[k]['particle']['N'][rows]) for k in kept])
    ni = np.stack([AP[k]['particle']['N'][rows] for k in kept], 1)
    nshell = np.nanmedian(np.diff(np.concatenate([np.zeros((len(rows), 1)), ni], 1), axis=1), axis=0)
    # forward-model check: mean over haloes of the fine stacked cumulative at each halo's own aperture radius
    fm = np.array([wmean(eval_fine(cbar, AP[k]['x'][rows]), w) for k in kept])
    usable = (nshell >= 50) & (e[1:] >= X200[1]) & ((e[:-1] == 0) | (e[:-1] >= X200[1]))
    return {'bin': [float(EDGES[b]), float(EDGES[b + 1])], 'n_halos': int(len(rows)), 'apertures': kept,
            'x_edges': e.tolist(), 'soap_A': estA.tolist(), 'soap_B': estB.tolist(), 'particle_same_radii_A': estA_p.tolist(),
            'particle_fine_same_interval': truth.tolist(), 'ratio_A_to_fine': (estA / truth).tolist(),
            'ratio_B_to_fine': (estB / truth).tolist(), 'ratio_A_to_particle_same_radii': (estA / estA_p).tolist(),
            'cum_soap': cs.tolist(), 'cum_fine_at_xbar': cf.tolist(), 'median_particles_inside_aperture': npart.tolist(),
            'median_particles_in_shell': nshell.tolist(), 'usable_shell': usable.tolist(),
            'forward_model_ratio': (cs / fm).tolist(), 'forward_model_ratio_particles': (cp / fm).tolist(),
            'fine_x': fine_x.tolist(), 'fine_profile': fine.tolist(), 'cum_fine_at_4R': float(cbar[-1])}


stacks = {'P': {}, 'rho_SO': {}, 'rho_SO_plus_kpc': {}}
for b in range(len(EDGES) - 1):
    if (BIN == b).sum() < 5:
        continue
    stacks['P'][b] = soap_stack(b, 'P', SO_KEYS)
    stacks['rho_SO'][b] = soap_stack(b, 'rho', SO_KEYS)
    stacks['rho_SO_plus_kpc'][b] = soap_stack(b, 'rho', SO_KEYS + INC_KEYS, min_sep=0.10)

# fraction of the particle Y(<4R200c) and M_gas(<4R200c) outside SOAP's SO coverage, per bin
coverage = {}
for b, st in stacks['P'].items():
    rows = np.where(BIN == b)[0]
    cbar = fine_stack(rows, 2, YNORM, [1, 2])
    x_in, x_out = st['x_edges'][1], st['x_edges'][-1]
    cY = eval_fine(cbar, [x_in, x_out])
    cbarM = fine_stack(rows, 1, MNORM, [0, 1, 2])
    cM = eval_fine(cbarM, [x_in, x_out])
    coverage[b] = {'bin': st['bin'], 'x_innermost_SO': x_in, 'x_outermost_SO': x_out,
                   'Y_frac_inside_innermost': float(cY[0] / cbar[-1]), 'Y_frac_beyond_outermost_to_4R': float(1 - cY[1] / cbar[-1]),
                   'M_frac_inside_innermost': float(cM[0] / cbarM[-1]), 'M_frac_beyond_outermost_to_4R': float(1 - cM[1] / cbarM[-1])}

audit2 = {'stacks': {f: {str(b): v for b, v in d.items()} for f, d in stacks.items()}, 'coverage': {str(b): v for b, v in coverage.items()}}
json.dump(audit2, open(OUT + 'soap_audit_part2.json', 'w'), indent=1)
print('part 2 done')
