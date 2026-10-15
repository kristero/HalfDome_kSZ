"""Column list for the study-row SOAP read (fiducial 0077) and its byte budget (from the inventory)."""
SO_NEW = ['50_crit', '100_crit', '1000_crit', '200_mean', 'BN98']
KPC = ['10kpc', '30kpc', '50kpc', '100kpc', '300kpc', '500kpc', '1000kpc', '3000kpc']
COLUMNS = []
for ap in SO_NEW:
    COLUMNS += [f'SO/{ap}/{q}' for q in ('SORadius', 'ComptonY', 'ComptonYWithoutRecentAGNHeating', 'GasMass', 'GasTemperature')]
COLUMNS += ['SO/2500_crit/SORadius', 'SO/2500_crit/ComptonY', 'SO/2500_crit/ComptonYWithoutRecentAGNHeating',
            'SO/5xR_500_crit/GasTemperature']
COLUMNS += [f'InclusiveSphere/{k}/{q}' for k in KPC for q in ('GasMass', 'GasTemperature')]
COLUMNS += [f'ExclusiveSphere/{k}/GasMass' for k in KPC]

if __name__ == '__main__':
    import json, numpy as np
    OUT = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'
    STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/'
    inv = json.load(open(OUT + 'inventory_raw_fid0077.json'))['datasets']
    rows = np.sort(np.load(STUDY + 'selected_halos.npz')['soap_row'])
    nrow = 15329653
    tot_full = tot_touch = 0
    for c in COLUMNS:
        v = inv[c]; ch = v['chunks'][0]; nch = -(-nrow // ch)
        touched = len(np.unique(rows // ch))
        tot_full += v['storage_bytes']; tot_touch += v['storage_bytes'] * touched / nch
    print(len(COLUMNS), 'columns; full compressed', tot_full / 1e6, 'MB; touched by study rows ~', tot_touch / 1e6, 'MB')
