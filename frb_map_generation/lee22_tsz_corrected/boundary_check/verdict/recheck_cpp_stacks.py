"""Independent re-read of the kept C++ group_particles outputs (c3fe9b8 y_prof/ne_prof, real CAMELS CV boxes,
actual box faces = translation s0). Recomputes the buggy/plain ratio of the halo-mean stacks at fixed x, without
using analyze_impact.py.  Usage: python recheck_cpp_stacks.py"""
import numpy as np, glob, os, json
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007/impact/cpp_runs'
N = 128
lnr = np.log(2.5/0.03)/(N-1)

def shell_index(x):
    return 1 + int(np.floor(np.log(x/0.03)/lnr))

def load(run, var, snap):
    d = f'{OUT}/{run}/{var}'
    M = np.fromfile(f'{d}/grp_M200c_{snap}.bin', dtype=np.float64)
    R = np.fromfile(f'{d}/grp_R200c_{snap}.bin', dtype=np.float64)
    P200 = np.fromfile(f'{d}/grp_P200c_{snap}.bin', dtype=np.float64)
    pe = np.fromfile(f'{d}/grp_pressure_prof_{snap}.bin', dtype=np.float64).reshape(-1, N)
    Mn = np.fromfile(f'{d}/grp_M200c.bin', dtype=np.float64)
    ne = np.fromfile(f'{d}/grp_ne_prof.bin', dtype=np.float64).reshape(-1, N)
    assert np.array_equal(M, Mn)
    return M, R, P200, pe, ne

res = {}
for snap, z in [('090', 0.0), ('060', 1.05), ('044', 2.0)]:
    runs = sorted(r for r in os.listdir(OUT) if r.endswith(f'_{snap}_s0'))
    for lo, hi in [(13.0, 14.2), (12.5, 13.0)]:
        acc = {k: [] for k in ['ne_b', 'ne_p', 'pe_b', 'pe_p', 'nlost']}
        nh = 0
        for run in runs:
            Mb, Rb, Pb, peb, neb = load(run, 'buggy', snap)
            Mp, Rp, Pp, pep, nep = load(run, 'plain', snap)
            assert np.array_equal(Mb, Mp)
            lm = np.log10(Mb*1e10)
            sel = (lm >= lo) & (lm < hi)
            nh += sel.sum()
            acc['ne_b'].append(neb[sel]); acc['ne_p'].append(nep[sel])
            acc['pe_b'].append(peb[sel]/Pb[sel, None]); acc['pe_p'].append(pep[sel]/Pp[sel, None])
        if nh == 0:
            continue
        ne_b = np.concatenate(acc['ne_b']); ne_p = np.concatenate(acc['ne_p'])
        pe_b = np.concatenate(acc['pe_b']); pe_p = np.concatenate(acc['pe_p'])
        naff = int(np.sum(np.any(np.abs(ne_b-ne_p) > 1e-12*np.abs(ne_p), axis=1)))
        row = {'n_halos': int(nh), 'n_affected_ne': naff}
        for x in [0.5, 1.0, 1.34, 2.5]:
            k = min(shell_index(x), N-1)
            row[f'ne_ratio_x{x}'] = float(ne_b[:, k].mean()/ne_p[:, k].mean())
            row[f'pe_ratio_x{x}'] = float(pe_b[:, k].mean()/pe_p[:, k].mean())
        res[f'z={z} logM {lo}-{hi}'] = row
        print(f'z={z} logM {lo}-{hi}', row)
json.dump(res, open('/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007/verdict/recheck_cpp_stacks.json', 'w'), indent=1)
