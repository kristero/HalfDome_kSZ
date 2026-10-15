"""Beam-consistent SO noise for the tSZ_64k rows, built and analysed in stages of 8,000 training rows (on idark).

Why: the frozen engine beams the signal with the 2' Gaussian beam but adds the SO Deproj-0 noise, which is already
beam-deconvolved, without that beam (observation.toml: noise_beam_applied = false). Per row, the stored production
spectrum is the masked split cross spectrum

    C_noisy = C_ss + C_sn + C_nn,    C_sn = C(ms, mn2) + C(mn1, ms),   C_nn = C(mn1, mn2),

and C_ss (masked clean) is stored too. regen_noise.jl regenerates the row's two noise maps from its seeds exactly as the
engine does and computes C_nn, which isolates C_sn = C_noisy - C_ss - C_nn. Beaming the noise multiplies C_sn by B_ell
and C_nn by B_ell^2, so the beam-consistent spectra are

    beam:  C_ss + B C_sn + B^2 C_nn               each split at full survey depth, as in production
    half:  C_ss + sqrt(2) B C_sn + 2 B^2 C_nn     each split at half the survey depth (twice the noise power)

The idark pilot (beam_pilot_analysis.py) compared these with spectra of beamed-noise maps: the error is at most 0.3 % of
the noise term per band, and B^2 C_nn is as accurate as the exact beamed noise cross.

Stages (from results_final/split.npz, so that every N is the same row set as in results_final):
    stage 1   the 3,000 test rows + training rows order[0:8000]
    stage k   training rows order[8000 (k-1) : 8000 k]           (stage 8: the last 5,440; 61,440 in all)
After stage k the NPE ensembles are trained at N = 8000 k on the uncorrected and the corrected spectra of the same rows,
with the same settings as results_final (3 seeds, logit-space MAF, early stopping). Every feature transform is fitted
on the stage-1 training rows and then frozen, so the feature vector of a row never changes between stages.

Variants (network sizes as in sbi_final.VARIANTS):
    bins40_noisy  40 bands of the production spectrum (the uncorrected reference, trained here with the same transform)
    bins40_beam   40 bands of the beam-consistent spectrum, full depth
    bins40_half   40 bands of the beam-consistent spectrum, half-depth splits
    bins40_clean  40 bands without noise (upper bound on any noise reduction)
    moped_noisy   the fixed B12 MOPED matrix of the 256-row test applied to the production spectrum
    moped_beam    the same matrix applied to the beam-consistent spectrum (the matrix is that of the uncorrected noise)

Layout of the run root (--run, default the folder of this script's parent):
    inputs/            uploaded once (see INPUT_FILES)
    tasks/stageKK.txt  regen_noise.jl task lists, written by plan
    nn/, claims/       regen_noise.jl outputs and claims
    upload/stageKK/bNNN/   relay.py uploads: rows.npy, noisy/<row>.npy, obs/<row>.toml, SHA256SUMS, UPLOADED, and
                           clean_dl_unbinned.npy (cut from inputs/clean_dl_unbinned.npy by extract-clean; batches
                           uploaded before 2026-10-01 11:00 carry it from the relay)
    stages/stageKK/    build outputs
    series/<variant>/N<size>/   features, networks, metrics (train_ensemble.py layout)
    results/           report outputs: the only files fetched back

Commands
    python beamfix.py plan
    python beamfix.py status
    python beamfix.py extract-clean DIR      (relay.py, on idark: DIR/clean_dl_unbinned.npy for the rows in DIR/rows.npy)
    python beamfix.py build K
    python beamfix.py series K [--workers 6 --threads 4 --variants ...]
    python beamfix.py report
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                     # sbi_final.py, train_ensemble.py, analyze_64k.py are copied next to it

STAGE_ROWS = int(os.environ.get('BEAMFIX_STAGE_ROWS', 8000))   # training rows added per stage (env: small tests)
ELL = np.arange(80, 7980)                         # multipoles of the analysis (unbinned arrays are ell = 80..7979)
DL = ELL * (ELL + 1) / (2 * np.pi)                # C_ell -> D_ell
EDGES = np.r_[np.arange(80, 7881, 200), 7980]     # 40 bands of 200 multipoles, weights 2 ell + 1
QA_EVERY = 64                                     # rows with design row % QA_EVERY == 0 also get the exact beamed C_nn
HASH_EVERY = 16                                   # rows with design row % HASH_EVERY == 0: noise hashes vs production
                                                  # (relay.py fetches observation.toml for exactly these rows)
HIGH_ELL = ELL >= 6000                            # multipoles of the noise-pair check (where C_nn is largest)
SLOPE_TESTABLE_SD = 0.1                           # the slope check is a hard check for rows whose slope sd is below this
SLOPE_FAIL = 0.5                                  # ... and fails a row whose slope is below this (right pair 1, wrong 0)
SLOPE_MEDIAN_TOLERANCE = 0.01                     # median |slope - 1| over the testable rows of a stage (expected ~0.0013;
                                                  # a uniform C_nn scale error eps gives ~eps)
B12_FITTED_PAIR = (39001001, 39001002)            # noise seeds of the B12 observation (the fitted SO mock)
VALIDATION_ROWS = {38694: 'row38694_faint', 18986: 'row18986_typical', 28448: 'row28448_bright'}   # beam pilot rows
VARIANTS = dict(bins40_noisy=dict(hidden=128, transforms=5), bins40_beam=dict(hidden=128, transforms=5),
                bins40_half=dict(hidden=128, transforms=5), bins40_clean=dict(hidden=128, transforms=5),
                moped_noisy=dict(hidden=64, transforms=5), moped_beam=dict(hidden=64, transforms=5))
SEEDS = ['1', '2', '3']
EPOCHS = '2000'                                   # epoch cap of sbi_final train (early stopping decides)
# inputs/ (uploaded once by driver.sh):
INPUT_FILES = dict(
    split='split.npz',                            # results_final/split.npz: test, order (positions in rows_64440), row_id
    theta='theta.npy', clean40='clean_dl.npy', noisy40='noisy_dl.npy',   # data/rows_64440 small arrays (checks)
    moped_raw_noisy='moped_raw_noisy.npy', meta='meta.json', row_id='row_id.npy',
    seeds='noise_seeds_65536.npy',                # design seeds (bundle design/), row r -> (seed1, seed2)
    moped_matrix='moped_matrix.npy',              # compression_context.npz['Battaglia12_rcond1e-06'], (7900, 9)
    b12_obs='Battaglia12.npz',                    # the B12 observation (clean/noisy D_ell, unbinned and 40 bands)
    b12_mocks='b12_mocks',                        # mocks/pass1_B12/B12: masked_clean_cl.npy, noisy_cross_<s1>_<s2>.npy
    reference='results_final_metrics.json',       # results_final metrics of every variant and N (report overlay)
    clean_unbinned='clean_dl_unbinned.npy')       # rows_65536 masked clean D_ell (65536, 7900), ell 80..7979: a link to
                                                  # ~/tsz64k_rows_65536/clean_dl_unbinned.npy (extract-clean reads it)


def run_root(args):
    return Path(args.run).resolve()


def rebin40(dl):
    """40 bands (weights 2 ell + 1) of D_ell arrays (..., 7900) on ELL."""
    w = 2 * ELL + 1
    out = [np.sum(dl[..., a - 80:b - 80] * w[a - 80:b - 80], axis=-1) / np.sum(w[a - 80:b - 80])
           for a, b in zip(EDGES[:-1], EDGES[1:])]
    return np.stack(out, axis=-1)


def load_inputs(root):
    """The small inputs as a dict; 'row_of_position' maps a rows_64440 position to its design row."""
    d = root / 'inputs'
    split = np.load(d / INPUT_FILES['split'])
    row_id = np.load(d / INPUT_FILES['row_id'])
    assert np.array_equal(split['row_id'], row_id), 'split.npz row_id differs from rows_64440 row_id'
    meta = json.loads((d / INPUT_FILES['meta']).read_text())
    return dict(test=split['test'], order=split['order'], row_of_position=row_id,
                theta=np.load(d / INPUT_FILES['theta']), clean40=np.load(d / INPUT_FILES['clean40']),
                noisy40=np.load(d / INPUT_FILES['noisy40']), moped_raw_noisy=np.load(d / INPUT_FILES['moped_raw_noisy']),
                seeds=np.load(d / INPUT_FILES['seeds']), matrix=np.load(d / INPUT_FILES['moped_matrix']),
                low=np.array(meta['lower']), high=np.array(meta['upper']))


def stage_positions(inp, k):
    """rows_64440 positions of stage k: (test positions or empty, training positions), both in stage order."""
    order = inp['order']
    stages = n_stages(inp)
    assert 1 <= k <= stages, f'stage {k} outside 1..{stages}'
    train = order[STAGE_ROWS * (k - 1):STAGE_ROWS * k]
    test = inp['test'] if k == 1 else np.array([], dtype=int)
    return test, train


def n_stages(inp):
    return int(np.ceil(len(inp['order']) / STAGE_ROWS))


def stage_size(inp, k):
    """Training-set size N after stage k."""
    return min(STAGE_ROWS * k, len(inp['order']))


def row_label(row):
    return f'r{int(row):05d}'


def b12_pairs(root):
    """(s1, s2) of the B12 SO mocks, the fitted observation first."""
    pairs = []
    for p in sorted((root / 'inputs' / INPUT_FILES['b12_mocks']).glob('noisy_cross_*.npy')):
        s1, s2 = (int(v) for v in re.match(r'noisy_cross_(\d+)_(\d+)\.npy', p.name).groups())
        pairs.append((s1, s2))
    pairs.sort(key=lambda p: p != B12_FITTED_PAIR)
    assert pairs[0] == B12_FITTED_PAIR and len(pairs) == 21, pairs
    return pairs


# ----------------------------------------------------------------------------- plan

def cmd_plan(args):
    """Write tasks/stageKK.txt ("<label> <seed1> <seed2> <qa> <hash>") and plan.json.

    Stage 1 starts with the 21 B12 mocks (qa) and the 3 pilot rows (qa; their exact beamed spectra are known), then the
    test rows, then its training rows. Every row with design row % QA_EVERY == 0 is a qa task; rows with design
    row % HASH_EVERY == 0, the B12 mocks and the pilot rows are hash tasks.
    """
    root = run_root(args)
    inp = load_inputs(root)
    (root / 'tasks').mkdir(parents=True, exist_ok=True)
    seeds = inp['seeds']
    plan = dict(stage_rows=STAGE_ROWS, stages=[])
    for k in range(1, n_stages(inp) + 1):
        test, train = stage_positions(inp, k)
        rows = inp['row_of_position'][np.r_[test, train].astype(int)]
        lines = []
        if k == 1:
            lines += [f'B12_{s1}_{s2} {s1} {s2} 1 1' for s1, s2 in b12_pairs(root)]
            lines += [f'{row_label(r)} {seeds[r, 0]} {seeds[r, 1]} 1 1' for r in VALIDATION_ROWS]
        lines += [f'{row_label(r)} {seeds[r, 0]} {seeds[r, 1]} {int(r % QA_EVERY == 0)} {int(r % HASH_EVERY == 0)}'
                  for r in rows]
        (root / 'tasks' / f'stage{k:02d}.txt').write_text('\n'.join(lines) + '\n')
        np.save(root / 'tasks' / f'stage{k:02d}_rows.npy', rows)
        plan['stages'].append(dict(stage=k, test_rows=int(len(test)), train_rows=int(len(train)),
                                   N_after=stage_size(inp, k), tasks=len(lines)))
    (root / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    print(json.dumps(plan, indent=1))


# ----------------------------------------------------------------------------- status

def read_tasks(root, k):
    """{label: (qa, hash)} of tasks/stageKK.txt."""
    tasks = {}
    for line in (root / 'tasks' / f'stage{k:02d}.txt').read_text().split('\n'):
        fields = line.split('#')[0].split()
        if fields:
            tasks[fields[0]] = (fields[3] == '1', fields[4] == '1')
    return tasks


def task_done(root, label, qa, want_hash):
    """The completion rule of regen_noise.jl task_done: the .toml, plus the beamed spectrum (qa) and hashes (hash)."""
    path = root / 'nn' / f'{label}.toml'
    if not path.exists():
        return False
    if qa and not (root / 'nn' / f'{label}_beamed.npy').exists():
        return False
    return not want_hash or toml_hashes(path) is not None


def all_tasks(root):
    tasks = {}
    for p in sorted((root / 'tasks').glob('stage*.txt')):
        tasks.update(read_tasks(root, int(p.stem[5:])))
    return tasks


def stage_state(root, k):
    rows = np.load(root / 'tasks' / f'stage{k:02d}_rows.npy')
    tasks = read_tasks(root, k)
    done = sum(task_done(root, label, *flags) for label, flags in tasks.items())
    failed = [p.stem for p in (root / 'nn').glob('*.failed') if p.stem in tasks] if (root / 'nn').exists() else []
    labels = list(tasks)
    uploaded = set()
    for batch in sorted((root / 'upload' / f'stage{k:02d}').glob('b*')):
        if (batch / 'UPLOADED').exists():
            uploaded.update(np.load(batch / 'rows.npy').tolist())
    return dict(stage=k, tasks=len(labels), regen_done=done, rows=len(rows), failed_tasks=failed,
                uploaded=len(uploaded & set(rows.tolist())),
                built=(root / 'stages' / f'stage{k:02d}' / 'qa.json').exists(),
                series_done=all((root / d / f'N{k}.done').exists() for d in SERIES_DIRS.values()),
                series_median_done=(root / 'series' / f'N{k}.done').exists(),
                series_noise_done=(root / 'series_noise' / f'N{k}.done').exists())


def cmd_status(args):
    root = run_root(args)
    inp = load_inputs(root)
    states = [stage_state(root, k) for k in range(1, n_stages(inp) + 1)]
    tasks = all_tasks(root)
    pending = [p.name for p in (root / 'claims').glob('*')
               if p.name in tasks and not task_done(root, p.name, *tasks[p.name])] if (root / 'claims').exists() else []
    done = len(list((root / 'nn').glob('*.toml'))) if (root / 'nn').exists() else 0
    print(json.dumps(dict(stages=states, pending_claims=len(pending), regen_done_total=done), indent=1))


def cmd_extract_clean(args):
    """Write DIR/clean_dl_unbinned.npy, the masked clean D_ell (rows, 7900) of the design rows in DIR/rows.npy, from
    the idark copy of rows_65536/clean_dl_unbinned.npy (row r of that file is design row r). build checks the result
    against the rows_64440 band powers."""
    root = run_root(args)
    folder = Path(args.folder)
    rows = np.load(folder / 'rows.npy')
    clean = np.load(root / 'inputs' / INPUT_FILES['clean_unbinned'], mmap_mode='r')
    assert clean.shape == (65536, 7900), clean.shape
    temporary = folder / 'clean_dl_unbinned.npy.tmp'
    with open(temporary, 'wb') as stream:
        np.save(stream, np.asarray(clean[rows], dtype=np.float64))
    temporary.replace(folder / 'clean_dl_unbinned.npy')


def cmd_reclaim(args):
    """Release claims of tasks without output (nn/<label>.toml) left by workers that died mid-task.

    A claim is released when it is older than --min-age seconds (so a claim whose owner file is still being written is
    never touched) and either its owner job is not in --alive (PBS job ids of the running regen workers, passed by the
    driver) or it is older than --max-age (a dead process inside a job that is still running; a task takes at most a
    few minutes). The claim's age is the modification time of its owner file, or of the folder if that is missing."""
    root = run_root(args)
    alive = set(args.alive)
    tasks = all_tasks(root)
    now = time.time()
    removed = []
    for folder in sorted((root / 'claims').glob('*')):
        if folder.name not in tasks or task_done(root, folder.name, *tasks[folder.name]):
            continue
        owner_file = folder / 'owner'
        try:
            owner = owner_file.read_text().strip() if owner_file.exists() else ''
            age = now - (owner_file if owner_file.exists() else folder).stat().st_mtime
        except FileNotFoundError:                                       # released by a worker meanwhile
            continue
        job = owner.split('@')[0].split('.')[0]
        if age > args.min_age and (job not in alive or age > args.max_age):
            for tmp in (root / 'nn').glob(f'{folder.name}*.tmp.*'):       # before the claim goes: a new claimant's
                tmp.unlink(missing_ok=True)                                 # files cannot be caught
            for p in folder.iterdir():
                p.unlink(missing_ok=True)
            folder.rmdir()
            removed.append(folder.name)
    print(json.dumps(dict(removed=removed)))


# ----------------------------------------------------------------------------- build

def load_nn(root, label, beamed=False):
    """D_ell (7900,) of the regenerated noise cross C_nn (beamed: C_bb) of task `label`."""
    cl = np.load(root / 'nn' / f'{label}{"_beamed" if beamed else ""}.npy')
    assert cl.shape == (7980,) and np.isfinite(cl).all()
    return cl[ELL] * DL


def correct(clean, noisy, nn, bl):
    """Beam-consistent D_ell from clean (C_ss), production noisy (C_noisy) and regenerated nn (C_nn), all (..., 7900)
    D_ell on ELL, and B_ell (7900,). Returns (beam, half, sn): full-depth and half-depth spectra and C_sn."""
    sn = noisy - clean - nn
    beam = clean + bl * sn + bl ** 2 * nn
    half = clean + np.sqrt(2) * bl * sn + 2 * bl ** 2 * nn
    return beam, half, sn


def noise_slope(clean, noisy, nn):
    """Does the regenerated C_nn belong to the production spectrum? Per row, over HIGH_ELL:

        slope = sum(d nn) / sum(nn^2),   d = noisy - clean = C_sn + C_nn (production),

    i.e. the regression coefficient of the production excess on the regenerated noise cross. It is 1 + <C_sn, nn> /
    <nn, nn>: about 1 when nn is the production C_nn and about 0 when it is an unrelated realization, whatever the signal
    strength (a variance ratio would not do: for bright rows C_sn dominates even at ell >= 6000). Returns (slope,
    residual variance ratio var(d - slope nn) / var(nn)); the slope's sd is sqrt(that ratio / n_eff), see slope_neff."""
    d = noisy[..., HIGH_ELL] - clean[..., HIGH_ELL]
    m = nn[..., HIGH_ELL]
    slope = np.sum(d * m, axis=-1) / np.sum(m * m, axis=-1)
    resid = np.var(d - slope[..., None] * m, axis=-1) / np.var(m, axis=-1)
    return slope, resid


def slope_neff(root):
    """Effective number of independent multipoles in the slope, from the 21 B12 mocks: the slope of mock i's excess on
    mock j's regenerated C_nn (i != j, unrelated realizations) has sd sqrt(var(d_i) / var(nn_j) / n_eff)."""
    d = root / 'inputs' / INPUT_FILES['b12_mocks']
    clean = np.load(d / 'masked_clean_cl.npy')[ELL] * DL
    pairs = b12_pairs(root)
    excess = np.stack([np.load(d / f'noisy_cross_{s1}_{s2}.npy')[ELL] * DL - clean for s1, s2 in pairs])[:, HIGH_ELL]
    nn = np.stack([load_nn(root, f'B12_{s1}_{s2}') for s1, s2 in pairs])[:, HIGH_ELL]
    slopes, scale = [], []
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if i != j:
                slopes.append(np.sum(excess[i] * nn[j]) / np.sum(nn[j] ** 2))
                scale.append(np.var(excess[i]) / np.var(nn[j]))
    return float(np.mean(scale) / np.var(slopes)), dict(wrong_pair_slope_mean=float(np.mean(slopes)),
                                                        wrong_pair_slope_sd=float(np.std(slopes)))


def toml_hashes(path):
    text = Path(path).read_text()
    m = re.search(r'^noise_sha256\s*=\s*\["([0-9a-f]{64})",\s*"([0-9a-f]{64})"\]', text, re.M)
    return [m.group(1), m.group(2)] if m else None


def toml_seeds(path):
    """split_seeds of a production observation.toml, as a list of int."""
    m = re.search(r'^split_seeds\s*=\s*\[([^\]]*)\]', Path(path).read_text(), re.M)
    return [int(v) for v in m.group(1).split(',') if v.strip()] if m else None


def pilot_hashes(path, pair):
    """[hash1, hash2] of noise pair "<s1>_<s2>" in a paint_mocks_beamtest.jl case.toml ([noise_sha256] table)."""
    m = re.search(rf'^"?{pair}"?\s*=\s*\["([0-9a-f]{{64}})",\s*"([0-9a-f]{{64}})"\]', Path(path).read_text(), re.M)
    return [m.group(1), m.group(2)] if m else None


def cmd_build(args):
    """Corrected spectra of stage K from the uploads and the regenerated C_nn, with checks.

    Writes stages/stageKK/: rows.npy (design rows, stage order), {clean,noisy,beam,half}40.npy (rows, 40) band D_ell,
    beam_dl_unbinned.npy, half_dl_unbinned.npy (rows, 7900) float64, moped_raw_{noisy,beam,half}.npy (rows, 9) and
    qa.json. Refuses to run unless every row is uploaded and regenerated; raises on any failed hard check:
      - the uploaded noisy and clean spectra rebin exactly (1e-10) to the rows_64440 band powers of the same rows,
      - the production MOPED projection is reproduced (to 1e-12 of the float64 summation scale),
      - the regenerated noise pair is the production one (noise_slope): every row whose slope sd is below
        SLOPE_TESTABLE_SD has slope >= SLOPE_FAIL, and the median |slope - 1| of those rows is below
        SLOPE_MEDIAN_TOLERANCE. Brighter rows (~7 %) cannot be tested this way (their C_sn swamps C_nn); they are
        counted, and they are covered by the next check,
      - the production split_seeds of every row (its observation.toml) equal the design seeds that regen used,
      - the B12 mocks' regenerated pairs pass the slope check (build_b12).
    Informational: noise map hashes against the production observation.toml (bitwise equality needs the same CPU
    SIMD path), and for qa rows the error of B^2 C_nn against the exact beamed noise cross.
    """
    root = run_root(args)
    k = args.stage
    inp = load_inputs(root)
    rows = np.load(root / 'tasks' / f'stage{k:02d}_rows.npy')
    position = {int(r): i for i, r in enumerate(inp['row_of_position'])}
    bl_full = np.load(root / 'beam_bl.npy')
    assert bl_full.shape == (7980,)
    bl = bl_full[ELL]
    # uploaded spectra, keyed by design row
    clean, noisy, obs_hash, obs_seeds = {}, {}, {}, {}
    for batch in sorted((root / 'upload' / f'stage{k:02d}').glob('b*')):
        if not (batch / 'UPLOADED').exists():
            continue
        batch_rows = np.load(batch / 'rows.npy')
        batch_clean = np.load(batch / 'clean_dl_unbinned.npy')
        assert batch_clean.shape == (len(batch_rows), 7900)
        for i, r in enumerate(batch_rows.tolist()):
            cl = np.load(batch / 'noisy' / f'{r:05d}.npy')
            assert cl.shape == (7980,) and np.isfinite(cl).all()
            noisy[r] = cl[ELL] * DL
            clean[r] = batch_clean[i]
            t = batch / 'obs' / f'{r:05d}.toml'
            if t.exists():
                obs_hash[r] = toml_hashes(t)
                obs_seeds[r] = toml_seeds(t)
    missing_upload = [int(r) for r in rows if int(r) not in noisy]
    missing_nn = [int(r) for r in rows if not (root / 'nn' / f'{row_label(r)}.toml').exists()]
    if missing_upload or missing_nn:
        raise SystemExit(f'stage {k}: {len(missing_upload)} rows not uploaded, {len(missing_nn)} not regenerated')
    clean_a = np.stack([clean[int(r)] for r in rows])
    noisy_a = np.stack([noisy[int(r)] for r in rows])
    nn_a = np.stack([load_nn(root, row_label(r)) for r in rows])
    pos = np.array([position[int(r)] for r in rows])

    qa = dict(stage=k, rows=int(len(rows)))
    # hard check 1: the uploads are the spectra behind the rows_64440 band powers
    c40, n40 = rebin40(clean_a), rebin40(noisy_a)
    qa['clean40_vs_rows64440_max_rel'] = float(np.max(np.abs(c40 / inp['clean40'][pos] - 1)))
    scale = np.abs(inp['noisy40'][pos]) + np.abs(inp['clean40'][pos])
    qa['noisy40_vs_rows64440_max_over_scale'] = float(np.max(np.abs(n40 - inp['noisy40'][pos]) / scale))
    assert qa['clean40_vs_rows64440_max_rel'] < 1e-10 and qa['noisy40_vs_rows64440_max_over_scale'] < 1e-10, qa
    # hard check 2: the production MOPED numbers
    raw_noisy = noisy_a @ inp['matrix']
    ref = inp['moped_raw_noisy'][pos]
    rounding = np.abs(noisy_a) @ np.abs(inp['matrix'])                   # scale of the float64 summation error
    qa['moped_raw_noisy_vs_rows64440_max_over_rounding_scale'] = float(np.max(np.abs(raw_noisy - ref) / rounding))
    assert qa['moped_raw_noisy_vs_rows64440_max_over_rounding_scale'] < 1e-12, qa
    # hard check: the production seeds (observation.toml, fetched for every row) are the design seeds used by regen
    seed_bad = [r for r in rows.tolist() if obs_seeds.get(int(r)) != [int(v) for v in inp['seeds'][int(r)]]]
    qa['production_seeds_equal_design'] = dict(checked=len(obs_seeds), mismatched_or_missing=len(seed_bad))
    assert not seed_bad, f'stage {k}: production seeds missing or different from the design for rows {seed_bad[:10]}'
    # hard check 3: the regenerated C_nn is the production noise pair
    n_eff, wrong = slope_neff(root)
    slope, resid = noise_slope(clean_a, noisy_a, nn_a)
    sd = np.sqrt(resid / n_eff)
    testable = sd < SLOPE_TESTABLE_SD
    bad = rows[testable & (slope < SLOPE_FAIL)]
    median_dev = float(np.median(np.abs(slope[testable] - 1))) if testable.any() else None
    qa['noise_slope'] = dict(n_eff=n_eff, **wrong, testable_rows=int(testable.sum()),
                             untestable_rows=int((~testable).sum()), testable_median_abs_dev=median_dev,
                             testable_min=float(slope[testable].min()) if testable.any() else None,
                             untestable_median=float(np.median(slope[~testable])) if (~testable).any() else None,
                             failed_rows=bad.tolist()[:100])
    if len(bad) or median_dev is None or median_dev > SLOPE_MEDIAN_TOLERANCE:
        (root / 'stages').mkdir(exist_ok=True)
        (root / 'stages' / f'stage{k:02d}_failed_rows.json').write_text(json.dumps(qa['noise_slope'], indent=1))
        raise SystemExit(f'stage {k}: noise-pair check failed: {len(bad)} rows with slope < {SLOPE_FAIL}, '
                         f'median |slope - 1| {median_dev} (see stages/stage{k:02d}_failed_rows.json)')
    # informational: noise hashes against production
    same = [toml_hashes(root / 'nn' / f'{row_label(r)}.toml') == h for r, h in obs_hash.items()
            if toml_hashes(root / 'nn' / f'{row_label(r)}.toml') is not None]              # hash tasks only
    qa['noise_hash_equal_production'] = dict(checked=len(same), equal=int(np.sum(same)))
    # informational: B^2 C_nn against the exact beamed noise cross (qa rows)
    beam, half, sn = correct(clean_a, noisy_a, nn_a, bl)
    errs = []
    for i, r in enumerate(rows):
        label = row_label(r)
        if (root / 'nn' / f'{label}_beamed.npy').exists():
            bb = load_nn(root, label, beamed=True)
            noise = rebin40(beam[i] - clean_a[i])                   # the noise terms of the corrected spectrum
            errs.append(float(np.max(np.abs(rebin40(bl ** 2 * nn_a[i] - bb))) / np.sqrt(np.mean(noise ** 2))))
    qa['b2_approximation_max_err_over_noise_rms'] = dict(rows=len(errs), max=float(np.max(errs)) if errs else None,
                                                         median=float(np.median(errs)) if errs else None)
    # band-level size of the correction
    b40, h40 = rebin40(beam), rebin40(half)
    noise_old = np.sqrt(np.mean((n40 - c40) ** 2, axis=0))
    qa['band_noise_rms_ratio_beam_over_production'] = (np.sqrt(np.mean((b40 - c40) ** 2, axis=0)) / noise_old).tolist()
    qa['band_noise_rms_ratio_half_over_production'] = (np.sqrt(np.mean((h40 - c40) ** 2, axis=0)) / noise_old).tolist()

    out = root / 'stages' / f'stage{k:02d}'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'rows.npy', rows)
    for name, value in dict(clean40=c40, noisy40=n40, beam40=b40, half40=h40,
                            moped_raw_noisy=raw_noisy, moped_raw_beam=beam @ inp['matrix'],
                            moped_raw_half=half @ inp['matrix']).items():
        np.save(out / f'{name}.npy', value)
    np.save(out / 'beam_dl_unbinned.npy', beam)
    np.save(out / 'half_dl_unbinned.npy', half)
    if k == 1:
        qa['b12'] = build_b12(root, bl, out, n_eff)
        qa['pilot_rows'] = check_pilot_rows(root, bl)
    (out / 'qa.json').write_text(json.dumps(qa, indent=1) + '\n')
    print(json.dumps({key: v for key, v in qa.items() if not key.startswith('band_')}, indent=1))


def build_b12(root, bl, out, n_eff):
    """Corrected B12 observation and its 21 SO mocks -> out/b12.npz; returns the qa record.

    The observation is the fitted mock (seeds B12_FITTED_PAIR); its noisy spectrum equals Battaglia12.npz to 1e-13.
    For all 21 mocks the exact beamed noise cross is known (qa tasks), which gives the correction error at B12."""
    d = root / 'inputs' / INPUT_FILES['b12_mocks']
    clean = np.load(d / 'masked_clean_cl.npy')[ELL] * DL
    obs = np.load(root / 'inputs' / INPUT_FILES['b12_obs'])
    rec = dict(obs_clean_vs_mock_max_rel=float(np.max(np.abs(obs['clean_dl_unbinned'] / clean - 1))))
    noisy, nn, bb = [], [], []
    for s1, s2 in b12_pairs(root):
        noisy.append(np.load(d / f'noisy_cross_{s1}_{s2}.npy')[ELL] * DL)
        nn.append(load_nn(root, f'B12_{s1}_{s2}'))
        has_beamed = (root / 'nn' / f'B12_{s1}_{s2}_beamed.npy').exists()      # qa tasks: always, unless seeded
        bb.append(load_nn(root, f'B12_{s1}_{s2}', beamed=True) if has_beamed else np.full(7900, np.nan))
    noisy, nn, bb = np.stack(noisy), np.stack(nn), np.stack(bb)
    rec['obs_noisy_vs_fitted_mock_max_over_max'] = float(np.max(np.abs(obs['noisy_dl_unbinned'] - noisy[0]))
                                                         / np.max(np.abs(noisy[0])))
    beam, half, sn = correct(clean[None], noisy, nn, bl)
    slope, resid = noise_slope(clean[None], noisy, nn)
    rec['noise_slope'] = slope.tolist()
    # hard check: the regenerated noise pairs are those of the B12 mocks (the observation of every variant)
    assert np.all(np.abs(slope - 1) < np.maximum(5 * np.sqrt(resid / n_eff), 0.02)), rec['noise_slope']
    noise = rebin40(beam - clean[None])
    err = np.max(np.abs(rebin40(bl ** 2 * nn - bb)), axis=1) / np.sqrt(np.mean(noise ** 2, axis=1))
    rec['b2_approximation_max_err_over_noise_rms'] = [None if np.isnan(v) else float(v) for v in err]
    np.savez(out / 'b12.npz', clean_dl_unbinned=clean, noisy_dl_unbinned=noisy, beam_dl_unbinned=beam,
             half_dl_unbinned=half, clean40=rebin40(clean), noisy40=rebin40(noisy), beam40=rebin40(beam),
             half40=rebin40(half), pairs=np.array(b12_pairs(root)))
    return rec


def check_pilot_rows(root, bl):
    """The 3 pilot rows: corrected spectrum from regen + production files vs the pilot's exact beamed-noise spectra."""
    pilot = Path(os.environ.get('BEAM_PILOT_OUT', Path.home() / 'tsz64k_beamtest_20260930' / 'out'))
    rec = {}
    for row, label in VALIDATION_ROWS.items():
        folder = pilot / label
        exact = sorted(folder.glob('noisy_cross_beamed_[0-9]*.npy'))
        if not exact:
            rec[label] = 'pilot output not found'
            continue
        clean = np.load(folder / 'masked_clean_cl.npy')[ELL] * DL
        noisy_path = sorted(folder.glob('noisy_cross_[0-9]*.npy'))[0]
        pair = noisy_path.stem.replace('noisy_cross_', '')
        noisy = np.load(noisy_path)[ELL] * DL
        nn = load_nn(root, row_label(row))
        beam, half, _ = correct(clean, noisy, nn, bl)
        exact_beam = np.load(exact[0])[ELL] * DL
        exact_half = np.load(sorted(folder.glob('noisy_cross_beamed_half_*.npy'))[0])[ELL] * DL
        pilot_nn = np.load(sorted(folder.glob('noise_cross_[0-9]*.npy'))[0])[ELL] * DL
        noise = rebin40(exact_beam - clean)
        rms = np.sqrt(np.mean(noise ** 2))
        rec[label] = dict(nn_vs_pilot_max_over_max=float(np.max(np.abs(nn - pilot_nn)) / np.max(np.abs(pilot_nn))),
                          beam_err_over_noise_rms=float(np.max(np.abs(rebin40(beam - exact_beam))) / rms),
                          half_err_over_noise_rms=float(np.max(np.abs(rebin40(half - exact_half)))
                                                        / np.sqrt(np.mean(rebin40(exact_half - clean) ** 2))),
                          hashes_equal_pilot=pilot_hashes(folder / 'case.toml', pair)
                          == toml_hashes(root / 'nn' / f'{row_label(row)}.toml'))
    return rec


# ----------------------------------------------------------------------------- series

SERIES_DIRS = dict(median='series', noise='series_noise')   # one folder per asinh-scale choice of the features


def stage_arrays(root, stages):
    """Concatenated build outputs of stages 1..stages, keyed by array name, plus 'rows' (design rows)."""
    parts = {}
    for k in range(1, stages + 1):
        folder = root / 'stages' / f'stage{k:02d}'
        if not (folder / 'qa.json').exists():
            raise SystemExit(f'stage {k} is not built')
        for name in ('rows', 'clean40', 'noisy40', 'beam40', 'half40', 'moped_raw_noisy', 'moped_raw_beam'):
            parts.setdefault(name, []).append(np.load(folder / f'{name}.npy'))
    return {name: np.concatenate(v) for name, v in parts.items()}


def variant_inputs(arrays, b12):
    """{variant: (row inputs, B12 observation input, kind)}; kind 'bins40' or 'moped'."""
    return dict(bins40_noisy=(arrays['noisy40'], b12['noisy40'][:1], 'bins40'),
                bins40_beam=(arrays['beam40'], b12['beam40'][:1], 'bins40'),
                bins40_half=(arrays['half40'], b12['half40'][:1], 'bins40'),
                bins40_clean=(arrays['clean40'], b12['clean40'][None], 'bins40'),
                moped_noisy=(arrays['moped_raw_noisy'], None, 'moped'),
                moped_beam=(arrays['moped_raw_beam'], None, 'moped'))


def cmd_series(args):
    """Features at N = size after stage K for every variant, 3-seed training, evaluation -> <series dir>/<variant>/
    N<size>/ (series/ for --asinh-scale median, series_noise/ for noise).

    Feature transforms (sbi_final.fit_bins40, sbi_final.fit_moped) are fitted on the stage-1 training rows and frozen.
    --asinh-scale median: the asinh scale of each band is the median |clean D_ell| of the training rows (MOPED: median
    |raw|), as in results_final. noise: the band's noise sd at zero signal, N_b / sqrt(nu_b), of the variant's own noise
    model (sbi_final.fit_noise_model on its stage-1 training rows; bins40_clean takes the production noise model, as
    sbi_final does); MOPED: 1, the unit noise of the fixed B12 matrix. The median scale puts every row fainter than the
    median in the linear part of the asinh, where noise-level differences are 1e-5..1e-3 of the input spread.
    The B12 observation of each variant is the corrected fitted mock (b12.npz).
    """
    import sbi_final as SF
    root = run_root(args)
    k = args.stage
    inp = load_inputs(root)
    size = stage_size(inp, k)
    arrays = stage_arrays(root, k)
    b12 = np.load(root / 'stages' / 'stage01' / 'b12.npz')
    matrix = inp['matrix']
    position = {int(r): i for i, r in enumerate(inp['row_of_position'])}
    rows = arrays['rows']
    n_test = len(inp['test'])
    test_idx = np.arange(n_test)                                     # stage 1 starts with the test rows
    train_idx = np.arange(n_test, len(rows))
    assert len(train_idx) == size
    pos = np.array([position[int(r)] for r in rows])
    assert np.array_equal(pos[:n_test], inp['test']) and np.array_equal(pos[n_test:], inp['order'][:size])
    theta = inp['theta'][pos]
    first = train_idx[:STAGE_ROWS]                                   # stage-1 training rows: the transform fit
    variants = args.variants or list(VARIANTS)
    choice = args.asinh_scale
    base = root / SERIES_DIRS[choice]
    for name in variants:
        x_rows, x_obs40, kind = variant_inputs(arrays, b12)[name]
        record = dict(variant=name, asinh_scale=choice, fit_rows=int(len(first)))
        if kind == 'bins40':
            scale40 = None
            if choice == 'noise':
                # the variant's own noise model; the noise-free variant has none and takes the production one
                noisy_for_model = arrays['noisy40'] if name == 'bins40_clean' else x_rows
                N_band, nu_band, redchi = SF.fit_noise_model(arrays['clean40'][first], noisy_for_model[first])
                scale40 = SF.bins40_scale('noise', N_band, nu_band)
                record.update(noise_model_from='bins40_noisy' if name == 'bins40_clean' else name,
                              N_band=N_band.tolist(), nu_band=nu_band.tolist(), fit_reduced_chi2=redchi,
                              scale=scale40.tolist())
            state = SF.fit_bins40(arrays['clean40'][first], x_rows[first], scale40)
            x, x_obs = SF.project_binned(state, x_rows), SF.project_binned(state, x_obs40)
        else:
            transform = SF.fit_moped(x_rows[first], choice)
            spectrum = 'noisy' if name == 'moped_noisy' else 'beam'
            x, x_obs = transform(x_rows), transform(b12[f'{spectrum}_dl_unbinned'][:1] @ matrix)
        folder = base / name / f'N{size}'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'transform.json').write_text(json.dumps(record, indent=1, default=float) + '\n')
        np.savez(folder / 'features.npz', x_pool=x[train_idx], theta_pool=theta[train_idx], x_held=x[test_idx],
                 theta_held=theta[test_idx], pool=pos[train_idx], held=pos[test_idx], low=inp['low'],
                 high=inp['high'], obs_labels=np.array(['Battaglia12']), x_obs=np.asarray(x_obs, np.float32))
    # training: one train_ensemble.py call per variant, all seeds in parallel processes
    for name in variants:
        folder = base / name / f'N{size}'
        if len(list(folder.glob('estimator_seed*.pt'))) == len(SEEDS) and not args.force:
            continue
        net = VARIANTS[name]
        cmd = [sys.executable, str(HERE / 'train_ensemble.py'), 'train', '--out', str(base / name),
               '--kinds', f'N{size}', '--seeds', *SEEDS, '--workers', str(args.workers), '--threads',
               str(args.threads), '--hidden', str(net['hidden']), '--transforms', str(net['transforms']),
               '--epochs', EPOCHS]
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    evaluate(base, variants, size, args)
    (base / f'N{k}.done').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')


def evaluate(base, variants, size, args):
    """metrics.json per variant at N = size, as sbi_final.cmd_evaluate, plus per-row log p and B12 samples."""
    import torch
    import analyze_64k as A
    from sbi_final import row_metrics
    from train_ensemble import Ensemble, summarize
    torch.set_num_threads(args.threads * args.workers)
    for name in variants:
        folder = base / name / f'N{size}'
        if (folder / 'metrics.json').exists() and not args.force:
            continue
        ens = Ensemble(folder, args.threads * args.workers)
        f, width = ens.f, ens.high - ens.low
        metrics, lp = row_metrics(ens, f['x_held'], f['theta_held'], args.samples)
        torch.manual_seed(2)
        sb, lpb, _ = ens.posterior(f['x_obs'][:1], 20000, rng=np.random.default_rng(2))
        b12 = summarize(sb[:, 0], lpb[:, 0], A.B12)
        hpd = ens.hpd_level(f['x_obs'][:1], A.B12, n=4000, rng=np.random.default_rng(3))
        b12['truth_hpd_level'] = float(hpd[0])
        train = [json.loads((folder / f'train_seed{s}.json').read_text()) for s in ens.seeds]
        m = dict(variant=name, N=size, seeds=ens.seeds, test_rows=int(len(f['theta_held'])), **metrics, b12=b12,
                 b12_width68_over_prior=((np.array(b12['q84']) - np.array(b12['q16'])) / width).tolist(),
                 training=[dict(seed=t['seed'], seconds=t['seconds'], epochs=t['epochs'],
                                best_validation_loss=t['best_validation_loss']) for t in train],
                 settings=ens.settings)
        (folder / 'metrics.json').write_text(json.dumps(m, indent=1) + '\n')
        np.save(folder / 'per_row_log_prob.npy', lp)
        np.save(folder / 'b12_samples.npy', sb[:5000, 0].astype(np.float32))
        print(f"{name} N{size}: log p {m['mean_log_prob_theta']:.2f} +- {m['mean_log_prob_theta_sem']:.2f}, "
              f"TARP {m['tarp68']:.2f}/{m['tarp95']:.2f}, mean r {np.mean(m['pearson_r']):.3f}, "
              f"B12 width {np.mean(m['b12_width68_over_prior']):.2f}", flush=True)


# ----------------------------------------------------------------------------- report

def collect_series(base):
    """Evaluated folders of one series dir: ({variant: {N: metrics}}, {(variant, N): per-row log p})."""
    series, lp = {}, {}
    for folder in sorted(base.glob('*/N*')):
        if not (folder / 'metrics.json').exists():
            continue
        m = json.loads((folder / 'metrics.json').read_text())
        series.setdefault(m['variant'], {})[m['N']] = m
        lp[(m['variant'], m['N'])] = np.load(folder / 'per_row_log_prob.npy')
    return series, lp


def paired_differences(lp):
    """Paired log p differences on the same test rows against the production-noise variant of the same compression."""
    paired = {}
    for (name, N), values in lp.items():
        for reference in ('bins40_noisy', 'moped_noisy'):
            if name != reference and name.split('_')[0] == reference.split('_')[0] and (reference, N) in lp:
                d = values - lp[(reference, N)]
                paired.setdefault(f'{name}-{reference}', {})[N] = dict(mean=float(d.mean()),
                                                                     sem=float(d.std() / np.sqrt(len(d))))
    return paired


SUMMARY_KEYS = ('mean_log_prob_theta', 'mean_log_prob_theta_sem', 'tarp68', 'tarp95', 'pearson_r',
                'median_width68_over_prior', 'median_width68_over_flat', 'edge_mass_ratio', 'rmse_over_prior_rmse',
                'b12_width68_over_prior', 'b12', 'training')


def cmd_report(args):
    """results/series.json and figures from every evaluated series folder (both asinh scales), the stage qa and
    results_final. Files of the median-scale series keep their names (fig_series.png, ...); the noise-scale series
    adds fig_series_noise.png, fig_series_params_noise.png, fig_b12_noise_N<N>.png."""
    root = run_root(args)
    inp = load_inputs(root)
    out = root / 'results'
    out.mkdir(exist_ok=True)
    collected = {choice: collect_series(root / d) for choice, d in SERIES_DIRS.items()}
    # the same data with the noise-level scale minus the median scale (same rows, same test rows)
    scale_change = {}
    for (name, N), values in collected['noise'][1].items():
        if (name, N) in collected['median'][1]:
            d = values - collected['median'][1][(name, N)]
            scale_change.setdefault(name, {})[N] = dict(mean=float(d.mean()), sem=float(d.std() / np.sqrt(len(d))))
    qa = {}
    for k in range(1, n_stages(inp) + 1):
        p = root / 'stages' / f'stage{k:02d}' / 'qa.json'
        if p.exists():
            q = json.loads(p.read_text())
            qa[k] = {key: v for key, v in q.items() if not key.startswith('band_')}
            qa[k]['band_noise_rms_ratio_beam_over_production'] = q['band_noise_rms_ratio_beam_over_production']
            qa[k]['band_noise_rms_ratio_half_over_production'] = q['band_noise_rms_ratio_half_over_production']
    reference = json.loads((root / 'inputs' / INPUT_FILES['reference']).read_text())
    summary = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), qa=qa, results_final=reference,
                   noise_scale_minus_median_scale=scale_change)
    for choice, (series, lp) in collected.items():
        suffix = '' if choice == 'median' else f'_{choice}'
        paired = paired_differences(lp)
        summary[f'paired_log_p{suffix}'] = paired
        summary[f'series{suffix}'] = {name: {N: {key: m[key] for key in SUMMARY_KEYS} for N, m in sorted(v.items())}
                                      for name, v in series.items()}
        if not series:
            continue
        label = ('asinh scale: median |clean| per band (as results_final)' if choice == 'median'
                 else 'asinh scale: noise level (each variant\'s own noise model; MOPED 1)')
        plot_series(series, paired, reference, out / f'fig_series{suffix}.png', title=label)
        plot_params(series, out / f'fig_series_params{suffix}.png', title=label)
        latest = max(N for v in series.values() for N in v)
        plot_b12(root, latest, out / f'fig_b12{suffix}_N{latest}.png', root / SERIES_DIRS[choice], title=f'; {label}')
    (out / 'series.json').write_text(json.dumps(summary, indent=1) + '\n')
    if qa:
        plot_qa(qa, out / 'fig_qa_noise_ratio.png')
    print(json.dumps(dict(paired_log_p=summary['paired_log_p'], paired_log_p_noise=summary.get('paired_log_p_noise'),
                          noise_scale_minus_median_scale=scale_change), indent=1))


# colour = noise treatment, line style and marker = compression
COLORS = dict(bins40_noisy='#2a78d6', bins40_beam='#e34948', bins40_half='#eda100', bins40_clean='#1baf7a',
              moped_noisy='#2a78d6', moped_beam='#e34948')
STYLE = dict(bins40=dict(ls='-', marker='o'), moped=dict(ls='--', marker='s'))


def style(name):
    return dict(color=COLORS.get(name, 'k'), **STYLE[name.split('_')[0]])
NAMES = dict(bins40_noisy='40 bands, production noise (no beam on noise)', bins40_beam='40 bands, beamed noise',
             bins40_half='40 bands, beamed noise, half-depth splits', bins40_clean='40 bands, no noise',
             moped_noisy='MOPED, production noise', moped_beam='MOPED, beamed noise')


def plot_series(series, paired, reference, path, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for name, v in series.items():
        Ns = sorted(v)
        st = style(name)
        axes[0, 0].errorbar(Ns, [v[N]['mean_log_prob_theta'] for N in Ns],
                            [v[N]['mean_log_prob_theta_sem'] for N in Ns], ms=4, lw=1.2, label=NAMES.get(name, name),
                            **st)
        axes[0, 1].plot(Ns, [np.mean(v[N]['pearson_r']) for N in Ns], ms=4, lw=1.2, **st)
        axes[1, 0].plot(Ns, [np.mean(v[N]['median_width68_over_flat']) for N in Ns], ms=4, lw=1.2, **st)
    for name in ('bins40_noisy', 'bins40_clean', 'moped_noisy'):
        ref = reference.get(name, {})
        Ns = sorted(int(N) for N in ref)
        if Ns:
            ref_ls = ':' if name.startswith('bins40') else '-.'
            axes[0, 0].plot(Ns, [ref[str(N)]['mean_log_prob_theta'] for N in Ns], color=COLORS[name], ls=ref_ls,
                            lw=1, alpha=0.6, label=f'{NAMES[name]} (results_final, transforms fitted on 61,440 rows)')
            axes[0, 1].plot(Ns, [np.mean(ref[str(N)]['pearson_r']) for N in Ns], color=COLORS[name], ls=ref_ls, lw=1,
                            alpha=0.6)
    for key, v in paired.items():
        Ns = sorted(v)
        axes[1, 1].errorbar(Ns, [v[N]['mean'] for N in Ns], [v[N]['sem'] for N in Ns], ms=4, lw=1.2,
                            label=key.replace('-', ' minus '), **style(key.split('-')[0]))
    axes[1, 1].axhline(0, color='0.5', lw=0.8)
    titles = ['mean log p of the truth on the test rows [nats]', 'mean Pearson r (posterior mean vs truth)',
              'posterior 68 % width / flat-prior width (1 = nothing learnt)', 'paired log p difference [nats]']
    for ax, title in zip(axes.flat, titles):
        ax.set_xscale('log')
        ax.set_xlabel('training rows N')
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
    axes[1, 1].legend(fontsize=6.5, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97 if title else 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


PARAMS = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']


def plot_params(series, path, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
    for i, ax in enumerate(axes.flat):
        for name, v in series.items():
            Ns = sorted(v)
            ax.plot(Ns, [v[N]['pearson_r'][i] for N in Ns], ms=3, lw=1, label=NAMES.get(name, name), **style(name))
        ax.set_title(f'{PARAMS[i]}: Pearson r', fontsize=9)
        ax.set_xscale('log')
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('training rows N')
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_b12(root, N, path, base, title=''):
    """1D marginals of the B12 posterior at N for every variant of series dir base (5,000 samples each)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import analyze_64k as A
    inp = load_inputs(root)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for i, ax in enumerate(axes.flat):
        for name in VARIANTS:
            p = base / name / f'N{N}' / 'b12_samples.npy'
            if p.exists():
                s = np.load(p)[:, i]
                ax.hist(s, bins=60, range=(inp['low'][i], inp['high'][i]), density=True, histtype='step',
                        color=COLORS[name], ls=STYLE[name.split('_')[0]]['ls'], lw=1.1, label=NAMES[name])
        ax.axvline(A.B12[i], color='k', lw=0.8, ls='--')
        ax.set_title(PARAMS[i], fontsize=9)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f'B12 observation: marginal posteriors at N = {N} (dashed: B12 truth; x range = prior){title}',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_qa(qa, path):
    """Band noise rms of the corrected spectra relative to production, per stage."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    centres = 0.5 * (EDGES[:-1] + EDGES[1:] - 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, q in qa.items():
        ax.plot(centres, q['band_noise_rms_ratio_beam_over_production'], color='#e34948', lw=1, alpha=0.7,
                label='beamed, full depth' if k == min(qa) else None)
        ax.plot(centres, q['band_noise_rms_ratio_half_over_production'], color='#eda100', lw=1, alpha=0.7,
                label='beamed, half-depth splits' if k == min(qa) else None)
    ax.axhline(1, color='0.5', lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('ell (band centre)')
    ax.set_ylabel('rms of (spectrum - clean) / production')
    ax.set_title('Noise in the 40 bands after the beam correction', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default=str(HERE.parent), help='run root')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    sub.add_parser('status')
    a = sub.add_parser('extract-clean')
    a.add_argument('folder')
    a = sub.add_parser('reclaim')
    a.add_argument('--alive', nargs='*', default=[], help='PBS job ids (numbers) of running regen workers')
    a.add_argument('--min-age', type=float, default=300)
    a.add_argument('--max-age', type=float, default=3600)
    a = sub.add_parser('build')
    a.add_argument('stage', type=int)
    a = sub.add_parser('series')
    a.add_argument('stage', type=int)
    a.add_argument('--variants', nargs='*', default=None)
    a.add_argument('--workers', type=int, default=6)
    a.add_argument('--threads', type=int, default=4)
    a.add_argument('--samples', type=int, default=1000)
    a.add_argument('--force', action='store_true')
    a.add_argument('--asinh-scale', choices=tuple(SERIES_DIRS), default='median')
    sub.add_parser('report')
    args = p.parse_args()
    commands = {'plan': cmd_plan, 'status': cmd_status, 'reclaim': cmd_reclaim, 'extract-clean': cmd_extract_clean,
                'build': cmd_build, 'series': cmd_series, 'report': cmd_report}
    commands[args.cmd](args)


if __name__ == '__main__':
    main()
