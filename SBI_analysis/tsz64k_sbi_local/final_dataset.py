"""Merge the finished rows of the collaborator's tSZ_64k production into the local SBI dataset.

The production (Flatiron run tsz_64k_flat_prior_SO_baseline_deproj0_20260926) finished 64,440 of the 65,536 design
rows; its 274 failed batches of 4 rows are being regenerated on idark as an independent validation set. This script
checks every finished row against the frozen design and writes them in the layout of pipeline/manage.py collect.

Reads
  ../../tSZ_64k_flat_prior_SO_baseline_deproj0/design/  frozen design: design_manifest.json (file names, SHA-256,
      parameter order and prior bounds), theta_design_65536.npy (65536 x 9), noise_seeds_65536.npy (65536 x 2)
  data/delivery/batch_plan.json  the production's batch plan (per batch: id, cases with row, label, theta, seeds)
  data/final_status/logs/tsz64k_prod_*.out  final production logs ("batch <id>: ran (" or "batch <id>: done (")
  data/rows_finished/  25,944 rows collected on 2026-09-27 from the same run (row_id, theta, noise_seeds and the
      unbinned clean/noisy D_ell)
  data/delivery/batches/<batch>/<label>/  per newly finished row, masked_clean_cl.npy and masked_noisy_cross_cl.npy
      (C_ell for ell = 0..7979), copied by Globus with checksums verified in transit
  ../tsz_diagnostic_256_accelerated_20260922/reference/compression_context.npz  the fixed B12 MOPED matrix

Checks: the plan's design hashes against the frozen design, every case's theta and seeds against the design, that
the final logs report "ran"/"done" for every batch used, complete file sizes, finite spectra, positive clean
spectra, one unique pair of noise seeds per row, and the old rows against the design.

Writes data/rows_64440/ (small, for the SBI analysis; n = 64,440 rows in ascending design row):
  theta.npy (n, 9), noise_seeds.npy (n, 2) int64, row_id.npy (n,) int64, ell_unbinned.npy (7900,) = 80..7979,
  clean_dl.npy and noisy_dl.npy (n, 40) band-averaged D_ell, moped_raw_clean.npy and moped_raw_noisy.npy (n, 9)
  = unbinned D_ell @ the fixed B12 MOPED matrix, and meta.json (provenance, units, binning, missing rows).
With --upload the float64 unbinned spectra clean/noisy_dl_unbinned.npy (n, 7900) and all small arrays are streamed
to idark over ssh (the unbinned ones are never written locally) and their SHA-256 are checked there; --small-only
uploads only the small arrays.

Run with the halfdome python from this folder:
  python final_dataset.py [--upload /lustre/work/kristero10/tsz64k_dataset_20260928/rows_64440 [--small-only]]
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent                                  # this project folder
BUNDLE = HERE.parent.parent / 'tSZ_64k_flat_prior_SO_baseline_deproj0'  # collaborator bundle, frozen design
TEST256 = HERE.parent / 'tsz_diagnostic_256_accelerated_20260922'       # holds the fixed MOPED compression matrix
ELL = np.arange(80, 7980)                    # multipoles kept from the delivered spectra (which start at ell = 0)
DL = ELL * (ELL + 1) / (2 * np.pi)           # C_ell -> D_ell = ell(ell+1)C_ell/2pi
NPY_BYTES = 63920                            # size of a complete delivered spectrum: 7980 float64 + 80-byte header
BAND_EDGES = np.r_[np.arange(80, 7881, 200), 7980]  # 40 bands: 39 of 200 multipoles, the last is 7880..7979
DESIGN_ROWS = 65536                          # rows of the frozen design
BATCH_SIZE = 4                               # rows per production batch: batch b holds rows 4b..4b+3
MOPED_KEY = 'Battaglia12_rcond1e-06'         # the B12 MOPED matrix (7900 x 9) inside compression_context.npz
SSH = ('ssh.exe', '-o', 'BatchMode=yes')     # ssh from WSL that fails instead of prompting
REMOTE_HOST = 'idark'                        # ssh host of the upload target
UPLOAD_ATTEMPTS = 3                          # tries per uploaded array
RETRY_SECONDS = 30                           # wait after a failed upload attempt


def sha256_file(path):
    """Return the SHA-256 hex digest (str) of the file at `path`, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def rebin40(values):
    """Average unbinned spectra into the 40 bands of BAND_EDGES.

    values: (n, 7900) array on ELL, e.g. D_ell. Returns (n, 40) float64. Each band is the mean weighted by
    2 ell + 1, the number of m modes at each ell, so every mode in the band counts equally.
    """
    bands = []
    for low, high in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        in_band = (ELL >= low) & (ELL < high)
        bands.append(np.average(values[:, in_band], axis=1, weights=2 * ELL[in_band] + 1))
    return np.stack(bands, axis=1)


def finished_batches(log_folder):
    """Return the set of batch ids (int) that the final production logs report as finished.

    log_folder: Path with the tsz64k_prod_*.out logs. A batch is finished when a log line starts with
    'batch <id>: ran (' (computed in that job) or 'batch <id>: done (' (found complete by a restarted job).
    """
    finished = set()
    for log_path in glob.glob(str(log_folder / 'tsz64k_prod_*.out')):
        with open(log_path) as handle:
            log = handle.read()
        finished.update(int(match.group(1)) for match in re.finditer(r'^batch (\d+): (ran|done) \(', log, re.M))
    return finished


def load_delivered_row(folder, row):
    """Read one newly finished row from the Globus delivery and convert it to D_ell on ELL.

    folder: Path data/delivery/batches/<batch>/<label>; row: its design row (for the error messages). Both files
    must be complete and hold 7980 finite C_ell (ell = 0..7979); the clean C_ell must be positive on ELL (the
    noisy spectrum is a signed cross spectrum of two splits and may be negative).
    Returns (clean_dl, noisy_dl), two (7900,) float64 arrays of dimensionless Compton-y D_ell.
    """
    clean_path = folder / 'masked_clean_cl.npy'
    noisy_path = folder / 'masked_noisy_cross_cl.npy'
    complete = clean_path.stat().st_size == NPY_BYTES and noisy_path.stat().st_size == NPY_BYTES
    assert complete, f'row {row}: incomplete files'
    clean_cl = np.load(clean_path)
    noisy_cl = np.load(noisy_path)
    valid = (clean_cl.shape == noisy_cl.shape == (7980,) and np.isfinite(clean_cl).all()
             and np.isfinite(noisy_cl).all() and (clean_cl[ELL] > 0).all())
    assert valid, f'row {row}'
    return clean_cl[ELL] * DL, noisy_cl[ELL] * DL


class _HashingPipe:
    """Writable file object that hashes every chunk it receives and forwards it to a pipe.

    np.lib.format.write_array writes the .npy header and data through write(), so the digest covers exactly the
    bytes sent.
    """

    def __init__(self, digest, pipe):
        """digest: hashlib object, updated in place; pipe: binary stdin of the ssh process."""
        self.digest = digest
        self.pipe = pipe

    def write(self, chunk):
        """Hash and send one bytes-like chunk; return the number of bytes written."""
        data = bytes(chunk)
        self.digest.update(data)
        self.pipe.write(data)
        return len(data)


class Remote:
    """Stream .npy arrays and text files to a folder on idark over ssh, hashing exactly the bytes sent.

    Nothing is written locally. `sums` maps each sent file name to the SHA-256 of its bytes; verify() has idark
    check them with sha256sum -c.
    """

    def __init__(self, folder):
        """folder: str, absolute path on idark; created with mkdir -p."""
        self.folder = folder
        subprocess.run([*SSH, REMOTE_HOST, f'mkdir -p {folder}'], check=True)
        self.sums = {}

    def save(self, name, array):
        """Send `array` as the .npy file `name` (allow_pickle=False, C order).

        The bytes go to <name>.part, which is renamed only when cat succeeds, so an interrupted upload never
        leaves a truncated file under the final name. Up to three attempts, 30 s apart; SystemExit if all fail.
        """
        for attempt in range(UPLOAD_ATTEMPTS):
            digest = hashlib.sha256()
            t0 = time.time()
            part = f'{self.folder}/{name}.part'
            remote_command = f'cat > {part} && mv {part} {self.folder}/{name}'
            proc = subprocess.Popen([*SSH, '-o', 'ServerAliveInterval=30', REMOTE_HOST, remote_command],
                                    stdin=subprocess.PIPE)
            try:
                np.lib.format.write_array(_HashingPipe(digest, proc.stdin), np.ascontiguousarray(array),
                                          allow_pickle=False)
                proc.stdin.close()
                if proc.wait() == 0:
                    self.sums[name] = digest.hexdigest()
                    print(f'  {name}: {array.nbytes / 1e6:.0f} MB in {time.time() - t0:.0f} s', flush=True)
                    return
            except (BrokenPipeError, OSError) as error:
                print(f'  {name}: attempt {attempt + 1} failed ({error}); retrying', flush=True)
                proc.kill()
            time.sleep(RETRY_SECONDS)
        raise SystemExit(f'upload of {name} failed three times')

    def text(self, name, content):
        """Send the str `content` (UTF-8) as the file `name` and record its SHA-256."""
        encoded = content.encode()
        subprocess.run([*SSH, REMOTE_HOST, f'cat > {self.folder}/{name}'], input=encoded, check=True)
        self.sums[name] = hashlib.sha256(encoded).hexdigest()

    def verify(self):
        """Send SHA256SUMS.local (the digests of everything sent so far) and run sha256sum -c on idark.

        Prints idark's report; SystemExit if any remote file differs from what was sent.
        """
        listing = '\n'.join(f'{digest}  {name}' for name, digest in sorted(self.sums.items())) + '\n'
        self.text('SHA256SUMS.local', listing)
        check = subprocess.run([*SSH, REMOTE_HOST, f'cd {self.folder} && sha256sum -c SHA256SUMS.local'],
                               capture_output=True, text=True)
        print(check.stdout.strip())
        if check.returncode != 0:
            raise SystemExit('remote checksums differ')


def main():
    """Check and merge the finished rows, write the small arrays to --out and optionally upload to idark."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, default=HERE / 'data' / 'delivery')
    parser.add_argument('--old', type=Path, default=HERE / 'data' / 'rows_finished')
    parser.add_argument('--logs', type=Path, default=HERE / 'data' / 'final_status' / 'logs')
    parser.add_argument('--out', type=Path, default=HERE / 'data' / 'rows_64440')
    parser.add_argument('--upload', default=None, help='idark folder for the dataset')
    parser.add_argument('--small-only', action='store_true',
                        help='upload only the 40-band, MOPED and parameter arrays (slow link)')
    args = parser.parse_args()

    # The frozen design, and a batch plan that must refer to exactly these design files.
    design = json.loads((BUNDLE / 'design' / 'design_manifest.json').read_text())
    design_theta = np.load(BUNDLE / 'design' / design['files']['theta_design'])
    design_seeds = np.load(BUNDLE / 'design' / design['files']['noise_seeds'])
    plan = json.loads((args.delivery / 'batch_plan.json').read_text())
    for name, digest in plan['design_sha256'].items():
        on_disk = sha256_file(BUNDLE / 'design' / name)
        assert digest == on_disk == design['sha256'][name], f'plan design hash mismatch: {name}'
    assert plan['rows'] == DESIGN_ROWS and plan['batch_size'] == BATCH_SIZE and not plan['deferred_rows']

    # Finished batches and their rows; the output is ordered by ascending design row.
    finished = finished_batches(args.logs)
    missing = sorted(set(range(len(plan['batches']))) - finished)
    old_rows = np.load(args.old / 'row_id.npy')
    assert np.array_equal(np.load(args.old / 'theta.npy'), design_theta[old_rows])
    assert np.array_equal(np.load(args.old / 'noise_seeds.npy'), design_seeds[old_rows])
    old_row_set = set(old_rows.tolist())
    rows = sorted(int(case['row']) for batch_id in finished for case in plan['batches'][batch_id]['cases'])
    n_rows = len(rows)
    position = {row: index for index, row in enumerate(rows)}          # design row -> output row
    print(f'{len(finished)} finished batches, {n_rows} rows ({len(old_row_set)} already collected); '
          f'{len(missing)} batches missing', flush=True)

    # Unbinned D_ell (n_rows, 7900): copy the rows collected before, then read the newly delivered ones.
    clean = np.empty((n_rows, len(ELL)))
    noisy = np.empty((n_rows, len(ELL)))
    old_clean = np.load(args.old / 'clean_dl_unbinned.npy', mmap_mode='r')
    old_noisy = np.load(args.old / 'noisy_dl_unbinned.npy', mmap_mode='r')
    for old_index, row in enumerate(old_rows):
        clean[position[int(row)]] = old_clean[old_index]
        noisy[position[int(row)]] = old_noisy[old_index]
    n_new = 0
    for batch_id in sorted(finished):
        batch = plan['batches'][batch_id]
        assert batch['id'] == batch_id
        for case in batch['cases']:
            row = int(case['row'])
            # Every case, also an old one, must carry the design's parameters and noise seeds.
            same = np.array_equal(case['theta'], design_theta[row]) and case['seeds'] == design_seeds[row].tolist()
            assert same, f'row {row} differs'
            if row in old_row_set:
                continue
            folder = args.delivery / 'batches' / f'{batch_id:05d}' / case['label']
            clean[position[row]], noisy[position[row]] = load_delivered_row(folder, row)
            n_new += 1
    assert n_new + len(old_row_set) == n_rows, (n_new, len(old_row_set), n_rows)
    row_ids = np.asarray(rows, dtype=np.int64)
    seeds = design_seeds[row_ids]
    # Two seeds per row (one per noise split); none may repeat, so the 2 n_rows noise realisations are independent.
    assert len(np.unique(seeds.ravel())) == 2 * n_rows, 'repeated noise seed'
    assert np.all(clean > 0) and np.isfinite(noisy).all()

    # Raw MOPED projections, moped_raw = D_ell @ matrix: the matrix is the fixed B12 MOPED compression built in
    # ../tsz_reuse_cache_20260921/analyze.py (build_weights) from the Battaglia12 anchor's parameter derivatives
    # and 64 noise draws (rcond 1e-6), already divided by the per-ell noise sigma so that it acts on raw D_ell.
    matrix = np.load(TEST256 / 'reference' / 'compression_context.npz')[MOPED_KEY]
    small = dict(theta=design_theta[row_ids], noise_seeds=seeds, row_id=row_ids, ell_unbinned=ELL,
                 clean_dl=rebin40(clean), noisy_dl=rebin40(noisy),
                 moped_raw_clean=clean @ matrix, moped_raw_noisy=noisy @ matrix)
    missing_rows = [int(row) for batch_id in missing
                    for row in range(BATCH_SIZE * batch_id, BATCH_SIZE * batch_id + BATCH_SIZE)]
    meta = dict(rows=int(n_rows), requested_rows=DESIGN_ROWS, missing_rows=missing_rows, missing_batches=missing,
                source='collaborator run tsz_64k_flat_prior_SO_baseline_deproj0_20260926 (Flatiron), '
                       'finished 2026-09-28 10:37 UTC; its final collector reported the same 1,096 missing rows',
                parameter_order=design['parameter_order'], lower=design['lower'], upper=design['upper'],
                units='D_ell = ell(ell+1)C_ell/2pi of Compton-y, dimensionless; ell = 80..7979 unbinned',
                binning='40 bins of 200 multipoles (last to 7979), weights 2l+1',
                spectra=dict(clean_dl_unbinned='masked clean signal (fsky=0.4 cap, 2 arcmin beam)',
                             noisy_dl_unbinned='signed cross spectrum of two masked signal+noise splits, '
                                               'SO baseline Deproj-0, two seeds per row'),
                moped=f'moped_raw_* = D_ell_unbinned @ compression_context.npz[{MOPED_KEY}] (7900 x 9)',
                design_sha256=design['sha256'], batch_plan_sha256=sha256_file(args.delivery / 'batch_plan.json'),
                independent_noise_realizations=int(2 * n_rows),
                created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    meta_text = json.dumps(meta, indent=1) + '\n'

    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in small.items():
        np.save(args.out / f'{name}.npy', value)
    (args.out / 'meta.json').write_text(meta_text)
    print(f'wrote the small arrays to {args.out}', flush=True)
    if args.upload:
        remote = Remote(args.upload)
        if args.small_only:
            uploads = small
        else:
            uploads = dict(clean_dl_unbinned=clean, noisy_dl_unbinned=noisy, **small)
        for name, value in uploads.items():
            remote.save(f'{name}.npy', value)
        remote.text('meta.json', meta_text)
        remote.verify()


if __name__ == '__main__':
    main()
