#!/usr/bin/env python3
"""Verify completed inference, publication figures and the reviewed final PDF.

Checks on idark read original inputs again; they do not run scientific
inference on the login node. Useful results and logs are retained. Cleanup
removes only obsolete transfer/review products inside this new experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def read(root, name):
    return json.loads((root/name).read_text())


def write(root, name, value):
    (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify(root, cluster=False):
    inference = read(root, 'INFERENCE_COMPLETE.json')
    plots = read(root, 'PLOTS_COMPLETE.json')
    publication = read(root, 'PUBLICATION_COMPLETE.json')
    for marker in [inference, plots]:
        for filename, expected in marker['outputs'].items():
            assert sha(root/filename) == expected, filename
    assert sha(root/'INFERENCE_COMPLETE.json') == plots['inference_marker_sha256']
    assert sha(root/'INFERENCE_COMPLETE.json') == publication['inference_marker_sha256']
    assert sha(root/'PLOTS_COMPLETE.json') == publication['plots_marker_sha256']
    for name, expected in publication['source_sha256'].items():
        assert sha(root/name) == expected, name
    # Inference sources were preserved before publication formatting changes.
    for name in ['diagnostic_metrics.py', 'evaluate_truth_mean.py',
                 'plot_truth_mean.py', 'submit_idark.py']:
        assert sha(root/'source_snapshot'/name) == inference['source'][name], name
    for name in ['diagnostic_metrics.py', 'evaluate_truth_mean.py']:
        assert sha(root/name) == inference['source'][name], name
    jobs = read(root, 'jobs.json')['jobs']
    assert len(jobs) == 2 and {j['id'] for j in jobs} == {'600740.idark', '600741.idark'}
    for job in jobs:
        number = job['id'].split('.')[0]
        prefix = 'publication_' if job.get('task') == 'publication' else ''
        assert 'Exit status: 0' in (root/'logs'/f'{prefix}resources_{number}.txt').read_text()
        log_name = 'publication' if prefix else 'truth_mean'
        assert 'finished=' in (root/'logs'/f'{log_name}_{number}.log').read_text()
    report = read(root, 'PDF_BUILT.json')
    review = read(root, 'PDF_REVIEW.json')
    pdf = root/report['pdf']
    assert sha(pdf) == report['sha256'] == review['pdf_sha256']
    assert review['all_pages_visually_reviewed'] and review['pages'] == 44
    assert review['reviewed_pages'] == list(range(1, 45))
    assert sha(root/'PLOTS_COMPLETE.json') == report['plots_marker_sha256']
    assert sha(root/'build_supplement.py') == report['builder_sha256']
    if cluster:
        for filename, expected in inference['previous_study_inputs_verified'].items():
            assert sha(Path(filename)) == expected, filename
        for filename, expected in inference['frozen_inputs'].items():
            assert sha(Path(filename)) == expected, filename
        previous = root.parent/'tsz_joint_report_20261005/output/pdf/HalfDome_Merlin_seed_mask_and_Pearson_report.pdf'
        assert sha(previous) == report['combined_report']['previous_report_sha256']
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(), complete=True,
        analysis_on_idark=True, new_jobs=2, jobs=[j['id'] for j in jobs],
        application_exit_statuses=[0, 0], pages=44, pdf_sha256=sha(pdf),
        inference_output_hashes_verified=len(inference['outputs']),
        final_plot_output_hashes_verified=len(plots['outputs']),
        frozen_inputs_verified_live=cluster, posterior_samples_not_regenerated_for_publication=True,
        physical_mean_definition='No prior clipping; original and band50 evaluated on identical reserved cases',
        limitation='Fixed halo100/mask12345 prior test; not sky-marginalized calibration over the pressure prior')


def cleanup(root):
    root = root.resolve()
    removed = []
    paths = [root/name for name in ['inference_delivery.zip', 'publication_delivery.zip', 'report_delivery.zip']]
    paths += list((root/'tmp/pdfs').glob('*.png'))+list((root/'tmp/pdfs').glob('*.pdf'))
    paths += list((root/'__pycache__').glob('*.pyc'))
    paths += [p for p in (root/'logs').glob('*.o6007*') if p.stat().st_size == 0]
    for path in paths:
        assert path.resolve().is_relative_to(root)
        if path.exists():
            removed.append(str(path.relative_to(root)))
            path.unlink()
    if (root/'__pycache__').exists() and not list((root/'__pycache__').iterdir()):
        (root/'__pycache__').rmdir()
    return removed


def main(args):
    root = args.root.resolve()
    assert root.name == 'tsz_truth_mean_validation_20261006'
    result = verify(root, args.cluster)
    if args.cleanup: result['obsolete_generated_files_removed'] = cleanup(root)
    result['files'] = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(root.rglob('*')) if p.is_file() and p.suffix != '.zip'
        and 'tmp' not in p.parts and '__pycache__' not in p.parts
        and p.name not in ['VERIFIED.json', 'CLUSTER_VERIFIED.json', 'RUN_STATUS.json']}
    filename = 'CLUSTER_VERIFIED.json' if args.cluster else 'VERIFIED.json'
    write(root, filename, result)
    write(root, 'RUN_STATUS.json', {key: value for key, value in result.items() if key != 'files'})
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['files', 'obsolete_generated_files_removed']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    main(parser.parse_args())
