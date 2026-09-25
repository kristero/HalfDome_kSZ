"""Inventory the review, hash its inputs, and record document checks.

Run after analyze.py, build_report.py and visual inspection of rendered pages.
Only this new review directory is written. No production files are touched.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PDF = REPO/'output/pdf/tsz_prior_pipeline_review_20260920.pdf'
QA = REPO/'tmp/pdfs/tsz_next8k_review'


def metadata(path):
    return dict(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def relative(path):
    return path.relative_to(REPO).as_posix()


inputs = []
study = REPO/'SBI_analysis/tsz_guardrail_study'
for case in ['Battaglia12', 'FL_L1_m9']:
    for variant in ['historical1', 'logz2', 'pixel8192_2', 'pixel16384_2']:
        folder = study/'maps'/case/variant
        inputs.extend(folder/name for name in ['status.json', 'time.txt', 'run.log'])
        if variant != 'historical1':
            inputs.append(folder/'masked_clean_cl.npy')
        if variant == 'pixel16384_2':
            inputs.append(folder/'unmasked_clean_cl.npy')
inputs.extend(study/name for name in [
    'audit/noise_covariance.npz', 'audit/noise_covariance.json',
    'METHODS.md', 'RESULTS.md', 'map_experiment.jl'])
base = REPO/'SBI_analysis/flamingo_linear_prior'
inputs.extend(base/name for name in [
    'prior.py', 'prior.json', 'stable_los.jl', 'paint_row.jl',
    'independent_noise.jl', 'cluster_results/completed_20260918/summary.json',
    'cluster_results/completed_20260918/manifest.json',
    'cluster_results/completed_20260918/dataset/theta.npy',
    'cluster_results/completed_20260918/dataset/validation_split.npy'])
followup = REPO/'SBI_analysis/tsz_beta_flat_followup_20260920'
inputs.extend(followup/name for name in [
    'RESULTS.md', 'continuous_flat.py', 'results/summary.json',
    'results/continuous_flat_validation.json', 'results/production_integrity.json'])
inputs.extend(REPO/name for name in [
    'SBI_analysis/linear_prior_sbi/SBI_LINEAR_8K_ANALYSIS_20260918.md',
    'SBI_analysis/linear_prior_sbi/sbi_linear_prior_pipeline.py',
    'SBI_analysis/linear_prior_sbi/cluster_results/figures/convergence_metrics.csv',
    'truncation_comparison/spherical_truncation_profiles.jl',
    'frb_map_generation/compare_takahashi_sightlines.py',
    'other_sims/SO/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt'])

pdf_info = subprocess.check_output(['pdfinfo', str(PDF)], text=True)
page_count = int(next(line.split(':', 1)[1] for line in pdf_info.splitlines()
                      if line.startswith('Pages:')))
assert page_count == 18, page_count
text = subprocess.check_output(['pdftotext', '-layout', str(PDF), '-'], text=True)
assert '\ufffd' not in text, 'Unicode replacement character in the PDF'
plan = json.loads((ROOT/'test_plan.json').read_text())
assert len({item['id'] for item in plan['tests']}) == 19
for item in plan['tests']:
    assert item['id'] + ' -' in text, item['id']

qa = dict(pages=page_count, all_19_test_ids_present=True,
          no_unicode_replacement_character=True,
          visual_review='All 18 pages rendered and inspected; revised pages checked again',
          scientific_scope='Reanalysis of saved experiments; new spherical full-sky tests remain pending')
(ROOT/'results/document_checks.json').write_text(json.dumps(qa, indent=2)+'\n')

outputs = sorted(path for path in ROOT.rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts
                 and path.name not in ['artifact_manifest.json', 'changed_files.txt'])
outputs.append(PDF)
inventory = outputs + [ROOT/'changed_files.txt', ROOT/'artifact_manifest.json']
(ROOT/'changed_files.txt').write_text(
    'New files only; no pre-existing production file edited.\n' +
    '\n'.join(sorted(relative(path) for path in inventory))+'\n')
outputs.append(ROOT/'changed_files.txt')
manifest = dict(
    input_hashes={relative(path):metadata(path) for path in inputs},
    output_hashes={relative(path):metadata(path) for path in outputs},
    halfdome_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    tracked_changes=subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=REPO, text=True).strip(),
    new_cluster_jobs_submitted=0,
    cluster_execution='Unavailable this turn: SSH timed out twice',
    document_checks=qa,
)
(ROOT/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(inputs=len(inputs), outputs=len(outputs),
                     pages=page_count, tracked_changes=manifest['tracked_changes']), indent=2))
