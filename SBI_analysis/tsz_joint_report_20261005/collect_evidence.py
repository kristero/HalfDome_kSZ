#!/usr/bin/env python3
"""Copy only the frozen evidence needed to rebuild this combined report.

Sources are read-only. SHA256 identities distinguish this delivery from earlier
reports and make the report independent of the original local directory layout.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent / 'tsz_robustness_methods_20261004' / 'final_results'
DIAG = ROOT.parent / 'tsz_seed_offset_diagnosis_20261005'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    copied = []
    groups = {
        'study': (STUDY / 'report', [
            'summary.json', 'bounded_summary.json', 'selection.json',
            'prior_final_test_linear.json', 'prior_final_test_logit.json',
            'joint_truth_linear.json', 'joint_truth_logit.json',
            'posterior_intervals.csv', 'method_contracts.csv',
            'calibration_qualification.json', 'training_summary.csv',
            'parameter_width_cost.png', 'posteriors_simulation.png',
            'posteriors_mask.png', 'posteriors_combined.png',
            'signal_variation.png', 'compression_scale_response.png',
            'selected_training_losses.png', 'combined_grid_original.png',
            'combined_grid_selected.png', 'joint_truth_linear.png',
        ]),
        'study_source': (STUDY / 'source_snapshot', [
            'paint_seed.jl', 'common.py', 'pipeline.py', 'report.py',
            'merlin_checks.py', 'joint_truth.py', 'final_prior_test.py',
        ]),
        'diagnosis': (DIAG, [
            'REPORT.md', 'diagnostics.json', 'original_linear.json',
            'original_bounded.json', 'extended_original_linear.json',
            'extended_original_bounded.json', 'band50_linear.json',
            'counterfactuals.json', 'extended_cases.json',
            'posterior_correlations.json', 'FINAL_COMPLETE.json',
            'VERIFIED.json', 'input_sha256.json', 'jobs.json',
            'diagnose.py', 'extended_controls.py', 'audit_inputs.py',
            'plots/mechanism_summary.png', 'plots/fingerprint_per_seed.png',
            'plots/mask_term_controls.png',
            'plots/feature_support_and_noise_distance.png',
            'plots/posterior_response_paths.png',
        ]),
    }
    for group, (source, names) in groups.items():
        for name in names:
            src = source / name
            dst = ROOT / 'evidence' / group / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            assert sha(src) == sha(dst)
            copied.append(dict(source=str(src), delivered=str(dst.relative_to(ROOT)),
                               bytes=dst.stat().st_size, sha256=sha(dst)))
    marker = STUDY / 'COMPLETE.json'
    if not marker.exists():
        marker = STUDY / 'comparison' / 'COMPLETE.json'
    # The local delivery may place completion markers at its root.
    matches = list(STUDY.glob('*COMPLETE*.json'))
    for marker in matches:
        dst = ROOT / 'evidence' / 'study' / marker.name
        shutil.copy2(marker, dst)
        copied.append(dict(source=str(marker), delivered=str(dst.relative_to(ROOT)),
                           bytes=dst.stat().st_size, sha256=sha(dst)))
    (ROOT / 'evidence_manifest.json').write_text(json.dumps(copied, indent=2)+'\n')
    print(f'Copied and verified {len(copied)} frozen evidence files.')


if __name__ == '__main__':
    main()
