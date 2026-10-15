#!/usr/bin/env python3
"""Copy the final seed/mask figures into fig/4_seedmask with clear names.

Only copies existing, already-published figures (PNG downscaled to <= 1800 px wide,
PDF copied unchanged beside it). No data are recomputed.
Run from anywhere: python copy_figures.py
"""
from pathlib import Path
import shutil
from PIL import Image

REPO = Path('/home/cbllover/HalfDome')
OUT = REPO / 'reports/halfdome_overview_20261007/fig/4_seedmask'
SBI = REPO / 'SBI_analysis'
JR = SBI / 'tsz_joint_report_20261005/figures'
RB = SBI / 'tsz_robustness_methods_20261004/final_results/report'
OD = SBI / 'tsz_seed_offset_diagnosis_20261005/plots'
TM = SBI / 'tsz_truth_mean_validation_20261006/figures'
SM = SBI / 'tsz_seed_mask_variation_20261002/final_results/figures'

FIGURES = [
    ('01_grid_design', JR / 'grid_design'),
    ('02_signal_variation', RB / 'signal_variation'),
    ('03_first_study_variation', SM / 'variation_unbounded_primary'),
    ('04_posterior_simulation', JR / 'posterior_simulation'),
    ('05_posterior_mask', JR / 'posterior_mask'),
    ('06_combined_P0_grid', JR / 'combined_P0_grid'),
    ('07_counterfactual_posterior_response', OD / 'counterfactual_posterior_response'),
    ('08_mechanism_summary', OD / 'mechanism_summary'),
    ('09_fingerprint_per_seed', OD / 'fingerprint_per_seed'),
    ('10_baseline_smoothing', JR / 'baseline_smoothing'),
    ('11_pearson_linear_delta', JR / 'pearson_linear_delta'),
    ('12_held_tradeoffs', JR / 'held_tradeoffs'),
    ('13_variance_width', JR / 'variance_width'),
    ('14_joint_truth_linear', RB / 'joint_truth_linear'),
    ('15_saved_scale_cut_comparison', TM / 'saved_scale_cut_comparison'),
    ('16_width68_ratio_full_prior', TM / 'width68_ratio_geomean'),
]

MAX_WIDTH = 1800

OUT.mkdir(parents=True, exist_ok=True)
for name, stem in FIGURES:
    png = stem.with_suffix('.png')
    pdf = stem.with_suffix('.pdf')
    img = Image.open(png)
    if img.mode not in ('RGB', 'L'):
        background = Image.new('RGB', img.size, 'white')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
    target = OUT / f'{name}.png'
    img.save(target, optimize=True)
    if pdf.exists():
        shutil.copy2(pdf, OUT / f'{name}.pdf')
    print(f'{name}: {img.width}x{img.height}, {target.stat().st_size/1e6:.2f} MB  <- {png.relative_to(REPO)}')
