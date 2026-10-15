"""Re-render results_final_noise/fig_convergence.png locally (the idark copy lost its minus glyphs).

Nothing is recomputed: the stored prepare.json and <variant>/N<size>/metrics.json of
SBI_analysis/tsz64k_sbi_local/results_final_noise are copied into a scratch folder and sbi_final.cmd_report (the
unchanged plotting code of the pipeline) is run on the copy. The figure is then copied into the report.

    LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python rerender_convergence_noise.py SCRATCH_DIR
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path('/home/cbllover/HalfDome')
SBI = REPO / 'SBI_analysis' / 'tsz64k_sbi_local'
SOURCE = SBI / 'results_final_noise'
FIG = REPO / 'reports' / 'halfdome_overview_20261007' / 'fig' / '3_sbi64k'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('scratch', type=Path)
    args = p.parse_args()
    work = args.scratch / 'results_final_noise_copy'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copy(SOURCE / 'prepare.json', work / 'prepare.json')
    for metrics in SOURCE.glob('*/N*/metrics.json'):
        target = work / metrics.parent.parent.name / metrics.parent.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(metrics, target / 'metrics.json')
    sys.path.insert(0, str(SBI))
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['axes.unicode_minus'] = False      # ASCII minus: no missing-glyph boxes
    import sbi_final
    sbi_final.cmd_report(argparse.Namespace(out=work))
    shutil.copy(work / 'fig_convergence.png', FIG / '13_convergence_40band_noise_variants.png')
    print('written', FIG / '13_convergence_40band_noise_variants.png')


if __name__ == '__main__':
    main()
