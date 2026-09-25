"""Render only completed evidence, keeping partial timing explicitly labelled."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
try:
    import tomllib
except ImportError:
    import toml as tomllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'tsz_8192_validation_recovery_20260921'
ELL=np.arange(80,7980)
FACTOR=ELL*(ELL+1)/(2*np.pi)


def dl(path):
    value=np.load(path)
    assert value.shape==(7980,) and np.isfinite(value).all()
    return value[ELL]*FACTOR


def distance(delta, samples):
    cov=np.atleast_2d(np.cov(samples,rowvar=False,ddof=1))
    eigenvalues,vectors=np.linalg.eigh(cov)
    keep=eigenvalues>max(eigenvalues[-1]*1e-10,1e-30)
    return float(np.linalg.norm((delta@vectors[:,keep])/np.sqrt(eigenvalues[keep])))


def main():
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':16,
                         'xtick.labelsize':12,'ytick.labelsize':12,'savefig.dpi':180})
    gate=tomllib.loads((ROOT/'gate/gate.toml').read_text())
    baseline=json.loads((ROOT/'followup/baseline_cache.json').read_text())
    probes=tomllib.loads((ROOT/'followup/probe/cache_probe.toml').read_text())['cases']
    context=np.load(ROOT/'followup/compression_context.npz')
    plan=json.loads((ROOT/'plan.json').read_text())
    records=[]
    for task in plan['tasks']:
        base=ROOT/'controls'/f"{task['id']:03d}"
        marker=base/'status.json'
        if not marker.exists():continue
        status=json.loads(marker.read_text())
        if status['returncode']!=0:
            records.append(dict(id=task['id'],failed=status));continue
        timing=tomllib.loads((base/'benchmark.toml').read_text())
        assert timing['ell_max']==7979 and set(timing['selected_halos_per_case'])=={85224251}
        cases=[]
        for case in task['cases']:
            label=case['label']
            a=dl(base/label/'masked_clean_cl.npy')
            b=dl(OLD/'controls'/f"{case['reference_id']:03d}"/'masked_clean_cl.npy')
            errors={name:distance((a-b)@context[name],context[label+'__'+name])
                    for name in context.files if '__' not in name}
            cases.append(dict(label=label,relative_Dell_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                              moped=errors))
        records.append(dict(id=task['id'],label=task['label'],seconds=status['seconds'],
                            timing=timing,comparisons=cases))
    probe_summary=[]
    for n in (512,256,128):
        subset=[r for r in probes if r['nodes'][0]==n]
        probe_summary.append(dict(nodes=subset[0]['nodes'],
            maximum_central_scaled_error=max(r['central_scaled_quantiles'][-1] for r in subset),
            median_cache_seconds=float(np.median([r['cache_seconds'] for r in subset]))))
    report=dict(utc=datetime.now(timezone.utc).isoformat(),gate=gate,
        default_cache_comparison=baseline,scalar_cache_summary=probe_summary,
        completed_controls=records,expected_controls=len(plan['tasks']),
        diagnostic_256_submitted=False)
    (ROOT/'results/progress.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_probes(probes,baseline)
    lines=['# Catalogue reuse: verified progress', '', 'Snapshot: '+report['utc'], '',
           'The 256-row dataset is held. All tests retain ell=80..7979 and unbinned MOPED.', '',
           '## Completed checks', '',
           f"Shared versus original painter: maximum relative map L2 error {max(gate['painting_relative_l2']):.3g} "
           'on 256 synthetic halos and four pressure models at NSIDE128. This is a small-map equivalence '
           'test; full-catalogue parity is reported separately below. Geometry is identical across those models.', '',
           'HEALPix nested-child tests passed: all child directions select the expected parent, a constant '
           'map remains constant and equal-area averaging preserves integrated flux.', '',
           'Scalar cache audit: 22 parameter combinations, 3 grids, 1024 common off-grid points per combination '
           'and grid (67,584 comparisons). The normalized finite-sphere quadrature remains unchanged.', '',
           '| Grid | Largest error / central y of that halo | Median cache build [s] |',
           '| --- | ---: | ---: |']
    for item in probe_summary:
        lines.append('| '+' x '.join(map(str,item['nodes']))+
                     f" | {item['maximum_central_scaled_error']:.6g} | {item['median_cache_seconds']:.3f} |")
    lines += ['', 'These are pointwise errors over the complete cache domain, including low-mass/low-z '
              'corners outside the actual catalogue. They include the historical angular floor and do not '
              'replace beam-, mask- and noise-weighted spectrum comparisons.', '',
              '![Scalar cache accuracy](plots/cache_probe_accuracy.png)', '',
              '## Newly completed observable-level cache check', '',
              'Default versus doubled cache at the same raw NSIDE8192, with the same beam and mask. '
              'Each model uses its own 64 held-out SO split-noise draws. Values below are local linear '
              'MOPED shifts in noise units, across the two anchor compressors; not posterior biases.', '',
              '| Model | Five retained directions | Nine retained directions |',
              '| --- | ---: | ---: |']
    for item in baseline['results']:
        v5=[r['distance'] for r in item['moped'] if r['rank']==5]
        v9=[r['distance'] for r in item['moped'] if r['rank']==9]
        lines.append(f"| {item['label']} | {min(v5):.4g} to {max(v5):.4g} | {min(v9):.4g} to {max(v9):.4g} |")
    lines += ['', 'The bright/shallow model fails the provisional 0.1-noise-unit interpolation budget '
              'even at the current default cache. This is a renderer-accuracy question; no parameter '
              'combination has been removed from the flat prior. The smaller-cache decision must use '
              'the full-sky results, and the bright case may need a different grid.', '',
              '![Noise-weighted cache accuracy](plots/cache_noise_accuracy.png)', '',
              '## Full-catalogue controls', '',
              f"Completed: {len([r for r in records if 'failed' not in r])}/{len(plan['tasks'])}.", '',
              '| Task | Catalogue + painting [s / sky] | Maximum relative spectrum error vs old 8192 | Largest MOPED shift |',
              '| --- | ---: | ---: | ---: |']
    for r in records:
        if 'failed' in r:
            lines.append(f"| {r['id']} | FAILED | inspect retained logs | unavailable |");continue
        lines.append(f"| {r['label']} | {r['timing']['catalogue_seconds']/r['timing']['cases']:.2f} | "
                     f"{max(c['relative_Dell_l2'] for c in r['comparisons']):.4g} | "
                     f"{max(max(c['moped'].values()) for c in r['comparisons']):.4g} |")
    for r in records:
        if r.get('id')!=3 or 'failed' in r:continue
        lines += ['', 'Completed half-theta test (256 x 256 x 128), relative to the old default cache:', '',
                  '| Model | Relative spectrum norm change | Largest tested MOPED shift |',
                  '| --- | ---: | ---: |']
        for c in r['comparisons']:
            lines.append(f"| {c['label']} | {c['relative_Dell_l2']:.5g} | {max(c['moped'].values()):.5g} |")
        t=r['timing']['timings']
        lines += ['',f"Reading mass/redshift and positions took {t['read_mass_redshift']+t['read_positions']:.3f} s, "
                  f"selection {t['selection']:.3f} s, and painting {t['painting']:.3f} s for all four maps. "
                  'Reading alone is a small fraction of this workload. The large last-chunk tail is the '
                  'motivation for the separately gated greedy-block scheduling experiment.']
    lines += ['', 'Spectrum differences for altered grids or NSIDE are approximation changes, not a '
              'pure refactoring-equivalence criterion. Full process wall time includes Julia startup, '
              'compilation, cache creation, map allocation, transforms and, in selected controls, '
              'additional pixel-averaging experiments. Clean-signal timings exclude per-row noise.', '',
              '## Still pending', '',
              '- Complete serial/read-once/shared-geometry runtime comparison and independent repeat.',
              '- One-axis and combined cache reductions, checked over the full multipole range.',
              '- Full-catalogue parent-pixel comparisons and bright/Battaglia12 16384 references.',
              '- Global renderer error across the prior and held-out SBI calibration after the numerical tests.', '',
              'No lowering of ellmax, narrowing of priors, or dataset submission has been used to obtain speed.', '',
              'The original auxiliary baseline-analysis attempt failed from a Python module-name collision. '
              'The corrected auxiliary entrypoint completed; its old log is retained. The simulation jobs '
              'and physical model were unaffected.', '',
              'Additional tests now running: the cache-exterior continuation that removes a derivative '
              'kink, and greedy scheduling of small halo blocks. Their small numerical gates passed. '
              'The continuation preserves interior values bitwise and the physical support; greedy '
              'scheduling agreed with the original painter to below 9e-17 in relative map norm on '
              '1024 synthetic halos. Full-catalogue validation of these two additions remains pending.', '',
              'Implementation, physics, file list and cluster commands: [README](README.md). '
              'Numeric evidence: [progress.json](results/progress.json). '
              'The completed four-child pixel experiment is reported separately in '
              '[PIXEL_PROGRESS.md](PIXEL_PROGRESS.md); it is not yet a converged coarse-pixel renderer.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=len(records),scalar_probes=len(probes)*1024),indent=2))


def plot_probes(probes,baseline):
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for n,color,marker in [(512,'#333333','o'),(256,'#277da8','s'),(128,'#c46b26','^')]:
        subset=[r for r in probes if r['nodes'][0]==n]
        label=' × '.join(map(str,subset[0]['nodes']))
        ax.plot([r['row'] for r in subset],[r['central_scaled_quantiles'][-1] for r in subset],
                marker+'-',color=color,label=label,ms=5,lw=1)
    ax.set(xlabel='Parameter test',ylabel=r'Maximum $|\Delta y|/y(0)$',yscale='log')
    ax.legend(fontsize=12,ncol=3)
    ax.grid(alpha=.2)
    fig.savefig(ROOT/'plots/cache_probe_accuracy.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    labels=['Battaglia12','FLAMINGO fit','Compact','Bright / shallow']
    for rank,offset,color in [(5,-.18,'#277da8'),(9,.18,'#c46b26')]:
        values=[max(r['distance'] for r in item['moped'] if r['rank']==rank) for item in baseline['results']]
        ax.bar(np.arange(4)+offset,np.maximum(values,1e-14),width=.34,color=color,label=f'{rank} directions')
    ax.axhline(.1,color='black',lw=1,ls='--')
    ax.set(xticks=np.arange(4),xticklabels=labels,yscale='log',ylabel='Local MOPED shift / noise',ylim=(1e-14,2))
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'plots/cache_noise_accuracy.png');plt.close(fig)


if __name__=='__main__':
    main()
