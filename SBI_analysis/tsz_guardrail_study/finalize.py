"""Refresh the measured-results report and record incomplete experiments honestly."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from summarize import prior_plots,quadrature_plot,map_comparisons,pixel_plot,mean_y_plot,native_quadrature_audit
from verify import verify
from audit import save

MAP_CASES=['Battaglia12','FL_L1_m9','xc_low_1','combined_tails','amP0_low_2',
           'amxc_low_2','azP0_low_2','azxc_high_2','azbeta_high_2','beta_steep']
PIXEL_CASES=['Battaglia12','xc_low_1','combined_tails','FL_L1_m9']


def finalize(root):
    prior_plots(root);quadrature_plot(root);map_comparisons(root);pixel_plot(root);mean_y_plot(root)
    native_quadrature_audit(root);verify(root)
    support=json.loads((root/'audit/support.json').read_text())
    quad=json.loads((root/'audit/quadrature_summary.json').read_text())
    comparisons=json.loads((root/'audit/map_comparison.json').read_text())['cases']
    check=json.loads((root/'audit/verification.json').read_text())
    expected=[(name,grid) for name in MAP_CASES for grid in ('historical1','logz1','logz2')]
    expected += [(name,'pixel8192_2') for name in PIXEL_CASES]
    expected += [(name,'pixel16384_2') for name in ('Battaglia12','FL_L1_m9')]
    completed=[];failed=[];pending=[];metadata_notes=[]
    for name,grid in expected:
        path=root/'maps'/name/grid/'status.json'
        if not path.exists():pending.append([name,grid]);continue
        d=json.loads(path.read_text())
        (completed if d['exit_code']==0 else failed).append([name,grid])
        if d.get('name')=='protocol.json':
            metadata_notes.append(dict(path=str(path.relative_to(root)),case=name,
                explanation='Early source-copy loop shadowed the display-name variable. Directory and theta retained the correct case; exact theta consistency checked. Raw record preserved.'))
    status=dict(updated_utc=datetime.now(timezone.utc).isoformat(),planned_maps=len(expected),
        completed_maps=completed,failed_maps=failed,pending_maps=pending,
        planned_map_experiments_complete=not failed and not pending,
        extended_prior_numerically_certified=False,
        missing_for_production_certification=['Converged full-observable pixel reference across admitted regimes',
            'Full-observable finite-LOS error budget across broad joint tails',
            'Covariance and inference calibration appropriate to the new observations and prior'],
        metadata_corrections=metadata_notes)
    save(root/'audit/study_status.json',status)
    lines=['# Guardrail study: measured results','',
        'Updated UTC: '+status['updated_utc'],'',
        '**The active 8k run is unchanged. The extended distribution plotted here is an analytic candidate, not a certified production prior.**','',
        f"Completed full-map experiments: {len(completed)}/{len(expected)}; failed: {len(failed)}; pending: {len(pending)}.",'',
        '## Main metric','',
        'Use the covariance-whitened error in the original 40 clean bandpowers. The reference covariance comes from 64 independent SO split-noise realizations at fixed Battaglia12. It does not include cosmic variance or pressure-model discrepancy. Away from that signal it is a reference precision scale, not the exact candidate posterior uncertainty.','',
        'Accuracy targets 0.05, 0.1 and 0.3 correspond to different explicit numerical budgets. At 0.1 the local identifiable Fisher bias is at most 0.1 standard deviations and the Gaussian mean-shift KL divergence is at most 0.005 nats. These implications, assumptions and derivations are in [METHODS.md](METHODS.md).','',
        '## Prior volume','',
        f"Of {support['proposals']:,} proposals, {100*support['old_acceptance']:.3f}% pass the production prior. Within the same rectangle, {100*support['analytic_same_box_acceptance']:.3f}% pass the finite-energy condition. In the extended rectangle the analytic acceptance is {100*support['analytic_extended_acceptance']:.3f}%.",'',
        f"Independent quadrature gives analytic acceptance {100*check['analytic_normalization']:.5f}%. All {support['production_total']:,} accepted production-prior test points also pass the analytic candidate. This does not certify the numerical accuracy of the newly admitted points.",'',
        '![All-parameter prior comparison](plots/prior_comparison_all_parameters.png)','',
        'Grey: exact frozen 8k design. Purple: analytic extended candidate. Dashed boundaries: 8k rectangle. Gold: original SBI bounds. Black and colored lines: Battaglia12 and effective FLAMINGO fits.','',
        '## Quadrature and LOS endpoint','',
        f"The joint stress audit contains {quad['tests']:,} probes, with {quad.get('failed',0)} tolerance failures. The maximum evolved beta tested is {quad['max_beta']:.5g}; the maximum evaluation count is {quad['max_evaluations']}. Tightening tolerances changes successful results by at most {quad['max_tolerance_change']:.3g} relatively.",'',
        f"Doubling the LOS endpoint changes columns inside 4R200 by at most {quad['max_endpoint_change_inside_painting']:.3g} over these cases. The maximum including unpainted radius 1000R200 is {quad['max_endpoint_change_all']:.3g}. There are {quad['underflows']} correctly recorded floating-point underflows in the successful log-integral tests. These do not by themselves establish map accuracy.",'',
        '![Quadrature and endpoint diagnostics](plots/quadrature_and_los.png)','',
        '## Complete-catalogue interpolation tests','',
        '| Case | Historical vs refined log-z, reference sigma | Coarse vs refined log-z, reference sigma | Historical maximum fractional difference |',
        '|---|---:|---:|---:|']
    for record in comparisons:
        if 'historical_to_logz2_sigma' in record:
            lines.append(f"| {record['case']} | {record['historical_to_logz2_sigma']:.5g} | {record['logz1_to_logz2_sigma']:.5g} | {100*record['historical_max_relative']:.5g}% |")
    lines += ['', 'These are full 85,224,251-halo maps with the same physical pressure model, beam, mask and painting radius. They isolate interpolation changes while retaining the historical pixel-center painter.','']
    if (root/'plots/full_map_interpolation_convergence.png').exists():
        lines+=['![Full-map interpolation convergence](plots/full_map_interpolation_convergence.png)','']
    support_path=root/'audit/los_catalogue_support.json'
    if support_path.exists():
        los=json.loads(support_path.read_text())
        lines+=['## LOS endpoint on occupied catalogue support','',
            f"The {los['tests']:,} additional probes use {los['hull_vertices']} catalogue-hull vertices and {los['occupied_bin_samples']} occupied-bin means for each tested parameter vector. The largest L-to-2L change is {los['worst']['doubled_endpoint_relative_change']:.5g}; its next 2L-to-4L change is {los['worst']['second_doubling_relative_change']:.5g}.",'',
            'The much larger 4.44% cache-domain result above occurs at M=1e12 Msun, z=5, outside the occupied catalogue range. These domains must not be conflated. The occupied-support probes are empirical column tests, not a complete-map bound or a guarantee between all tested points.','']
    timings={grid:[d['timing'][grid] for d in comparisons if grid in d['timing']]
             for grid in ('historical1','logz1','logz2','pixel8192_2','pixel16384_2')}
    lines+=['## Measured resource use','',
        '| Operator | Time range [s] | Largest peak RSS [GiB] |',
        '|---|---:|---:|']
    for grid,rows in timings.items():
        if rows:
            lines.append(f"| {grid} | {min(d['seconds'] for d in rows):.0f}--{max(d['seconds'] for d in rows):.0f} | {max(d['peak_rss_gib'] for d in rows):.2f} |")
    lines+=['','These isolated study processes use eight threads. They measure clean map experiments, including their stated diagnostics; they are not a throughput forecast for the complete noisy 8k production pipeline. Runtime budgets are computational choices, not astrophysical boundaries.','']
    lines+=['## Pixel sampling','',
        'The exact single-halo random-position bound is E[F_pix^2]/F^2 >= max(1, Omega_pix/A_eff), where A_eff=(integral y)^2/integral y^2. It explains rare-hit variance and gives a physically defined resolution diagnostic. It is a bound on flux squared, not on every multipole of a masked catalogue.','']
    if (root/'plots/effective_area_bound.png').exists():
        lines+=['![Effective-area bound](plots/effective_area_bound.png)','']
    lines+=['| Case | 4096 vs 8192 raw sampling, reference sigma | Maximum fractional bandpower difference |',
            '|---|---:|---:|']
    for record in comparisons:
        key='pixel8192_2_from_4096_sigma'
        if key in record:lines.append(f"| {record['case']} | {record[key]:.5g} | {100*record['pixel8192_2_from_4096_max_relative']:.5g}% |")
    for record in comparisons:
        if 'pixel16384_2_from_8192_sigma' in record:
            lines+=['',f"Further refinement for {record['case']}: 4096 to 16384 gives {record['pixel16384_2_from_4096_sigma']:.5g} reference sigma and maximum bandpower change {100*record['pixel16384_2_from_4096_max_relative']:.5g}%. The 8192 to 16384 step gives {record['pixel16384_2_from_8192_sigma']:.5g} reference sigma; maximum relative bandpower change {100*record['pixel16384_2_from_8192_max_relative']:.5g}%."]
    if (root/'plots/full_map_pixel_refinement.png').exists():
        lines+=['','![Further pixel refinement](plots/full_map_pixel_refinement.png)','']
    for record in comparisons:
        if record['case']=='Battaglia12' and 'pixel8192_2_amplitude_only_fit' in record:
            reference='16384' if 'pixel16384_2_amplitude_only_fit' in record else '8192'
            a=record['pixel'+reference+'_2_amplitude_only_fit']
            lines+=['',f"As an identifiable one-parameter example, treating the {reference} map as synthetic data and fitting only P0 with the 4096 template gives P0/P0_input={a['P0_ratio']:.6g}, or a shift {a['signed_power_amplitude_shift_sigma']:.4g} in the conditional power-amplitude standard deviation. This is not a nine-parameter posterior result and does not assert that {reference} is the continuum reference.",'']
    lines+=['','For this paired full-map test the denser raw map is smoothed with the same harmonic beam and band limit, then synthesized at 4096 before the original mask and binning. Agreement at two resolutions alone is not a proof of the continuum limit.','',
        '## Interpretation of the requested tails','',
        'All six requested directions have nonzero analytic support in the candidate rectangle. They are not all unconditionally safe to paint with the historical renderer. The numerical gate must depend on the observable error and amplitude, rather than universal xc/beta or Y200/B12 ratios. At fixed other parameters and fixed covariance, the clean-spectrum error scales as P0^2; therefore a measured complete error epsilon_ref implies P0_max=P0_ref sqrt(budget/epsilon_ref).','',
        'The new rectangle is P0 [1,60], xc [0.025,4], beta [2.8,16], alpha_m_P0 [-0.6,1.5], alpha_m_xc [-1,0.4], alpha_m_beta [-0.2,0.4], alpha_z_P0 [-6,0.5], alpha_z_xc [-1.5,3], alpha_z_beta [-0.5,2]. Its endpoints define an exploration window; they are not claimed physical singularities.','',
        'The old upper Y ratio 30 has no universal pressure-only physical derivation. FIRAS mean-y measurements provide a separate empirical test of total thermal energy; they are not silently inserted into the sampling prior. See the independent mean-y artifacts when available.','',
        '## Remaining evidence','',
        'A usable production revision requires the point-specific fidelity gate to pass, with adequate reference convergence and the intended covariance. Missing pixel or LOS evidence is an explicit failure of certification, not permission to assume safety. The existing 8k run has not been restarted, changed or relabelled.','',
        'Pending planned maps: '+(', '.join(name+'/'+grid for name,grid in pending) or 'none')+'.','',
        'Failed planned maps: '+(', '.join(name+'/'+grid for name,grid in failed) or 'none')+'.','']
    mean_path=root/'audit/mean_y_prior_summary.json'
    if mean_path.exists():
        mean=json.loads(mean_path.read_text())
        lines+=['## Independent mean-y diagnostic','',
            'These comparisons are not imposed as prior exclusions. The numbers refer to a conservative positive halo contribution evaluated with approximate catalogue histogram integration. They omit additional positive components and are not a measurement of total cosmic y.','',
            '| Distribution | Above 5.2e-6 | Above 15e-6 |','|---|---:|---:|']
        for name,d in mean['groups'].items():
            lines.append(f"| {name} ({d['count']} points) | {100*d['lower_bound_above_5p2e6']/d['count']:.3f}% | {100*d['lower_bound_above_15e6']/d['count']:.3f}% |")
        lines+=['','![Independent thermal-energy diagnostic](plots/independent_mean_y_check.png)','']
        means=json.loads((root/'audit/mean_y_cases.json').read_text())['cases']
        lines += [f"Across {len(means)} case checks, maximum change on histogram refinement is {100*max(d['histogram_relative_change'] for d in means.values()):.4g}%; maximum radial-quadrature refinement change is {100*max(d['radial_relative_change'] for d in means.values()):.4g}%.",'']
    lines+=['## Numerical identity check','',
        f"Thirty independent finite-energy checks against 70-digit arithmetic give a maximum normalized quadrature error {check['finite_integral_max_relative_error']:.3g}. The installed SciPy special-function path has maximum error {check['scipy_special_max_relative_error']:.3g}; the detailed version-specific comparison is retained in audit/finite_integral_validation.json.",'']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    files=[p for folder in ('audit','plots','inputs') for p in (root/folder).rglob('*') if p.is_file()]
    files += [root/'RESULTS.md']
    source_root=root/'code' if (root/'code').is_dir() else root
    files += [p for p in source_root.iterdir() if p.is_file() and p.suffix in ('.py','.jl','.pbs','.md','.json')
              and p.name!='artifact_manifest.json']
    files += [p for p in (source_root/'baseline').glob('*') if p.is_file()]
    map_artifacts={'status.json','parameters.toml','numerics.toml','operator_probe.toml',
                   'columns.csv','masked_clean_cl.npy','unmasked_clean_cl.npy','time.txt'}
    files += [p for p in (root/'maps').glob('*/*/*') if p.is_file() and p.name in map_artifacts]
    files=sorted(set(files))
    (root/'changed_files.txt').write_text(
        'All listed files were added in the new study directory. Existing production source and design were not edited.\n\n'
        +'\n'.join(p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')
    files.append(root/'changed_files.txt')
    save(root/'artifact_manifest.json',dict(updated_utc=status['updated_utc'],
        sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();finalize(args.root)
