"""Readable, dated PDF and Markdown report from saved numerical evidence."""
import json
from datetime import datetime,timezone
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OUTPUT=REPO/'output/pdf/tsz_resolution_and_original_xgpaint_20260921.pdf'

def main():
    data=json.loads((ROOT/'results/snapshot.json').read_text())
    stock=json.loads((ROOT/'results/stock_summary.json').read_text())
    jobs=json.loads((ROOT/'results/recovery_submission.json').read_text())
    repeats=json.loads((ROOT/'results/recheck_validation.json').read_text())
    passed_repeats=[r for r in repeats['records'] if r['status']=='passed']
    repeat_summary=(f"{len(passed_repeats)}/6 fresh repeat controls passed; maximum clean-spectrum "
                    f"relative difference {max(r['relative_dl_norm'] for r in passed_repeats):.2e}.")
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyD',fontName='DejaVu',fontSize=10.4,leading=15,spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallD',parent=styles['BodyD'],fontSize=8.6,leading=12))
    styles.add(ParagraphStyle(name='TitleD',fontName='DejaVu-Bold',fontSize=23,leading=28,spaceAfter=17,textColor=colors.HexColor('#18364B')))
    styles.add(ParagraphStyle(name='HeadingD',fontName='DejaVu-Bold',fontSize=16,leading=20,spaceAfter=13,textColor=colors.HexColor('#18364B')))
    story=[];markdown=[]
    def title(text):story.append(Paragraph(escape(text),styles['HeadingD']));markdown.append('## '+text+'\n')
    def p(text,small=False):
        story.append(Paragraph(escape(text).replace('\n','<br/>'),styles['SmallD' if small else 'BodyD']))
        markdown.append(text+'\n')
    def table(rows,widths):
        cells=[[Paragraph(escape(str(v)),styles['SmallD']) for v in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF3')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#638397')),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D4DEE4'))]))
        story.extend([t,Spacer(1,12)])
        markdown.append('| '+' | '.join(map(str,rows[0]))+' |')
        markdown.append('| '+' | '.join(['---']*len(rows[0]))+' |')
        markdown.extend(['| '+' | '.join(map(str,row))+' |' for row in rows[1:]]);markdown.append('')
    def fig(name,width,height,caption):
        story.append(Image(str(ROOT/'plots'/name),width=width,height=height));p(caption,small=True)
        markdown.append(f'![{caption}](plots/{name})\n')
    def page():story.append(PageBreak())
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    story.append(Paragraph('tSZ resolution and original-XGPaint audit',styles['TitleD']))
    p(stamp)
    p('Decision: NSIDE 8192 remains the candidate raw painting resolution; 4096 is not adequate for the full multipole range near Battaglia12 in the existing frozen MOPED comparison. A lower-ell 4096 analysis remains possible, but its information loss and the new-model parameter sensitivity are still being tested. NSIDE 16384 is an accuracy reference only.')
    p('The extended independent uniform prior is numerically evaluable with the finite-sphere wrapper. That does not certify the current point painter over every corner. Extreme-profile sampling errors must be assessed against their own signal and noise levels, not by relative percentages alone.')
    table([['Evidence','Status'],['Cluster small-runtime gate','Passed: bounded LOS, noise algebra, entrypoint allocation, analysis and recovery software tests'],
        ['Independent resolution controls','Both anchors completed at 4096, 8192 and 16384'],
        ['New catalogue controls',f"{data['completed']}/80 completed; 51 pending when the first campaign was stopped"],
        ['Frozen unbinned MOPED','Available: 128 paired split-noise draws per anchor'],
        ['New-model nine-parameter MOPED','Pending the complete two-step derivative set'],
        ['256-row diagnostic dataset','Not submitted']], [177,328])
    p('Operational correction: the first four workers used flock on a Lustre mount configured with localflock. Those locks only exclude writers on the same node. The workers repeated tasks and could overwrite identical-task files. I stopped those jobs and preserved their outputs. The local software test had not exercised cross-node locking.')
    p('Recovery uses disjoint task-ID partitions and atomic directory claims. All retained arrays passed shape, finiteness and source/request checks. Both anchor clean spectra also agree with independent four-thread controls to better than 8e-16 in relative D-ell norm. '+repeat_summary+' The measurements below are a dated snapshot, not a completed production certificate.')
    p('Active replacement workers: '+', '.join(jobs['workers'])+'. Dependent analysis: '+jobs['analysis']+'. A four-node claim test admitted exactly one owner. An initial recovery submission failed before execution because of CRLF script endings; those were corrected to LF and the failed logs were preserved.',small=True)
    page();title('What the existing unbinned MOPED notices')
    p('The same frozen signed-asinh MOPED transform is applied to 7900 individual D-ell values, ell=80..7979. Each noise pair is reused across the two resolutions, while the two splits and different draws are independent. The first 64 draws estimate compressed covariance; the last 64 estimate the paired mean shift. Reported distances use the scatter of one observation, not the error on a 64-mock average.')
    rows=[['ell max','Battaglia12','Historical FLAMINGO fit']]
    for a,b in zip(data['anchors'][0]['cuts'],data['anchors'][1]['cuts']):
        rows.append([a['ell_max'],f"{a['distance']:.3f}",f"{b['distance']:.3f}"])
    table(rows,[95,170,240])
    fig('frozen_moped_cutoffs.png',505,210,'Left: covariance from the first 64 draws. Right: sensitivity to using the independent second covariance half. These are compressed observation shifts, not posterior parameter biases.')
    p('At the full cutoff, changing covariance halves gives 4.16 versus 4.75 for Battaglia12 and 0.830 versus 0.832 for the FLAMINGO-fit anchor. The qualitative B12 conclusion is stable, but the distances have Monte Carlo covariance uncertainty. The old transform was trained under a different prior/noise context. Restricting its weights at lower ell does not retrain the old SBI network.')
    p('No cosmic variance, foreground residuals or model discrepancy is included. Both noise splits retain the historical table multiplier of one; equal half-depth observing splits would be a different noise convention. The new local spherical-model MOPED and retained-information calculation are still pending.')
    page();title('Resolution convergence at the two anchors')
    table([['Anchor','4096 vs 16384','8192 vs 16384'],
        ['Battaglia12','9.789 noise units','0.383 noise units'],
        ['Historical FLAMINGO fit','1.880 noise units','0.0258 noise units']], [185,160,160])
    p('These reference comparisons use the saved conditional 40-bin B12 noise covariance. They are a separate rendering diagnostic, not unbinned MOPED results. That covariance is only approximate at the FLAMINGO-fit anchor. A finite 16384 reference is not the exact continuum solution.')
    fig('resolution_reference.png',505,342,'Matched spherical projection, interpolation, 2 arcmin beam, mask and output grid. Only the raw painting resolution changes; 16384 is reference only.')
    p('For a 2 arcmin Gaussian, beam power transmission is 0.783 at ell=2000, 0.376 at 4000, 0.111 at 6000 and 0.0205 at 7979. The important discrepancy accumulates at intermediate ell, before the most suppressed tail. A 4096 output grid is distinct from painting an unsmoothed compact halo directly at 4096 pixel centres.')
    p('A missed or oversampled central halo column changes the map before smoothing. A later Gaussian beam cannot recover the true flux. Likewise, raising NSIDE does not guarantee every compact profile in an extended prior is resolved. HEALPix documentation distinguishes already band-limited fields from unresolved inputs that alias during discretization.')
    page();title('Joint extremes and interpolation convergence')
    fig('extreme_resolution.png',470,352,'Relative power response to changing raw NSIDE, with identical observation operators. Large relative differences can occur in low-amplitude signals and are not, by themselves, a noise-weighted significance.')
    table([['Profile','Relative D-ell norm: 4096 vs 8192','Doubled-cache change at 8192'],
        ['Compact','266.0%','0.00000462%'],
        ['Extended / shallow','0.0203%','0.000544%'],
        ['Combined tails','8.83%','Not yet tested'],
        ['High-amplitude tails','24.38%','Not yet tested']], [145,190,170])
    p('Default interpolation is 512 x 256 x 128 on log(theta), log(z), log10(M). Refinement doubles every dimension, with the same endpoints and padding. The B12 and FL anchor norm changes are 1.88e-7 and 3.41e-7. The compact case changes by only 4.62e-8 when the cache is refined, despite its large NSIDE response. Increasing interpolation nodes therefore does not fix that pixel-sampling error.')
    p('The compact case uses xc amplitude 0.025 and beta amplitude 16, with the other parameters at Battaglia12. Its 4096 power is about 3.65 times the 8192 power near ell=4000, but its peak D-ell is only 8.9e-22. The extended/shallow peak is 7.8e-9. Therefore a large compact-case percentage alone cannot justify excluding it, while a tiny bright-case error can matter. Jobs 598533 and 598534 now measure 128 paired noise realizations for these two extremes using their own covariances. Those results are pending.')
    page();title('Where the original XGPaint implementation fails')
    p('Original means the local cluster-branch commit 5dd0b57, without the stable-LOS, cache-floor and spherical-wrapper changes. These small probes ran locally under Julia 1.12.2; the full-catalogue tests ran on idark. The numerical LOS file profiles_y.jl is byte-identical to that commit. The literal committed profiles.jl also contains syntax/name errors, unrelated to the prior. Numerical probes use the loadable checkout with those trivial defects corrected.')
    table([['Stage','Failure demonstrated'],
        ['Raw LOS integration','At xc=0.0688725, beta_raw=256 and x=1e-6, stock returns 0; stable integration of the SAME cylinder gives 0.00366282.'],
        ['Reason','All initial Gauss-Kronrod samples underflow to zero over 0..100000 R200. Estimated integral and error are both zero, so refinement stops.'],
        ['Positive-cache cleanup','Original floor=min_positive*1e-6. With min_positive=1e-320 this rounds to zero; log(0) becomes -infinity.'],
        ['Cubic interpolation','Exact original cleanup on a 4x4x4 synthetic cache leaves one nonfinite log node; all 343 interpolated test values are nonfinite.'],
        ['Physical truncation','Cutting the projected disc at 4 R200 does not remove gas beyond the 3D sphere from each LOS.'],
        ['Point painting','No exception is required: unresolved haloes can yield finite but resolution-dependent maps.']], [138,367])
    p('The steep example is inside the proposed parameter ranges: M=1e14 solar masses, z=3, beta_amp=16, alpha_z_beta=2, xc_amp=0.025 and alpha_z_xc=0.731. At x=1e-6 the pressure column is genuinely nonzero. A positive floor cannot repair a central column that the quadrature has incorrectly set to zero.')
    p('At z=5 the same amplitudes give beta_raw=576 and xc=0.0926336: stock again returns zero, while the reference is 0.00277815. B12 and the compact pivot beta=16 control do compute correctly to floating-point accuracy. The statement is not that every draw fails, or that beta_amp=16 is intrinsically invalid; mass/redshift evolution and combinations matter.')
    p('The probes completed without a numerical timeout. The original routine does not impose the new 4096-evaluation budget and ignores the returned error estimate, so difficult cases have a weaker runtime/error contract. This audit demonstrates silent wrong zeros and cache failure, not a claim that these particular probes stalled.')
    page();title('Original profiles versus stable physical projections')
    fig('original_profile_comparison.png',505,379,'Dimensionless projected pressure J(x), before multiplication by the physical halo amplitude. Solid: finite sphere. Dashed: stable integration of the original cylinder. Markers: actual original LOS routine. Zero-valued original outputs are stated explicitly rather than placed on a logarithmic axis.')
    p('For the shallow example beta_raw=0.55735, xc=0.497 and x=0.1, the original cylinder gives J=33.3105, versus 3.01494 inside the 4 R200 sphere: a factor 11.05. Increasing its arbitrary LOS endpoint from 10000 to 100000 to 1000000 R200 changes J from 22.1730 to 33.3105 to 48.7788.')
    p('With alpha=1 and gamma=-0.3, pressure at large radius scales as r^[-(beta_raw+0.3)]. An infinite LOS converges only for beta_raw>0.7; the pressure-volume integral, proportional to total thermal energy, requires beta_raw>2.7. Equality gives a logarithmic divergence. A finite 4 R200 sphere removes those infinite-radius divergences, while the central cusp remains integrable. That permits flat independent beta coordinates mathematically; it does not establish every model as astrophysically realistic.')
    p('The older custom stable-cylinder wrapper additionally asserts beta_raw>0.7. It would explicitly reject this shallow evolved slice. That assertion and HalfDome parameter guards are wrapper behaviour, not a universal restriction imposed by vanilla XGPaint.')
    page();title('Normalization, code changes and measured cost')
    p('The finite-sphere column is J(x)=2 integral from 0 to sqrt(16-x²) of p(sqrt(x²+l²)) dl. The physical signal is y=A(M,z) J. The amplitude A contains P0 and the physical pressure-to-Compton-y conversion. It is never renormalized to unit flux in the production-style map controls.')
    p('For x>0, use l=x sinh(u), so r=x cosh(u) and dl=r du. Set g=log[r p(r)] and c=max(g) on the integration interval. The numerical integral K=integral exp(g-c) du is well scaled; restore J=exp(log(2K)+c). At the centre, use a log-radius integral. The temporary peak scale is restored, so changing it does not change P0 or the physical normalization.')
    p('The new wrapper checks a 4096-evaluation budget and relative error 1e-10. It passed 10180 independent columns, with maximum relative error 7.26e-13, plus 36864 corner projections. It caches the chord mean, restores the exact chord after interpolation, uses log-z, and floors underflow tails at a fixed positive representable value. None of those changes alone repairs point-pixel sampling.')
    table([['Raw NSIDE','Clean B12 walltime, 4 threads','Peak RSS'],
        ['4096','16.0 min','9.43 GiB'],['8192','42.6 min','19.91 GiB'],
        ['16384 reference','148.1 min','58.54 GiB']], [122,255,128])
    p('The 26-thread clean derivative controls have a median walltime of 14.86 minutes. A noise pair costs about 33-34 seconds. These are per-control measurements; the original cross-node duplication wasted aggregate cluster work. The current campaign peak RSS reaches about 27.84 GiB. Removing the unused 6-GiB temporary map and the 13-versus-26-thread comparison are still awaiting their clean benchmarks.')
    p('At 8192, default cache construction takes about 11-15 seconds, while catalogue reading/painting takes roughly 793-998 seconds. Painting dominates. Cache refinement takes 74-103 seconds. Future speed work should prioritize repeated catalogue geometry, interpolation coordinates per halo, lock contention and persistent processes; optimizing the already short scalar quadrature will not remove the main cost.')
    page();title('What remains before a science decision')
    p('1. Finish the remaining independent repeat controls: '+repeat_summary+' 2. Finish derivatives for all eight non-amplitude parameters at both step sizes. 3. Evaluate the new local MOPED directions, derivative convergence and information retained by each cutoff. 4. Finish the allocation and thread benchmarks. 5. Resolve extreme-profile observable fidelity with matched noise and, where needed, a full-catalogue pre-beam or pixel-integrated renderer. Sparse-halo pre-beam tests do not replace full-catalogue validation.')
    p('A 256-row run can then be an engineering/inference diagnostic with explicitly stated numerical limits. It cannot by itself demonstrate precise, calibrated nine-parameter constraints throughout this broad prior. A flat prior removes a sampling preference; it does not cure weak identifiability or simulator error.')
    table([['Independent uniform','P0','xc','beta'],['Pivot amplitude','1 to 60','0.025 to 4','2.8 to 16'],
        ['Mass exponent','-0.6 to 1.5','-1 to 0.4','-0.2 to 0.4'],
        ['Redshift exponent','-6 to 0.5','-1.5 to 3','-0.5 to 2']], [157,116,116,116])
    p('Files added for this report: collect_results.py; run_stock_audit.py; stock_probe.jl; stock_cache_profiles.jl; plot_audit.py; build_report.py; original source snapshots, results and plots. Operational additions: prepare_recovery.py, partition_worker.py, test_partition.py, deploy_recovery.py, repair_submission.py, fetch_recovery.py and test_deployment.py. Further noise tests: queue_extreme_noise.py and analyze_extreme_noise.py. The recovery and extreme-noise directories retain byte-identical physical producer files and add PBS/task orchestration. No new prior limits or map physics were introduced during this recovery.')
    p('Reproducible evidence roots:\nSBI_analysis/tsz_8192_results_20260921\nSBI_analysis/tsz_8192_validation_20260920\nSBI_analysis/tsz_8192_validation_recovery_20260921\nSBI_analysis/tsz_spherical_preflight_20260920',small=True)
    p('Source references: XGPaint commit 5dd0b57, src/profiles_y.jl lines 100-105 (original LOS); src/profiles.jl original cleanup function and logarithmic B-spline construction. Exact source snapshots and SHA256 comparisons are included. HalfDome tSZ_visuals/config.jl validate_battaglia_params and the older flamingo_linear_prior/stable_los.jl distinguish wrapper rejection from library behaviour.',small=True)
    p('QuadGK API: https://juliamath.github.io/QuadGK.jl/stable/api/ - adaptive Gauss-Kronrod rules, error estimate, maximum evaluations and endpoint sampling. HEALPix anafast: https://healpix.sourceforge.io/html/fac_anafast.htm - band limitation, aliasing and harmonic-analysis accuracy. These support the numerical interpretation; quantitative results above come from the saved experiments.',small=True)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('DejaVu',8);canvas.setFillColor(colors.HexColor('#526878'))
        canvas.drawString(45,25,'HalfDome tSZ | test snapshot 2026-09-21');canvas.drawRightString(550,25,str(doc.page))
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=(595.28,841.89),leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=44,
        title='tSZ resolution and original-XGPaint audit',author='HalfDome validation',subject='Dated numerical and cluster test report')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'REPORT.md').write_text('\n'.join(markdown)+'\n')
    print(OUTPUT)

if __name__=='__main__':main()
