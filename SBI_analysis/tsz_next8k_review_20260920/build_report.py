"""Build the review from preserved measurements; never change simulation inputs.

Run with a Python environment containing ReportLab, for example WSL python3.
All narrative lives here. REVIEW.md and the PDF are generated from the same
content, while PRE_RUN_TESTS.md is generated from the editable test_plan.json.
"""
from html import escape, unescape
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT = REPO / 'output/pdf/tsz_prior_pipeline_review_20260920.pdf'
PLAN = json.loads((ROOT / 'test_plan.json').read_text())
BEAM = json.loads((ROOT / 'results/beam_noise_resolution.json').read_text())
COST = json.loads((ROOT / 'results/costs.json').read_text())
BASELINE = json.loads((ROOT / 'results/prior_only_baseline.json').read_text())

FONT_ROOT = Path('/usr/share/fonts/truetype/dejavu')
for name, filename in [('DejaVu', 'DejaVuSans.ttf'),
                       ('DejaVu-Bold', 'DejaVuSans-Bold.ttf'),
                       ('DejaVu-Italic', 'DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / filename)))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold',
                             italic='DejaVu-Italic', boldItalic='DejaVu-Bold')
INK = colors.HexColor('#203142')
BLUE = colors.HexColor('#006D93')
BODY = ParagraphStyle('body', fontName='DejaVu', fontSize=10.2, leading=14.8,
                      textColor=INK, spaceAfter=9, splitLongWords=True)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8.6, leading=12.2)
H1 = ParagraphStyle('h1', parent=BODY, fontName='DejaVu-Bold', fontSize=20,
                    leading=25, spaceAfter=16, textColor=BLUE)
H2 = ParagraphStyle('h2', parent=BODY, fontName='DejaVu-Bold', fontSize=11.7,
                    leading=15.5, spaceBefore=7, spaceAfter=7)
EQ = ParagraphStyle('equation', parent=BODY, leftIndent=10, fontSize=10.4,
                    leading=16, spaceBefore=2, spaceAfter=12)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=8.8, leading=12,
                      spaceAfter=0)
HEAD_CELL = ParagraphStyle('headcell', parent=CELL, fontName='DejaVu-Bold',
                           textColor=colors.white)
WIDTH = A4[0] - 38*mm
story, markdown = [], []
section_count = 0


def plain(text):
    return unescape(re.sub('<[^>]+>', '', text))


def section(title):
    global section_count
    if section_count:
        story.append(PageBreak())
    section_count += 1
    story.append(Paragraph(title, H1))
    markdown.append('\n## ' + plain(title) + '\n')


def p(text, style=BODY):
    story.append(Paragraph(text, style))
    markdown.append(plain(text) + '\n')


def sub(title):
    story.append(Paragraph(title, H2))
    markdown.append('\n### ' + plain(title) + '\n')


def table(headers, rows, widths):
    cells = [[Paragraph(escape(str(x)), HEAD_CELL) for x in headers]]
    cells += [[Paragraph(escape(str(x)), CELL) for x in row] for row in rows]
    obj = Table(cells, colWidths=[WIDTH*f for f in widths], repeatRows=1,
                hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#EEF4F7'), colors.white]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#BDCFD8')),
    ]))
    story.extend([obj, Spacer(1, 10)])
    markdown.append('| ' + ' | '.join(headers) + ' |')
    markdown.append('| ' + ' | '.join(['---']*len(headers)) + ' |')
    markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    markdown.append('')


def figure(name, caption, width=WIDTH):
    path = ROOT/'plots'/(name+'.png')
    img = Image(str(path))
    img.drawHeight *= width/img.drawWidth
    img.drawWidth = width
    story.extend([img, Spacer(1, 8), Paragraph(caption, SMALL), Spacer(1, 8)])
    markdown.extend([f'![{plain(caption)}](plots/{name}.png)', ''])


def tests(ids):
    for item in PLAN['tests']:
        if item['id'] not in ids:
            continue
        sub(item['id'] + ' - ' + item['title'])
        # Add spaces to numerical identifiers kept compact in the source plan.
        clean = lambda value: re.sub(
            r'\b(at|with|and|count|selected|Use|use|least|completed|sizes|raw|sphere|diagnostic)(?=\d)',
            r'\1 ', value)
        p(escape(clean(item['test'])))
        p('<b>Acceptance:</b> '+escape(clean(item['pass'])), SMALL)
        p('<b>Evidence now:</b> '+escape(clean(item['status'])), SMALL)


section('tSZ priors, rendering and the next 8k dataset')
p('Technical review | 20 September 2026', H2)
p('<b>Decision supported by the evidence:</b> use the HalfDome spherical-truncation '
  'update as the physical starting point, retain the requested wide physical-uniform '
  'ranges for testing, and validate the renderer and inference controls before '
  'starting the next diagnostic dataset. No new production run was submitted in this review.')
table(['Finding', 'Consequence'], [
    ['The continuous tSZ column is already computed before mapping.',
     'The unresolved-halo problem is pixel sampling of that column. A map cross-spectrum does not repair it.'],
    ['A 2 arcmin beam makes the highest bins relatively unimportant.',
     'Measured errors at ell roughly 2000-6000 remain important in the existing noise metric.'],
    ['Matched clean tests give a 2.02-2.03 time ratio for raw 8192 versus 4096.',
     'This is not a measurement of the proposed spherical, noisy production pipeline.'],
    ['MOPED beats a prior-mean prediction on held-out simulations.',
     'Prior influence remains; flat priors alone cannot remove degeneracies or guarantee calibrated inference.'],
], [.43, .57])
sub('Verified versions and scope')
p('HalfDome branch <b>cluster</b> was fetched and its local HEAD matches origin at '
  '<b>10947ce</b>. The user-confirmed spherical update is <b>3522b71</b>. '
  'The public XGPaint cluster branch remains at <b>5dd0b57</b>; the sphere is implemented '
  'in the HalfDome wrapper, not in that upstream line-of-sight routine.')
p('Two cluster SSH attempts timed out. Calculations here use current fetched source '
  'and preserved outputs from completed cluster experiments. They are a fresh analysis '
  'of those outputs, not fresh cluster timings or a verification of the live cluster checkout.')
p('The completed 8,192-row dataset, its seeds, trained models and earlier study outputs '
  'were left intact. The tables below explicitly distinguish historical production, '
  'separate experiments, and proposed changes.', SMALL)

section('Where tSZ is computed, and what FRB does differently')
sub('Current forward calculation')
p('XGPaint first evolves the gNFW pressure parameters with mass and redshift, then '
  'integrates electron pressure along a continuous line of sight. It stores the projected '
  'profile in an interpolation cache. The painter evaluates that profile at HEALPix '
  'pixel centres, adds the haloes, and only then applies the Gaussian beam. Independent '
  'noise maps, the common mask, cross-power estimation, 40-bin compression and SBI follow.')
p('y(θ) = [σ<sub>T</sub>/(m<sub>e</sub>c²)] ∫ P<sub>e</sub>(r) dl', EQ)
p('m<sub>A</sub> = W [B * y + n<sub>A</sub>], &nbsp; '
  'm<sub>B</sub> = W [B * y + n<sub>B</sub>]', EQ)
p('A compact halo can fall between pixel centres or have its central value overweighted. '
  'The resulting flux error exists before either noise realization is drawn. Smoothing '
  'a wrongly sampled map cannot reconstruct missed halo flux; independent-noise '
  'cross-correlation preserves the same erroneous clean signal in both maps.')
sub('The FRB comparison')
p('The FRB DM calculation evaluates halo columns along selected sightlines, so it '
  'does not require a pixel average for the DM value on a ray. However, '
  '<b>compare_takahashi_sightlines.py::sample_y</b> reads a HEALPix tSZ map, filters '
  'its harmonics with an annular filter and beam, then samples that map at the source '
  'pixels. Its tSZ side therefore has the same map-representation issue.')
sub('Useful alternatives to test')
p('<b>Pre-beam or pixel-integrated painting:</b> project each halo continuously, '
  'convolve with the 2 arcmin beam before sampling, or integrate it over pixels. '
  'This is an implementation option to test at 4096; it is not implemented by this review.')
p('<b>Direct harmonics:</b> transform each axisymmetric profile and sum its harmonic '
  'contribution at the actual halo positions. This retains inter-halo cross terms. '
  'Summing only individual halo powers gives a Poisson/one-halo reference, not the '
  'full masked map statistic. A naive direct harmonic sum over the full catalogue '
  'is expensive and needs an independently benchmarked acceleration.')
p('HEALPix provides sampling, transforms and pixel windows; its pixel window does '
  'not infer a continuous compact halo that the painter never represented. '
  'See the HEALPix discretisation reference [R1].', SMALL)

section('Spherical truncation: geometry and implementation')
p('A projected cut b &lt; X R<sub>200c</sub> with a long line-of-sight integral '
  'includes gas outside the sphere. If the intended gas domain is r &lt; X R<sub>200c</sub>, '
  'the line-of-sight endpoint must depend on the impact parameter. This is a change '
  'in the physical model, not simply a faster integration method.')
p('x = b/R<sub>200c</sub>, &nbsp; L(x) = √(X² - x²), &nbsp; '
  'y<sub>sph</sub>(x) = 2A ∫<sub>0</sub><super>L(x)</super> p(√(x² + u²)) du', EQ)
p('The column is exactly zero for x ≥ X. The default comparison radius is X = 4. '
  'The projected integral must satisfy 2π ∫ x y<sub>sph</sub>(x) dx = '
  '4πA ∫ r²p(r) dr, with both outer bounds X and consistent dimensional factors.')
sub('What commit 3522b71 implements')
p('<b>truncation_comparison/spherical_truncation_profiles.jl</b> introduces '
  '<b>ChordMeanProfile</b>. Rather than log-interpolating a column that goes to '
  'zero at the sphere edge, it caches the positive chord mean '
  'g = y<sub>sph</sub>/(2L). The painter restores the exact factor '
  '2X√[1 - (θ/θ<sub>max</sub>)²]. This separates the known edge geometry from the '
  'smooth quantity being interpolated.')
sub('Remaining dependency exposed by source review')
p('The wrapper currently recovers the normalization using '
  '<b>inner(θ200, M, z) / old_LOS_integral(1)</b>. The ratio is algebraically correct '
  'when both evaluations are reliable, but it unnecessarily invokes the old long-column '
  'integrator. With the previous stable override it can encounter the β &gt; 0.7 assertion; '
  'extreme underflow can also make a ratio unsafe.')
p('<b>Required before production:</b> obtain the physical amplitude directly from '
  'the compatible inner prepared-profile API, or factor that normalization into a '
  'shared tested function. Use a normalized positive quadrature for broad spherical '
  'profiles. These changes are proposed here, not silently applied to the wrapper.')
p('After convolution the observed halo extends beyond its physical sphere. '
  'Pre-beam painting must include beam wings and apply the beam exactly once. '
  'Cutting the convolved profile again at 4 R200 would discard signal.', SMALL)

section('Numerical changes already made for the wider prior')
sub('1. Normalized line-of-sight quadrature - used in the completed 8k')
p('<b>flamingo_linear_prior/stable_los.jl</b> replaces a difficult direct long-column '
  'integration by u = asinh(l/x), so r = x cosh(u). For α = 1 and γ = -0.3, '
  'the transformed positive integrand has log shape:')
p('log f(u) = 0.7 log r + 0.3 log x<sub>c</sub> '
  '- β log(1 + r/x<sub>c</sub>).', EQ)
p('The code subtracts its peak log amplitude before quadrature and restores the '
  'normalization afterwards. For β &gt; 0.7 the stationary radius is '
  'r* = 0.7 x<sub>c</sub>/(β - 0.7), clamped to the integration interval. '
  'This avoids asking the integrator to resolve extremely small absolute values '
  'against an inappropriate scale. The historical integration endpoint and map '
  'definition were retained in that run.')
sub('2. Positive cache floor - used in the completed 8k')
p('A floor max(minimum positive cache value × 10<super>-6</super>, nextfloat(0)) '
  'prevents log(0) when profiles underflow. This removes a numerical failure; it '
  'does not prove the modified faint tail has zero effect on the observable. '
  'A convergence test must bound that effect. True zeros outside a sphere should '
  'be represented by the analytic support factor, not a positive physical tail.')
sub('3. Separate follow-up implementations')
p('<b>tsz_guardrail_study/map_experiment.jl</b> tests log-redshift interpolation, '
  'grid refinement and raw map resolution independently. '
  '<b>tsz_beta_flat_followup_20260920</b> generalizes the finite-LOS normalization '
  'for β ≤ 0.7 by placing the maximum at the upper endpoint, tests finite spheres, '
  'and constructs a continuous correlated beta prior with flat marginals.')
p('Those study implementations did not retroactively change the completed '
  '8k dataset. Nor do their profile-level tests certify the current spherical wrapper '
  'through the full noisy map pipeline. No change to the physical baryon fraction '
  'normalization is proposed; pressure normalization and physical M200c units must '
  'remain consistent with the pinned XGPaint model.')

section('Guardrails: physical statements versus engineering cuts')
table(['Old restriction', 'Reason and disposition for the next model'], [
    ['Evolved beta between 2.8 and 50 on the full cache rectangle',
     '2.8 was a margin above the untruncated finite-energy threshold 2.7; 50 is not a physical singularity. A finite sphere does not need either outer-tail convergence cut.'],
    ['(xc/beta)/(xc/beta)B12 between 0.4 and 8',
     'A compactness proxy, not a measured HEALPix error. Replace with flux and bandpower convergence of the renderer; do not reject compact physical profiles merely because point painting fails.'],
    ['Y200/Y200,B12 between 0.003 and 30',
     'An exploration/dynamic-range choice, not a universal energy law. Test finite integrated pressure directly; retain observational or thermodynamic bounds only as explicit scientific assumptions.'],
    ['Missing central long-LOS tail at most 1 percent',
     'A conservative bound combining extreme xc and beta values, sometimes from different locations. For a sphere, gas outside X is excluded by definition; vary X as model sensitivity.'],
    ['1200-second row timeout',
     'An operational limit. A timeout must remain a recorded failure or be rerun; dropping it changes the prior.'],
], [.38, .62])
p('For p(r) ∝ r<super>-0.3</super>(1 + r/x<sub>c</sub>)<super>-β</super>, '
  'the untruncated outer pressure slope is -(β + 0.3). Total integrated thermal '
  'energy requires β &gt; 2.7; a column to infinity requires β &gt; 0.7. '
  'A finite-radius sphere removes both outer-infinity divergences, while the fixed '
  'central r<super>-0.3</super> cusp remains integrable.')
p('<b>The unused cache corner test</b> checks the entire interpolation rectangle, '
  'including high-mass/high-redshift combinations absent from the catalogue. '
  'Those cells must be numerically computable if the cache builds them, but their '
  'hypothetical infinite-radius energy need not constrain a catalogue-only physical '
  'prior. Changing the catalogue later requires checking its support again.')
p('Finite radius does not establish that every pressure amplitude is astrophysically '
  'realistic. Very shallow profiles can place most thermal energy near the chosen '
  'outer boundary. Treat that radius and pressure normalization as explicit model '
  'assumptions rather than describing all formerly excluded cases as impossible.', SMALL)
p('Measured finite-radius examples at M = 10<super>13</super> M⊙ and z = 2 '
  'illustrate the distinction: increasing the sphere from 4 to 16 R200 gives '
  'energy ratios 1.032, 7.474 and 20.612 for evolved β = 6.269, 1.617 and 0.644. '
  'The last two profiles are finite inside each sphere, but their prediction '
  'depends strongly on the chosen outer radius.', SMALL)

section('Flat priors and the proposed exploration ranges')
p('For each of P0, xc and beta, q(M,z) = q0 (M/10<super>14</super> M⊙) '
  '<super>αm</super>(1+z)<super>αz</super>. The amplitudes and exponents below '
  'are the nine sampled parameters, in the established order.')
table(['Parameter', 'Battaglia12', 'Completed 8k box', 'Extended test box'], [
    ['P0', '18.1', '[1, 60]', '[1, 60]'],
    ['xc', '0.497', '[0.1, 4]', '[0.025, 4]'],
    ['beta0', '4.35', '[2.8, 16]', '[2.8, 16]'],
    ['alpha_m_P0', '0.154', '[-0.2, 1.5]', '[-0.6, 1.5]'],
    ['alpha_m_xc', '-0.00865', '[-0.6, 0.4]', '[-1, 0.4]'],
    ['alpha_m_beta', '0.0393', '[-0.2, 0.4]', '[-0.2, 0.4]'],
    ['alpha_z_P0', '-0.758', '[-4.5, 0.5]', '[-6, 0.5]'],
    ['alpha_z_xc', '0.731', '[-1.5, 2]', '[-1.5, 3]'],
    ['alpha_z_beta', '0.415', '[-0.5, 1.5]', '[-0.5, 2]'],
], [.25, .19, .28, .28])
p('The completed 8k used a uniform proposal in physical values followed by joint '
  'guards. Its accepted density was constant on the surviving joint support, '
  'but its one-dimensional marginals were not uniform. About 2.20 percent of '
  'that particular proposal survived; the earlier 1.70 percent figure belongs '
  'to a different base proposal and must not be reused for this dataset.')
p('<b>Preferred candidate to test:</b> independent physical uniforms in all nine '
  'extended ranges, with the intended finite spherical gas radius. Finite integrals '
  'permit this mathematically. Full numerical stability, catalogue-scale costs and '
  'observable accuracy across the rectangle are still unverified. No further narrowing '
  'or FLAMINGO-fit weighting was introduced here.')
p('<b>Alternative if untruncated finite energy is retained:</b> the separate '
  'continuous beta sampler has flat one-dimensional marginals but correlated '
  'joint support. Its 48³ whole-cell construction was validated on the actual '
  'catalogue hull, with 131,072 draws and no energy violations. This is not a '
  'fully independent rectangular beta prior or a complete new nine-parameter '
  'production configuration.')
p('FLAMINGO overlays remain effective spectrum fits, not nine independent pressure '
  'measurements. The new spherical model requires new clean fits before those '
  'coordinates can be interpreted with it.', SMALL)

section('Beam suppression and where sampling errors matter')
figure('beam_noise_resolution',
       'Same refined cache grid; raw 4096 and 8192 compared with raw 16384. '
       'All clean comparisons include the existing 2 arcmin beam and fsky = 0.4 mask. '
       'The FLAMINGO curve is the HalfDome effective-fit model, not the hydrodynamic map. '
       'Noise power is shown separately from the bandpower uncertainty.')
p('B<sub>ℓ</sub>² = exp[-ℓ(ℓ+1)σ<sub>b</sub>²], &nbsp; '
  'σ<sub>b</sub> = (2 arcmin in radians)/√(8 ln 2).', EQ)
p('The retained power is 0.783 at ℓ = 2000, 0.376 at 4000, 0.111 at 6000 '
  'and 0.0205 at 7979. These are power factors, not the beam amplitudes. '
  'The convention agrees with healpy gauss_beam [R2].')
p('For B12, the last-bin raw-4096 discrepancy is 18.29 percent of the signal '
  'but only 0.0527 of that bin’s noise standard deviation. The largest '
  'noise-scaled discrepancy is instead 3.24 at ℓ ≈ 4180. A large fractional '
  'error in a nearly invisible bin is therefore a poor numerical guardrail.')

section('An accuracy metric tied to the actual statistic')
p('For two implementations of the same physical model, form the difference '
  'ΔD of the same 40 beam-smoothed, masked bandpowers and evaluate:')
p('ε = √(ΔD<super>T</super>C<super>-1</super>ΔD).', EQ)
p('A proposed allowance ε = 0.1 is an explicit precision budget, not a law of '
  'physics. Report sensitivity to 0.05 and 0.3. For equal-covariance Gaussian '
  'likelihoods, the mean-shift KL divergence is ε²/2; in a local linear identifiable '
  'model, the Fisher-metric parameter shift cannot exceed ε. Those statements '
  'motivate the metric, but do not certify a nonlinear SBI posterior.')
table(['Maximum ell', 'B12 ε4096', 'B12 ε8192', 'B12 amplitude information kept'], [
    [str(r['ell_max']), f"{r['error4096']:.3f}", f"{r['error8192']:.3f}",
     f"{100*r['amplitude_information_fraction']:.2f}%"]
    for r in BEAM['cases']['Battaglia12']['cuts']
], [.22, .22, .22, .34])
p('Across all bins, ε4096 is 9.79 for B12 and 1.88 for the FLAMINGO fit; '
  'ε8192 is 0.383 and 0.0260 respectively. Thus raw 8192 strongly reduces '
  'the measured error, but B12 still exceeds the proposed 0.1 budget. '
  'Raw 16384 is a comparison reference, not a proven continuum solution.')
p('Discarding bins above ℓ = 6079 loses only 0.045 percent of the calculated '
  'B12 amplitude information, yet leaves ε4096 = 9.73. The dominant sampling '
  'problem has already entered lower multipoles. The information calculation '
  'uses ∂D/∂ln P0 = 2D with all other parameters fixed; it is not a nine-parameter '
  'information or posterior forecast.')
sub('Limits of the covariance')
p('C comes from 64 independent split-noise realizations on one fixed B12 sky, '
  'using shrinkage for its 40-bin correlations. It excludes cosmic variance, '
  'foreground/model discrepancy and variation in the underlying halo catalogue. '
  'Its use for the FLAMINGO fit is a common reference metric. A new-run certification '
  'must check covariance sampling uncertainty and relevant signal dependence. '
  'Do not label the ε values as a detection significance or a measured posterior bias.')

section('Measured NSIDE cost: compare identical cache grids')
table(['Raw NSIDE / grid', 'B12 time', 'FL fit time', 'B12 / FL peak GiB'], [
    [f"{a['nside']} / {'old' if a['grid_refinement']==1 else 'fine'}",
     f"{a['seconds']/60:.2f} min", f"{b['seconds']/60:.2f} min",
     f"{a['rss_gib']:.2f} / {b['rss_gib']:.2f}"]
    for a,b in zip(COST['clean_eight_thread_experiments'][0]['rows'],
                   COST['clean_eight_thread_experiments'][1]['rows'])
], [.31, .21, .21, .27])
figure('resolution_costs', 'Saved eight-thread clean cluster experiments. The fine cache '
       'takes about 4.3-4.4 minutes independently of raw NSIDE. These tests return '
       'the final map to NSIDE 4096 and use the same harmonic bandlimit.')
p('<b>Matched result:</b> raw 8192 takes 2.02-2.03 times the total clean-test time '
  'and 1.59 times the process peak memory of raw 4096 on the fine grid. '
  'The earlier approximately 3.4 ratio compared 8192/fine with 4096/old and '
  'mixed interpolation-grid cost with resolution cost.')
p('Doubling NSIDE increases raw pixel count fourfold. A float64 map is 1.5 GiB '
  'at 4096 and 6 GiB at 8192. Keeping the final beam-smoothed/noise map at 4096 '
  'avoids making every downstream map four times larger. The physical beam and '
  'observed angular information are unchanged; the purpose is more faithful rendering.')
p('The completed noisy 26-thread production had a 261.45-second median and '
  '602.39 worker-hours for 8,169 generated maps plus 23 imported spectra. '
  'Those settings differ from this clean experiment. New spherical, pre-beam and '
  'full-noise runtime forecasts require the matched pilot in P07.', SMALL)

section('What the completed SBI analysis actually establishes')
figure('prior_only_baseline', 'Normalized held-out RMSE for the completed guarded prior. '
       'The black control always predicts the training-pool parameter mean and never '
       'reads the observation. MAF results are taken from the preserved convergence tables.')
table(['Estimator / control', 'Mean RMSE / range', 'Pooled correlation', 'Test rows'], [
    ['Prior-mean control', f"{BASELINE['mean_normalized_rmse']:.4f}",
     f"{BASELINE['pooled_pearson']:.3f}", '843'],
    ['40 bins, MAF', '0.1819', '0.646', '842'],
    ['MOPED, MAF', '0.1666', '0.714', '843'],
    ['PCA, MAF', '0.1876', '0.618', '843'],
], [.35, .25, .24, .16])
p('MOPED improves the mean normalized RMSE by about 23.4 percent over the '
  'constant prior-mean predictor, so the model has learned some dependence on '
  'the spectra. But even a constant prediction gives pooled correlation 0.403 '
  'when parameters with different marginal means are concatenated. Its '
  'within-parameter correlation is undefined. Pooled correlation alone is '
  'therefore not evidence of successful parameter recovery.')
p('The result is compatible with weak directions and broad degeneracies in a '
  'nine-parameter fit to one power spectrum. A posterior mean displaced from '
  'B12 need not be a network failure if that truth sits along a broad '
  'prior-sensitive degeneracy. Flat priors are the requested design choice, '
  'but this observation alone does not prove they will eliminate inference bias.')
p('The 40-bin table includes 842 of 843 test rows after its finite-result and '
  'minimum-sample filtering. '
  'The next analysis must record every failed posterior rather than omit it '
  'without an outcome. New tests should use per-parameter metrics, known-truth '
  'recovery, density scores and a shuffled-pair training control.', SMALL)

section('Noise cross-spectra and realistic SO observations')
p('Two maps of the same sky with independent noise can be cross-correlated '
  'to remove the mean noise auto-bias. This is a legitimate observing strategy. '
  'For ideal independent Gaussian noise, E[C<sub>AB</sub>] = S; for a fixed '
  'signal and ν effective modes the approximate conditional variance is:')
p('Var(C<sub>AB</sub> | s) = [S(N<sub>A</sub> + N<sub>B</sub>) '
  '+ N<sub>A</sub>N<sub>B</sub>]/ν.', EQ)
p('Consequently, S &lt; N per mode does not imply that a bandpower is '
  'uninformative: many modes reduce its uncertainty. The actual mask couples '
  'modes, which is why the measured pipeline covariance is used for the '
  'numerical comparison rather than the ideal formula alone.')
sub('Two full-depth draws are not two half-survey splits')
p('The existing mock draws each noise map with the tabulated residual Nℓ. '
  'For purely instrumental white noise, splitting a fixed total observing time '
  'into equal halves instead gives approximately 2Nℓ in each half. In the '
  'noise-dominated limit its cross variance is four times that of two '
  'independent full-depth draws. This is a choice of mock definition, not a '
  'reason to alter the finished dataset.')
p('A post-component-separation y-noise table can include residual sky '
  'foregrounds. They are generally shared between observing splits and '
  'cannot be made independent just by choosing different random seeds. '
  'Their cross power can remain in the mean, and their scaling need not '
  'follow observing time. Do not simply double the full ILC table without '
  'identifying its components. The SO forecast is context [R3]; the exact '
  'mock contract is the saved table plus the simulation code.')
sub('Independent rows and FLAMINGO parity')
p('The prior follow-up audited all 16,384 row/split seeds as unique and '
  'separate from the two observation seeds. That establishes the intended '
  'old mock construction; it does not establish that it includes every '
  'real-SO residual or sky-to-sky fluctuation.')
p('For the next run, preserve independent noise between rows and use the '
  'same declared beam, mask, noise convention, bins and compression for '
  'HalfDome and FLAMINGO. Fit clean FLAMINGO spectra separately from '
  'noise. Hydrodynamic y maps contain diffuse gas and cosmology differences '
  'that a finite-radius halo model may not reproduce; a good pipeline can '
  'correctly expose that mismatch.')

section('Before 8k: projection and interpolation tests')
p('These are blockers for declaring the new spherical production pipeline '
  'ready. Existing cylinder-model results are useful controls, not a substitute '
  'for exercising the intended new operator.', SMALL)
tests(['P01', 'P02', 'P03', 'P04'])

section('Before 8k: rendering, resources and the prior')
tests(['P05', 'P06', 'P07', 'D01'])

section('Before 8k: data and inference-interface integrity')
tests(['D02', 'D03', 'D04', 'D05'])
p('A changed physical sphere or renderer invalidates reuse of old clean spectra '
  'as new-model rows. Parameter vectors, catalogue data and other unchanged '
  'inputs may be reused with provenance, but output reuse requires the '
  'complete forward-model identity to match. Hold-out copies must be grouped '
  'by clean sky/parameter setting when repeated noise draws are added.', SMALL)

section('Small-pilot inference controls, then diagnostic 8k')
tests(['I01', 'I02', 'I03', 'I04'])
p('Uniform SBC ranks alone are insufficient: a method that always returns '
  'the prior can pass unconditional SBC while ignoring observations. Pair '
  'calibration with proper held-out density scores, likelihood-positive '
  'controls, conditional coverage and observation dependence.', SMALL)

section('What the diagnostic dataset must decide')
tests(['I05', 'I06', 'I07'])
sub('Recommended sequence')
p('<b>First:</b> repair the spherical normalization dependency in an isolated '
  'implementation, test its integral identities and broad-prior columns, '
  'then compare a flux-conserving 4096 renderer with raw 8192 and selected '
  '16384 controls. No currently measured resolution is an automatic certificate '
  'for every extended-prior point.')
p('<b>Next:</b> freeze the physical-uniform density, complete the small inference '
  'controls and benchmark the entire noisy pipeline. Use 8,192 as the diagnostic '
  'count, with prefix-stable parameter/noise IDs and 524,288 as an adjustable '
  'larger target. The repeated-noise suite should be reserved explicitly '
  'within that design or budgeted as additional noise processing of '
  'saved clean maps or harmonics, not silently counted twice.')
p('<b>Then:</b> assess conditional bias, calibration, prior information and '
  'learning curves before scaling. If nine parameters remain weakly '
  'identified, investigate the statistic or scientific model; more training '
  'examples alone do not create information absent from the observation.')

section('Files created, reproduction and evidence sources')
p('All new analysis code and small outputs live under '
  '<b>SBI_analysis/tsz_next8k_review_20260920/</b>. No existing production source '
  'or historical result was edited for this review.')
table(['New file', 'Purpose'], [
    ['analyze.py', 'Recompute beam/noise comparisons, matched timing tables and the prior-mean inference baseline from saved outputs.'],
    ['test_plan.json', 'Editable structured specification of the 19 tests, pass criteria, current evidence and stages.'],
    ['build_report.py', 'Generate this report, REVIEW.md and PRE_RUN_TESTS.md from measurements and the test plan.'],
    ['finalize.py', 'Hash consumed inputs and delivered outputs; check document text and record page inspection.'],
    ['results/ and plots/', 'Machine-readable numbers, bandpower CSV and three presentation-ready figures in PNG/PDF.'],
    ['inputs/', 'Pinned public XGPaint source snapshots used in the review.'],
    ['artifact_manifest.json and changed_files.txt', 'Hashes, input provenance and inventory for this isolated review.'],
], [.35, .65])
sub('Reproduction')
p('Run <b>python analyze.py</b> in an environment with NumPy and Matplotlib. '
  'Its default entrypoint refreshes pinned public source snapshots over HTTPS. '
  'Run <b>python3 build_report.py</b> in the WSL environment containing ReportLab '
  'and DejaVu fonts. Source data are read from the preserved relative repository '
  'paths, so this review is reproducible within the existing workspace rather '
  'than a standalone copy of the large simulations. After inspecting rendered '
  'pages, run <b>python3 finalize.py</b> to refresh hashes and the file inventory.')
sub('Local evidence')
p('[L1] flamingo_linear_prior: prior.py, prior.json, stable_los.jl, '
  'paint_row.jl, independent_noise.jl and cluster_results/completed_20260918/.<br/>'
  '[L2] tsz_guardrail_study: METHODS.md, RESULTS.md, map_experiment.jl, '
  'audit/noise_covariance.npz and maps/{Battaglia12,FL_L1_m9}/.<br/>'
  '[L3] tsz_beta_flat_followup_20260920: RESULTS.md, continuous_flat.py and results/.<br/>'
  '[L4] linear_prior_sbi: SBI_LINEAR_8K_ANALYSIS_20260918.md, sbi_linear_prior_pipeline.py '
  'and cluster_results/figures/convergence_metrics.csv.<br/>'
  '[L5] truncation_comparison/spherical_truncation_profiles.jl and '
  'frb_map_generation/compare_takahashi_sightlines.py.', SMALL)
p('The manifest records exact source and measurement hashes. Statements about '
  'older tests are based on those preserved outputs; unperformed tests are '
  'explicitly marked pending in the plan.', SMALL)

section('References and interpretation limits')
references = [
    ('R1', 'HEALPix: Discretisation of Functions on the Sphere',
     'https://healpix.sourceforge.io/html/intro_Discretisation_Functions_on.htm',
     'Sampling rationale and pixel-count convention.'),
    ('R2', 'healpy: gauss_beam',
     'https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.gauss_beam.html',
     'Gaussian beam amplitude versus power convention.'),
    ('R3', 'Simons Observatory Collaboration, Science goals and forecasts (2019)',
     'https://arxiv.org/abs/1808.07445',
     'Primary forecast context; the local saved residual table defines this particular mock.'),
    ('R4', 'HalfDome spherical-truncation update, commit 3522b71',
     'https://github.com/kristero/HalfDome_kSZ/commit/3522b71',
     'User-confirmed wrapper implementation reviewed here.'),
    ('R5', 'XGPaint public cluster source, commit 5dd0b57',
     'https://github.com/kristero/XGPaint.jl/tree/5dd0b57cae243598cef9608de77f6807689db712',
     'Pinned upstream profile integration and amplitude definitions.'),
]
for key, title, url, note in references:
    p(f'<b>[{key}] <link href="{url}" color="#006D93">{escape(title)}</link></b>')
    p(escape(note), SMALL)
    markdown.append(f'[{title}]({url})\n')
sub('What is established')
p('The published source implements continuous projection before point painting. '
  'The fetched HalfDome wrapper supplies chord-limited spherical geometry. '
  'Saved matched-grid experiments quantify a large reduction in the existing '
  'sampling discrepancy at raw 8192, and their timings establish the quoted '
  'clean-run cost ratios. The completed SBI results improve on a matched '
  'prior-mean baseline.')
sub('What remains to establish')
p('The new spherical model, full independent flat rectangle, improved compact-halo '
  'renderer and noise/observation contract have not jointly passed a full-sky '
  'preflight. The report is a source audit, measured reanalysis and staged '
  'test specification, not a claim that the next 8k job is ready to submit.')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#BDD0DB'))
    canvas.line(19*mm, 16*mm, A4[0]-19*mm, 16*mm)
    canvas.setFont('DejaVu', 8)
    canvas.setFillColor(INK)
    canvas.drawString(19*mm, 11*mm, 'HalfDome / FLAMINGO | Technical review | 2026-09-20')
    canvas.drawRightString(A4[0]-19*mm, 11*mm, str(doc.page))
    canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
document = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=19*mm,
    rightMargin=19*mm, topMargin=18*mm, bottomMargin=22*mm,
    title='tSZ priors, rendering and the next 8k dataset',
    author='HalfDome analysis review', pageCompression=1)
document.build(story, onFirstPage=footer, onLaterPages=footer)
(ROOT/'REVIEW.md').write_text('\n'.join(markdown)+'\n', encoding='utf-8')
lines = ['# Tests for the next diagnostic tSZ dataset', '',
         'No new production run has been submitted. Current statuses distinguish completed',
         'measurements from tests that must still be run. See REVIEW.md for the physics.', '']
for stage, label in [('before_8k', 'Before starting the new 8k'),
                     ('diagnostic_8k', 'Use the diagnostic 8k to evaluate'),
                     ('before_524k', 'Before scaling to 524k')]:
    lines.extend(['## '+label, ''])
    for item in PLAN['tests']:
        if item['stage'] != stage:
            continue
        lines.extend(['### '+item['id']+' - '+item['title'], '', item['test'], '',
                      '**Pass:** '+item['pass'], '', '**Current status:** '+item['status'], ''])
(ROOT/'PRE_RUN_TESTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({'pdf':str(OUT), 'planned_pages':section_count,
                  'actual_pages':document.page}, indent=2))
if document.page != section_count:
    raise RuntimeError('Unexpected page spill: inspect and adjust page content before delivery')
