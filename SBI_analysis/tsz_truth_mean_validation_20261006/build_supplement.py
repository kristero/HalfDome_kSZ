#!/usr/bin/env python3
"""Build the follow-up PDF only after idark inference and plot verification.

Run the PDF skill's artifact-operation marker immediately before first PDF
authoring. Render and visually review every resulting page before delivery.
This builder is intentionally unusable with preparation-only evidence.
"""
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               PageBreak, Table, TableStyle, Preformatted)
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter

from diagnostic_metrics import PARAMETERS

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = landscape(A4)
MARGIN = 42
STYLES = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=25, leading=30,
                            textColor=colors.HexColor('#15324F'), spaceAfter=15),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=12, leading=17, spaceAfter=11),
    'caption': ParagraphStyle('caption', fontName='Helvetica', fontSize=10.5, leading=14, spaceAfter=8),
    'code': ParagraphStyle('code', fontName='Courier', fontSize=9, leading=12),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def verify_inputs(root):
    marker = root/'PLOTS_COMPLETE.json'
    if not marker.exists():
        raise RuntimeError('PDF pending: completed idark truth-mean and corner results are required')
    manifest = json.loads(marker.read_text())
    assert sha(root/'INFERENCE_COMPLETE.json') == manifest['inference_marker_sha256']
    for name, expected in manifest['outputs'].items():
        assert sha(root/name) == expected, f'Plot/input changed: {name}'
    return json.loads((root/'TRUTH_MEAN_SUMMARY.json').read_text())


def paragraph(text, style='body'):
    return Paragraph(text, STYLES[style])


def figure(root, name, maximum_height=385):
    filename = root/'figures'/f'{name}.png'
    w, h = ImageReader(str(filename)).getSize()
    scale = min((WIDTH-2*MARGIN)/w, maximum_height/h)
    return Image(str(filename), w*scale, h*scale)


def equation(root, latex, name, maximum_width=700):
    fig = plt.figure(figsize=(10, .7))
    fig.text(.01, .5, '$'+latex+'$', fontsize=22, va='center')
    filename = root/'tmp/pdfs'/f'equation_{name}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    w, h = ImageReader(str(filename)).getSize()
    scale = min(maximum_width/w, 55/h)
    return Image(str(filename), w*scale, h*scale)


def grid_table(rows, widths=None):
    table = Table([[paragraph(str(x), 'caption') for x in row] for row in rows],
                  colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E9F0F5')),
        ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#15324F')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return table


def footer(canvas, document):
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#526170'))
    canvas.drawString(MARGIN, 20, 'HalfDome | Truth-mean validation | Conditional halo100 / mask12345')
    canvas.drawRightString(WIDTH-MARGIN, 20, str(document.page))


def assemble_combined(root, supplement):
    """Preserve the earlier seed/mask evidence and replace its Pearson section.

    Source pages remain intact except for a uniform, current footer. The old
    report is never edited. One updated cover and continuous numbering make
    the combined artifact usable without the conversation or a second PDF.
    """
    previous = root.parent/'tsz_joint_report_20261005/output/pdf/HalfDome_Merlin_seed_mask_and_Pearson_report.pdf'
    expected = '3facbec25ad685c85394805394f017a817dfe8d33f86b6ca364dcc7d9b91f01f'
    if sha(previous) != expected:
        raise RuntimeError('Previously verified seed/mask report changed')
    old = PdfReader(previous)
    new = PdfReader(supplement)
    assert len(old.pages) == 36
    # New cover; earlier contract/method/results/diagnosis; new truth-mean
    # pages; preserved future-MOPED discussion and references.
    selected = [('new', 0, new.pages[0]), ('new', 1, new.pages[1])]
    selected += [('old', i, old.pages[i]) for i in range(2, 26)]
    selected += [('new', i, new.pages[i]) for i in range(2, len(new.pages))]
    selected += [('old', i, old.pages[i]) for i in [33, 35]]
    writer = PdfWriter()
    page_manifest = []
    for number, (source, source_index, page) in enumerate(selected, 1):
        buffer = io.BytesIO()
        overlay = canvas.Canvas(buffer, pagesize=(WIDTH, HEIGHT))
        overlay.setFillColor(colors.white)
        if source == 'old' and source_index == 35:
            # Update the reused references page to name this experiment's
            # completion records. Keep its references and prior-run hashes.
            overlay.rect(MARGIN, 181, WIDTH-2*MARGIN, 54, fill=1, stroke=0)
            provenance = Paragraph(
                'The tables and figures derive from completed HalfDome experiments. '
                'Earlier sources are archived in evidence/. New results are recorded in '
                'INFERENCE_COMPLETE.json and PLOTS_COMPLETE.json. VERIFIED.json and '
                'CLUSTER_VERIFIED.json record checksums for the final PDF and scientific '
                'artifacts; source_snapshot/ preserves the inference code.',
                ParagraphStyle('provenance', fontName='Helvetica', fontSize=11,
                               leading=15.95, textColor=colors.HexColor('#172D40')))
            _, paragraph_height = provenance.wrap(WIDTH-2*MARGIN, 54)
            assert paragraph_height <= 54
            provenance.drawOn(overlay, MARGIN, 229-paragraph_height)
        overlay.rect(0, 0, WIDTH, 36, fill=1, stroke=0)
        overlay.setFillColor(colors.HexColor('#526170'))
        overlay.setFont('Helvetica', 9)
        overlay.drawString(MARGIN, 20, '06 October 2026 | idark | Seed/mask and truth-mean validation | Unbounded primary')
        overlay.drawRightString(WIDTH-MARGIN, 20, str(number))
        overlay.save()
        page.merge_page(PdfReader(buffer).pages[0])
        writer.add_page(page)
        page_manifest.append(dict(page=number, source=source, source_page=source_index+1))
    writer.add_metadata({'/Title': 'HalfDome tSZ: seeds, masks, scale cuts and truth-mean validation',
                         '/Author': 'HalfDome research analysis'})
    writer.add_outline_item('Updated findings', 0)
    writer.add_outline_item('Scientific contract, methods and seed/mask results', 1)
    writer.add_outline_item('Band50 and reserved-test validation', 26)
    writer.add_outline_item('Future MOPED and references', len(selected)-2)
    filename = root/'output/pdf/HalfDome_Merlin_complete_seed_mask_truth_mean_report.pdf'
    with filename.open('wb') as stream: writer.write(stream)
    (root/'PAGE_MANIFEST.json').write_text(json.dumps(page_manifest, indent=2)+'\n')
    return filename, dict(previous_report=str(previous), previous_report_sha256=expected,
                          pages=len(selected), replaced_old_pages=[1, 2]+list(range(27, 34))+[35])


def build(root=ROOT):
    summary = verify_inputs(root)  # fail before creating a misleading partial PDF
    metrics = summary['models']
    (root/'tmp/pdfs').mkdir(parents=True, exist_ok=True)
    (root/'output/pdf').mkdir(parents=True, exist_ok=True)
    story = []

    def page(title):
        if story: story.append(PageBreak())
        story.append(paragraph(title, 'title'))

    page('HalfDome: seeds, masks and parameter recovery')
    story.append(paragraph('One updated report combines the completed 10-by-10 seed/mask study, '
        'bias-control experiments and offset diagnosis with the new reserved-test evaluation. '
        'This corrects the requested correlation definition: '
        '<b>true parameters versus posterior means across distinct reserved test cases</b>. '
        'The earlier report measured correlations among posterior parameters for one observation; '
        'those are different quantities.'))
    story.append(paragraph('All 1,500 reserved odd held-row positions are compared without changing '
        'the earlier method selection. The simulation catalogue is halo100 and the mask is 12345; '
        'pressure parameters and SO noise vary across cases. The 10-by-10 sky experiment remains '
        'the separate fixed-Battaglia test of simulation/mask sensitivity.'))
    story.append(paragraph('Frozen original and band50 ensembles are evaluated in unbounded '
        '(primary) and bounded coordinates. The already trained unbounded band50+580 model is '
        'an additional control; no bounded 580 model was available. Use the coverage, bias, '
        'width and density diagnostics alongside Pearson r.'))
    story.append(paragraph('Each test case uses 9,000 posterior draws. Each illustrated corner '
        'uses 45,000 draws per model. Unbounded samples are not clipped. Exact per-case statistics, '
        'input hashes, examples and full nine-parameter corner PDFs accompany this document.'))
    original = metrics['original_linear']['pearson']
    band = metrics['band50_linear']['pearson']
    story.append(paragraph('<b>New primary result:</b> band50 increases robustness between skies, '
        'while reducing conditional truth-mean correlation: '
        f'P0 {original[0]:.3f} to {band[0]:.3f}; xc {original[1]:.3f} to {band[1]:.3f}; '
        f'beta {original[2]:.3f} to {band[2]:.3f}. '
        'The full-prior mean width cost is +45.6% for band50 and +56.5% for the 580 cut, '
        'larger than the earlier fixed-Battaglia costs. Agreement alone does not establish '
        'minimal information loss.'))

    page('Reader\u2019s guide and evidence hierarchy')
    rows = [['Part', 'Pages and scope'],
        ['Physical and statistical contract', '3-8: pressure parameters, finite projection, signed SO cross-spectrum, fixed noise draws, full grid and shared painting.'],
        ['Seed/mask mitigation experiment', '9-21: all 23 new recipes, selection rule, separate simulation/mask/joint results, widths, joint truth and earlier reserved-prior tests.'],
        ['Offset diagnosis', '22-26: common-reference statistics, clean/noise factorization, smooth/fine controls and implementation/support checks.'],
        ['Banding and scale cuts', '27-29: definition of band50, alternative widths, catalogue reuse and the saved 580-cut tradeoff.'],
        ['Corrected truth-mean comparison', '30-37: 1,500 parameter/noise cases; per-parameter Pearson, scatter, RMSE, signed bias, widths and coverage.'],
        ['Posterior examples', '38-40: original versus band50, bounded and unbounded; two random cases and an explicitly selected stress case.'],
        ['Recommendation and reproducibility', '41-42: conclusions after the prior-wide precision test, code snippets, execution and artifact identities.'],
        ['Future compression and references', '43-44: proposed nuisance-aware MOPED and primary references.']]
    story.append(grid_table(rows, [190, 545]))
    story.append(Spacer(1, 12))
    story.append(paragraph('<b>Evidence hierarchy:</b> full-grid observations test sky sensitivity '
        'at one known Battaglia point; reserved parameter/noise cases test the fixed training sky; '
        'counterfactual spectra test mechanisms. None alone establishes calibration over new '
        'catalogues and masks throughout the pressure prior. Earlier grid posteriors use 90,000 '
        'draws per observation; the new test uses 9,000 per case and 45,000 per corner model.'))

    page('What 50 means, and why it was shared')
    story.append(equation(root, r'\overline{D}_b=\frac{\sum_{\ell\in b}(2\ell+1)D_\ell}'
        r'{\sum_{\ell\in b}(2\ell+1)},\qquad C_{\rm band}=A C A^{\mathsf{T}}', 'bands'))
    story.append(paragraph('D<sub>ell</sub> is the saved signed pseudo angular power spectrum in '
        'D units; b denotes a contiguous multipole band; A is the matrix of normalized weights; '
        'C is the saved full-spectrum covariance. Band50 means 80-129, 130-179, through 7930-7979: '
        '7,900 entries become 158 bandpowers. The factor 2ell+1 counts full-sky harmonic modes. '
        'For a masked sky this is an averaging prescription, not a claim of optimal inverse-variance weighting.'))
    story.append(paragraph('The new recipes shared 50-wide input to make matched covariance and '
        'training comparisons tractable. The original Merlin uses its original unbinned '
        'whitening/PCA chain. Width 50 was not optimized. Widths 25 or 100 are equally implementable, '
        'but require refitted transforms and retrained, independently validated models. '
        'They are not Gaussian sigma=50 observation controls.'))
    story.append(figure(root, 'band_width_illustration', maximum_height=210))
    story.append(paragraph('Actual archived 100/12345 spectrum. Width 25/100 panels illustrate '
        'averaging only; they provide no inference-performance evidence.', 'caption'))

    page('Banding does not require a fresh sky per parameter draw')
    story.append(equation(root, r'y=\frac{\sigma_{\rm T}}{m_e c^2}\int P_e\,\mathrm{d}l_{\rm phys}', 'tsz'))
    story.append(paragraph('The Compton y signal integrates physical electron pressure P<sub>e</sub> '
        'along physical path length l<sub>phys</sub>; sigma<sub>T</sub> is the Thomson cross-section, '
        'm<sub>e</sub> is electron mass, and c is the speed of light. P0, xc and beta control pressure '
        'normalization, radial scale and outer behavior. Six alpha parameters specify their '
        'mass and redshift scalings. All nine definitions and priors are unchanged.'))
    story.append(paragraph('Different halo phases alter the realized spectrum even at identical '
        'pressure parameters. A mask changes the sampled sky and couples multipoles. A fixed-sky '
        'training set can teach the network fine patterns specific to one realization. Banding '
        'removes some of those patterns while also losing potentially useful pressure information. '
        'Robustness therefore has to be measured, rather than inferred from bin width.'))
    story.append(paragraph('Apply the same deterministic A to every parameter/noise simulation '
        'and every catalogue. No fresh N-body seed is required by this operation. For sky-marginalized '
        'SBI, vary catalogue and mask independently of the pressure parameters during training, '
        'possibly drawing repeatedly from a finite cached catalogue pool. Hold out entire catalogues '
        'and masks and check multiple pressure truths. Ten skies at one Battaglia truth do not '
        'establish that validation across the whole prior.'))

    page('The 580 cut is a valid tradeoff candidate')
    story.append(paragraph('The saved cut also uses band50: it keeps 148 bands, ell=580-7979, '
        'and removes 6.33% of raw multipole entries. This discards the largest angular scales; '
        'it is not a small-scale cut or a statement that only 6.33% of parameter information is lost.'))
    story.append(figure(root, 'saved_scale_cut_comparison', maximum_height=205))
    story.append(paragraph('Saved held-out-sky screening results. Truth error and seed/mask spread '
        'are RMS displacements normalized by the original baseline half-width. Width ratios are '
        'geometric means relative to the original model on the same cases; the dotted line marks '
        'the former, arbitrary 1.25 budget. These three diagnostics have different meanings.', 'caption'))
    story.append(paragraph('Mean width costs are +28.7% for simulation seeds, +22.5% for masks, '
        'and +27.0% for the combined subset. The cut slightly improves simulation agreement '
        'and mask spread; the combined result is similar or slightly worse than band50 alone. '
        'Worst individual-parameter costs are +68.1%, +60.0% and +80.7%. Accepting a wider mean '
        'budget makes this candidate eligible, but does not itself establish it as better.'))

    page('Pearson across truths, with errors and uncertainty')
    story.append(equation(root, r'r_p={\rm Corr}_i(\theta_{ip},\mu_{ip}),\qquad'
        r'\mu_{ip}=\mathbb{E}_{q(\theta\mid x_i)}[\theta_p]', 'pearson'))
    story.append(paragraph('i indexes the 1,500 parameter/noise observations, p a pressure parameter, '
        'theta<sub>ip</sub> its known truth, x<sub>i</sub> the trained observation, q the learned '
        'posterior, and mu<sub>ip</sub> its physical posterior mean. The fixed-Battaglia grid cannot '
        'measure r because its truth does not vary.'))
    story.append(figure(root, 'pearson', maximum_height=285))
    story.append(paragraph('Intervals use 1,000 paired row resamples across the same cases. They '
        'describe this test cohort, not cosmic variance or estimator-training uncertainty. '
        'Finite-draw mean errors and split-sample r differences are saved separately.', 'caption'))

    page('All nine truth-mean correlations')
    names = ['original_linear', 'band50_linear', 'lmin580_linear', 'original_logit', 'band50_logit']
    rows = [['Parameter', 'Original U', 'Band50 U', '580 U', 'Original B', 'Band50 B']]
    rows += [[parameter]+[f"{metrics[name]['pearson'][j]:.3f}" for name in names]
             for j, parameter in enumerate(PARAMETERS)]
    story.append(grid_table(rows, [145, 118, 118, 118, 118, 118]))
    story.append(Spacer(1, 12))
    story.append(paragraph('U: primary unbounded; B: bounded. The same 1,500 truths and noisy '
        'observations are used throughout. Paired correlation changes and row-resampling '
        'intervals are saved in paired_pearson_changes.csv. These are point-estimation '
        'diagnostics; coverage is tested separately.'))

    for space, title in [('linear', 'Unbounded: truth versus posterior mean'),
                          ('logit', 'Bounded: truth versus posterior mean')]:
        page(title)
        story.append(figure(root, f'truth_mean_{space}', maximum_height=430))
        story.append(paragraph('Each point is one reserved parameter/noise case. Blue: original; '
            'orange: band50. The dashed identity line indicates exact point recovery; posterior '
            'means need not lie on it for weakly measured parameters.', 'caption'))

    page('High correlation can coexist with bias or shrinkage')
    story.append(equation(root, r'{\rm RMSE}_p=\sqrt{\frac{1}{N}\sum_i'
        r'\left(\frac{\mu_{ip}-\theta_{ip}}{w_p}\right)^2},\qquad'
        r'B_p=\frac{1}{N}\sum_i\frac{\mu_{ip}-\theta_{ip}}{w_p}', 'errors'))
    story.append(paragraph('N is the number of cases; w<sub>p</sub> is the parameter prior width. '
        'RMSE is a scale-normalized point-estimation error; B is signed average bias. '
        'Regression slopes and intercepts distinguish offsets from compression toward the prior. '
        'A slope below one can arise from legitimate Bayesian shrinkage; coverage assesses '
        'posterior uncertainty separately.'))
    story.append(figure(root, 'normalized_rmse', maximum_height=250))
    story.append(paragraph('All primary means retain out-of-prior tails. Small Monte Carlo errors '
        'do not establish accurate density estimation or an unbiased sky-marginalized posterior.', 'caption'))

    page('Signed offsets across the parameter prior')
    story.append(figure(root, 'normalized_bias', maximum_height=290))
    story.append(paragraph('Bias is the signed mean error divided by prior width. Averaging over '
        'the prior can cancel errors of opposite sign near opposite boundaries. Consequently, '
        'small overall bias does not rule out the fixed-Battaglia offsets diagnosed earlier. '
        'The truth-mean scatter and regression slopes also expose shrinkage toward the prior.'))
    for name, label in [('original_linear', 'Original unbounded'), ('band50_linear', 'Band50 unbounded')]:
        slopes = metrics[name]['regression_slope']
        story.append(paragraph(f'{label}, slopes for P0 / xc / beta: '
            f'{slopes[0]:.3f} / {slopes[1]:.3f} / {slopes[2]:.3f}. '
        'Exact point identity has slope 1; a Bayesian posterior mean need not have slope 1.'))

    page('Width costs across the full parameter test')
    story.append(equation(root, r'W_{ip}=q_{0.84,ip}-q_{0.16,ip},\qquad'
        r'G_p=\exp\!\left[\frac{1}{N}\sum_i\ln\frac{W_{ip}^{\rm method}}'
        r'{W_{ip}^{\rm original}}\right]', 'width'))
    story.append(paragraph('q is a physical-parameter posterior quantile; W is the equal-tailed 68% '
        'width; i indexes cases and p parameters. G is the geometric mean width ratio across cases. '
        'Bounded methods use the bounded original as reference; unbounded methods use the unbounded '
        'original. This is a full-prior, fixed-sky test, separate from the earlier Battaglia seed/mask widths.'))
    story.append(figure(root, 'width68_ratio_geomean', maximum_height=270))
    values = [(label, metrics[name]['width68_ratio_all_parameters']) for name, label in
        [('band50_linear', 'Band50 unbounded'), ('band50_logit', 'Band50 bounded'),
         ('lmin580_linear', 'Band50+580 unbounded')]]
    story.append(paragraph('Across all parameters and cases: '+ '; '.join(
        f'{label}: {value:.3f} times the matching original width' for label, value in values)+'.', 'caption'))
    story.append(paragraph('These are empirical costs of the completed preprocessing and retrained '
        'SBI pipelines. They do not isolate irreducible loss from the averaging operator alone.', 'caption'))

    page('Marginal coverage checks the uncertainty statement')
    story.append(figure(root, 'coverage68', maximum_height=260))
    story.append(paragraph('Coverage is the fraction of truths inside each equal-tailed 16%-84% '
        'posterior interval. The dashed line is nominal 68%. The saved CSV also reports 95% '
        'coverage, physical truth log density, normalized widths, bias and regression fits. '
        'Correlations across pressure parameters can make marginal and joint coverage differ.'))
    for space in ['linear', 'logit']:
        original = metrics[f'original_{space}']
        band = metrics[f'band50_{space}']
        values = [(min(r['coverage68']), max(r['coverage68'])) for r in [original, band]]
        label = 'Unbounded' if space == 'linear' else 'Bounded'
        story.append(paragraph(f'{label}: original 68% coverage spans '
            f'{100*values[0][0]:.1f}-{100*values[0][1]:.1f}% across parameters; '
            f'band50 spans {100*values[1][0]:.1f}-{100*values[1][1]:.1f}%.'))

    examples = json.loads((root/'corner_examples.json').read_text())['cases']
    for i, example in enumerate(examples):
        description = 'random reserved case' if i < 2 else 'original-model stress case'
        page(f'Corner {i+1}: '+description)
        story.append(figure(root, f'corner_example{i+1}_compact', maximum_height=410))
        story.append(paragraph(f"Design row {example['design_id']}; held-row position "
            f"{example['held_position']}. Original/band50, bounded/unbounded, with known truth "
            'and 68%/95% two-parameter marginal contours. Full nine-parameter corners are exported separately at large '
            'size. Contour-display smoothing does not smooth data or change inference.', 'caption'))

    page('Recommendation after the reserved-test comparison')
    original = metrics['original_linear']
    band = metrics['band50_linear']
    cut = metrics['lmin580_linear']
    story.append(paragraph('<b>Retain full-range band50 as the tested robustness diagnostic, and '
        'the original as the conditional-information reference.</b> The earlier seed/mask tests '
        'show improved agreement and truth-region recovery for band50. The new truth-mean test '
        'shows an empirical precision cost that cannot be inferred from the small number of removed multipoles. '
        'The change includes preprocessing and retraining; it does not measure averaging loss in isolation.'))
    story.append(paragraph(f"For the primary beta estimate, Pearson r is {original['pearson'][2]:.3f} "
        f"for original, {band['pearson'][2]:.3f} for band50 and {cut['pearson'][2]:.3f} "
        'for the 580 cut. Relaxing the old 25% mean-width budget makes the cut eligible, '
        'but the completed comparisons do not show an overall advantage over full-range band50.'))
    story.append(paragraph('Bounded and unbounded comparisons retain the same observation definitions. '
        'Bounding changes prior support and some coverage results; it does not recover the spectral '
        'information discarded by banding. High truth-mean correlation alone is not a calibration test, '
        'and small signed mean bias can conceal errors that change sign over the prior.'))
    story.append(paragraph('<b>Unresolved optimization:</b> widths 25/50/100 have not been compared '
        'with separately trained models. A bandwidth sweep can reuse saved spectra. A calibrated '
        'sky-marginalized pipeline additionally needs pressure-parameter variation across catalogue '
        'and mask choices, followed by validation on entire unseen catalogues/masks. No completed '
        'test here establishes unchanged original constraints with that full nuisance marginalization.'))

    page('Implementation changes and verification')
    story.append(paragraph('The earlier models and outputs are read-only. The new scripts verify '
        'held-row identifiers, physical truths and priors before sampling. Bounded companions '
        'must use identical observation features. Different parameter transforms run in fresh '
        'interpreters; every network is in evaluation mode.'))
    story.append(Preformatted(
        "positions = np.arange(len(design['held']))[1::2]  # reserved 1,500 cases\n"
        "ids = design['held'][positions]\n"
        "np.testing.assert_array_equal(features['theta_held'][positions], truth)\n"
        "u = ens.sample_u(x, 9000, rng=rng)\n"
        "draws = te.from_u(u, low, high)                  # physical, no clipping\n"
        "mean = draws.mean(axis=0, dtype=np.float64)\n"
        "r = pearson_columns(truth, mean)                 # across test cases",
        STYLES['code']))
    story.append(Spacer(1, 15))
    story.append(paragraph('INFERENCE_COMPLETE.json records source/model hashes, outputs and '
        'runtime. PLOTS_COMPLETE.json verifies the consumed inference products. Per-case moments '
        'are retained; unnecessary full test-row draw cubes are not written. The PBS run uses '
        'one free eligible node, 26 CPUs, 16 GB and a two-hour wall limit.'))
    story.append(paragraph('Files: diagnostic_metrics.py; evaluate_truth_mean.py; plot_truth_mean.py; '
        'submit_idark.py; truth_mean.pbs; build_supplement.py; README.md; FILES_CREATED.md. '
        'The complete metrics and pairwise differences are in data/truth_mean_metrics.csv and '
        'data/paired_pearson_changes.csv. All examples and source provenance are inspectable.'))

    draft = root/'tmp/pdfs/truth_mean_sections.pdf'
    doc = SimpleDocTemplate(str(draft), pagesize=(WIDTH, HEIGHT), leftMargin=MARGIN,
                           rightMargin=MARGIN, topMargin=35, bottomMargin=35,
                           title='HalfDome band50 truth-mean validation', author='HalfDome analysis')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    filename, combined = assemble_combined(root, draft)
    (root/'PDF_BUILT.json').write_text(json.dumps(dict(pdf=str(filename.relative_to(root)),
        sha256=sha(filename), plots_marker_sha256=sha(root/'PLOTS_COMPLETE.json'),
        builder_sha256=sha(Path(__file__)), combined_report=combined,
        visual_review='Required before delivery; PDF building is not layout verification'), indent=2)+'\n')
    print(filename)


if __name__ == '__main__':
    build()
