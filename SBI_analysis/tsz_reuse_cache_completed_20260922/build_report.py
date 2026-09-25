"""Build the final PDF and Markdown report from verified numerical evidence.

Uses system Python + reportlab, separately from the scientific plotting env.
"""
from html import escape
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PDF = REPO / "output/pdf/tsz_completed_performance_20260922.pdf"
DATA = json.loads((ROOT / "results.json").read_text())
SOURCE = ROOT.parent / "tsz_reuse_cache_20260921"
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle("BodyCustom", fontName="Helvetica", fontSize=10.4, leading=14,
                         spaceAfter=8, textColor=colors.HexColor("#243140")))
STYLES.add(ParagraphStyle("SmallCustom", parent=STYLES["BodyCustom"], fontSize=9, leading=12))
STYLES.add(ParagraphStyle("CellCustom", parent=STYLES["BodyCustom"], fontSize=8.8, leading=11, spaceAfter=0))
STYLES.add(ParagraphStyle("EquationCustom", fontName="Courier", fontSize=10, leading=15, spaceAfter=10,
                         leftIndent=12, textColor=colors.HexColor("#174e72")))
STYLES["Title"].fontSize = 22
STYLES["Title"].leading = 26
STYLES["Title"].alignment = TA_LEFT
STYLES["Title"].textColor = colors.HexColor("#174e72")
STORY, MARKDOWN = [], []


def para(text, small=False):
    STORY.append(Paragraph(text, STYLES["SmallCustom" if small else "BodyCustom"]))
    MARKDOWN.append(re.sub(r"<[^>]+>", "", text))


def title(text, first=False):
    if not first:
        STORY.append(PageBreak())
    STORY.append(Paragraph(text, STYLES["Title"]))
    STORY.append(Spacer(1, 8))
    MARKDOWN.append("# " + text)


def equation(text):
    STORY.append(Paragraph(escape(text).replace("\n", "<br/>"), STYLES["EquationCustom"]))
    MARKDOWN.append("```text\n" + text + "\n```")


def figure(name, height, caption):
    from PIL import Image as PILImage
    path = ROOT / "plots" / (name + ".png")
    with PILImage.open(path) as im:
        width, image_height = im.size
    factor = min(495 / width, height / image_height)
    STORY.append(Image(str(path), width=width * factor, height=image_height * factor))
    STORY.append(Spacer(1, 6))
    para(caption, small=True)
    MARKDOWN.append(f"![{name}](plots/{name}.png)")


def table(headers, rows, widths):
    cells = [[Paragraph(escape(str(value)), STYLES["CellCustom"]) for value in row]
             for row in [headers] + rows]
    t = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eff5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#7a93a5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    STORY.extend([t, Spacer(1, 10)])
    MARKDOWN.extend(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    MARKDOWN.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)


def get_time(group, identifier):
    return next(t for t in DATA["timings"] if t["group"] == group and t["id"] == identifier)


def get_error(group, identifier, label):
    return next(r for r in DATA["errors"] if r["group"] == group and r["task"] == identifier and r["label"] == label)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#617385"))
    canvas.drawString(45, 28, "HalfDome tSZ | Completed performance controls | 22 September 2026")
    canvas.drawRightString(A4[0] - 45, 28, str(doc.page))
    canvas.restoreState()


def main():
    title("Catalogue reuse and cache accuracy", first=True)
    para("<b>Completed results - 22 September 2026.</b> All 13 main full-catalogue controls, three smooth-boundary controls and one balanced-painter control succeeded. Every sky contains 85,224,251 selected halos. The live queue was empty, and 312 retrieved products plus 37 frozen source/input files passed SHA256 checks locally and on idark.")
    para("<b>Recommendation:</b> use shared geometry and balanced halo blocks; the smooth cache boundary with a 256 x 128 x 64 grid is a strong candidate. These components have separate full-catalogue validation; their combined production entrypoint still needs an integration check. The 256-row dataset has not been submitted.")
    figure("painting_time", 235, "Same four pressure models and default cache. These are measured painting times, excluding compilation, reads and harmonic transforms. The repeat exposes runtime variability.")
    rows = []
    for group, identifier, name in [("original_exterior", 0, "Four catalogue passes"), ("original_exterior", 1, "Read once, four painters"),
                                     ("original_exterior", 2, "Shared geometry*"), ("original_exterior", 8, "Shared geometry repeat"),
                                     ("balanced", 0, "Shared + balanced blocks")]:
        t = get_time(group, identifier)
        rows.append([name, f"{t['process_seconds']/60:.2f}", f"{t['painting_seconds']/60:.2f}", f"{t['peak_RSS_GiB']:.2f}"])
    table(["Four-sky control", "Process [min]", "Paint [min]", "RSS [GiB]"], rows, [220, 95, 90, 90])
    para("*Shared-geometry task 002 includes eight extra pixel-averaging observation transforms; use its painting time for a fair kernel comparison. The balanced run took 18.70 minutes end to end: 4.67 minutes per clean sky, 4.67x faster than the four-pass control. Painting alone improved by 5.52x. These are single-run measurements, not a hardware-independent speed guarantee.", small=True)

    title("Cache error: ordinary models and the bright extreme")
    para("The user accepts a larger numerical discrepancy at the bright, shallow extreme; it is retained in the flat prior. Among these four full-sky cases, large cache errors are concentrated there. This is not evidence that every other point in the nine-dimensional prior is accurate.")
    figure("cache_common_reference", 285, "All cells use the SAME reference: smooth-boundary 1024 x 512 x 256 cache at raw NSIDE8192. Values are the larger nine-direction local MOPED shift across the Battaglia12 and FLAMINGO anchors. No multipole binning is used in the metric.")
    rows = []
    names = ["Battaglia12", "FLAMINGO fit", "Compact / faint", "Bright / shallow"]
    for case, name in zip(DATA["cases"], names):
        a = get_error("common_reference", 6, case["label"])
        b = get_error("smooth_exterior", 1, case["label"])
        rows.append([name, f"{a['distance9']:.4g}", f"{b['distance9']:.4g}", f"{100*b['relative_Dell_l2']:.3g}%"])
    table(["Model", "Half grid, old boundary", "Half grid, smooth boundary", "Smooth spectrum norm change"], rows, [135, 120, 120, 120])
    para("For continuity with the earlier report, half-all versus the OLD DEFAULT cache gives 3.051 noise units at the bright extreme (half-theta alone: 3.076). Against the common smooth reference the half-all value is 3.234. These are different reference comparisons, not a change of noise or compression. The smooth half-grid value is 0.00749.", small=True)
    para("The same-worker smooth-grid tests reduced cache construction for four models from 54.45 to 12.12 seconds (4.49x), and the raw values array per model from 128 to 16 MiB. Total process time changed only from 29.72 to 29.46 minutes. Cache construction is a small runtime component; maps dominate memory. The quarter grid offers little additional total-time benefit and has not been tested with the smooth boundary.", small=True)

    title("What the cache-boundary fix changes")
    para("For a finite sphere with radius X = 4 in R200c units, let x be projected radius and p(r) the dimensionless pressure shape. Instead of interpolating a quantity that vanishes with a square-root edge, the code caches the positive chord mean h(x):")
    equation("L = sqrt(X^2 - x^2);  y(x) = A (2L) h(x)\nh(x) = integral_0^1 p(sqrt(x^2 + (X^2-x^2)u^2)) du")
    para("At the edge, h(X) = p(X) and h'(X-) = (2/3)p'(X). The previous dummy continuation outside the sphere used h(x) = p(x), whose derivative is p'(X). Thus values were continuous but the derivative had a kink. Cubic interpolation stencils sample that exterior even for interior evaluation, contaminating pressure near the boundary.")
    para("The new continuation uses the same h(x) expression for x &gt; X. Its square-root argument is a positive convex combination of x squared and X squared. A 16-point Gauss-Legendre rule with log-sum-exp evaluates it. The physical interior branch is unchanged, and the exact chord/disc cut still paints zero outside 4 R200c: <b>no exterior gas is added</b>. The small gate verified identical interior values and matching edge derivatives; the completed doubled-grid test measures the observable-level benefit.")
    figure("cache_spectral_residuals", 260, "Signed spectrum residuals relative to the smooth doubled grid. Units are parts per million. Averages over 100 multipoles are used only for display. Ordinary-anchor and bright/faint cases use separate vertical ranges.")
    para("A small fractional spectrum error can still be many conditional-noise units for the deliberately bright sky. Conversely, tiny noise-weighted errors in the faint compact case do not imply that its profile is well resolved or its parameters are inferable. Interpolation accuracy and information content are separate questions.", small=True)

    title("The LOS normalization is numerical, not physical")
    para("The spherical thermal SZ signal is y(x) = A times the integral of the pressure shape along a finite chord. A contains the dimensional pressure, Thomson-scattering factor and radial-unit conversion supplied by the tSZ profile. It must not be fitted away or divided out of the final sky.")
    equation("y(x) = A * 2 * integral_0^L p(sqrt(x^2+l^2)) dl")
    para("For x &gt; 0 the substitution l = x sinh(u) gives r = x cosh(u) and dl = r du. The integrand is therefore r p(r), not simply p(r). The upper bound is asinh(L/x). At x = 0, the code instead integrates in log(r), again with dr = r dlog(r). This avoids evaluating the integrable central cusp directly; with fixed gamma = -0.3 its central LOS is finite.")
    equation("g(u) = r p(r);  m = max(log g) over the radial interval\nI = integral exp(log(g)-m) du\ny = A * exp(log(2I)+m)")
    para("Subtracting m keeps a positive integrand on a useful numerical scale across extreme amplitudes and concentrations. Restoring exp(m) and the factor of two preserves the physical amplitude exactly, apart from quadrature error. It is analogous to changing units during the calculation, not imposing a gas-energy normalization. The cache then divides the shape integral by 2L, and painting multiplies by the same exact 2L.")
    para("The revised tSZ wrapper obtains A directly from prepare_profile_slice(...).amplitude. Previously, recovering it by dividing a projected profile by the legacy long-LOS integral required evaluating that integral merely to cancel it. That could retain the old slow computation even after introducing spherical projection. The generic non-tSZ fallback remains, but the tSZ and BreakModel methods avoid it.")
    table(["Numerical change", "Reason / contract"], [
        ["Finite chord; exact outer cut", "Counts only gas inside the 4 R200c sphere, unlike a projected disc retaining the long LOS column."],
        ["Logarithmic / hyperbolic LOS variable", "Resolves the central scale and includes the exact integration Jacobian."],
        ["Peak-scaled positive integral", "Improves conditioning; restores the scale in the returned physical integral."],
        ["rtol = 1e-10; maxevals = 4096", "Checks the reported relative integration error and raises a recorded failure instead of silently accepting a bad integral."],
        ["Direct tSZ amplitude", "Avoids calling the expensive legacy LOS merely to normalize it away."],
        ["Log-cache floor = 1e-300", "Prevents log(0) in numerical tails; is not a physical pressure floor or a prior rejection."]], [175, 320])
    para("These changes were already present in the frozen spherical projector used by the completed controls. The new speed experiments do not relax LOS accuracy. A finite outer radius removes the need to require finite total energy at infinite radius, but does not by itself establish observational plausibility throughout the prior.", small=True)

    title("Resolution after the 2 arcmin beam")
    figure("resolution_spectra", 320, "Beam-smoothed masked pseudo-spectra and signed differences from the 16384 reference. The production choice remains 4096 versus 8192; 16384 is only an accuracy reference. Curves use display averaging; errors below use all 7,900 individual multipoles.")
    rows = []
    for label, name in [("Battaglia12", "Battaglia12"), ("FL_L1_m9", "FLAMINGO fit"), ("extended_shallow", "Bright / shallow")]:
        values = [next(r for r in DATA["resolution"] if r["label"] == label and r["pair"] == pair)
                  for pair in ("4096 vs 8192", "8192 vs 16384")]
        rows.append([name, f"{values[0]['distance5']:.3f} / {values[0]['distance9']:.3f}",
                     f"{values[1]['distance5']:.3f} / {values[1]['distance9']:.3f}"])
    table(["Model", "4096 vs 8192: 5 / 9 directions", "8192 vs 16384: 5 / 9 directions"], rows, [155, 170, 170])
    para("<b>For this centre-sampled painter and full ell range, 4096 is not supported as numerically interchangeable with 8192.</b> MOPED does notice the difference at both ordinary anchors. That statement is specific to this renderer and conditional-noise model; it is not a theorem that an accurately pre-beamed or pixel-integrated 4096 map is inadequate for the science.")
    para("At ell = 7979, a 2 arcmin Gaussian beam leaves about 2.05% of the pre-beam power. It suppresses high-ell signal but cannot undo aliasing already created by sampling unresolved halos. Contributions across many multipoles can add coherently in compressed directions. The 8192 residual is much smaller, but Battaglia12 still differs by about 0.39 noise units from 16384 and the bright case by 2.33: 8192 is a practical candidate, not a zero-error reference.", small=True)

    title("Pixel integration: what the finished test establishes")
    para("The tests average four or sixteen fine-grid centre samples into each actual 4096 HEALPix parent pixel, before applying the single Gaussian beam. Parent assignment, preservation of a constant map and integrated-flux conservation passed the numerical gate. These checks establish correct averaging/indexing; they do not establish a converged pixel integral.")
    figure("pixel_refinement", 255, "Both quadrature refinements are compared with the SAME centre-sampled 16384 sky. Points show the larger nine-direction MOPED shift across anchors. Parent-window removal improves with refinement but has not reached a negligible residual.")
    para("For the parent-window-corrected 4096 map, increasing from four to sixteen samples reduces the Battaglia12 discrepancy from 8.58 to 2.17 noise units, and the bright discrepancy from 734 to 187. Without removing the averaging response, the spectra are intentionally more smoothed, so proximity to a centre-sampled reference need not improve monotonically.")
    para("HEALPix defines pixel values as area averages and describes an isotropic pixel-window approximation for their spectrum. A finite set of child-centre samples has a different response from the exact area average. On a regular one-dimensional grid, its discrete response is the parent sinc window divided by the child sinc window; dividing only by the parent window leaves a finite-child residual. That identity is illustrative, not an exact HEALPix correction.")
    para('Reference: <link href="https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm" color="#17618d">HEALPix documentation: Pixel window functions</link>. The documentation also states that the window treatment assumes simplified pixel shapes. Applying its isotropic window is not a substitute for measuring the particular renderer response.', small=True)
    para("<b>Implication:</b> the tested four-child/dewindow recipe is not ready to replace the 8192 painter. A direct coarse-pixel integral or a controlled pre-beam painter remains a possible route, with a measured transfer function and phase/position tests. Building a 16384 map merely to average it is an accuracy experiment, not a production speed improvement.")
    para("The tSZ pressure and LOS integral are already computed before map painting. What is missing for unresolved halos is accurate angular sampling/integration of that projected profile, not moving the pressure calculation ahead of the harmonic spectrum. Cross-correlating noise splits cannot restore signal missed during painting.")

    title("Code changes and the next diagnostic")
    table(["Implementation already tested", "What it does / measured evidence"], [
        ["benchmark.jl: catalogue_pass!", "Reads the original catalogue in chunks once for several pressure models; keeps masses, redshifts and positions common, with independent maps."],
        ["benchmark.jl: paint_shared!", "Reuses halo directions, R200c angular radii, ring/pixel intersections and interpolation coordinates at fixed cosmology and 4 R200c support. Full spectra agree within 8e-16 in relative norm."],
        ["work_balance/balanced_painter.jl", "Distributes 256-halo blocks using greedy scheduling, retaining the same numerical operations and ring locks. Full-catalogue parity is within 8e-16."],
        ["edge_extension/smooth_exterior.jl", "Removes the unused exterior derivative kink without changing physical support; smooth half-grid error is below 0.0075 local noise units on these four controls."],
        ["Cache node counts", "512/256/128 to 256/128/64 nodes; identical coordinate ranges, cubic interpolation, numerical floor and LOS tolerance."]], [180, 315])
    para("Reading fell from 43.73 seconds for four catalogue passes to 7.44 seconds for one pass. That saving alone cannot explain the much larger process-time reduction; painting dominates. The shared-kernel repeat varied from 26.87 to 35.42 painting minutes. Greedy scheduling reduced this to 14.64 minutes, and the final nearby-halo chunk fell from about 865 to 215 seconds. These observations support a load-balancing benefit without pretending the runs were controlled CPU microbenchmarks.")
    para("<b>Candidate configuration:</b> raw NSIDE8192, four pressure combinations per catalogue pass, greedy blocks of 256, smooth cache exterior, 256 x 128 x 64 nodes, output NSIDE4096, unchanged ellmax7979, beam and mask. Keep all nine physical-value priors flat and retain the bright extreme. Shared geometry is valid only while cosmology and support radius are common; each row must still get its own independent SO noise splits.")
    para("Before submitting 256 rows: (1) exercise the combined candidate entrypoint with its independent row seeds, resume bookkeeping and held-out split; (2) measure its end-to-end runtime including noise, since the speed controls are clean; (3) sample renderer error across the flat prior, using per-row checks/refinement rather than rejection based on arbitrary pressure limits. The 22-point scalar probe is useful stress evidence, but only four pressure models have full-sky cache-reduction comparisons.")
    para("For planning only, 64 batches at the measured balanced-default time would be 19.94 worker-hours for 256 clean skies, about 2.49 hours with eight equally fast concurrent workers. This excludes queueing, new noise realizations, final diagnostics and SBI analysis; it is not a promised completion time or a measured combined-configuration forecast.", small=True)
    para("The diagnostic should then test training/held-out residuals, posterior predictive agreement, parameter recovery and simulation-based calibration using unbinned MOPED. Flat priors remove a sampling-density preference but cannot create information about degenerate or noise-dominated parameters. A posterior resembling the prior can reflect weak information as well as a failed pipeline.", small=True)

    title("Metric, scope and reproducibility")
    para("The observable is D_l = l(l+1) C_l/(2 pi) for l = 80,...,7979: 7,900 unbinned entries. All clean comparisons use the same 2 arcmin beam, output NSIDE4096 and apodized f_sky = 0.4 mask (seed 12345, 60 arcmin apodization). The intermediate harmonic transform extends to l = 12287 as in the frozen pipeline. The FLAMINGO anchor is the existing nine-parameter fit, not a newly refitted FLAMINGO map.")
    equation("delta_t = (D_candidate - D_reference) W\nd^2 = delta_t Cov(t_noise)^(-1) delta_t^T")
    para("W is fixed from local derivatives at Battaglia12 or the fiducial FLAMINGO fit. The compression context uses 64 noise draws for the regularized covariance fit and 64 held-out draws for the comparison scale. Each sky uses its own signal-dependent split-noise ensemble. Relative singular cutoffs 1e-3 and 1e-6 retain five and nine directions; the report gives both for resolution and the more sensitive nine-direction values for cache comparisons. Both anchor results are retained in results.json.")
    para("These distances are local linear shifts measured in joint conditional-noise units, not nine-parameter posterior biases or confidence levels. The scale excludes catalogue cosmic variance, foreground residuals and model discrepancy. Maximizing across anchors is a diagnostic convention, not a hypothesis-test calibration. The former 0.1-unit numerical target is not a physical guardrail, and no model is discarded because it exceeds it.")
    para("The 22-model cache probe covers four sky controls, two combined-tail cases and sixteen flat-prior points, at 1,024 common off-grid positions for each of three grids: 67,584 evaluations. It includes unused corners of the interpolation domain, which can stress cache construction even if the catalogue never paints those coordinates. It does not establish global prior coverage or improve the physical plausibility of those unused corners.")
    table(["Artifact", "Purpose"], [
        ["results.json; spectra.npz", "Recomputed metrics, input SHA256 values and plotting spectra."],
        ["timings.csv; errors.csv", "All 17 controls and reference-specific cache comparisons."],
        ["resolution.csv; pixel_quadrature.csv", "Unbinned resolution and finite-quadrature comparisons."],
        ["plots/*.png and *.svg", "Five presentation-size figures, plus editable vector versions."],
        ["cluster_audit.json", "Live empty-queue snapshot and remote source/product verification."],
        ["FILES.md; delivery_manifest.json", "Created files and final deliverable checksums."],
        ["cleanup.json", "Verified duplicate transfer archives removed; scientific evidence retained."]], [230, 265])
    para("New analysis code: analyze_completed.py, make_plots.py, build_report.py, cluster_evidence.py and publish_and_cleanup.py, under SBI_analysis/tsz_reuse_cache_completed_20260922/. Frozen simulation sources and prior partial reports remain as provenance. The new report documents the completed numerical changes; it does not silently enable them in a production generator.", small=True)
    para("Commands: run analyze_completed.py and make_plots.py in the HalfDome scientific Python environment; run build_report.py with system Python + reportlab. Run cluster_evidence.py and publish_and_cleanup.py with Windows Python using the configured idark SSH alias. No script here submits a dataset job.", small=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=42, bottomMargin=45,
                            title="HalfDome tSZ: completed performance and numerical controls", author="HalfDome analysis")
    doc.build(STORY, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "REPORT.md").write_text("\n\n".join(MARKDOWN) + "\n")
    print(PDF)


if __name__ == "__main__":
    main()
