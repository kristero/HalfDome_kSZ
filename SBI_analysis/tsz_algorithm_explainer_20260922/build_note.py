"""Create a standalone illustrated code note and its Markdown counterpart."""
from html import escape
import json
from pathlib import Path
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
PDF = ROOT.parents[1] / "output/pdf/tsz_algorithms_illustrated_20260922.pdf"
DATA = json.loads((ROOT / "illustration_data.json").read_text())
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle("BodyNote", fontName="Helvetica", fontSize=10.5, leading=14.2, spaceAfter=8))
STYLES.add(ParagraphStyle("CaptionNote", parent=STYLES["BodyNote"], fontSize=9, leading=12))
STYLES.add(ParagraphStyle("CellNote", parent=STYLES["BodyNote"], fontSize=9, leading=12, spaceAfter=0))
STYLES.add(ParagraphStyle("CodeNote", fontName="Courier", fontSize=8.5, leading=12, spaceAfter=9,
                         textColor=colors.HexColor("#24516c")))
STYLES["Title"].fontSize = 21
STYLES["Title"].leading = 25
STYLES["Title"].alignment = TA_LEFT
STYLES["Title"].textColor = colors.HexColor("#175978")
STORY, MD = [], []


def title(text, first=False):
    if not first: STORY.append(PageBreak())
    STORY.extend([Paragraph(text, STYLES["Title"]), Spacer(1, 8)])
    MD.append("# " + text)


def p(text, small=False):
    STORY.append(Paragraph(text, STYLES["CaptionNote" if small else "BodyNote"]))
    # Retain readable links in the Markdown version.
    text = re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>', r'[\2](\1)', text)
    MD.append(re.sub(r"<[^>]+>", "", text))


def code(text):
    STORY.append(Preformatted(text, STYLES["CodeNote"]))
    MD.append("```text\n" + text + "\n```")


def fig(name, maxheight, caption):
    path = ROOT / "plots" / (name + ".png")
    with PILImage.open(path) as image: width, height = image.size
    factor = min(495 / width, maxheight / height)
    STORY.extend([Image(str(path), width=width*factor, height=height*factor), Spacer(1, 5)])
    p(caption, small=True)
    MD.append(f"![{name}](plots/{name}.png)")


def table(headers, rows, widths):
    content = [[Paragraph(escape(str(value)), STYLES["CellNote"]) for value in row] for row in [headers]+rows]
    t = Table(content, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eff5")),
                          ("VALIGN", (0, 0), (-1, -1), "TOP"),
                          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f8")]),
                          ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    STORY.extend([t, Spacer(1, 10)])
    MD.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"])
    MD.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#657785"))
    canvas.drawString(44, 27, "HalfDome tSZ | Illustrated algorithm note | 22 September 2026")
    canvas.drawRightString(A4[0]-44, 27, str(doc.page)); canvas.restoreState()


def main():
    title('What "smooth boundary; half all" means', first=True)
    p("This label combines two cache changes: a smooth mathematical continuation across the halo's sphere edge, and half as many interpolation nodes on each of three axes. It does not refer to smoothing the physical gas edge, changing the Gaussian beam, reducing map NSIDE or halving the prior. Greedy scheduling and child-pixel averaging are separate experiments.")
    fig("spherical_chord", 245, "A cross-section through the adopted spherical halo. The blue segment is the allowed LOS chord. Gas beyond the sphere does not contribute. X = 4 and all distances here are in R200c units.")
    p("For projected radius x = R_perp/R200c, the physical Compton-y is a finite LOS integral. Let p(r) be the dimensionless gNFW pressure shape, A its physical tSZ normalization, and L the half-length of the chord:")
    code("L(x) = sqrt(X^2 - x^2),  X = 4\ny(x) = A * 2 * integral_0^L p(sqrt(x^2 + l^2)) dl\nh(x) = integral_0^1 p(sqrt(x^2 + (X^2-x^2)u^2)) du\ny(x) = 2L * [A h(x)]       for x < X; otherwise y = 0")
    p("The cache stores A h(x), not the final y(x). Dividing the LOS integral by the chord length gives a finite, positive quantity at the edge, suitable for log interpolation. Painting restores the exact 2L and enforces zero outside. The square-root edge remains in the physical projection; the cache alone is smooth.")
    p("The boundary is the moving surface theta = 4 theta200c(M,z) inside the rectangular interpolation domain. It is not the edge of the sky mask or the minimum/maximum cache coordinate. Its location changes with halo mass and redshift, so interpolation stencils on all three axes can cross it.")

    title("Why the previous cache had a kink")
    fig("smooth_boundary", 295, "Analytic-shape illustrations at fixed alpha = 1 and gamma = -0.3. Shaded regions are outside the physical sphere. The old and new interior functions coincide before interpolation; the final gas projection remains zero outside. The derivative panel exposes the removed kink.")
    p("Inside the sphere, both versions use the same chord mean. The earlier cache used p(x) as a positive placeholder outside the sphere. This matched the value at x = X but not the derivative:")
    code("h(X) = p(X)\nh'(X-) = p'(X) * integral_0^1 (1-u^2) du = (2/3)p'(X)\nprevious exterior: h'(X+) = p'(X)")
    p("A cubic spline uses neighboring cache coefficients. The non-smooth placeholder can therefore influence the interpolated result just inside the sphere, although the painter never directly paints exterior gas. Taking logarithms does not eliminate this derivative discontinuity. A coarser grid can make its effect more visible.")
    p("The new exterior uses the same integral expression for h(x) when x &gt; X. Its radial argument stays real and positive: r squared = x squared times (1-u squared) + X squared times u squared. This is an auxiliary continuation of the analytic pressure shape; it is not the physical pressure outside the sphere. It matches the limiting value and slope at X.")
    p("The exterior is evaluated with a 16-point Gauss-Legendre rule on u in [0,1]. The code computes log-pressure, subtracts its largest value before summing, then restores that scale. This avoids underflow in unused steep cache tails. The physical interior LOS still uses the existing bounded adaptive quadrature. There is no new physical amplitude normalization.")

    title("Exact code path and the half-sized grid")
    p("edge_extension/entrypoint.jl loads the frozen benchmark with automatic execution disabled, includes smooth_exterior.jl to replace the chord_mean method in that process, and then runs main_benchmark(). The experimental override does not edit the installed XGPaint package. The change to the branch is:")
    code("# Interior: unchanged in the experiment\nif x < X\n    L = sqrt((X-x)*(X+x))\n    return chord_quadrature(x, xc, alpha, beta, gamma, L)/(2L)\nend\n# Previous: generalized_nfw(x, xc, alpha, beta, gamma)\n# New:\nreturn smooth_exterior_mean(x, xc, alpha, beta, gamma, X)")
    fig("cache_nodes", 200, "Actual node counts and the Float64 values array only. Additional interpolation coefficients and temporary arrays are not included. The sky maps are separate allocations.")
    p("The tested task sets nodes = [256,128,64], in the order ln(theta), ln(z), log10(M/Msun). build_cache passes these counts to LinRange, leaving each axis endpoint unchanged. The count changes from 16,777,216 to 2,097,152 values: an eightfold reduction. The grid spacings increase by about two, rather than exactly two, because an N-node axis has N-1 intervals.")
    p("The angular range still comes from RadialFourierTransform(n=512, pad=256); that n is not changed to 256. The redshift interval stays 0.001 to 5; the mass interval starts at log10(M/Msun) = 12 and retains the configured maximum. Cache values retain the absolute 1e-300 floor, log transformation and cubic B-spline interpolation. The exact halo chord is restored after interpolation.")
    p("Pressure parameters, halo selection, physical sphere radius, LOS tolerance, beam, mask, raw NSIDE8192 and ellmax7979 are unchanged by these two cache edits. The completed smooth-half task used shared-geometry painting with static scheduling and pixel_targets = []; its accuracy result does not already include the separate greedy or pixel-averaging changes.", small=True)

    title("What was demonstrated, and where to inspect it")
    p("The boundary gate evaluated the original and replacement interior methods on four pressure shapes and six radii. Interior values were bitwise identical; the physical support check passed. Finite-difference edge derivatives agreed with the analytic limit to better than 1.7e-5 relative error. These are source-level checks, separate from the full-catalogue spectrum test.")
    rows=[]
    pretty={"Battaglia12":"Battaglia12", "FL_L1_m9":"FLAMINGO fit", "compact":"Compact / faint", "extended_shallow":"Bright / shallow"}
    for row in DATA["measured_smooth_half_errors"]:
        rows.append([pretty[row["label"]], f"{row['distance5']:.4g}", f"{row['distance9']:.4g}", f"{row['relative_Dell_l2']:.3g}"])
    table(["Smooth half vs smooth doubled", "5 directions", "9 directions", "Relative spectrum norm"], rows, [180,95,95,125])
    p("All models use 85,224,251 halos, the same 2 arcmin beam and mask, and all unbinned multipoles ell=80..7979. MOPED distances use each model's held-out conditional SO noise scale, taking the larger result from two local anchor compressions. They are not posterior biases. The doubled cache is a convergence reference, not an exact analytic sky.")
    p("The four-model smooth cache build fell from 54.45 to 12.12 seconds; total clean process time changed from 29.72 to 29.46 minutes. The maps and painting dominate cost. A larger error at the bright extreme is acceptable under the user's stated criterion; no pressure combination is removed on that basis. A good result on four models is not a global numerical-error guarantee over all nine parameters.")
    table(["Existing source (under tsz_reuse_cache_20260921)", "Role"], [
        ["spherical_truncation_profiles.jl:69", "Bounded normalized physical LOS quadrature."],
        ["spherical_truncation_profiles.jl:113", "Exact chord factor and physical support."],
        ["edge_extension/smooth_exterior.jl:8", "16-node positive exterior continuation."],
        ["edge_extension/smooth_exterior.jl:26", "Interior-preserving method override."],
        ["edge_extension/entrypoint.jl:1", "Load order that activates the experimental method."],
        ["edge_extension/controls/001/task.toml:1", "The actual smooth-half task: nodes and raw NSIDE."],
        ["benchmark.jl:26", "Grid construction and cubic log interpolation."],
        ["benchmark.jl:77", "Restore the chord during pixel painting."],
        ["benchmark.jl:154", "Signal transform, beam and output-map synthesis."]], [315,180])
    p("This note adds explanation and figures, not a new simulator implementation. make_figures.py saves hashes of these sources. Its boundary curves are independent numerical illustrations of the same formulas, while the table above comes from the already completed cluster results.", small=True)

    title("Child-centre averaging: the sample points")
    p("A parent pixel is one pixel of the lower-resolution map. HEALPix's nested hierarchy splits it into four equal-area children when NSIDE doubles, and sixteen descendants when NSIDE quadruples. A child centre is simply the sky direction at the centre of one such smaller pixel; it is not a halo centre.")
    fig("child_centres", 235, "Actual HEALPix boundaries and centres for one equatorial NSIDE4096 parent, drawn in a tangent projection. All three panels cover the same sky area. The dots mark the evaluation locations; geometry was checked by assigning every child centre back to its parent.")
    p("The centre-painted fine map contains y(n_child) at each dot. average_children sums those map values and divides by the number of children. For the four-child experiment:")
    code("y_parent ~= [y(n1)+y(n2)+y(n3)+y(n4)]/4\nexact area average = (1/Omega_parent) * integral_pixel y(n) dOmega")
    p("The division matters: y is a surface-brightness-like field, not a total flux per pixel. Because the children have equal area, this arithmetic mean conserves the integrated flux of the discrete fine map and preserves a constant map. It does not guarantee the true continuous halo flux: a sufficiently narrow peak can fall between all sample points.")
    p("For 8192 to 4096 there are four samples; for 16384 to 4096 there are sixteen. Increasing their number is numerical quadrature refinement. The implementation changes RING indices to NESTED indices to locate descendants, reads the RING fine map and returns a RING parent map. This index conversion does not rotate or smooth the sky by itself.")
    p('The <link href="https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm" color="#17618d">HEALPix pixel-window documentation</link> defines area averaging and an isotropic approximation to its harmonic response. The finite child-centre quadrature has its own response, so dividing by the parent pixel window alone is not an exact correction. The completed four- and sixteen-point tests have not established a converged replacement painter.')
    p("This was an experimental averaging path. The ordinary 8192-to-4096 pipeline uses harmonic reconstruction after the beam, described on the final page. It does not call average_children.")

    title("Greedy scheduling: keep available workers busy")
    p("Each halo is a piece of computational work. Nearby or large-angular-size halos touch more pixels and rings, so equal halo counts need not take equal time. Static scheduling assigns fixed contiguous chunks to workers. If expensive halos are concentrated near the end of the catalogue, one worker can remain busy after others finish.")
    fig("greedy_scheduling", 250, "Illustrative schedule for the SAME sixteen blocks on four workers, with identical block costs in both panels. Each color identifies a block. White gaps are idle time. This is an algorithm illustration, not a captured trace of the cluster threads.")
    code("# Previous shared painter\nThreads.@threads :static for i in eachindex(masses)\n    # paint halo i\nend\n\n# Separate balanced-painter experiment\nblock_size = 256\nThreads.@threads :greedy for first in 1:block_size:length(masses)\n    for i in first:min(first+block_size-1, length(masses))\n        # same painting arithmetic and ring locks\n    end\nend")
    p("When a worker finishes one block, it takes the next available block. The block size avoids creating scheduling overhead for every individual halo. It is a performance setting, not a mass cut, halo selection or pressure parameter. All halos and all four maps are still processed; no block is subdivided midway through its work.")
    p("The completed four-map default-cache test reduced painting from 1,612 to 879 seconds, and the last catalogue chunk from about 865 to 215 seconds. The full spectra agree to about 8e-16 relative norm. Different addition order can change floating-point rounding, so the expected criterion is numerical equivalence, not bitwise-identical sums.", small=True)
    p('The implementation is work_balance/balanced_painter.jl:10. Julia describes <link href="https://docs.julialang.org/en/v1/base/multi-threading/" color="#17618d">:greedy scheduling</link> as workers taking further iterator values as they become available, suited to unequal workloads. The code keeps ring locks to protect shared map additions; it does not use thread-ID-indexed mutable buffers.', small=True)

    title("Why paint at 8192 and form the output at 4096?")
    fig("output_pipeline", 128, "The signal path. Noise splits are added and masked consistently afterward in the diagnostic observation operator. Clean performance controls stop at the masked signal spectrum. The Gaussian beam is applied once.")
    p("Here output means the map used for masking and noise; the stored spectrum still has 7,900 unbinned multipoles. Raw halo profiles contain sharp cores and an edge before the beam; a denser painting grid reduces errors in their harmonic coefficients. After retaining coefficients through ell=12287 and applying the 2 arcmin beam, the code evaluates the smoother harmonic field on a 4096 grid. It does not average four raw child pixels.")
    fig("sampling_before_after_beam", 185, "The left two panels are a 1D Gaussian-halo illustration, not a full-sky convergence test. The beam plot uses the actual 2 arcmin transfer function. More accurate initial sampling and a smaller final smooth-map grid can coexist.")
    table(["Output NSIDE", "Pixels", "sqrt(pixel area)", "One Float64 map"], [
        ["4096", "201,326,592", "0.859 arcmin", "1.5 GiB"],
        ["8192", "805,306,368", "0.429 arcmin", "6 GiB"]], [100,140,130,125])
    p("Using 4096 therefore saves a factor of four in map storage for the output signal, mask and each noise split. Harmonic coefficient storage at fixed lmax is unchanged, and transform runtime does not necessarily improve by four. The science cutoff stays ell=7979. The beam leaves 2.05% of signal power there and about 0.00995% at ell=12287; these factors motivate the choice but do not prove sufficient accuracy.")
    p("<b>Qualification to the earlier recommendation:</b> the completed raw-resolution tests held output NSIDE at 4096. They measure the effect of raw painting resolution, but do not independently certify the final 4096 map. The current map2alm calls also use niter=0. A definitive output test must use the same beam-smoothed alms, synthesize 4096 and 8192 maps, apply the same continuous mask and matched harmonic noise splits, then compare unbinned spectra and MOPED through ell=7979. Mask sampling and final-transform error belong in that test.")
    p('Thus output4096 is a justified cost-saving candidate, conditional on that check. A value of lmax below 3*NSIDE-1 is a library convention, not an exact accuracy theorem. The <link href="https://juliaastro.org/Healpix/stable/alm/" color="#17618d">Healpix.jl harmonic-transform documentation</link> describes the approximation and optional iterations. No output-resolution validation or production change was performed for this explanatory note.', small=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=44, rightMargin=44, topMargin=40, bottomMargin=43,
                            title="HalfDome tSZ: smooth cache boundary, pixel sampling and scheduling", author="HalfDome analysis")
    doc.build(STORY, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "EXPLANATION.md").write_text("\n\n".join(MD) + "\n")
    print(PDF)


if __name__ == "__main__":
    main()
