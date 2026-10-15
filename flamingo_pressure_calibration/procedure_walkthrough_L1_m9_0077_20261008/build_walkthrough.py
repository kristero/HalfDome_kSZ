"""Build a short, source-linked explanation of the completed calibration.

This documentation builder reads compact local products only. It does not edit
the completed study, read snapshot particles, or submit work to a cluster.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from walkthrough_examples import DEFAULT_STUDY, inspect_and_refit


HERE = Path(__file__).resolve().parent
STUDY = DEFAULT_STUDY
OUTPUT = HERE / "FLAMINGO_snapshot_to_stack_fits_walkthrough.pdf"
W, H = landscape(A4)
LEFT, RIGHT = 36, W - 36
INK = colors.HexColor("#123448")
MUTED = colors.HexColor("#536570")
FONTS = Path("/usr/share/fonts/truetype/dejavu")
for name, filename in (
    ("Body", "DejaVuSans.ttf"),
    ("Bold", "DejaVuSans-Bold.ttf"),
    ("Mono", "DejaVuSansMono.ttf"),
):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Walkthrough:
    """Fixed, generous page layouts with explicit overflow checks."""

    def __init__(self):
        self.c = canvas.Canvas(str(OUTPUT), pagesize=(W, H))
        self.c.setTitle("FLAMINGO calibration: from snapshot to fitted stacks")
        self.c.setAuthor("HalfDome research")
        self.page = 0

    def start(self, title, source):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.c.setFillColor(INK)
        self.c.setFont("Bold", 22)
        assert pdfmetrics.stringWidth(title, "Bold", 22) < RIGHT - LEFT
        self.c.drawString(LEFT, H - 43, title)
        self.c.setFillColor(MUTED)
        self.c.setFont("Body", 9)
        self.c.drawString(LEFT, H - 64, source)
        self.c.setStrokeColor(colors.HexColor("#ccd8df"))
        self.c.line(LEFT, 35, RIGHT, 35)
        self.c.setFont("Body", 8.5)
        self.c.drawString(LEFT, 20, "HalfDome | L1_m9, snapshot 0077 | 8 October 2026")
        self.c.drawRightString(RIGHT, 20, f"{self.page} / 12")
        self.c.bookmarkPage(f"page{self.page}")
        self.c.addOutlineEntry(title, f"page{self.page}", level=0)
        return H - 90

    def text(self, value, y, x=LEFT, width=None, size=11, leading=16):
        width = RIGHT - x if width is None else width
        self.c.setFillColor(colors.HexColor("#182730"))
        self.c.setFont("Body", size)
        for paragraph in value.split("\n"):
            words = paragraph.split()
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if pdfmetrics.stringWidth(candidate, "Body", size) > width:
                    assert line, f"Unbreakable word: {word}"
                    assert y > 43, "Text crosses footer"
                    self.c.drawString(x, y, line)
                    y -= leading
                    line = word
                else:
                    line = candidate
            if line:
                assert y > 43, "Text crosses footer"
                self.c.drawString(x, y, line)
                y -= leading
            y -= 7
        return y

    def code(self, value, y, size=10):
        lines = textwrap.dedent(value).strip("\n").splitlines()
        leading = size * 1.42
        height = len(lines) * leading + 18
        assert y - height > 43, "Code crosses footer"
        self.c.setFillColor(colors.HexColor("#eef3f6"))
        self.c.roundRect(LEFT, y - height, RIGHT - LEFT, height, 5, fill=1, stroke=0)
        self.c.setFillColor(colors.HexColor("#162f43"))
        self.c.setFont("Mono", size)
        for j, line in enumerate(lines):
            assert pdfmetrics.stringWidth(line, "Mono", size) < RIGHT - LEFT - 24, line
            self.c.drawString(LEFT + 12, y - 17 - j * leading, line)
        return y - height - 13

    def equation(self, latex, y, size=20, max_height=43):
        fig = plt.figure(figsize=(10, .6))
        fig.text(.01, .3, latex, fontsize=size)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=240, bbox_inches="tight", transparent=True)
        plt.close(fig)
        buffer.seek(0)
        image = ImageReader(buffer)
        iw, ih = image.getSize()
        scale = min((RIGHT - LEFT - 10) / iw, max_height / ih)
        height = ih * scale
        assert y - height > 43
        self.c.drawImage(image, LEFT + 4, y - height, width=iw * scale, height=height, mask="auto")
        return y - height - 17

    def table(self, headers, rows, y, widths, size=10, row_height=26):
        assert sum(widths) <= RIGHT - LEFT + 1
        for i, row in enumerate([headers] + rows):
            top = y - i * row_height
            assert top - row_height > 43
            self.c.setFillColor(colors.HexColor("#dfeaf0" if i == 0 else "#f5f8fa"))
            self.c.rect(LEFT, top - row_height, sum(widths), row_height, fill=1, stroke=0)
            x = LEFT
            for value, width in zip(row, widths):
                font = "Bold" if i == 0 else "Body"
                assert pdfmetrics.stringWidth(str(value), font, size) < width - 12, value
                self.c.setFont(font, size)
                self.c.setFillColor(INK)
                self.c.drawString(x + 6, top - row_height + 9, str(value))
                x += width
        return y - (len(rows) + 1) * row_height - 14

    def image(self, path, top, max_height):
        image = ImageReader(str(path))
        iw, ih = image.getSize()
        scale = min((RIGHT - LEFT) / iw, max_height / ih)
        width, height = iw * scale, ih * scale
        assert top - height > 43
        self.c.drawImage(image, (W - width) / 2, top - height, width=width, height=height)
        return top - height - 13

    def flow(self, y):
        names = ["SOAP centrals", "Cells + exact sphere", "Shell sums", "Normalize + stack", "gNFW fits"]
        width, gap, height = 137, 20, 51
        for i, name in enumerate(names):
            x = LEFT + i * (width + gap)
            self.c.setFillColor(colors.HexColor("#e1edf3"))
            self.c.roundRect(x, y - height, width, height, 6, fill=1, stroke=0)
            self.c.setFillColor(INK)
            self.c.setFont("Bold", 9.4)
            self.c.drawCentredString(x + width / 2, y - 30, name)
            if i < len(names) - 1:
                self.c.setStrokeColor(INK)
                self.c.line(x + width + 3, y - 25, x + width + gap - 3, y - 25)
                self.c.line(x + width + gap - 7, y - 21, x + width + gap - 3, y - 25)
                self.c.line(x + width + gap - 7, y - 29, x + width + gap - 3, y - 25)
        return y - height - 24

    def finish(self):
        assert self.page == 12
        self.c.save()


def make_example_figure():
    """Plot stored, shell-averaged data; do not introduce a new physical fit."""
    sys.path.insert(0, str(STUDY))
    from models import shell
    case = read_json(STUDY / "products/bin_14p5_14p7_fits.json")
    with np.load(STUDY / "products/bin_14p5_14p7_profiles.npz") as saved:
        data = dict(saved)
    x = np.cbrt((data["edges"][:-1]**3 + data["edges"][1:]**3) / 2)
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 17, "legend.fontsize": 10,
                         "axes.titlesize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 6.2), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.35]}, layout="constrained")
    for column, (field, family, ylabel) in enumerate([
        ("pressure", "b12", r"$\langle P_e/P_{200}\rangle$"),
        ("electrons", "b16", r"$\langle\rho_\tau/(f_b\rho_c)\rangle$"),
    ]):
        mean = data[f"all_{field}_mean"]
        p16, _, p84 = data[f"all_{field}_percentiles"]
        supported = data["fit_base_mask"] & (mean > 0)
        top, bottom = axes[:, column]
        top.fill_between(x, p16, p84, color="0.86", label="Halo 16–84%")
        top.plot(x, mean, "k.", markersize=4.5, label="Mean stack")
        top.plot(x, data[f"{field}_published_reference"], "--", color="0.4", label="Published reference")
        for ri, radius in enumerate((2, 3, 4)):
            fit = case["selections"]["all"][field][ri]["fit"]
            prediction = shell(family, fit["parameters"])
            color = ["#2875a5", "#b36c18", "#2d8a66"][ri]
            top.plot(x[supported], prediction[supported], color=color, lw=1.8, label=f"Fit to {radius}R")
            bottom.plot(x[supported], 100*(prediction[supported]/mean[supported]-1), color=color, lw=1.8)
        top.set(xscale="log", yscale="log", ylabel=ylabel, title="Pressure" if field=="pressure" else "Electrons")
        bottom.set(xscale="log", ylabel="Residual [%]", xlabel=r"$r/R_{200c}$")
        bottom.axhline(0, color="0.5", lw=.8)
        for ax in (top, bottom):
            ax.set_xlim(.03, 4.1)
            ax.grid(alpha=.15)
        top.legend(ncol=2, loc="lower left")
    fig.savefig(HERE / "stack_fit_example.png", dpi=210)
    fig.savefig(HERE / "stack_fit_example.svg")
    plt.close(fig)


def build():
    # Six inexpensive refits also exercise both fields and all requested radii.
    checks = [inspect_and_refit(STUDY, 14.5, r, f) for f in ("pressure", "electrons") for r in (2, 3, 4)]
    (HERE / "example_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    make_example_figure()
    case = read_json(STUDY / "products/bin_14p5_14p7_fits.json")
    report = Walkthrough()

    y = report.start("From FLAMINGO particles to fitted stacks", "Current executed procedure: radii234_contamination_L1_m9_0077_20261008")
    y = report.text("The pipeline measures electron pressure and free-electron density around central halos, stacks their normalized radial profiles, and fits Battaglia-shaped models. This guide follows the executed code; short snippets explain its essential operations.", y)
    y = report.flow(y - 3)
    y = report.table(["Quantity", "Meaning"], [
        ("M200c, R200c", "Mass and radius enclosing mean density 200 times the critical density"),
        ("a, x = r/R200c", "Cosmic scale factor; dimensionless physical halo-centric radius"),
        ("fb, rho_c, P200", "Cosmic baryon fraction; critical density; halo pressure normalization"),
        ("Y_i, n_e,i", "Particle-integrated Compton signal; stored free-electron number density"),
    ], y, [145, 625], size=10)
    y = report.text("Current sample: L1_m9 / HYDRO_FIDUCIAL, snapshot 0077, z = 0; 4,375 centrals in 15 bins of width 0.2 dex, from log10(M200c/Msun) = 12.5 to 15.5. Each halo is extracted once to 4R200c; the 2R and 3R spheres are nested subsets.", y)
    y = report.text("What is fitted: three pressure coefficients and three electron-density coefficients per mass bin and radial range. This is an empirical calibration of mean profiles at one redshift. The nine-coefficient mass/redshift law has not been globally recalibrated; one snapshot cannot determine redshift evolution.", y)
    y = report.text("Reading order: selection and units (2), extraction and SWIFT (3), physical profiles (4–5), stacking and resolution (6), models and fitting (7–8), worked plots (9), scatter (10), projection (11), code map and usage (12).", y, size=10)
    report.text("Measured extraction cost: 8 workers, 71.27 minutes, 7.01 CPU-hours and peak 30.14 GiB on STRW. This guide uses compact local products; no new cluster jobs or snapshot particle reads.", y, size=10)

    y = report.start("1. Select centrals; read the units", "Source: prepare_study.py:43; dependencies/flamingo_io.py:263,597; dependencies/common.py:285")
    y = report.text("SOAP is the halo-property catalogue. It supplies M200c, R200c, the InputHalos/HaloCentre position, central flags and halo-finder IDs. A previously audited central-only candidate catalogue feeds prepare_study.py. The script does not independently rescan IsCentral; that input is a required dependency.", y)
    y = report.code("""
        logm = np.log10(m200c_msun)       # physical Msun, not Msun/h
        in_bin = np.flatnonzero((logm >= low) & (logm < high))
        selected = (rng.choice(in_bin, 300, replace=False)
                    if low < 14.5 else in_bin)
        # Same selected halo rows are used for the 2R, 3R and 4R comparisons.
    """, y)
    y = report.table(["Snapshot field", "Working unit", "Role"], [
        ("Coordinates; SmoothingLengths", "comoving Mpc", "Periodic positions; resolution gate"),
        ("Masses; Densities", "Msun; physical Msun/Mpc^3", "Particle volume V_i = m_i/rho_i"),
        ("ComptonYParameters", "physical Mpc^2", "Integrated electron pressure"),
        ("ElectronNumberDensities", "physical Mpc^-3", "Free electrons per physical volume"),
    ], y, [245, 225, 300], size=9.7)
    y = report.code("""
        factor = dataset.attrs[
            'Conversion factor to physical CGS (including cosmological corrections)']
        physical_values = raw.astype(np.float64) * float(np.asarray(factor)[0])
        mass_msun = physical_mass_g / MSUN_G
        # Coordinates instead retain the base CGS factor, giving comoving Mpc.
    """, y, size=9.5)
    report.text("Use the dataset attributes, including cosmological corrections, then convert CGS to the working unit. Do not append an assumed h factor. This snapshot has a = 1 and h-scale exponents zero for these fields; the code also carries a explicitly for positions. Particle IDs remain integers.", y, size=10.2, leading=15)

    y = report.start("2. Extract particles; keep membership explicit", "Source: extract_study.py:55,190; measure.py:35,43; swift_reader.py:102")
    y = report.text("Metadata identifies conservative SWIFT cells intersecting the selected 4R spheres and indexes cells to halos. Read each relevant cell once per chunk, then apply the exact sphere to each candidate halo. Cell selection limits I/O; the periodic sphere defines the measurement.", y)
    y = report.code("""
        run = Path('/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL')
        snapshot = run / 'snapshots/flamingo_0077/flamingo_0077.hdf5'
        soap = run / 'SOAP-HBT/halo_properties_0077.hdf5'
        delta = coordinates_cmpc - centre_cmpc
        delta -= box_cmpc * np.rint(delta / box_cmpc)  # nearest periodic image
        relative = delta * scale_factor / radius_physical_mpc
        x = np.linalg.norm(relative, axis=1)
        inside = x <= 4.0
        # Each particle can appear around multiple centrals if their spheres overlap.
    """, y)
    y = report.text("Membership_0077 gives the bound halo-finder ID of each gas particle, or a negative value for unbound gas. Compare it with InputHalos/HaloCatalogueIndex, not the SOAP row number. Satellites are identified through SOAP HostHaloIndex and mapped to their halo-finder IDs.", y)
    y = report.code("""
        category = np.where(membership < 0, 3, 2).astype(np.int64)
        category[np.isin(membership, satellite_ids)] = 1
        category[membership == central_id] = 0
        # 0 central-bound; 1 own satellite-bound; 2 other halo-bound; 3 unbound
        all_gas = [0, 1, 2, 3]; no_other = [0, 1, 3]
        no_satellites_or_other = [0, 3]; host_bound = [0, 1]
    """, y)
    y = report.text("SWIFT cross-check: a fresh swiftsimio mask loads the cuboid, followed by the same sphere and estimators. Forty-five halos (three per bin) match sorted particle-ID hashes and shell counts exactly; small weighted-sum differences arise from conversion constants and arithmetic.", y, size=10.5)
    report.text("Identical SWIFT and direct fits are expected for matched particles: this checks extraction completeness and conversion, not independent simulation physics. A matched cutout is spatially smaller, not a randomly downsampled snapshot.", y, size=10.5)

    y = report.start("3. Turn additive sums into physical profiles", "Source: measure.py:43; pilot_physics.py:106; extract_study.py:154")
    y = report.equation(r"$V_j=\frac{4\pi}{3}R_{200c}^3(x_{j+1}^3-x_j^3),\qquad V_i=\frac{m_i}{\rho_i}$", y)
    y = report.text("V_j is the full geometric volume of radial shell j. V_i is an SPH particle's effective volume, used to convert its electron density to an electron count. These two volumes have different purposes; the profile is divided by V_j even when some gas components are removed.", y)
    y = report.equation(r"$\overline{P}_{e,j}=\frac{m_e c^2}{\sigma_T V_j}\sum_{i\in j}Y_i,\qquad \overline{n}_{e,j}=\frac{1}{V_j}\sum_{i\in j}n_{e,i}V_i$", y, max_height=62)
    y = report.text("Y_i is the particle-integrated Compton signal, with area units. The Thomson cross section sigma_T and electron rest energy m_e c^2 turn its shell sum into electron pressure. n_e,i already contains the simulation's ionization/chemistry calculation. c is the speed of light; m_e is electron mass.", y)
    y = report.code("""
        particle_volume_mpc3 = mass_msun / density_msun_mpc3
        electron_number = ne_mpc3 * particle_volume_mpc3
        volume_mpc3 = (4*np.pi/3) * R_mpc**3 * np.diff(edges**3)
        Pe_pa = sum_Y_mpc2 * m_e*c**2 / (sigma_T * volume_mpc3 * MPC_M)
        ne_shell_mpc3 = sum_electron_number / volume_mpc3
        rho_gas_msun_mpc3 = sum_mass_msun / volume_mpc3
    """, y)
    report.text("The last pressure expression includes the Mpc-to-m conversion: Mpc^2/Mpc^3 becomes 1/(MPC_M metres). The code accumulates 11 sums per component and shell using numpy.bincount; file products can be added without retaining particles. Counts, gas mass, no-AGN Y, n_e k_B T, H/He masses and particle volumes provide diagnostics.", y, size=10.5)

    y = report.start("4. Normalize consistently; do not count H twice", "Source: pilot_physics.py:18,30,106; extract_study.py:176; metadata_plan.json field_schema")
    y = report.equation(r"$\rho_c=\frac{M_{200c}}{200(4\pi/3)R_{200c}^3},\quad f_b=\frac{\Omega_b}{\Omega_m+\Omega_{\nu,0}},\quad P_{200}=\frac{G M_{200c}(200\rho_c)f_b}{2R_{200c}}$", y, size=19)
    y = report.text("rho_c is the physical critical density inferred from each halo's matching mass and radius. G is Newton's constant. The snapshot convention puts massive neutrinos outside Omega_m, so the denominator adds Omega_nu,0. The resulting baryon fraction is 0.15882349. The pressure stack uses Pe/P200, with P200 evaluated in SI units.", y)
    y = report.equation(r"$\rho_\tau=\mu_{\rm XGP}n_e,\qquad \mu_{\rm XGP}=\frac{m_e+2m_p/(1+0.76)}{0.9},\qquad D_e=\frac{\rho_\tau}{f_b\rho_c}$", y)
    y = report.text("The primary B16 target D_e is an electron-derived effective density, using the fixed XGPaint mass-per-electron convention mu_XGP (including its 0.9 factor). m_p is proton mass. This mapping preserves compatibility with optical-depth painting. It is distinct from actual gas mass density, whose separate fit is an auxiliary comparison.", y)
    y = report.code("""
        pressure_profile = Pe_pa / P200_pa
        electron_profile = ne_mpc3 * MU_XGP_MSUN / (fb * rho_c_msun_mpc3)
        gas_profile = rho_gas_msun_mpc3 / (fb * rho_c_msun_mpc3)
        # Stored n_e already includes chemistry: do NOT multiply it by X_H.
        nH200 = 200*rho_c_msun_mpc3*fb*MSUN_KG*XH / proton_mass
        hydrogen_normalized_electrons = ne_mpc3 / nH200  # diagnostic only
    """, y)
    y = report.text("XH is the mass-weighted hydrogen fraction inside 4R; H and He also define a fully ionized reference. Those references do not replace the stored free electrons. Native Y and n_e are both zero for star-forming particles according to this snapshot's headers, so the current target excludes their unresolved ISM signal.", y, size=10.5)
    report.text("Recently AGN-heated gas remains in the primary target: its thermal signal is part of the snapshot. A separately labelled filtered Y diagnostic is retained; it is not the primary pressure fit. Native total Pressures and n_e k_B T are consistency checks, not substitutions for Y-based pressure.", y, size=10.5)

    y = report.start("5. Stack halos; enforce a common resolution mask", "Source: fit_study.py:60,80; measure.py:103; products/*_profiles.npz")
    y = report.code("""
        values = pressure_components[halo_rows].sum(axis=1)  # [halo, shell]
        mean = values.mean(axis=0)                         # equal halo weights
        scatter = values.std(axis=0, ddof=1)               # halo-to-halo scatter
        p16, median, p84 = np.percentile(values, [16, 50, 84], axis=0)
        # Normalize each halo BEFORE taking the arithmetic mean.
    """, y)
    y = report.text("This estimates the mean dimensionless profile of a central in that mass bin. It does not pool all gas particles, which would favor massive or gas-rich halos. Component profiles use the same halo normalization and shell volume, so they add exactly to the all-gas profile. The mean of a nonlinear family is generally not a member of that family.", y)
    y = report.code("""
        x_inner = edges[:-1]
        geometric = x_inner[None, :] >= np.maximum(0.04, 4*eps/R[:, None])
        resolved = (geometric & (particle_counts >= 20)
                    & (x_inner[None, :] >= 2*hsml_median_upper_x))
        strict = resolved.all(axis=0)      # require every selected halo
        supported = np.count_nonzero(strict & (edges[1:] <= 2)) >= 6
        mask_R = strict & (edges[1:] <= Rfit) & (mean > 0)
    """, y)
    y = report.text("eps is the physical gravitational softening (0.0057 Mpc here); R is each halo's physical R200c. Smoothing lengths are converted to h/R. Their conservative median uses the upper middle order statistic and upper histogram-bin edge. The gate requires a shell inner edge beyond both 4eps and twice that smoothing length.", y)
    y = report.text("The same base mask is used for 2R, 3R, 4R and membership controls. If fewer than six strict shells exist inside 2R, the code shows diagnostic fits using geometric cuts, at least 200 pooled particles and at least half of halos nonempty per shell. It does not silently drop unresolved halos.", y)
    report.text("Eleven of 15 bins pass the common strict-support rule; the four bins below log10 M = 13.3 are diagnostic only. A stable optimizer result in those bins is not evidence of resolved physics. Component controls inherit all-gas resolution gates and have no separate component-count certification.", y, size=10.5)

    y = report.start("6. What the two Battaglia models mean", "Source: models.py:8,11,20,28; pilot_physics.py:135")
    y = report.equation(r"$F_P(x)=\frac{P_e}{P_{200}}=0.5176\,P_0\left(\frac{x}{x_c}\right)^{-0.3}\left(1+\frac{x}{x_c}\right)^{-\beta_P}$", y)
    y = report.text("B12 pressure: P0 sets amplitude, xc sets the transition radius in units of R200c, and beta_P controls the outer decline. The inner slope is fixed at −0.3 and transition sharpness at 1. The asymptotic outer slope is −(0.3 + beta_P). The fixed 0.5176 converts the reference thermal-pressure convention to electron pressure; the measured Y-derived Pe is not multiplied by this factor again.", y)
    y = report.equation(r"$F_e(x)=D_e=\rho_0\,q^{-0.2}\left(1+q^\alpha\right)^{-(\beta_e-0.2)/\alpha},\qquad q=x/0.5$", y)
    y = report.text("B16 electrons: rho0 sets amplitude, alpha sets transition sharpness, and beta_e is the magnitude of the outer slope. The inner slope is fixed at −0.2 and transition radius at 0.5R200c. The same shape fitted to actual gas density is a separate, auxiliary model.", y)
    y = report.equation(r"$\overline{F}_j(\boldsymbol{p})=\frac{3}{x_{j+1}^3-x_j^3}\int_{x_j}^{x_{j+1}}F(x;\boldsymbol{p})\,x^2\,dx$", y, size=19)
    y = report.text("The data are shell averages. The model must therefore be volume-averaged over each shell, rather than evaluated only at the plotted radius. models.shell uses 32-point Gauss–Legendre quadrature. This matters most where a profile varies steeply across a shell.", y)
    y = report.table(["Family", "Parameter order", "Allowed positive ranges"], [
        ("Pressure", "P0, xc, beta_P", "[0.001, 10^4], [0.01, 10], [0.5, 20]"),
        ("Electrons", "rho0, alpha, beta_e", "[0.1, 10^7], [0.1, 10], [0.5, 20]"),
    ], y, [130, 225, 415], size=10)
    report.text("Published reference curves apply the original mass and redshift scaling to each selected halo's actual mass, then average. New per-bin fits float only the three shape coefficients; they do not inherit the original mass exponents as fitted constraints.", y, size=10.5)

    y = report.start("7. Fit the mean profile; understand the objective", "Source: fit_study.py:56,73,119; models.py:32; projection_model.py:33")
    y = report.equation(r"$s_j=\max\!\left(\frac{\sigma_{{\rm halo},j}}{\overline{F}_j},0.05\right),\qquad r_j(\boldsymbol{\theta})=\frac{\ln F_{{\rm model},j}(e^{\boldsymbol{\theta}})-\ln\overline{F}_j}{s_j}$", y, size=18)
    y = report.text("The optimizer minimizes the sum of r_j squared in log-parameter space, theta = ln(p). This keeps parameters positive and compares fractional profile differences across a large dynamic range. sigma_halo is the standard deviation among halos, not the error on the mean. The 5% floor prevents very quiet shells from dominating.", y)
    y = report.code("""
        from fit_study import fit_one                 # imports only; no main()
        from models import shell
        mask = fit_base_mask & (edges[1:] <= 4) & (mean > 0)
        fit = fit_one('b12', mean, scatter, mask)
        model = shell('b12', fit['parameters'])        # shell-averaged prediction
        rms = 100*np.sqrt(np.mean((model[mask]/mean[mask] - 1)**2))
    """, y)
    y = report.text("Zero or invalid mean shells are excluded from logarithms. At least six shells are required. scipy.optimize.least_squares uses a maximum of 500 evaluations and 10^-9 termination tolerances. The default starting values are the first three published coefficients.", y)
    y = report.text("A fit is marked stable only if optimization succeeds, its minimum log-distance from bounds exceeds 10^-5, and its residual-Jacobian condition number is below 10^6. These detect numerical fragility; they do not impose a goodness-of-fit threshold. A stable fit can still have a large RMS.", y)
    y = report.table(["14.5–14.7 bin; 895 centrals", "4R coefficients", "In-range RMS"], [
        ("Pressure: P0, xc, beta_P", "17.9515, 0.642988, 5.11131", "17.50%"),
        ("Electrons: rho0, alpha, beta_e", ", ".join(f"{p:.5g}" for p in case['selections']['all']['electrons'][2]['fit']['parameters']), f"{case['selections']['all']['electrons'][2]['fit']['rms_percent']:.2f}%"),
    ], y, [315, 290, 165], size=10)
    report.text("The reported radial RMS is unweighted percentage RMS; it is different from the minimized scatter-weighted log objective. For a fair radius comparison, also evaluate every fitted model on the same resolved inner-2R and full-4R targets. A 2R fit evaluated at 4R is an extrapolation.", y, size=10.5)

    y = report.start("8. A complete stack-to-fit example", "Existing data: products/bin_14p5_14p7_profiles.npz and bin_14p5_14p7_fits.json")
    y = report.image(HERE / "stack_fit_example.png", y, 350)
    y = report.text("Points are the arithmetic mean of 895 individually normalized profiles; the gray band spans the halo 16th–84th percentiles, not uncertainty in the mean. All curves are shell averages. Colored curves use the 2R, 3R or 4R fit but are displayed on common available support to expose extrapolation. Residual = 100 × (model/stack − 1).", y, size=10.5)
    report.text("The larger fit range constrains the outskirts but can compromise the inner fit because one gNFW shape must describe both regions. Across ten supported bins with holdouts, median full-4R pressure RMS falls from 20.36% (2R fit) to 18.13% (3R) to 15.61% (4R). This does not establish resolution convergence or a universal optimum for inner-halo painting.", y, size=10.5)

    y = report.start("9. Three different kinds of scatter", "Source: fit_study.py:43,98,140–167,251; products/*_parameters.npz")
    y = report.table(["Saved quantity", "What it describes", "What it does not establish"], [
        ("Halo std / 16–84% profile band", "Variation among individual halo profiles", "Precision of the mean"),
        ("128 bootstrap mean-fit draws", "Sensitivity to the sampled halo population", "Full inference covariance / cosmic variance"),
        ("Individual fitted coefficients", "Conditional distribution for resolved, stable halos", "Deconvolved intrinsic parameter scatter"),
    ], y, [240, 285, 245], size=9)
    y = report.code("""
        draws = rng.integers(0, N, size=(128, N))  # resample whole halos with replacement
        for chosen in draws:
            sample = values[chosen]             # keep each entire radial profile
            trial = fit_one(family, sample.mean(0), sample.std(0, ddof=1), mask)
            # Save coefficients only if the optimizer passes the stability checks.
        # Reuse the same chosen indices across fields, radii and membership controls.
    """, y)
    y = report.text("Whole-halo resampling preserves radial correlations inside a halo. The fitting objective itself still uses diagonal shell weights. Bootstrap percentiles are conditional on successful refits; rejected fits are not represented. With only 128 draws, percentile tails are approximate. Shared gas, spatial correlations, one-box cosmic variance, resolution and model errors are not captured.", y)
    y = report.text("For individual halos, the code uses each halo's resolved shells and constant 5% log weights. Coefficient percentiles and log covariance are computed only for stable fits. This selection can suppress or distort physical scatter. The joint six-parameter distribution preserves pressure–electron correlations, but small-N bins can have deficient covariance rank.", y)
    y = report.code("""
        good = saved['accepted'][:, 2].all(axis=1)       # radius index 2 means 4R
        joint = saved['parameters'][good, 2].reshape(-1, 6)
        draw = joint[rng.integers(len(joint))]          # sample one whole joint row
        # Order: P0, xc, beta_P, rho0, alpha, beta_e. Do not draw six independent errors.
    """, y, size=9.5)
    report.text("The worked bin has 894 accepted joint 4R halo fits. This empirical draw is a demonstration, not a validated painting model. A separate 80/20 halo split tests mean-profile prediction when N ≥ 20; it is not a spatially independent simulation test. The six-halo highest bin has no holdout.", y, size=10.5)

    y = report.start("10. Project the same sphere; inspect subtraction", "Source: measure.py:75–100; projection_model.py:10,23,39; orientations.py")
    y = report.equation(r"$\frac{y(b)}{y_{200}}=\int_{-\sqrt{R_{\rm cut}^2-b^2}}^{+\sqrt{R_{\rm cut}^2-b^2}} F_P\!\left(\sqrt{b^2+\ell^2}\right)d\ell,\quad y_{200}=\frac{\sigma_T P_{200}R_{200c}}{m_ec^2}$", y, size=18)
    y = report.text("b, ell and Rcut are measured in R200c units. The integral follows a finite spherical chord; Rcut = 2, 3 or 4. Pressure maps are y/y200; electron maps use the corresponding integral of D_e. The model map uses 128-point line-of-sight quadrature and 3 × 3 subpixel averaging on a shared 64 × 64 grid spanning ±4R200c.", y, size=10.5)
    y = report.image(STUDY / "figures/bin_14p5_14p7_all_pressure_R4_maps.png", y, 250)
    y = report.text("Particles are deposited by their centers, without SPH kernel smoothing. Each normalized halo is viewed along ten fixed directions with image-plane rolls, projected, then averaged and stacked. This reduces directional features but cannot change the 3D radial profile or create independent halos; the ten views are correlated.", y, size=10.3, leading=15)
    report.text("The right panel is stack − model (red: missing model signal); radial residuals on page 9 use the opposite sign. Map L2 is 100||model−stack||/||stack||, while integrated bias uses model/stack−1. Central pixels can dominate whole-map metrics, so the analysis also reports core-excluded and outer-ring residuals. Projected refits are additional fits to 2D annular data, separate from projecting a 3D fit.", y, size=10.3, leading=15)

    y = report.start("11. Follow the files; reproduce a fit locally", "The existing study remains unchanged. This walkthrough and its examples use a separate output directory.")
    y = report.table(["File / entry point", "Responsibility"], [
        ("prepare_study.py; metadata_plan.json", "Choose halos, satellites, cells, units, checks and resource budget"),
        ("extract_study.py; measure.py", "Read gas and membership; exact spheres; additive sums and maps"),
        ("pilot_physics.py; measurements.npz", "Physical profiles, normalizations and fixed reference conventions"),
        ("fit_study.py; models.py", "Masks, stacks, log fits, bootstrap, individual scatter and holdouts"),
        ("projection_model.py; projection_stacks.npz", "Matched finite-sphere model maps; measured map accumulators"),
        ("products/*_profiles.npz; *_fits.json", "Mean/scatter/reference/masks; coefficients and diagnostics"),
        ("products/*_parameters.npz", "[halo, radius, field, coefficient]; accepted/status masks and draws"),
        ("plot_study.py; profile_explorer.html", "Static comparison figures; interactive radial/spherical visualization"),
    ], y, [310, 460], size=9.4, row_height=23)
    y = report.code("""
        # In the new procedure_walkthrough_L1_m9_0077_20261008 directory:
        /home/cbllover/miniconda3/bin/python -B walkthrough_examples.py --bin-low 14.5 --radius 4
        # Add --field electrons, or change --radius to 2 or 3; no files are overwritten.
    """, y, size=9.5)
    y = report.text("Array orders: sums [4375, 4 components, 11 statistics, 36 shells]; normalized fields [4375, 4, 36]; individual parameters [N, 3 radii, 2 fields, 3 coefficients]. Radii: 2/3/4; fields: pressure/electrons; status −1/0/1: absent/unstable/stable.", y, size=10.3, leading=15)
    y = report.text("Edit a fresh study using the file map above. Some selection settings, grid sizes and array dimensions are hardcoded: study_config.json alone does not retarget the pipeline. Never reuse cached file products after changing physics or selection. The executable examples read existing outputs and leave them unchanged.", y, size=10.3, leading=15)
    report.text("Physical limits: one redshift and resolution; low-mass diagnostic fits; overlapping all-gas spheres can double count in additive painting. Removing bound satellites does not remove merger heating. Disturbed and companion flags are snapshot proxies, not merger histories. The local spherical viewer reconstructs a radial profile; it does not retain measured asymmetry.", y, size=10.3, leading=15)
    report.finish()

    sources = [
        "prepare_study.py", "extract_study.py", "measure.py", "pilot_physics.py",
        "fit_study.py", "models.py", "projection_model.py", "orientations.py",
        "swift_reader.py", "check_fresh_mask.py", "metadata_plan.json", "complete.json",
        "swift_checks.json", "scientific_checks.json", "products/study_summary.json",
        "products/bin_14p5_14p7_profiles.npz", "products/bin_14p5_14p7_fits.json",
        "products/bin_14p5_14p7_parameters.npz", "selected_halos.npz", "measurements.npz",
        "figures/bin_14p5_14p7_all_pressure_R4_maps.png",
        "dependencies/common.py", "dependencies/flamingo_io.py",
    ]
    evidence = {
        "report": str(OUTPUT),
        "pages": len(PdfReader(OUTPUT).pages),
        "report_sha256": sha256(OUTPUT),
        "six_local_refits_passed": all(c["rebuild_stack_and_refit_checks_passed"] for c in checks),
        "maximum_profile_rebuild_relative_difference": max(c["rebuilt_profile_max_relative_difference"] for c in checks),
        "snapshot_particle_reads": 0,
        "cluster_jobs_submitted": 0,
        "completed_study_files_modified": [],
        "source_sha256": {name: sha256(STUDY / name) for name in sources},
        "new_source_sha256": {name: sha256(HERE / name) for name in ("build_walkthrough.py", "walkthrough_examples.py")},
        "visual_review_complete": False,
    }
    (HERE / "walkthrough_verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"pdf": str(OUTPUT), "pages": evidence["pages"], "sha256": evidence["report_sha256"]}))


if __name__ == "__main__":
    build()
