"""Build two readable PDF reports from verified cluster products.

Run after fetching the delivery bundle. Scientific numbers are read from
the cluster JSON outputs; figure pages use the cluster-rendered PNGs.
"""
from pathlib import Path
import hashlib
import html
import json
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfdoc
from reportlab.lib import utils as reportlab_utils

import physics as P

# ReportLab uses MD5 only for document/image identifiers here. Python 3.8's
# OpenSSL wrapper lacks the later non-security keyword; keep hashlib global
# behaviour unchanged and adapt only these two ReportLab module aliases.
try:
    hashlib.md5(usedforsecurity=False)
except TypeError:
    def document_fingerprint(data=b'', **kwargs):
        kwargs.pop('usedforsecurity', None)
        return hashlib.md5(data, **kwargs)
    pdfdoc.md5 = document_fingerprint
    reportlab_utils.md5 = document_fingerprint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs'
FIG = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
FONT = Path('/usr/share/fonts/truetype/dejavu')
if not FONT.exists():
    FONT = Path(matplotlib.get_data_path())/'fonts/ttf'
for name, filename in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),
                       ('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT/filename)))

W, H = landscape(A4)
M = 36
STYLE = ParagraphStyle('body', fontName='Body', fontSize=11, leading=15,
                       textColor=colors.HexColor('#263445'), spaceAfter=8)
SMALL = ParagraphStyle('small', parent=STYLE, fontSize=9, leading=13)


def load(name):
    return json.loads((ROOT/name).read_text())


class Report:
    def __init__(self, filename, subtitle):
        self.path = OUT/filename
        self.pending = self.path.with_suffix('.pdf.pending')
        self.c = canvas.Canvas(str(self.pending), pagesize=(W,H))
        self.c.setTitle(subtitle)
        self.c.setAuthor('HalfDome / FLAMINGO analysis')
        self.page = 0
        self.y = 0

    def start(self, title):
        self.section_title = title.replace(' (continued)', '')
        if self.page:
            self.c.showPage()
        self.page += 1
        self.c.setFillColor(colors.HexColor('#153a58'))
        self.c.setFont('Bold', 21)
        self.c.drawString(M,H-44,title)
        self.c.setStrokeColor(colors.HexColor('#b9cddc'))
        self.c.line(M,H-56,W-M,H-56)
        self.c.setFont('Body',8)
        self.c.setFillColor(colors.HexColor('#657487'))
        self.c.drawString(M,18,'FLAMINGO L1_m9 / HalfDome | z = 0 | 14 October 2026')
        self.c.drawRightString(W-M,18,str(self.page))
        self.y = H-76

    def text(self, content, small=False):
        para = Paragraph(content, SMALL if small else STYLE)
        _,height = para.wrap(W-2*M, H)
        if self.y-height < 36:
            self.start(self.section_title+' (continued)')
        para.drawOn(self.c,M,self.y-height)
        self.y -= height+6

    def table(self, rows, widths=None, size=10):
        style = ParagraphStyle('cell', parent=STYLE, fontSize=size, leading=size+3)
        cells=[[Paragraph(html.escape(str(v)),style) for v in row] for row in rows]
        table=Table(cells,colWidths=widths or [(W-2*M)/len(rows[0])]*len(rows[0]),hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef5')),
            ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#7894a9')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),
            ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        _,height=table.wrap(W-2*M,H)
        if self.y-height<36:
            self.start(self.section_title+' (continued)')
        table.drawOn(self.c,M,self.y-height)
        self.y -= height+10

    def equation(self, expression, name):
        path=TMP/(name+'.png')
        fig=plt.figure(figsize=(10,.65),dpi=180)
        fig.text(.5,.5,'$'+expression+'$',ha='center',va='center',fontsize=20)
        fig.savefig(path,bbox_inches='tight',pad_inches=.12,transparent=True)
        plt.close(fig)
        iw,ih=Image.open(path).size
        width=min(W-2*M,iw/2)
        height=width*ih/iw
        self.c.drawImage(str(path),(W-width)/2,self.y-height,width,height,mask='auto')
        self.y-=height+6

    def code(self, value):
        lines=textwrap.dedent(value).strip('\n').splitlines()
        width=W-2*M
        height=14*len(lines)+20
        if self.y-height<36:
            self.start(self.section_title+' (continued)')
        self.c.setFillColor(colors.HexColor('#f0f4f7'))
        self.c.roundRect(M,self.y-height,width,height,5,fill=1,stroke=0)
        self.c.setFont('Mono',9)
        self.c.setFillColor(colors.HexColor('#172b3d'))
        for i,line in enumerate(lines):
            if len(line)>116:
                raise ValueError('Code line too long')
            self.c.drawString(M+10,self.y-17-14*i,line)
        self.y-=height+12

    def figure(self,name,title,caption):
        self.start(title)
        path=FIG/(name+'.png')
        iw,ih=Image.open(path).size
        maxheight=H-180
        scale=min((W-2*M)/iw,maxheight/ih)
        width,height=iw*scale,ih*scale
        top=self.y
        self.c.drawImage(str(path),(W-width)/2,top-height,width,height)
        self.y=top-height-8
        self.text(caption,small=True)

    def finish(self):
        self.c.save()
        self.pending.replace(self.path)
        return dict(file=str(self.path.relative_to(ROOT)),pages=self.page,
                    bytes=self.path.stat().st_size,
                    sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())


def summary_table(findings):
    units={'pressure':'Pa','electrons':'cm^-3','y':'dimensionless','tau':'dimensionless',
           'dm':'pc cm^-3','deltaT':'microkelvin'}
    labels={'pressure':'Electron pressure','electrons':'Electron density','y':'Compton-y',
            'tau':'Optical depth','dm':'DM','deltaT':'kSZ Delta T (RMS)'}
    rows=[['Observable','FL gas mean / RMS*','FL painted / gas','HD / FL painted']]
    for name in units:
        r=findings[name]
        if name=='deltaT':
            true=r['measured_fl_rms'];fl=r['fl_painted_rms'];hd=r['hd_painted_rms']
        else:
            true=r['measured_fl_mean'];fl=r['fl_painted_mean'];hd=r['hd_painted_mean']
        rows.append([labels[name],f'{true:.4g} {units[name]}',f'{fl/true:.3f}',f'{hd/fl:.3f}'])
    return rows


def resources(report):
    job_count=load('CLUSTER_VERIFIED.json')['job_instances']
    if (ROOT/'run_report_layout.job.json').exists():
        job_count=max(job_count,16)
    elif (ROOT/'run_report_final.job.json').exists():
        job_count=max(job_count,15)
    elif (ROOT/'run_report_retry.job.json').exists():
        job_count=max(job_count,14)
    elif (ROOT/'run_report.job.json').exists() and job_count==12:
        job_count+=1
    report.text(f'{job_count} PBS job instances were submitted, including a cached-profile weighting correction and retries of cache synchronization and plotting. '
                'Every idark request used the mini queue, 26 CPUs and 32 GB; the longest wall-time cap was 8 hours. '
                'The five painting workers were the only five-node stage. STRW used four bounded CPU processes on maas, '
                'with a 24 GB process-tree guard; it did not use five nodes.')
    data=load('strw_execution_metadata.json')['extraction_complete.json']
    used=load('resource_accounting_summary.json')
    report.table([
        ['Stage','Measured usage / limit'],
        ['STRW full gas pass',f"{data['native_pass_max_worker_wall_seconds']/60:.1f} min max worker wall; "
                              f"{data['worker_cpu_seconds']/3600:.2f} CPU-hours total"],
        ['Raw fields','5,271,821,058 gas particles; 295.2 GB decoded; 124.5 GB stored datasets'],
        ['idark timed applications',f"{used['timed_idark_cpu_hours']:.2f} CPU-hours across 15 unique job logs; final PBS usage unavailable"],
        ['Painting peak','PBS reported approximately 29.3 GiB for the largest worker; 32 GB requested'],
        ['Quota recovery','3.9 GB home quota exceeded during merge; verified data moved to /data1; zero particle rereads'],
        ['Cleanup','1.424 GB verified redundant partials removed after conservation checks']
    ],[180,W-2*M-180])
    report.text('42.3 MB compact exports were transferred to idark. The sum of per-worker STRW RSS maxima is about 5.17 GiB; '
                'it is not a measured simultaneous tree maximum. '
                'Stored dataset sizes are not a measurement of physical network traffic. '
                'PBS allocation CPU-hours and actual consumed CPU-hours are reported separately in the machine-readable accounting.',small=True)


def make_summary(findings,mapping):
    report=Report('FLAMINGO_HalfDome_z0_summary.pdf','FLAMINGO to HalfDome: concise comparison')
    report.start('What the z=0 comparison establishes')
    report.text('<b>The transfer is numerically feasible and the direct FLAMINGO gas comparison is complete.</b> '
                '57 painted cases plus two derived common-support controls cover fit/cut radii, cosmology normalization, '
                'halo scatter, massive-halo removal and alternative mass calibrations. Eight 250 Mpc cubes have identical '
                'physical depth and resolution. No raw FLAMINGO snapshot was downloaded.')
    report.table(summary_table(findings),[170,190,180,W-2*M-540])
    report.text('*Mean is used for positive fields; kSZ uses RMS because the signed mean cancels. '
                'These ratios compare a selected-mass halo model to the full gas field; they do not isolate profile-fit error.',small=True)
    report.text(f"<b>Mass calibration:</b> held-out recovery of the released assigned masses has "
                f"{mapping['summary']['heldout_rms_percent']:.3f}% RMS; near z=0 it is "
                f"{mapping['summary']['near_z0_rms_percent']:.3f}%. This does not validate true spherical-overdensity masses. "
                'Alternative z=0 polynomial reconstructions differ by up to 25.1% at the high-mass edge, so their map changes '
                'are sensitivity controls, not a confidence bound.')
    report.text('<b>Recommended use:</b> retain Pe fit 2R / cut 4R and ne fit 3R / cut 3R as observable-specific reference templates. '
                'Inspect the fit/cut matrices and inner residuals before treating this as an optimum. For a joint gas fluid, '
                'also supply electrons in the pressure-bearing 3-4R shell; the common-support control tests ne cut 4R.')
    for field,title in [('y','Compton-y: equal 250 Mpc slabs'),('dm','DM: equal 250 Mpc slabs'),
                        ('deltaT','kSZ: equal 250 Mpc slabs')]:
        report.figure('maps_'+field,title,
            'Top: measured FLAMINGO gas, its halo-painted model, and HalfDome halo painting. Bottom: same-simulation '
            'absolute differences. FL gas uses actual gas velocities; painting uses halo bulk velocities. '
            'The independent FLAMINGO and HalfDome pixels are not subtracted from one another.')
    report.start('How to interpret this result')
    report.text('<b>HalfDome is an N-body simulation:</b> there is no independent HalfDome gas truth to refit. '
                'The shape parameters are transferred from FLAMINGO; population and velocity differences change the maps. '
                'Cosmology corrections change reference density, baryon fraction and halo radius, but cannot correct feedback '
                'history, halo abundance or initial-condition differences.')
    report.text('<b>Statistics:</b> all pixels/cells of the defined grids are included, with zero-probability atoms and '
                'histogram tails recorded. Spatial bands describe variation among eight regions. They are not independent-mock '
                'covariance. z=0 slab DM is not a source-redshift-one FRB prediction. The power spectra are physical-plane P2D(k), '
                'not lightcone angular C-ell. Coarse voxel masks cannot validate halo cores.')
    resources(report)
    return report.finish()


def method_pages(report,findings,mapping,fits):
    report.start('Scope, inputs and measured conclusions')
    report.text('The purpose is to test transferring z=0 FLAMINGO Battaglia-family pressure and electron-density calibrations '
                'to the HalfDome halo catalogue. Independent initial phases prevent halo/pixel matching. '
                'We therefore compare equal-volume PDFs and projected power, and use FLAMINGO gas versus FLAMINGO painting '
                'for same-realization residuals.')
    report.table(summary_table(findings),[170,190,180,W-2*M-540])
    report.text('Pressure/electron-density PDFs are uniform-grid cell statistics at 0.977 Mpc, with 1.953 Mpc resolution controls. '
                'Projected maps use 512 squared pixels over 250 Mpc. Whole-box figures use 1024 squared pixels, but the FLAMINGO '
                'and HalfDome depths are 1000 and 5535.87 Mpc respectively; they are qualitative overviews.')
    report.text('The shared painting interval is physical M200c = 10^13 to 10^15.5 solar masses. FLAMINGO source profiles '
                'cover 12.5-15.5 in 0.2 dex bins: 300 halos per lower bin and all available high-mass centrals. '
                'The two upper bins contain 21 and 6 halos. The HalfDome selection rejects RFoF groups below 320 particles '
                'and any mass mapping outside its calibrated raw-mass support.')

    report.start('Main findings and sensitivity controls')
    report.text('<b>The selected halo model recovers most thermal pressure, but a minority of all electrons.</b> '
                f"In FLAMINGO it captures {100*findings['y']['fl_painted_mean_over_full_gas']:.1f}% of mean y and "
                f"{100*findings['dm']['fl_painted_mean_over_full_gas']:.1f}% of mean DM. "
                'The remainder combines diffuse gas, excluded halo masses, finite support and profile/overlap errors; '
                'it cannot be assigned uniquely to diffuse gas by this experiment. '
                'Pressure-only halo painting is therefore more complete than an all-electron / FRB-DM description.')
    controls=findings['controls']
    rows=[['HalfDome control / reference','Mean y','Mean DM','y power','kSZ power']]
    for key,label in [('normalization_control','Source-cosmology normalization'),
                       ('high_mass_cut_15p1','Remove M >= 10^15.1 Msun'),
                       ('joint_support_hybrid','Extend fixed ne fit3 to 4R'),
                       ('paired_scatter_42','Correlated parameter scatter: seed 42'),
                       ('mass_mapping_alternative_2','Alternative mass calibration 2')]:
        c=controls[key]
        rows.append([label,f"{c['y']['mean_ratio']:.3f}",f"{c['dm']['mean_ratio']:.3f}",
                     f"{c['y']['power_ratio_k005_1']:.3f}",f"{c['deltaT']['power_ratio_k005_1']:.3f}"])
    report.table(rows,[315,105,105,105,W-2*M-630])
    report.text('Power ratios use the mean physical-plane power over 0.05 <= k <= 1 Mpc^-1; '
                'they are not universal multipole-independent responses. The matrices use a separate 0.05-0.5 band.',small=True)
    report.text('Removing poorly constrained massive halos changes y power far more than mean DM. They should be retained '
                'with a regularized mass trend or a wider, uncertainty-labelled bin, rather than silently dropped. '
                'The alternative mass reconstruction gives a large tSZ response even though released-mass interpolation '
                'is accurate on held-out data: the high-mass z=0 extrapolation remains consequential.')
    report.text('Increasing electron support from 3R to 4R at fixed fit raises the mean electron column substantially, '
                'yet cannot restore the full gas field. Cosmology normalization is only a percent-level map effect here. '
                'Scatter changes higher moments and power; two seeds and eight templates per bin do not establish a converged '
                'physical scatter model. The original Pe2/4, ne3/3 choice is a useful reference, not a uniquely validated optimum.')
    report.text('Against the coarse measured same-cut halo union, Pe fit2/cut4 gives 30.2% map L2 residual; '
                'ne fit3/cut3 gives 23.0% DM residual and 52.1% kSZ residual. Fit-radius changes are small in this global metric. '
                'The target changes with cut, and voxel resolution, overlap and velocity errors prevent using it as a unique '
                'core-fit optimization criterion.',small=True)

    report.start('Extraction: native particle fields to observables')
    report.text('One streamed pass reads six fields from all 64 L1000N1800/HYDRO_FIDUCIAL snapshot 0077 files: '
                'Coordinates, Velocities, Masses, Densities, ElectronNumberDensities and ComptonYParameters. '
                'HDF5 physical-CGS metadata supplies unit conversion. Gas volume is particle mass divided by density. '
                'Native electron density already includes composition and ionization, so applying a hydrogen fraction again would be wrong.')
    report.equation(r'y=\frac{\sum_i Y_i}{A},\quad \tau=\sigma_T\int n_e\,dl,\quad {\rm DM}=\int n_e\,dl','observables')
    report.text('Y_i is the native integrated particle Compton-y area, in physical Mpc squared; A is pixel area in Mpc squared. '
                'n_e is electron density in inverse cubic centimetres; l is physical line-of-sight distance. '
                'Thomson cross section sigma_T converts electron column to optical depth; DM is reported in pc cm^-3. '
                'The factor 10^6 converts Mpc to pc for DM, rather than a hydrogen normalization.')
    report.equation(r'\Delta T=-T_{\rm CMB}\frac{\sigma_T}{c}\int n_e v_\parallel\,dl,\quad P_e=n_e k_B T_e','ksz')
    report.text('v_parallel is physical peculiar gas velocity, c is the speed of light and T_CMB = 2.7255 K. '
                'Positive receding velocity gives negative temperature shift. P_e is electron thermal pressure, '
                'k_B is Boltzmann constant and T_e is electron temperature. The measured pressure is recovered from native Y_i, '
                'rather than multiplying Y by the electron-to-total pressure factor again.')
    report.text('Deposition is nearest-grid-point at particle centres, with conservation tests. No smoothing-length field was read; '
                'this is a coarse statistical map, not an SPH-resolved gas rendering. Native ComptonYParameters excludes '
                'star-forming gas pressure by the release convention; recently AGN-heated gas is retained.')

    report.start('Cosmology transfer at fixed physical halo mass')
    report.equation(r'\rho_c=\frac{3H_0^2}{8\pi G},\quad R_{200c}=\left(\frac{3M_{200c}}{800\pi\rho_c}\right)^{1/3},\quad f_b=\frac{\Omega_b}{\Omega_m}','cosmology')
    report.text('rho_c is z=0 critical density, H_0 the present Hubble rate, G Newton constant, and M200c the mass inside '
                '200 times critical density. f_b is the cosmic baryon fraction. FLAMINGO total matter includes its separately '
                'stored neutrino density: Omega_m = 0.3060000794. HalfDome has Omega_m = 0.3089. '
                'Their h values are 0.681 and 0.6774; both use Omega_b = 0.0486.')
    report.equation(r'P_{200}=\frac{GM_{200c}}{2R_{200c}}\,200\rho_c f_b,\quad n_{e,0}=\frac{\rho_c f_b}{\mu_{\rm XGP}}','normalizations')
    report.text('P200 is the pressure normalization. mu_XGP is the electron-mass conversion used by the existing XGPaint '
                'density convention, including its 0.9 ionization factor; the fitted density amplitude absorbs this convention. '
                'The native FLAMINGO electron field is not altered. Dimensionless radius x = r/R200c and calibrated shape '
                'are held fixed; R scales as H0^-2/3, P200 as f_b H0^8/3, and electron normalization as f_b H0^2.')
    report.table([['Target/source factor','Value']]+[[k,f'{v:.8f}'] for k,v in P.cosmology_ratios().items()],
                 [350,W-2*M-350],size=10)
    report.text('These factors correct units and self-similar normalization only. They cannot remove different sigma8, '
                'growth, massive neutrinos, baryonic halo-mass shifts, feedback histories or velocity fields.',small=True)

    report.start('Profile fits: objective, constraints and scatter')
    report.equation(r'\frac{P_e}{P_{200}}=0.5176P_0 q^{-0.3}(1+q)^{-\beta},\quad q=\frac{x}{x_c}','b12')
    report.equation(r'\frac{\mu_{\rm XGP}n_e}{\rho_c f_b}=\rho_0 u^{-0.2}(1+u^\alpha)^{-(\beta-0.2)/\alpha},\quad u=\frac{x}{0.5}','b16')
    report.text('The pressure parameters are amplitude P0, transition radius xc, and outer-slope parameter beta. '
                'The electron parameters are amplitude rho0, transition sharpness alpha, and outer slope beta. '
                'The B12 factor 0.5176 is the inherited total-pressure-to-electron convention in the model definition. '
                'It is applied only there. A single z=0 snapshot cannot constrain redshift exponents.')
    report.equation(r'\chi^2_{\rm diag}=\sum_{j\in\mathcal{F}}\left[\frac{\ln f_j(\theta)-\ln\bar p_j}{\max(s_j/\bar p_j,0.05)}\right]^2','objective')
    report.text('j labels radial shells; F is the resolved set inside the selected fit radius; p-bar_j is the halo mean '
                'normalized profile, s_j its halo standard deviation and f_j the shell-averaged model. theta contains the three '
                'free parameters. Fractional halo SD weights retain the established empirical-calibration objective; '
                'SEM weights were retained only as a sensitivity control. This diagonal objective ignores radial covariance.')
    report.text('Bounds: Pe [P0, xc, beta] = [0.001,0.01,0.5] to [10000,10,20]; '
                'ne [rho0, alpha, beta] = [0.1,0.1,0.5] to [10^7,10,20]. '
                '128 paired halo-bootstrap resamples refit the empirical mean and recompute SD. '
                'Their parameter spread is uncertainty of that mean calibration; it is not intrinsic halo scatter or full inference uncertainty. '
                'Individual fits use each halo\'s resolved bins and a 5% fractional objective. Accepted paired Pe/ne fits generate '
                'correlated scatter templates, with selection, parameter-degeneracy and finite-resolution limitations.',small=True)

    report.start('HalfDome mass assignment and binning')
    report.text('Raw RFoF particle counts define RFoF masses, not spherical-overdensity M200c. The released processed lightcone '
                'contains assigned M200c values. A smooth two-dimensional Chebyshev reconstruction in log RFoF mass and redshift '
                'was trained on 43,263 balanced objects and checked against one million held-out released objects. '
                'The physical catalogue divides masses and positions in h-scaled units by h = 0.6774.')
    s=mapping['summary']
    report.table([['Validation quantity','Result'],['Held-out RMS',f"{s['heldout_rms_percent']:.6f}%"],
                  ['Held-out 95th percentile absolute error',f"{s['heldout_p95_absolute_percent']:.6f}%"],
                  ['Near-z=0 held-out sample',f"{s['near_z0_rows']:,} objects; RMS {s['near_z0_rms_percent']:.6f}%"],
                  ['Training redshift support',str(s['redshift_training_range'])],
                  ['Alternative z=0 maximum difference','25.06% at selected high-mass edge']],
                 [350,W-2*M-350])
    report.text('This validates recovery of released assigned masses. It does not establish true direct-SO masses, '
                'nor a baryonic-to-DMO halo correspondence. z=0 is slightly outside the redshift support. '
                'Alternative basis orders and low-redshift training windows probe numerical instability, not a physical error bar.')
    report.text('The FLAMINGO fits use actual geometric-mean mass pivots in each 0.2 dex bin. Positive parameters are interpolated '
                'in log mass and log parameter. This reduces a nominal-bin-centre bias but does not make a stacked fit equivalent '
                'to a conditional individual-halo model. Higher statistics or a smooth mass/concentration model are preferable '
                'for the 21- and 6-halo bins; independent draws of each parameter would destroy their observed covariance.')

    report.start('Overlap, support and kSZ interpretation')
    report.text('All-gas radial profiles include central-bound, own-satellite-bound, other-halo-bound and unbound gas. '
                'The no-other control removes only the other-halo-bound component of the calibration target; it is not '
                'a global particle ownership model. Painting overlapping all-gas templates adds them, so gas from shared '
                'environments can be represented twice. The measured control instead masks the unique geometric union of '
                'central spheres, counting each coarse gas voxel once.')
    report.text('Union masks use 0.977 Mpc voxel centres and are not exact particle-sphere membership. Small halos and 1R apertures '
                'are poorly resolved; the associated RMS cannot alone choose the optimal halo-core model. No binding labels '
                'are removed from this union. The native union and model are compared at the same 256 squared projected resolution.')
    report.text('Changing Rfit modifies the inferred central amplitude and shape even with fixed Rcut. Changing Rcut modifies '
                'the line-of-sight column at smaller projected radii, because outer gas can lie in front of or behind the centre. '
                'The finite-sphere quadrature includes this effect; no average over correlated rotations is used. '
                'The matrices explicitly compare each off-diagonal cell to its same-cut or same-fit diagonal reference.')
    report.text('The proposed Pe fit2/cut4 and ne fit3/cut3 templates optimize different observables. They do not constitute a '
                'single complete gas fluid in 3-4R: nonzero thermal pressure requires electrons. A common-support ne cut4 control '
                'is provided. kSZ additionally depends on internal gas motion and density-velocity correlation; painting assigns '
                'each halo its bulk velocity, while the direct gas map uses each particle velocity. A kSZ discrepancy cannot be '
                'attributed solely to the density profile.')

    report.start('Resources, verification and limits')
    resources(report)
    report.text('Numerical gates verify physical normalization, finite-sphere projection, uniform-velocity kSZ closure, '
                'projection signal conservation and catalogue unit conversion. The corrected SD fits reproduce the established '
                'cached profiles to a maximum fractional difference of 1.65e-5 over 180 comparisons. '
                'All case files and compact exports are SHA-256 checked. The cache-lock retry changes synchronization only; '
                'science-function syntax trees were compared to the preserved executed source.',small=True)

    report.start('Available systems and accounting limits')
    used=load('resource_accounting_summary.json')
    report.text('The following shared-resource observations were captured during this study on 14 October 2026. '
                'They describe available hardware and filesystem space at that time, not reserved capacity or personal quotas. '
                'idark was unreachable during final report delivery, so its live availability was not reverified.')
    report.table([
        ['System / resource','Recorded capacity or policy'],
        ['STRW maas','24 physical CPU cores / 48 hardware threads; 188 GiB RAM, about 176 GiB available at capture'],
        ['STRW data filesystem','/data1: 129 GB total, 80 GB available; separate home quota about 3.9 GB'],
        ['idark mini queue','One node per job; at most 52 CPUs per job; max_user_run = 6 at capture'],
        ['idark Lustre','About 151 TB available; 95% occupied. Filesystem free space does not establish a user quota.'],
        ['This study',f"15 job instances; {used['timed_idark_cpu_hours']:.3f} measured CPU-hours in explicitly matched GNU time logs"],
        ['Allocation lower bound',f"{used['allocation_core_hours_lower_bound']:.3f} core-hours = 26 CPUs times timed application durations"]
    ],[205,W-2*M-205])
    report.text('The allocation lower bound excludes untimed setup, PDF rendering and archive creation. It is not a billing '
                'measurement. The final PBS queries returned no usage fields, so missing CPU/allocation totals are null in '
                'resource_accounting_summary.json. This avoids mistaking unavailable data for zero resource use. '
                'Adding the separate STRW particle pass gives about 4.48 measured CPU-hours, plus untimed stages.')
    report.text('The final report layout was completed locally from verified compact outputs under a Windows/WSL memory guard. '
                'The guard requires 8 GiB available to start and stops the local process below 5 GiB. '
                'No additional cluster jobs, particle reads or profile refits were required for this delivery.',small=True)


def appendix_figures(report):
    for field,title in [('y','Compton-y'),('dm','Dispersion measure'),('deltaT','kSZ temperature'),('tau','Optical depth')]:
        report.figure('maps_'+field,title+' maps and absolute differences',
            'Equal 250 Mpc regions and depths. Shared signal limits across the top row; a separate symmetric residual scale below. '
            'Different simulations have different initial phases. Only same-simulation residuals are meaningful pixel by pixel.')
    for field,title in [('pressure','Electron pressure'),('electrons','Electron density'),('y','Compton-y'),
                        ('tau','Optical depth'),('dm','DM'),('deltaT','kSZ temperature')]:
        report.figure('pdf_'+field,title+': pixel / cell probability distributions',
            'Every defined grid cell or map pixel is included. Positive fields are shown per log10 interval; kSZ is signed. '
            'Zero atoms and histogram tails are retained in JSON, outside the plotted continuous density. '
            'Bands show 16-84% spatial-region variation. Resolution or same-population power controls appear at right.')
    for field,title in [('y','Compton-y'),('tau','Optical depth'),('dm','DM'),('deltaT','kSZ temperature')]:
        report.figure('power_'+field,title+': projected physical-plane power',
            'P2D uses mean-subtracted 250 Mpc maps and physical transverse wavenumber, not angular multipole. '
            'The curve is k squared P2D / (2 pi). The vertical guide marks half the 512-pixel Nyquist wavenumber. '
            'Cell assignment and finite resolution affect high k; spatial bands are not covariance estimates.')
        report.figure('matrix_'+field,title+': fit radius by truncation radius',
            'Mean power over 0.05 <= k <= 0.5 Mpc^-1 in HalfDome. Rows change fit radius; columns change spherical cut. '
            'References: fit4/cut4, same-cut diagonal, and same-fit diagonal. The last two make each matched fit/cut diagonal equal to one. '
            'These are ratios of band-averaged power, not fitted-parameter ratios.')
    for field,title in [('y','Compton-y'),('dm','DM'),('deltaT','kSZ temperature')]:
        report.figure('whole_box_'+field,title+': whole-box overview',
            'Left and centre project the full 1000 Mpc FLAMINGO box; right projects the 5535.87 Mpc HalfDome box. '
            'Both use 1024 squared pixels, so physical resolution and line-of-sight depth differ. '
            'Use the equal-region figures for quantitative comparisons.')
    for field,title in [('pressure','Pressure'),('electrons','Electron density'),('y','Compton-y'),
                        ('tau','Optical depth'),('dm','DM'),('deltaT','kSZ temperature')]:
        report.figure('halo_union_pdf_'+field,title+': measured union-of-halos controls',
            'Measured gas is retained in the unique voxel-centre union of 1, 2, 3 or 4R200c central spheres. '
            'Overlap is counted once; gas binding is not removed. All cells/pixels, including exterior zeros, define the sample space. '
            'Projected model maps are rebinned to the same 0.977 Mpc pixels. Coarse small-halo geometry limits this diagnostic.')
    for b in range(15):
        lo=12.5+.2*b
        report.figure(f'profiles_bin_{b:02d}',f'Profile fits: {lo:.1f}-{lo+.2:.1f} in log10 solar mass',
            'Top: measured spherical-shell mean, halo scatter and 1-4R fits. Parameters are shown in legend order '
            '[P0, xc, beta] and [rho0, alpha, beta]. Dashed portions extrapolate beyond Rfit to 4R. '
            'Bottom: absolute normalized model-minus-mean residual. Open points are excluded fit bins; low-mass support needs resolution review.')
    for field,title in [('pressure','Pressure parameters'),('electrons','Electron-density parameters')]:
        report.figure('parameters_'+field,title+': mean uncertainty and halo diversity',
            'Top: best-fit curves and bootstrap median with 16-84% error bars for the empirical mean. '
            'Bottom: accepted individual-fit 16-84% spread, including measurement/fit degeneracy and selection effects. '
            'Parameter draws should be correlated. Neither panel is a calibrated full inference covariance.')
    for b in [3,10,14]:
        report.figure(f'scatter_closure_bin_{b:02d}',f'Does fitted-parameter scatter reproduce profile scatter? Bin {b}',
            'Black: standard deviation of all measured halo profiles. Orange: standard deviation of profiles generated by '
            'accepted individual parameter fits. The samples differ when fits are rejected; mismatch combines selection and '
            'model inadequacy. This check is necessary before using parameter scatter as a realistic painting prescription.')


def code_pages(report):
    report.start('Code example: unit-correct native deposits')
    report.text('The real extraction obtains conversion factors from each dataset\'s physical-CGS attributes. '
                'The following compact example shows the exact physical quantities accumulated; variable names state their units. '
                'The snapshot is streamed in bounded chunks, rather than loading all particles.')
    report.code('''
        volume_mpc3 = mass_msun / density_msun_mpc3
        electron_volume = electron_density_cm3 * volume_mpc3
        momentum_volume = electron_volume * velocity_z_km_s
        # Y_native_mpc2 already integrates electron pressure over particle volume.
        deposits = [Y_native_mpc2, electron_volume, momentum_volume]
        y, tau, dm, deltaT = physics.map_observables(pixel_sums, pixel_area_mpc2)
    ''')
    report.text('For reference, the production converter in physics.py is:')
    report.code('''
        y = sums[0] / pixel_area
        tau = NE_VOLUME_TO_TAU_AREA * sums[1] / pixel_area
        dm = 1e6 * sums[1] / pixel_area
        dt = -T_CMB * 1e6 / (c / 1000) * NE_VOLUME_TO_TAU_AREA * sums[2] / pixel_area
    ''')
    report.text('sums has the three integrated deposits in the order above. NE_VOLUME_TO_TAU_AREA = sigma_T[cm^2] '
                'times one Mpc in cm. c/1000 converts SI speed of light to km/s. The 10^6 in dt converts kelvin to microkelvin. '
                'No additional hydrogen mass fraction is applied to native electron density.')

    report.start('Code example: access compact maps and fits')
    report.code('''
        import h5py
        import json
        import physics

        with h5py.File('measured_flamingo_compact.h5', 'r') as f:
            deposits = f['region_sums'][0]       # first 250 Mpc cube
        y, tau, dm, deltaT = physics.map_observables(deposits, (250 / 512)**2)

        fits = json.load(open('fits.json'))
        result = next(r for r in fits['results']
                      if r['field'] == 'pressure' and r['component'] == 'all'
                      and r['bin'] == 10 and r['cut'] == 2)
        print(result['parameters'])             # [P0, xc, beta]
        print(result['bootstrap_parameters'])   # paired halo resampling
    ''')
    report.text('Full case maps and their receipts remain on idark under cases/. Measured full-precision compact maps and '
                'cached 3D grids remain on STRW. Only compact map exports and statistics are transferred. '
                'The local delivery bundle contains figures, fit distributions, metrics and accounting; it does not duplicate '
                'all 59 case map files. export receipts specify float32 map precision; histogram/moment precision is retained.')
    report.text('Cluster access uses existing authenticated SSH aliases: <b>ssh idark</b> and <b>ssh strw-maas</b>. '
                'STRW gateway login is nagainis@ssh.strw.leidenuniv.nl; Microsoft Authenticator remains a private interactive step '
                'when the existing connection is unavailable. Simulation data are under /net/hypernova/data2/FLAMINGO. '
                'The study\'s home path links to /data1/nagainis/HalfDome/flamingo_pressure_calibration/.')

    report.start('Files created, provenance and references')
    report.text('All edited or created analysis files are in this new isolated study directory. Existing calibration products '
                'and raw simulation files were not edited. FILES_CHANGED.md lists every source file and its role; '
                'delivery_manifest.json records local delivery hashes, and CLUSTER_VERIFIED.json records live cluster verification.')
    report.table([
        ['File group','Purpose'],
        ['physics.py; prepare_idark.py','Units, cosmology, released mass mapping and profile fits'],
        ['extract_strw.py; compact_strw.py; halo_union_strw.py','One particle pass, conservation merge and cached halo-union controls'],
        ['painting.py; run_comparisons.py; verify_numerics.py','Finite-sphere painting, 57 cases and numerical gates'],
        ['field_statistics.py; aggregate_and_plot.py','All-pixel PDFs, projected power, residuals and figures'],
        ['cluster_control.py; sync_metadata.py; finalize_control.py; *.pbs','Bounded deployment, job submission and receipts'],
        ['make_reports.py; README.md','Reproducible PDF authoring and documented contracts']
    ],[350,W-2*M-350])
    report.text('Primary references: '
                '<link href="https://arxiv.org/abs/1109.3711" color="#17638a">Battaglia et al. (2012), pressure / tSZ</link>; '
                '<link href="https://arxiv.org/abs/1607.02442" color="#17638a">Battaglia (2016), electron-density / optical-depth profiles</link>; '
                '<link href="https://arxiv.org/html/2205.01710" color="#17638a">Lee et al. (2022), cluster tSZ/kSZ profiles</link>; '
                '<link href="https://arxiv.org/html/2407.17462v2" color="#17638a">HalfDome release paper</link>. '
                'Lee22 uses 0.2 dex h-scaled mass bins, a narrower fitted radial range and covariance-aware fitting; '
                'our wider-radius diagonal empirical fit does not reproduce its likelihood. '
                'Its TNG electron abundance requires a hydrogen conversion, whereas FLAMINGO native electron density does not.',small=True)


def main():
    (ROOT/'report.pid').write_text(str(os.getpid()))
    assert load('cluster_analysis_complete.json')['completed']
    findings=load('quantitative_findings.json')
    mapping=load('mass_mapping.json')
    fits=load('fits.json')
    outputs=[make_summary(findings,mapping)]
    report=Report('FLAMINGO_HalfDome_z0_detailed_report.pdf','FLAMINGO to HalfDome: methods and full diagnostics')
    method_pages(report,findings,mapping,fits)
    appendix_figures(report)
    code_pages(report)
    outputs.append(report.finish())
    P.save_json(ROOT/'reports_complete.json',dict(completed=True,outputs=outputs,
        authoring_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(outputs,indent=2))


if __name__=='__main__':
    main()
