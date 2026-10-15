#!/usr/bin/env python3
"""Build one illustrated report from verified, frozen numerical evidence.

Run collect_evidence.py, obtain PEARSON_COMPLETE.json from idark, then run
report_figures.py and this builder. All plots and equations are real renderings;
source excerpts are extracted from delivered files, not recreated pseudocode.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence';S=E/'study';D=E/'diagnosis';F=ROOT/'figures'
OUT=ROOT/'output/pdf';TMP=ROOT/'tmp/pdfs'
for folder in [OUT,TMP]:folder.mkdir(parents=True,exist_ok=True)
font_directory=Path('/usr/share/fonts/truetype/dejavu')
if not font_directory.exists():
    font_directory=Path(matplotlib.get_data_path())/'fonts/ttf'
for family,name in [('DejaVuSans','Body'),('DejaVuSans-Bold','Bold'),('DejaVuSansMono','Mono')]:
    pdfmetrics.registerFont(TTFont(name,str(font_directory/(family+'.ttf'))))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
W,H=landscape(A4);M=42;CW=W-2*M;COL=(CW-30)/2
INK='#182a3a';BLUE='#2765a4';ORANGE='#cc682e';TEAL='#218671';LIGHT='#edf3f7'
PARAMS=['P0','xc','beta','alpha_m_P0','alpha_m_xc','alpha_m_beta','alpha_z_P0','alpha_z_xc','alpha_z_beta']
TRUTH=[18.1,.497,4.35,.154,-.00865,.0393,-.758,.731,.415]
LOW=[1,.025,2.8,-.6,-1,-.2,-6,-1.5,-.5]
HIGH=[60,4,16,1.5,.4,.4,.5,3,2]
EXCERPTS=[];PAGES=[]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class Report:
    def __init__(self,draft=False):
        self.path=(TMP/'prior_sections_layout.pdf') if draft else (OUT/'HalfDome_Merlin_seed_mask_and_Pearson_report.pdf')
        self.c=canvas.Canvas(str(self.path),pagesize=(W,H),pageCompression=1)
        self.c.setTitle('HalfDome tSZ: simulation seeds, masks, bias controls and posterior correlations')
        self.c.setAuthor('HalfDome research analysis')
        self.number=0

    def page(self,title,section='HalfDome / Battaglia12 / SO baseline deproj0'):
        if self.number:self.c.showPage()
        self.number+=1;PAGES.append(dict(page=self.number,title=title))
        key=f'p{self.number}';self.c.bookmarkPage(key);self.c.addOutlineEntry(title,key,level=0)
        self.c.setFillColor(colors.HexColor(BLUE));self.c.setFont('Bold',10)
        self.c.drawString(M,H-29,section.upper())
        title_size=min(25,25*CW/pdfmetrics.stringWidth(title,'Bold',25))
        self.c.setFillColor(colors.HexColor(INK));self.c.setFont('Bold',title_size)
        self.c.drawString(M,H-68,title)
        self.c.setStrokeColor(colors.HexColor('#d5e1e9'));self.c.line(M,H-81,W-M,H-81)
        self.c.setFont('Body',9);self.c.setFillColor(colors.HexColor('#667789'))
        self.c.drawString(M,24,'05 October 2026   |   idark analysis   |   Unbounded model is primary')
        self.c.drawRightString(W-M,24,str(self.number))
        return H-103

    def text(self,text,y,x=M,width=CW,size=12,leading=None,color=INK):
        style=ParagraphStyle('p',fontName='Body',fontSize=size,leading=leading or size*1.45,
                             textColor=colors.HexColor(color),spaceAfter=0)
        p=Paragraph(text,style);_,height=p.wrap(width,1000)
        if y-height<48:raise ValueError(f'Page {self.number} text overflow: {text[:65]}')
        p.drawOn(self.c,x,y-height);return y-height-12

    def heading(self,text,y,x=M,width=CW):
        return self.text('<b>'+text+'</b>',y,x,width,size=15,color=BLUE)

    def equation(self,expression,y,width=CW):
        name=hashlib.sha256(expression.encode()).hexdigest()[:16]
        path=TMP/(name+'.png')
        if not path.exists():
            fig=plt.figure(figsize=(12,1));fig.text(.01,.5,'$'+expression+'$',fontsize=22,va='center')
            fig.savefig(path,dpi=350,bbox_inches='tight',pad_inches=.10,facecolor='white');plt.close(fig)
        im=Image.open(path);h=width*im.height/im.width
        if h>50:width*=50/h;h=50
        self.c.drawImage(str(path),M,y-h,width=width,height=h,mask='auto')
        return y-h-13

    def picture(self,path,y,height=330,width=CW,x=M):
        path=Path(path);im=Image.open(path);scale=min(width/im.width,height/im.height)
        w,h=im.width*scale,im.height*scale
        if y-h<48:raise ValueError(f'Page {self.number} image overflow {path}')
        self.c.drawImage(str(path),x+(width-w)/2,y-h,width=w,height=h,mask='auto')
        return y-h-14

    def figure_page(self,title,path,caption,section='Results',height=335):
        y=self.page(title,section);y=self.picture(path,y,height=height)
        return self.text(caption,y,size=11.5)

    def table(self,headers,rows,y,widths=None,size=11):
        widths=widths or [CW/len(headers)]*len(headers)
        st=ParagraphStyle('t',fontName='Body',fontSize=size,leading=size*1.3,textColor=colors.HexColor(INK))
        data=[[Paragraph('<b>'+html.escape(str(v))+'</b>',st) for v in headers]]
        data.extend([[Paragraph(str(v),st) for v in row] for row in rows])
        t=Table(data,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f2')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),9),
            ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#afc6d8'))]))
        _,height=t.wrap(CW,1000)
        if y-height<48:raise ValueError(f'Page {self.number} table overflow ({height})')
        t.drawOn(self.c,M,y-height);return y-height-15

    def code(self,path,lo,hi,y,label=None,size=10):
        lines=path.read_text().splitlines();raw='\n'.join(lines[lo-1:hi])
        EXCERPTS.append(dict(file=str(path.relative_to(ROOT)),first_line=lo,last_line=hi,
                             source_sha256=sha(path),code=raw))
        if label:y=self.text(label,y,size=10,color=BLUE)
        visible=[]
        for num,line in enumerate(lines[lo-1:hi],lo):
            prefix=f'{num:3} '
            maxchars=int((CW-20)/pdfmetrics.stringWidth('M','Mono',size))
            chunks=textwrap.wrap(line,maxchars-4,replace_whitespace=False,drop_whitespace=False) or ['']
            visible.append(prefix+chunks[0])
            visible.extend('    '+chunk for chunk in chunks[1:])
        h=len(visible)*(size*1.38)+18
        if y-h<48:raise ValueError(f'Page {self.number} code overflow: y={y:.1f}, height={h:.1f}, {path}')
        self.c.setFillColor(colors.HexColor(LIGHT));self.c.roundRect(M,y-h,CW,h,5,fill=1,stroke=0)
        self.c.setFillColor(colors.HexColor(INK));self.c.setFont('Mono',size)
        for i,line in enumerate(visible):self.c.drawString(M+10,y-13-i*size*1.38,line)
        return y-h-14

    def finish(self):
        self.c.save()
        (ROOT/'source_excerpts.json').write_text(json.dumps(EXCERPTS,indent=2)+'\n')
        (ROOT/'page_manifest.json').write_text(json.dumps(PAGES,indent=2)+'\n')
        print(self.path, 'pages=',self.number,'sha256=',sha(self.path))


def main(draft=False):
    summary=read(S/'summary.json')
    if draft:
        definition={'target':'relative_change'}
    else:
        pearson=read(ROOT/'pearson_linear.json');bounded=read(ROOT/'pearson_logit.json')
        definition=read(ROOT/'control_definition.json')
        assert (ROOT/'PEARSON_COMPLETE.json').exists()
    r=Report(draft=draft);src=E/'study_source'
    y=r.page('One pressure law, many skies and windows','Combined research report')
    y=r.text('Simulation-seed variation, mask variation, bias diagnosis, and the posterior Pearson comparison',y,size=20,leading=28)
    y=r.text('Battaglia12 tSZ • ten Gaussian HalfDome lightcones • ten masks • one SO baseline deproj0 noise prescription',y,size=13)
    y=r.heading('What the completed experiments establish',y)
    y=r.text('The retrained 50-multipole Merlin model is the best tested compromise under the requested <b>25% average width budget</b>. On unused skies/windows it reduces truth displacement by 18-33% and variation between observations by 68-79%. It retains the full multipole range, while discarding structure within each band.',y)
    y=r.text('The original model learned a posterior conditional on <b>simulation 100 and mask 12345</b>. Fine changes in clean tSZ power drive most of the common upward P₀ shift on new skies. The shifts are similar rather than exactly constant. Restoring agreement with the training sky does not guarantee unbiased marginal medians.',y)
    target='saved fractional smooth-change control: baseline plus the smooth clean difference of the other nine simulations, with its SO residual fixed'
    if draft:
        y=r.text('The new baseline Pearson section is pending a scientific definition of what to smooth. This intermediate layout draft contains only completed earlier analyses and is not the final delivery.',y)
    else:
        y=r.text('The new Pearson comparison uses seed 100 / mask 12345 as its reference and the <b>unchanged original models</b>. Its smoothing target is the '+target+'. The result is a diagnostic of the posterior response, not a separately calibrated inference method.',y)
    y=r.text('Scope: this report concerns B12 tSZ SBI. It does not draw conclusions about f<sub>NL</sub>, Lee22, kSZ or FRB DM. All numerical results are tied to the delivered completion records and source excerpts.',y,size=11)

    y=r.page('Reader’s guide and evidence hierarchy')
    y=r.table(['Part','What to look for'],[
        ['Physical and statistical contract','Pages 3-8: pressure parameters, finite projection, signed cross-spectrum, fixed noise draws, full-grid design and shared painting.'],
        ['Mitigation experiment','Pages 9-21: all 23 new recipes, selection criterion, separate simulation/mask/joint results, width costs, joint truth and reserved prior tests.'],
        ['Why the original offsets share a direction','Pages 22-26: common-reference statistics, clean/noise factorization, smoothing/fine controls, implementation and support checks.'],
        ['New baseline Pearson experiment','Pages 27-32: smoothing definition, physical-parameter matrices, changes in correlation, joint clouds, bounded sensitivity and numerical tables.'],
        ['Recommendations and reproducibility','Pages 33-36: conclusions within the width budget, a possible nuisance-aware MOPED, artifact and code inventory, references.'],
    ],y,[180,CW-180])
    y=r.text('<b>Evidence hierarchy.</b> Full-grid observations test sensitivity at one known B12 point. Reserved parameter/noise rows test the fixed training sky. Counterfactual spectra identify mechanisms. None of these alone establishes calibration over independent halo realizations throughout the prior.',y)
    y=r.text('Original and selected full-grid posteriors use 90,000 mixture draws per observation. The earlier mechanism controls use 9,000 draws. New Pearson matrices use all 90,000 physical-parameter draws per observation; Monte Carlo precision is reported separately.',y,size=11.5)

    y=r.page('The pressure law and physical projection','Physical setup')
    y=r.equation(r'P_e/P_{200}=P_0(M,z)\,u^{-0.3}(1+u)^{-\beta(M,z)},\quad u=\frac{r/R_{200c}}{x_c(M,z)}',y)
    y=r.text('P<sub>e</sub> is electron pressure and P<sub>200</sub> is the common physical mass/redshift pressure scale. R<sub>200c</sub> encloses a mean density 200 times the critical density. P₀ sets amplitude, x<sub>c</sub> the dimensionless radius scale, and β the outer decline. The inner slope is fixed at -0.3 and the transition exponent at 1. [1]',y)
    y=r.equation(r'A(M,z)=A_0\left(\frac{M}{10^{14}M_\odot}\right)^{\alpha_{m,A}}(1+z)^{\alpha_{z,A}},\quad A\in\{P_0,x_c,\beta\}',y)
    y=r.text('A₀ is the value at the pivot mass and z=0. The six α parameters control the mass and redshift dependences of the three shape parameters. Both scaling factors multiply. M is physical M<sub>200c</sub>, obtained from the supplied M⊙/h catalogue masses using h=0.68.',y)
    y=r.equation(r'y=\frac{\sigma_T}{m_ec^2}\int_{r<4R_{200c}}P_e\,dl_{\rm phys}',y,width=520)
    y=r.text('Compton-y is the pressure integral along physical line-of-sight length l<sub>phys</sub>; σ<sub>T</sub> is the Thomson cross section, m<sub>e</sub> the electron mass and c the speed of light. Halo contributions are summed. The finite <b>4R<sub>200c</sub> sphere</b> matches this tSZ training pipeline. All available finite nonnegative redshifts are included; there is no z&lt;1 cut.',y)

    y=r.page('Nine parameters: truth, order and training prior','Physical setup')
    names=['P₀','x<sub>c</sub>','β','α<sub>m,P₀</sub>','α<sub>m,xc</sub>','α<sub>m,β</sub>',
           'α<sub>z,P₀</sub>','α<sub>z,xc</sub>','α<sub>z,β</sub>']
    y=r.table(['Parameter','B12 generating value','Uniform training interval'],
              [[n,f'{t:g}',f'[{lo:g}, {hi:g}]'] for n,t,lo,hi in zip(names,TRUTH,LOW,HIGH)],y,[220,230,CW-450],size=11)
    y=r.text('This order is used in every spectrum, posterior file, matrix and numerical table. The primary model uses linear parameter coordinates and permits posterior draws beyond the training intervals. Such extrapolation is not validated physics. The secondary model uses a logit transform and stays inside the intervals.',y,size=11.5)

    y=r.page('Observation: one noise prescription, two fixed draws','Observation contract')
    y=r.text('Each observation is a cross-spectrum of signal plus two independent noise draws from the <b>same SO baseline deproj0 prescription</b>. Draw seeds 39001001 and 39001002 are held fixed throughout the experiment. This preserves the trained observation, removes auto-noise bias in expectation, and isolates sky/window effects in paired comparisons. Signal-noise and noise-noise fluctuations still remain.',y)
    y=r.equation(r'D_\ell=\frac{\ell(\ell+1)}{2\pi}C_\ell,\quad 80\leq\ell\leq7979',y,width=550)
    y=r.text('C<sub>ℓ</sub> is the signed masked angular cross-power and D<sub>ℓ</sub> its conventional rescaling. The cap covers 40% of the sky with a 60-arcmin cosine apodization. These are pseudo-spectra: no f<sub>sky</sub> division, mask deconvolution or auto-spectrum noise subtraction is introduced.',y)
    y=r.equation(r'D_{\rm obs}=D_{ss}+B(D_{\rm prod}-D_{ss}-D_{nn})+B^2D_{nn}',y)
    y=r.text('The saved trained correction uses the 2-arcmin Gaussian harmonic beam B<sub>ℓ</sub>. ss is the beam-smoothed clean signal, prod the original noisy cross-spectrum and nn the unfiltered noise-noise cross-spectrum. The B term contains signal-noise power. Masking and beam smoothing do not commute exactly; the map-exact check was much smaller in feature space than the seed/mask effect.',y)
    y=r.code(src/'pipeline.py',61,63,y,'Executed beam-correction implementation: pipeline.py',size=10)

    y=r.page('What the original Merlin was trained to infer','SBI contract')
    y=r.equation(r'q_\phi(\theta\mid D)\simeq p(\theta\mid D,H_{100},W_{12345})',y,width=650)
    y=r.text('θ is the ordered nine-parameter pressure vector; D the corrected pseudo-spectrum; H the halo lightcone and W the apodized mask. Training varies θ and SO noise while holding H and W fixed. The network therefore learns a conditional experiment, rather than marginalizing over cosmic realizations and masks.',y)
    y=r.equation(r'p(\theta\mid D)\ \propto\ p(\theta)\int p(D\mid\theta,H,W)\,p(H,W)\,dH\,dW',y)
    y=r.text('General-sky inference requires the integral over nuisance realizations H and W. The current dataset does not implement that integral. A broad parameter prior can contain spectra resembling new skies at incorrect θ, even when the new observation is close to the global training support.',y)
    y=r.heading('Saved preprocessing and models',y)
    y=r.text('The original fine-spectrum model whitens with a noise-only Cholesky factor, retains 83 PCA coordinates, applies an asinh transform scaled by the SO residual median absolute deviation, and standardizes the transformed features. A learned head feeds a masked autoregressive flow. Three independently trained networks form an equal-mixture ensemble.',y)
    y=r.text('The reused corrected dataset has 64,440 rows: 61,440 training-pool rows and 3,000 reserved parameter/noise rows. Reconstructed saved feature rows agree exactly (64-row audit). New compressions are retrained on the same corrected input dataset; input smoothing in the new Pearson control keeps the original models frozen.',y)

    y=r.page('A full 10 × 10 design with separate tests','Experiment design')
    y=r.picture(F/'grid_design.png',y,height=315)
    y=r.text('Simulation seeds are 100, 102, …, 118; masks are 12345, …, 12354. The black cell is the training baseline. The initial separate arms had 19 distinct observations; the full grid has 100 and reveals interactions. Selection uses only the 30 discovery cells. Untouched tests have 4 simulation, 5 mask and 20 jointly changed cases. The remaining 41 mixed cells are descriptive.',y,size=11.5)
    y=r.text('The masks overlap and every observation shares the same two noise maps. These paired, correlated cases measure sensitivity; they are not 100 independent cosmic/noise realizations or an independent-mock covariance estimate. Seed 120, the non-Gaussian lightcone, is excluded from this Gaussian-seed study.',y,size=11)

    y=r.page('One painting per lightcone; reuse across all masks','Implementation and checks')
    y=r.code(src/'paint_seed.jl',216,217,y,'Shared catalogue pass: paint_seed.jl',size=10)
    y=r.code(src/'paint_seed.jl',232,242,y,'Noise maps created once; ten windows reuse the same painted signal',size=10)
    y=r.text('Seed 100 reuses the validated earlier painting. Nine other jobs paint each catalogue once. The count assertion applies only to the known reference seed; other seeds legitimately have different halo counts (85.216-85.268 million selected halos). The common physical mass cut is M<sub>200c</sub> ≥10¹² M⊙, with Ω<sub>b</sub>=0.049 and Ω<sub>cdm</sub>=0.261.',y,size=11.5)
    y=r.text('Raw HEALPix NSIDE=8192 is beam-smoothed to 2 arcmin and reduced to NSIDE=4096. The interpolation cache spans z=0.001-5; its largest relative visible-profile error is 2.41×10⁻⁴ and relative L₂ error 1.83×10⁻⁵. Consistent cache and selection contracts were checked for every lightcone.',y,size=11)

    r.figure_page('The spectra vary much less than the inferred P₀',S/'signal_variation.png',
        'The clean B12 spectra differ between Gaussian skies and between windows. Relative to seed 100, the other nine simulations average -6.14% at ℓ=80-579, -0.063% at 580-2979 and +0.742% at 2980-7979. These block averages exclude the reference itself. P₀ medians nevertheless shift from about 12 to about 50 in the original model; this is not explained by a comparable pressure-amplitude change.',height=345)

    y=r.page('Selection rewards truth recovery and agreement','Metrics and width budget')
    y=r.equation(r'R=\left[\frac{1}{9N}\sum_{i,p}\left(\frac{m_{ip}-\theta_p^*}{s_p}\right)^2\right]^{1/2},\quad A=\left[\frac{1}{9N}\sum_{i,p}\left(\frac{m_{ip}-\bar m_p}{s_p}\right)^2\right]^{1/2}',y)
    y=r.text('m<sub>ip</sub> is posterior median p in observation i, θ* the known B12 truth, N the number of observations in the arm, and m̄<sub>p</sub> their mean median. The fixed scale s<sub>p</sub> is half the original baseline 68% width. R measures truth displacement and A agreement between observations. Broadening a candidate cannot shrink either metric by changing the normalization.',y)
    y=r.equation(r'G_p=\exp\left[\frac{1}{N}\sum_i\ln\frac{w_{ip}}{w_{ip}^{\rm original}}\right],\quad G=\exp\left[\frac{1}{9}\sum_p\ln G_p\right]',y)
    y=r.text('w is the marginal q₈₄-q₁₆ width. G is the geometric mean width ratio over observations and parameters; max G<sub>p</sub> records the worst parameter. Selection minimizes √(R²+A²) on discovery skies, requires both R and A to improve, G≤1.25, and broad marginal/joint reserved-row screens to pass. Final test skies are not used for re-ranking.',y)
    y=r.code(src/'report.py',44,51,y,'The fixed-scale metric implementation: report.py',size=10)

    y=r.page('All 23 new inference recipes were tested','Mitigation methods')
    y=r.table(['Family','Variants','Physical/statistical purpose'],[
        ['MOPED, full range (2)','Noise-only covariance; total SO residual covariance','Separate the effect of the covariance used to optimize nine pressure-score directions.'],
        ['MOPED, upper cuts (4)','ℓ<sub>max</sub>=5979, 4979, 3979, 2979','Remove progressively more small-angular-scale information.'],
        ['MOPED, lower/both cuts (4)','ℓ<sub>min</sub>=280, 580, 1080; 580-4979','Remove large angular scales and/or constrain the retained window.'],
        ['MOPED, covariance taper (1)','Smooth high-ℓ downweighting above ℓ≈4000','Reduce small-scale score weight while preserving the multipole range.'],
        ['MOPED, sky covariance (4)','Simulation, mask, combined; weak combined','Use discovery residual covariance as a nuisance penalty in the score weights.'],
        ['MOPED, augmentation (1)','Discovery clean-spectrum ratios','Approximate sky variation during retraining.'],
        ['Merlin (7)','Full band50; lower cuts 280/580/1080; upper 4979/3979; augmentation','Retest learned compression with banding, cuts or approximate sky variation.'],
    ],y,[175,245,CW-420],size=10.5)
    y=r.text('All recipes use 50-multipole weighted bands before their alternative compression. Requested cut boundaries are rounded inward to complete bands; the values above are effective inclusive bounds. The original 7,900-multipole / 83-PCA Merlin model is the additional reference. [2]',y,size=11)

    y=r.page('Banding and MOPED have different information costs','Compression implementation')
    y=r.equation(r'd_b=\frac{\sum_{\ell\in b}(2\ell+1)D_\ell}{\sum_{\ell\in b}(2\ell+1)},\quad b=1,\ldots,158',y,width=600)
    y=r.text('d<sub>b</sub> averages each 50-multipole band. The selected Merlin model keeps all 158 bands across ℓ=80-7979, but loses variations inside bands. Its learned head is retrained. The executed Merlin branch explicitly uses <b>noise-only C<sub>nn</sub></b>, even though a generic recipe metadata field says total noise.',y)
    y=r.code(src/'pipeline.py',164,170,y,'The actual covariance branch: pipeline.py',size=10)
    y=r.code(src/'common.py',48,54,y,'The (2ℓ+1)-weighted band operator: common.py',size=10)
    y=r.page('MOPED implementation and its local guarantee','Compression implementation')
    y=r.equation(r'C=LL^T,\quad L^{-1}J\,{\rm diag}(\Delta\theta)=QR,\quad W=L^{-T}Q,\quad W^TCW=I_9',y)
    y=r.text('C is the chosen band covariance, J the band-mean derivatives with respect to θ, Δθ the prior widths, and QR the stable thin factorization. W maps bands to nine MOPED summaries. All nine locally resolved pressure directions are retained. Local mean-derivative Fisher preservation for this C does not establish nonlinear, global SBI recovery or nuisance marginalization. [2]',y)
    y=r.code(src/'common.py',89,98,y,'Stable QR weights and orthogonality gate: common.py',size=9.5)

    y=r.page('Sky penalties and augmentation: what changed','Nuisance implementations')
    y=r.equation(r'C_{\rm eff}=C_{\rm SO}+\lambda S_{\rm sky},\qquad d^{\rm aug}=d^{\rm noisy}+d^{\rm clean}(r_k-1)',y)
    y=r.text('S<sub>sky</sub> is the discovery covariance of clean spectrum residuals normalized to the B12 reference; λ is 1 or 0.1. r<sub>k</sub> is a sampled discovery clean-spectrum ratio. Augmentation changes the clean component while preserving the saved SO residual. It approximates the nuisance response at other pressure parameters and omits the corresponding changed signal-noise term.',y)
    y=r.text('Adding S only to MOPED’s weighting covariance downweights nuisance-sensitive directions; it does not integrate H and W out of a fixed-sky training simulator. The high-ℓ taper instead multiplies C by outer products of a smooth factor that rises from 1 above ℓ≈4000 toward √10 at the highest ℓ.',y)
    y=r.code(src/'merlin_checks.py',59,67,y,'Corrected Merlin augmentation preprocessing: merlin_checks.py',size=10)
    y=r.text('The asinh noise scale is computed after subtracting the <b>augmented clean signal</b>; otherwise sky variation would contaminate the supposedly noise-only residual scale. The earlier generic MOPED augmentation path used the baseline clean subtraction. These are approximate controlled recipes, not certified replacements for many-catalogue simulations. [3,5]',y)

    r.figure_page('Simulation variation: the common shift is reduced',F/'posterior_simulation.png',
        'Fixed mask 12345; points are medians, bars central 68% intervals and dashed lines B12 truth. The orange retrained band50 results agree across skies, including seed 100. On the four unused simulation seeds: R decreases 1.746→1.169, A 0.157→0.050, average width increases 24.7%, and the worst parameter increases 59.1%. Agreement does not imply unbiased marginal medians.',height=335)

    r.figure_page('Mask variation: noise-only changes are insufficient',F/'posterior_mask.png',
        'Fixed simulation 100, changing only the cap/window seed. On five unused masks: R decreases 1.419→1.162 and A 0.459→0.098. Average width grows 17.9%, with a worst-parameter increase of 55.4%. The original clean pseudo-spectrum’s window response drives most of the P₀ change; independent control of noise-noise and signal-noise terms is shown later.',height=335)

    y=r.page('Combined variation: test both factors together','Full-grid results')
    y=r.picture(F/'combined_P0_grid.png',y,height=225)
    y=r.table(['Unused test','Truth R, original → band50','Agreement A, original → band50','Mean / worst width increase'],[
        ['Simulation (4)','1.746 → 1.169','0.157 → 0.050','24.7% / 59.1%'],
        ['Mask (5)','1.419 → 1.162','0.459 → 0.098','17.9% / 55.4%'],
        ['Joint (20)','1.660 → 1.162','0.342 → 0.097','22.6% / 74.4%'],
    ],y,[110,220,235,CW-565],size=10.5)
    y=r.text('The complete 100-cell grid includes discovery observations and is descriptive: its combined R is 1.620→1.150, A 0.444→0.092 and G=1.215. The table uses only previously unused skies/windows. Reductions in A on the full grid are larger because the original training baseline is unusually different from the other cases.',y,size=10.5)

    y=r.page('The unused observations support the selected tradeoff','Tradeoffs')
    y=r.picture(F/'held_tradeoffs.png',y,height=300)
    y=r.text('All 23 recipes on unused observations: blue MOPED, green Merlin, black original and orange selected full-band50. The vertical line is the 25% mean width cap. Selection was fixed on discovery skies before these results were examined. Lower combined truth/agreement cost is better. The width budget can prevent choosing the method with the best truth displacement alone.',y,size=12)

    y=r.page('Interaction variance falls; individual width costs differ','Tradeoffs')
    y=r.picture(F/'variance_width.png',y,height=300)
    y=r.text('Left: exact balanced full-grid decomposition of normalized median variance. Right: widths on unused observations. A 25% geometric mean cap is not a 25% per-parameter cap. The interaction fraction falls from 41.9% to 14.4%; absolute variation also falls, so the larger relative mask fraction is not increased mask variance.',y,size=12)

    y=r.page('Alternatives, secondary model and truth regions','Constraint checks')
    y=r.table(['Result','What it means'],[
        ['No winner under 10% width growth','None of the tested recipes meets both improvement conditions and the diagnostic screen inside that narrower budget.'],
        ['Sky-weighted MOPED, λ=1','Lower truth displacement on unused skies, but widths grow about 35-39% on the three arms. It exceeds the main 25% budget.'],
        ['Merlin lower cut ℓ≥580','Average width growth is 28.7% (sim), 22.5% (mask), 27.0% (joint). It does not qualify across all three arms.'],
        ['Bounded secondary, same selected recipe','Unused widths grow 27.0% (sim), 17.1% (mask), about 25.9% (joint), relative to the bounded original. Sim/joint exceed 25%. No secondary re-ranking was performed.'],
        ['Full-grid joint truth, physical θ density','Original unbounded includes truth at joint 68% in 1/100 cases and at 95% in 2/100. Selected band50 includes truth at joint 68% in all 100 cases, for both parameter transforms.'],
    ],y,[250,CW-250],size=11)
    y=r.text('The selected x<sub>c</sub> and β marginal 68% intervals still miss their truth throughout the grid; all nine marginal 95% intervals include truth. Joint high-density containment and marginal median recovery are different questions. This fixed-truth correlated grid is not a calibrated coverage experiment.',y)

    y=r.page('Reserved prior tests and the density-coordinate fix','Validation')
    y=r.picture(F/'prior_calibration.png',y,height=185)
    y=r.text('Screening uses even reserved rows; this final test uses 1,500 untouched odd rows. Joint 68%/95% containment is 64.1%/94.3% (original) and 66.4%/96.1% (band50, unbounded). These parameter/noise tests use the fixed training sky, not new catalogues.',y,size=11)
    y=r.code(src/'joint_truth.py',46,49,y,'Corrected physical-coordinate densities: joint_truth.py',size=9.5)
    y=r.text('Densities of samples and truth both include the parameter-transform Jacobian. The HPD rank is posterior mass at density above the truth; smaller rank means higher-density truth. Mean physical-coordinate log density at reserved truth falls from -3.013±0.136 to -5.481±0.097 (row SEM), indicating a fixed-sky information cost despite improved realization robustness.',y,size=11)

    y=r.page('Why random skies share a non-random-looking offset','Offset diagnosis')
    y=r.equation(r'\Delta_i=\epsilon_i-\epsilon_{100},\qquad \mathbb{E}[\Delta_i\mid\epsilon_{100}]=-\epsilon_{100}',y)
    y=r.text('ε<sub>i</sub> is the spectral fluctuation of independent lightcone i around an ensemble mean. Every comparison subtracts the same ε₁₀₀. Conditional on that reference, the differences share a common mean. For equal independent sky covariance, distinct Δ<sub>i</sub> also have unconditional cross-covariance equal to the reference covariance.',y)
    y=r.text('The original network can map the common fine-spectrum difference into the same parameter direction. The nine other simulations give P₀ medians <b>49.89±2.32</b> and β medians <b>13.01±0.21</b>, where ± is scatter of medians across skies, not posterior uncertainty. Seed 100 gives P₀≈12.15 and β≈9.70; truth is 18.1 and 4.35.',y)
    y=r.heading('Why several parameters shift together',y)
    y=r.equation(r'd\ln P_e=d\ln P_0+\left(0.3+\frac{\beta u}{1+u}\right)d\ln x_c-\ln(1+u)\,d\beta',y)
    y=r.text('At fixed mass, redshift and radius, this equation shows amplitude/shape compensation, with u=(r/R<sub>200c</sub>)/x<sub>c</sub>. The six exponents additionally change the halo populations contributing at each angular scale. A coordinated marginal-median shift is not a positive P₀-β correlation within a posterior, and the vector of marginal medians need not itself be a best-fit physical profile.',y)
    y=r.text('The other seeds’ mean α<sub>m,P₀</sub>≈0.688 and α<sub>m,β</sub>≈0.118 increase, while α<sub>m,xc</sub>≈-0.404, α<sub>z,P₀</sub>≈-2.061 and α<sub>z,xc</sub>≈-0.027 decrease. This is a coupled fit, not a uniform rescaling of all coordinates.',y,size=11)

    y=r.page('Controlled decomposition identifies the clean signal','Mechanism tests')
    y=r.equation(r'D_i-D_0=\delta D_{\rm clean}+\delta D_{yn}+\delta D_{nn}^{\rm corrected}',y)
    y=r.text('D₀ is the noisy baseline. δD<sub>yn</sub> is the corrected signal-noise change and δD<sub>nn</sub><sup>corrected</sup>=B²δD<sub>nn</sub>. For different simulations at the same mask the noise-noise term is identical and cancels exactly. For different masks, the saved B²δD<sub>nn</sub> is isolated separately.',y)
    y=r.equation(r'\delta D^{\rm smooth}=D_0^{\rm clean}\,G_\sigma\!\left(\delta D^{\rm clean}/D_0^{\rm clean}\right),\quad \delta D^{\rm fine}=\delta D^{\rm clean}-\delta D^{\rm smooth}',y)
    y=r.text('G<sub>σ</sub> is Gaussian smoothing in integer multipole with σ<sub>ℓ</sub>=5 or 50. The old controls smooth the <b>change from the training sky</b> and add either its smooth or fine part to D₀. Baseline fine structure is retained in the smooth-change control. Absolute-change smoothing was tested independently to exclude a ratio artifact.',y)
    y=r.code(E/'diagnosis/diagnose.py',143,145,y,'Executed smooth/fine controls: diagnose.py',size=9.5)
    y=r.text('These hybrid inputs need not be physical mock spectra. They isolate the estimator’s response while leaving the original whitening, PCA, asinh and learned model unchanged. The saved smooth controls add a smoothed change from other skies to the baseline; they do not smooth the baseline itself.',y,size=11)

    r.figure_page('The mechanism persists for simulation, mask and joint changes',F/'mechanism_summary.png',
        'Each column applies one mean change to the same baseline observation: other simulations, other masks, or the 81 joint-only cells (both factors changed). “Smooth” uses the σℓ=5 smooth clean change; “fine” its complement. Clean-only changes retain most of the upward P₀ response. Smooth changes return to the conditional baseline, while the fine residual keeps the shift. Dashed truth and dotted baseline are different references.',height=330)

    r.figure_page('Individual seeds reproduce the smoothing/fine split',D/'plots/fingerprint_per_seed.png',
        'All nine non-training skies individually show the same mechanism. Smooth-change controls give P₀ medians 12.13-15.46, while fine-only controls give 33.89-42.39. Averaging the nine actual observations still gives P₀≈51.81; reversing the fine change moves P₀ to about 4.67. A +0.742% pure clean-power amplitude change alone gives about 12.43. The response is directional and sensitive to fine structure, rather than a constant high-P₀ output.',height=345)

    y=r.page('What was ruled out, and what remains uncertain','Diagnostic checks')
    y=r.table(['Check','Verified result / limitation'],[
        ['Painting and preprocessing parity','Same M200c/h conversion, physical mass cut, B12 vector, all-redshift selection, 4R200c sphere, beam and fixed noise hashes. Band reconstruction error 8.9×10⁻¹⁶; original features exact for all ten fixed-mask seeds.'],
        ['BatchNorm / ensemble artifact','All modules are in evaluation mode. Changing evaluation cohort alters baseline log density by ≤1.91×10⁻⁶. All three ensemble members shift upward; no single bad network explains it.'],
        ['Bounded / unbounded coordinates','The mechanism survives both: bounded mean-new P₀≈52.16, smooth≈11.67, fine≈44.50. Extrapolation is not the sole cause.'],
        ['Global support / noise distance','New observations are near the 22nd-30th held-out nearest-neighbor percentiles in the broad training pool. Clean PCA noise distances are 21.6-39.7; these norms are not formal sigma significance. Conditional mismatch can occur inside global support.'],
        ['Local inverse / noise-only baseline draws','Prior-scaled derivatives have condition ≈85,000; full linear coordinate shifts are unreliable. The 21 old noise controls give P₀ medians 9.3-23.6, but 20 built whitening and are not independent calibration data.'],
    ],y,[215,CW-215],size=10.5)
    y=r.text('Stored contracts do not re-establish raw simulation provenance beyond those metadata checks. Fine-pattern conditional mismatch is strongly supported by the controlled response. Remaining baseline median offsets may include nonlinear degeneracy, prior volume and finite density-estimator error; their relative contributions are not uniquely separated. [4,5]',y,size=11)

    if draft:
        r.finish()
        return

    y=r.page('New Pearson comparison: the saved smooth-change input','Seed 100 / mask 12345')
    assert definition['target']=='relative_change'
    y=r.equation(r'D_0^{(\sigma)}=D_0^{\rm obs}+D_0^{\rm clean}G_\sigma\!\left[\frac{\bar D_{\rm other}^{\rm clean}-D_0^{\rm clean}}{D_0^{\rm clean}}\right]',y)
    y=r.text('This uses the <b>exact same saved “smooth” input</b> from the simulation column of the actual/clean/smooth/fine plot. D₀ is seed 100 / mask 12345; D̄<sub>other</sub><sup>clean</sup> is the mean of the other nine simulations at that same mask. The control adds only their smoothed fractional clean change to D₀. The original baseline fine pattern and corrected SO residual remain intact.',y)
    y=r.text('The plotted σ<sub>ℓ</sub>=5 control is primary; σ<sub>ℓ</sub>=50 is a secondary check. The Gaussian kernel is truncated at four σ with nearest-value extension at ℓ=80/7979. σ is a multipole standard deviation, not a band width or real-space beam. The baseline itself and its noisy spectrum are not smoothed; no model is retrained.',y)
    lines=(ROOT/'baseline_pearson.py').read_text().splitlines();lo=next(i+1 for i,s in enumerate(lines) if 'for sigma in [5,50]' in s)
    y=r.code(ROOT/'baseline_pearson.py',lo,lo+8,y,'New implementation: baseline_pearson.py',size=10)
    y=r.text('The baseline feature reconstruction agrees with its saved input to maximum error '+f"{definition['original_input_max_error']:.2g}"+'. The unbounded original baseline reuses the exact verified full-grid posterior draws. Each changed input uses 90,000 equal-mixture draws from the same three original trained networks.',y,size=11)

    r.figure_page('Original vs smooth-change control: posterior Pearson r',F/'pearson_linear_primary.png',
        'Primary unbounded model and the requested σℓ=5 control. Each matrix uses all 90,000 physical-parameter draws, without prior clipping. The main compensation directions persist. Pearson r describes dependence within each posterior, rather than correlations of parameter medians across simulations. The σℓ=50 check appears in the next panel and numerical table.',height=350)

    y=r.page('How much the posterior correlations change','Primary Pearson result')
    y=r.picture(F/'pearson_linear_delta.png',y,height=245)
    rr=np.asarray(pearson['correlation'])
    pairs=[(0,2,'P₀ - β'),(0,1,'P₀ - x<sub>c</sub>'),(1,2,'x<sub>c</sub> - β')]
    y=r.table(['Pair','Original','σℓ=5','σℓ=50'],[[n]+[f'{rr[k,a,b]:+.3f}' for k in range(3)] for a,b,n in pairs],y,[310,145,145,CW-600],size=11)
    y=r.text('Largest |Δr| is P₀-β: +0.061±0.0043 (σℓ=5), +0.021±0.0043 (σℓ=50). The ± is sampling MCSE, not training uncertainty. Correlations stay close to the original; pearson_pairs.csv contains all 36 pairs for both models and controls.',y,size=10.5)

    y=r.page('Joint clouds and marginal shifts help interpret r','Primary Pearson interpretation')
    y=r.picture(F/'pearson_joint_clouds.png',y,height=245)
    rows=[]
    for d,label in zip(pearson['cases'],['Original','σℓ=5','σℓ=50']):
        rows.append([label,f"{d['median'][0]:.2f}",f"{d['median'][1]:.3f}",f"{d['median'][2]:.2f}",f"{d['outside_any_prior_fraction']:.1%}"])
    y=r.table(['Input','P₀ median','xc median','β median','Any coordinate beyond prior'],rows,y,[180,135,135,135,CW-585],size=11)
    y=r.text('Clouds show thinned joint draws without clipping. Pearson r can change along a curved compensation direction, even when the posteriors remain similar. Marginal medians alone are not a joint best-fit point.',y,size=11)

    r.figure_page('Bounded posterior: the same smooth-change diagnostic',F/'pearson_logit_primary.png',
        'Secondary bounded model, transformed back to physical θ. P₀-β changes -0.352→-0.297 at σℓ=5 and to -0.330 at σℓ=50. The unbounded inside-prior-only check changes -0.361→-0.292 and -0.336. Truncating primary draws changes the distribution; it is a sensitivity test, not the primary comparison. Neither prior tails nor the logit transform alone explain the response.',height=350)

    y=r.page('Pearson precision, widths and diagnostic status','Numerical summary')
    rows=[]
    for space,d in [('Unbounded',pearson),('Bounded',bounded)]:
        original=np.asarray(d['cases'][0]['width68'])
        for i,label in [(1,'σℓ=5'),(2,'σℓ=50')]:
            changes=d['largest_changes'][i-1];p=changes[0]
            width=np.exp(np.mean(np.log(np.asarray(d['cases'][i]['width68'])/original)))
            rows.append([space+' / '+label, p['first']+' / '+p['second'],
                         f"{p['delta']:+.3f} ± {p['delta_mcse']:.3f}",f'{width:.3f}',
                         f"{d['cases'][i]['truth_hpd_theta']:.3f}"])
    y=r.table(['Model / control','Largest pair change','Δr ± MCSE','Mean width ratio','Truth HPD rank'],rows,y,[155,230,140,115,CW-640],size=10.5)
    y=r.equation(r'r_{ab}=\frac{\operatorname{Cov}(\theta_a,\theta_b)}{\sigma_{\theta_a}\sigma_{\theta_b}}',y,width=580)
    y=r.text('Cov is the covariance of physical posterior parameters a,b; σ<sub>θa</sub>, σ<sub>θb</sub> are their standard deviations. Correlation uses all 90,000 draws and is symmetric, with unit diagonal and |r|≤1.',y,size=11.5)
    lines=(ROOT/'baseline_pearson.py').read_text().splitlines()
    corr_line=next(i+1 for i,line in enumerate(lines) if 'correlation=np.corrcoef' in line)
    y=r.code(ROOT/'baseline_pearson.py',corr_line,corr_line,y,'Pearson calculation in physical θ: baseline_pearson.py',size=10)
    y=r.text('MCSE is batch-correlation SD/√90; original/control errors combine in quadrature. It measures sampling precision only. Widths use the original baseline reference; truth HPD uses 6,000 physical-density draws. These changed-input outputs are diagnostic, not calibrated constraints.',y,size=11.5)
    y=r.text('This is the saved smooth <b>mean-other-simulation change applied to the baseline</b>, preserving the original model and its conditional fine pattern. The retrained 50-multipole method is a separate experiment.',y,size=11.5)

    y=r.page('Recommendation under the requested information budget','What to use now')
    y=r.heading('Best completed tradeoff: retrained Merlin, full band50 range',y)
    y=r.text('Use the unbounded band50 ensemble as the main tested robustness option, keeping the bounded ensemble as a sensitivity result. Across unused simulation/mask/joint tests it satisfies the primary average width budget and improves both truth displacement and agreement. Report the largest individual width costs alongside the mean. Retaining ℓ=80-7979 minimizes scale removal among the tested winners; within-band information is nevertheless reduced.',y)
    y=r.heading('Keep three statements separate',y)
    y=r.text('<b>Agreement:</b> the selected model varies much less across observations. <b>Truth:</b> joint density regions recover B12 more consistently, while some marginal medians remain displaced. <b>Calibration:</b> reserved fixed-sky parameter/noise checks are encouraging but do not certify inference over new halo realizations. The offset diagnosis and Pearson controls explain sensitivity, rather than supplying a deployable truth-based subtraction.',y)
    y=r.heading('Further work with the greatest scientific value',y)
    y=r.text('Train pressure models with several catalogue and mask realizations, and hold out entire lightcones and windows before tuning any method. Preserve the trained noise/beam/cross-spectrum contract. Evaluate at multiple pressure points, so a nuisance model measured at B12 does not silently stand in for the whole prior. This is the direct test of the desired sky-marginalized posterior.',y)
    y=r.text('The smooth-change controls should remain diagnostic. The frozen model was not trained on these hybrid baseline-plus-other-sky inputs. A method intended for real observations must apply its transformation consistently during simulation/training and be validated on unused skies.',y)

    y=r.page('A MOPED that relies less on nuisance-sensitive scales','Future method, not yet validated')
    y=r.equation(r'C_{\rm eff}(\theta)=C_{\rm SO}(\theta)+S_{H,W}(\theta),\quad W_{\rm eff}=L_{\rm eff}^{-T}Q_{\rm eff}',y)
    y=r.text('S<sub>H,W</sub> describes halo/window variance, including fine multipole directions rather than only an ℓ cutoff. Whitening derivatives with C<sub>eff</sub> reduces the weight of unstable spectral features wherever they occur. This is the rationale of the already tested sky-weighted MOPED recipe, but a θ-dependent nuisance model and consistent retraining still need validation.',y)
    y=r.equation(r't_{\rm hard}=t_\theta-F_{\theta\eta}F_{\eta\eta}^{-1}t_\eta',y,width=580)
    y=r.text('An alternative is nuisance-hardened score compression: t<sub>θ</sub> is the pressure score, t<sub>η</sub> scores of explicit nuisance templates, and F blocks their local Fisher matrix. Subtracting the correlated nuisance score removes local nuisance sensitivity while preserving pressure directions as far as possible. This is a proposed extension inspired by [3], not an implemented result of this study.',y)
    y=r.text('Estimate nuisance directions on discovery skies only; validate their pressure dependence at multiple θ. Compare full bands, high-ℓ taper and nuisance projection using the same fixed-scale R/A metrics and width budget, then hold out whole lightcones. A small-scale cut is simpler but may discard both nuisance structure and real pressure information. Local Fisher retention alone does not ensure globally calibrated SBI. [2,3]',y)
    y=r.text('Ten skies at a single pressure point and overlapping masks constrain an interim model; they are insufficient for a precise global nuisance covariance. No training-scale claim or implementation beyond the tested recipes is implied here.',y,size=11)

    y=r.page('Reproducibility, execution and files changed','Delivery record')
    y=r.text('<b>Authoritative cluster root for this delivery:</b><br/><font face="Mono" size="10">/lustre/work/kristero10/tsz_joint_report_20261005</font><br/>The earlier study and diagnosis roots remain frozen. The full-grid study used 12 jobs; its mechanism follow-up used three. This request submits one 26-CPU Pearson job to an available idark queue; jobs.json records its node, timing and status. No repainting or model training is needed for the new comparison.',y)
    y=r.table(['Created / edited in the new report root','Purpose'],[
        ['baseline_pearson.py; pearson.pbs','New fixed-baseline input controls, physical-parameter Pearson matrices, Monte Carlo precision, bounded sensitivity, physical-coordinate truth checks and source/output hashes.'],
        ['collect_evidence.py; evidence_manifest.json','Read-only copy and identities of the prior result tables, selected source files and publication figures needed to rebuild the combined report.'],
        ['report_figures.py; figures/','Readable plotting code, full-grid design, new metric/width/validation plots and Pearson figures; every plot has PNG and vector PDF output.'],
        ['build_report.py; source_excerpts.json','Editable report builder, rendered equations and exact source excerpts with line numbers and source SHA256 identities.'],
        ['FILES_CREATED.md; README.md; VERIFIED.json','Complete delivery inventory, execution/reproduction steps, PDF page review and artifact verification record.'],
    ],y,[350,CW-350],size=10.5)
    y=r.text('Previous source fixes documented here include shared painting and per-seed count checks, feature parity, inward band cuts, stable MOPED QR, augmented clean subtraction for Merlin’s SO residual scale, and applying Jacobians to both sample/truth densities. The report’s excerpts are the executed files. Existing scientific outputs were not overwritten.',y,size=11)

    y=r.page('References and provenance anchors','References')
    references=[
        ('[1] Battaglia et al. (2012), pressure model and tSZ power','https://arxiv.org/abs/1109.3711'),
        ('[2] Heavens, Jimenez & Lahav (2000), MOPED / massive data compression','https://arxiv.org/abs/astro-ph/9911102'),
        ('[3] Alsing & Wandelt (2019), nuisance-hardened data compression','https://arxiv.org/abs/1903.01473'),
        ('[4] Cannon, Ward & Schmon (2022), SBI under model misspecification','https://arxiv.org/abs/2209.01845'),
        ('[5] Ward et al. (2022), robust neural posterior estimation','https://arxiv.org/abs/2210.06564'),
    ]
    for name,url in references:y=r.text(html.escape(name)+f'<br/><a href="{url}" color="{BLUE}">{url}</a>',y,size=11.5)
    y=r.heading('Numerical evidence and artifact identities',y)
    y=r.text('The numerical tables, figures and conclusions in this report derive from the completed HalfDome experiments, rather than the reference papers. Sources are archived in evidence/; new inference products carry PEARSON_COMPLETE.json. evidence_manifest.json and VERIFIED.json contain full SHA256 values, so the data, source excerpts and delivered PDF can be checked independently.',y,size=11)
    y=r.text('Prior full-grid completion SHA256:<br/><font face="Mono" size="9">1172e56d35b97d14382a9b06df12b9af212c6c52aeeab8cb943bd917a27d9a0f</font><br/>Prior diagnosis final completion SHA256:<br/><font face="Mono" size="9">57204a5751fa5578e6368c76b901a299fde370b36fd4c1a3d4c1d7924e7d0c73</font>',y,size=10)
    y=r.text('Raw halo catalogues were reused through saved spectra and stored painting contracts; they were not reread or rehashed for the new inference. The mechanism follow-up’s report-formatting failure was repaired after its inference finished, preserving the samples; FINAL_COMPLETE supersedes the earlier intermediate marker. This report distinguishes those repairs from scientific validation.',y,size=10.5)
    r.finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-draft',action='store_true',help='Layout-check only the completed earlier analyses.')
    main(draft=parser.parse_args().prior_draft)
