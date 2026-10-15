"""Render measured radius comparisons and an unrestricted methodology proposal."""
import json
from pathlib import Path
import numpy as np
from report_layout import Report,OUTPUT,W,H
from reportlab.lib.utils import ImageReader
from fit_study import accepted

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text())
def number(value):return '--' if value is None or not np.isfinite(value) else f'{value:.2f}'


def code(report,lines,y,size=9):
    c=report.c;c.setFont('Code',size)
    for line in lines.splitlines():
        c.drawString(38,y,line);y-=14
    return y-12


def figure(report,title,path,caption):
    y=report.start(title)
    report.figure(path,y)
    report.text(caption,72,size=9,leading=12,width=137)


def main():
    summary=read('products/study_summary.json');cases=summary['bins']
    plan=read('metadata_plan.json');usage=read('complete.json');checks=read('scientific_checks.json')
    halos=dict(np.load(HERE/'selected_halos.npz'))
    report=Report()
    y=report.start('FLAMINGO: 2, 3 and 4 R200c','Radius, satellites, disturbed systems, empirical scatter and a full-snapshot methodology')
    y=report.text('L1_m9 / HYDRO_FIDUCIAL / snapshot 0077 / z = 0. The same 4,375 centrals are measured once to 4R200c, then restricted to exact nested 2R, 3R and 4R spheres. Ten 0.2-dex bins from log10(M200c/Msun) = 12.5 to 14.5 contain 300 randomly selected centrals each; the five higher bins contain every available central.',y)
    y=report.text('Primary target: the native snapshot observables from all gas particles inside each sphere, including recently AGN-heated gas and satellites. Headers specify that Compton-Y and electron density are zero for star-forming particles. Membership controls separate neighboring halos. Geometric stacks include a correlated environment and can overlap; they are not automatically additive painting templates.',y)
    supported=[c for c in cases if c['strict_resolution_supported']]
    y=report.text(f'Strict common resolution support is present in {len(supported)} of the 15 mass bins. The remaining mean-stack fits are explicitly diagnostic. More low-mass halos reduce sample noise but cannot recover unresolved gas structure. No low-mass resolution convergence is claimed.',y)
    y=report.text('This report distinguishes the acquisition boundary, the fitted radial range, and the final painting boundary. A larger cutout retains more measured signal; it does not by itself make a single gNFW function a better physical description.',y)
    y=report.text('All figures use physical M200c in Msun, not Msun/h. Fits use shell averages and empirical diagonal weights. Profile error bars represent halo standard deviation. Parameter intervals from mean-stack bootstraps and individual-halo distributions have different meanings.',y)

    figure(report,'Full-volume gas and DMO overview',HERE/'figures/whole_snapshot_gas_dmo.png',
        'Reused verified overview: complete top-level cell-count projections, normalized by their own means. Pixels are coarse (31.25 cMpc); this is neither a resolved particle map nor a pressure map. No new whole-volume particle read was needed.')

    y=report.start('What the radius comparison shows','Only bins with common resolved support enter the aggregate ranking')
    ranking={}
    for field in ('pressure','electrons'):
        valid=[c for c in supported if all(e.get('accepted') and 'heldout' in e
            and accepted(e['heldout']['fit']) for e in c['selections']['all'][field])]
        if valid:
            inner=np.array([[e['heldout']['inner_2R_RMS_percent'] for e in c['selections']['all'][field]] for c in valid])
            full=np.array([[e['heldout']['all_4R_RMS_percent'] for e in c['selections']['all'][field]] for c in valid])
            wins=np.bincount(np.argmin(full,axis=1),minlength=3)
            ranking[field]={'supported_bins_with_holdout':len(valid),'median_heldout_inner_RMS_percent':np.nanmedian(inner,axis=0).tolist(),
                'median_heldout_4R_RMS_percent':np.nanmedian(full,axis=0).tolist(),'full_4R_wins':wins.tolist()}
            y=report.text(f'{field.capitalize()}: {len(valid)} supported bins have a halo holdout. Median held-out full-4R RMS for 2R / 3R / 4R fits is '+ ' / '.join(number(v)+'%' for v in np.nanmedian(full,axis=0))+'. The respective numbers of bin-wise wins are '+ ' / '.join(str(v) for v in wins)+'.',y)
            y=report.text('Median held-out inner-2R RMS: '+' / '.join(number(v)+'%' for v in np.nanmedian(inner,axis=0))+'. This quantifies the cost paid in the inner region when outer shells enter the fit.',y)
    y=report.text('Recommendation for data extraction: retain a 4R200c cutout. It contains all three experiments and allows an outer component to be measured rather than guessed. The ranking includes only bins where all three mean and training fits pass the optimizer / boundary / conditioning stability gate. The fit range must follow the observable and ownership target.',y)
    y=report.text('For additive HalfDome painting, do not paint every all-gas 4R stack on every halo: neighboring halos are already present in these stacks. Calibrate uniquely assigned host gas and a separate diffuse field. Neighboring painted hosts generate a two-halo contribution through clustering; painting that contribution again around each host would double count it. A smaller boundary cannot fully solve ownership.',y)
    y=report.text('The highest bins have only 21 and 6 halos. Ten views suppress orientation-dependent structure but do not create ten independent halo samples, remove satellites, or establish a smooth intrinsic parameter distribution.',y)
    (HERE/'products/recommendation.json').write_text(json.dumps(ranking,indent=2)+'\n')

    measured=read('products/interpretation.json');p=measured['pressure'];e=measured['electrons']
    y=report.start('Recommended cutout: 4R200c','Acquire the outer signal; distinguish a good outer fit from a good inner fit')
    y=report.text('For the measured all-gas target, 4R is the best acquisition boundary of the three tested. Among the ten stable, supported bins with a holdout, it wins the full-4R pressure RMS in nine bins and electron-density RMS in eight. This is evidence within one snapshot and one halo split, not a universal optimal radius.',y)
    y=report.text(f'Across supported bins, 2R / 3R retain median {100*p["median_retained_fraction_of_4R"][0]:.1f}% / {100*p["median_retained_fraction_of_4R"][1]:.1f}% of measured pressure signal inside 4R, and {100*e["median_retained_fraction_of_4R"][0]:.1f}% / {100*e["median_retained_fraction_of_4R"][1]:.1f}% of electrons. The 3-4R extension therefore matters more for tau than for tSZ. These are equal-bin summaries, not cosmic signal fractions.',y)
    y=report.text('Use the 4R fit when predicting the complete measured 4R radial target. Use the 2R fit, or a separate inner component, when inner-profile fidelity is the objective. Median inner pressure RMS is '+ ' / '.join(number(v)+'%' for v in p['median_training_population_inner_2R_RMS_percent'])+' for 2R / 3R / 4R fits. A single gNFW trades core-region accuracy for outer-shape accuracy.',y)
    y=report.text('The core-excluded common-4R pressure-map L2 medians are '+ ' / '.join(number(v)+'%' for v in p['median_core_excluded_4R_map_L2_by_fit_radius_percent'])+'. Here 3R and 4R are practically similar. In the 3-4R projected ring, the corresponding L2 values are '+ ' / '.join(number(v)+'%' for v in p['median_outer_ring_map_L2_by_fit_radius_percent']['3to4'])+'. Extension helps the outskirts, but substantial residuals favor a separate outer component.',y)
    y=report.text('For HalfDome, keep a 4R data product and calibrate an ownership-aware inner-plus-outer model before selecting a final painting truncation. A 3R boundary is a usable comparison convention, but the test gives no reason to discard the measured 3-4R electrons or to assume their signal can be assigned to every halo independently.',y)

    y=report.start('Satellites, neighbors and merger interpretation','Small cumulative fractions can have a large outer-profile effect')
    y=report.text(f'At 4R, median own-satellite and other-halo pressure fractions are {100*p["median_cumulative_satellite_fraction"][2]:.2f}% and {100*p["median_cumulative_other_halo_fraction"][2]:.2f}%. In the 3-4R shell alone, the median other-halo fraction rises to {100*p["median_3_to_4R_ring_component_fraction"][2]:.1f}%; about {100*p["median_3_to_4R_ring_component_fraction"][3]:.1f}% is unbound. Unbound gas includes real ejected and correlated gas and is not a clean-background label.',y)
    values=p['median_all_4R_RMS_by_target']
    y=report.text(f'Refitting after removing other-halo gas changes median full-4R pressure RMS from {values["all"]:.2f}% to {values["no_other"]:.2f}%. Removing own satellites as well gives {values["no_satellites_or_other"]:.2f}%. Neighbor ownership explains part of the mismatch; satellite removal provides no uniform additional benefit and removes genuine host-system signal.',y)
    y=report.text('Major-companion and centre-offset contrasts do not show a universal increase of native fit RMS in every bin. A merger can change the profile mean and scatter without making its conditional mean harder to fit. The groups differ in finite sample size and mass distribution, and the broad offset flag is not a merger history. Do not infer that removing flagged systems physically improves the population calibration.',y)
    y=report.text(f'Ten-view averaging lowers median whole-4R pressure-map L2 from {p["median_native_4R_map_L2_original_z_percent"]:.2f}% to {p["median_native_4R_map_L2_ten_views_percent"]:.2f}%. It reduces angular residuals while leaving the radial stack and integrated signal unchanged. It neither removes merger-heated gas nor increases the number of independent halos.',y)

    figure(report,'Common-region and held-out RMS',HERE/'figures/overview_RMS.png',
        'Gray mass intervals are diagnostic only. All three fits are evaluated on the same region in each panel. Held-out points use a fixed 20% halo split; the six-halo bin has no holdout.')
    figure(report,'Signal retention and bound neighbors',HERE/'figures/overview_signal_contamination.png',
        'Integrated quantities use equal normalized-halo weights. Satellite signal belongs physically to the host system; other-halo signal is an ownership concern for additive painting. Unbound gas can still be correlated environment.')
    figure(report,'How membership removal changes profile fit RMS',HERE/'figures/overview_membership_RMS.png',
        'Within each panel the radial domain is fixed, but the physical gas target changes between curves. Removing a component can improve the gNFW approximation while removing real observable signal.')
    figure(report,'Other-halo gas and coefficient changes',HERE/'figures/overview_no_other_parameter_shifts.png',
        'Percent changes relative to all gas use paired whole-halo bootstrap draws. Bars span the 16th-84th percentiles of the change. These are empirical finite-sample sensitivities, not covariance-aware significance intervals.')
    figure(report,'3D fits, projected refits and orientation averaging',HERE/'figures/overview_projection_accuracy.png',
        'Map L2 and integrated bias include the whole image and its unresolved core. Projected refits use b >= 0.25R plus the common shell mask. Orientation averaging can improve spherical agreement without changing the 3D radial stack.')
    figure(report,'Common outer-image comparison',HERE/'figures/overview_resolved_projection_accuracy.png',
        'Each fit is projected to the same 4R sphere. Pixels below the larger of 0.25R and the first fitted shell boundary plus half a pixel diagonal are excluded. This avoids bright unresolved cores dominating the comparison; it is not an SPH or resolution-convergence test.')
    figure(report,'Outer rings retain substantial model residuals',HERE/'figures/overview_outer_map_accuracy.png',
        'All models are projected through the same 4R sphere. Each ring has its own relative L2 denominator, so bright inner pixels cannot hide outer mismatch. Radius extension improves the approximation without guaranteeing an adequate single gNFW outer component.')
    figure(report,'Major companions, offsets and fit mismatch',HERE/'figures/overview_morphology_RMS.png',
        'Groups use current bound satellite mass ratios and bound-central COM offsets. Curves describe different halo subsets and are not a causal estimate of merger effects. Group fits require at least six halos; low-mass fits remain diagnostic.')

    y=report.start('Sample and numerical support','Uniform random lower-mass samples; all higher-mass centrals')
    rows=[]
    for b,c in enumerate(cases):
        r=np.flatnonzero(halos['mass_bin']==b)
        rows.append([f'{c["mass_edges"][0]:.1f}-{c["mass_edges"][1]:.1f}',c['halos'],c['population'],
            int(np.median(halos['ngas_200_crit'][r])),c['disturbed_count'],c['strict_shell_count_to_2R'],
            'resolved' if c['strict_resolution_supported'] else 'diagnostic'])
    y=report.table(['log10 M','N','population','median Ngas','disturbed','shells <2R','status'],rows,y,[95,65,100,115,90,105,150],size=9,height=23)
    report.text('Ngas is the SOAP spherical R200c count. SOAP general-property zeros below its catalog threshold are not evidence for an empty gas halo. Selection uses central status and M200c, not gas detectability.',y,size=9,width=135)
    figure(report,'Sample, gas counts and morphology proxies',HERE/'figures/sample_resolution_overview.png',
        'The lowest selected bin has median 55 gas particles inside R200c. The combined disturbance flag is dominated by the bound-central COM offset; separate major-companion contrasts avoid interpreting it as a merger rate.')

    y=report.start('Measured fields and normalization','Stored particle quantities and header conversions define the measurements')
    y=report.equation(r'$\overline{P}_{e,j}=\frac{m_e c^2}{\sigma_T V_j}\sum_{i\in j}Y_i,\qquad V_j=\frac{4\pi}{3}(r_{j+1}^3-r_j^3)$',y)
    y=report.text('Y_i is a particle Compton-Y area, converted to square metres; V_j is the physical shell volume in cubic metres. m_e is the electron mass, c the speed of light and sigma_T the Thomson cross-section. Summing particle Y gives the volume-integrated electron pressure without assuming a temperature or an ionization fraction.',y)
    y=report.equation(r'$\overline{n}_{e,j}=\frac{1}{V_j}\sum_{i\in j}n_{e,i}\frac{m_i}{\rho_i},\qquad n_{200,H}=\frac{200\rho_c f_b X_H}{m_p}$',y)
    y=report.text('n_e,i is the stored cooling-table free-electron density, with star-forming particles set to zero; m_i/rho_i is the particle volume, rho_c the critical density and m_p the proton mass. X_H is mass weighted over the common 4R gas sample and fixed across nested comparisons. It enters the hydrogen reference once and must not multiply a stored electron count again.',y)
    y=report.text('All field conversions use the snapshot CGS conversion, a exponent and h exponent. At z=0 a=1; retaining the full header conversion prevents accidental redshift or h errors when this test is extended.',y)
    y=report.text(f'Here f_b = {plan["baryon_fraction"]:.9f}. SWIFT header Omega_m excludes massive neutrinos, so the code uses Omega_b/(Omega_m + Omega_nu_0), equivalent to Omega_b divided by total matter density. Stored particle electron abundances remain unchanged by this reference normalization.',y)

    y=report.start('B12 / B16 functions and finite projections','Three shape parameters are fitted independently in each mass bin')
    y=report.equation(r'$\frac{P_e(x)}{P_{200}}=0.5176\,P_0 (x/x_c)^{-0.3}(1+x/x_c)^{-\beta},\quad P_{200}=\frac{G M_{200c}\,200\rho_c f_b}{2R_{200c}}$',y)
    y=report.text('x = r/R200c; M200c is the mass inside R200c, whose mean enclosed density is 200rho_c; G is the gravitational constant. P0 is the B12 amplitude, xc its scale radius and beta its outer-shape parameter. The transition exponent is fixed to 1 and inner slope to -0.3. The model factor 0.5176 converts thermal to electron pressure for the fixed XGPaint composition; measured Y is not multiplied again.',y)
    y=report.equation(r'$\frac{\rho_\tau(x)}{f_b\rho_c}=\rho_0 q^{-0.2}(1+q^{\alpha})^{-(\beta-0.2)/\alpha},\quad q=x/0.5,\quad \rho_\tau=\mu_e n_e$',y)
    y=report.text('rho0 is the B16 amplitude, alpha the transition exponent and beta the outer slope. The scale radius is fixed to 0.5R200c. mu_e = [m_e + 2m_p/(1+0.76)]/0.9 retains the XGPaint electron-to-mass and ionized-fraction convention. The fitted amplitude absorbs this conversion. The plotted density equals 200 X_H (mu_e/m_p) times n_e/n200,H; it differs from actual gas density, whose shape fit is auxiliary.',y)
    y=report.equation(r'$S_R(b)=2\int_0^{\sqrt{R^2-b^2}} f(\sqrt{b^2+z^2})\,dz\quad (b<R),\qquad S_R(b)=0\quad (b\geq R)$',y)
    report.text('R is the dimensionless sphere boundary (2, 3 or 4); b and z are projected radius and line-of-sight distance in R200c units. f is normalized pressure or density. S_R = y/y200 for pressure, where y200 = sigma_T P200 R200c/(m_e c^2), with R200c in metres. For density, S_R = Sigma/(f_b rho_c R200c), where Sigma is the line-of-sight surface density. Models average this integral over the measured square pixels.',y,size=9,width=135)

    y=report.start('Observable contracts for HalfDome painting','Pressure, free-electron density and electron momentum have different roles')
    y=report.equation(r'$y(b)=\frac{\sigma_T}{m_e c^2}\int P_e(b,l)\,dl,\qquad \tau(b)=\sigma_T\int n_e(b,l)\,dl$',y)
    y=report.text('y is the dimensionless thermal Compton parameter and tau the Thomson optical depth. b is projected physical separation, l the physical line-of-sight distance and the integration runs through the chosen finite sphere. The pressure fit calibrates y; the electron-density fit calibrates tau.',y)
    y=report.equation(r'$\frac{\Delta T_{\rm kSZ}(b)}{T_{\rm CMB}}=-\frac{\sigma_T}{c}\int n_e(b,l)\,v_{\parallel}(b,l)\,dl$',y)
    y=report.text('Delta T_kSZ is the kinetic SZ temperature perturbation, T_CMB the cosmic microwave background temperature, and v_parallel the physical peculiar velocity along the line of sight (positive away from the observer). kSZ depends on electron momentum, not just electron density.',y)
    y=report.text('This test does not read gas velocities or validate kSZ momentum. Painting a tau template times one halo bulk velocity assumes that relevant electrons share that velocity. Merger motions, satellite orbits and diffuse flows can violate this assumption. A production full-snapshot analysis should also accumulate and validate electron-weighted momentum, velocity bias and its correlation with the density / pressure scatter.',y)

    y=report.start('Fit objective, resolution and fair comparisons','A successful optimizer is not a physical validation')
    y=report.equation(r'$\chi^2_{\rm emp}=\sum_{j\in\mathcal{F}}\left[\frac{\ln f_j(\theta)-\ln\overline{d}_j}{\max(s_j/\overline{d}_j,0.05)}\right]^2,\quad {\rm RMS}=100\sqrt{\left\langle(f_j/\overline{d}_j-1)^2\right\rangle}$',y)
    y=report.text('dbar_j is the equally weighted mean of individually normalized halo shell profiles; s_j is their standard deviation. f_j is the volume-averaged model over that shell. F is the radial fitting mask and theta the three positive fitted parameters. The 5% floor limits domination by very small empirical scatter. This objective is an empirical calibration, not a covariance-aware likelihood.',y)
    y=report.text('A halo shell passes when its inner edge exceeds max(0.04R200c, 4 times the physical softening), contains at least 20 gas particles and starts beyond twice its smoothing-length median. The median uses a conservative upper histogram edge. Strict stack fits require every selected halo to pass and at least six common shells inside 2R.',y)
    y=report.text('If that strict requirement fails, diagnostic fits use the common geometric floor, at least 200 pooled particles per shell and a non-empty shell in at least half the halos. These weaker cuts keep the population fixed but do not establish resolution. The same base mask is used for 2R, 3R and 4R fits within a bin.',y)
    y=report.text('Native-domain RMS changes the tested region and cannot alone rank cutouts. We therefore also report common-inner RMS, full-4R RMS including extrapolation, outer-ring map errors and a fixed 20% halo holdout where N >= 20. The holdout is not spatially independent. Whole-image map L2 includes the unresolved core and is a projection diagnostic, not a resolved-profile goodness-of-fit statistic.',y)

    y=report.start('What membership and disturbed controls mean','Four exclusive components; no silent fallback to all gas')
    y=report.text('SOAP is the Spherical Overdensity and Aperture Processor: it measures halo aperture properties. HBT provides tracked bound subhalo identities and host hierarchy. Full radial profiles here come from snapshot particles, not from fitting the single SOAP R200c aperture.',y)
    y=report.text('Central bound: GroupNr_bound equals the selected central original halo index. Own satellites: the bound ID belongs to a subhalo whose SOAP top-level HostHaloIndex points to that central. Other halos: all remaining non-negative bound IDs. Unbound: GroupNr_bound = -1. Their shell sums and maps add exactly to all gas.',y)
    y=report.text('Membership lives in separate SOAP-HBT/membership_0077 chunk files, in original snapshot particle order. The IDs equal InputHalos/HaloCatalogueIndex, not the SOAP row number. HostHaloIndex uses SOAP rows. The code maps both conventions explicitly, checks chunk row counts, and selects satellites globally before applying spatial boundaries.',y)
    y=report.text('Controls retain the same geometry: all gas; all minus other-halo gas; central plus unbound gas; and host-bound gas (central plus own satellites). Host-bound gas is a distinct target because it excludes diffuse gas. Control fits reuse the all-gas mask for paired comparability; they do not independently certify component resolution.',y)
    y=report.text('Disturbed proxy: the largest own-satellite bound-mass ratio inside 2R is at least 0.25, or the bound central centre-of-mass offset exceeds 0.07R. The bound centre of mass includes the whole bound central, not just its R200c aperture; the broad combined flag is dominated by offsets. Major-companion-only and offset-only contrasts are therefore also saved. These are snapshot labels, not measured merger histories.',y)
    y=report.text('Merger-heated gas can already be central bound or unbound. Membership identifies present ownership, not heating origin; removing satellite-bound gas does not remove a merger shock. The group contrasts are descriptive associations at fixed mass-bin width.',y)
    y=report.text('The ten projections use uniform viewing directions and balanced image rolls. Rotation changes projected substructure, but preserves every integrated field and the 3D radial profile. Averaging orientations suppresses angular features; it cannot erase their radial pressure or create independent halos.',y)

    figure(report,'Pressure parameter uncertainty and halo scatter',HERE/'figures/overview_pressure_parameters.png',
        'Top: paired halo-bootstrap mean-fit intervals. Bottom: 16-84% spread of accepted individual fits. Missing low-mass points indicate unsupported individual fits. Different radius fits use the same halos and bootstrap draws.')
    figure(report,'Electron parameter uncertainty and halo scatter',HERE/'figures/overview_electrons_parameters.png',
        'Individual parameter distributions retain their covariance and pressure-density pairing. Accepted samples exclude unresolved or unstable fits, so their distribution can be selection biased; it is not deconvolved intrinsic scatter.')

    y=report.start('Bootstrap, scatter and stochastic painting','Resampling a mean and sampling a population answer different questions')
    y=report.text('For each mass bin, draw N halo indices with replacement and refit the mean stack; repeat 128 times. The same index draws are used for both physical fields and all three fit radii. Percentiles describe mean-calibration sensitivity to the finite halo sample. Paired radius changes retain this correlation.',y)
    y=report.text('Only stable bootstrap refits enter coefficient intervals; retained counts are recorded. These intervals are conditional on acceptance. A high failure fraction can bias or invalidate them and must not be hidden by plotting only the surviving percentile bars.',y)
    y=report.text('Profile standard deviation and the halo 16th-84th percentiles describe variation among measured halos. Separately, each halo is fitted on its own resolved shells with equal log-shell weights. Coefficients are accepted only if the optimizer succeeds, the solution avoids a boundary and the log-parameter Jacobian condition number is below one million. Failures and unsupported fits remain recorded.',y)
    y=report.text('For a preliminary painting experiment, draw a complete accepted six-parameter pressure-density tuple from a single halo, retaining its identity. Independent marginal draws can generate unrealistic shapes. The average of nonlinear profiles is not the profile at average coefficients: check and correct the ensemble mean before painting. Do not use the mean-bootstrap width as halo scatter.',y)
    y=report.text('The individual-fit sample omits unresolved halos and may bias the distribution toward smoother, gas-rich systems. Low-mass diagnostic stacks and the six-halo top bin cannot establish robust population distributions. Resampling halos also ignores correlations from overlapping spheres, large-scale modes and the common simulation volume. A hierarchical residual model and spatial-block validation are needed for a production stochastic painter.',y)

    # Every bin has its own profile page. Detailed maps for every bin and radius
    # remain standalone artifacts; representative maps keep this report usable.
    for case in cases:
        figure(report,'Profiles: '+case['label'].replace('bin_','').replace('p','.'),
            HERE/'figures'/(case['label']+'_profiles.png'),
            'Error bars: halo SD; bands: halo 16-84%. Solid model segments are fitted shells; dashed segments are extrapolations. The hatched core is excluded from fitting. Membership fractions include genuine satellite and diffuse signal.')
        y=report.start('Fit diagnostics: '+case['label'].replace('bin_','').replace('p','.'),case['fit_status'])
        rows=[]
        for field in ('pressure','electrons'):
            for entry in case['selections']['all'][field]:
                fit=entry['fit']
                if fit is None:
                    rows.append([field,entry['radius_R200c'],'unsupported','--','--','--','--']);continue
                rows.append([field,entry['radius_R200c'],', '.join(f'{v:.3g}' for v in fit['parameters']),
                    number(fit['rms_percent'])+(' *' if not entry.get('accepted') else ''),number(entry['common_inner_RMS_percent']),number(entry['all_4R_RMS_percent']),
                    number(entry.get('heldout',{}).get('all_4R_RMS_percent'))])
        y=report.table(['field','R','coefficients','native RMS','inner RMS','4R RMS','test 4R RMS'],rows,y,[85,30,225,95,95,95,145],size=8.5,height=27)
        y=report.text('Coefficient order: pressure (P0, xc, beta); electrons (rho0, alpha, beta). RMS values are percentages. Native RMS covers the fitted region; 4R RMS includes extrapolation. An asterisk marks a mean fit that fails the optimizer / bound / conditioning stability gate.',y,size=9,width=135)
        rows=[]
        for field in ('pressure','electrons'):
            f=case['integrated_fractions'][field]
            for ri in range(3):
                rows.append([field,ri+2,number(100*f['retained_fraction_of_4R'][ri]),
                    number(100*f['component_fraction'][ri][1]),number(100*f['component_fraction'][ri][2]),
                    case['selections']['all'][field][ri].get('individual_fit_accepted',0)])
        y=report.table(['field','R','signal / 4R (%)','satellite (%)','other halos (%)','accepted halo fits'],rows,y,[85,30,145,145,145,220],size=9,height=23)
        report.text(f'Disturbed proxy: {case["disturbed_count"]}/{case["halos"]}; major companion: {case["groups"]["major_companion"]["count"]}/{case["halos"]}. Fractions use equal normalized-halo weighting, not tSZ power-spectrum weighting. Full bootstrap intervals, component fits, projected refits and correlated samples are in the machine-readable products.',y,size=9,width=135)
    for b in (0,7,10,13,14):
        case=cases[b]
        figure(report,'Membership and morphology: '+case['label'],HERE/'figures'/(case['label']+'_contamination.png'),
            'Removal controls change the physical target. The disturbed contrast uses a snapshot morphology proxy, not a historical merger classification. All components and fit radii are available for every bin in the figure directory.')
        for ri in range(3):
            path=HERE/'figures'/(case['label']+f'_all_pressure_R{ri+2}_maps.png')
            if path.exists():figure(report,'Finite-sphere pressure: '+case['label']+f' | {ri+2}R',path,
                'Measured ten-view stack, native 3D fit projected through the same sphere, and measured minus model. The first two panels share a logarithmic scale; the signed residual uses its full symmetric range. Coefficients appear above the panels.')
        path=HERE/'figures'/(case['label']+'_all_gas_R4_maps.png')
        if path.exists():figure(report,'Gas mass projection: '+case['label'],path,
            'Gas mass is an auxiliary density target, distinct from electron density. Particle-centre maps are conserved pixel histograms, not SPH-kernel reconstructions. Large smoothing lengths limit the physical meaning of small-scale low-mass map features.')

    y=report.start('Best full-snapshot methodology: define the target first','Proposal independent of the execution budget of this test')
    y=report.text('1. Calibrate the observable that HalfDome will paint: electron pressure for tSZ, free-electron density for tau/kSZ, and actual gas density only as an auxiliary field. Preserve the header unit contract, physical mass definition, hydrogen accounting and the finite spherical projection in every route. Include recently heated gas in total electron pressure; make a heating-excluded sensitivity product rather than removing physical signal by default.',y)
    y=report.text('Keep the native zero-for-star-forming convention explicit. A separate ISM sensitivity model would require physical ionization and temperature assumptions: inserting a subgrid effective equation-of-state temperature directly can create artificial tSZ. This test does not reconstruct that unresolved component.',y)
    y=report.text('2. Acquire at least 4R around selected centrals and a larger context field for the correlated environment. Keep an explicit particle-level ownership rule. For an additive painter, assign bound satellite gas to its host once, assign other-halo gas to its own host, and model diffuse gas with a conservation-consistent prescription. Retain an all-gas geometric stack as a separate observational target.',y)
    y=report.text('3. Use a cell- or spatial-indexed pass that streams each snapshot chunk once and updates all intersecting halos. Accumulate physical shell sums, component sums, SPH-kernel projections and context/environment statistics together. Save compact per-halo sufficient statistics, particle-ID audit samples and a versioned manifest; avoid independent overlapping cutout reads for the production pass.',y)
    y=report.text('4. Cover mass and redshift with all useful centrals. Use L1_m8 / zoom studies for low-mass convergence and the larger L2p8_m9 volume for rare clusters. State whether feedback parameters change with resolution. Count emitting particles and effective weighted sample size separately from total gas. Gate shells per halo; publish completeness in mass, radius, gas fraction and disturbance.',y)

    y=report.start('Best methodology: mergers, environment and model shape','Separate physical diversity from a poor functional form')
    y=report.text('5. Use HBT merger trees to identify recent events over a stated fraction of the dynamical time, with progenitor peak masses and time since merger. Keep mass-ratio thresholds, central/satellite transitions and disrupted progenitors explicit. Compare these labels against centre offsets, substructure fraction, accretion rate, concentration and a tidal / local-density environment measure.',y)
    y=report.text('6. Fit a resolved inner host component and a physically distinct outer / correlated component when one gNFW function fails. Test free inner slope, transition shape and a broken mass scaling, motivated by Lee et al. (2022), against held-out data. Do not force fixed B12/B16 shape constants to absorb merger shocks, neighboring halos or an incorrect pressure normalization.',y)
    y=report.text('7. Fit mass, redshift and concentration dependence jointly, with continuous functions and hierarchical halo scatter. Keep 0.2-dex stacks as diagnostics; rare upper bins need wider descriptive bins or hierarchical pooling for population scatter. Lee et al. use h^-1 Msun, so convert their edges before comparing with the physical-Msun bins here.',y)
    y=report.text('8. Keep three-dimensional and projected calibrations distinct. Model the exact spherical truncation, pixel response, SPH smoothing, line-of-sight integration and intended observational beam. Validate several cameras plus random orientations; average views to estimate orientation sensitivity, while keeping the halo count unchanged.',y)

    y=report.start('Best methodology: statistics and painting validation','Population scatter requires more than marginal coefficient error bars')
    y=report.text('9. Estimate radial correlations from whole-halo resampling and spatial blocks, with shrinkage or a hierarchical likelihood when the covariance is poorly determined. Use spatially disjoint train/test regions and independent simulations or volumes to assess cosmic variance. Do not blindly apply a Hartlap correction to a shrinkage estimate or a covariance with too few independent samples.',y)
    y=report.text('10. Fit a joint distribution of pressure and electron profiles, conditioned on mass, redshift, concentration, disturbance and environment. Distinguish finite-resolution noise, uncertain per-halo coefficients, intrinsic physical variation and model inadequacy. Use a compact correlated log-profile residual basis when a six-parameter gNFW distribution cannot represent the diversity.',y)
    y=report.text('11. Preserve integrated electron number and thermal signal while sampling stochastic profiles; impose positivity and physically motivated limits. Draw a whole correlated profile or parameter vector. Model asymmetric and satellite components separately if those are needed in the maps; a random spherical radial profile cannot restore angular merger morphology.',y)
    y=report.text('12. Paint held-out spatial volumes, then compare directly with the particle tSZ / tau fields: radial and map residuals, integrated Y and electron counts, one- and two-point statistics, cross-correlations with halos / lensing, overlap regions and mass/redshift dependence. Choose the fitting boundary and model complexity by these observable-level results, rather than native fit RMS alone.',y)

    y=report.start('Full-snapshot implementation and expected scale','A production design should exploit the shared particle field')
    full=read('../mass12p5_15p5_r4_preflight_L1_m9_0077_20261008/resource_preflight.json')['alternatives']['all_centrals_12p5_15p5']
    y=report.text(f'The preceding metadata preflight counted {full["halos"]:,} centrals from 12.5 to 15.5 and {full["candidate_particle_rows"]:,} gas rows in their union of 4R candidate cells: {100*full["fraction_snapshot_gas_rows"]:.3f}% of snapshot gas. At 112 requested bytes per particle including membership, one logical pass is approximately {112*full["candidate_particle_rows"]/1e9:.1f} GB before kernel padding. This is a metadata estimate, not measured physical I/O.',y)
    y=report.text(f'A naive cell-to-every-intersecting-halo distance loop requires about {full["candidate_particle_halo_comparisons"]/1e9:.1f} billion candidate comparisons. Use a periodic spatial join, sorted particle / halo tiles and suitable neighbor searches. Benchmark spatial indexing and field accumulation independently; simply multiplying workers on a shared NFS volume can saturate storage before CPU.',y)
    y=report.text('Parallelize disjoint snapshot chunks or tiles, write additive component accumulators and merge deterministically. Use a node-local read cache if policy permits; limit simultaneous readers per storage server. Adaptive radial grids and kernel-integrated shell / map weights should be tied to physical resolution rather than fixed pixel counts alone. Read enough context to include kernels intersecting each boundary.',y)
    y=report.text('Keep per-halo radial profiles, selection labels, physical integrals and compact projection diagnostics as durable products. Accumulate stack maps directly. Full voxel cubes or ten maps for every halo can cost hundreds of GiB and are unnecessary for most calibration fits; store them only for a representative morphology set or stream their sufficient statistics. This is an algorithmic design choice, even with ample resources.',y)

    y=report.start('Verification and limits','Extraction completeness, numerical gates and physical validation are separate')
    y=report.text(f'The independent swiftsimio route checks {checks["SWIFT_checks"]["checked_halos"]} halos: three representatives from every mass bin. Particle-ID hashes and shell counts agree exactly. The largest relative Y-shell difference is {checks["SWIFT_checks"]["max_relative_shell_difference"]["y"]:.3g}. Matching SWIFT and full-snapshot fits are expected because both loaders select the same particles and apply the same physical estimator.',y)
    y=report.text('All four membership components reproduce all-gas profiles exactly; all 1,375 earlier high-mass shell sums are reproduced after recombining the new shell boundary at 2R. Original and ten-view map integrals agree. Analytic synthetic-profile recovery and projection-volume quadrature checks pass. These numerical gates do not establish resolution convergence or the adequacy of the chosen profile family.',y)
    y=report.text('Membership assignment relies on the documented original-particle-order and original-halo-index contract, with chunk length checks. The membership file does not provide particle IDs for a second independent membership join. SOAP aperture differences and zero-valued catalog entries are reported in scientific_checks.json rather than hidden.',y)
    y=report.text('Gas maps now use Sigma_gas/(f_b rho_c R200c). Earlier maps normalized by f_b M200c/R200c^2; multiply those older dimensionless values by 800pi/3 to obtain this normalization. The gate verifies this conversion. Physical gas masses and radial profiles are unchanged.',y)
    y=report.text('Limitations: one snapshot and simulation model; diagonal empirical fitting; overlapping environments; one halo holdout split; unresolved lower masses; sparse upper bins; snapshot disturbance proxies; unmeasured physical NFS traffic; particle-centre maps. Low-mass published Battaglia curves are extrapolations. This study cannot determine redshift exponents or validate the complete mass-redshift parametrization.',y)
    figure(report,'Independent matched swiftsimio profiles',HERE/'figures/SWIFT_matched_comparison.png',
        'Top: three matched direct/SWIFT cutouts compared with the full 14.5-14.7 population stack. Bottom: matched loader differences across every mass bin. Identical particles imply agreeing fit parameters; a small cutout subset can differ from the full population.')

    y=report.start('Resources and reproducibility','The unrestricted methodology proposal does not imply an unrestricted run')
    prior=sum(a['resources'].get('sampled_cpu_seconds',0) for a in [json.loads(p.read_text()) for p in (HERE/'attempts').glob('*/failed.json')])
    y=report.text(f'Completed extraction: {usage["wall_seconds"]/60:.2f} minutes, {(usage["main_cpu_seconds"]+usage["children_cpu_seconds"])/3600:.3f} CPU-hours, peak process-tree RSS {usage["peak_tree_rss_bytes"]/1024**3:.2f} GiB. Eight single-thread workers, nice priority 10, a 40-GiB cap and a 180-minute stop. Zero PBS jobs; data remained on STRW/maas. The failed first attempt is retained and cost at least {prior/60:.2f} sampled CPU-minutes.',y)
    y=report.text(f'The metadata plan contains {plan["logical_bytes_one_direct_pass"]/1e9:.2f} GB of requested particle/membership field slices plus at most {plan["swift_logical_bytes_upper_bound"]/1e9:.2f} GB for SWIFT checks. These are logical uncompressed field bytes, not measured disk or network traffic. Compression chunk rereads, cache behavior and the failed attempt change physical I/O.',y)
    y=report.text(f'Local fitting: {summary["wall_seconds"]/60:.2f} minutes, {summary["CPU_seconds"]/3600:.3f} CPU-hours, peak RSS {summary["peak_RSS_MiB"]/1024:.2f} GiB. Host availability is a point-in-time shared-resource measurement; metadata and launch receipts record the state. No full snapshot was downloaded.',y)
    capacity=read('resources_during.json')
    y=report.text(f'STRW/maas has {capacity["physical_CPU_cores"]} physical cores, {capacity["logical_CPUs"]} logical CPUs and {capacity["RAM_total_GiB"]:.1f} GiB RAM. Before launch it reported {plan["resources"]["available_memory_GiB"]:.1f} GiB available and a one-minute load of {plan["resources"]["load"][0]:.1f}. These are shared resources, not reserved allocations. Execution stayed beside the STRW snapshot to avoid transferring it to idark.',y,size=10,width=118)
    filesystems=list(capacity['shared_filesystems'].values())
    y=report.text(f'The during-run filesystem snapshot reported {filesystems[0]["free_bytes"]/1e12:.2f} TB free on the home filesystem and {filesystems[1]["free_bytes"]/1e12:.2f} TB free on the simulation filesystem. These are shared free-space figures, not a per-user disk quota. See resources_during.json and resources_after.json for timestamps and raw-input file-stat provenance.',y,size=10,width=118)
    y=report.text('Every new source and output is isolated in radii234_contamination_L1_m9_0077_20261008. Older studies and fullsnap_calibration production sources are preserved. Source pins, input SHA256 verification, attempt receipts, completion markers, checks and the final file inventory make the result inspectable.',y)

    y=report.start('Extraction example: units and membership','Illustrative code; executable loaders in this study enforce the full contract')
    y=code(report,"""import h5py
BASE = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL"
snapshot_chunk = BASE + "/snapshots/flamingo_0077/flamingo_0077.0.hdf5"
membership_chunk = BASE + "/SOAP-HBT/membership_0077/membership_0077.0.hdf5"
with h5py.File(snapshot_chunk, "r") as f:
    ds = f["PartType0/ComptonYParameters"]
    attrs = ds.attrs
    to_cgs = float(attrs["Conversion factor to CGS (not including cosmological corrections)"])
    a_exp = float(attrs["a-scale exponent"])
    h_exp = float(attrs["h-scale exponent"])
    Y_m2 = ds[start:stop] * to_cgs * a**a_exp * h**h_exp * 1e-4
with h5py.File(membership_chunk, "r") as f:
    bound_id = f["PartType0/GroupNr_bound"][start:stop]
# Same chunk and original particle rows; selected IDs are original halo IDs.
# Apply periodic distances and r <= chosen_radius * R200c after unit conversion.
Pe_shell = (electron_mass * c**2 / sigma_T) * Y_m2.sum() / V_shell_m3""",y,size=8.2)
    report.text('snapshot_chunk and membership_chunk name the matching physical files; start:stop is a cell row range. a and h are header cosmological factors. V_shell_m3 is a geometric physical shell volume, not the sum of particle volumes. The actual code verifies row counts and uses separate index mappings for central and satellite membership.',y,size=10,width=118)

    y=report.start('swiftsimio comparison and local use','Matched spatial masks test extraction, not a separate simulation population')
    y=code(report,"""from swiftsimio import mask, load
from unyt import unyt_array
m = mask(snapshot_path, spatial_only=True)
centre = unyt_array(centre_cmpc, "Mpc")
halfwidth = unyt_array(4 * R200c_physical / a, "Mpc")
m.constrain_spatial([[centre[k]-halfwidth, centre[k]+halfwidth] for k in range(3)])
data = load(snapshot_path, mask=m)
# Convert data.gas.coordinates to the same physical/comoving convention.
# Retain the exact periodic sphere; the mask is a conservative cuboid/cell cut.
# Compare selected ParticleIDs and linear shell sums before comparing fit params.

python -B fit_study.py
python -B verify_study.py
python -B plot_study.py
python -B build_viewer.py
python -B build_report.py""",y,size=9)
    report.text('swift_reader.py is the executed independent route and handles swiftsimio coordinate scaling explicitly. The offline profile_explorer.html contains measurements, fit-radius controls, a rotatable spherical reconstruction and correlated individual-halo coefficient draws. NPZ products retain per-halo profiles and coefficient arrays for independent analysis.',y,size=10,width=118)

    y=report.start('Primary sources and provenance','Papers and upstream code; field units were also checked in actual snapshot headers')
    sources=[('FLAMINGO snapshot fields','https://dataweb.cosma.dur.ac.uk:8443/flamingo/snapshots/snapshot_particle_properties.html'),
        ('SOAP original-particle-order membership writer','https://raw.githubusercontent.com/SWIFTSIM/SOAP/flamingo-hbt/group_membership.py'),
        ('SOAP HBT original halo-index reader','https://raw.githubusercontent.com/SWIFTSIM/SOAP/flamingo-hbt/read_hbtplus.py'),
        ('Battaglia et al. pressure model (B12)','https://arxiv.org/abs/1109.3711'),
        ('Battaglia electron-density / tau model (B16)','https://arxiv.org/abs/1607.02442'),
        ('Lee et al. (2022): conditional gas profiles','https://arxiv.org/abs/2205.01710'),
        ('FLAMINGO simulations and resolution','https://flamingo.strw.leidenuniv.nl/')]
    for title,url in sources:
        y=report.text(title,y,size=11,width=110)
        report.c.setFillColorRGB(.05,.25,.55);report.c.setFont('Text',8.5)
        report.c.drawString(36,y,url);report.c.linkURL(url,(36,y-3,W-36,y+11),relative=0)
        y-=29
    y=report.start('Files created and edited','All changes are confined to this new isolated study directory')
    y=report.text('No earlier analysis outputs or production calibration files were edited. The following source/configuration files were created here; the final delivery_manifest.json lists every data product, figure, checksum and receipt. README.txt documents entry points and array dimensions.',y)
    source_names=sorted(str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and
        (p.suffix in ('.py','.mjs') or p.name in ('study_config.json','viewer_template.html','README.txt')))
    pairs=[source_names[i:i+2]+['']*(2-len(source_names[i:i+2])) for i in range(0,len(source_names),2)]
    y=report.table(['source/configuration','source/configuration'],pairs,y,[385,385],size=9,height=21)
    report.text('Copied helpers preserve the prior physical estimators; pilot_physics.py adds an exact 2R shell boundary. New modules implement exclusive membership components, nested projections, paired radius/control fits, scatter distributions, verification and delivery. Runtime sources and final analysis sources have separate hash manifests.',y,size=9,width=135)
    report.finish()
    print(json.dumps({'report':str(OUTPUT),'pages':report.page}))


if __name__=='__main__':main()
