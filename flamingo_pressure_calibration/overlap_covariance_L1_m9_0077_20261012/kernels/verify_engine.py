"""Gates for the profile-space engine (Step 0.1).

1. Hard-cut parity: the original 256-node grids and coefficient sets reproduce
   tsz_halo_terms.npz (16 fit x cut cases, 83 ell, 1h/2h/total, by-z terms,
   mean y, cosine-taper and Y/shape-matched controls).
2. Shared fine grid (1024 log nodes on 1e-5..4R plus shell breaks) against the
   original grid, the existing 1024-node refinement and its own doubling.
3. DM definitions: the 0.02-dex scenario table reproduces analytic_spectra.npz
   (baseline_R2/3/4: C_ell^yy self/linear2h, mean y, mean DM z_s=1, lambda).
4. DM variance: projection operator on analytic profiles, Parseval against the
   Fourier kernel, and the compound-Poisson ray control's sample variance.
5. Transforms against analytic top-hat and Gaussian Fourier transforms.
Writes verification.json (gates) and gate_spectra.npz in this directory.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_engine", HERE/"engine.py")
E = importlib.util.module_from_spec(spec)
sys.modules["kernel_engine"] = E
spec.loader.exec_module(E)

import numpy as np
from scipy import integrate

WORKERS = 8


# The frozen products were made with scipy 1.11.4 (CODATA 2018). Newer scipy
# (CODATA 2022) changes sigma_T/m_e by -5.4e-9 and m_p by +1.4e-9; these exact
# factors convert the current-environment values to the producing constants.
CODATA2018 = dict(sigma_T=6.6524587321e-29, electron_mass=9.1093837015e-31, proton_mass=1.67262192369e-27)


def codata_factors():
    """Multiply current y amplitudes / DM by these to use CODATA 2018 constants."""
    from scipy.constants import physical_constants
    now = dict(sigma_T=E.P.SIGMA_T, electron_mass=E.P.electron_mass,
               proton_mass=physical_constants["proton mass"][0])
    mu = lambda c: (c["electron_mass"]+c["proton_mass"]*2/(1+E.P.XH_XGP))/.9
    y = (CODATA2018["sigma_T"]/CODATA2018["electron_mass"])/(now["sigma_T"]/now["electron_mass"])
    return dict(y_amplitude=y, dm=mu(now)/mu(CODATA2018), current=now, reference=CODATA2018)


def max_relative(new, old, floor=0.):
    new, old = np.asarray(new, float), np.asarray(old, float)
    use = np.abs(old) > floor
    return float(np.max(np.abs(new[use]/old[use]-1))) if use.any() else 0.


def gate_parity(engine, support):
    """Original grids, the 16 fit x cut cases and the taper/matched controls."""
    saved = dict(np.load(E.TRUNCATION/"halo_model/tsz_halo_terms.npz"))
    variants = []
    grids = {cut: E.original_grid(cut) for cut in E.H.RADII}
    for fi, ci in E.H.CASES:
        fit, cut = E.H.RADII[fi], E.H.RADII[ci]
        variants.append(E.Variant(f"fit{int(fit)}_cut{int(cut)}", E.pressure_fit(fit, support), grids[cut]))
    for cut in E.H.RADII:
        variants.append(E.Variant(f"taper_cut{int(cut)}", E.pressure_fit(4, support), grids[cut],
                                  window=E.cosine_window(grids[cut].x, cut)))
    for cut in E.H.RADII:
        variants.append(E.Variant(f"taper_Ymatched_cut{int(cut)}",
                                  derived=(f"taper_cut{int(cut)}", f"fit4_cut{int(cut)}")))
    for label in list(saved["labels"]):
        variants.append(E.Variant(f"shape_{label}", derived=(str(label), "fit4_cut4")))
    result = engine.run(variants, ell=E.ELL, workers=WORKERS)
    labels = list(result["labels"])
    main = [labels.index(str(l)) for l in saved["labels"]]
    taper = [labels.index(f"taper_cut{int(c)}") for c in E.H.RADII]
    tmatch = [labels.index(f"taper_Ymatched_cut{int(c)}") for c in E.H.RADII]
    shape = [labels.index(f"shape_{l}") for l in saved["labels"]]
    factors = codata_factors()
    raw_one, raw_two = result["cl_1h"], result["cl_2h"]
    raw_checks = dict(cl_1h=max_relative(raw_one[main], saved["cl_1h"]),
                      cl_linear_2h=max_relative(raw_two[main], saved["cl_linear_2h"]),
                      cl_total=max_relative(raw_one[main]+raw_two[main], saved["cl_total"]),
                      mean_compton_y=max_relative(result["mean"][main], saved["mean_compton_y"]))
    fy = factors["y_amplitude"]
    one, two = raw_one*fy**2, raw_two*fy**2
    corrected = dict(mean=result["mean"]*fy, cl_1h_by_z=result["cl_1h_by_z"]*fy**2,
                     cl_2h_by_z=result["cl_2h_by_z"]*fy**2)
    checks = dict(
        cl_1h=max_relative(one[main], saved["cl_1h"]),
        cl_linear_2h=max_relative(two[main], saved["cl_linear_2h"]),
        cl_total=max_relative(one[main]+two[main], saved["cl_total"]),
        dl_total_uK2=max_relative((one[main]+two[main])*E.ELL*(E.ELL+1)/(2*np.pi)*E.A.T_CMB_UK**2, saved["dl_total_uK2"]),
        mean_compton_y=max_relative(corrected["mean"][main], saved["mean_compton_y"]),
        cl_1h_by_z=max_relative(corrected["cl_1h_by_z"][main], saved["cl_1h_by_z"], 1e-300),
        cl_linear_2h_by_z=max_relative(corrected["cl_2h_by_z"][main], saved["cl_linear_2h_by_z"], 1e-300),
        taper_cl_1h=max_relative(one[taper], saved["taper_cl_1h"]),
        taper_cl_linear_2h=max_relative(two[taper], saved["taper_cl_linear_2h"]),
        taper_mean_y=max_relative(corrected["mean"][taper], saved["taper_mean_y"]),
        taper_Ymatched_cl_1h=max_relative(one[tmatch], saved["taper_Ymatched_cl_1h"]),
        taper_Ymatched_cl_linear_2h=max_relative(two[tmatch], saved["taper_Ymatched_cl_linear_2h"]),
        shape_matched_cl_1h=max_relative(one[shape], saved["shape_matched_cl_1h"]),
        shape_matched_cl_linear_2h=max_relative(two[shape], saved["shape_matched_cl_linear_2h"]))
    worst = max(checks[k] for k in ("cl_1h", "cl_linear_2h", "cl_total"))
    i100, i1000 = (int(np.flatnonzero(E.ELL == e)[0]) for e in (100., 1000.))
    total = one+two
    known = dict(
        truncation_fit4_cut123_ell100_percent=[100*(total[main[12+k], i100]/total[main[15], i100]-1) for k in range(3)],
        taper_cut4_change_percent_ell100_1000=[100*(total[taper[3], i]/total[main[15], i]-1) for i in (i100, i1000)],
        taper_cut4_Ymatched_change_percent_ell100_1000=[100*(total[tmatch[3], i]/total[main[15], i]-1) for i in (i100, i1000)])
    gate = dict(passed=bool(worst <= 1e-10 and max(checks.values()) <= 1e-10), tolerance=1e-10,
                max_relative_deviation=checks, worst_main_spectra=worst, known_numbers=known,
                raw_max_relative_deviation_current_constants=raw_checks,
                constants=dict(factors, note="checks are for CODATA-2018 constants (the producing scipy 1.11.4); "
                               "raw values use this interpreter's scipy.constants. The raw offset equals "
                               "y_amplitude**2-1 for C_ell and y_amplitude-1 for mean y."),
                cases=[str(l) for l in saved["labels"]], ell_count=len(E.ELL),
                grid="np.unique(np.r_[np.geomspace(1e-5,cut,256),.9*cut]) per cut, as compute_halo_terms.py:173",
                timing=result["timing"])
    return gate, result


def fine_grid(cut, refine=1):
    breaks = [e for e in E.SHELL_EDGES[1:] if e <= cut]+[.9*cut]
    return E.make_grid(cut, breaks=breaks, refine=refine)


def gate_fine(engine, support, parity):
    """Shared fine grid against the original grid and the 1024-node refinement."""
    saved = dict(np.load(E.TRUNCATION/"halo_model/tsz_halo_terms.npz"))
    refined = dict(np.load(E.TRUNCATION/"halo_model/independent_checks.npz"))
    outputs = {}
    for refine in (1, 2):
        grids = {cut: fine_grid(cut, refine) for cut in E.H.RADII}
        variants = [E.Variant(f"fit{int(E.H.RADII[fi])}_cut{int(E.H.RADII[ci])}",
                              E.pressure_fit(E.H.RADII[fi], support), grids[E.H.RADII[ci]])
                    for fi, ci in E.H.CASES]
        outputs[refine] = engine.run(variants, ell=E.ELL, workers=WORKERS)
        outputs[refine]["nodes"] = [len(grids[c]) for c in E.H.RADII]
    fine, double = outputs[1], outputs[2]
    select = np.searchsorted(E.ELL, E.H.SELECT_ELL)
    rows = {}
    for term, key, ref in (("1h", "cl_1h", "refined_cl_1h"), ("2h", "cl_2h", "refined_cl_linear_2h")):
        rows[term] = dict(
            fine_vs_original_max_abs=float(np.max(np.abs(fine[key]/saved["cl_1h" if term == "1h" else "cl_linear_2h"]-1))),
            fine_vs_existing_1024_refinement_max_abs=float(np.max(np.abs(fine[key][:, select]/refined[ref]-1))),
            fine_vs_doubled_max_abs=float(np.max(np.abs(fine[key]/double[key]-1))))
    total = dict(fine_vs_original_max_abs=float(np.max(np.abs(fine["cl_total"]/saved["cl_total"]-1))),
                 fine_vs_existing_1024_refinement_max_abs=float(np.max(np.abs(
                     fine["cl_total"][:, select]/(refined["refined_cl_1h"]+refined["refined_cl_linear_2h"])-1))),
                 fine_vs_doubled_max_abs=float(np.max(np.abs(fine["cl_total"]/double["cl_total"]-1))),
                 original_vs_doubled_max_abs=float(np.max(np.abs(saved["cl_total"]/double["cl_total"]-1))))
    existing = json.loads((E.TRUNCATION/"halo_model/independent_checks_verification.json").read_text())
    gate = dict(passed=bool(total["fine_vs_original_max_abs"] <= 1.1*existing["maximum_relative_spectrum_error"]
                            and total["fine_vs_doubled_max_abs"] < 1e-3),
                existing_256_vs_1024_node_check_max=existing["maximum_relative_spectrum_error"],
                one_halo=rows["1h"], two_halo=rows["2h"], total=total,
                nodes_fine=fine["nodes"], nodes_doubled=double["nodes"],
                note="Fine grid: 1024 log nodes per 1e-5..4R (fixed log step), shell edges and 0.9*cut as exact "
                     "segment breaks. The original 256-node grid is biased high by the chord of a convex profile; "
                     "the fine grid matches the existing independent 1024-node refinement.",
                timing_fine=fine["timing"], timing_doubled=double["timing"])
    return gate, fine


def gate_dm(engine):
    """Reproduce analytic_spectra.npz baseline_R2/3/4 with the scenario table."""
    scenarios = dict(np.load(E.SENSITIVITY/"profile_scenarios.npz"))
    analytic = np.load(E.SENSITIVITY/"analytic_observables/analytic_spectra.npz")
    variants = []
    C_RADII = (2., 3., 4.)
    for ri, cut in enumerate(C_RADII):
        grid = E.Grid(E.A.radial_grid(scenarios["x"], cut), exact=True)
        for field in (0, 1):
            variants.append(E.Variant(f"baseline_{'y' if field == 0 else 'e'}_R{int(cut)}",
                                      E.ScenarioProfile(scenarios, 0, field, ri, "baseline"), grid))
    result = engine.run(variants, ell=E.A.ELL[1:], workers=WORKERS)
    labels = list(result["labels"])
    y = [labels.index(f"baseline_y_R{int(c)}") for c in C_RADII]
    e = [labels.index(f"baseline_e_R{int(c)}") for c in C_RADII]
    old = dict(cl_y_self=analytic["cl_y_self"][:3, 1:], cl_y_linear2h=analytic["cl_y_linear2h"][:3, 1:],
               mean_Compton_y=analytic["mean_Compton_y"][:3], dm=analytic["frb_z1_mean_DM_pc_cm3"][:3],
               lam=analytic["poisson_interception_lambda"][:3])
    assert list(analytic["labels"][:3]) == ["baseline_R2", "baseline_R3", "baseline_R4"]
    factors = codata_factors()
    fy, fd = factors["y_amplitude"], factors["dm"]
    raw_checks = dict(cl_y_self=max_relative(result["cl_1h"][y], old["cl_y_self"]),
                      mean_Compton_y=max_relative(result["mean"][y], old["mean_Compton_y"]),
                      frb_z1_mean_DM_pc_cm3=max_relative(result["dm_mean"][e], old["dm"]))
    checks = dict(cl_y_self=max_relative(result["cl_1h"][y]*fy**2, old["cl_y_self"]),
                  cl_y_linear2h=max_relative(result["cl_2h"][y]*fy**2, old["cl_y_linear2h"]),
                  mean_Compton_y=max_relative(result["mean"][y]*fy, old["mean_Compton_y"]),
                  frb_z1_mean_DM_pc_cm3=max_relative(result["dm_mean"][e]*fd, old["dm"]),
                  poisson_interception_lambda=max_relative(result["lambda_hit"][e], old["lam"]))
    values = dict(mean_DM_R2_R3_R4=result["dm_mean"][e], DM_variance_R2_R3_R4=result["dm_var"][e],
                  DM_std_R2_R3_R4=np.sqrt(result["dm_var"][e]))
    # Compound-Poisson ray control: same profiles/counts, Poissonized centres.
    control = np.load(E.SENSITIVITY/"lightcone_maps/frb_poisson_control.npz")
    names = [str(n) for n in control["task_names"]]
    mc = []
    rng = np.random.default_rng(20261012)
    for k, cut in enumerate(C_RADII):
        sample = control["dm_pc_cm3"][names.index(f"baseline_r{int(cut)}")]
        boot = np.array([np.var(sample[rng.integers(0, len(sample), len(sample))]) for _ in range(200)])
        mc.append(dict(radius=cut, mc_mean=float(sample.mean()), mc_variance=float(sample.var(ddof=1)),
                       mc_variance_bootstrap_sd=float(boot.std(ddof=1)),
                       analytic_mean=float(result["dm_mean"][e[k]]), analytic_variance=float(result["dm_var"][e[k]]),
                       variance_ratio_minus1=float(result["dm_var"][e[k]]/sample.var(ddof=1)-1),
                       pull=float((result["dm_var"][e[k]]-sample.var(ddof=1))/boot.std(ddof=1))))
    # kSZ self term with the matched catalogue velocities (existing apply_fine_velocity.py
    # product, which squares float32 tau kernels: agreement is limited to ~1e-7).
    ksz_old = np.load(E.SENSITIVITY/"analytic_observables/ksz_self_fine_velocity.npz")
    assert [str(l) for l in ksz_old["labels"][:3]] == ["baseline_R2", "baseline_R3", "baseline_R4"]
    tau_factor = (CODATA2018["sigma_T"]/E.P.SIGMA_T)*fd      # tau ~ sigma_T/mu_XGP
    ksz = dict(cl_ksz_self_max_relative_current_constants=max_relative(result["ksz_cl_1h"][e], ksz_old["cl_ksz_self"][:3, 1:]),
               cl_ksz_self_max_relative=max_relative(result["ksz_cl_1h"][e]*tau_factor**2, ksz_old["cl_ksz_self"][:3, 1:]),
               tolerance=1e-6,
               note="existing product squares float32 tau kernels (tau_kernel_cache.npz), so agreement is limited to ~1e-7; "
                    "constants converted to CODATA 2018 as for the other checks")
    ksz["passed"] = bool(ksz["cl_ksz_self_max_relative"] <= 1e-6)
    # Compound-Poisson PDF from the FFT against the existing Poisson ray control.
    levels = np.array([.5, .9, .99, .999])
    pdf_rows = []
    for k, cut in enumerate(C_RADII):
        v = e[k]
        row = list(result["dm_pdf_rows"]).index(v)
        pmf = E.compound_poisson_pdf(result["dm_intensity"][row], result["dm_overflow_intensity"][row])
        mine = quantiles_at(pmf, levels)*fd
        sample = control["dm_pc_cm3"][names.index(f"baseline_r{int(cut)}")]
        boot = np.array([np.quantile(sample[rng.integers(0, len(sample), len(sample))], levels) for _ in range(100)])
        reference = np.quantile(sample, levels)
        pdf_rows.append(dict(radius=cut, levels=levels, fft_quantiles=mine, control_quantiles=reference,
                             control_bootstrap_sd=boot.std(axis=0, ddof=1),
                             pull=(mine-reference)/boot.std(axis=0, ddof=1),
                             pdf_mean_over_exact_minus1=float(result["dm_pdf_mean"][v]/result["dm_mean"][v]-1),
                             pdf_variance_over_exact_minus1=float(result["dm_pdf_variance"][v]/result["dm_var"][v]-1)))
    pdf_gate = dict(passed=bool(all(np.all(np.abs(r["pull"]) < 4) and abs(r["pdf_mean_over_exact_minus1"]) < 1e-5
                                    and abs(r["pdf_variance_over_exact_minus1"]) < 1e-3 for r in pdf_rows)),
                    rows=pdf_rows, note="FFT compound-Poisson PDF of the same scenario profiles vs the 120k-ray seeded "
                                        "Poisson control (frb_poisson_control.npz; 64-node z column tables); pulls use bootstrap SDs")
    worst = max(checks.values())
    gate = dict(passed=bool(worst <= 1e-10 and ksz["passed"] and pdf_gate["passed"]), tolerance=1e-10,
                max_relative_deviation=checks, ksz_self_term=ksz, dm_pdf_vs_poisson_control=pdf_gate,
                raw_max_relative_deviation_current_constants=raw_checks,
                constants=dict(y_amplitude=fy, dm=fd, note="DM depends on mu_XGP (m_e, m_p); sigma_T cancels"),
                definitions=dict(
                    mean_DM="sum_{z<=1 cells} N tau_0/(4 pi sigma_T pc_cm3 (1+z)); analytic_DM_moments.json mean_halo_DM_z1_pc_cm3",
                    DM_variance="compound-Poisson (Campbell) variance of halo DM for independent centres: "
                                "sum_{z<=1 cells} N/(4 pi) int d^2theta DM_h(theta)^2, the analytic counterpart of "
                                "frb_poisson_control.py; not in analytic_DM_moments.json, which has no quadratic DM term",
                    lambda_="Poisson interception expectation, profile independent"),
                values=values, poisson_ray_control=mc,
                poisson_control_note="MC uses the painter's column tables (64-node z interpolation, exact cap geometry); "
                                     "agreement is expected within MC error plus table interpolation.",
                timing=result["timing"])
    return gate, result


def quantiles_at(pmf, levels):
    grid = np.arange(E.DM_POINTS)*E.DM_DX
    return np.interp(levels, np.cumsum(np.maximum(pmf, 0.)), grid+.5*E.DM_DX)


def gate_pdf_monte_carlo(engine, support):
    """FFT compound-Poisson PDF vs a fresh seeded Monte Carlo (2 seeds), E4 hard 4R."""
    spec_mc = importlib.util.spec_from_file_location("kernel_dm_pdf_mc", HERE/"dm_pdf_mc.py")
    MC = importlib.util.module_from_spec(spec_mc)
    spec_mc.loader.exec_module(MC)
    grid = E.make_grid(4., breaks=list(E.SHELL_EDGES[1:]))
    profile = E.electron_fit(4, support)
    result = engine.run([E.Variant("E4", profile, grid, 4.)], ell=np.array([100.]), workers=WORKERS, verbose=False)
    fft = result["dm_quantiles"][0]
    rows = []
    for seed in (1, 2):
        started = time.perf_counter()
        dm, events = MC.sample(engine.sky, profile, grid, 4., rays=200000, seed=seed)
        summary = MC.summary(dm)
        rows.append(dict(seed=seed, rays=200000, events=int(events), quantiles=summary["quantiles"],
                         bootstrap_sd=summary["quantile_bootstrap_sd"], pull=(summary["quantiles"]-fft)/summary["quantile_bootstrap_sd"],
                         mean_over_exact_minus1=summary["mean"]/result["dm_mean"][0]-1,
                         variance_over_exact_minus1=summary["variance"]/result["dm_var"][0]-1, seconds=time.perf_counter()-started))
    seed_difference = np.abs(rows[0]["quantiles"]-rows[1]["quantiles"])/np.sqrt(rows[0]["bootstrap_sd"]**2+rows[1]["bootstrap_sd"]**2)
    return dict(passed=bool(all(np.all(np.abs(r["pull"]) < 4) for r in rows)), levels=E.QUANTILES, fft_quantiles=fft,
                exact_mean=result["dm_mean"][0], exact_std=float(np.sqrt(result["dm_var"][0])),
                monte_carlo=rows, seed_to_seed_pull=seed_difference,
                note="MC numerical noise = bootstrap SD per seed; the FFT result has no sampling noise (CIC grid dx=0.125 pc cm^-3)")


def gate_projection():
    """Projection operator and DM-variance quadratic form on analytic profiles."""
    rows = {}
    grid = E.Grid(np.r_[1e-12, np.linspace(1e-3, 1., 400)])
    b = np.array([0., .1, .5, .9, .99, .999])
    sigma = E.column(grid, np.ones(len(grid)), b)
    rows["tophat_column_max_abs_error"] = float(np.max(np.abs(sigma-2*np.sqrt(1-b*b))))
    rows["tophat_variance_relative_error"] = float(abs(np.ones(len(grid))@E.variance_forms(grid, 1.)["total"]@np.ones(len(grid))/(2*np.pi)-1))
    grid = E.Grid(np.r_[1e-12, np.linspace(1e-3, 12., 6000)])
    f = np.exp(-grid.x**2/2)
    b = np.linspace(0., 4., 41)
    sigma = E.column(grid, f, b)
    rows["gaussian_column_max_abs_error_over_peak"] = float(np.max(np.abs(sigma-np.sqrt(2*np.pi)*np.exp(-b*b/2)))/np.sqrt(2*np.pi))
    rows["gaussian_variance_relative_error"] = float(abs(f@E.variance_forms(grid, 12.)["total"]@f/(2*np.pi**2)-1))
    # Parseval: int d^2b Sigma^2 = (1/2pi) int q |K(q)|^2 dq for a hard-cut B16 profile.
    support = E.support_definition()
    electrons = E.electron_fit(4, support)
    grid = E.make_grid(4., breaks=E.SHELL_EDGES[1:])
    profile = electrons.base(np.array([10**14.6]), grid.x)[0]
    direct = float(profile@E.variance_forms(grid, 4.)["total"]@profile)
    q = np.geomspace(1e-4, 3e4, 6000)
    kernel = E.transform(grid, profile, q)
    integrand = q*kernel**2/(2*np.pi)
    edge = 4*np.pi*4.*profile[-1]
    tail = (edge**2/(2*np.pi))/(2*q[-1]**2)/2     # <cos^2>=1/2 of the q^-3 edge tail
    parseval = float(integrate.trapezoid(integrand*q, np.log(q)))+tail
    rows["parseval_hard_cut_B16_relative_difference"] = parseval/direct-1
    rows["parseval_note"] = "log-q trapezoid on 6000 nodes to q=3e4 plus the analytic q^-3 edge tail; limited by the q quadrature"
    passed = (rows["tophat_column_max_abs_error"] < 1e-10 and rows["tophat_variance_relative_error"] < 1e-10
              and rows["gaussian_column_max_abs_error_over_peak"] < 1e-6 and rows["gaussian_variance_relative_error"] < 1e-6
              and abs(rows["parseval_hard_cut_B16_relative_difference"]) < 1e-3)
    return dict(passed=bool(passed), **rows)


def gate_transforms():
    """Exact transform against analytic top-hat and Gaussian Fourier transforms."""
    q = np.r_[0., 1e-3, .05, .1999, .2001, .5, 1., 2., 5., 10., 20., 50.]
    grid = E.Grid(np.r_[1e-6, np.linspace(1e-3, 1., 50)])
    tophat = E.transform(grid, np.ones(len(grid)), q)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.where(q > 0, 4*np.pi*(np.sin(q)-q*np.cos(q))/np.maximum(q, 1e-300)**3, 4*np.pi/3)
    small = q < 1e-2
    exact[small] = 4*np.pi*(1/3-q[small]**2/30)
    tophat_error = float(np.max(np.abs(tophat-exact))/(4*np.pi/3))
    grid = E.Grid(np.r_[1e-6, np.linspace(1e-3, 12., 12000)])
    f = np.exp(-grid.x**2/2)
    gauss = E.transform(grid, f, q)
    exact_g = (2*np.pi)**1.5*np.exp(-q*q/2)
    gauss_error = float(np.max(np.abs(gauss-exact_g))/(2*np.pi)**1.5)
    big = exact_g > 1e-3*(2*np.pi)**1.5
    gauss_relative = float(np.max(np.abs(gauss[big]/exact_g[big]-1)))
    # Extended-grid guard: series vs closed form for small q on a 22R grid.
    x = np.geomspace(1e-5, 22.4, 1500)
    profile = E.b16(np.array([2605., 1.118, 3.777]), x)
    qs = np.array([.1, .15])
    guarded = E.radial_weights(x, qs)@profile
    # Independent reference: 24-point Gauss-Legendre per interval of the same
    # piecewise-linear profile times the smooth sinc (exact to rounding here).
    t, w = np.polynomial.legendre.leggauss(24)
    lo, hi = x[:-1, None], x[1:, None]
    r = lo+(hi-lo)*(t+1)/2
    f_lin = profile[:-1, None]+(profile[1:]-profile[:-1])[:, None]*(r-lo)/(hi-lo)
    reference = np.array([np.sum(4*np.pi*r*r*f_lin*np.sinc(qq*r/np.pi)*(hi-lo)*w/2) for qq in qs])
    unguarded = E.A.linear_radial_kernel(x, qs)@profile
    return dict(passed=bool(tophat_error < 1e-6 and gauss_error < 1e-6),
                tophat_max_abs_error_over_zero_mode=tophat_error,
                gaussian_max_abs_error_over_zero_mode=gauss_error,
                gaussian_max_relative_error_where_above_1e3_of_peak=gauss_relative,
                q_tested=q, gaussian_nodes=len(grid),
                extended_grid_guard=dict(q=qs, guarded_relative_error=np.abs(guarded/reference-1),
                                         frozen_series_relative_error=np.abs(unguarded/reference-1),
                                         grid="1500 log nodes to 22.4R, B16 14.5-14.7 4R fit",
                                         note="24-point Gauss-Legendre per interval of the same piecewise-linear profile; "
                                              "shows why the guard exists on grids extended beyond 7.5R"))


def embed(verification):
    """Attach the Jacobian and original-interpreter receipts; overall pass needs all."""
    jac = HERE/"jacobians_verification.json"
    if jac.exists():
        verification["jacobian_gates"] = json.loads(jac.read_text())
    other = HERE/"verification_base_interpreter.json"
    if other.exists():
        base = json.loads(other.read_text())
        verification["original_interpreter_rerun"] = dict(
            passed=base["passed"], interpreter=base["interpreter"], scipy=base["scipy"],
            max_relative_deviation={name: base["gates"][name]["max_relative_deviation"] for name in base["gates"]},
            note="parity and DM gates rerun with the interpreter that produced the frozen products (CODATA 2018)")
    verification["passed"] = bool(all(g["passed"] for g in verification["gates"].values())
                                  and verification["frozen_inputs_unchanged"]
                                  and verification.get("jacobian_gates", {}).get("passed", False)
                                  and verification.get("original_interpreter_rerun", {}).get("passed", False))
    return verification


def main():
    """All gates (default) or a subset: --only parity,dm --output name.json;
    --only embed refreshes the attached receipts of an existing verification.json."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--output", default="verification.json")
    args = parser.parse_args()
    only = set(filter(None, args.only.split(",")))
    if only == {"embed"}:
        verification = embed(json.loads((HERE/"verification.json").read_text()))
        E.write_json(HERE/"verification.json", verification)
        print(json.dumps(dict(passed=verification["passed"])), flush=True)
        return
    wanted = lambda name: not only or name in only
    started = time.perf_counter()
    hashes = E.source_hashes()
    engine = E.Engine()
    support = E.support_definition()
    gates = {}
    if wanted("transforms"):
        gates["transforms_analytic"] = gate_transforms()
    if wanted("projection"):
        gates["projection_and_variance"] = gate_projection()
    if gates:
        print(json.dumps(E.C.plain(gates), indent=1), flush=True)
    parity = fine = dm = None
    if wanted("parity"):
        gates["hard_cut_parity"], parity = gate_parity(engine, support)
        print(json.dumps(E.C.plain(gates["hard_cut_parity"]["max_relative_deviation"]), indent=1), flush=True)
    if wanted("fine"):
        gates["fine_grid"], fine = gate_fine(engine, support, parity)
        print(json.dumps(E.C.plain(gates["fine_grid"]), indent=1), flush=True)
    if wanted("dm"):
        gates["dm_definitions"], dm = gate_dm(engine)
        print(json.dumps(E.C.plain(gates["dm_definitions"]), indent=1), flush=True)
    if wanted("pdfmc"):
        gates["dm_pdf_monte_carlo"] = gate_pdf_monte_carlo(engine, support)
        print(json.dumps(E.C.plain(gates["dm_pdf_monte_carlo"]), indent=1), flush=True)
    if parity is not None and fine is not None and dm is not None:
        np.savez_compressed(HERE/"gate_spectra.npz", ell=E.ELL, labels=parity["labels"],
                            parity_cl_1h=parity["cl_1h"], parity_cl_2h=parity["cl_2h"], parity_mean=parity["mean"],
                            fine_labels=fine["labels"], fine_cl_1h=fine["cl_1h"], fine_cl_2h=fine["cl_2h"],
                            dm_labels=dm["labels"], dm_mean=dm["dm_mean"], dm_var=dm["dm_var"],
                            note="raw engine outputs with this interpreter's scipy.constants")
    unchanged = E.source_hashes() == hashes
    verification = dict(passed=bool(all(g["passed"] for g in gates.values()) and unchanged),
                        gates=gates, frozen_inputs_SHA256=hashes, frozen_inputs_unchanged=unchanged,
                        interpreter=sys.executable, python=sys.version, numpy=np.__version__,
                        scipy=__import__("scipy").__version__, elapsed_seconds=time.perf_counter()-started)
    if args.output == "verification.json":
        embed(verification)
    E.write_json(HERE/args.output, verification)
    print(json.dumps(dict(passed=verification["passed"], elapsed=verification["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
