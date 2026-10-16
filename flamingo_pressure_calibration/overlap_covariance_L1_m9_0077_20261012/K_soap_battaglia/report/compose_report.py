"""Plain-Markdown version of the Test K report (for the hand-back to the workflow), assembled from report/narrative.json,
report/param_tables.md.txt and the equation definitions below.  Prints to stdout (or --out FILE).

  python -B report/compose_report.py [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

K = Path(__file__).resolve().parent.parent

EQUATIONS = r"""
1. **B12 pressure profile** (Battaglia et al. 2012, eq. 10; XGPaint convention):
   P_e(x)/P200 = 0.5176 · P0 · (x/x_c)^γ · [1 + (x/x_c)^α]^(−β),  P200 = G M200c · 200ρ_c(z) · f_b / (2 R200c).
   - x = r/R200c, with r the physical distance from the halo centre [Mpc] (x dimensionless); P_e = n_e k_B T_e, the electron pressure [Pa];
     P200 = the halo's self-similar thermal-pressure scale [Pa].
   - 0.5176 = (2+2X_H)/(3+5X_H) = P_e/P_th for a fully ionised gas with hydrogen mass fraction X_H = 0.76 (dimensionless).
   - P0 = amplitude (thermal pressure in units of P200; dimensionless); x_c = core radius [R200c]; α = transition sharpness (1 in B12) and
     γ = inner slope (−0.3 in B12) (dimensionless); β = outer exponent, P_e ∝ x^(γ−αβ) for x ≫ x_c (dimensionless).
   - G = gravitational constant [m³ kg⁻¹ s⁻²]; M200c = mass inside R200c (SOAP SO/200_crit TotalMass) [Msun]; R200c = radius of the sphere of
     mean density 200ρ_c(z) (SOAP SO/200_crit SORadius, converted to physical) [Mpc]; z = redshift of the halo's SOAP output (dimensionless);
     ρ_c(z) = 3H(z)²/(8πG), the critical density at z [kg m⁻³]; f_b = Ω_b/Ω_m = 0.0486/0.306 = 0.15882 (FLAMINGO, Ω_m incl. massive neutrinos; dimensionless).
   - XGPaint's Battaglia16ThermalSZProfile takes this β and converts it internally to the asymptotic slope αβ − γ.
2. **B16 gas-density profile** (Battaglia 2016, eq. A1, as implemented in XGPaint's BattagliaTauProfile):
   ρ_gas(x)/(f_b ρ_c(z)) = ρ0 · (x/x_c)^γ · [1 + (x/x_c)^α]^(−(β+γ)/α).
   - x = r/R200c (dimensionless); ρ_gas = gas mass density (all gas; no electron conversion) [Msun Mpc⁻³]; ρ0 = amplitude in units of f_b ρ_c(z)
     (dimensionless); x_c = scale radius (0.5 published) [R200c]; α = transition sharpness; β = outer slope (ρ_gas ∝ x^(−β) for x ≫ x_c);
     γ = inner slope (ρ_gas ∝ x^γ for x ≪ x_c; −0.2 published) (all dimensionless).
   - f_b = 0.15882 (dimensionless); ρ_c(z) = critical density at the redshift z of the halo's SOAP output [Msun Mpc⁻³]; R200c as in 1 [Mpc].
   - The paper's eq. A1 prints −(β−γ)/α and no f_b; the convention fitted here is XGPaint's (profiles_tau.jl). With it the published law gives
     M̃(<R200c) = 0.701 at 10^14 Msun, z = 0 (`report/report_numbers.json` published_integral_R200c_1e14_z0|G).
3. **Mass and redshift laws** for each free profile parameter Q (P0, x_c, β for B12; ρ0, α, β for B16; plus γ (B12) or x_c and γ (B16) in 'free'):
   - power law (orig, free): Q = A · (M200c / 10^14 Msun)^α_m · (1+z)^α_z, with M200c the halo mass [Msun], z the redshift of its SOAP output,
     A = value of Q at 10^14 Msun (no h) and z = 0 [units of Q], α_m, α_z = mass and redshift exponents (dimensionless);
   - brk_all (Lee22 BPLall style): ln Q = ln Q_b + α_m^low|high · ln(M200c/M_b) + α_z ln(1+z), with α_m^low for M200c < M_b and α_m^high for M200c ≥ M_b,
     M_b = the break mass shared by the three parameters [Msun], Q_b = value of Q at M_b and z = 0 [units of Q];
   - brk_amp (XGPaint BreakModel): the orig laws, with the amplitude Q_amp (P0 or ρ0) multiplied by (M200c/M_b)^α_break for M200c < M_b
     (α_break dimensionless; M_b = XGPaint M_break [Msun]);
   - the sign of each Q is fixed (γ < 0, all others > 0).
4. **Data** (per halo, from SOAP aperture integrals inside physical radius r = x R200c):
   Ỹ(<x) = Y(<r) (m_e c²/σ_T) / (P200 R200c³) = 4π ∫₀ˣ (P_e(x')/P200) x'² dx',   M̃(<x) = M_gas(<r)/(f_b M200c) = (3/200) ∫₀ˣ ρ_gas(x')/(f_b ρ_c(z)) x'² dx'.
   - r = aperture radius: SORadius/(1+z) (SORadius is stored comoving; z = redshift of the output) or 30–3000 physical kpc [Mpc]; x' = integration variable
     (radius in units of R200c; dimensionless).
   - Y(<r) = (σ_T/m_e c²) ∫_{<r} P_e dV, SOAP ComptonYWithoutRecentAGNHeating (ComptonY in the check) [physical Mpc²]; m_e = electron mass [kg];
     c = speed of light [m s⁻¹]; σ_T = Thomson cross-section [m²]; M_gas(<r) = SOAP GasMass of the SO aperture or InclusiveSphere (all gas inside r) [Msun];
     Ỹ, M̃ = normalised aperture integrals (dimensionless); P_e, P200, ρ_gas, M200c, R200c, f_b, ρ_c(z) as in 1 and 2.
     The right-hand equalities use M200c = (4π/3)·200ρ_c(z)·R200c³.
   - The integral I(x) = ∫₀ˣ F(x') x'² dx' of either profile F is computed in closed form: with q = x'/x_c, s = q^α, t = s/(1+s),
     I = pref · amp · x_c³/α · B(t; a_B, b_B), a_B = (3+γ)/α, b_B = e − a_B, e = the outer exponent (β for B12, (β+γ)/α for B16), pref = 0.5176 (B12) or 1 (B16),
     amp = P0 or ρ0, B(t; a_B, b_B) = ∫₀ᵗ u^(a_B−1)(1−u)^(b_B−1) du the incomplete beta function (u the integration variable; finite for t < 1 and any real b_B;
     b_B < 0 means an outer slope shallower than x⁻³); ln t and ln(1−t) = −ln(1+s) are passed separately so that the result stays exact for large s;
     I = +∞ where γ ≤ −3 (non-integrable core).
5. **Forward model of a cell mean**: d_{c,a} = ln[(1/N_{c,a}) Σ_{h∈c} F̃_{h,a}],  m_{c,a}(θ) = ln[Σ_{k=1}^{32} w_k · norm · I(x_{k,a}; Q(M_k, z_c; θ))].
   - c = cell index (one 0.2-dex M200c bin at one SOAP output); a = aperture index; h = halo index, the sum running over the N_{c,a} central haloes of
     cell c with aperture a valid; k = quadrature-node index (1–32; one node per halo when the cell has ≤ 32 haloes).
   - d_{c,a}, m_{c,a} = data and model (natural log; dimensionless); F̃_{h,a} = Ỹ (pressure) or M̃ (gas) of halo h in aperture a; N_{c,a} = count of those haloes.
   - Nodes = 4 mass-quantile strata × 8 ln-x-quantile strata of those haloes: w_k = stratum share (dimensionless), M_k = exp(stratum mean of ln M200c) [Msun],
     x_{k,a} = exp(stratum mean of ln x) [R200c], so every halo enters at its own aperture radius; z_c = redshift of the output of cell c;
     Q(M, z; θ) = the profile parameters given by the laws of 3; θ = the law parameters (9 to 15); norm = 4π (Ỹ) or 3/200 (M̃).
   - Accuracy (`results/k7_fit_audit.json`): FORWARD_NOTE_PLACEHOLDER
6. **Likelihood**: χ²(θ) = Σ_c r_cᵀ [C_c + (σ_d² + σ_0²) 𝟏]⁻¹ r_c,  r_c = d_c − m_c(θ);  σ_d is solved so that χ²_min/(N_pts − n_θ) = 1.
   - d_c, m_c(θ), r_c = vectors of the cell's p data points, model values and residuals (ln; dimensionless); ᵀ = transpose;
     C_c = p × p bootstrap covariance of the cell's log means (200 halo resamples), its correlation shrunk towards the redshift-pooled correlation with weight
     λ = clip(p/N, 0.05, 1) (p = points in the cell, N = haloes in the cell); 𝟏 = the p × p identity matrix; σ_d = model-discrepancy floor [ln units];
     σ_0 = 0.002 numerical floor [ln units]; χ²_min = χ² at the best-fitting θ; N_pts = number of data points of the fit; n_θ = number of law parameters.
   - BIC = χ²(θ; σ_ref) + n_θ ln N_pts, with σ_ref = the orig fit's σ_d of the same field and range.
   - Convergence: CHI2_NOTE_PLACEHOLDER
"""


def strip_page_syntax(s):
    s = re.sub(r"\[\]\(eq:[^)]*\)", "the equations below", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nar = json.loads((K / "report" / "narrative.json").read_text())
    tabs = (K / "report" / "param_tables.md.txt").read_text()
    parts = ["## Battaglia parameters refitted to FLAMINGO (SOAP), Test K run 3", "", "### In short"]
    parts += [f"- {b}" for b in nar["in_short"]]
    eqs = EQUATIONS.strip().replace("FORWARD_NOTE_PLACEHOLDER", nar["forward_note"]).replace("CHI2_NOTE_PLACEHOLDER", nar["chi2_note"])
    parts += ["", "### Data, selection and definitions", nar["data"], "", "#### Haloes, cells and points per output", nar["per_output_table"], "",
              "### Equations (every symbol defined)", eqs, "",
              "### Fit quality", tabs.strip().split("\n\n")[-1], "",
              "n.c. = not converged: the least-squares solve stopped at its iteration limit, so the value is not the χ²/ν = 1 floor.", "",
              nar["quality_text"], "",
              "### Reading the parameters", nar["results_text"], "",
              "### Aperture integrals", nar["apertures_text"], "",
              "### Profiles", nar["profiles_text"], "", nar["trends_text"], "",
              "### Electrons for DM and kSZ", nar["electrons_text"], "",
              "### Using the parameters",
              "- XGPaint: `report/xgpaint_refit_snippet.jl` (Float64 literals; runs with the validated XGPaint and reproduces the fitted profiles, "
              "`report/xgpaint_snippet_check.json`). Python: `report/k_params_usage.py`. Every fit with theta, Hessian covariance, bootstrap sd, "
              "sigma_d, data range, conventions and fit_status (converged, usable; G|R1|free is not usable): `results/k_params.json`.",
              "- Full-particle run: pass per-halo aperture integrals (kind 0) or shell profiles (kind 1) for each cell to `k_fit_core.summarise_cell`, "
              "build the dataset with `build_dataset`, fit with `fit_global(D, Law(form, kind))`, then `bootstrap_global` and `law_table`; with the same "
              "Law kinds and cells the parameters are directly comparable. Compare profiles and aperture integrals (results/k4_profiles.npz, "
              "k4_profile_ratios.json), since P0, x_c and beta are degenerate.",
              "", "### Known limits"]
    parts += [f"- {b}" for b in nar["limits"]]
    parts += ["", "### Recommendations"] + [f"- {b}" for b in nar["recommendations"]]
    parts += ["", "### Run record", nar["run"], "", "### Changes from earlier versions"] + [f"- {b}" for b in nar["changes"]]
    text = strip_page_syntax("\n".join(parts)) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
