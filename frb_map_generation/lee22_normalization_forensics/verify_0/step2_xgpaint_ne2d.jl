# verify_0 / step 2: paint single halos through XGPaint's UNMODIFIED BattagliaTauProfilePhysical.ne2d
# (profiles_tau.jl rho_2d -> ne2d) with Lee22 Table-8 parameters (parsed from TeX in step 1), using
# per-halo constant PowerLawParams, for three amplitude mappings:
#   F = 1                      P0 = 200 n0              (production :xgpaint_ne2d)
#   F = 1.6621                 P0 = 1.6621 x 200 n0     (claim's rounded faithful mapping)
#   F = 1/0.6016316602183194   P0 = exact faithful mapping
# beta_XGPaint = beta' + 0.3, alpha = 1, gamma = -0.3.
# Also queries the repository implementation (read-only include) for comparison.
using XGPaint, Unitful, UnitfulAstro, Printf

const REPO = "/home/cbllover/HalfDome/frb_map_generation"
const OUT = joinpath(REPO, "outputs/lee22_normalization_forensics_20260928/verify_0")
include(joinpath(REPO, "lee2022_frb_dm_profile.jl"))

lines = readlines(joinpath(OUT, "step1_param_grid.csv"))
rows = [split(l, ",") for l in lines[2:end]]
xs = (0.01, 0.05, 0.2, 0.5, 1.0, 1.34, 3.0)
Fs = (1.0, 1.6621, 1 / 0.6016316602183194)

open(joinpath(OUT, "step2_constants.txt"), "w") do io
    println(io, "M_sun_kg=", ustrip(u"kg", XGPaint.M_sun))
    println(io, "G_SI=", ustrip(u"m^3/kg/s^2", XGPaint.constants.G))
    println(io, "m_p_kg=", ustrip(u"kg", XGPaint.constants.ProtonMass))
    println(io, "m_e_kg=", ustrip(u"kg", XGPaint.constants.ElectronMass))
    println(io, "Mpc_m=", ustrip(u"m", 1.0u"Mpc"))
    println(io, "pc_cm=", ustrip(u"cm", 1.0u"pc"))
end

io = open(joinpath(OUT, "step2_xgpaint_ne2d.csv"), "w")
println(io, "cosmo,M_msun,z,x,r200c_m,rho_crit_kgm3,F,xgp_ne2d_m2,repo_literal_col_m2,repo_xgpnative_col_m2,repo_literal_n3d_m3,repo_xgpnative_n3d_m3,repo_n0,repo_xc,repo_beta_prime,repo_xgp_P0,repo_xgp_beta,E2_xgpaint")
for r in rows
    cname = String(r[1]); Oc = parse(Float64, r[2]); Ob = parse(Float64, r[3]); h = parse(Float64, r[4])
    M = parse(Float64, r[5]); z = parse(Float64, r[6])
    n0 = parse(Float64, r[7]); xc = parse(Float64, r[8]); bp = parse(Float64, r[9])
    # repository models, same cosmology
    lit = Lee2022NoConcentrationDMProfile(Omega_c=Oc, Omega_b=Ob, h=h, normalization=:literal, n0_pivot=:mcut)
    nat = Lee2022NoConcentrationDMProfile(Omega_c=Oc, Omega_b=Ob, h=h, normalization=:xgpaint_ne2d, n0_pivot=:mcut)
    xgpnat = Lee2022XGPaintDMProfile(nat)
    lp = lee2022_parameters(lit, M, z)
    gp = XGPaint.get_params(xgpnat, M * XGPaint.M_sun, z)
    for F in Fs
        prof = XGPaint.BattagliaTauProfilePhysical(Omega_c=Oc, Omega_b=Ob, h=h,
            P0_amp=F * 200 * n0, P0_alpha_m=0.0, P0_alpha_z=0.0,
            x_c_amp=xc, x_c_alpha_m=0.0, x_c_alpha_z=0.0,
            alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0,
            beta_amp=bp + 0.3, beta_alpha_m=0.0, beta_alpha_z=0.0,
            gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)
        m200 = M * XGPaint.M_sun
        r200 = uconvert(u"m", XGPaint.R_Δ(prof, m200, z, 200))
        rhoc = uconvert(u"kg/m^3", XGPaint.ρ_crit(prof, z))
        E2 = ustrip(rhoc / XGPaint.ρ_crit(prof, 0.0))
        for x in xs
            rr = x * r200
            ne = ustrip(u"m^-2", XGPaint.ne2d(prof, rr, m200, z))
            # repository: literal projected column (pc cm^-3 -> m^-2) and XGPaint-native via XGPaint's own ne2d
            lit_pc = lee2022_projected_electron_column_pc_cm3(lit, x, M, z)
            lit_m2 = lit_pc / ustrip(u"pc*cm^-3", 1.0u"m^-2")
            theta = x * XGPaint.angular_size(xgpnat, XGPaint.R_Δ(xgpnat, m200, z, 200), z)
            nat_m2 = ustrip(u"m^-2", XGPaint.ne2d(xgpnat, theta, m200, z))
            lit3 = halo_electron_density_m3(lit, x, M, z)
            nat3 = halo_electron_density_m3(Battaglia16DensityDMProfile(xgpnat), x, M, z)
            @printf(io, "%s,%.6e,%.3f,%.3f,%.15e,%.15e,%.15f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n",
                cname, M, z, x, ustrip(r200), ustrip(rhoc), F, ne, lit_m2, nat_m2, lit3, nat3,
                lp.n0, lp.x_c, lp.beta_prime, gp.P₀, gp.β, E2)
        end
    end
end
close(io)

# pivot convention of the repository constructor default
open(joinpath(OUT, "step2_pivot_defaults.txt"), "w") do io2
    d = Lee2022NoConcentrationDMProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
    println(io2, "default_constructor_n0_pivot=", d.n0_pivot)
    m = Lee2022NoConcentrationDMProfile(Omega_c=0.261, Omega_b=0.049, h=0.68, n0_pivot=:mcut)
    for lmh in (13.3, 13.9)
        M = 10.0^lmh / 0.68
        pd = lee2022_parameters(d, M, 0.0); pm = lee2022_parameters(m, M, 0.0)
        println(io2, "log10(M h)=", lmh, " n0_default=", pd.n0, " n0_mcut=", pm.n0, " ratio_default_over_mcut=", pd.n0 / pm.n0)
    end
end
println("done")
