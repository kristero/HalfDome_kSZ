using XGPaint, Healpix, Interpolations

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "cosmology_helpers.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "output.jl"))
include(joinpath(@__DIR__, "painting.jl"))
include(joinpath(@__DIR__, "model.jl"))

Base.@kwdef struct SingleProfileConfig
    base_cfg::VisualConfig
    mass_msun::Float64
    redshift::Float64
    ra_deg::Float64
    dec_deg::Float64
    radius_comoving_mpc::Float64
    radius_arcmin::Float64
    profile_tag::String
    y_output_path::String
    mass_output_path::String
    cl_output_path::String
end

function default_single_profile_tag(
    mass_msun::Real,
    redshift::Real,
    radius_arcmin::Real,
    ra_deg::Real,
    dec_deg::Real
)
    return join(
        (
            "single_tsz_profile",
            "m$(fmt_param_value(Float64(mass_msun)))",
            "z$(fmt_param_value(Float64(redshift)))",
            "r$(fmt_param_value(Float64(radius_arcmin)))arcmin",
            "ra$(fmt_param_value(Float64(ra_deg)))",
            "dec$(fmt_param_value(Float64(dec_deg)))"
        ),
        "_"
    )
end

function load_single_profile_config()
    base_cfg = load_visual_config()

    mass_msun = get_float_arg("mass_msun", 1.0e14; env="TSZ_SINGLE_MASS_MSUN")
    redshift = get_float_arg("redshift", 0.5; env="TSZ_SINGLE_REDSHIFT")
    ra_deg = get_float_arg("ra_deg", 0.0; env="TSZ_SINGLE_RA_DEG")
    dec_deg = get_float_arg("dec_deg", 0.0; env="TSZ_SINGLE_DEC_DEG")
    radius_comoving_mpc = get_float_arg("radius_comoving_mpc", 1.0; env="TSZ_SINGLE_RADIUS_COMOVING_MPC")
    radius_arcmin = get_float_arg("radius_arcmin", NaN; env="TSZ_SINGLE_RADIUS_ARCMIN")

    isfinite(mass_msun) && mass_msun > 0.0 || error("mass_msun must be positive.")
    isfinite(redshift) && redshift >= 0.0 || error("redshift must be non-negative.")
    isfinite(ra_deg) || error("ra_deg must be finite.")
    isfinite(dec_deg) && abs(dec_deg) <= 90.0 || error("dec_deg must lie in [-90, 90] degrees.")
    isfinite(radius_comoving_mpc) && radius_comoving_mpc > 0.0 || error("radius_comoving_mpc must be positive.")

    if !(isfinite(radius_arcmin) && radius_arcmin > 0.0)
        chi_of_z = make_chi_of_z_itp(omegam=base_cfg.cosmo_omegam, h_value=base_cfg.cosmo_h)
        chi_comoving_mpc = chi_of_z(redshift)
        radius_rad = radius_to_angular_extent(radius_comoving_mpc, chi_comoving_mpc)
        radius_arcmin = rad2deg(radius_rad) * 60.0
    end

    radius_arcmin > 0.0 || error("radius_arcmin must evaluate to a positive value.")

    default_tag = default_single_profile_tag(mass_msun, redshift, radius_arcmin, ra_deg, dec_deg)
    profile_tag = safe_filename_tag(
        get_string_arg("profile_tag", default_tag; env="TSZ_SINGLE_PROFILE_TAG")
    )

    y_output_path = joinpath(
        base_cfg.output_dir,
        "$(profile_tag)_tSZ_nside$(base_cfg.nside)_$(base_cfg.param_tag)_$(base_cfg.cosmology_tag)_$(base_cfg.beam_tag).fits"
    )
    mass_output_path = joinpath(
        base_cfg.output_dir,
        "$(profile_tag)_mass_nside$(base_cfg.nside)_$(base_cfg.param_tag)_$(base_cfg.cosmology_tag)_$(base_cfg.beam_tag).fits"
    )
    cl_output_path = joinpath(
        base_cfg.output_dir,
        "$(profile_tag)_tSZ_cl_nside$(base_cfg.nside)_$(base_cfg.param_tag)_$(base_cfg.cosmology_tag)_$(base_cfg.beam_tag).fits"
    )

    return SingleProfileConfig(
        base_cfg=base_cfg,
        mass_msun=mass_msun,
        redshift=redshift,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_comoving_mpc=radius_comoving_mpc,
        radius_arcmin=radius_arcmin,
        profile_tag=profile_tag,
        y_output_path=y_output_path,
        mass_output_path=mass_output_path,
        cl_output_path=cl_output_path
    )
end

function print_single_profile_config(cfg::SingleProfileConfig)
    println("Running a single-halo tSZ profile with fiducial Battaglia defaults unless overridden.")
    println("Output directory: $(cfg.base_cfg.output_dir)")
    println("Battaglia parameter tag: $(cfg.base_cfg.param_tag)")
    println("Cosmology tag: $(cfg.base_cfg.cosmology_tag)")
    println("Cosmology: h=$(cfg.base_cfg.cosmo_h), Omega_b=$(cfg.base_cfg.cosmo_omegab), Omega_c=$(cfg.base_cfg.cosmo_omegac), Omega_m=$(cfg.base_cfg.cosmo_omegam)")
    println("Gaussian beam: apply=$(cfg.base_cfg.apply_gaussian_beam), fwhm_arcmin=$(cfg.base_cfg.gaussian_beam_fwhm_arcmin)")
    println("NSIDE: $(cfg.base_cfg.nside)")
    println("Halo mass M200c [Msun]: $(cfg.mass_msun)")
    println("Halo redshift: $(cfg.redshift)")
    println("Sky position [deg]: RA=$(cfg.ra_deg), Dec=$(cfg.dec_deg)")
    println("Mass-map radius [comoving Mpc]: $(cfg.radius_comoving_mpc)")
    println("Mass-map radius [arcmin]: $(cfg.radius_arcmin)")
    println("y-map output: $(abspath(cfg.y_output_path))")
    if cfg.base_cfg.save_mass_map
        println("mass-map output: $(abspath(cfg.mass_output_path))")
    end
    println("C_l output: $(abspath(cfg.cl_output_path))")
end

function save_single_profile_maps!(cfg::SingleProfileConfig, state)
    ensure_output_dir(cfg.base_cfg)
    Healpix.saveToFITS(state.m_hp, "!" * cfg.y_output_path, typechar="D")
    println("Saved tSZ map to $(abspath(cfg.y_output_path))")
    if cfg.base_cfg.save_mass_map
        Healpix.saveToFITS(state.mass_hp, "!" * cfg.mass_output_path, typechar="D")
        println("Saved mass map to $(abspath(cfg.mass_output_path))")
    end
    return nothing
end

function run_single_tsz_profile()
    t0 = time()
    cfg = load_single_profile_config()

    print_single_profile_config(cfg)

    y_model_interp = build_visual_interpolator(cfg.base_cfg)
    state = init_visual_maps(cfg.base_cfg)

    ra = deg2rad(cfg.ra_deg)
    dec = deg2rad(cfg.dec_deg)
    radius_rad = deg2rad(cfg.radius_arcmin / 60.0)

    paint!(state.m_hp, state.workspace, y_model_interp, [cfg.mass_msun], [cfg.redshift], [ra], [dec])
    if state.mass_hp !== nothing
        build_halo_mass_map!(state.mass_hp, state.workspace, [ra], [dec], [cfg.mass_msun], [radius_rad])
    end

    if cfg.base_cfg.apply_gaussian_beam
        println("Applying Gaussian beam to single-profile tSZ map with FWHM=$(cfg.base_cfg.gaussian_beam_fwhm_arcmin) arcmin.")
    end
    output_y_map = prepare_tsz_map_for_output(cfg.base_cfg, state.m_hp)
    save_single_profile_maps!(cfg, (m_hp=output_y_map, mass_hp=state.mass_hp))

    cl = anafast(output_y_map, niter=0)
    write_cl_fits_overwrite(cfg.cl_output_path, cl)

    elapsed = time() - t0
    println("Finished single-halo tSZ profile generation in $(round(elapsed; digits=2)) s.")
    return (cfg=cfg, state=state, cl=cl)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_single_tsz_profile()
end
