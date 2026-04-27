function build_tsz_model(cfg::VisualConfig)
    p = cfg.battaglia_params
    return Battaglia16ThermalSZProfile(
        Omega_c=TSZ_OMEGAC,
        Omega_b=TSZ_OMEGAB,
        h=TSZ_H_VALUE,
        P0_amp=p.P0_amp,
        P0_alpha_m=p.P0_alpha_m,
        P0_alpha_z=p.P0_alpha_z,
        x_c_amp=p.x_c_amp,
        x_c_alpha_m=p.x_c_alpha_m,
        x_c_alpha_z=p.x_c_alpha_z,
        beta_amp=p.beta_amp,
        beta_alpha_m=p.beta_alpha_m,
        beta_alpha_z=p.beta_alpha_z,
        alpha_amp=p.alpha_amp,
        alpha_alpha_m=p.alpha_alpha_m,
        alpha_alpha_z=p.alpha_alpha_z,
        gamma_amp=p.gamma_amp,
        gamma_alpha_m=p.gamma_alpha_m,
        gamma_alpha_z=p.gamma_alpha_z
    )
end

function build_visual_interpolator(cfg::VisualConfig)
    model = build_tsz_model(cfg)
    cache_sim_tag = cfg.catalog_source == "halfdome" ? "HalfDome" : "Websky"
    y_cache_file = joinpath(repo_root(), "cached_tSZ_$(cache_sim_tag)_cosmo_$(cfg.param_tag).jld2")

    if cfg.model_exists
        return build_interpolator(
            model,
            cache_file=y_cache_file,
            overwrite=false
        )
    end

    return build_interpolator(
        model;
        cache_file=y_cache_file,
        pad=256,
        logM_max=15.7,
        overwrite=true,
        verbose=true
    )
end
