# Load the real control entrypoint and exercise both allocation modes at NSIDE32.
ENV["CONTROL_LOAD_ONLY"]="1"
include(joinpath(@__DIR__,"control.jl"))
replace_or_add_arg!(ARGS,"nside",32)
replace_or_add_arg!(ARGS,"cl_lmax",63)
replace_or_add_arg!(ARGS,"so_noise_lmax",63)
replace_or_add_arg!(ARGS,"enforce_battaglia_guardrails",false)
cfg=load_halfdome_fullsky_so_noise_config()
state=init_visual_maps(cfg.base_cfg)
@assert length(state.m_hp.pixels)==12*32^2
model=ChordMeanProfile(build_tsz_model(cfg.base_cfg),4.)
struct DirectTestProfile{M,I}
    model::M
    itp::I
end
(profile::DirectTestProfile)(theta,mass,z)=profile.model(theta,mass,z)
profile=DirectTestProfile(model,(ranges=(LinRange(log(1e-11),log(1e5),512),),))
paint_visual_batch!(state,profile,[1.,0.],[0.,1.],[0.,0.],[1.,1.],[1e15,1e15],[.01,.01])
@assert any(state.m_hp.pixels .>0) && all(isfinite,state.m_hp.pixels)
write_npy_float64_vector(joinpath(OUT,"allocation_test_map.npy"),state.m_hp.pixels,"allocation parity test")
println("PASS small real-entrypoint painting; lean=",get(ENV,"VALIDATION_LEAN_MAPS","0"))
