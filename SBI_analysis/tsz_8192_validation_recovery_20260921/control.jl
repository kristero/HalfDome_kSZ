# One isolated catalogue control, optionally with paired-resolution noise draws.
ENV["PREFLIGHT_LOAD_ONLY"]="1"
include(joinpath(@__DIR__,"fullsky_test.jl"))
using Dates
const PHASES=Dict{String,Float64}()

function timed_phase(f,key)
    t=time();value=f();PHASES[key]=get(PHASES,key,0.)+time()-t
    return value
end

# The custom spherical painter only touches m_hp/workspace. The alternative
# allocation removes the unused tmp_hp; mass and bin-map modes are prohibited.
if get(ENV,"VALIDATION_LEAN_MAPS","0")=="1"
    function init_visual_maps(cfg::VisualConfig)
        @assert !cfg.save_mass_map && !cfg.save_bin_maps
        map=HealpixMap{Float64,RingOrder}(cfg.nside);fill!(map.pixels,0.)
        workspace=XGPaint.HealpixRingProfileWorkspace{Float64}(Healpix.Resolution(cfg.nside))
        return (;m_hp=map,mass_hp=nothing,tmp_hp=nothing,batch_mass_hp=nothing,
            bin_y_hp=nothing,bin_mass_hp=nothing,workspace)
    end
end

function measured_signal(cfg)
    prefix="raw$(cfg.nside)_"
    model=timed_phase(prefix*"cache") do;build_visual_interpolator(cfg);end
    state=timed_phase(prefix*"allocation") do;init_visual_maps(cfg);end
    timed_phase(prefix*"catalogue_and_painting") do;run_halfdome_visuals!(cfg,state,model);end
    raw=state.m_hp
    alms=timed_phase(prefix*"map2alm") do
        Healpix.map2alm(raw;lmax=healpix_default_lmax(4096),niter=0)
    end
    timed_phase(prefix*"beam") do
        Healpix.almxfl!(alms,Healpix.gaussbeam(deg2rad(2/60),alms.lmax))
    end
    return timed_phase(prefix*"output_alm2map") do;Healpix.alm2map(alms,4096);end
end

function total_cross(signal_alm,a,b)
    return Healpix.alm2cl(signal_alm)+Healpix.alm2cl(signal_alm,a)+
        Healpix.alm2cl(signal_alm,b)+Healpix.alm2cl(a,b)
end

function main_control()
    cfg=load_halfdome_fullsky_so_noise_config()
    @assert cfg.base_cfg.nside in (4096,8192)
    @assert cfg.base_cfg.cl_lmax==7979 && cfg.base_cfg.cl_niter==0
    @assert cfg.fsky==.4 && cfg.mask_seed==12345
    @assert cfg.base_cfg.gaussian_beam_fwhm_arcmin==2.
    @assert cfg.mask_apodization_arcmin==60. && cfg.noise_deprojection==0 && cfg.noise_lmax==7979
    paired=get(ENV,"VALIDATION_PAIRED_4096","0")=="1"
    draws=parse(Int,get(ENV,"VALIDATION_NOISE_DRAWS","0"))
    task_id=parse(Int,ENV["VALIDATION_TASK_ID"])
    mask=timed_phase("mask") do
        random_apodized_cap_mask(4096,.4,cfg.mask_apodization_arcmin,MersenneTwister(12345))
    end
    reference_alm=nothing
    if paired
        @assert cfg.base_cfg.nside==8192
        replace_or_add_arg!(ARGS,"nside",4096)
        low_cfg=load_halfdome_fullsky_so_noise_config()
        low=measured_signal(low_cfg.base_cfg);low.pixels .*= mask.mask.pixels
        reference_alm=Healpix.map2alm(low;lmax=7979,niter=0)
        write_npy_float64_vector(joinpath(OUT,"paired4096_clean_cl.npy"),Healpix.alm2cl(reference_alm),"paired4096 clean")
        low=nothing;GC.gc()
        replace_or_add_arg!(ARGS,"nside",8192)
    end
    start=time();signal=measured_signal(cfg.base_cfg);GC.gc()
    write_npy_float64_vector(joinpath(OUT,"unmasked_clean_cl.npy"),compute_cl(cfg.base_cfg,signal),"unmasked clean")
    signal.pixels .*= mask.mask.pixels
    signal_alm=timed_phase("masked_signal_map2alm") do;Healpix.map2alm(signal;lmax=7979,niter=0);end
    clean=Healpix.alm2cl(signal_alm)
    @assert all(isfinite,clean) && all(clean .>=0)
    write_npy_float64_vector(joinpath(OUT,"masked_clean_cl.npy"),clean,"masked clean")
    noise_dir=joinpath(OUT,"noise");mkpath(noise_dir)
    ell,native=read_so_noise_native_cl(cfg.baseline_noise_path,7979;column=2,input_is_dl=false)
    noise_cl=so_noise_cl_vector_for_synalm(ell,native,7979)
    noise_records=Dict[]
    for draw in 0:draws-1
        target=joinpath(noise_dir,lpad(string(draw),3,'0')*".npy")
        record_path=target*".toml"
        if isfile(target) && isfile(record_path)
            push!(noise_records,TOML.parsefile(record_path));continue
        end
        tick=time();seeds=[62000000+task_id*10000+2draw+s for s in (1,2)]
        first=generate_gaussian_noise_map(noise_cl,4096,7979,MersenneTwister(seeds[1]))
        second=generate_gaussian_noise_map(noise_cl,4096,7979,MersenneTwister(seeds[2]))
        first_hash=pixel_sha(first);second_hash=pixel_sha(second)
        @assert first_hash!=second_hash
        first.pixels .*=mask.mask.pixels;second.pixels .*=mask.mask.pixels
        a=Healpix.map2alm(first;lmax=7979,niter=0)
        b=Healpix.map2alm(second;lmax=7979,niter=0)
        cross=total_cross(signal_alm,a,b)
        @assert all(isfinite,cross)
        parity=0.
        if draw==0
            first.pixels .+=signal.pixels;second.pixels .+=signal.pixels
            original=compute_cross_cl(cfg.base_cfg,first,second)
            parity=maximum(abs.(original-cross))/max(maximum(abs.(cross)),1e-300)
            @assert parity<1e-10
            write_npy_float64_vector(joinpath(OUT,"first_draw_map_route.npy"),original,"original map route")
        end
        write_npy_float64_vector(target*".tmp",cross,"noise draw")
        mv(target*".tmp",target;force=true)
        if paired
            value=total_cross(reference_alm,a,b)
            write_npy_float64_vector(target*".paired4096.npy",value,"paired4096 noise")
        end
        record=Dict("draw"=>draw,"split_seeds"=>seeds,"seconds"=>time()-tick,
            "noise1_sha256"=>first_hash,"noise2_sha256"=>second_hash,"map_alm_parity"=>parity)
        save_toml(record_path,record);push!(noise_records,record)
        first=nothing;second=nothing;a=nothing;b=nothing;GC.gc()
    end
    save_toml(joinpath(OUT,"control.toml"),Dict("phases"=>PHASES,"cache"=>CACHE_STATS,
        "current_signal_and_noise_seconds"=>time()-start,"raw_nside"=>cfg.base_cfg.nside,
        "output_nside"=>4096,"threads"=>Threads.nthreads(),"noise_draws"=>noise_records,
        "mask_sha256"=>pixel_sha(mask.mask),"noise_table_sha256"=>file_sha(cfg.baseline_noise_path),
        "lean_maps"=>get(ENV,"VALIDATION_LEAN_MAPS","0"),"beam_arcmin"=>2.,
        "noise_beam_applied"=>false,"split_Nell_multiplier"=>1.,"production_certified"=>false))
end
get(ENV,"CONTROL_LOAD_ONLY","0")=="1" || main_control()
