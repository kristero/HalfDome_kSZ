#!/usr/bin/env julia
# Both density families on identical pixel centres, with exact 3-D truncation.
# The paper-normalized Lee22 map is an exact constant multiple of the native map.
using XGPaint, Unitful, UnitfulAstro, Healpix, HDF5, Interpolations, TOML, Random, SHA
include(joinpath(@__DIR__, "..", "lee2022_frb_dm_profile.jl"))

const SIGMA_T = Float64(ustrip(uconvert(u"m^2", XGPaint.constants.ThomsonCrossSection)))
const PC_CM3_TO_M2 = 1 / M2_TO_PC_CM3_LOCAL

function models(cfg)
    c = cfg["cosmology"]
    lee = Lee2022NoConcentrationDMProfile(; Omega_c=c["Omega_c"],Omega_b=c["Omega_b"],h=c["h"],
        normalization=:xgpaint_ne2d,n0_pivot=:mcut,shape_clip_mass_msun=10.0^14.8/c["h"],redshift_scaling=:physical)
    return (Lee2022XGPaintDMProfile(lee),XGPaint.HaloDMProfile(;Omega_c=c["Omega_c"],Omega_b=c["Omega_b"],h=c["h"]))
end

"""Dimensionless shape integral averaged over a finite chord; u=b/(sphere*R200c)."""
function shape_mean(par,u,sphere)
    u == 1 && return XGPaint.generalized_nfw(sphere,par.xc,par.α,par.β,par.γ)
    value,_ = lee2022_quadgk_function()(t -> XGPaint.generalized_nfw(
        sphere*sqrt(u*u+(1-u*u)*t*t),par.xc,par.α,par.β,par.γ),0.,1.;rtol=2e-9)
    return value
end

function electron_amplitude(model,mass,z)
    par = XGPaint.get_params(model,mass*XGPaint.M_sun,z)
    rho = Float64(ustrip(uconvert(u"kg/m^3",XGPaint.ρ_crit(model,z))))
    radius = Float64(ustrip(uconvert(u"m",XGPaint.R_Δ(model,mass*XGPaint.M_sun,z,200))))
    return par, 0.9*par.P₀*model.f_b*rho/xgpaint_mass_per_electron_kg()*radius*SIGMA_T
end

function make_tables(cfg, bounds, model_pair, out)
    g=cfg["interpolation"]
    lu=range(log(g["radius_fraction_min"]),0.;length=g["n_radius"])
    lz=range(log1p(bounds["z_min"]),log1p(bounds["z_max"]);length=g["n_redshift"])
    lm=range(log10(bounds["mass_min_msun"]),log10(bounds["mass_max_msun"]);length=g["n_mass"])
    path=joinpath(out,"chord_shape_tables.h5")
    tables=[]
    for (name,model) in zip(("lee22","battaglia16"),model_pair)
        if isfile(path)
            values=h5open(path,"r") do f;read(f[name]);end
        else
            values=Array{Float64}(undef,length(lu),length(lz),length(lm))
            println("Tabulating $name dimensionless chord means: ",size(values));flush(stdout)
            Threads.@threads for im in eachindex(lm)
                for iz in eachindex(lz)
                    par=XGPaint.get_params(model,10.0^lm[im]*XGPaint.M_sun,expm1(lz[iz]))
                    for ir in eachindex(lu)
                        values[ir,iz,im]=log(shape_mean(par,exp(lu[ir]),cfg["sphere_r200c"]))
                    end
                end
            end
        end
        itp=Interpolations.scale(Interpolations.interpolate(values,BSpline(Cubic(Line(OnGrid())))),lu,lz,lm)
        rng=MersenneTwister(42022);worst=0.;tests=0
        for i in 1:800
            q=rand(rng,3)
            if i<=8;q=Float64.([(i-1)&1,((i-1)>>1)&1,((i-1)>>2)&1]);end
            ul=first(lu)+q[1]*(last(lu)-first(lu));zl=first(lz)+q[2]*(last(lz)-first(lz));ml=first(lm)+q[3]*(last(lm)-first(lm))
            # Stress both mass-law kinks, the grazing boundary, and nearby halos.
            if 9<=i<=80
                ml=clamp((isodd(i) ? 13.61 : 14.8)-log10(cfg["cosmology"]["h"])+(q[3]-.5)*.02,first(lm),last(lm))
                ul=log(0.97+q[1]*.03)
            end
            par=XGPaint.get_params(model,10.0^ml*XGPaint.M_sun,expm1(zl))
            direct=shape_mean(par,exp(ul),cfg["sphere_r200c"])
            worst=max(worst,abs(exp(itp(ul,zl,ml))/direct-1));tests+=1
        end
        println("$name: $tests checks, maximum relative interpolation error=$worst");flush(stdout)
        worst < g["validation_rtol"] || error("Chord table needs finer sampling")
        push!(tables,(itp=itp,values=values,error=worst,name=name))
    end
    if !isfile(path)
        h5open(path,"w") do f
            for table in tables;f[table.name]=table.values;end
            f["log_radius_fraction"]=collect(lu);f["log1p_z"]=collect(lz);f["log10_mass_msun"]=collect(lm)
            attributes(f)["sphere_r200c"]=cfg["sphere_r200c"]
        end
    end
    return (tables[1],tables[2])
end

"""Workers own disjoint HEALPix rings; four maps share the same geometry queries."""
function paint_chunk!(maps,workspace,tables,model_pair,mass,z,pos,vel,cfg)
    n=length(mass);theta=zeros(n);phi=zeros(n);sizes=zeros(n);vc=zeros(n)
    amp=zeros(2,n);starts=zeros(Int,n);stops=zeros(Int,n)
    logm=log10.(mass);logz=log1p.(z)
    restricted=(1e13 .<= mass) .& (mass .< 1e14) .& (z .<= 1)
    Threads.@threads for i in eachindex(mass)
        r=sqrt(sum(abs2,@view pos[:,i]));direction=pos[:,i]./r
        theta[i],phi[i]=Healpix.vec2ang(direction...)
        r200=XGPaint.R_Δ(model_pair[1],mass[i]*XGPaint.M_sun,z[i],200)
        sizes[i]=Float64(XGPaint.angular_size(model_pair[1],cfg["sphere_r200c"]*r200,z[i]))
        starts[i],stops[i]=XGPaint.get_relevant_rings(workspace.res,theta[i],sizes[i])
        vc[i]=-sum(direction.*vel[:,i])/299792.458
        for j in 1:2;_,amp[j,i]=electron_amplitude(model_pair[j],mass[i],z[i]);end
    end
    nrings=length(workspace.ring_first_pixels);workers=Threads.nthreads()
    rmin=cfg["interpolation"]["radius_fraction_min"]
    Threads.@threads :static for worker in 1:workers
        first_ring=fld((worker-1)*nrings,workers)+1;last_ring=fld(worker*nrings,workers)
        for i in eachindex(mass)
            lo=max(starts[i],first_ring);hi=min(stops[i],last_ring);lo>hi && continue
            x0,y0,z0=Healpix.ang2vec(theta[i],phi[i])
            for ring in lo:hi
                ranges=XGPaint.get_ring_disc_ranges(workspace,ring,theta[i],phi[i],sizes[i])
                first_pixel=workspace.ring_first_pixels[ring]
                for range in ranges,pixel in range
                    p=first_pixel+pixel-1
                    x1,y1,z1=Healpix.pix2vecRing(workspace.res,p)
                    d2=(x1-x0)^2+(y1-y0)^2+(z1-z0)^2
                    # asin from chord distance is stable for small angular separations.
                    angle=2asin(min(1.,sqrt(d2)/2));u=angle/sizes[i]
                    u<1 || continue
                    chord=2cfg["sphere_r200c"]*sqrt(max(0.,1-u*u))
                    logu=log(max(rmin,u))
                    for j in 1:2
                        signal=chord*amp[j,i]*exp(tables[j].itp(logu,logz[i],logm[i]))*vc[i]
                        maps[j].pixels[p]+=signal
                        restricted[i] && (maps[j+2].pixels[p]+=signal)
                    end
                end
            end
        end
    end
    return sum(restricted)
end

function verify_density_and_painter(cfg,model_pair,tables)
    # Independent physical chord routine checks factors of two, redshift, and sigma_T.
    worst=0.
    for model in model_pair,mass in (1e13,5e13,3e14,1e15),z in (.01,.5,2.,3.5),u in (.0001,.1,.7,.999)
        par,amp=electron_amplitude(model,mass,z)
        sphere=cfg["sphere_r200c"];half=sphere*sqrt(1-u*u)
        direct=chord_dm_pc_cm3(Battaglia16DensityDMProfile(model),sphere*u,half,mass,z)*(1+z)*PC_CM3_TO_M2*SIGMA_T
        actual=2half*amp*shape_mean(par,u,sphere)
        worst=max(worst,abs(actual/direct-1))
    end
    @assert worst<1e-7
    println("Independent physical chord maximum error=$worst")
    # Direct all-pixel summation checks seams, polar halos, overlap, and restricted routing.
    small=copy(cfg);small["nside"]=128
    maps=[HealpixMap{Float64,RingOrder}(128) for _ in 1:4]
    foreach(m->fill!(m.pixels,0.),maps);ws=XGPaint.HealpixRingProfileWorkspace(maps[1].resolution)
    mass=[5e13,8e13,2e14,7e13];zs=[.025,.04,.03,1.2]
    dirs=[(pi/2,.0001),(pi/2+.01,2pi-.001),(.008,1.),(pi-.008,4.)]
    pos=hcat([collect(Healpix.ang2vec(t,p)).*100 for (t,p) in dirs]...);vel=pos.*3
    paint_chunk!(maps,ws,tables,model_pair,mass,zs,pos,vel,small)
    direct=zeros(length(maps[1].pixels),4)
    for i in eachindex(mass)
        t,p=dirs[i];v=Healpix.ang2vec(t,p)
        radius=Float64(XGPaint.angular_size(model_pair[1],cfg["sphere_r200c"]*XGPaint.R_Δ(model_pair[1],mass[i]*XGPaint.M_sun,zs[i],200),zs[i]))
        for pixel in eachindex(maps[1].pixels)
            q=Healpix.pix2vecRing(ws.res,pixel);angle=2asin(min(1.,sqrt(sum((q[k]-v[k])^2 for k in 1:3))/2));u=angle/radius
            u<1 || continue
            for j in 1:2
                par,amp=electron_amplitude(model_pair[j],mass[i],zs[i])
                signal=2cfg["sphere_r200c"]*sqrt(1-u*u)*amp*shape_mean(par,u,cfg["sphere_r200c"])*(-300/299792.458)
                direct[pixel,j]+=signal
                mass[i]<1e14 && zs[i]<=1 && (direct[pixel,j+2]+=signal)
            end
        end
    end
    for j in 1:4
        relative=sqrt(sum(abs2,maps[j].pixels-direct[:,j])/sum(abs2,direct[:,j]))
        @assert relative < cfg["interpolation"]["validation_rtol"]
        println("Painter versus independent sum map$j relative L2=$relative")
    end
    return worst
end

function main()
    cfg=TOML.parsefile(ARGS[1]);bounds=TOML.parsefile(ARGS[2]);out=cfg["output"]
    model_pair=models(cfg);tables=make_tables(cfg,bounds,model_pair,out)
    chord_error=verify_density_and_painter(cfg,model_pair,tables)
    "--validate-only" in ARGS && return
    dest=joinpath(out,"ksz_maps.h5");isfile(dest) && error("Preserving completed maps")
    maps=[HealpixMap{Float64,RingOrder}(cfg["nside"]) for _ in 1:4]
    foreach(m->fill!(m.pixels,0.),maps);workspace=XGPaint.HealpixRingProfileWorkspace(maps[1].resolution)
    count=0;restricted_count=0;t0=time()
    h5open(cfg["catalog"],"r") do f
        for start in 1:cfg["chunk_size"]:bounds["total_catalog_rows"]
            stop=min(start+cfg["chunk_size"]-1,bounds["total_catalog_rows"]);rows=start:stop
            mass=Float64.(f["halo_mass_m200c"][rows])./cfg["cosmology"]["h"]
            z=Float64.(f["redshift"][rows]);pos=Float64.(f["Position"][:,rows]);vel=Float64.(f["Velocity"][:,rows])
            @assert all(isfinite,mass) && all(>(0),mass) && all(isfinite,z) && all(>(0),z) && all(isfinite,pos) && all(isfinite,vel)
            restricted_count+=paint_chunk!(maps,workspace,tables,model_pair,mass,z,pos,vel,cfg)
            count+=length(mass)
            println("Painted $count full / $restricted_count restricted; elapsed=$(round(time()-t0,digits=1)) s");flush(stdout)
        end
    end
    @assert count==bounds["total_catalog_rows"]
    h5open(dest*".partial","w") do f
        for (i,name) in enumerate(("full_lee22_previous","full_battaglia16","restricted_lee22_previous","restricted_battaglia16"))
            @assert all(isfinite,maps[i].pixels);f[name]=maps[i].pixels
        end
        attributes(f)["nside"]=cfg["nside"];attributes(f)["sphere_r200c"]=cfg["sphere_r200c"]
        attributes(f)["full_halos"]=count;attributes(f)["restricted_halos"]=restricted_count
        attributes(f)["lee22_paper_over_previous"]=1/lee2022_normalization_factor(model_pair[1].lee)
        attributes(f)["lee22_table_max_error"]=tables[1].error;attributes(f)["b16_table_max_error"]=tables[2].error
        attributes(f)["independent_chord_max_error"]=chord_error
        attributes(f)["complete"]=true;attributes(f)["ordering"]="RING"
        attributes(f)["profile_sha256"]=bytes2hex(sha256(read(joinpath(@__DIR__,"..","lee2022_frb_dm_profile.jl"))))
    end
    mv(dest*".partial",dest);println("Saved $dest")
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
