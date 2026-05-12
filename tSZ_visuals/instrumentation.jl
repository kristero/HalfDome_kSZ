import Pkg

struct ProcessCpuClockTimespec
    tv_sec::Clong
    tv_nsec::Clong
end

function process_cpu_time_seconds()
    if Sys.islinux()
        ts = Ref{ProcessCpuClockTimespec}()
        ret = ccall(:clock_gettime, Cint, (Cint, Ref{ProcessCpuClockTimespec}), 2, ts)
        if ret == 0
            return Float64(ts[].tv_sec) + 1.0e-9 * Float64(ts[].tv_nsec)
        end
    end
    return NaN
end

function start_phase_timing()
    return (wall=time(), cpu=process_cpu_time_seconds())
end

function trim_process_memory()
    GC.gc()
    if Sys.islinux()
        try
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
        catch err
            println("malloc_trim unavailable ($(typeof(err))); continuing after GC.")
        end
    end
    return nothing
end

function phase_usage_stats(start_state; thread_capacity::Integer=Base.Threads.nthreads())
    wall_elapsed = max(time() - start_state.wall, 0.0)
    cpu_end = process_cpu_time_seconds()
    cpu_elapsed = if isfinite(start_state.cpu) && isfinite(cpu_end)
        max(cpu_end - start_state.cpu, 0.0)
    else
        NaN
    end

    avg_cpus_used = if wall_elapsed > 0.0 && isfinite(cpu_elapsed)
        cpu_elapsed / wall_elapsed
    else
        NaN
    end

    capacity = max(thread_capacity, 1)
    efficiency_pct = if isfinite(avg_cpus_used)
        100.0 * avg_cpus_used / capacity
    else
        NaN
    end

    return (
        wall_elapsed=wall_elapsed,
        cpu_elapsed=cpu_elapsed,
        avg_cpus_used=avg_cpus_used,
        thread_capacity=capacity,
        efficiency_pct=efficiency_pct
    )
end

function print_phase_usage(label::AbstractString, start_state; thread_capacity::Integer=Base.Threads.nthreads())
    stats = phase_usage_stats(start_state; thread_capacity=thread_capacity)
    if isfinite(stats.cpu_elapsed)
        println(
            "$(label) usage: wall=$(round(stats.wall_elapsed; digits=2)) s, " *
            "cpu=$(round(stats.cpu_elapsed; digits=2)) s, " *
            "avg_cpus=$(round(stats.avg_cpus_used; digits=2))/$(stats.thread_capacity), " *
            "efficiency=$(round(stats.efficiency_pct; digits=1))%"
        )
    else
        println(
            "$(label) usage: wall=$(round(stats.wall_elapsed; digits=2)) s " *
            "(process CPU time unavailable on this platform)"
        )
    end
    return stats
end

function print_module_path(module_name::AbstractString)
    modsym = Symbol(module_name)
    if isdefined(Main, modsym)
        mod = getfield(Main, modsym)
        println("Julia module path $(module_name): $(pathof(mod))")
    else
        package_path = Base.find_package(module_name)
        package_path === nothing || println("Julia package path $(module_name): $(package_path)")
    end
    return nothing
end

function package_info_field(info, field::Symbol, default)
    return field in propertynames(info) ? getproperty(info, field) : default
end

function print_package_version(package_name::AbstractString)
    try
        matches = filter(
            info -> package_info_field(info, :name, "") == package_name,
            collect(values(Pkg.dependencies()))
        )
        if isempty(matches)
            println("Julia package version $(package_name): <not found in active manifest>")
        else
            for info in matches
                version = package_info_field(info, :version, nothing)
                source = package_info_field(info, :source, nothing)
                version_text = version === nothing ? "<dev/unversioned>" : string(version)
                source_text = source === nothing ? "<unknown source>" : string(source)
                println("Julia package version $(package_name): $(version_text), source=$(source_text)")
            end
        end
    catch err
        println("Julia package version $(package_name): unavailable ($(typeof(err)))")
    end
    return nothing
end

function print_runtime_environment()
    println("Julia version: $(VERSION)")
    println("Julia bindir: $(Sys.BINDIR)")
    println("Julia executable path: $(joinpath(Sys.BINDIR, Base.julia_exename()))")
    println("Julia threads available: $(Base.Threads.nthreads())")
    println("Visible CPU threads: $(Sys.CPU_THREADS)")
    println("Machine: $(Sys.MACHINE)")
    println("JULIA_NUM_THREADS env: $(get(ENV, "JULIA_NUM_THREADS", "<unset>"))")
    println("JULIA_DEPOT_PATH env: $(get(ENV, "JULIA_DEPOT_PATH", "<default>"))")
    println("LOAD_PATH: $(join(LOAD_PATH, ":"))")
    for module_name in ("XGPaint", "QuadGK", "Healpix", "HDF5", "Interpolations")
        print_module_path(module_name)
    end
    for package_name in ("XGPaint", "QuadGK", "Healpix", "HDF5", "Interpolations")
        print_package_version(package_name)
    end
    return nothing
end
