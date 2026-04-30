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

function phase_usage_stats(start_state; thread_capacity::Integer=nthreads())
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

function print_phase_usage(label::AbstractString, start_state; thread_capacity::Integer=nthreads())
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
