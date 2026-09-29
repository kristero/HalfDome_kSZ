# Read-only provenance probe: list the XGPaint source files (size, CRC32c) recorded in each precompile cache (.ji)
# of a depot, so the source a Julia session actually loaded can be matched to a git tree (git_tree_crc32c.jl).
# Usage: julia ji_source_hashes.jl <depot> [outfile]   (run with the Julia version that wrote the caches;
# caches of another version report "Incompatible header"). Put a scratch depot first in JULIA_DEPOT_PATH.
depot = length(ARGS) >= 1 ? ARGS[1] : "/home/kn18001/.julia"
io = length(ARGS) >= 2 ? open(ARGS[2], "a") : stdout
dir = joinpath(depot, "compiled", "v1.12", "XGPaint")
println(io, "# julia ", VERSION, " depot ", depot)
for f in sort(filter(endswith(".ji"), readdir(dir)))
    path = joinpath(dir, f)
    so = replace(path, ".ji" => ".so")
    println(io, "== ", f, "  so_mtime(compile time)=", isfile(so) ? Base.Libc.strftime("%Y-%m-%d %H:%M:%S", mtime(so)) : "?")
    try
        includes = Base.parse_cache_header(path)[2][1]
        for inc in includes
            occursin("XGPaint", inc.filename) || continue
            b = basename(inc.filename)
            b in ("XGPaint.jl", "model.jl", "profiles.jl", "profiles_y.jl", "profiles_tau.jl", "profiles_frb.jl", "util.jl", "healpixworkspace.jl") || continue
            println(io, "   ", rpad(b, 22), " fsize=", inc.fsize, " crc32c=", string(inc.hash, base=16))
        end
    catch err
        println(io, "   not readable by this Julia: ", first(sprint(showerror, err), 60))
    end
end
io === stdout || close(io)
