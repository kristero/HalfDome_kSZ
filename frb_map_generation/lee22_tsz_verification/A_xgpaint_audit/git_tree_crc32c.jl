# Read-only: CRC32c and size of XGPaint src files in the working tree and at selected git commits,
# for comparison with the hashes stored in the precompile caches (ji_source_hashes.jl).
using CRC32c
io = length(ARGS) >= 1 ? open(ARGS[1], "w") : stdout
repo = "/home/kn18001/.julia/dev/XGPaint"
files = ["profiles.jl", "profiles_y.jl", "model.jl", "XGPaint.jl", "profiles_tau.jl", "profiles_frb.jl", "util.jl", "healpixworkspace.jl"]
for rev in ["WORKTREE", "HEAD", "35f79ad", "07fc686", "5dd0b57", "ce1d239", "e67a8a1", "cbf5cbd", "b924ab1"]
    println(io, "== ", rev)
    for f in files
        data = try
            rev == "WORKTREE" ? read(joinpath(repo, "src", f)) : read(pipeline(`git -C $repo show $rev:src/$f`; stderr=devnull))
        catch
            println(io, "   ", rpad(f, 22), " (absent)"); continue
        end
        println(io, "   ", rpad(f, 22), " fsize=", length(data), " crc32c=", string(crc32c(data), base=16))
    end
end
io === stdout || close(io)
