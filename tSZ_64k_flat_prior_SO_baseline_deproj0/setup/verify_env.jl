# Check the instantiated julia_env against the runtime of the validated 256-row test:
# Julia version, key package versions, and the SHA-256 of every XGPaint source file.
# Run with: julia --startup-file=no --project=julia_env setup/verify_env.jl
using Pkg, SHA, TOML

const EXPECTED = TOML.parsefile(joinpath(@__DIR__, "validated_runtime.toml"))

function main()
    failures = String[]
    julia_expected = VersionNumber(EXPECTED["julia_version"])
    VERSION == julia_expected || push!(failures, "Julia $(VERSION), expected $(julia_expected)")
    packages = Dict(info.name => info for info in values(Pkg.dependencies()))
    for (name, version) in sort(collect(EXPECTED["package_versions"]))
        found = haskey(packages, name) ? string(packages[name].version) : "missing"
        found == version || push!(failures, "$(name) $(found), expected $(version)")
    end
    xgpaint = get(packages, "XGPaint", nothing)
    if xgpaint === nothing
        push!(failures, "XGPaint is not installed; run Pkg.instantiate()")
    else
        for (relative, digest) in sort(collect(EXPECTED["xgpaint_sha256"]))
            path = joinpath(xgpaint.source, relative)
            found = isfile(path) ? bytes2hex(open(sha256, path)) : "missing"
            found == digest || push!(failures, "XGPaint/$(relative): $(found), expected $(digest)")
        end
        println("XGPaint ", xgpaint.version, " from ", something(xgpaint.git_source, xgpaint.source),
                ", tree ", xgpaint.tree_hash, ", installed at ", xgpaint.source)
    end
    if isempty(failures)
        println("verify_env: OK - Julia ", VERSION, ", ", length(EXPECTED["package_versions"]),
                " package versions and ", length(EXPECTED["xgpaint_sha256"]),
                " XGPaint files match the validated 256-row test")
    else
        foreach(message -> println(stderr, "verify_env FAILED: ", message), failures)
        exit(1)
    end
end

main()
