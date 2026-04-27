if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

using HDF5, Interpolations, Plots
using Random
using Statistics
using Base.Threads

const C_KMS = 299_792.458
const OMEGA_B = 0.049
const OMEGA_M = 0.31
const OMEGA_C = OMEGA_M - OMEGA_B
const H_VALUE = 0.68

function get_int_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return parse(Int, ENV[env])
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse(Int, split(a, "=", limit=2)[2])
        end
    end
    return Int(default)
end

function get_float_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return parse(Float64, ENV[env])
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse(Float64, split(a, "=", limit=2)[2])
        end
    end
    return Float64(default)
end

function get_string_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return ENV[env]
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return split(a, "=", limit=2)[2]
        elseif startswith(a, prefix2)
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

function parse_bool_arg(value)
    value_norm = lowercase(strip(String(value)))
    if value_norm in ("1", "true", "t", "yes", "y", "on")
        return true
    elseif value_norm in ("0", "false", "f", "no", "n", "off")
        return false
    end
    error("Could not parse boolean value $(repr(value)). Use true/false, yes/no, on/off, or 1/0.")
end

function get_bool_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return parse_bool_arg(ENV[env])
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse_bool_arg(split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse_bool_arg(split(a, "=", limit=2)[2])
        end
    end
    return Bool(default)
end

function fmt_param_value(x)
    s = string(x)
    s = replace(s, "-" => "m")
    s = replace(s, "." => "p")
    s = replace(s, "+" => "")
    return s
end

function save_plot_accessible(plot_obj, preferred_path::AbstractString)
    preferred_dir = dirname(preferred_path)
    isdir(preferred_dir) || mkpath(preferred_dir)

    savefig(plot_obj, preferred_path)

    if isfile(preferred_path) && filesize(preferred_path) > 0
        return preferred_path
    end

    error("Plot save completed without producing a readable file at $(preferred_path).")
end

catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env="FRB_DM_CATALOG_SOURCE"))
catalog_source in ("halfdome", "websky") || error("Unsupported catalog_source=$(repr(catalog_source)). Use \"halfdome\" or \"websky\".")

halfdome_path = get_string_arg("halfdome_path", "lightcone_100.hdf5"; env="FRB_DM_HALFDOME_PATH")
websky_path = get_string_arg("websky_path", "halos-light.pksc"; env="FRB_DM_WEBSKY_PATH")
chunkN = get_int_arg("chunkN", 1_000_000; env="FRB_REDSHIFT_TEST_CHUNKN")
frb_count = get_int_arg("frb_count", 30_000; env="FRB_COUNT")
frb_seed = get_int_arg("frb_seed", 12345; env="FRB_SEED")
apply_mass_cut = get_bool_arg("apply_mass_cut", true; env="FRB_REDSHIFT_TEST_APPLY_MASS_CUT")
mass_min = get_float_arg("mass_min", 1.0e13; env="FRB_REDSHIFT_TEST_MASS_MIN")
frb_z_cut = get_float_arg("frb_z_cut", 0.5; env="FRB_Z_CUT")
test_z_min = get_float_arg("test_z_min", 0.0; env="FRB_REDSHIFT_TEST_Z_MIN")
test_z_max = get_float_arg("test_z_max", 2.0; env="FRB_REDSHIFT_TEST_Z_MAX")
display_bin_count = get_int_arg("display_bin_count", 13; env="FRB_REDSHIFT_TEST_DISPLAY_BINS")
candidate_pdf_bin_count = get_int_arg("candidate_pdf_bin_count", 200; env="FRB_REDSHIFT_TEST_CANDIDATE_PDF_BINS")
save_plot = get_bool_arg("save_plot", true; env="FRB_REDSHIFT_TEST_SAVE_PLOT")

frb_count > 0 || error("frb_count must be positive.")
mass_min > 0.0 || error("mass_min must be positive.")
test_z_max > test_z_min || error("test_z_max must be greater than test_z_min.")
display_bin_count > 0 || error("display_bin_count must be positive.")
candidate_pdf_bin_count > 1 || error("candidate_pdf_bin_count must be at least 2.")
frb_z_cut > 0.0 || error("frb_z_cut must be positive.")

function make_chi_and_z_of_chi_itp(; omegam, h_value, z1=0.0, z2=6.0, nz=100_000)
    H0 = 100 * h_value
    H(z) = H0 * sqrt(omegam * (1 + z)^3 + 1 - omegam)
    dchidz(z) = C_KMS / H(z)

    za = collect(range(z1, z2; length=nz))
    dz = za[2] - za[1]
    chia = similar(za)

    chia[1] = 0.0
    s = 0.0
    @inbounds for i in 2:length(za)
        s += 0.5 * (dchidz(za[i - 1]) + dchidz(za[i])) * dz
        chia[i] = s
    end

    chi_of_z_itp = linear_interpolation(za, chia; extrapolation_bc=Line())
    z_of_chi_itp = linear_interpolation(chia, za; extrapolation_bc=Line())
    return chi_of_z_itp, z_of_chi_itp
end

const H0 = 100 * H_VALUE
@inline H_of_z(z::Float64) = H0 * sqrt(OMEGA_M * (1.0 + z)^3 + 1.0 - OMEGA_M)
@inline luminosity_distance(z::Float64, chi_of_z_itp) = (1.0 + z) * chi_of_z_itp(z)

# Munoz & Loeb (2018), Eq. (24): dN_FRB/dz is proportional to
# c * chi(z)^2 / ((1+z) H(z)) * exp(-d_L(z)^2 / d_L(z_cut)^2)
function paper_frb_redshift_pdf_weight(z::Float64, chi_of_z_itp, d_l_cut_sq::Float64)
    chi = chi_of_z_itp(z)
    d_l = luminosity_distance(z, chi_of_z_itp)
    return C_KMS * chi^2 / ((1.0 + z) * H_of_z(z)) * exp(-(d_l * d_l) / d_l_cut_sq)
end

function normalize_pdf_grid!(z_grid, pdf_values)
    integral = 0.0
    @inbounds for i in 1:length(z_grid)-1
        integral += 0.5 * (pdf_values[i] + pdf_values[i + 1]) * (z_grid[i + 1] - z_grid[i])
    end
    integral > 0.0 || error("PDF normalization must be positive.")
    pdf_values ./= integral
    return pdf_values
end

function evaluate_normalized_paper_pdf(z_grid, chi_of_z_itp, d_l_cut_sq, z_max)
    pdf_values = Vector{Float64}(undef, length(z_grid))
    @inbounds for i in eachindex(z_grid)
        z = Float64(z_grid[i])
        pdf_values[i] = (0.0 <= z <= z_max) ? paper_frb_redshift_pdf_weight(z, chi_of_z_itp, d_l_cut_sq) : 0.0
    end
    return normalize_pdf_grid!(z_grid, pdf_values)
end

function histogram_counts(values, bin_edges)
    counts = zeros(Int, length(bin_edges) - 1)
    @inbounds for value in values
        if value < first(bin_edges) || value > last(bin_edges)
            continue
        end
        bin_idx = searchsortedlast(bin_edges, value)
        if bin_idx == 0
            continue
        elseif bin_idx >= length(bin_edges)
            bin_idx = length(bin_edges) - 1
        end
        counts[bin_idx] += 1
    end
    return counts
end

function pdf_from_histogram_counts(counts, bin_edges)
    total_count = sum(counts)
    total_count > 0 || error("Histogram counts must sum to a positive value.")
    pdf = Vector{Float64}(undef, length(counts))
    @inbounds for i in eachindex(counts)
        bin_width = bin_edges[i + 1] - bin_edges[i]
        pdf[i] = counts[i] / (total_count * bin_width)
    end
    return pdf
end

function evaluate_histogram_pdf(z_grid, bin_edges, bin_pdf)
    pdf_values = zeros(Float64, length(z_grid))
    @inbounds for i in eachindex(z_grid)
        z = z_grid[i]
        if z < first(bin_edges) || z > last(bin_edges)
            continue
        end
        bin_idx = searchsortedlast(bin_edges, z)
        if bin_idx == 0
            continue
        elseif bin_idx >= length(bin_edges)
            bin_idx = length(bin_edges) - 1
        end
        pdf_values[i] = bin_pdf[bin_idx]
    end
    return pdf_values
end

function bin_probabilities_from_pdf(z_grid, pdf_values, bin_edges)
    probabilities = zeros(Float64, length(bin_edges) - 1)
    @inbounds for i in 1:length(z_grid)-1
        z_left = z_grid[i]
        z_right = z_grid[i + 1]
        z_mid = 0.5 * (z_left + z_right)
        if z_mid < first(bin_edges) || z_mid > last(bin_edges)
            continue
        end
        bin_idx = searchsortedlast(bin_edges, z_mid)
        if bin_idx == 0
            continue
        elseif bin_idx >= length(bin_edges)
            bin_idx = length(bin_edges) - 1
        end
        probabilities[bin_idx] += 0.5 * (pdf_values[i] + pdf_values[i + 1]) * (z_right - z_left)
    end
    total_probability = sum(probabilities)
    total_probability > 0.0 || error("Binned PDF probability must be positive.")
    probabilities ./= total_probability
    return probabilities
end

function summarize_match(label::AbstractString, sampled_redshifts, target_bin_probabilities, display_bin_edges)
    counts = histogram_counts(sampled_redshifts, display_bin_edges)
    observed_probabilities = counts ./ sum(counts)
    l1_distance = sum(abs.(observed_probabilities .- target_bin_probabilities))
    expected_counts = length(sampled_redshifts) .* target_bin_probabilities

    chi_square = 0.0
    @inbounds for i in eachindex(counts)
        if expected_counts[i] > 0.0
            chi_square += (counts[i] - expected_counts[i])^2 / expected_counts[i]
        end
    end

    println("$(label):")
    println("  sample_count=$(length(sampled_redshifts))")
    println("  L1 distance to target bin probabilities = $(round(l1_distance; digits=4))")
    println("  chi-square over $(length(counts)) bins = $(round(chi_square; digits=2))")
    return counts
end

function draw_weighted_sample_positions(rng, weights::AbstractVector{<:Real}, sample_count::Int)
    positive_count = count(>(0.0), weights)
    sample_count <= positive_count || error(
        "sample_count=$(sample_count) exceeds the number of positive-weight candidates=$(positive_count)."
    )

    if sample_count == length(weights)
        return collect(eachindex(weights))
    end

    selection_keys = Vector{Float64}(undef, length(weights))
    @inbounds for i in eachindex(weights)
        weight = Float64(weights[i])
        if weight > 0.0
            selection_keys[i] = randexp(rng) / weight
        else
            selection_keys[i] = Inf
        end
    end

    return partialsortperm(selection_keys, 1:sample_count)
end

function sample_redshifts_from_pdf(rng, z_grid, pdf_values, sample_count::Int)
    cdf = similar(pdf_values)
    cdf[1] = 0.0
    @inbounds for i in 2:length(z_grid)
        dz = z_grid[i] - z_grid[i - 1]
        cdf[i] = cdf[i - 1] + 0.5 * (pdf_values[i] + pdf_values[i - 1]) * dz
    end
    cdf ./= cdf[end]

    samples = Vector{Float64}(undef, sample_count)
    @inbounds for i in 1:sample_count
        u = rand(rng)
        idx = searchsortedfirst(cdf, u)
        if idx <= 1
            samples[i] = z_grid[1]
        elseif idx > length(z_grid)
            samples[i] = z_grid[end]
        else
            cdf_left = cdf[idx - 1]
            cdf_right = cdf[idx]
            if cdf_right <= cdf_left
                samples[i] = z_grid[idx]
            else
                t = (u - cdf_left) / (cdf_right - cdf_left)
                samples[i] = z_grid[idx - 1] + t * (z_grid[idx] - z_grid[idx - 1])
            end
        end
    end

    return samples
end

@inline function m200m_to_m200c_scalar(m200m::Float64, z::Float64)
    one_plus_z = 1.0 + z
    ez_num = OMEGA_M * one_plus_z^3
    omegamz = ez_num / (ez_num + 1.0 - OMEGA_M)
    return m200m * omegamz^0.35
end

function compute_redshift_and_mass(x, y, z, R, itp_z_of_chi, rho_m)
    n = length(x)
    redshift = Vector{Float64}(undef, n)
    halo_mass = Vector{Float64}(undef, n)
    mass_prefactor = (4.0 * pi / 3.0) * rho_m

    @threads for i in 1:n
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        ri = Float64(R[i])

        chi = sqrt(xi * xi + yi * yi + zi * zi)
        zi_redshift = itp_z_of_chi(chi)

        redshift[i] = zi_redshift
        halo_mass[i] = m200m_to_m200c_scalar(mass_prefactor * ri^3, zi_redshift)
    end

    return redshift, halo_mass
end

function stream_catalog_batches(
    process_batch!::F,
    catalog_source::AbstractString,
    halfdome_path::AbstractString,
    websky_path::AbstractString,
    chunkN::Int,
    itp_z_of_chi,
    rho_m
) where {F}
    if catalog_source == "halfdome"
        return h5open(halfdome_path, "r") do h5
            pos_ds = h5["Position"]
            mass_ds = h5["halo_mass_m200c"]
            redshift_ds = h5["redshift"]

            total_halo_count = size(pos_ds, 2)
            for batch_start_halo_index in 1:chunkN:total_halo_count
                batch_end_halo_index = min(batch_start_halo_index + chunkN - 1, total_halo_count)
                idx = batch_start_halo_index:batch_end_halo_index

                pos = pos_ds[:, idx]
                halo_mass = vec(mass_ds[idx])
                redshift = vec(redshift_ds[idx])
                x = @view pos[1, :]
                y = @view pos[2, :]
                z = @view pos[3, :]

                process_batch!(batch_start_halo_index, x, y, z, halo_mass, redshift)
            end

            total_halo_count
        end
    end

    open(websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        _ = read(io, Float32)
        _ = read(io, Float32)

        buf = Matrix{Float32}(undef, 10, chunkN)
        batch_start_halo_index = 1
        nleft = total_halo_count

        while nleft > 0
            nthis = min(chunkN, nleft)

            rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
            read!(io, rawview)

            cat = @view buf[:, 1:nthis]
            x = @view cat[1, :]
            y = @view cat[2, :]
            z = @view cat[3, :]
            R = @view cat[7, :]

            redshift, halo_mass = compute_redshift_and_mass(x, y, z, R, itp_z_of_chi, rho_m)
            process_batch!(batch_start_halo_index, x, y, z, halo_mass, redshift)

            batch_start_halo_index += nthis
            nleft -= nthis
        end

        total_halo_count
    end
end

function collect_eligible_redshifts(
    catalog_source,
    halfdome_path,
    websky_path,
    chunkN,
    itp_z_of_chi,
    rho_m;
    apply_mass_cut,
    mass_min,
    z_min,
    z_max
)
    eligible_redshifts = Float64[]
    total_halo_count = stream_catalog_batches(
        catalog_source,
        halfdome_path,
        websky_path,
        chunkN,
        itp_z_of_chi,
        rho_m
    ) do _batch_start_halo_index, _x, _y, _z, halo_mass, redshift
        @inbounds for i in eachindex(redshift)
            if apply_mass_cut && halo_mass[i] < mass_min
                continue
            end

            halo_redshift = Float64(redshift[i])
            if !isfinite(halo_redshift) || halo_redshift < z_min || halo_redshift > z_max
                continue
            end

            push!(eligible_redshifts, halo_redshift)
        end
    end

    return eligible_redshifts, total_halo_count
end

chi_of_z_itp, itp_z_of_chi = make_chi_and_z_of_chi_itp(omegam=OMEGA_M, h_value=H_VALUE)
rho_m = 2.775e11 * OMEGA_M * H_VALUE^2
d_l_cut = luminosity_distance(frb_z_cut, chi_of_z_itp)
d_l_cut_sq = d_l_cut^2

display_bin_edges = collect(range(test_z_min, test_z_max; length=display_bin_count + 1))
candidate_pdf_bin_edges = collect(range(test_z_min, test_z_max; length=candidate_pdf_bin_count + 1))
plot_z_grid = collect(range(test_z_min, test_z_max; length=4_001))
display_bin_width = display_bin_edges[2] - display_bin_edges[1]

paper_pdf_values = evaluate_normalized_paper_pdf(plot_z_grid, chi_of_z_itp, d_l_cut_sq, test_z_max)
paper_display_bin_probabilities = bin_probabilities_from_pdf(plot_z_grid, paper_pdf_values, display_bin_edges)
paper_curve_counts = frb_count .* paper_pdf_values .* display_bin_width

rng = MersenneTwister(frb_seed)
paper_direct_redshifts = sample_redshifts_from_pdf(rng, plot_z_grid, paper_pdf_values, frb_count)

println("FRB redshift test configuration:")
println("  catalog_source=$(catalog_source)")
println("  frb_count=$(frb_count), frb_seed=$(frb_seed)")
println("  apply_mass_cut=$(apply_mass_cut), mass_min=$(mass_min)")
println("  z range = [$(test_z_min), $(test_z_max)]")
println("  display bins=$(display_bin_count), candidate_pdf_bins=$(candidate_pdf_bin_count)")
println("  paper z_cut=$(frb_z_cut)")

eligible_redshifts, total_halo_count = collect_eligible_redshifts(
    catalog_source,
    halfdome_path,
    websky_path,
    chunkN,
    itp_z_of_chi,
    rho_m;
    apply_mass_cut=apply_mass_cut,
    mass_min=mass_min,
    z_min=test_z_min,
    z_max=test_z_max
)

eligible_count = length(eligible_redshifts)
println("  total_halo_count=$(total_halo_count)")
println("  eligible_halo_count_in_test_range=$(eligible_count)")
frb_count <= eligible_count || error(
    "frb_count=$(frb_count) exceeds the number of eligible halos in $(test_z_min) <= z <= $(test_z_max), which is $(eligible_count)."
)

candidate_hist_counts = histogram_counts(eligible_redshifts, candidate_pdf_bin_edges)
candidate_hist_pdf = pdf_from_histogram_counts(candidate_hist_counts, candidate_pdf_bin_edges)
candidate_pdf_values = evaluate_histogram_pdf(plot_z_grid, candidate_pdf_bin_edges, candidate_hist_pdf)
candidate_display_bin_probabilities = bin_probabilities_from_pdf(plot_z_grid, candidate_pdf_values, display_bin_edges)
candidate_curve_counts = frb_count .* candidate_pdf_values .* display_bin_width

random_positions = randperm(rng, eligible_count)[1:frb_count]
random_redshifts = eligible_redshifts[random_positions]

naive_weights = Vector{Float64}(undef, eligible_count)
corrected_weights = Vector{Float64}(undef, eligible_count)

@inbounds for i in eachindex(eligible_redshifts)
    z = eligible_redshifts[i]
    target_weight = paper_frb_redshift_pdf_weight(z, chi_of_z_itp, d_l_cut_sq)
    bin_idx = searchsortedlast(candidate_pdf_bin_edges, z)
    if bin_idx == 0
        candidate_density = 0.0
    elseif bin_idx >= length(candidate_pdf_bin_edges)
        candidate_density = candidate_hist_pdf[end]
    else
        candidate_density = candidate_hist_pdf[bin_idx]
    end

    naive_weights[i] = target_weight
    corrected_weights[i] = candidate_density > 0.0 ? target_weight / candidate_density : 0.0
end

naive_positions = draw_weighted_sample_positions(rng, naive_weights, frb_count)
corrected_positions = draw_weighted_sample_positions(rng, corrected_weights, frb_count)

naive_redshifts = eligible_redshifts[naive_positions]
corrected_redshifts = eligible_redshifts[corrected_positions]

paper_direct_counts = summarize_match(
    "Direct draws from paper PDF",
    paper_direct_redshifts,
    paper_display_bin_probabilities,
    display_bin_edges
)
random_counts = summarize_match(
    "Random halo sample",
    random_redshifts,
    candidate_display_bin_probabilities,
    display_bin_edges
)
naive_counts = summarize_match(
    "Naive halo weighting with w(z) = paper PDF",
    naive_redshifts,
    paper_display_bin_probabilities,
    display_bin_edges
)
corrected_counts = summarize_match(
    "Corrected halo weighting with w(z) = paper PDF / halo-density(z)",
    corrected_redshifts,
    paper_display_bin_probabilities,
    display_bin_edges
)

paper_peak = maximum(paper_curve_counts)
candidate_peak = maximum(candidate_curve_counts)
y_limit = 1.15 * max(
    maximum(paper_direct_counts),
    maximum(random_counts),
    maximum(naive_counts),
    maximum(corrected_counts),
    paper_peak,
    candidate_peak
)

common_hist_kwargs = (
    bins=display_bin_edges,
    xlims=(test_z_min, test_z_max),
    ylims=(0.0, y_limit),
    color=:grey,
    linecolor=:grey,
    alpha=0.9,
    label="",
    xlabel="FRB redshift z",
    ylabel="counts / bin"
)

p1 = histogram(
    paper_direct_redshifts;
    title="Direct draws from paper PDF",
    common_hist_kwargs...
)
plot!(p1, plot_z_grid, paper_curve_counts; color=:red, linewidth=2, label="")

p2 = histogram(
    random_redshifts;
    title="Random halo sample",
    common_hist_kwargs...
)
plot!(p2, plot_z_grid, candidate_curve_counts; color=:red, linewidth=2, label="")

p3 = histogram(
    naive_redshifts;
    title="Naive halo weighting",
    common_hist_kwargs...
)
plot!(p3, plot_z_grid, paper_curve_counts; color=:red, linewidth=2, label="")

p4 = histogram(
    corrected_redshifts;
    title="Corrected halo weighting",
    common_hist_kwargs...
)
plot!(p4, plot_z_grid, paper_curve_counts; color=:red, linewidth=2, label="")

comparison_plot = plot(
    p1,
    p2,
    p3,
    p4;
    layout=(2, 2),
    size=(1400, 1000),
    plot_title="FRB redshift distribution tests: paper target vs halo-based sampling"
)

if save_plot
    output_dir = joinpath(@__DIR__, "batched_data", "plots")
    output_path = joinpath(
        output_dir,
        "FRB_redshift_tests_$(catalog_source)_n$(frb_count)_zmax$(fmt_param_value(test_z_max))_zcut$(fmt_param_value(frb_z_cut))_mass$(apply_mass_cut ? fmt_param_value(mass_min) : "none").png"
    )
    saved_path = save_plot_accessible(comparison_plot, output_path)
    println("Saved FRB redshift comparison plot to $(saved_path)")
end
