using XGPaint

# with Sigmoid
println("Test with Sigmoid model")
model = BattagliaTauProfile(Omega_c=0.2603, Omega_b=0.0486,  h=0.6774)

print("Starting the model build_interpolator: fast version \n")
#model_interp = XGPaint.load_precomputed_battaglia_tau()
@time y_small = build_interpolator(
    model;
    cache_file = "cached_model_density-z_websky.jld2",
    N_logθ     = 512,
    pad        = 256,
    logM_max=16.5,
    overwrite  = true,
    verbose    = true,
)
print("Finished the model build_interpolator \n")