using XGPaint

# with Sigmoid
println("Test with Sigmoid model")
model = SigmoidBattagliaTauProfile(Omega_c=0.2603, Omega_b=0.0486,  h=0.6774)

print("Starting the model build_interpolator: fast version \n")
#model_interp = XGPaint.load_precomputed_battaglia_tau()
@time y_small = build_interpolator(
    model;
    cache_file = "cached_model_sigmoid_Ntheta512_pad256_integral_reduced_acc.jld2",
    N_logθ     = 512,
    pad        = 256,
    overwrite  = true,
    verbose    = true,
)
print("Finished the model build_interpolator \n")