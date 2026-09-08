# ============================================================================
# Two-Gaussian fit demo
#
# Recovers (μ₁, σ₁, A₁, μ₂, σ₂, A₂) from 100 noisy observations of a sum of
# two 1D Gaussians, using only VarInf.jl. No correlated-field machinery,
# no FFTs; this exercises the AbstractInferenceProblem protocol on a small
# parametric problem.
#
# Usage:
#   julia --project=examples examples/two_gaussians.jl
# ============================================================================

using VarInf
using Random
using Printf
using LinearAlgebra
using Statistics: mean

include(joinpath(@__DIR__, "two_gaussians_problem.jl"))

println("=== Two-Gaussian fit demo ===")
truth_tuple = (-1.5, 0.7, 1.0, 1.8, 1.0, 1.2)
println("Truth (μ₁, σ₁, A₁, μ₂, σ₂, A₂) = ", truth_tuple)

prob, _, x, y_clean = generate_synthetic_two_gaussians(;
    truth=truth_tuple, n_obs=100, noise=0.05, seed=42)
println(@sprintf("n_obs = %d, noise σ = %.3f", length(x), prob.sigma_noise))

# ---- MAP ----
println("\n---- MAP (L-BFGS) ----")
@time z_map = reconstruct_map(prob; maxiter=200, verb=false)
phys_map = _unpack(z_map, prob)
@printf("MAP: μ₁=%+.3f σ₁=%.3f A₁=%.3f  μ₂=%+.3f σ₂=%.3f A₂=%.3f\n", phys_map...)

# ---- GeoVI posterior ----
println("\n---- GeoVI posterior (4 iter, 4 samples each) ----")
@time z_geo, samples_geo = reconstruct_geovi(
    prob; z0=z_map, n_iterations=4, n_samples=iter -> 4,
    map_maxiter=0, kl_maxiter=30, cg_maxiter=30, cg_tol=0.05,
    geo_newton_maxiter=6, geo_cg_maxiter=15, geo_tol=1e-5, verb=false)

# Summarize posterior in physical-parameter space
nsamples = length(samples_geo)
phys_samples = [collect(_unpack(z_geo .+ s, prob)) for s in samples_geo]
phys_mat = reduce(hcat, phys_samples)              # 6 × n
post_mean = vec(mean(phys_mat, dims=2))
post_std  = vec(sqrt.(mean((phys_mat .- post_mean) .^ 2, dims=2)))

names = ("μ₁", "σ₁", "A₁", "μ₂", "σ₂", "A₂")
println("\nParameter   truth     posterior mean ± std         z-error")
for i in 1:6
    truth_i = truth_tuple[i]
    z_err   = (post_mean[i] - truth_i) / max(post_std[i], 1e-12)
    @printf("  %-3s     %+7.3f    %+7.3f ± %.3f          %+5.2f σ\n",
            names[i], truth_i, post_mean[i], post_std[i], z_err)
end

# Data-space fit quality
y_map_clean = let
    μ1, σ1, A1, μ2, σ2, A2 = phys_map
    @. A1 * exp(-(x - μ1)^2 / (2σ1^2)) + A2 * exp(-(x - μ2)^2 / (2σ2^2))
end
y_geo_clean = let
    μ1, σ1, A1, μ2, σ2, A2 = post_mean
    @. A1 * exp(-(x - μ1)^2 / (2σ1^2)) + A2 * exp(-(x - μ2)^2 / (2σ2^2))
end
println(@sprintf("\nMAP   fit rms vs clean truth = %.4f", norm(y_map_clean .- y_clean) / sqrt(length(x))))
println(@sprintf("GeoVI fit rms vs clean truth = %.4f", norm(y_geo_clean .- y_clean) / sqrt(length(x))))
println("=== Two-Gaussian demo complete ===")
