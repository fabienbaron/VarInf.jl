# ============================================================================
# Point-source recovery demo (generic, no interferometry).
#
# Three point sources on a 32×32 pixel image, blurred by a Gaussian PSF and
# observed with Gaussian noise. Latent z is 9-dimensional (3 sources × 3
# params). Recovers (x, y, F) for each source via MAP + GeoVI and reports
# posterior calibration.
# ============================================================================

using VarInf
using LinearAlgebra
using Random
using Printf
using Statistics: mean

include(joinpath(@__DIR__, "point_sources_problem.jl"))

println("=== Point-source recovery demo ===")
truth_xyF = [
    (10.0, 12.0, 1.0),
    (20.0, 18.0, 0.6),
    (15.0, 25.0, 0.8),
]
println("True sources:")
for (i, t) in enumerate(truth_xyF)
    @printf("  #%d  x=%5.2f  y=%5.2f  F=%.3f\n", i, t[1], t[2], t[3])
end

prob, tx, ty, tF, _ = generate_synthetic_point_sources(;
    truth_xyF=truth_xyF, npix=32, sigma_psf=1.5, noise=0.05, seed=42)
@printf("Grid = %d×%d, PSF σ = %.1f pix, noise σ = %.3f\n",
        prob.npix, prob.npix, prob.sigma_psf, prob.sigma_noise)
@printf("Prior offset from truth: max |x_mean - x_true| ≈ %.2f pix\n",
        maximum(abs.(prob.x_mean .- tx)))

# ---- MAP ----
println("\n---- MAP (L-BFGS) ----")
@time z_map = reconstruct_map(prob; maxiter=500, gtol=(1e-8, 1e-8), verb=false)
map_x, map_y, map_F = _unpack(z_map, prob)
println("MAP estimate:")
for k in 1:prob.n_sources
    @printf("  #%d  x=%5.2f  y=%5.2f  F=%.3f      (truth %5.2f, %5.2f, %.3f)\n",
            k, map_x[k], map_y[k], map_F[k], tx[k], ty[k], tF[k])
end

# ---- GeoVI posterior ----
println("\n---- GeoVI posterior (3 iter, 4 samples) ----")
@time z_geo, samples_geo = reconstruct_geovi(
    prob; z0=z_map, n_iterations=3, n_samples=iter -> 4,
    map_maxiter=0, kl_maxiter=30, cg_maxiter=30, cg_tol=0.05,
    geo_newton_maxiter=5, geo_cg_maxiter=15, geo_tol=1e-5, verb=false)

# Posterior in physical space
function _physical_samples(z_center, samples, prob)
    out = Vector{Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}()
    for s in samples
        push!(out, _unpack(z_center .+ s, prob))
    end
    return out
end
ps = _physical_samples(z_geo, samples_geo, prob)

function _stats(getter)
    vals = reduce(hcat, [getter(s) for s in ps])      # N × n_samples
    means = vec(mean(vals, dims=2))
    stds  = vec(sqrt.(mean((vals .- means) .^ 2, dims=2)))
    return means, stds
end
x_mean, x_std = _stats(s -> s[1])
y_mean, y_std = _stats(s -> s[2])
F_mean, F_std = _stats(s -> s[3])

println("\nPosterior (mean ± std) vs truth:")
@printf("  %-3s  %-22s %-22s %-22s   %-s\n",
        "src", "x  truth → estimate", "y  truth → estimate", "F  truth → estimate", "z-error (x, y, F)")
for k in 1:prob.n_sources
    zx = (x_mean[k] - tx[k]) / max(x_std[k], 1e-12)
    zy = (y_mean[k] - ty[k]) / max(y_std[k], 1e-12)
    zF = (F_mean[k] - tF[k]) / max(F_std[k], 1e-12)
    @printf("  #%d  %5.2f → %5.2f ± %.3f   %5.2f → %5.2f ± %.3f   %.3f → %.3f ± %.3f   (%+.2f, %+.2f, %+.2f)σ\n",
            k, tx[k], x_mean[k], x_std[k],
               ty[k], y_mean[k], y_std[k],
               tF[k], F_mean[k], F_std[k],
               zx, zy, zF)
end

println("\n=== Point-source recovery demo complete ===")
