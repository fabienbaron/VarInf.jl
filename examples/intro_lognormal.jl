# ============================================================================
# Log-normal correlated-field demo: VarInf's analogue of NIFTy's 0_intro.py.
#
# Reconstruct a 2-D log-normal signal whose correlated-field hyperparameters
# (slope, fluctuations, flexibility, asperity, per-bin spectrum) are all
# learned from the data, along with a multiplicative scaling. Identity
# forward operator, Gaussian noise.
#
# Usage:
#   julia --project=examples examples/intro_lognormal.jl
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Printf
using Statistics: mean

include(joinpath(@__DIR__, "intro_lognormal_problem.jl"))

const N            = 64
const SCALE_MEAN   = 2.0
const SCALE_STD    = 0.5
const NOISE_FRAC   = 0.03
const SEED         = 2026

println("=== Log-normal correlated-field demo ===")
@printf("Grid: %d×%d, scaling lognormal(mean=%.1f, std=%.1f), noise = %.0f%% of peak\n",
        N, N, SCALE_MEAN, SCALE_STD, 100 * NOISE_FRAC)
println("Hyperparameters slope/fluct/flex/asp + per-bin spectrum are all learned.")
println()

cfg = CorrFieldConfig(slope_prior=(-2.0, 0.5),
                      fluct_prior=(0.4, 0.04),
                      flex_prior=(1.0, 0.5),
                      asp_prior=(0.6, 0.06),
                      offset_prior=(0.0, 0.1))

prob, z_true, signal_true, amp_true =
    generate_synthetic_lognormal(; npix=N, dx=1.0/N,
        scaling_mean=SCALE_MEAN, scaling_std=SCALE_STD,
        cfg=cfg, noise_frac=NOISE_FRAC, seed=SEED)

# True hypers (for ground-truth printing)
let u = _unpack(z_true, prob)
    true_slope = cfg.slope_mean + cfg.slope_std * u.xi_slope
    true_fluct = exp(cfg.fluct_mean + cfg.fluct_std * u.xi_fluct)
    true_scaling = exp(SCALE_MEAN + SCALE_STD * u.xi_scaling)
    @printf("Truth: slope = %+.3f, fluct = %.4f, scaling = %.3f\n",
            true_slope, true_fluct, true_scaling)
    @printf("Signal: range = [%.3f, %.3f]\n",
            minimum(signal_true), maximum(signal_true))
    @printf("Noise σ = %.4f\n", prob.sigma_noise)
end

# 1. MAP warm-start near truth (avoid global mode-hopping in the hypers)

println("\n---- MAP (L-BFGS) warm-started near truth ----")
Random.seed!(123)
z_init = z_true .+ 0.1 .* randn(length(z_true))
e0, _ = VarInf.energy_and_gradient(prob, z_init)
@printf("Initial energy   = %.4f\n", e0)
@time z_map = reconstruct_map(prob; z0=z_init, maxiter=200,
                              gtol=(1e-8, 1e-8), verb=false)
e_map, _ = VarInf.energy_and_gradient(prob, z_map)
@printf("Final MAP energy = %.4f\n", e_map)

# Recovered hypers + signal
let u_map = _unpack(z_map, prob)
    map_slope = cfg.slope_mean + cfg.slope_std * u_map.xi_slope
    map_fluct = exp(cfg.fluct_mean + cfg.fluct_std * u_map.xi_fluct)
    map_scaling = exp(SCALE_MEAN + SCALE_STD * u_map.xi_scaling)
    @printf("MAP    : slope = %+.3f, fluct = %.4f, scaling = %.3f\n",
            map_slope, map_fluct, map_scaling)

    signal_map = let fwd = _forward(z_map, prob); fwd.signal; end
    rel_err_map = norm(signal_map .- signal_true) / norm(signal_true)
    @printf("MAP signal rel err vs truth        = %.3f\n", rel_err_map)
end

# 2. GeoVI posterior

println("\n---- GeoVI posterior (3 iter, 3 samples) ----")
@time z_geo, samples_geo = reconstruct_geovi(
    prob; z0=z_map, n_iterations=3, n_samples=3,
    map_maxiter=0, kl_maxiter=20, cg_maxiter=30, cg_tol=0.05,
    geo_newton_maxiter=4, geo_cg_maxiter=15, geo_tol=1e-4, verb=false)

# Posterior over the SIGNAL and over the AMPLITUDE SPECTRUM
function _posterior_stats_of(getter, samples, z_center, prob)
    refs = [getter(_forward(z_center .+ s, prob)) for s in samples]
    stack = reduce((a, b) -> cat(a, b; dims=ndims(refs[1]) + 1),
                   [reshape(r, size(r)..., 1) for r in refs])
    K = size(stack, ndims(stack))
    mean_arr = dropdims(sum(stack, dims=ndims(stack)) ./ K,
                         dims=ndims(stack))
    sq_arr   = dropdims(sum(stack .^ 2, dims=ndims(stack)) ./ K,
                         dims=ndims(stack))
    std_arr  = sqrt.(max.(sq_arr .- mean_arr .^ 2, 0.0))
    return mean_arr, std_arr
end

signal_mean, signal_std = _posterior_stats_of(fwd -> fwd.signal,
                                               samples_geo, z_geo, prob)
amp_mean,    amp_std    = _posterior_stats_of(fwd -> fwd.amp,
                                               samples_geo, z_geo, prob)

rel_err_geo = norm(signal_mean .- signal_true) / norm(signal_true)
@printf("GeoVI posterior-mean signal rel err = %.3f\n", rel_err_geo)
covered = mean(abs.(signal_mean .- signal_true) .< 2 .* (signal_std .+ 1e-12))
@printf("GeoVI ±2σ band covers %d%% of truth pixels (signal space)\n",
        round(Int, 100 * covered))

# 3. Amplitude-spectrum recovery (the headline plot of 0_intro)

println("\nAmplitude spectrum (radial bins 2..end-1, truth vs posterior mean):")
@printf("  %-8s %-12s %-12s %-12s %-s\n",
        "bin", "k_length", "truth", "posterior", "posterior σ")
n_show = min(8, prob.grid.n_bins - 2)
step = max(1, (prob.grid.n_bins - 2) ÷ n_show)
for b in 2:step:prob.grid.n_bins-1
    @printf("  %-8d %-12.4g %-12.4g %-12.4g  %-.3g\n",
            b, prob.grid.mode_lengths[b], amp_true[b], amp_mean[b], amp_std[b])
end

# Recovered hypers
let u_geo = _unpack(z_geo, prob)
    geo_slope = cfg.slope_mean + cfg.slope_std * u_geo.xi_slope
    geo_fluct = exp(cfg.fluct_mean + cfg.fluct_std * u_geo.xi_fluct)
    geo_scaling = exp(SCALE_MEAN + SCALE_STD * u_geo.xi_scaling)
    @printf("\nGeoVI hypers (point estimate at posterior mean z):\n")
    @printf("  slope   = %+.3f\n", geo_slope)
    @printf("  fluct   = %.4f\n", geo_fluct)
    @printf("  scaling = %.3f\n", geo_scaling)
end

println("\n=== Log-normal demo complete ===")
