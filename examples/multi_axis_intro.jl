# ============================================================================
# Multi-axis correlated-field intro demo.
#
# Reconstructs a 32 × 32 × 8 log-normal field whose Fourier kernel factorises
# across a spatial 2-D radial axis (the (32,32) image plane) and a 1-D
# spectral axis (the 8 wavelengths). All correlated-field hyperparameters
# on both axes (slope, fluctuations, flexibility, asperity, per-bin
# spectrum deviations) plus a single global offset are learned from the
# data. Identity forward operator, Gaussian noise.
#
# Usage:
#   julia --project=examples examples/multi_axis_intro.jl
# ============================================================================

using VarInf
using LinearAlgebra
using Random
using Printf
using Statistics: mean

include(joinpath(@__DIR__, "multi_axis_intro_problem.jl"))

const SPATIAL_DIM  = 32
const SPECTRAL_DIM = 8
const NOISE_FRAC   = 0.03
const SEED         = 2026

println("=== Multi-axis correlated-field demo ===")
@printf("Field shape: %d × %d × %d, noise = %.0f%% of peak\n",
        SPATIAL_DIM, SPATIAL_DIM, SPECTRAL_DIM, 100 * NOISE_FRAC)
println("Learned per-axis hypers: slope, fluct, flex (spatial), asp (spatial), spectrum")
println()

sp_cfg = CorrFieldConfig(slope_prior=(-2.0, 0.5),
                          fluct_prior=(0.4, 0.04),
                          flex_prior=(1.0, 0.5),
                          asp_prior=(0.6, 0.06))
sc_cfg = CorrFieldConfig(slope_prior=(-1.5, 0.3),
                          fluct_prior=(0.3, 0.03))

prob, z_true, field_true, signal_true =
    generate_synthetic_multi_axis(; spatial_dim=SPATIAL_DIM,
                                    spectral_dim=SPECTRAL_DIM,
                                    sp_cfg=sp_cfg, sc_cfg=sc_cfg,
                                    offset_prior=(0.0, 0.1),
                                    noise_frac=NOISE_FRAC, seed=SEED)

# Print ground-truth hypers (read directly from z_true via latent_unpack)
_, true_axes, true_xi_offset = latent_unpack(prob.mcf, z_true)
let
    sp = true_axes[1]
    sc = true_axes[2]
    true_sp_slope = sp_cfg.slope_mean + sp_cfg.slope_std * sp.xi_slope
    true_sp_fluct = exp(sp_cfg.fluct_mean + sp_cfg.fluct_std * sp.xi_fluct)
    true_sc_slope = sc_cfg.slope_mean + sc_cfg.slope_std * sc.xi_slope
    true_sc_fluct = exp(sc_cfg.fluct_mean + sc_cfg.fluct_std * sc.xi_fluct)
    true_offset   = exp(0.0 + 0.1 * true_xi_offset)
    @printf("Truth:  spatial  slope = %+.3f, fluct = %.4f\n",
            true_sp_slope, true_sp_fluct)
    @printf("        spectral slope = %+.3f, fluct = %.4f\n",
            true_sc_slope, true_sc_fluct)
    @printf("        DC bin (offset) = %.4f\n", true_offset)
end
@printf("Signal: range = [%.3f, %.3f]\n",
        minimum(signal_true), maximum(signal_true))
@printf("Noise σ = %.4f\n", prob.sigma_noise)

# MAP warm-start (near truth, avoid global mode hopping)

println("\n---- MAP (L-BFGS) warm-started near truth ----")
Random.seed!(123)
z_init = z_true .+ 0.1 .* randn(length(z_true))
e0, _ = VarInf.energy_and_gradient(prob, z_init)
@printf("Initial energy   = %.4f\n", e0)
@time z_map = reconstruct_map(prob; z0=z_init, maxiter=200,
                              gtol=(1e-8, 1e-8), verb=false)
e_map, _ = VarInf.energy_and_gradient(prob, z_map)
@printf("Final MAP energy = %.4f\n", e_map)

# MAP signal recovery
let unpacked = latent_unpack(prob.mcf, z_map); map_axes = unpacked[2]; map_xi_offset = unpacked[3]
    sp = map_axes[1]; sc = map_axes[2]
    map_sp_slope = sp_cfg.slope_mean + sp_cfg.slope_std * sp.xi_slope
    map_sp_fluct = exp(sp_cfg.fluct_mean + sp_cfg.fluct_std * sp.xi_fluct)
    map_sc_slope = sc_cfg.slope_mean + sc_cfg.slope_std * sc.xi_slope
    map_sc_fluct = exp(sc_cfg.fluct_mean + sc_cfg.fluct_std * sc.xi_fluct)
    map_offset   = exp(0.0 + 0.1 * map_xi_offset)
    @printf("MAP:    spatial  slope = %+.3f, fluct = %.4f\n",
            map_sp_slope, map_sp_fluct)
    @printf("        spectral slope = %+.3f, fluct = %.4f\n",
            map_sc_slope, map_sc_fluct)
    @printf("        DC bin (offset) = %.4f\n", map_offset)
end

signal_map = exp.(prob.mcf(z_map))
rel_err_map = norm(signal_map .- signal_true) / norm(signal_true)
@printf("MAP signal rel err vs truth = %.3f\n", rel_err_map)

# GeoVI posterior

println("\n---- GeoVI posterior (3 iter, 3 samples) ----")
@time z_geo, samples_geo = reconstruct_geovi(
    prob; z0=z_map, n_iterations=3, n_samples=3,
    map_maxiter=0, kl_maxiter=20, cg_maxiter=30, cg_tol=0.05,
    geo_newton_maxiter=4, geo_cg_maxiter=15, geo_tol=1e-4, verb=false)

# Posterior over the signal: collect samples, compute mean / std
function _posterior_signal_stats(z_center, samples, prob)
    signals = [exp.(prob.mcf(z_center .+ s)) for s in samples]
    K = length(signals)
    mean_sig = sum(signals) ./ K
    sq_sig = sum(s -> s .^ 2, signals) ./ K
    std_sig = sqrt.(max.(sq_sig .- mean_sig .^ 2, 0.0))
    return mean_sig, std_sig
end

signal_mean, signal_std = _posterior_signal_stats(z_geo, samples_geo, prob)
rel_err_geo = norm(signal_mean .- signal_true) / norm(signal_true)
covered = mean(abs.(signal_mean .- signal_true) .< 2 .* (signal_std .+ 1e-12))
@printf("GeoVI posterior-mean signal rel err = %.3f\n", rel_err_geo)
@printf("GeoVI ±2σ band covers %d%% of truth voxels\n", round(Int, 100 * covered))

# Recovered hypers from GeoVI posterior mean
let unpacked = latent_unpack(prob.mcf, z_geo); geo_axes = unpacked[2]; geo_xi_offset = unpacked[3]
    sp = geo_axes[1]; sc = geo_axes[2]
    geo_sp_slope = sp_cfg.slope_mean + sp_cfg.slope_std * sp.xi_slope
    geo_sp_fluct = exp(sp_cfg.fluct_mean + sp_cfg.fluct_std * sp.xi_fluct)
    geo_sc_slope = sc_cfg.slope_mean + sc_cfg.slope_std * sc.xi_slope
    geo_sc_fluct = exp(sc_cfg.fluct_mean + sc_cfg.fluct_std * sc.xi_fluct)
    geo_offset   = exp(0.0 + 0.1 * geo_xi_offset)
    @printf("\nGeoVI hypers (point estimate at posterior-mean z):\n")
    @printf("  spatial  slope = %+.3f, fluct = %.4f\n", geo_sp_slope, geo_sp_fluct)
    @printf("  spectral slope = %+.3f, fluct = %.4f\n", geo_sc_slope, geo_sc_fluct)
    @printf("  DC bin (offset) = %.4f\n", geo_offset)
end

println("\n=== Multi-axis demo complete ===")
