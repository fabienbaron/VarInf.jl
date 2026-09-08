# ============================================================================
# Tiny deconvolution demo (32 × 32), runnable in a few seconds.
#
# Usage:
#   julia --project=. examples/deconv_tiny.jl
#
# This script:
#   1. Draws a synthetic image from a NIFTy-style correlated-field prior.
#   2. Convolves it with a Gaussian PSF and adds Gaussian noise.
#   3. Reconstructs via MAP → MGVI → GeoVI using only VarInf.jl
#      (no MGVIImaging / OITOOLS / NFFT in scope).
#   4. Prints reconstruction quality vs ground truth and notes whether
#      the posterior std covers the residual.
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Printf
using Statistics: mean

include(joinpath(@__DIR__, "deconvolution_problem.jl"))

const N        = 32         # grid side length
const DX       = 1.0        # pixel size (arbitrary units)
const PSF_SIG  = 1.5        # Gaussian PSF std (pixels)
const N_FRAC   = 0.05       # noise std as fraction of |image|_∞
const SEED     = 42

println("=== Tiny deconvolution demo ===")
println("Grid: $(N)×$(N), PSF σ = $(PSF_SIG) pix, S/N noise frac = $(N_FRAC)")
println()

grid = FourierGridInfo(N, DX)
cfg  = CorrFieldConfig(slope_prior=(-3.0, 0.0), fluct_prior=(1.0, 0.0))
prob, z_true, image_true, clean_obs =
    generate_synthetic_deconvolution(grid, cfg, PSF_SIG;
                                      noise_frac=N_FRAC, seed=SEED)

println(@sprintf("True latent  : ||z_true||    = %.3f", norm(z_true)))
println(@sprintf("True image   : range         = [%.3f, %.3f]",
                 minimum(image_true), maximum(image_true)))
println(@sprintf("Noise σ      : %.4f", prob.sigma))
println()

println("---- MAP reconstruction ----")
@time z_map = reconstruct_map(prob; maxiter=300, verb=false)

# Two quantities to inspect:
#   image_recovered = CF(z_map)       (the underlying latent image, no PSF)
#   model_map       = CF(z_map) ⊛ PSF (what is compared to the data)
# Bare amplitude kernel (no PSF). The problem stores combined = amp .* psf,
# so we recompute amp separately rather than dividing combined by psf.
amp_kernel = amplitude_spectrum(cfg, grid)[grid.bin_index]

image_recovered = real.(ifft(fft(reshape(z_map, N, N)) .* amp_kernel))
model_map       = real.(ifft(fft(reshape(z_map, N, N)) .* prob.combined_kernel))

img_rel  = norm(image_recovered .- image_true) / norm(image_true)
data_rel = norm(model_map .- clean_obs)        / norm(clean_obs)
chi2_pp  = sum(abs2, model_map .- prob.data) / (prob.sigma^2 * N^2)
println(@sprintf("MAP image  rel err vs true latent image       = %.3f", img_rel))
println(@sprintf("MAP model  rel err vs noise-free observation  = %.3f", data_rel))
println(@sprintf("MAP χ² per pixel (vs noisy data)              = %.3f  (target ≈ 1)",
                 chi2_pp))
println()

println("---- MGVI posterior (3 iter, 3 samples) ----")
@time z_mgvi, samples_mgvi = reconstruct_mgvi(
    prob; z0=z_map, n_iterations=3, n_samples=3,
    map_maxiter=0, kl_maxiter=40, cg_maxiter=40, cg_tol=0.05, verb=false)

# Posterior mean / std image.
#
#   `mode = :antithetic` applies ±1 to each sample (MGVI convention: the
#     stored samples are one side of each pair, build the negation here).
#   `mode = :pairs`      expects samples to already contain explicit pairs as adjacent
#     entries (GeoVI convention after nonlinear refinement: samples[2k-1] and
#     samples[2k] are independent δ₊ and δ₋).
function _posterior_stats(z_center, samples, kernel; mode::Symbol)
    n = size(kernel, 1)
    mean_img = zeros(n, n)
    sq_img   = zeros(n, n)
    count    = 0
    function _accumulate!(s)
        img = real.(ifft(fft(reshape(z_center .+ s, n, n)) .* kernel))
        mean_img .+= img
        sq_img   .+= img .^ 2
        count    += 1
    end
    if mode == :antithetic
        for s in samples
            _accumulate!(s)
            _accumulate!(-s)
        end
    else
        for s in samples
            _accumulate!(s)
        end
    end
    mean_img ./= count
    std_img = sqrt.(max.(sq_img ./ count .- mean_img .^ 2, 0.0))
    return mean_img, std_img
end

# Posterior is over the latent image (CF only), so we use amp_kernel, not
# the combined_kernel which also includes the (fixed, known) PSF blur.
mgvi_mean, mgvi_std = _posterior_stats(z_mgvi, samples_mgvi, amp_kernel; mode=:antithetic)
mgvi_rel = norm(mgvi_mean .- image_true) / norm(image_true)
covered  = mean(abs.(mgvi_mean .- image_true) .< 2 .* mgvi_std)
println(@sprintf("MGVI posterior mean rel err = %.3f", mgvi_rel))
println(@sprintf("MGVI ±2σ band covers %d%% of true-image pixels", round(Int, 100*covered)))
println()

println("---- GeoVI posterior (2 iter, 2 samples) ----")
@time z_gv, samples_gv = reconstruct_geovi(
    prob; z0=z_map, n_iterations=2, n_samples=2,
    map_maxiter=0, kl_maxiter=30, cg_maxiter=40, cg_tol=0.05,
    geo_newton_maxiter=5, geo_cg_maxiter=20, geo_tol=1e-4, verb=false)
gv_mean, gv_std = _posterior_stats(z_gv, samples_gv, amp_kernel; mode=:pairs)
gv_rel  = norm(gv_mean .- image_true) / norm(image_true)
covered_gv = mean(abs.(gv_mean .- image_true) .< 2 .* gv_std)
println(@sprintf("GeoVI posterior mean rel err = %.3f", gv_rel))
println(@sprintf("GeoVI ±2σ band covers %d%% of true-image pixels", round(Int, 100*covered_gv)))
println()

println("=== Tiny demo complete ===")
