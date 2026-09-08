# ============================================================================
# Saturn deconvolution demo (128 × 128).
#
# Requires FITSIO. Install in your active environment if missing:
#     using Pkg; Pkg.add("FITSIO")
#
# This script:
#   1. Loads /home/baron/SOFTWARE/numericalmethods/src/saturn.fits (256x256 UInt8)
#      and downsamples to 128x128 by 2x2 block averaging.
#   2. Blurs with a Gaussian PSF (FWHM ≈ 3.5 pixels ⇒ σ ≈ 1.5).
#   3. Adds Gaussian noise at S/N ≈ 30 dB (≈3% of peak).
#   4. Reconstructs via MAP and GeoVI using VarInf.jl's protocol with a
#      power-law (slope ≈ -3) correlated-field regularization.
#   5. Prints quality metrics and saves four FITS files in the cwd:
#         saturn_truth.fits, saturn_data.fits,
#         saturn_map.fits,   saturn_geovi_mean.fits
#
# Notes:
# - Saturn is a real image, not drawn from the assumed N(0, I) prior, so the
#   reconstruction is biased toward the prior's preferred smoothness. This is
#   the expected behavior of a regularized deconvolution.
# - The combined kernel (CF amplitude × PSF) is extremely ill-conditioned
#   at high frequencies, so MAP via L-BFGS converges slowly and ends with
#   χ² well above 1. GeoVI's inner Newton-CG handles the conditioning much
#   better and produces a posterior mean noticeably closer to the truth.
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Printf
using Statistics: mean
using FITSIO

include(joinpath(@__DIR__, "deconvolution_problem.jl"))

const SATURN_PATH = "/home/baron/SOFTWARE/numericalmethods/src/saturn.fits"
const N           = 128       # working grid size after 2× downsampling
const DX          = 1.0       # pixel size (arbitrary)
const PSF_SIG     = 1.5       # PSF Gaussian σ in pixels
const NOISE_FRAC  = 0.03      # noise std as fraction of peak

println("=== Saturn deconvolution demo ===")
@printf("Working grid: %d×%d, PSF σ = %.1f pix, noise = %.0f%% of peak\n",
        N, N, PSF_SIG, 100 * NOISE_FRAC)
println()

# 1. Load and downsample

println("Loading $SATURN_PATH ...")
raw = let f = FITS(SATURN_PATH)
    img = read(f[1])
    close(f)
    img
end
@assert size(raw) == (256, 256) "expected 256×256, got $(size(raw))"

# 2×2 block average → 128×128 Float64
saturn_256 = Float64.(raw)
saturn = 0.25 .* (saturn_256[1:2:end, 1:2:end] .+
                   saturn_256[2:2:end, 1:2:end] .+
                   saturn_256[1:2:end, 2:2:end] .+
                   saturn_256[2:2:end, 2:2:end])
saturn ./= maximum(saturn)      # normalize: peak = 1
@printf("Truth image: range = [%.3f, %.3f]\n", minimum(saturn), maximum(saturn))

# 2. Build the (data, problem) pair

Random.seed!(2026)
grid = FourierGridInfo(N, DX)
# slope = −2 (closer to a power-spectrum slope appropriate for natural images)
# is much better-conditioned for L-BFGS than slope = −3.
cfg  = CorrFieldConfig(slope_prior=(-2.0, 0.0), fluct_prior=(1.0, 0.0))

# Bare amplitude kernel (CF prior) and PSF kernel (Gaussian)
amp_kernel = amplitude_spectrum(cfg, grid)[grid.bin_index]
psf_kernel = make_smoothing_kernel(N, N, DX, PSF_SIG * DX)
combined_kernel = amp_kernel .* psf_kernel

# Forward-blur Saturn (without any CF; we are not claiming Saturn was drawn
# from the prior; we just want a realistic noisy blurred observation)
saturn_blurred = real.(ifft(fft(saturn) .* psf_kernel))
sigma_n = NOISE_FRAC * maximum(saturn)
data    = saturn_blurred .+ sigma_n .* randn(N, N)

@printf("Noise σ = %.4f  (peak/σ ≈ %.1f → ~%.0f dB)\n",
        sigma_n, 1.0 / sigma_n, 20 * log10(1.0 / sigma_n))

prob = DeconvolutionProblem(grid, combined_kernel, data, sigma_n)

# 3. MAP reconstruction

println("\n---- MAP reconstruction (L-BFGS) ----")
@time z_map = reconstruct_map(prob; maxiter=800, gtol=(1e-8, 1e-8), verb=false)

image_map = real.(ifft(fft(reshape(z_map, N, N)) .* amp_kernel))
model_map = real.(ifft(fft(reshape(z_map, N, N)) .* combined_kernel))
img_rel  = norm(image_map .- saturn)            / norm(saturn)
data_rel = norm(model_map .- saturn_blurred)    / norm(saturn_blurred)
chi2_pp  = sum(abs2, model_map .- prob.data) / (sigma_n^2 * N^2)
@printf("MAP image rel err vs Saturn truth          = %.3f\n", img_rel)
@printf("MAP model rel err vs noise-free observation = %.3f\n", data_rel)
@printf("MAP χ² per pixel (vs noisy data)           = %.3f  (target ≈ 1)\n", chi2_pp)

# 4. GeoVI posterior (single iteration to keep runtime modest)

println("\n---- GeoVI posterior (2 iter, 2 samples) ----")
@time z_geo, samples_geo = reconstruct_geovi(
    prob; z0=z_map, n_iterations=2, n_samples=2,
    map_maxiter=0, kl_maxiter=20, cg_maxiter=30, cg_tol=0.1,
    geo_newton_maxiter=4, geo_cg_maxiter=15, geo_tol=1e-4, verb=false)

# Posterior over the underlying IMAGE (CF only, no PSF)
function posterior_stats(z_center, samples, kernel)
    n = size(kernel, 1)
    mean_img = zeros(n, n)
    sq_img   = zeros(n, n)
    for s in samples
        img = real.(ifft(fft(reshape(z_center .+ s, n, n)) .* kernel))
        mean_img .+= img
        sq_img   .+= img .^ 2
    end
    K = length(samples)
    mean_img ./= K
    std_img = sqrt.(max.(sq_img ./ K .- mean_img .^ 2, 0.0))
    return mean_img, std_img
end

geo_mean, geo_std = posterior_stats(z_geo, samples_geo, amp_kernel)
geo_rel = norm(geo_mean .- saturn) / norm(saturn)
cov_band = mean(abs.(geo_mean .- saturn) .< 2 .* (geo_std .+ 1e-9))
@printf("GeoVI posterior mean rel err vs Saturn truth = %.3f\n", geo_rel)
@printf("GeoVI ±2σ band covers %d%% of truth pixels   (prior-bias caveat)\n",
        round(Int, 100 * cov_band))

# 5. Save outputs

function save_fits(filename::String, arr::Matrix{Float64})
    f = FITS(filename, "w")
    write(f, arr)
    close(f)
end

save_fits("saturn_truth.fits",      saturn)
save_fits("saturn_data.fits",       data)
save_fits("saturn_map.fits",        image_map)
save_fits("saturn_geovi_mean.fits", geo_mean)
println("\nSaved: saturn_truth.fits, saturn_data.fits, saturn_map.fits, saturn_geovi_mean.fits")
println("=== Saturn demo complete ===")
