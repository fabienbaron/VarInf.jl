# ============================================================================
# DeconvolutionProblem: a minimal AbstractInferenceProblem implementation
# for image deconvolution with a correlated-field prior on the latent image.
#
# Use case: observed pixel grid `data` is the convolution of a latent image
# with a Gaussian PSF, plus Gaussian noise of known scale σ. We want the
# posterior over the latent image.
#
# Forward model (all linear in the latent vector z):
#
#   field(z)        = ifft(fft(reshape(z, n, n)) .* amp_kernel)   [real part]
#   image(z)        = field(z)
#   model(z)        = ifft(fft(image(z)) .* psf_kernel)            [real part]
#
# Because both `amp_kernel` (radial power spectrum on the Fourier grid) and
# `psf_kernel` (Gaussian) are real and even, the cascade simplifies to a
# single Fourier multiplication: `model(z) = ifft(fft(z_2d) .* combined)`
# with `combined = amp_kernel .* psf_kernel`. The operator is self-adjoint,
# so `right_sqrt_metric` and `left_sqrt_metric` apply the same kernel.
#
# Latent layout: `z ∈ R^{n²}`, with a standard-normal prior. The amplitude
# kernel is fixed (no learned slope/fluct) to keep the example small.
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random

struct DeconvolutionProblem{T<:AbstractFloat} <: AbstractInferenceProblem
    grid::FourierGridInfo{T}
    combined_kernel::Matrix{T}   # amp_kernel .* psf_kernel (Fourier domain, real)
    data::Matrix{T}              # observed pixels (npix × npix)
    sigma::T                     # noise std (uniform)
end

Base.eltype(::DeconvolutionProblem{T}) where {T} = T

"""
    DeconvolutionProblem(grid, cfg, psf_sigma_pix, data; sigma)

Convenience constructor. Builds the amplitude kernel from `cfg` (a
`CorrFieldConfig` with `use_iwp=false`) at zero slope/fluct latents, builds
a Gaussian PSF kernel with the given Fourier-domain width, and combines them.
"""
function DeconvolutionProblem(grid::FourierGridInfo{T}, cfg::CorrFieldConfig,
                              psf_sigma_pix::Real, data::AbstractMatrix{<:Real};
                              sigma::Real) where {T<:AbstractFloat}
    @assert !cfg.use_iwp "DeconvolutionProblem expects CorrFieldConfig with use_iwp=false"
    amp_kernel = amplitude_spectrum(cfg, grid)[grid.bin_index]
    psf_kernel = make_smoothing_kernel(grid.npix, grid.npix, grid.dx,
                                        T(psf_sigma_pix) * grid.dx)
    combined = amp_kernel .* psf_kernel
    return DeconvolutionProblem(grid, combined, Matrix{T}(data), T(sigma))
end

# Protocol implementation

VarInf.latent_size(p::DeconvolutionProblem) = p.grid.npix^2
VarInf.data_size(p::DeconvolutionProblem)   = p.grid.npix^2

# Internal: apply the combined Fourier operator to a real image
function _apply_forward(x_2d::AbstractMatrix{<:Real},
                        p::DeconvolutionProblem)
    return real.(ifft(fft(x_2d) .* p.combined_kernel))
end

function VarInf.transformation(p::DeconvolutionProblem,
                                z::AbstractVector{<:Real})
    n = p.grid.npix
    return vec(_apply_forward(reshape(z, n, n), p)) ./ p.sigma
end

function VarInf.right_sqrt_metric(p::DeconvolutionProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    n = p.grid.npix
    return vec(_apply_forward(reshape(v, n, n), p)) ./ p.sigma
end

function VarInf.left_sqrt_metric(p::DeconvolutionProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    n = p.grid.npix
    return vec(_apply_forward(reshape(w ./ p.sigma, n, n), p))
end

function VarInf.energy_and_gradient(p::DeconvolutionProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = vec(p.data) ./ p.sigma
    resid = T_z .- d_white
    chi2  = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end

# Convenience: ground-truth generation for examples / tests

"""
    generate_synthetic_deconvolution(grid, cfg, psf_sigma_pix; noise_frac=0.05, seed=42)
        -> (prob, z_true, image_true, clean_obs)

Draw a synthetic ground-truth latent `z_true ~ N(0, I)`, build the
corresponding latent image, blur with a Gaussian PSF, add Gaussian noise
of std `noise_frac * max(|image_true|)`. Returns the assembled problem,
the true latent, the true image, and the noise-free observation.
"""
function generate_synthetic_deconvolution(grid::FourierGridInfo,
                                          cfg::CorrFieldConfig,
                                          psf_sigma_pix::Real;
                                          noise_frac::Real=0.05,
                                          seed::Int=42,
                                          T::Type{<:AbstractFloat}=eltype(grid))
    Random.seed!(seed)
    n = grid.npix
    amp_kernel = amplitude_spectrum(cfg, grid)[grid.bin_index]
    psf_kernel = make_smoothing_kernel(n, n, grid.dx, T(psf_sigma_pix) * grid.dx)
    combined = amp_kernel .* psf_kernel

    z_true = randn(T, n^2)
    image_true = real.(ifft(fft(reshape(z_true, n, n)) .* amp_kernel))
    clean_obs  = real.(ifft(fft(image_true) .* psf_kernel))
    sigma_n = T(noise_frac) * maximum(abs.(image_true))
    data = clean_obs .+ sigma_n .* randn(T, n, n)

    prob = DeconvolutionProblem(grid, combined, data, sigma_n)
    return prob, z_true, image_true, clean_obs
end
