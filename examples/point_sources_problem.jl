# ============================================================================
# PointSourcesProblem: fit N point sources (position + flux) to a noisy
# 2D image observation.
#
# This is the generic version of the OIVI/MGVIImaging point-source model:
# direct pixel observations, no interferometric visibilities, no NFFT. It
# demonstrates VarInf.jl on a parametric 2D problem that lives outside
# either the correlated-field or the interferometry world.
#
# Model:
#   data[i,j] = Σ_k F_k · exp(−((i − x_k)² + (j − y_k)²) / (2 σ_psf²))
#               + N(0, σ_noise²) per pixel
#
# Latent layout (3·N entries, z ~ N(0, I)):
#   For each source k in 1..N (offset = 3·(k-1)):
#     z[offset+1] → x_k  via  x_k = x_mean[k] + xy_std · z[offset+1]
#     z[offset+2] → y_k  via  y_k = y_mean[k] + xy_std · z[offset+2]
#     z[offset+3] → F_k  via  F_k = F_scale · exp(z[offset+3])
# ============================================================================

using VarInf
using LinearAlgebra
using Random

struct PointSourcesProblem{T<:AbstractFloat} <: AbstractInferenceProblem
    npix::Int                       # image is npix × npix
    sigma_psf::T                    # PSF std (pixels)
    data::Matrix{T}                 # observed image (npix × npix)
    sigma_noise::T                  # per-pixel noise std

    # Prior parameters (length-N vectors / scalars)
    n_sources::Int
    x_mean::Vector{T}               # prior means for x positions
    y_mean::Vector{T}               # prior means for y positions
    xy_std::T                       # prior std on positions (shared)
    F_scale::T                      # F = F_scale * exp(z) (Lognormal)
end

Base.eltype(::PointSourcesProblem{T}) where {T} = T

VarInf.latent_size(p::PointSourcesProblem) = 3 * p.n_sources
VarInf.data_size(p::PointSourcesProblem)   = p.npix * p.npix

# Latent → physical (returns x, y, F as length-N vectors)
function _unpack(z::AbstractVector{<:Real}, p::PointSourcesProblem)
    N = p.n_sources
    x = similar(p.x_mean)
    y = similar(p.y_mean)
    F = Vector{eltype(p)}(undef, N)
    @inbounds for k in 1:N
        x[k] = p.x_mean[k] + p.xy_std * z[3*(k-1) + 1]
        y[k] = p.y_mean[k] + p.xy_std * z[3*(k-1) + 2]
        F[k] = p.F_scale * exp(z[3*(k-1) + 3])
    end
    return x, y, F
end

# Forward model: sum of Gaussian point-spread spots on the pixel grid
function _model(z::AbstractVector{<:Real}, p::PointSourcesProblem)
    x, y, F = _unpack(z, p)
    npix = p.npix
    Tp = eltype(p)
    inv2σ2 = inv(2 * p.sigma_psf^2)
    img = zeros(Tp, npix, npix)
    @inbounds for j in 1:npix, i in 1:npix
        s = zero(Tp)
        for k in 1:p.n_sources
            dx = i - x[k]
            dy = j - y[k]
            s += F[k] * exp(-(dx^2 + dy^2) * inv2σ2)
        end
        img[i, j] = s
    end
    return img
end

VarInf.transformation(p::PointSourcesProblem, z::AbstractVector{<:Real}) =
    vec(_model(z, p)) ./ p.sigma_noise

# JVP: J · v, forward-mode through latent → physical → image
function VarInf.right_sqrt_metric(p::PointSourcesProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    x, y, F = _unpack(z, p)
    # Tangent in physical-parameter space
    Tp = eltype(p)
    N = p.n_sources
    dx = Vector{Tp}(undef, N)
    dy = Vector{Tp}(undef, N)
    dF = Vector{Tp}(undef, N)
    @inbounds for k in 1:N
        dx[k] = p.xy_std * v[3*(k-1) + 1]
        dy[k] = p.xy_std * v[3*(k-1) + 2]
        dF[k] = F[k] * v[3*(k-1) + 3]   # F = scale · exp(z) ⇒ dF = F · dz
    end

    npix = p.npix
    inv_σ2 = inv(p.sigma_psf^2)
    inv2σ2 = inv_σ2 / 2
    out = zeros(Tp, npix, npix)
    @inbounds for j in 1:npix, i in 1:npix
        s = zero(Tp)
        for k in 1:N
            ddx = i - x[k]
            ddy = j - y[k]
            r2  = ddx^2 + ddy^2
            e   = exp(-r2 * inv2σ2)
            g   = F[k] * e
            # ∂g/∂x = g · ddx/σ², ∂g/∂y = g · ddy/σ², ∂g/∂F = e
            s += g * ddx * inv_σ2 * dx[k] +
                 g * ddy * inv_σ2 * dy[k] +
                 e * dF[k]
        end
        out[i, j] = s
    end
    return vec(out) ./ p.sigma_noise
end

# VJP: J' · w, reverse-mode
function VarInf.left_sqrt_metric(p::PointSourcesProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    x, y, F = _unpack(z, p)
    Tp = eltype(p)
    N = p.n_sources
    npix = p.npix
    w_scaled = reshape(w ./ p.sigma_noise, npix, npix)

    g_x = zeros(Tp, N)
    g_y = zeros(Tp, N)
    g_F = zeros(Tp, N)
    inv_σ2 = inv(p.sigma_psf^2)
    inv2σ2 = inv_σ2 / 2
    @inbounds for j in 1:npix, i in 1:npix
        wk = w_scaled[i, j]
        wk == 0.0 && continue
        for k in 1:N
            ddx = i - x[k]
            ddy = j - y[k]
            e = exp(-(ddx^2 + ddy^2) * inv2σ2)
            g = F[k] * e
            g_x[k] += wk * g * ddx * inv_σ2
            g_y[k] += wk * g * ddy * inv_σ2
            g_F[k] += wk * e
        end
    end

    # Chain to latent space
    out = Vector{Tp}(undef, 3 * N)
    @inbounds for k in 1:N
        out[3*(k-1) + 1] = g_x[k] * p.xy_std
        out[3*(k-1) + 2] = g_y[k] * p.xy_std
        out[3*(k-1) + 3] = g_F[k] * F[k]   # chain through exp
    end
    return out
end

function VarInf.energy_and_gradient(p::PointSourcesProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = vec(p.data) ./ p.sigma_noise
    resid = T_z .- d_white
    chi2  = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end

# Convenience: generate a synthetic point-source observation

"""
    generate_synthetic_point_sources(; truth_xyF, npix=32, sigma_psf=1.5, noise=0.05, seed=42)
        -> (prob, truth_x, truth_y, truth_F, clean_obs)

`truth_xyF` is a length-N vector of (x, y, F) tuples. The prior is set up to
be loose (±5 pixels around each true position) but informative enough that
multi-source identification doesn't get hopelessly confused.
"""
function generate_synthetic_point_sources(;
        truth_xyF::Vector{<:NTuple{3, Real}}=[
            (10.0, 12.0, 1.0),
            (20.0, 18.0, 0.6),
            (15.0, 25.0, 0.8),
        ],
        npix::Int=32,
        sigma_psf::Real=1.5,
        noise::Real=0.05,
        seed::Int=42,
        T::Type{<:AbstractFloat}=Float64)
    Random.seed!(seed)
    N = length(truth_xyF)
    truth_x = T[t[1] for t in truth_xyF]
    truth_y = T[t[2] for t in truth_xyF]
    truth_F = T[t[3] for t in truth_xyF]

    # Synthesize the noise-free observation
    inv2σ2 = inv(2 * T(sigma_psf)^2)
    clean_obs = zeros(T, npix, npix)
    @inbounds for j in 1:npix, i in 1:npix
        s = zero(T)
        for k in 1:N
            s += truth_F[k] * exp(-((i - truth_x[k])^2 + (j - truth_y[k])^2) * inv2σ2)
        end
        clean_obs[i, j] = s
    end
    data = clean_obs .+ T(noise) .* randn(T, npix, npix)

    prob = PointSourcesProblem(
        npix, T(sigma_psf), data, T(noise),
        N,
        # Prior centered ~3 pixels off the truth (so MAP has work to do but
        # doesn't have to globally search):
        truth_x .+ T(3) .* (rand(T, N) .- T(1//2)),
        truth_y .+ T(3) .* (rand(T, N) .- T(1//2)),
        T(4),                                 # ±4 pixel position prior
        T(1),                                 # F prior scale (F ≈ exp(z))
    )
    return prob, truth_x, truth_y, truth_F, clean_obs
end
