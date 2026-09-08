# ============================================================================
# TwoGaussiansProblem: fit a sum of two 1D Gaussians to noisy observations.
#
# This example exercises the AbstractInferenceProblem protocol WITHOUT any
# correlated-field machinery, no FFTs, no Fourier grid. It is the cleanest
# evidence that VarInf.jl is a general variational-inference toolkit, not
# specifically a correlated-field one.
#
# Model:
#   f(x; θ) = A₁ exp(-(x-μ₁)² / (2σ₁²)) + A₂ exp(-(x-μ₂)² / (2σ₂²))
#   y_k ~ N(f(x_k), σ_noise²)
#
# Latent layout (6 entries, z ~ N(0, I)):
#   z[1] → μ₁  via  μ₁ = μ_mean[1] + μ_std * z[1]
#   z[2] → σ₁  via  σ₁ = σ_scale * exp(z[2])
#   z[3] → A₁  via  A₁ = A_scale * exp(z[3])
#   z[4..6] → μ₂, σ₂, A₂  (same transforms)
# ============================================================================

using VarInf
using LinearAlgebra
using Random

struct TwoGaussiansProblem{T<:AbstractFloat} <: AbstractInferenceProblem
    x::Vector{T}           # observation locations
    data::Vector{T}        # observed y values
    sigma_noise::T         # noise std
    μ_mean::Vector{T}      # prior mean for μ₁, μ₂ (length 2)
    μ_std::T               # prior std for μ (shared)
    σ_scale::T             # log-σ prior scale (exp(z) is Lognormal)
    A_scale::T             # log-A prior scale
end

Base.eltype(::TwoGaussiansProblem{T}) where {T} = T

VarInf.latent_size(::TwoGaussiansProblem) = 6
VarInf.data_size(p::TwoGaussiansProblem)  = length(p.x)

# Internal: latent z → physical (μ, σ, A) for each Gaussian component
function _unpack(z::AbstractVector{<:Real}, p::TwoGaussiansProblem)
    μ1 = p.μ_mean[1] + p.μ_std * z[1]
    σ1 = p.σ_scale * exp(z[2])
    A1 = p.A_scale * exp(z[3])
    μ2 = p.μ_mean[2] + p.μ_std * z[4]
    σ2 = p.σ_scale * exp(z[5])
    A2 = p.A_scale * exp(z[6])
    return μ1, σ1, A1, μ2, σ2, A2
end

# Forward model: f(x; z) evaluated at all observation locations
function _model(z::AbstractVector{<:Real}, p::TwoGaussiansProblem)
    μ1, σ1, A1, μ2, σ2, A2 = _unpack(z, p)
    f = similar(p.x)
    @inbounds for k in eachindex(p.x)
        x = p.x[k]
        e1 = exp(-(x - μ1)^2 / (2 * σ1^2))
        e2 = exp(-(x - μ2)^2 / (2 * σ2^2))
        f[k] = A1 * e1 + A2 * e2
    end
    return f
end

VarInf.transformation(p::TwoGaussiansProblem, z::AbstractVector{<:Real}) =
    _model(z, p) ./ p.sigma_noise

# JVP: J · v.  Forward-mode through the parameter unpack + Gaussian sums.
function VarInf.right_sqrt_metric(p::TwoGaussiansProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    μ1, σ1, A1, μ2, σ2, A2 = _unpack(z, p)
    # Tangent of the latent → physical transform:
    dμ1 = p.μ_std * v[1]
    dσ1 = σ1 * v[2]   # σ = scale * exp(z) ⇒ dσ = σ * dz
    dA1 = A1 * v[3]
    dμ2 = p.μ_std * v[4]
    dσ2 = σ2 * v[5]
    dA2 = A2 * v[6]

    df = similar(p.x)
    @inbounds for k in eachindex(p.x)
        x = p.x[k]
        # Gaussian 1
        u1 = (x - μ1) / σ1
        e1 = exp(-u1^2 / 2)
        g1 = A1 * e1
        dg1 = g1 * (u1 / σ1) * dμ1 + g1 * (u1^2 / σ1) * dσ1 + e1 * dA1
        # Gaussian 2
        u2 = (x - μ2) / σ2
        e2 = exp(-u2^2 / 2)
        g2 = A2 * e2
        dg2 = g2 * (u2 / σ2) * dμ2 + g2 * (u2^2 / σ2) * dσ2 + e2 * dA2
        df[k] = dg1 + dg2
    end
    return df ./ p.sigma_noise
end

# VJP: J' · w.  Reverse-mode: accumulate gradients into the physical params,
# then chain through the latent transform.
function VarInf.left_sqrt_metric(p::TwoGaussiansProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    μ1, σ1, A1, μ2, σ2, A2 = _unpack(z, p)
    w_scaled = w ./ p.sigma_noise

    Tz = eltype(p)
    g_μ1 = zero(Tz);  g_σ1 = zero(Tz);  g_A1 = zero(Tz)
    g_μ2 = zero(Tz);  g_σ2 = zero(Tz);  g_A2 = zero(Tz)
    @inbounds for k in eachindex(p.x)
        x = p.x[k]
        u1 = (x - μ1) / σ1
        e1 = exp(-u1^2 / 2)
        g1 = A1 * e1
        wk = w_scaled[k]
        g_μ1 += wk * g1 * (u1 / σ1)
        g_σ1 += wk * g1 * (u1^2 / σ1)
        g_A1 += wk * e1

        u2 = (x - μ2) / σ2
        e2 = exp(-u2^2 / 2)
        g2 = A2 * e2
        g_μ2 += wk * g2 * (u2 / σ2)
        g_σ2 += wk * g2 * (u2^2 / σ2)
        g_A2 += wk * e2
    end
    # Chain through latent → physical (Jacobian is diagonal in this layout):
    return [g_μ1 * p.μ_std,
            g_σ1 * σ1,           # ∂σ/∂z = σ
            g_A1 * A1,           # ∂A/∂z = A
            g_μ2 * p.μ_std,
            g_σ2 * σ2,
            g_A2 * A2]
end

function VarInf.energy_and_gradient(p::TwoGaussiansProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = p.data ./ p.sigma_noise
    resid = T_z .- d_white
    chi2  = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end

# Convenience: generate a synthetic two-Gaussian dataset

"""
    generate_synthetic_two_gaussians(; truth=(μ₁,σ₁,A₁,μ₂,σ₂,A₂), n_obs=100, noise=0.05, seed=42)
        -> (prob, truth_params, x, y_clean)
"""
function generate_synthetic_two_gaussians(;
        truth::NTuple{6, Real}=(-1.5, 0.7, 1.0, 1.8, 1.0, 1.2),
        x_range::Tuple{Real,Real}=(-5.0, 5.0),
        n_obs::Int=100,
        noise::Real=0.05,
        seed::Int=42,
        T::Type{<:AbstractFloat}=Float64)
    Random.seed!(seed)
    x = collect(T, range(x_range[1], x_range[2], length=n_obs))
    μ₁, σ₁, A₁, μ₂, σ₂, A₂ = T.(truth)
    y_clean = @. A₁ * exp(-(x - μ₁)^2 / (2 * σ₁^2)) +
                  A₂ * exp(-(x - μ₂)^2 / (2 * σ₂^2))
    data = y_clean .+ T(noise) .* randn(T, n_obs)

    # Prior priors centered around the right region but loose enough to be
    # informative-but-not-cheating.
    prob = TwoGaussiansProblem(
        x, data, T(noise),
        T[-2.0, 2.0],          # μ prior means (rough region of each peak)
        T(2),                   # μ prior std
        T(1),                   # σ prior scale (σ = exp(z) ≈ 1)
        T(1),                   # A prior scale (A = exp(z) ≈ 1)
    )
    return prob, truth, x, y_clean
end
