# ============================================================================
# MultiAxisIntroProblem: VarInf's analogue of NIFTy 0_intro extended to two
# axes (spatial 2-D + spectral 1-D). Compared to `intro_lognormal_problem.jl`,
# this one delegates ALL correlated-field machinery (per-axis hyperparameters,
# composite Fourier kernel, global offset, JVP/VJP through them) to a
# single `CorrelatedField` instance; the problem itself only adds the
# log-normal transform on top.
#
# Forward chain:
#     z ──┐
#         │
#         ▼
#       mcf(z) → 3-D real field           (CorrelatedField composite)
#         │
#         ▼
#     signal = exp(field)                  (per-pixel positivity)
#         │
#         ▼
#     data = signal + N(0, σ²)             (identity forward + Gaussian noise)
#
# Latent layout: whatever `CorrelatedField` says; the problem doesn't unpack
# z explicitly, just hands it to `mcf` / `mcf_jvp` / `mcf_vjp`.
# ============================================================================

using VarInf
using LinearAlgebra
using Random

struct MultiAxisIntroProblem{T<:AbstractFloat,CF<:CorrelatedField{T}} <: AbstractInferenceProblem
    mcf::CF                      # concrete CF type ⇒ p.mcf stays type-stable
    data::Array{T}               # shape == mcf.field_shape
    sigma_noise::T
end

Base.eltype(::MultiAxisIntroProblem{T}) where {T} = T

VarInf.latent_size(p::MultiAxisIntroProblem) = latent_size(p.mcf)
VarInf.data_size(p::MultiAxisIntroProblem)   = prod(p.mcf.field_shape)

# Cache forward intermediates so JVP/VJP don't recompute the field.
struct _MAFwd{T<:AbstractFloat}
    field::Array{T}
    signal::Array{T}
end

function _forward(z::AbstractVector{<:Real}, p::MultiAxisIntroProblem)
    field = p.mcf(z)
    signal = exp.(field)
    return _MAFwd(field, signal)
end

function VarInf.transformation(p::MultiAxisIntroProblem,
                                z::AbstractVector{<:Real})
    fwd = _forward(z, p)
    return vec(fwd.signal) ./ p.sigma_noise
end

function VarInf.right_sqrt_metric(p::MultiAxisIntroProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    fwd = _forward(z, p)
    d_field = mcf_jvp(p.mcf, z, v)
    d_signal = fwd.signal .* d_field
    return vec(d_signal) ./ p.sigma_noise
end

function VarInf.left_sqrt_metric(p::MultiAxisIntroProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    fwd = _forward(z, p)
    w_scaled = reshape(w ./ p.sigma_noise, p.mcf.field_shape)
    g_field = w_scaled .* fwd.signal     # chain through exp
    return mcf_vjp(p.mcf, z, g_field)
end

function VarInf.energy_and_gradient(p::MultiAxisIntroProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = vec(p.data) ./ p.sigma_noise
    resid = T_z .- d_white
    chi2 = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end


# Convenience: synthetic data generator

"""
    generate_synthetic_multi_axis(; spatial_dim, spectral_dim, sp_cfg, sc_cfg,
                                    offset_prior, noise_frac, seed)
        -> (prob, z_true, field_true, signal_true)
"""
function generate_synthetic_multi_axis(;
        spatial_dim::Int=32,
        spectral_dim::Int=8,
        sp_cfg::CorrFieldConfig=CorrFieldConfig(slope_prior=(-2.0, 0.5),
                                                 fluct_prior=(0.4, 0.04),
                                                 flex_prior=(1.0, 0.5),
                                                 asp_prior=(0.6, 0.06)),
        sc_cfg::CorrFieldConfig=CorrFieldConfig(slope_prior=(-1.5, 0.3),
                                                 fluct_prior=(0.3, 0.03)),
        offset_prior::Tuple{Real,Real}=(0.0, 0.1),
        noise_frac::Real=0.05,
        seed::Int=42,
        T::Type{<:AbstractFloat}=eltype(sp_cfg))
    Random.seed!(seed)
    spatial  = FourierGridInfo(spatial_dim, inv(T(spatial_dim)))
    spectral = Axis1DInfo(spectral_dim, inv(T(spectral_dim)))
    mcf = CorrelatedField([spatial, spectral], [sp_cfg, sc_cfg];
                           offset_prior=offset_prior)
    n_z = latent_size(mcf)

    # Stub problem to compute synthetic ground truth
    stub = MultiAxisIntroProblem(mcf, zeros(T, mcf.field_shape), one(T))
    z_true = randn(T, n_z)
    fwd_true = _forward(z_true, stub)
    sigma_n = T(noise_frac) * maximum(fwd_true.signal)
    data = fwd_true.signal .+ sigma_n .* randn(T, mcf.field_shape...)

    prob = MultiAxisIntroProblem(mcf, data, sigma_n)
    return prob, z_true, fwd_true.field, fwd_true.signal
end
