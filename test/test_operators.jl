# Operator-identity unit tests — the per-operator regression net that the
# comparison/ harness (end-to-end only) does not provide. Ported from the niche
# NIFTy's test_likelihood_impl.py covers. Checks, on both a LINEAR
# (DeconvolutionProblem) and a NONLINEAR (TwoGaussiansProblem) problem, at a
# random latent z:
#   • metric symmetry:    ⟨M v, w⟩ ≈ ⟨v, M w⟩
#   • Gauss-Newton form:  M == JᵀJ + I     (J = right_sqrt_metric, Jᵀ = left_sqrt_metric)
#   • jvp vs finite diff: right_sqrt_metric(z, ·) ≈ dT/dz  (central difference)
#
# These are analytic, cheap, need no NIFTy/Python, and are precision-parameterized
# on T — the strongest guard for the {T} refactor's metric/sqrt-metric path.

using VarInf
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "deconvolution_problem.jl"))
include(joinpath(@__DIR__, "..", "examples", "two_gaussians_problem.jl"))

# Apply the posterior metric M = JᵀJ + I via the public sqrt-metric operators.
_metric_mul(prob, z, v) = left_sqrt_metric(prob, z, right_sqrt_metric(prob, z, v)) .+ v

# Materialize a dense matrix for a linear operator `op(basis_vector) -> vector`.
function _dense(op, nrow::Int, ncol::Int, ::Type{T}) where {T}
    A = Matrix{T}(undef, nrow, ncol)
    e = zeros(T, ncol)
    for i in 1:ncol
        fill!(e, zero(T)); e[i] = one(T)
        A[:, i] = op(e)
    end
    return A
end

function check_operators(prob, ::Type{T}; label::String) where {T}
    n = latent_size(prob)
    m = data_size(prob)
    rng = MersenneTwister(1234)
    z = T(0.1) .* randn(rng, T, n)

    @testset "$label — metric symmetry" begin
        v = randn(rng, T, n); w = randn(rng, T, n)
        Mv = _metric_mul(prob, z, v)
        Mw = _metric_mul(prob, z, w)
        @test abs(dot(Mv, w) - dot(v, Mw)) / max(abs(dot(Mv, w)), one(T)) < adjoint_tol(T)
    end

    @testset "$label — M == JᵀJ + I" begin
        A = _dense(v -> right_sqrt_metric(prob, z, v), m, n, T)   # J
        M = _dense(v -> _metric_mul(prob, z, v),       n, n, T)   # M
        Mref = A' * A + I
        scale = max(one(T), maximum(abs, Mref))
        @test maximum(abs, M .- Mref) < adjoint_tol(T) * scale
        # left_sqrt_metric is the exact adjoint of right_sqrt_metric:
        L = _dense(w -> left_sqrt_metric(prob, z, w), n, m, T)    # Jᵀ
        @test maximum(abs, L .- A') < adjoint_tol(T) * max(one(T), maximum(abs, A))
    end

    @testset "$label — right_sqrt_metric == dT/dz (finite diff)" begin
        v = randn(rng, T, n)
        Jv = right_sqrt_metric(prob, z, v)
        ε = fd_step(T)
        Tp = transformation(prob, z .+ ε .* v)
        Tm = transformation(prob, z .- ε .* v)
        fd = (Tp .- Tm) ./ (2ε)
        @test norm(Jv .- fd) / max(norm(fd), one(T)) < fd_reltol(T)
    end
end

@testset "Operator identities" begin
    for T in (Float32, Float64)
        # LINEAR problem (self-adjoint Fourier operator)
        let n = 16
            grid = FourierGridInfo(n, one(T))
            cfg  = CorrFieldConfig(slope_prior=(T(-3), zero(T)),
                                   fluct_prior=(one(T), zero(T)))
            prob, = generate_synthetic_deconvolution(grid, cfg, one(T);
                                                     noise_frac=T(0.05), seed=7, T=T)
            check_operators(prob, T; label="Deconvolution (linear, $T)")
        end
        # NONLINEAR problem (no FFT, analytic Jacobian)
        let
            prob, = generate_synthetic_two_gaussians(; seed=7, T=T)
            check_operators(prob, T; label="TwoGaussians (nonlinear, $T)")
        end
    end
end
