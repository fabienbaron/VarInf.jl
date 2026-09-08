using VarInf
using Test
using Random
using LinearAlgebra

# ── Precision-scaled tolerance helpers ──────────────────────────────────────
# Every numeric tolerance in the suite is derived from `T` so the same test runs
# at Float32 and Float64. These collapse to ≈ the historical Float64 bars:
#   adjoint_tol(Float64) ≈ 1.5e-8,  fd_step(Float64) ≈ 6.1e-6,  fd_reltol ≈ 1.2e-4
#   adjoint_tol(Float32) ≈ 3.5e-4,  fd_step(Float32) ≈ 4.9e-3,  fd_reltol ≈ 9.8e-2
adjoint_tol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))   # attainable inner-product accuracy
fd_step(::Type{T})     where {T<:AbstractFloat} = cbrt(eps(T))   # optimal central-difference step
fd_reltol(::Type{T})   where {T<:AbstractFloat} = 20 * cbrt(eps(T))  # loose central-diff error bound

# Absolute allowance for central-difference cancellation noise.
#
# A central difference of a function of magnitude |E| with step h carries
# cancellation error of order |E|*eps(T)/(2h). With h = fd_step(T) = cbrt(eps(T))
# that is |E|*eps(T)^(2/3)/2, which for |E| ~ 1e4 at Float32 comes to = 0.12 -
# larger than fd_reltol(Float32) = 0.098. So a gradient component compared
# against max(|g_fd|, 1) is asserted BELOW the finite-difference noise floor
# whenever |g| <= 1, and the outcome then moves with anything that perturbs the
# Float32 summation order. `--check-bounds=yes`, which `Pkg.test()` passes by
# default, disables @inbounds and does exactly that: on
# test_blind_phase_retrieval the worst ratio goes 0.063 -> 0.197 against a 0.098
# bound, deterministically in both directions.
#
# `fd_grad_ok` ADDS this allowance to the previous threshold rather than
# replacing it, so it is strictly more permissive and cannot tighten any
# assertion that already passed. The factor 3 is slack on a derived scale, in
# the spirit of the 20 in fd_reltol.
fd_abstol(::Type{T}, energy::Real) where {T<:AbstractFloat} =
    3 * abs(energy) * eps(T) / (2 * fd_step(T))

# Gradient-component check: the original relative bound, plus the noise floor.
fd_grad_ok(::Type{T}, g::Real, g_fd::Real, energy::Real) where {T<:AbstractFloat} =
    abs(g - g_fd) <= fd_reltol(T) * max(abs(g_fd), one(T)) + fd_abstol(T, energy)

@testset "VarInf" begin
    @testset "CG solver ($T)" for T in (Float32, Float64)
        # Tiny SPD system — CG should match the direct solve to eps(T).
        A = T[4 1; 1 3]
        b = T[1, 2]
        x_cg, _ = VarInf._nifty_cg(v -> A * v, b; tol=sqrt(eps(T)))
        @test eltype(x_cg) == T
        @test norm(x_cg - (A \ b)) < sqrt(eps(T))
    end

    @testset "Newton-CG optimizer ($T)" for T in (Float32, Float64)
        # Quadratic: argmin 0.5*x'Ax - b'x = A\b
        A = T[4 1; 1 3]
        b = T[1, 2]
        fg(x) = (dot(x, A * x) / 2 - dot(b, x), A * x .- b)
        hp(_, v) = A * v
        x_opt = VarInf._nifty_newton_cg(fg, hp, zeros(T, 2); maxiter=20, xtol=sqrt(eps(T)))
        @test eltype(x_opt) == T
        @test norm(x_opt - (A \ b)) < sqrt(eps(T))
    end

    include("test_deconvolution.jl")
    include("test_two_gaussians.jl")
    include("test_point_sources.jl")
    include("test_phase_retrieval.jl")
    include("test_intro_lognormal.jl")
    include("test_axis1d.jl")
    include("test_multi_axis_corr_field.jl")
    include("test_multi_axis_intro.jl")

    # ── Step-0 additions (test-driven Float32 refactor) ──────────────────────
    # Class-(A) characterization tests: must pass on the current Float64 code.
    include("test_operators.jl")           # metric symmetry, M == JᵀJ + I, jvp-vs-FD
    include("test_analytic_posterior.jl")  # linear-Gaussian posterior mean/cov vs closed form
    include("test_ncg.jl")                 # minimizer correctness on ill-conditioned systems
    include("test_phase_retrieval_latent.jl")  # learnable-CF problem (was untested)
    include("test_blind_phase_retrieval.jl")   # blind learnable-CF problem (was untested)
    # Class-(B) target-spec: Float32 path + type purity. Expected-broken until the
    # {T} refactor lands (Step 7 removes the guards).
    include("test_float32_purity.jl")
    include("test_minisanity.jl")    # minisanity statistic + hybrid scheduling
    include("test_prior_covariance.jl")  # optional non-unit prior covariance
    include("test_matern.jl")            # Matérn amplitude port (NIFTy.re parity)
end
