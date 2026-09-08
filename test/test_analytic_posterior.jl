# Analytic linear-Gaussian posterior test — the only *posterior-correctness*
# check in the automated suite (every other end-to-end test only asserts
# `isfinite` + energy-decrease). Promotes what comparison/step1 validates into a
# self-contained CI test with NO Python / NIFTy dependency.
#
# For a LINEAR forward model T(z) = A z with standard-normal prior and whitened
# Gaussian likelihood, the posterior is exactly Gaussian:
#     M = AᵀA + I,   μ = M⁻¹ Aᵀ d_white,   Σ = M⁻¹
# The MAP minimizes 0.5‖A z − d_white‖² + 0.5‖z‖², so z_map == μ; and the MGVI
# linear sampler draws δ ~ N(0, Σ). DeconvolutionProblem is exactly linear.
#
# This is an ACCURACY test on a deliberately ill-conditioned Wiener problem
# (steep k^-3 spectrum), so it runs at Float64 only — Float32 accuracy on an
# ill-conditioned solve is capped at ~eps·cond and would make the ground-truth
# comparison meaningless. Float32 *numerical correctness* is covered instead by
# the dual-precision operator identities (test_operators), the CF core
# (test_multi_axis_corr_field), and the per-problem adjoint/FD-gradient tests.

using VarInf
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "deconvolution_problem.jl"))

# Dense matrix of a linear operator applied to basis vectors.
function _dense_op(op, nrow::Int, ncol::Int, ::Type{T}) where {T}
    A = Matrix{T}(undef, nrow, ncol)
    e = zeros(T, ncol)
    for i in 1:ncol
        fill!(e, zero(T)); e[i] = one(T)
        A[:, i] = op(e)
    end
    return A
end

@testset "Analytic linear-Gaussian posterior" begin
    T = Float64
    n = 8
    grid = FourierGridInfo(n, 1.0)
    cfg  = CorrFieldConfig(slope_prior=(-3.0, 0.0), fluct_prior=(1.0, 0.0))
    prob, = generate_synthetic_deconvolution(grid, cfg, 1.0; noise_frac=0.1, seed=11)

    nz = latent_size(prob)               # = n^2
    z0 = zeros(T, nz)
    # A = right_sqrt_metric (== Jacobian of the linear transformation).
    A = _dense_op(v -> right_sqrt_metric(prob, z0, v), data_size(prob), nz, T)
    d_white = vec(prob.data) ./ prob.sigma
    M = A' * A + I
    μ = M \ (A' * d_white)
    Σ = inv(Matrix(M))

    @testset "Analytic mean μ is a stationary point of the code's energy" begin
        # Exact, optimizer-independent: ∇E(z) = M z − Aᵀd_white, which is 0 at μ.
        # Verifies the code's energy/gradient matches the closed-form posterior.
        _, g = energy_and_gradient(prob, μ)
        @test norm(g) / max(norm(μ), one(T)) < adjoint_tol(T)
    end

    @testset "MAP optimizer converges toward μ" begin
        z_map = reconstruct_map(prob; z0=copy(z0), maxiter=2000,
                                gtol=(1e-12, 1e-12), verb=false)
        # vmlmb on this (steep-spectrum ⇒ ill-conditioned) problem reaches ~1e-3;
        # μ itself is pinned exactly by the stationarity test above.
        @test norm(z_map .- μ) / max(norm(μ), one(T)) < 5e-3
    end

    @testset "MGVI linear samples reproduce the analytic covariance Σ" begin
        Random.seed!(2024)
        ns = 3000
        # Draw δ ~ N(0, M⁻¹) at the posterior mean via the exact-metric CG sampler.
        samples, _, _ = VarInf._draw_samples_geovi(prob, μ, ns;
                                                   cg_maxiter=4nz, cg_tol=1e-10)
        S = reduce(hcat, samples)        # nz × ns, mean 0 by construction
        covE = (S * S') ./ ns
        # Marginal variances (the physically meaningful uncertainties) converge
        # fastest; the full matrix carries more off-diagonal MC noise.
        drel = norm(diag(covE) .- diag(Σ)) / norm(diag(Σ))
        frel = norm(covE .- Σ) / norm(Σ)
        @test drel < 0.12                # diag MC floor ~ sqrt(2/ns) ≈ 0.026
        @test frel < 0.25
    end
end
