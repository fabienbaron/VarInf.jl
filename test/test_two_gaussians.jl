# TwoGaussiansProblem tests — confirms the protocol works on a parametric
# (non-correlated-field, no-FFT) inference problem.

using VarInf
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "two_gaussians_problem.jl"))

@testset "TwoGaussiansProblem" begin
    truth_tuple = (-1.5, 0.7, 1.0, 1.8, 1.0, 1.2)
    prob, _, _, _ = generate_synthetic_two_gaussians(;
        truth=truth_tuple, n_obs=100, noise=0.05, seed=42)

    @test latent_size(prob) == 6
    @test data_size(prob)   == 100

    # Precision-sensitive correctness checks run at BOTH Float32 and Float64.
    @testset "Adjoint identity + FD gradient ($T)" for T in (Float32, Float64)
        probT, = generate_synthetic_two_gaussians(; truth=truth_tuple, n_obs=100,
                                                  noise=0.05, seed=42, T=T)
        rng = MersenneTwister(0)
        z0 = randn(rng, T, 6); v = randn(rng, T, 6); w = randn(rng, T, 100)
        Rv = right_sqrt_metric(probT, z0, v)
        Lw = left_sqrt_metric(probT, z0, w)
        @test abs(dot(Rv, w) - dot(v, Lw)) / max(abs(dot(Rv, w)), one(T)) < adjoint_tol(T)

        z1 = T(0.3) .* randn(rng, T, 6)
        e0, g0 = energy_and_gradient(probT, z1)
        @test isfinite(e0)
        @test all(isfinite, g0)
        ε = fd_step(T)
        for i in 1:6
            zp = copy(z1); zp[i] += ε
            zm = copy(z1); zm[i] -= ε
            g_fd = (energy_and_gradient(probT, zp)[1] - energy_and_gradient(probT, zm)[1]) / (2ε)
            @test fd_grad_ok(T, g0[i], g_fd, e0)
        end
    end

    @testset "MAP recovers physical parameters near truth" begin
        z_map = reconstruct_map(prob; maxiter=200, verb=false)
        phys = _unpack(z_map, prob)
        # μ accuracy: with prior std 2 and S/N of ~20 per peak, the MAP
        # should land within 0.3 of each true μ.
        @test abs(phys[1] - truth_tuple[1]) < 0.3   # μ₁
        @test abs(phys[4] - truth_tuple[4]) < 0.3   # μ₂
        # Amplitude positivity is automatic (A = exp(z)).
        @test phys[3] > 0
        @test phys[6] > 0
    end

    @testset "GeoVI runs end-to-end and returns finite physical posterior" begin
        z_map = reconstruct_map(prob; maxiter=100, verb=false)
        z, samples = reconstruct_geovi(
            prob; z0=z_map, n_iterations=2, n_samples=2,
            map_maxiter=0, kl_maxiter=15, cg_maxiter=20, cg_tol=0.05,
            geo_newton_maxiter=4, geo_cg_maxiter=10, geo_tol=1e-4, verb=false)
        @test all(isfinite, z)
        @test length(samples) == 4
        for s in samples
            phys = _unpack(z .+ s, prob)
            @test all(isfinite, phys)
            @test phys[2] > 0 && phys[5] > 0   # σ stays positive
            @test phys[3] > 0 && phys[6] > 0   # A stays positive
        end
    end
end
