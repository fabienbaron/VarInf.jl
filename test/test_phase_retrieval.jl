# PhaseRetrievalProblem tests — exercises a deeply nonlinear forward chain
# (phase screen → |FFT|² → convolution) on a small grid for speed.

using VarInf
using Test
using Random
using LinearAlgebra
using FFTW

include(joinpath(@__DIR__, "..", "examples", "phase_retrieval_problem.jl"))

@testset "PhaseRetrievalProblem" begin
    n = 16          # small grid for fast FD checks
    Random.seed!(0)
    saturn   = abs.(randn(n, n)) .* 0.5 .+ 0.5   # bright, non-negative pseudo-object
    aperture = circular_aperture(n; radius_frac=0.4)
    prob, z_true, phase_true, _, data_clean =
        generate_synthetic_phase_retrieval(saturn;
            aperture=aperture, kolmogorov_strength=1.0,
            noise_frac=0.01, seed=42)

    @test latent_size(prob) == n * n
    @test data_size(prob)   == n * n

    # Precision-sensitive correctness checks run at BOTH Float32 and Float64.
    @testset "Adjoint identity + FD gradient ($T)" for T in (Float32, Float64)
        probT, = generate_synthetic_phase_retrieval(saturn; aperture=aperture,
                                                    kolmogorov_strength=1.0,
                                                    noise_frac=0.01, seed=42, T=T)
        rng = MersenneTwister(1)
        z0 = T(0.1) .* randn(rng, T, latent_size(probT))
        v  = randn(rng, T, latent_size(probT))
        w  = randn(rng, T, data_size(probT))
        Rv = right_sqrt_metric(probT, z0, v)
        Lw = left_sqrt_metric(probT, z0, w)
        @test abs(dot(Rv, w) - dot(v, Lw)) / max(abs(dot(Rv, w)), one(T)) < adjoint_tol(T)

        z1 = T(0.1) .* randn(rng, T, latent_size(probT))
        e0, g0 = energy_and_gradient(probT, z1)
        @test isfinite(e0)
        @test all(isfinite, g0)
        ε = fd_step(T)
        for _ in 1:10
            i = rand(rng, 1:latent_size(probT))
            zp = copy(z1); zp[i] += ε
            zm = copy(z1); zm[i] -= ε
            g_fd = (energy_and_gradient(probT, zp)[1] - energy_and_gradient(probT, zm)[1]) / (2ε)
            @test fd_grad_ok(T, g0[i], g_fd, e0)
        end
    end

    @testset "MAP improves on a slightly-noisy initialization near truth" begin
        # Warm-start with truth + small noise so we don't tangle with the
        # twin-image / phase-sign global ambiguity.
        Random.seed!(3)
        z0 = z_true .+ 0.05 .* randn(length(z_true))
        e0, _ = energy_and_gradient(prob, z0)
        z_map = reconstruct_map(prob; z0=z0, maxiter=80,
                                 gtol=(1e-8, 1e-8), verb=false)
        e1, _ = energy_and_gradient(prob, z_map)
        @test e1 < e0
        @test all(isfinite, z_map)
    end
end
