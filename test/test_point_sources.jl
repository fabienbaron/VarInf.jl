# PointSourcesProblem tests — confirms the protocol works on a 2D
# parametric (non-correlated-field, non-FFT) inference problem.

using VarInf
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "point_sources_problem.jl"))

@testset "PointSourcesProblem" begin
    truth_xyF = [(10.0, 12.0, 1.0), (20.0, 18.0, 0.6), (15.0, 25.0, 0.8)]
    prob, tx, ty, tF, _ =
        generate_synthetic_point_sources(; truth_xyF=truth_xyF,
                                          npix=32, sigma_psf=1.5,
                                          noise=0.05, seed=42)

    @test latent_size(prob) == 9
    @test data_size(prob)   == 32 * 32

    # Precision-sensitive correctness checks run at BOTH Float32 and Float64.
    @testset "Adjoint identity + FD gradient ($T)" for T in (Float32, Float64)
        probT, = generate_synthetic_point_sources(; truth_xyF=truth_xyF, npix=32,
                                                  sigma_psf=1.5, noise=0.05, seed=42, T=T)
        rng = MersenneTwister(0)
        z0 = randn(rng, T, 9); v = randn(rng, T, 9); w = randn(rng, T, 32 * 32)
        Rv = right_sqrt_metric(probT, z0, v)
        Lw = left_sqrt_metric(probT, z0, w)
        @test abs(dot(Rv, w) - dot(v, Lw)) / max(abs(dot(Rv, w)), one(T)) < adjoint_tol(T)

        z1 = T(0.2) .* randn(rng, T, 9)
        e0, g0 = energy_and_gradient(probT, z1)
        @test isfinite(e0)
        @test all(isfinite, g0)
        ε = fd_step(T)
        for i in 1:9
            zp = copy(z1); zp[i] += ε
            zm = copy(z1); zm[i] -= ε
            g_fd = (energy_and_gradient(probT, zp)[1] - energy_and_gradient(probT, zm)[1]) / (2ε)
            @test fd_grad_ok(T, g0[i], g_fd, e0)
        end
    end

    @testset "MAP recovers source positions within ~1 pixel" begin
        z_map = reconstruct_map(prob; maxiter=500, gtol=(1e-8, 1e-8), verb=false)
        x_est, y_est, F_est = _unpack(z_map, prob)
        for k in 1:prob.n_sources
            @test abs(x_est[k] - tx[k]) < 1.0
            @test abs(y_est[k] - ty[k]) < 1.0
            @test F_est[k] > 0
            # Flux within 20% of truth
            @test abs(F_est[k] - tF[k]) / tF[k] < 0.2
        end
    end

    @testset "GeoVI runs end-to-end" begin
        z_map = reconstruct_map(prob; maxiter=300, verb=false)
        z, samples = reconstruct_geovi(
            prob; z0=z_map, n_iterations=2, n_samples=2,
            map_maxiter=0, kl_maxiter=15, cg_maxiter=20, cg_tol=0.05,
            geo_newton_maxiter=4, geo_cg_maxiter=10, geo_tol=1e-4, verb=false)
        @test all(isfinite, z)
        @test length(samples) == 4
        for s in samples
            x_s, y_s, F_s = _unpack(z .+ s, prob)
            @test all(isfinite, x_s)
            @test all(isfinite, y_s)
            @test all(>(0), F_s)
        end
    end
end
