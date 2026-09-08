# DeconvolutionProblem tests — exercises the AbstractInferenceProblem protocol
# end-to-end on a problem that has nothing to do with interferometry.

using VarInf
using Test
using Random
using LinearAlgebra

# Pull in the example DeconvolutionProblem definition
include(joinpath(@__DIR__, "..", "examples", "deconvolution_problem.jl"))

@testset "DeconvolutionProblem" begin
    n = 16
    grid = FourierGridInfo(n, 1.0)
    cfg  = CorrFieldConfig(slope_prior=(-3.0, 0.0), fluct_prior=(1.0, 0.0))
    prob, z_true, image_true, _ =
        generate_synthetic_deconvolution(grid, cfg, 1.0; noise_frac=0.05, seed=42)

    @test latent_size(prob) == n^2
    @test data_size(prob) == n^2
    @test prob.sigma > 0
    @test all(isfinite, prob.data)
    @test all(isfinite, image_true)

    # Precision-sensitive correctness checks run at BOTH Float32 and Float64.
    @testset "Adjoint identity + FD gradient ($T)" for T in (Float32, Float64)
        gridT = FourierGridInfo(n, one(T))
        cfgT  = CorrFieldConfig(slope_prior=(T(-3), zero(T)), fluct_prior=(one(T), zero(T)))
        probT, = generate_synthetic_deconvolution(gridT, cfgT, one(T);
                                                  noise_frac=T(0.05), seed=42, T=T)
        rng = MersenneTwister(0)
        z0 = randn(rng, T, latent_size(probT))
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

    @testset "MAP reconstruction tracks the noise-free observation" begin
        z_map = reconstruct_map(prob; maxiter=300, verb=false)
        @test all(isfinite, z_map)
        # Compare in MODEL space, not latent space: the latent has unobservable
        # high-frequency modes that the Wiener-filter MAP correctly shrinks to
        # ~0, so ||z_map - z_true|| stays large by design. The model output is
        # what's actually constrained by the data — it should match the
        # noise-free observation to within roughly the noise scale.
        n = prob.grid.npix
        model_map = real.(ifft(fft(reshape(z_map, n, n)) .* prob.combined_kernel))
        # The clean (noise-free) observation = forward(z_true).
        model_true = real.(ifft(fft(reshape(z_true, n, n)) .* prob.combined_kernel))
        rel_err = norm(model_map .- model_true) / norm(model_true)
        @test rel_err < 0.5
        # Also: residual fit to actual data should be near the noise level
        chi2_per_pixel = sum(abs2, model_map .- prob.data) / (prob.sigma^2 * n^2)
        @test chi2_per_pixel < 5.0  # generous: should be ~1 for a well-converged MAP
    end

    @testset "MGVI returns finite antithetic samples" begin
        z, samples = reconstruct_mgvi(prob;
                                      n_iterations=2, n_samples=2,
                                      map_maxiter=50, kl_maxiter=20,
                                      cg_maxiter=20, cg_tol=0.1,
                                      verb=false)
        @test all(isfinite, z)
        # n_samples=2 → 2 base × 2 antithetic (±δ, NIFTy mirror) = 4 stored samples
        @test length(samples) == 4
        @test all(s -> all(isfinite, s), samples)
    end

    @testset "GeoVI returns finite antithetic samples" begin
        z, samples = reconstruct_geovi(prob;
                                        n_iterations=1, n_samples=2,
                                        map_maxiter=50, kl_maxiter=10,
                                        cg_maxiter=20, cg_tol=0.1,
                                        geo_newton_maxiter=3, geo_cg_maxiter=10,
                                        verb=false)
        @test all(isfinite, z)
        # n_samples=2 → 2 base × 2 antithetic = 4 stored samples
        @test length(samples) == 4
        @test all(s -> all(isfinite, s), samples)
    end

    @testset "Hybrid returns finite samples" begin
        z, samples = reconstruct_hybrid(prob;
                                         n_mgvi=1, n_geovi=1, n_samples=2,
                                         map_maxiter=50, kl_maxiter=10,
                                         cg_maxiter=20, cg_tol=0.1,
                                         geo_newton_maxiter=3, geo_cg_maxiter=10,
                                         verb=false)
        @test all(isfinite, z)
        @test length(samples) == 4
        @test all(s -> all(isfinite, s), samples)
    end
end
