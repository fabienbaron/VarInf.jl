# LogNormalFieldProblem tests — exercises the only example so far that
# learns correlated-field hyperparameters (slope / fluct / flex / asperity /
# per-bin spectrum). On a small grid for FD tractability.

using VarInf
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "intro_lognormal_problem.jl"))

@testset "LogNormalFieldProblem" begin
    n = 8                                # smallest grid that still has IWP bins
    cfg = CorrFieldConfig(slope_prior=(-2.0, 0.5),
                          fluct_prior=(0.4, 0.04),
                          flex_prior=(1.0, 0.5),
                          asp_prior=(0.6, 0.06),
                          offset_prior=(0.0, 0.1))
    prob, z_true, signal_true, amp_true =
        generate_synthetic_lognormal(; npix=n, dx=1.0/n,
            scaling_mean=2.0, scaling_std=0.5,
            cfg=cfg, noise_frac=0.05, seed=42)

    N = n * n
    n_spec = 2 * (prob.grid.n_bins - 2)

    @testset "Latent layout" begin
        @test latent_size(prob) == N + 4 + n_spec + 2
        @test data_size(prob)   == N
        @test prob.cfg.use_iwp
        @test prob.cfg.use_offset
    end

    # Entries to FD-check: one per latent block + several xi_field samples.
    function _checked_indices(p, n_field_samples::Int=8)
        N = p.grid.npix^2
        n_spec = 2 * (p.grid.n_bins - 2)
        indices = Int[]
        for i in shuffle(MersenneTwister(99), 1:N)[1:min(n_field_samples, N)]
            push!(indices, i)
        end
        append!(indices, [N + 1, N + 2, N + 3, N + 4])      # slope, fluct, flex, asp
        append!(indices, [N + 5, N + 4 + n_spec ÷ 2, N + 4 + n_spec])  # xi_spectrum
        push!(indices, N + 5 + n_spec)                      # xi_offset
        push!(indices, N + 4 + n_spec + 2)                  # xi_scaling
        return indices
    end

    # Precision-sensitive correctness checks run at BOTH Float32 and Float64.
    @testset "Adjoint identity + FD gradient ($T)" for T in (Float32, Float64)
        cfgT = CorrFieldConfig(slope_prior=(T(-2), T(0.5)),
                               fluct_prior=(T(0.4), T(0.04)),
                               flex_prior=(one(T), T(0.5)),
                               asp_prior=(T(0.6), T(0.06)),
                               offset_prior=(zero(T), T(0.1)))
        probT, = generate_synthetic_lognormal(; npix=n, dx=inv(T(n)),
                                              scaling_mean=T(2), scaling_std=T(0.5),
                                              cfg=cfgT, noise_frac=T(0.05), seed=42)
        rng = MersenneTwister(0)
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
        for i in _checked_indices(probT)
            zp = copy(z1); zp[i] += ε
            zm = copy(z1); zm[i] -= ε
            g_fd = (energy_and_gradient(probT, zp)[1] - energy_and_gradient(probT, zm)[1]) / (2ε)
            @test fd_grad_ok(T, g0[i], g_fd, e0)
        end
    end

    @testset "MAP improves on a noisy warm-start near truth" begin
        Random.seed!(3)
        z0 = z_true .+ 0.05 .* randn(length(z_true))
        e0, _ = energy_and_gradient(prob, z0)
        z_map = reconstruct_map(prob; z0=z0, maxiter=80,
                                 gtol=(1e-8, 1e-8), verb=false)
        e1, _ = energy_and_gradient(prob, z_map)
        @test e1 < e0
        @test all(isfinite, z_map)
    end

    @testset "GeoVI runs end-to-end with positive signal samples" begin
        Random.seed!(5)
        z0 = z_true .+ 0.05 .* randn(length(z_true))
        z_map = reconstruct_map(prob; z0=z0, maxiter=40, verb=false)
        z, samples = reconstruct_geovi(prob;
            z0=z_map, n_iterations=1, n_samples=2,
            map_maxiter=0, kl_maxiter=10, cg_maxiter=20, cg_tol=0.05,
            geo_newton_maxiter=3, geo_cg_maxiter=10, geo_tol=1e-4, verb=false)
        @test all(isfinite, z)
        @test length(samples) == 4   # antithetic refinements of 2 base samples
        for s in samples
            fwd = VarInf.transformation(prob, z .+ s)   # whitened signal
            @test all(isfinite, fwd)
            # exp(field)·scaling is strictly positive
            @test all(>(0), fwd)
        end
    end
end
