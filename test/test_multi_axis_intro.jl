# Phase-3 test: MultiAxisIntroProblem on a small 2-axis grid. Mirrors
# test_intro_lognormal.jl's pattern but the problem delegates the
# correlated-field internals to a CorrelatedField composite.

using VarInf
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "multi_axis_intro_problem.jl"))

@testset "MultiAxisIntroProblem" begin
    # Small grids so finite-difference checks are tractable.
    sp_cfg = CorrFieldConfig(slope_prior=(-2.0, 0.5),
                              fluct_prior=(0.4, 0.04),
                              flex_prior=(1.0, 0.5),
                              asp_prior=(0.6, 0.06))
    sc_cfg = CorrFieldConfig(slope_prior=(-1.5, 0.3),
                              fluct_prior=(0.3, 0.03))
    prob, z_true, field_true, signal_true =
        generate_synthetic_multi_axis(; spatial_dim=8, spectral_dim=6,
            sp_cfg=sp_cfg, sc_cfg=sc_cfg,
            offset_prior=(0.0, 0.1), noise_frac=0.05, seed=42)

    @testset "Layout + forward sanity" begin
        @test latent_size(prob) == latent_size(prob.mcf)
        @test data_size(prob)   == prod(prob.mcf.field_shape)
        @test prob.mcf.field_shape == (8, 8, 6)
        @test all(>(0), signal_true)
        @test all(isfinite, prob.data)
    end

    # Precision-sensitive correctness checks run at BOTH Float32 and Float64.
    @testset "Adjoint identity + FD gradient ($T)" for T in (Float32, Float64)
        sp_cfgT = CorrFieldConfig(slope_prior=(T(-2), T(0.5)),
                                  fluct_prior=(T(0.4), T(0.04)),
                                  flex_prior=(one(T), T(0.5)),
                                  asp_prior=(T(0.6), T(0.06)))
        sc_cfgT = CorrFieldConfig(slope_prior=(T(-1.5), T(0.3)), fluct_prior=(T(0.3), T(0.03)))
        probT, = generate_synthetic_multi_axis(; spatial_dim=8, spectral_dim=6,
                                               sp_cfg=sp_cfgT, sc_cfg=sc_cfgT,
                                               offset_prior=(zero(T), T(0.1)),
                                               noise_frac=T(0.05), seed=42)
        rng = MersenneTwister(0)
        n = latent_size(probT)
        z0 = T(0.05) .* randn(rng, T, n)
        v  = randn(rng, T, n)
        w  = randn(rng, T, data_size(probT))
        Rv = right_sqrt_metric(probT, z0, v)
        Lw = left_sqrt_metric(probT, z0, w)
        @test abs(dot(Rv, w) - dot(v, Lw)) / max(abs(dot(Rv, w)), one(T)) < adjoint_tol(T)

        z1 = T(0.05) .* randn(rng, T, n)
        e0, g0 = energy_and_gradient(probT, z1)
        @test isfinite(e0)
        @test all(isfinite, g0)
        ε = fd_step(T)
        N_field = prod(probT.mcf.field_shape)
        m_sp = 4 + 2 * (probT.mcf.grids[1].n_bins - 2)
        sp_off = N_field; sc_off = N_field + m_sp
        indices = [rand(rng, 1:N_field),
                   sp_off + 1, sp_off + 2, sp_off + 3, sp_off + 4, sp_off + 5,
                   sc_off + 1, sc_off + 2, n]
        for i in indices
            zp = copy(z1); zp[i] += ε
            zm = copy(z1); zm[i] -= ε
            g_fd = (energy_and_gradient(probT, zp)[1] - energy_and_gradient(probT, zm)[1]) / (2ε)
            @test fd_grad_ok(T, g0[i], g_fd, e0)
        end
    end

    @testset "MAP improves on a noisy warm-start near truth" begin
        Random.seed!(2)
        z0 = z_true .+ 0.05 .* randn(length(z_true))
        e0, _ = energy_and_gradient(prob, z0)
        z_map = reconstruct_map(prob; z0=z0, maxiter=80,
                                 gtol=(1e-8, 1e-8), verb=false)
        e1, _ = energy_and_gradient(prob, z_map)
        @test e1 < e0
        @test all(isfinite, z_map)
    end

    @testset "GeoVI runs end-to-end and returns positive signal samples" begin
        Random.seed!(3)
        z0 = z_true .+ 0.05 .* randn(length(z_true))
        z_map = reconstruct_map(prob; z0=z0, maxiter=40, verb=false)
        z, samples = reconstruct_geovi(prob;
            z0=z_map, n_iterations=1, n_samples=2,
            map_maxiter=0, kl_maxiter=10, cg_maxiter=20, cg_tol=0.05,
            geo_newton_maxiter=3, geo_cg_maxiter=10, geo_tol=1e-4, verb=false)
        @test all(isfinite, z)
        @test length(samples) == 4
        for s in samples
            T = VarInf.transformation(prob, z .+ s)   # whitened signal
            @test all(isfinite, T)
            @test all(>(0), T)        # exp(field) > 0 always
        end
    end
end
