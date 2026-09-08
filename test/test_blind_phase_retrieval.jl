# BlindPhaseRetrievalProblem tests — previously untested. The second learnable
# correlated-field problem: joint inference of an object (correlated-field prior
# with a learnable slope + additive offset) AND an unknown aberrating phase
# screen. Exercises the optional report_latents / latent_blocks / whitened_data
# hooks and the CF offset/azm path.
#
# Self-contained synthetic object (smooth positive blob) — no external FITS file.

using VarInf
using Test
using Random
using LinearAlgebra
using FFTW
using Statistics

include(joinpath(@__DIR__, "..", "examples", "blind_phase_retrieval_problem.jl"))

@testset "BlindPhaseRetrievalProblem" begin
    T = Float64
    n = 16
    xs = range(-1, 1, length=n)
    object = [exp(-((x)^2 + (y)^2) / 0.3) for x in xs, y in xs] .+ 0.2   # positive image

    grid_obj = FourierGridInfo(n, 1.0)
    cfg_obj  = CorrFieldConfig(slope_prior=(-2.0, 1.0), fluct_prior=(1.0, 0.3))
    cf_obj   = CorrelatedField([grid_obj], [cfg_obj];
                               offset_prior=(log(mean(object)), 0.3))

    prob, white_true, tstrength, phase_true, _, psf_true, data_clean =
        generate_synthetic_blind(object, cf_obj; true_strength=0.4,
                                 noise_frac=0.02, seed=3)

    @test data_size(prob) == n^2
    @test latent_size(prob) > n^2               # object field + hypers + phase white field
    @test all(isfinite, data_clean)
    @test all(isfinite, phase_true)

    # Optional hooks are wired
    @test length(latent_blocks(prob)) >= 1
    @test whitened_data(prob) !== nothing
    @test length(whitened_data(prob)) == n^2

    # Precision-sensitive correctness checks run at BOTH Float32 and Float64.
    @testset "Adjoint identity + FD gradient ($TT)" for TT in (Float32, Float64)
        grid_objT = FourierGridInfo(n, one(TT))
        cfg_objT  = CorrFieldConfig(slope_prior=(TT(-2), one(TT)), fluct_prior=(one(TT), TT(0.3)))
        cf_objT   = CorrelatedField([grid_objT], [cfg_objT];
                                    offset_prior=(TT(log(mean(object))), TT(0.3)))
        probT, = generate_synthetic_blind(object, cf_objT; true_strength=0.4,
                                          noise_frac=0.02, seed=3)
        rng = MersenneTwister(0)
        z = TT(0.1) .* randn(rng, TT, latent_size(probT))
        v = randn(rng, TT, latent_size(probT))
        w = randn(rng, TT, data_size(probT))
        Rv = right_sqrt_metric(probT, z, v)
        Lw = left_sqrt_metric(probT, z, w)
        @test abs(dot(Rv, w) - dot(v, Lw)) / max(abs(dot(Rv, w)), one(TT)) < adjoint_tol(TT)

        z1 = TT(0.1) .* randn(rng, TT, latent_size(probT))
        e0, g0 = energy_and_gradient(probT, z1)
        @test isfinite(e0)
        @test all(isfinite, g0)
        ε = fd_step(TT)
        for _ in 1:8
            i = rand(rng, 1:latent_size(probT))
            zp = copy(z1); zp[i] += ε
            zm = copy(z1); zm[i] -= ε
            g_fd = (energy_and_gradient(probT, zp)[1] - energy_and_gradient(probT, zm)[1]) / (2ε)
            @test fd_grad_ok(TT, g0[i], g_fd, e0)
        end
    end

    @testset "MAP improves on a noisy warm start" begin
        rng = MersenneTwister(2)
        z0 = T(0.05) .* randn(rng, T, latent_size(prob))
        e0, _ = energy_and_gradient(prob, z0)
        z_map = reconstruct_map(prob; z0=z0, maxiter=60, verb=false)
        e1, _ = energy_and_gradient(prob, z_map)
        @test e1 < e0
        @test all(isfinite, z_map)
    end
end
