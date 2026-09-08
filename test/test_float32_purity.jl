# Float32 type-purity tests — verify the {T} refactor runs fully in single
# precision with NO silent promotion to Float64, across problem families:
#   • DeconvolutionProblem   — linear, correlated-field kernel + FFT
#   • TwoGaussiansProblem    — nonlinear, no FFT / no correlated field
#   • MultiAxisIntroProblem  — composite CorrelatedField (2-D × 1-D) + FFT
#
# For each: every observable exchanged with the solver (transformation, energy,
# gradient, sqrt-metric, MAP result) must be Float32, and the forward must be
# type-stable (@inferred). A stray Float64 literal anywhere in the pipeline
# would surface here as an eltype mismatch or an @inferred failure.

using VarInf
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "deconvolution_problem.jl"))
include(joinpath(@__DIR__, "..", "examples", "two_gaussians_problem.jl"))
include(joinpath(@__DIR__, "..", "examples", "multi_axis_intro_problem.jl"))

# Assert every solver-facing observable of `prob` stays Float32 at latent `z`.
function check_float32_purity(prob, z)
    @test eltype(z) == Float32
    Tz = transformation(prob, z)
    @test eltype(Tz) == Float32
    e, g = energy_and_gradient(prob, z)
    @test e isa Float32
    @test eltype(g) == Float32
    v = randn(MersenneTwister(1), Float32, length(z))
    @test eltype(right_sqrt_metric(prob, z, v)) == Float32
    w = randn(MersenneTwister(2), Float32, data_size(prob))
    @test eltype(left_sqrt_metric(prob, z, w)) == Float32
    zmap = reconstruct_map(prob; z0=copy(z), maxiter=30, verb=false)
    @test eltype(zmap) == Float32
    # Type stability: the compiler can prove the forward returns Float32.
    @test eltype(@inferred(transformation(prob, z))) == Float32
end

@testset "Float32 type purity" begin
    @testset "DeconvolutionProblem (linear, FFT)" begin
        grid = FourierGridInfo(16, 1.0f0)
        cfg  = CorrFieldConfig(slope_prior=(-3.0f0, 0.0f0), fluct_prior=(1.0f0, 0.0f0))
        prob, = generate_synthetic_deconvolution(grid, cfg, 1.0f0;
                                                 noise_frac=0.05f0, seed=7, T=Float32)
        @test eltype(prob) == Float32
        check_float32_purity(prob, zeros(Float32, latent_size(prob)))
    end

    @testset "TwoGaussiansProblem (nonlinear)" begin
        prob, = generate_synthetic_two_gaussians(; seed=7, T=Float32)
        @test eltype(prob) == Float32
        check_float32_purity(prob, 0.1f0 .* randn(MersenneTwister(0), Float32,
                                                  latent_size(prob)))
    end

    @testset "MultiAxisIntroProblem (composite CF)" begin
        sp_cfg = CorrFieldConfig(slope_prior=(-2.0f0, 0.5f0), fluct_prior=(0.4f0, 0.04f0))
        sc_cfg = CorrFieldConfig(slope_prior=(-1.5f0, 0.3f0), fluct_prior=(0.3f0, 0.03f0))
        prob, = generate_synthetic_multi_axis(; spatial_dim=8, spectral_dim=6,
                                              sp_cfg=sp_cfg, sc_cfg=sc_cfg,
                                              offset_prior=(0.0f0, 0.1f0), seed=7)
        @test eltype(prob) == Float32
        check_float32_purity(prob, 0.1f0 .* randn(MersenneTwister(0), Float32,
                                                  latent_size(prob)))
    end
end
