# Tests for Axis1DInfo — the 1-D sibling of FourierGridInfo. The
# amplitude_spectrum* family is duck-typed on the grid argument, so the
# substance of these tests is verifying that
#   • axis_mode_distributor returns sensible (multiplicities, mode lengths)
#   • amplitude_spectrum / _jvp / _adjoint all run on an Axis1DInfo
#   • adjoint identity ⟨J·v, w⟩ ≈ ⟨v, J'·w⟩ holds for amplitude_spectrum_*
# Run at both Float32 and Float64 (tolerances scale with eps(T)).

using VarInf
using Test
using LinearAlgebra
using Random

@testset "Axis1DInfo + 1-D amplitude_spectrum ($T)" for T in (Float32, Float64)
    npix = 16
    dx = inv(T(npix))
    axis = Axis1DInfo(npix, dx)

    @testset "Mode distributor sanity" begin
        @test axis.npix == npix
        @test axis.dx == dx
        @test eltype(axis) == T
        @test eltype(axis.mode_lengths) == T
        @test length(axis.bin_index) == npix
        @test all(axis.bin_index .>= 1)
        @test maximum(axis.bin_index) == axis.n_bins
        @test sum(axis.mode_multiplicity) == npix
        @test axis.mode_lengths[1] ≈ 0 atol=sqrt(eps(T))
        @test axis.mode_multiplicity[1] == 1
        @test axis.rel_log_mode_lengths[1] == 0
        @test length(axis.log_volume) == axis.n_bins - 2
        @test axis.total_volume ≈ npix * dx
    end

    cfg = CorrFieldConfig(slope_prior=(T(-2), T(0.5)),
                          fluct_prior=(T(0.4), T(0.04)),
                          flex_prior=(one(T), T(0.5)),
                          asp_prior=(T(0.6), T(0.06)),
                          offset_prior=(zero(T), T(0.1)))

    @testset "amplitude_spectrum runs on 1-D axis" begin
        rng = MersenneTwister(0)
        xi_spectrum = reshape(T(0.1) .* randn(rng, T, 2 * (axis.n_bins - 2)), axis.n_bins - 2, 2)
        amp = amplitude_spectrum(T(0.1), T(0.2), T(-0.1), T(0.3), xi_spectrum, cfg, axis;
                                  xi_offset=T(0.2))
        @test length(amp) == axis.n_bins
        @test eltype(amp) == T
        @test all(isfinite, amp)
        @test all(amp .>= 0)
    end

    @testset "JVP vs finite differences" begin
        rng = MersenneTwister(1)
        xi_slope, xi_fluct, xi_flex, xi_asp, xi_offset = randn(rng, T, 5) .* T(0.3)
        xi_spectrum = T(0.1) .* randn(rng, T, axis.n_bins - 2, 2)
        v_slope, v_fluct, v_flex, v_asp, v_offset = randn(rng, T, 5)
        v_spectrum = randn(rng, T, axis.n_bins - 2, 2)

        d_amp_jvp = amplitude_spectrum_jvp(v_slope, v_fluct, v_flex, v_asp, v_spectrum,
                                            xi_slope, xi_fluct, xi_flex, xi_asp,
                                            xi_spectrum, cfg, axis;
                                            xi_offset=xi_offset, v_offset=v_offset)
        ε = fd_step(T)
        amp_p = amplitude_spectrum(xi_slope + ε*v_slope, xi_fluct + ε*v_fluct,
                                    xi_flex + ε*v_flex, xi_asp + ε*v_asp,
                                    xi_spectrum .+ ε .* v_spectrum, cfg, axis;
                                    xi_offset=xi_offset + ε*v_offset)
        amp_m = amplitude_spectrum(xi_slope - ε*v_slope, xi_fluct - ε*v_fluct,
                                    xi_flex - ε*v_flex, xi_asp - ε*v_asp,
                                    xi_spectrum .- ε .* v_spectrum, cfg, axis;
                                    xi_offset=xi_offset - ε*v_offset)
        d_amp_fd = (amp_p .- amp_m) ./ (2ε)
        @test norm(d_amp_jvp .- d_amp_fd) / (norm(d_amp_fd) + eps(T)) < fd_reltol(T)
    end

    @testset "Adjoint identity ⟨J·v, w⟩ ≈ ⟨v, J'·w⟩" begin
        rng = MersenneTwister(2)
        xi_slope, xi_fluct, xi_flex, xi_asp, xi_offset = randn(rng, T, 5) .* T(0.3)
        xi_spectrum = T(0.1) .* randn(rng, T, axis.n_bins - 2, 2)
        v_slope, v_fluct, v_flex, v_asp, v_offset = randn(rng, T, 5)
        v_spectrum = randn(rng, T, axis.n_bins - 2, 2)
        g_amp = randn(rng, T, axis.n_bins)

        d_amp = amplitude_spectrum_jvp(v_slope, v_fluct, v_flex, v_asp, v_spectrum,
                                        xi_slope, xi_fluct, xi_flex, xi_asp,
                                        xi_spectrum, cfg, axis;
                                        xi_offset=xi_offset, v_offset=v_offset)
        g_slope, g_fluct, g_flex, g_asp, g_spectrum, g_offset =
            amplitude_spectrum_adjoint(g_amp, xi_slope, xi_fluct, xi_flex, xi_asp,
                                        xi_spectrum, cfg, axis;
                                        xi_offset=xi_offset)
        lhs = dot(g_amp, d_amp)
        rhs = g_slope * v_slope + g_fluct * v_fluct +
              g_flex * v_flex   + g_asp * v_asp +
              dot(g_spectrum, v_spectrum) + g_offset * v_offset
        @test abs(lhs - rhs) / max(abs(lhs), one(T)) < adjoint_tol(T)
    end
end
