# Phase-2 tests: CorrelatedField composing a 2-D spatial radial axis with a
# 1-D wavelength axis. Validates the outer-product forward, the product-rule
# JVP, and the dual VJP against finite differences + the adjoint identity.
# Run at both Float32 and Float64 — this exercises the Complex{T} FFT plans and
# scratch buffers that make the correlated-field path single-precision-capable.

using VarInf
using Test
using LinearAlgebra
using Random
using FFTW

@testset "CorrelatedField (2-D spatial × 1-D spectral) ($T)" for T in (Float32, Float64)
    n_sp = 8
    n_sw = 6
    spatial  = FourierGridInfo(n_sp, inv(T(n_sp)))
    spectral = Axis1DInfo(n_sw, inv(T(n_sw)))

    sp_cfg = CorrFieldConfig(slope_prior=(T(-2), T(0.5)),
                              fluct_prior=(T(0.4), T(0.04)),
                              flex_prior=(one(T), T(0.5)),
                              asp_prior=(T(0.6), T(0.06)))
    sc_cfg = CorrFieldConfig(slope_prior=(T(-1.5), T(0.3)),
                              fluct_prior=(T(0.3), T(0.03)))   # no IWP on spectral axis

    mcf = CorrelatedField([spatial, spectral], [sp_cfg, sc_cfg];
                           offset_prior=(zero(T), T(0.1)))

    n_z = latent_size(mcf)

    @testset "Layout + forward sanity" begin
        @test eltype(mcf) == T
        @test eltype(mcf.cs1) == Complex{T}       # FFT scratch buffers are single/double precision
        @test mcf.field_shape == (n_sp, n_sp, n_sw)
        n_field = n_sp * n_sp * n_sw
        m_sp = 4 + 2 * (spatial.n_bins  - 2)
        m_sc = 2
        @test n_z == n_field + m_sp + m_sc + 1

        z = T(0.05) .* randn(MersenneTwister(0), T, n_z)
        field = mcf(z)
        @test eltype(field) == T
        @test size(field) == mcf.field_shape
        @test all(isfinite, field)
    end

    @testset "Adjoint identity ⟨J·v, w⟩ ≈ ⟨v, J'·w⟩" begin
        rng = MersenneTwister(1)
        z = T(0.05) .* randn(rng, T, n_z)
        v = randn(rng, T, n_z)
        w = randn(rng, T, mcf.field_shape...)
        Jv = mcf_jvp(mcf, z, v)
        Jtw = mcf_vjp(mcf, z, w)
        @test abs(dot(Jv, w) - dot(v, Jtw)) / max(abs(dot(Jv, w)), one(T)) < adjoint_tol(T)
    end

    @testset "JVP vs finite differences (one entry from each latent block)" begin
        rng = MersenneTwister(2)
        z0 = T(0.05) .* randn(rng, T, n_z)
        ε = fd_step(T)

        N_total = prod(mcf.field_shape)
        m_sp = 4 + 2 * (spatial.n_bins - 2)
        sp_off = N_total
        sc_off = N_total + m_sp
        indices = [rand(rng, 1:N_total),
                   sp_off + 1, sp_off + 2, sp_off + 3, sp_off + 4, sp_off + 5,
                   sc_off + 1, sc_off + 2, n_z]

        for i in indices
            v = zeros(T, n_z); v[i] = one(T)
            Jv = mcf_jvp(mcf, z0, v)
            zp = copy(z0); zp[i] += ε
            zm = copy(z0); zm[i] -= ε
            fd = (mcf(zp) .- mcf(zm)) ./ (2ε)
            @test norm(Jv .- fd) / (norm(fd) + eps(T)) < fd_reltol(T)
        end
    end

    @testset "use_offset = false leaves DC at legacy convention" begin
        mcf2 = CorrelatedField([spatial, spectral], [sp_cfg, sc_cfg])
        @test !mcf2.use_offset
        z2 = T(0.05) .* randn(MersenneTwister(3), T, latent_size(mcf2))
        field2 = mcf2(z2)
        @test eltype(field2) == T
        @test all(isfinite, field2)
    end

    @testset "Single-axis 1-D CorrelatedField" begin
        mcf1 = CorrelatedField([spectral], [sc_cfg]; offset_prior=(zero(T), T(0.1)))
        @test mcf1.field_shape == (n_sw,)
        rng = MersenneTwister(4)
        z1 = T(0.05) .* randn(rng, T, latent_size(mcf1))
        field1 = mcf1(z1)
        @test size(field1) == (n_sw,)
        @test all(isfinite, field1)

        v1 = randn(rng, T, latent_size(mcf1))
        w1 = randn(rng, T, n_sw)
        Jv = mcf_jvp(mcf1, z1, v1)
        Jtw = mcf_vjp(mcf1, z1, w1)
        @test abs(dot(Jv, w1) - dot(v1, Jtw)) / max(abs(dot(Jv, w1)), one(T)) < adjoint_tol(T)
    end

    @testset "Three-axis composite (catches N-axis loop off-by-ones)" begin
        temporal = Axis1DInfo(4, inv(T(4)))
        tm_cfg = CorrFieldConfig(slope_prior=(T(-1), T(0.3)), fluct_prior=(T(0.4), T(0.04)))
        mcf3 = CorrelatedField([spatial, spectral, temporal],
                                [sp_cfg, sc_cfg, tm_cfg]; offset_prior=(zero(T), T(0.1)))
        @test mcf3.field_shape == (n_sp, n_sp, n_sw, 4)

        rng = MersenneTwister(10)
        n3 = latent_size(mcf3)
        z = T(0.05) .* randn(rng, T, n3)
        field = mcf3(z)
        @test size(field) == mcf3.field_shape
        @test all(isfinite, field)

        v = randn(rng, T, n3)
        w = randn(rng, T, mcf3.field_shape...)
        Jv = mcf_jvp(mcf3, z, v)
        Jtw = mcf_vjp(mcf3, z, w)
        @test abs(dot(Jv, w) - dot(v, Jtw)) / max(abs(dot(Jv, w)), one(T)) < adjoint_tol(T)
    end

    @testset "Single-axis 2-D matches hand-rolled forward (self-consistency)" begin
        only_cfg = CorrFieldConfig(slope_prior=(T(-2), T(0.5)),
                                    fluct_prior=(T(0.4), T(0.04)),
                                    flex_prior=(one(T), T(0.5)),
                                    asp_prior=(T(0.6), T(0.06)))
        mcf_a = CorrelatedField([spatial], [only_cfg]; offset_prior=(T(0.1), T(0.1)))
        @test mcf_a.field_shape == (n_sp, n_sp)

        z = T(0.05) .* randn(MersenneTwister(20), T, latent_size(mcf_a))
        field_mcf = mcf_a(z)

        xi_field, per_axis, xi_offset = latent_unpack(mcf_a, z)
        amp = amplitude_spectrum(per_axis[1].xi_slope, per_axis[1].xi_fluct,
                                  per_axis[1].xi_flex, per_axis[1].xi_asp,
                                  per_axis[1].xi_spectrum, only_cfg, spatial)
        amp_kernel = amp[spatial.bin_index]
        amp_kernel[1, 1] = T(0.1) * sqrt(T(n_sp^2))     # azm·√P (azm = 0.1, pinned)
        field_hand = real.(ifft(fft(reshape(xi_field, n_sp, n_sp)) .* amp_kernel)) .+ T(0.1)

        @test maximum(abs.(field_mcf .- field_hand)) < sqrt(eps(T)) * max(one(T), maximum(abs, field_hand))
    end

    @testset "IWP infeasibility check (axis with n_bins < 3)" begin
        tiny_axis = Axis1DInfo(2, T(0.5))
        @test length(tiny_axis.log_volume) == 0
        bad_cfg = CorrFieldConfig(slope_prior=(T(-2), T(0.5)),
                                   fluct_prior=(T(0.4), T(0.04)),
                                   flex_prior=(one(T), T(0.5)),
                                   asp_prior=(T(0.6), T(0.06)))
        @test_throws ErrorException CorrelatedField([tiny_axis], [bad_cfg];
                                                     offset_prior=(zero(T), T(0.1)))
        plain_cfg = CorrFieldConfig(slope_prior=(T(-2), T(0.5)), fluct_prior=(T(0.4), T(0.04)))
        mcf_tiny = CorrelatedField([tiny_axis], [plain_cfg]; offset_prior=(zero(T), T(0.1)))
        @test mcf_tiny.field_shape == (2,)
    end

    @testset "Both axes without IWP" begin
        plain_sp_cfg = CorrFieldConfig(slope_prior=(T(-2), T(0.5)), fluct_prior=(T(0.4), T(0.04)))
        plain_sc_cfg = CorrFieldConfig(slope_prior=(T(-1.5), T(0.3)), fluct_prior=(T(0.3), T(0.03)))
        mcf_p = CorrelatedField([spatial, spectral],
                                 [plain_sp_cfg, plain_sc_cfg]; offset_prior=(zero(T), T(0.1)))
        @test latent_size(mcf_p) == n_sp * n_sp * n_sw + 2 + 2 + 1

        rng = MersenneTwister(30)
        z = T(0.05) .* randn(rng, T, latent_size(mcf_p))
        field = mcf_p(z)
        @test size(field) == mcf_p.field_shape
        @test all(isfinite, field)

        v = randn(rng, T, latent_size(mcf_p))
        w = randn(rng, T, mcf_p.field_shape...)
        Jv = mcf_jvp(mcf_p, z, v)
        Jtw = mcf_vjp(mcf_p, z, w)
        @test abs(dot(Jv, w) - dot(v, Jtw)) / max(abs(dot(Jv, w)), one(T)) < adjoint_tol(T)
    end
end
