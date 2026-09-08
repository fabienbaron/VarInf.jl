# Matérn-kernel amplitude: port of NIFTy.re `matern_amplitude`.
#
# The reference values here are recomputed independently from NIFTy's formula
# (nifty/src/re/correlated_field.py, the `correlate` closure of
# `matern_amplitude`) rather than from VarInf's own helpers, so the test cannot
# agree with the implementation merely by sharing its code.

using VarInf
using Test
using Random
using LinearAlgebra

# NIFTy's spectrum, written out directly from the Python, with norm = 1
# (renormalize_amplitude is not ported). Takes PHYSICAL hyperparameters.
function nifty_matern(k, totvol, scale, cutoff, slope; kind=:amplitude)
    ln_spectrum = 0.25 .* slope .* log1p.((k ./ cutoff) .^ 2)
    spectrum = exp.(ln_spectrum)
    spectrum = scale .* sqrt(totvol) .* spectrum     # norm == 1
    spectrum[1] = totvol                             # spectrum.at[0].set(totvol)
    kind === :power && (spectrum = sqrt.(spectrum))
    return spectrum
end

# The log-space (μ, σ) NIFTy's lognormal_prior uses, recomputed here.
ln_params(m, s) = (log(m) - log1p((s / m)^2) / 2, sqrt(log1p((s / m)^2)))

@testset "Matérn amplitude" begin
    @testset "matches NIFTy's formula ($T, $kind)" for T in (Float32, Float64),
                                                       kind in (:amplitude, :power)
        p = FourierGridInfo(16, one(T))
        cfg = MaternConfig(; scale_prior=(T(2), T(0.5)),
                             cutoff_prior=(T(3), T(1)),
                             loglogslope_prior=(T(-4), T(1)), kind=kind)
        for (xs, xc, xl) in ((0, 0, 0), (0.7, -0.3, 1.1), (-1.2, 0.9, -0.6))
            xs, xc, xl = T(xs), T(xc), T(xl)
            A = matern_amplitude(xs, xc, xl, cfg, p)
            @test eltype(A) == T

            # Physical hyperparameters, from the priors, recomputed here.
            sm, ss = ln_params(2.0, 0.5)
            cm, cs = ln_params(3.0, 1.0)
            scale  = exp(sm + ss * Float64(xs))
            cutoff = exp(cm + cs * Float64(xc))
            slope  = -4.0 + 1.0 * Float64(xl)
            ref = nifty_matern(Float64.(p.mode_lengths), Float64(p.total_volume),
                               scale, cutoff, slope; kind=kind)
            @test maximum(abs.(Float64.(A) .- ref) ./ max.(abs.(ref), 1e-30)) <
                  (T === Float32 ? 1e-4 : 1e-12)
        end
    end

    @testset "zero mode is exactly total_volume ($kind)" for kind in (:amplitude, :power)
        p = FourierGridInfo(8, 1.0)
        cfg = MaternConfig(; scale_prior=(2.0, 0.5), cutoff_prior=(3.0, 1.0),
                             loglogslope_prior=(-4.0, 1.0), kind=kind)
        A = matern_amplitude(0.4, -0.2, 0.3, cfg, p)
        want = kind === :amplitude ? p.total_volume : sqrt(p.total_volume)
        @test A[1] ≈ want
        # ...and carries no latent dependence.
        @test matern_amplitude_jvp(1.0, 1.0, 1.0, 0.4, -0.2, 0.3, cfg, p)[1] == 0.0
    end

    @testset "priors follow the CODE, not NIFTy's docstring" begin
        # scale and cutoff lognormal (median at ξ=0), loglogslope normal.
        p = FourierGridInfo(8, 1.0)
        cfg = MaternConfig(; scale_prior=(2.0, 0.5), cutoff_prior=(3.0, 1.0),
                             loglogslope_prior=(-4.0, 1.0))
        sm, _ = ln_params(2.0, 0.5)
        A0 = matern_amplitude(0.0, 0.0, 0.0, cfg, p)
        # At ξ=0 the slope is exactly its normal mean, so A/(scale√V) = exp(0.25·(-4)·u).
        cm, _ = ln_params(3.0, 1.0)
        u = log1p.((p.mode_lengths ./ exp(cm)) .^ 2)
        pred = exp(sm) * sqrt(p.total_volume) .* exp.(0.25 .* (-4.0) .* u)
        @test A0[2:end] ≈ pred[2:end]
        # A normal slope prior must admit negative slopes; a lognormal could not.
        @test cfg.slope_mean == -4.0
        # A lognormal cutoff is positive for any latent, however large.
        @test exp(cfg.cutoff_mean + cfg.cutoff_std * (-50.0)) > 0
    end

    @testset "JVP against central differences ($T)" for T in (Float32, Float64)
        p = FourierGridInfo(16, one(T))
        cfg = MaternConfig(; scale_prior=(T(2), T(0.5)), cutoff_prior=(T(3), T(1)),
                             loglogslope_prior=(T(-4), T(1)))
        x = (T(0.3), T(-0.4), T(0.6))
        h = cbrt(eps(T))
        # One basis direction at a time, so each partial is checked separately.
        for (j, v) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
            vT = T.(v)
            jvp = matern_amplitude_jvp(vT..., x..., cfg, p)
            xp = collect(x); xp[j] += h
            xm = collect(x); xm[j] -= h
            fd = (matern_amplitude(xp..., cfg, p) .-
                  matern_amplitude(xm..., cfg, p)) ./ (2h)
            @test norm(jvp[2:end] .- fd[2:end]) / max(norm(fd[2:end]), one(T)) <
                  20 * cbrt(eps(T))
        end
    end

    @testset "adjoint identity ($T, $kind)" for T in (Float32, Float64),
                                                kind in (:amplitude, :power)
        p = FourierGridInfo(16, one(T))
        cfg = MaternConfig(; scale_prior=(T(2), T(0.5)), cutoff_prior=(T(3), T(1)),
                             loglogslope_prior=(T(-4), T(1)), kind=kind)
        rng = MersenneTwister(5)
        x = (T(0.3), T(-0.4), T(0.6))
        for _ in 1:5
            v = (randn(rng, T), randn(rng, T), randn(rng, T))
            g = randn(rng, T, p.n_bins)
            jvp = matern_amplitude_jvp(v..., x..., cfg, p)
            adj = matern_amplitude_adjoint(g, x..., cfg, p)
            lhs = dot(jvp, g)
            rhs = sum(v[i] * adj[i] for i in 1:3)
            @test abs(lhs - rhs) / max(abs(lhs), one(T)) < sqrt(eps(T))
        end
    end

    @testset ":power is the square root of :amplitude" begin
        p = FourierGridInfo(16, 1.0)
        kw = (; scale_prior=(2.0, 0.5), cutoff_prior=(3.0, 1.0),
                loglogslope_prior=(-4.0, 1.0))
        a = matern_amplitude(0.2, 0.1, -0.3, MaternConfig(; kw..., kind=:amplitude), p)
        q = matern_amplitude(0.2, 0.1, -0.3, MaternConfig(; kw..., kind=:power), p)
        @test q ≈ sqrt.(a)
    end

    @testset "shape of the spectrum" begin
        p = FourierGridInfo(32, 1.0)
        cfg = MaternConfig(; scale_prior=(1.0, 0.1), cutoff_prior=(2.0, 0.1),
                             loglogslope_prior=(-4.0, 0.1))
        A = matern_amplitude(0.0, 0.0, 0.0, cfg, p)
        # A negative loglogslope must give an amplitude falling with k.
        @test issorted(A[2:end]; rev=true) || all(diff(A[2:end]) .<= 1e-12)
        # Raising the cutoff pushes the knee out, so high-k amplitude rises.
        A_hi = matern_amplitude(0.0, 3.0, 0.0, cfg, p)
        @test A_hi[end] > A[end]
    end

    @testset "overflow safety and argument checking" begin
        # k/cutoff enormous: log1p(r²) would overflow r² in Float32.
        p = FourierGridInfo(16, 1.0f0)
        cfg = MaternConfig(; scale_prior=(1.0f0, 0.1f0), cutoff_prior=(1.0f-30, 1.0f-31),
                             loglogslope_prior=(-4.0f0, 0.1f0))
        A = matern_amplitude(0.0f0, 0.0f0, 0.0f0, cfg, p)
        @test all(isfinite, A)
        j = matern_amplitude_jvp(1.0f0, 1.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0, cfg, p)
        @test all(isfinite, j)

        @test_throws ArgumentError MaternConfig(; scale_prior=(1.0, 0.1),
            cutoff_prior=(1.0, 0.1), loglogslope_prior=(-4.0, 0.1), kind=:bogus)
        # lognormal priors need a positive mean; the normal slope prior does not.
        @test_throws ArgumentError MaternConfig(; scale_prior=(-1.0, 0.1),
            cutoff_prior=(1.0, 0.1), loglogslope_prior=(-4.0, 0.1))
        @test MaternConfig(; scale_prior=(1.0, 0.1), cutoff_prior=(1.0, 0.1),
                             loglogslope_prior=(-99.0, 0.1)) isa MaternConfig
    end
end
