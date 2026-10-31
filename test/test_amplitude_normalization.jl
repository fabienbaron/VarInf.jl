# Regression tests for the `fluctuations` normalisation of `amplitude_spectrum`.
#
# The guarantee under test: a field synthesised as `real(ifft(fft(xi) .* amp))`
# from white `xi` has per-pixel standard deviation `fluct`, on ANY grid,
# independent of the grid's size and dimension.
#
# Equivalently (Parseval, with Julia's `ifft` carrying 1/N):
#   var(field) = (1/N) * sum_modes amp^2,   N = sum(mode_multiplicity)
# so the non-DC bins must satisfy
#   sum_{b>=2} mult[b]*amp[b]^2 == fluct^2 * N.
#
# HISTORY: before 2026-10-31 the prefactor was `p.npix` rather than `sqrt(N)`.
# Those agree for a square 2-D `FourierGridInfo` (npix^2 == N) but not for an
# `Axis1DInfo` (npix != sqrt(npix)), so every 1-D axis came out too wide by
# `sqrt(npix)` -- a factor 6.9 at npix = 48 -- and every multi-axis field by
# `sqrt(npix)` for each 1-D axis it carried. Reported from OITOOLS.jl, whose
# learned limb-darkening prior is a 1-D field on 48 pixels.
#
# The pre-existing 1-D/multi-axis tests all passed throughout, because they
# check self-consistency (JVP vs FD, adjoint vs JVP) and structure, never
# whether `fluct` means what it says. That is the gap these tests close.

# Exact non-DC spectral power: sum_{b>=2} mult[b]*amp[b]^2.
_nondc_power(amp, mult) = sum(mult[b] * amp[b]^2 for b in 2:length(amp))

# The fluctuation VALUE at latent `xi`. `fluct_prior = (mean, std)` is a
# lognormal prior, so the value at xi = 0 is `mean*exp(-sigma^2/2)`, not `mean`
# -- a 1e-6 offset at std = 1e-3 that would otherwise swamp an exact identity.
_fluct_at(cfg, xi) = exp(cfg.fluct_mean + cfg.fluct_std * xi)

# Per-pixel std of `nsamp` CorrelatedField draws, field mean removed from each
# (the DC mode is not part of the normalisation being tested).
function _draw_pixel_std(cf, nsamp::Int, rng)
    acc = Float64[]
    nz = latent_size(cf)
    for _ in 1:nsamp
        fld = vec(cf(randn(rng, eltype(cf), nz)))
        m = sum(fld) / length(fld)
        append!(acc, Float64.(fld) .- m)
    end
    return sqrt(sum(abs2, acc) / (length(acc) - 1))
end

@testset "amplitude_spectrum fluctuations normalisation ($T)" for T in (Float32, Float64)

    # ── The exact identity, 1-D: the case that was wrong ────────────────────
    @testset "Axis1DInfo: non-DC power == fluct^2 * N" begin
        for npix in (16, 48, 128)
            axis = Axis1DInfo(npix, inv(T(npix ÷ 2 - 1)))
            N = sum(axis.mode_multiplicity)
            @test N ≈ T(npix)                      # N is the mode count, not npix^2
            for f in (T(0.1), T(0.3), T(1.0)), slope in (T(-2), T(-4))
                cfg = CorrFieldConfig(; slope_prior=(slope, T(1e-3)),
                                        fluct_prior=(f, T(1e-3)))
                amp = amplitude_spectrum(cfg, axis)
                fv = _fluct_at(cfg, zero(T))
                @test _nondc_power(amp, axis.mode_multiplicity) ≈ fv^2 * N rtol=sqrt(eps(T))
                # i.e. the implied per-pixel std is `fluct`, NOT the old
                # `fluct*sqrt(npix)`; check against `f` on a loose bar too, so
                # the intent survives even if the prior convention changes.
                @test sqrt(_nondc_power(amp, axis.mode_multiplicity) / N) ≈ fv rtol=sqrt(eps(T))
                @test sqrt(_nondc_power(amp, axis.mode_multiplicity) / N) ≈ f rtol=1e-3
            end
        end
    end

    # ── The same identity in 2-D: must be byte-identical to the old code ────
    @testset "FourierGridInfo: unchanged by the fix" begin
        for npix in (8, 16, 32)
            g = FourierGridInfo(npix, inv(T(npix)))
            # The fix is a no-op here precisely because sqrt(N) == npix.
            @test sum(g.mode_multiplicity) ≈ T(npix)^2
            @test sqrt(sum(g.mode_multiplicity)) ≈ T(npix)
            N = sum(g.mode_multiplicity)
            for f in (T(0.1), T(0.3), T(1.0))
                cfg = CorrFieldConfig(; slope_prior=(T(-4), T(1e-3)),
                                        fluct_prior=(f, T(1e-3)))
                amp = amplitude_spectrum(cfg, g)
                fv = _fluct_at(cfg, zero(T))
                @test _nondc_power(amp, g.mode_multiplicity) ≈ fv^2 * N rtol=sqrt(eps(T))
            end
        end
    end

    # ── Holds with the IWP layer on, and at nonzero hyper-latents ───────────
    @testset "normalisation is exact with IWP and off-mean latents" begin
        rng = MersenneTwister(4242)
        for (nm, grid) in (("axis", Axis1DInfo(64, inv(T(31)))),
                           ("fourier", FourierGridInfo(16, inv(T(16)))))
            cfg = CorrFieldConfig(; slope_prior=(T(-3), T(0.5)),
                                    fluct_prior=(T(0.7), T(0.07)),
                                    flex_prior=(T(0.8), T(0.08)),
                                    asp_prior=(T(0.2), T(0.02)))
            N = sum(grid.mode_multiplicity)
            for _ in 1:4
                xs, xf, xx, xa = randn(rng, T), randn(rng, T), randn(rng, T), randn(rng, T)
                xspec = randn(rng, T, grid.n_bins - 2, 2)
                amp = amplitude_spectrum(xs, xf, xx, xa, xspec, cfg, grid)
                # fluct is lognormal in the latent; recover it and check the identity.
                f = _fluct_at(cfg, xf)
                @test _nondc_power(amp, grid.mode_multiplicity) ≈ f^2 * N rtol=sqrt(eps(T))
                @test nm isa String
            end
        end
    end

    # ── JVP and adjoint carry the same scale ────────────────────────────────
    # The prefactor is linear, so a scale error in one of the three sites would
    # show up here as a JVP/FD or adjoint/JVP mismatch on a 1-D axis.
    @testset "jvp and adjoint agree with the rescaled forward" begin
        rng = MersenneTwister(99)
        axis = Axis1DInfo(32, inv(T(15)))
        cfg = CorrFieldConfig(; slope_prior=(T(-2.5), T(0.4)),
                                fluct_prior=(T(0.5), T(0.05)))
        xs, xf = randn(rng, T), randn(rng, T)
        z = zeros(T, 0, 0)
        h = fd_step(T)
        for (vs, vf) in ((one(T), zero(T)), (zero(T), one(T)))
            d_an = amplitude_spectrum_jvp(vs, vf, zero(T), zero(T), z,
                                          xs, xf, zero(T), zero(T), z, cfg, axis)
            ap = amplitude_spectrum(xs + h*vs, xf + h*vf, zero(T), zero(T), z, cfg, axis)
            am = amplitude_spectrum(xs - h*vs, xf - h*vf, zero(T), zero(T), z, cfg, axis)
            d_fd = (ap .- am) ./ (2h)
            @test norm(d_an .- d_fd) / (norm(d_fd) + eps(T)) < fd_reltol(T)
        end
        # <g, J v> == <Jᵀ g, v>
        g = randn(rng, T, axis.n_bins)
        vs, vf = randn(rng, T), randn(rng, T)
        Jv = amplitude_spectrum_jvp(vs, vf, zero(T), zero(T), z,
                                    xs, xf, zero(T), zero(T), z, cfg, axis)
        gs, gf, = amplitude_spectrum_adjoint(g, xs, xf, zero(T), zero(T), z, cfg, axis)
        @test dot(g, Jv) ≈ gs*vs + gf*vf rtol=adjoint_tol(T)
    end
end

# ── End-to-end: the property a user actually observes ──────────────────────
# Monte Carlo, Float64 only (a Float32 draw's sample std is not tight enough to
# separate 1.0 from the old 6.9 any better than this already does).
@testset "CorrelatedField draws have std == fluct" begin
    rng = MersenneTwister(20261031)

    @testset "1-D axis: std is fluct, independent of npix" begin
        for npix in (16, 48, 128)
            axis = Axis1DInfo(npix, inv(1.0 * (npix ÷ 2 - 1)))
            for f in (0.1, 0.3, 1.0)
                cfg = CorrFieldConfig(; slope_prior=(-4.0, 1e-3), fluct_prior=(f, 1e-3))
                cf = CorrelatedField([axis], [cfg])
                s = _draw_pixel_std(cf, 400, rng)
                # Tight enough to fail the old code by a factor 4 -- 11.3.
                @test s ≈ f rtol=0.12
            end
        end
    end

    @testset "2-D grid: std is fluct (unchanged behaviour)" begin
        for npix in (16, 32)
            g = FourierGridInfo(npix, inv(1.0 * npix))
            for f in (0.1, 0.3)
                cfg = CorrFieldConfig(; slope_prior=(-4.0, 1e-3), fluct_prior=(f, 1e-3))
                cf = CorrelatedField([g], [cfg])
                @test _draw_pixel_std(cf, 200, rng) ≈ f rtol=0.12
            end
        end
    end

    # Multi-axis composition. The per-axis DC bin is replaced by `√N_i`
    # (`_normalize_axis_dc`), which is NIFTy's `total_volume_i` expressed in
    # VarInf's mode-count convention, so a marginal-DC mode (k_i = 0, k_j ≠ 0)
    # carries axis j's FULL fluctuation power. Summing the four mode classes of
    # a 2-axis field and dropping the composite DC (removed with the mean):
    #
    #   var = fluct_1^2*fluct_2^2 + fluct_1^2 + fluct_2^2 = (1+f_1^2)(1+f_2^2) - 1
    #
    # and in general `var = prod_i (1 + fluct_i^2) - 1`, which collapses to
    # `fluct^2` for a single axis -- one law for both cases.
    #
    # NOTE this is NOT the product of the per-axis flucts: for small flucts it
    # is dominated by the marginal terms, ~sqrt(f_1^2 + f_2^2). Verified against
    # NIFTy (two 1-D axes, azm pinned to 1): f = (0.5,0.4) -> 0.659 / 0.674 and
    # f = (1.0,0.3) -> 1.062 / 1.128, vs this formula's 0.6708 / 1.0863.
    # Before 2026-10-31 the marginal terms were suppressed by 1/N_other, giving
    # 0.285 / 0.257 -- low by 2.3-2.7x.
    @testset "multi-axis: var == prod(1 + fluct_i^2) - 1" begin
        composite_std(fs) = sqrt(prod(1 .+ fs .^ 2) - 1)

        # Matches the single-axis guarantee for one axis.
        @test composite_std([0.3]) ≈ 0.3
        # Matches NIFTy's measured two-axis values.
        @test composite_std([0.5, 0.4]) ≈ 0.6708 atol=1e-4
        @test composite_std([1.0, 0.3]) ≈ 1.0863 atol=1e-4
        # Marginal-dominated, NOT the product.
        @test composite_std([0.5, 0.4]) > 3 * (0.5 * 0.4)

        g = FourierGridInfo(16, 1/16)
        for n_ax in (8, 24, 48), (f_sp, f_ax) in ((0.5, 0.4), (1.0, 0.3))
            axis = Axis1DInfo(n_ax, inv(1.0 * (n_ax ÷ 2 - 1)))
            sp_cfg = CorrFieldConfig(; slope_prior=(-4.0, 1e-3), fluct_prior=(f_sp, 1e-3))
            ax_cfg = CorrFieldConfig(; slope_prior=(-3.0, 1e-3), fluct_prior=(f_ax, 1e-3))
            expected = composite_std([_fluct_at(sp_cfg, 0.0), _fluct_at(ax_cfg, 0.0)])

            cf = CorrelatedField([g, axis], [sp_cfg, ax_cfg])
            @test _draw_pixel_std(cf, 60, rng) ≈ expected rtol=0.15
            # Independent of the axis length -- that dependence WAS the bug.
            @test expected ≈ composite_std([f_sp, f_ax]) rtol=1e-3
        end

        # Three axes compose the same way.
        g2 = FourierGridInfo(8, 1/8)
        ax1 = Axis1DInfo(6, 1/2)
        ax2 = Axis1DInfo(4, 1/1)
        fs = (0.4, 0.3, 0.2)
        cfgs = [CorrFieldConfig(; slope_prior=(-3.0, 1e-3), fluct_prior=(f, 1e-3))
                for f in fs]
        cf3 = CorrelatedField([g2, ax1, ax2], cfgs)
        exp3 = composite_std([_fluct_at(c, 0.0) for c in cfgs])
        @test _draw_pixel_std(cf3, 120, rng) ≈ exp3 rtol=0.15
    end

    # `offset_prior = nothing` keeps its documented contract: the COMPOSITE DC
    # stays pinned at 1 (field variance 1/P), even though every per-axis DC is
    # now √N_i. So the single-axis case above is unchanged by that fix, and
    # callers projecting the mean out are unaffected.
    @testset "composite DC stays pinned at 1 without offset_prior" begin
        g = FourierGridInfo(16, 1/16)
        axis = Axis1DInfo(12, 1/5)
        cfgs = [CorrFieldConfig(; slope_prior=(-3.0, 1e-3), fluct_prior=(0.4, 1e-3)),
                CorrFieldConfig(; slope_prior=(-2.0, 1e-3), fluct_prior=(0.3, 1e-3))]
        cf = CorrelatedField([g, axis], cfgs)
        amps = [amplitude_spectrum(cfgs[i], [g, axis][i]) for i in 1:2]
        K = VarInf._build_amp_kernel(amps, cf, 0.0)
        @test K[1, 1, 1] == 1
        # ... while the per-axis DC values are √N_i, so the marginal-DC slices
        # are NOT suppressed: K at (k_spatial=0, k_axis=bin b) is √N_spatial·a_ax[b].
        a = VarInf._normalize_axis_dc(amps, cf)
        @test a[1][1] ≈ sqrt(sum(g.mode_multiplicity))
        @test a[2][1] ≈ sqrt(sum(axis.mode_multiplicity))
    end
end
