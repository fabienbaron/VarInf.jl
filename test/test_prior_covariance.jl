# Optional non-unit prior covariance (ROTIR request B).
#
# Checks that the three hooks (prior_inv_covariance_mul,
# prior_inv_sqrt_covariance_mul, prior_energy) are wired consistently into the
# metric, the metric-sample draw, the GeoVI coordinate map and the minisanity
# table — and, critically, that the default C = I path is unchanged.

using VarInf
using Test
using Random
using LinearAlgebra

# Linear-Gaussian problem T(z) = F·z with a diagonal prior covariance.
# `cinv === nothing` leaves every hook at its default (C = I).
struct CenteredProblem <: AbstractInferenceProblem
    F::Matrix{Float64}
    d::Vector{Float64}
    cinv::Union{Nothing,Vector{Float64}}
end

VarInf.latent_size(p::CenteredProblem) = size(p.F, 2)
VarInf.data_size(p::CenteredProblem)   = size(p.F, 1)
VarInf.transformation(p::CenteredProblem, z::AbstractVector{<:Real}) = p.F * z
VarInf.right_sqrt_metric(p::CenteredProblem, _, v::AbstractVector{<:Real}) = p.F * v
VarInf.left_sqrt_metric(p::CenteredProblem, _, w::AbstractVector{<:Real})  = p.F' * w
VarInf.whitened_data(p::CenteredProblem) = p.d
VarInf.latent_blocks(p::CenteredProblem) =
    [("blk_a", 1:2), ("blk_b", 3:size(p.F, 2))]

# The hooks are only defined when `cinv` is present, so a `nothing` problem
# exercises the package defaults rather than an identity re-implementation.
VarInf.prior_inv_covariance_mul(p::CenteredProblem, ::AbstractVector{<:Real},
                                v::AbstractVector{<:Real}) =
    p.cinv === nothing ? v : p.cinv .* v
VarInf.prior_inv_sqrt_covariance_mul(p::CenteredProblem, ::AbstractVector{<:Real},
                                     v::AbstractVector{<:Real}) =
    p.cinv === nothing ? v : sqrt.(p.cinv) .* v
VarInf.prior_energy(p::CenteredProblem, z::AbstractVector{<:Real}) =
    p.cinv === nothing ? 0.5 * sum(abs2, z) : 0.5 * dot(z, p.cinv .* z)

function VarInf.energy_and_gradient(p::CenteredProblem, z::AbstractVector{<:Real})
    r = p.F * z .- p.d
    cz = p.cinv === nothing ? collect(float(z)) : p.cinv .* z
    return 0.5 * sum(abs2, r) + 0.5 * dot(z, cz), p.F' * r .+ cz
end

# Mixed parameterization: latent 1 is a non-centred hyper-latent (C = I), and
# latents 2:5 are a centred field block with C(z) = amp(z)²·I, amp = exp(z[1]).
struct MixedProblem <: AbstractInferenceProblem
    F::Matrix{Float64}
    d::Vector{Float64}
end

_amp2(z) = exp(2 * z[1])
_field_rng = 2:5

VarInf.latent_size(p::MixedProblem) = size(p.F, 2)
VarInf.data_size(p::MixedProblem)   = size(p.F, 1)
VarInf.transformation(p::MixedProblem, z::AbstractVector{<:Real}) = p.F * z
VarInf.right_sqrt_metric(p::MixedProblem, _, v::AbstractVector{<:Real}) = p.F * v
VarInf.left_sqrt_metric(p::MixedProblem, _, w::AbstractVector{<:Real})  = p.F' * w
VarInf.whitened_data(p::MixedProblem) = p.d

function VarInf.prior_inv_covariance_mul(::MixedProblem, z::AbstractVector{<:Real},
                                         v::AbstractVector{<:Real})
    out = collect(float(v))
    out[_field_rng] ./= _amp2(z)
    return out
end
function VarInf.prior_inv_sqrt_covariance_mul(::MixedProblem, z::AbstractVector{<:Real},
                                              v::AbstractVector{<:Real})
    out = collect(float(v))
    out[_field_rng] ./= sqrt(_amp2(z))
    return out
end
# 0.5‖θ‖² + 0.5·yᵀC(θ)⁻¹y + 0.5·logdet C(θ), log-det explicit.
VarInf.prior_energy(p::MixedProblem, z::AbstractVector{<:Real}) =
    0.5 * z[1]^2 + 0.5 * sum(abs2, view(z, _field_rng)) / _amp2(z) +
    0.5 * length(_field_rng) * log(_amp2(z))

function VarInf.energy_and_gradient(p::MixedProblem, z::AbstractVector{<:Real})
    r = p.F * z .- p.d
    g = p.F' * r .+ VarInf.prior_inv_covariance_mul(p, z, z)
    # d/dz[1] of the prior: z[1] from the hyper term, the amp-derivative of the
    # quadratic, and the log-det.
    g[1] += z[1] - sum(abs2, view(z, _field_rng)) / _amp2(z) + length(_field_rng)
    g[1] -= z[1]   # prior_inv_covariance_mul already contributed z[1] at index 1
    return 0.5 * sum(abs2, r) + VarInf.prior_energy(p, z), g
end

# A DENSE prior covariance: a squared-exponential GP kernel on a 1-D ring, so
# C⁻¹ and C^{-1/2} are full matrices rather than elementwise scalings. Same
# interface, different regime for the matrix-free path.
struct DenseGPProblem <: AbstractInferenceProblem
    F::Matrix{Float64}
    d::Vector{Float64}
    Cinv::Matrix{Float64}
    Cinvsqrt::Matrix{Float64}
end

function DenseGPProblem(F, d; theta_c=1.5)
    n = size(F, 2)
    ang = [2pi * (i - 1) / n for i in 1:n]
    # great-circle-style separation on a ring
    sep = [min(abs(a - b), 2pi - abs(a - b)) for a in ang, b in ang]
    C = exp.(-sep .^ 2 ./ (2 * theta_c^2)) + 1e-6I
    Ci = inv(Symmetric(C))
    return DenseGPProblem(F, d, Matrix(Ci), Matrix(sqrt(Symmetric(Ci))))
end

VarInf.latent_size(p::DenseGPProblem) = size(p.F, 2)
VarInf.data_size(p::DenseGPProblem)   = size(p.F, 1)
VarInf.transformation(p::DenseGPProblem, z::AbstractVector{<:Real}) = p.F * z
VarInf.right_sqrt_metric(p::DenseGPProblem, _, v::AbstractVector{<:Real}) = p.F * v
VarInf.left_sqrt_metric(p::DenseGPProblem, _, w::AbstractVector{<:Real})  = p.F' * w
VarInf.whitened_data(p::DenseGPProblem) = p.d
VarInf.prior_inv_covariance_mul(p::DenseGPProblem, ::AbstractVector{<:Real},
                                v::AbstractVector{<:Real}) = p.Cinv * v
VarInf.prior_inv_sqrt_covariance_mul(p::DenseGPProblem, ::AbstractVector{<:Real},
                                     v::AbstractVector{<:Real}) = p.Cinvsqrt * v
VarInf.prior_energy(p::DenseGPProblem, z::AbstractVector{<:Real}) =
    0.5 * dot(z, p.Cinv * z)
function VarInf.energy_and_gradient(p::DenseGPProblem, z::AbstractVector{<:Real})
    r = p.F * z .- p.d
    cz = p.Cinv * z
    return 0.5 * sum(abs2, r) + 0.5 * dot(z, cz), p.F' * r .+ cz
end

_dense(op, n) = reduce(hcat, [op(Float64[i == j for j in 1:n]) for i in 1:n])

@testset "prior covariance hooks" begin
    rng = MersenneTwister(3)
    nlat, ndat = 5, 7
    F = randn(rng, ndat, nlat)
    d = randn(rng, ndat)
    cinv = [0.5, 2.0, 4.0, 0.25, 1.5]
    prob  = CenteredProblem(F, d, cinv)
    probI = CenteredProblem(F, d, nothing)
    z = 0.4 .* randn(rng, nlat)

    @testset "posterior metric is JᵀJ + C⁻¹" begin
        M = _dense(v -> VarInf._posterior_metric_mul(prob, z, v), nlat)
        @test M ≈ F' * F + Diagonal(cinv)
        @test M ≈ M'                                   # still symmetric

        MI = _dense(v -> VarInf._posterior_metric_mul(probI, z, v), nlat)
        @test MI ≈ F' * F + I                          # default path untouched
    end

    @testset "GeoVI Hessian equals AᵀA for A = C^{-1/2} + JᵀJ" begin
        center = 0.2 .* randn(rng, nlat)
        x = center .+ 0.1 .* randn(rng, nlat)
        S = Diagonal(sqrt.(cinv))
        A = S + F' * F                                  # ∂g/∂x, constant here
        H = _dense(v -> VarInf._geovi_metric_mul(prob, x, v, center), nlat)
        @test H ≈ A' * A
        @test H ≈ H'

        HI = _dense(v -> VarInf._geovi_metric_mul(probI, x, v, center), nlat)
        AI = I + F' * F
        @test HI ≈ AI' * AI                             # default path untouched
    end

    @testset "GeoVI gradient is consistent with its energy (finite difference)" begin
        center = 0.2 .* randn(rng, nlat)
        x = center .+ 0.1 .* randn(rng, nlat)
        T_at_e = VarInf.transformation(prob, center)
        ms = 0.3 .* randn(rng, nlat)

        _, grad = VarInf._geovi_residual_vg(prob, x, center, T_at_e, ms)
        h = 1e-6
        fd = similar(x)
        for i in eachindex(x)
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            ep, _ = VarInf._geovi_residual_vg(prob, xp, center, T_at_e, ms)
            em, _ = VarInf._geovi_residual_vg(prob, xm, center, T_at_e, ms)
            fd[i] = (ep - em) / (2h)
        end
        @test norm(fd .- grad) / max(norm(fd), 1.0) < 1e-5
    end

    @testset "metric sample has covariance M" begin
        # ms = Jᵀw_data + C^{-1/2}w_latent  ⇒  Cov(ms) = JᵀJ + C⁻¹ = M
        Random.seed!(99)
        N = 200_000
        acc = zeros(nlat, nlat)
        for _ in 1:N
            ms = VarInf._draw_metric_sample(prob, z)
            acc .+= ms * ms'
        end
        M = F' * F + Diagonal(cinv)
        @test norm(acc ./ N - M) / norm(M) < 0.02
    end

    @testset "minisanity latent rows report the whitened statistic" begin
        s = fill(0.3, nlat)
        samples = [s, -s]
        out = mktemp() do path, io
            redirect_stdout(() -> VarInf._report_latents(prob, z, samples), io)
            flush(io); read(path, String)
        end
        w = sqrt.(cinv) .* (z .+ s)
        wm = sqrt.(cinv) .* (z .- s)
        for (label, rng_b) in VarInf.latent_blocks(prob)
            nd = length(rng_b)
            want = 0.5 * (sum(abs2, view(w, rng_b)) + sum(abs2, view(wm, rng_b))) / nd
            line = only([l for l in split(out, '\n') if occursin(label, l)])
            got = parse(Float64, match(r"[-+]?\d+\.\d+", line).match)
            @test isapprox(got, want; atol=5e-3)
        end
    end

    @testset "mixed, z-dependent C (hyper block non-centred, field block centred)" begin
        # The usage a centred hierarchical model actually needs: one call on the
        # full vector returns v unchanged on the hyper index and amp(z)⁻²·v on
        # the field block, with amp a function of the hyper latent.
        M_ = MixedProblem(F, d)
        zz = [0.3, 0.1, -0.2, 0.4, 0.05]
        amp2 = exp(2 * zz[1])
        cinv_expected = [1.0; fill(1 / amp2, 4)]

        Mm = _dense(v -> VarInf._posterior_metric_mul(M_, zz, v), nlat)
        @test Mm ≈ F' * F + Diagonal(cinv_expected)
        @test Mm ≈ Mm'

        # z-dependence is real: a different z must give a different metric.
        z2 = copy(zz); z2[1] += 0.7
        M2 = _dense(v -> VarInf._posterior_metric_mul(M_, z2, v), nlat)
        @test !(M2 ≈ Mm)
        @test M2 ≈ F' * F + Diagonal([1.0; fill(exp(-2 * z2[1]), 4)])

        # C^{-1/2} must square to C⁻¹ on the mixed vector.
        S = _dense(v -> VarInf.prior_inv_sqrt_covariance_mul(M_, zz, v), nlat)
        Ci = _dense(v -> VarInf.prior_inv_covariance_mul(M_, zz, v), nlat)
        @test S * S ≈ Ci

        Random.seed!(23)
        z_opt, smp = reconstruct_hybrid(M_; z0=zeros(nlat), n_mgvi=1, n_geovi=1,
                                        n_samples=2, kl_maxiter=4, map_maxiter=5,
                                        geo_newton_maxiter=2, geo_cg_maxiter=10,
                                        cg_maxiter=20, verb=false)
        @test all(isfinite, z_opt)
        @test all(s -> all(isfinite, s), smp)
    end

    @testset "dense C (GP prior): C⁻¹ and C^{-1/2} are full operators" begin
        gp = DenseGPProblem(F, d)
        zz = 0.3 .* randn(rng, nlat)

        # The dense C⁻¹ must be genuinely non-diagonal, or this proves nothing.
        Ci = _dense(v -> VarInf.prior_inv_covariance_mul(gp, zz, v), nlat)
        offdiag = Ci - Diagonal(diag(Ci))
        @test norm(offdiag) / norm(Ci) > 0.1

        Mg = _dense(v -> VarInf._posterior_metric_mul(gp, zz, v), nlat)
        @test Mg ≈ F' * F + gp.Cinv
        @test Mg ≈ Mg'

        # C^{-1/2} must square to C⁻¹ and be symmetric (the metric sample and
        # the GeoVI coordinate map both rely on the symmetric square root).
        S = _dense(v -> VarInf.prior_inv_sqrt_covariance_mul(gp, zz, v), nlat)
        @test S ≈ S'
        @test S * S ≈ gp.Cinv

        # GeoVI Hessian still equals AᵀA with a dense A.
        center = 0.2 .* randn(rng, nlat)
        x = center .+ 0.1 .* randn(rng, nlat)
        A = gp.Cinvsqrt + F' * F
        H = _dense(v -> VarInf._geovi_metric_mul(gp, x, v, center), nlat)
        @test H ≈ A' * A

        # And the gradient stays consistent under a dense C.
        T_at_e = VarInf.transformation(gp, center)
        ms = 0.3 .* randn(rng, nlat)
        _, grad = VarInf._geovi_residual_vg(gp, x, center, T_at_e, ms)
        h = 1e-6
        fd = similar(x)
        for i in eachindex(x)
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            ep, _ = VarInf._geovi_residual_vg(gp, xp, center, T_at_e, ms)
            em, _ = VarInf._geovi_residual_vg(gp, xm, center, T_at_e, ms)
            fd[i] = (ep - em) / (2h)
        end
        @test norm(fd .- grad) / max(norm(fd), 1.0) < 1e-5

        Random.seed!(31)
        z_opt, smp = reconstruct_hybrid(gp; z0=zeros(nlat), n_mgvi=1, n_geovi=1,
                                        n_samples=2, kl_maxiter=4, map_maxiter=10,
                                        geo_newton_maxiter=2, geo_cg_maxiter=10,
                                        cg_maxiter=20, verb=false)
        @test all(isfinite, z_opt)
        @test all(s -> all(isfinite, s), smp)
    end

    @testset "default path is bitwise unchanged by the new hooks" begin
        # A problem that leaves the hooks at their defaults must give exactly the
        # same answer as before they existed; compare against the explicit I.
        kw = (; z0=zeros(nlat), n_mgvi=1, n_geovi=1, n_samples=2, kl_maxiter=3,
                map_maxiter=5, geo_newton_maxiter=2, geo_cg_maxiter=5,
                cg_maxiter=10, verb=false)
        Random.seed!(5); z1, s1 = reconstruct_hybrid(probI; kw...)
        Random.seed!(5); z2, s2 = reconstruct_hybrid(probI; kw...)
        @test z1 == z2
        @test s1 == s2
        @test all(isfinite, z1)
    end

    @testset "GeoVI runs end-to-end with C ≠ I" begin
        Random.seed!(17)
        z_opt, samples = reconstruct_hybrid(prob; z0=zeros(nlat), n_mgvi=0,
                                            n_geovi=2, n_samples=2, kl_maxiter=5,
                                            geo_newton_maxiter=3, geo_cg_maxiter=10,
                                            cg_maxiter=20, verb=false)
        @test all(isfinite, z_opt)
        @test !isempty(samples)
        @test all(s -> all(isfinite, s), samples)
    end
end
