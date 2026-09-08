# minisanity / hybrid-scheduling regression tests.
#
# Covers three defects reported from the ROTIR spherical-harmonic imaging work
# (plans/varinf_requests.md items A, C, D):
#   A. the latent rows evaluated the statistic on the mean-zero offsets s, so
#      they reported tr Σ/n instead of (‖μ‖² + tr Σ)/n, and the mean column was
#      driven to 0 by antithetic construction whatever the posterior did.
#   C. reconstruct_hybrid could not run GeoVI-only (n_mgvi = 0).
#   D. n_samples was announced but ignored in :nonlinear_update iterations.

using VarInf
using Test
using Random
using LinearAlgebra

# Minimal whitened linear-Gaussian problem: T(z) = z, two declared blocks.
struct MiniSanityProblem <: AbstractInferenceProblem
    n::Int
    d::Vector{Float64}
end

VarInf.latent_size(p::MiniSanityProblem) = p.n
VarInf.data_size(p::MiniSanityProblem)   = p.n
VarInf.transformation(p::MiniSanityProblem, z::AbstractVector{<:Real}) = collect(float(z))
VarInf.right_sqrt_metric(::MiniSanityProblem, _, v::AbstractVector{<:Real}) = collect(float(v))
VarInf.left_sqrt_metric(::MiniSanityProblem, _, w::AbstractVector{<:Real})  = collect(float(w))
VarInf.whitened_data(p::MiniSanityProblem) = p.d
VarInf.energy_and_gradient(p::MiniSanityProblem, z::AbstractVector{<:Real}) =
    (0.5 * sum(abs2, z .- p.d) + 0.5 * dot(z, z), (z .- p.d) .+ z)
VarInf.latent_blocks(p::MiniSanityProblem) =
    [("blk_a", 1:(p.n ÷ 2)), ("blk_b", (p.n ÷ 2 + 1):p.n)]

# `redirect_stdout` needs a real file descriptor, so capture through a temp file.
function _capture(f)
    mktemp() do path, io
        redirect_stdout(f, io)
        flush(io)
        return read(path, String)
    end
end

# Same protocol, but with blocks declared as non-contiguous index vectors.
struct IndexBlockProblem <: AbstractInferenceProblem
    n::Int
end
VarInf.latent_size(p::IndexBlockProblem) = p.n
VarInf.data_size(p::IndexBlockProblem)   = p.n
VarInf.transformation(p::IndexBlockProblem, z::AbstractVector{<:Real}) = collect(float(z))
VarInf.right_sqrt_metric(::IndexBlockProblem, _, v::AbstractVector{<:Real}) = collect(float(v))
VarInf.left_sqrt_metric(::IndexBlockProblem, _, w::AbstractVector{<:Real})  = collect(float(w))
VarInf.whitened_data(p::IndexBlockProblem) = zeros(p.n)
VarInf.energy_and_gradient(p::IndexBlockProblem, z::AbstractVector{<:Real}) =
    (0.5 * dot(z, z), collect(float(z)))
VarInf.latent_blocks(p::IndexBlockProblem) =
    [("odd_idx", collect(1:2:p.n)), ("even_idx", collect(2:2:p.n))]

# Pull the (reduced χ², mean) pair out of one row of the printed table.
function _row_stats(out::AbstractString, label::AbstractString)
    for line in split(out, '\n')
        occursin(label, line) || continue
        nums = [parse(Float64, m.match) for m in eachmatch(r"[-+]?\d+\.\d+", line)]
        length(nums) >= 3 && return (nums[1], nums[3])   # rχ², its std, mean
    end
    return nothing
end

@testset "minisanity latent rows use the absolute sample z + s" begin
    n = 8
    prob = MiniSanityProblem(n, zeros(n))

    # A deliberately off-centre mean, and one antithetic pair. For samples
    # ±s the absolute statistic collapses to (‖z‖² + ‖s‖²)/nd per block and
    # the mean column to mean(z); the offset-only version gives ‖s‖²/nd and 0.
    z = collect(range(0.5, 2.0; length=n))
    s = fill(0.25, n)
    samples = [s, -s]

    out = _capture(() -> VarInf._report_latents(prob, z, samples))

    for (label, rng) in VarInf.latent_blocks(prob)
        nd  = length(rng)
        zb  = view(z, rng)
        sb  = view(s, rng)
        want_rchi = (sum(abs2, zb) + sum(abs2, sb)) / nd      # (‖μ‖² + tr Σ)/n
        want_mean = sum(zb) / nd
        bad_rchi  = sum(abs2, sb) / nd                        # the old, offset-only value

        got = _row_stats(out, label)
        @test got !== nothing
        got_rchi, got_mean = got
        @test isapprox(got_rchi, want_rchi; atol=5e-3)
        @test isapprox(got_mean, want_mean; atol=5e-3)
        # Guard against a regression to the offset-only statistic.
        @test !isapprox(want_rchi, bad_rchi; atol=1e-6)
        @test !isapprox(got_rchi, bad_rchi; atol=5e-3)
        @test abs(want_mean) > 0.1          # the mean column carries information
    end
end

@testset "reconstruct_hybrid runs GeoVI-only (n_mgvi = 0)" begin
    rng  = MersenneTwister(7)
    n    = 6
    prob = MiniSanityProblem(n, 0.3 .* randn(rng, n))

    z, samples = reconstruct_hybrid(prob; z0=zeros(n), n_mgvi=0, n_geovi=2,
                                    n_samples=2, kl_maxiter=3, map_maxiter=5,
                                    geo_newton_maxiter=2, geo_cg_maxiter=5,
                                    cg_maxiter=10, verb=false)
    @test length(z) == n
    @test all(isfinite, z)
    @test !isempty(samples)
    @test all(s -> all(isfinite, s), samples)

    # A user-supplied schedule that asks for :nonlinear_update first still
    # errors, but now says what to do about it.
    err = try
        reconstruct_hybrid(prob; z0=zeros(n), n_mgvi=0, n_geovi=1, n_samples=2,
                           sample_mode=_ -> :nonlinear_update, verb=false)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("nonlinear_resample", err.msg)
end

@testset ":nonlinear_update reports the sample count actually in use" begin
    rng  = MersenneTwister(11)
    n    = 6
    prob = MiniSanityProblem(n, 0.3 .* randn(rng, n))

    # Ask for 2 pairs on the resample iteration, then 64 on the update
    # iteration, which cannot honour it. The header must not announce 64.
    out = _capture() do
        reconstruct_hybrid(prob; z0=zeros(n), n_mgvi=1, n_geovi=1,
                           n_samples=iter -> iter == 1 ? 2 : 64,
                           kl_maxiter=3, map_maxiter=5, geo_newton_maxiter=2,
                           geo_cg_maxiter=5, cg_maxiter=10, verb=true)
    end

    upd = [l for l in split(out, '\n') if occursin("nonlinear_update iteration", l)]
    @test length(upd) == 1
    @test occursin("n_samples=2", upd[1])
    @test !occursin("n_samples=64", upd[1])
    @test occursin("reuses the 2 existing", out)
end

# A badly conditioned quadratic, so a small budget cannot converge it.
struct StiffProblem <: AbstractInferenceProblem
    n::Int
    cond::Float64
end
_stiff_diag(p::StiffProblem) = exp10.(range(0, log10(p.cond); length=p.n))
VarInf.latent_size(p::StiffProblem) = p.n
VarInf.data_size(p::StiffProblem)   = p.n
VarInf.transformation(p::StiffProblem, z::AbstractVector{<:Real}) =
    sqrt.(_stiff_diag(p)) .* z
VarInf.right_sqrt_metric(p::StiffProblem, _, v::AbstractVector{<:Real}) =
    sqrt.(_stiff_diag(p)) .* v
VarInf.left_sqrt_metric(p::StiffProblem, _, w::AbstractVector{<:Real}) =
    sqrt.(_stiff_diag(p)) .* w
function VarInf.energy_and_gradient(p::StiffProblem, z::AbstractVector{<:Real})
    d = _stiff_diag(p)
    shifted = z .- 1.0
    return 0.5 * sum(d .* abs2.(shifted)), d .* shifted
end

@testset "reconstruct_map reports convergence, and the real evaluation count" begin
    easy = MiniSanityProblem(6, zeros(6))
    z, info = reconstruct_map_with_info(easy; z0=0.5 .* ones(6), maxiter=500,
                                        verb=false)
    @test info.converged
    @test info.gnorm <= info.gtest
    @test info.n_evals >= 1
    @test info.n_evals <= 500
    @test info.maxiter == 500
    @test isfinite(info.energy)
    @test all(isfinite, z)

    # reconstruct_map must still return a bare vector (non-breaking).
    z_plain = reconstruct_map(easy; z0=0.5 .* ones(6), maxiter=500, verb=false)
    @test z_plain isa AbstractVector
    @test z_plain ≈ z

    # Optimiser-limited: a stiff problem on a tiny budget must report it rather
    # than claiming it used the whole budget, which the old code always did.
    stiff = StiffProblem(40, 1e9)
    _, bad = reconstruct_map_with_info(stiff; z0=zeros(40), maxiter=3, verb=false)
    @test !bad.converged
    @test bad.gnorm > bad.gtest

    # The printed line must carry the real count and the verdict, not `maxiter`.
    out = _capture() do
        reconstruct_map(stiff; z0=zeros(40), maxiter=3, verb=true)
    end
    @test occursin("NOT converged", out)
    @test occursin("evals", out)
    @test !occursin("(3 iters)", out)
end

@testset "latent_blocks accepts non-contiguous index vectors" begin
    # The blocks that matter can be geometric rather than positional, so they
    # need not be contiguous in the latent layout.
    n = 6
    prob = IndexBlockProblem(n)
    z = collect(range(0.5, 2.0; length=n))
    s = fill(0.25, n)
    out = _capture() do
        VarInf._report_latents(prob, z, [s, -s])
    end

    for (label, idx) in VarInf.latent_blocks(prob)
        nd = length(idx)
        want_rchi = (sum(abs2, view(z, idx)) + sum(abs2, view(s, idx))) / nd
        want_mean = sum(view(z, idx)) / nd
        line = only([l for l in split(out, '\n') if occursin(label, l)])
        nums = [parse(Float64, m.match) for m in eachmatch(r"[-+]?\d+\.\d+", line)]
        @test isapprox(nums[1], want_rchi; atol=5e-3)
        @test isapprox(nums[3], want_mean; atol=5e-3)
        @test occursin(string(nd), split(line)[end])      # correct # dof
    end
end
