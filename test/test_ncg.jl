# Minimizer correctness tests — port of the reliable core of NIFTy's test_ncg.py.
# The suite's existing CG/Newton-CG checks (runtests.jl) use a trivial 2×2 system;
# these exercise ILL-CONDITIONED SPD problems, where a mistuned tolerance or
# restart actually shows up. Precision-parameterized on T.

using VarInf
using Test
using Random
using LinearAlgebra

# Build a random SPD matrix with a prescribed condition number.
function _spd(n::Int, cond::Real, ::Type{T}; seed::Int=0) where {T}
    rng = MersenneTwister(seed)
    Q = qr(randn(rng, T, n, n)).Q
    λ = exp.(range(zero(T), log(T(cond)), length=n))   # eigenvalues in [1, cond]
    return Matrix(Q) * Diagonal(λ) * Matrix(Q)'
end

@testset "Minimizer correctness (ill-conditioned)" begin
    T = Float64
    n = 30

    @testset "CG solves an ill-conditioned SPD system" begin
        A = _spd(n, 1e3, T; seed=1)
        A = (A + A') ./ 2                       # symmetrize away round-off
        b = randn(MersenneTwister(2), T, n)
        x_cg, _ = VarInf._nifty_cg(v -> A * v, b; tol=1e-11, maxiter=5000)
        @test norm(x_cg .- (A \ b)) / norm(A \ b) < 1e-6
    end

    @testset "Newton-CG minimizes an ill-conditioned quadratic" begin
        A = _spd(n, 1e3, T; seed=3)
        A = (A + A') ./ 2
        b = randn(MersenneTwister(4), T, n)
        fg(x) = (0.5 * dot(x, A * x) - dot(b, x), A * x .- b)
        hp(_, v) = A * v
        x_opt = VarInf._nifty_newton_cg(fg, hp, zeros(T, n);
                                        maxiter=200, cg_maxiter=1000, xtol=1e-11)
        @test norm(x_opt .- (A \ b)) / norm(A \ b) < 1e-5
    end
end

# A well-conditioned system solves accurately at BOTH precisions (an
# ill-conditioned one cannot: Float32 accuracy is capped at ~eps·cond, so the
# tight bars above are Float64-only). Bar scales with sqrt(eps(T)).
@testset "Minimizer correctness (well-conditioned, $T)" for T in (Float32, Float64)
    n = 20
    A = _spd(n, 10, T; seed=5); A = (A + A') ./ 2
    b = randn(MersenneTwister(6), T, n)
    xref = A \ b
    x_cg, _ = VarInf._nifty_cg(v -> A * v, b; tol=sqrt(eps(T)), maxiter=2000)
    @test norm(x_cg .- xref) / norm(xref) < sqrt(eps(T))
    fg(x) = (dot(x, A * x) / 2 - dot(b, x), A * x .- b)
    hp(_, v) = A * v
    x_opt = VarInf._nifty_newton_cg(fg, hp, zeros(T, n);
                                    maxiter=100, cg_maxiter=500, xtol=sqrt(eps(T)))
    @test norm(x_opt .- xref) / norm(xref) < sqrt(eps(T))
end
