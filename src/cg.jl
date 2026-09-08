# ============================================================================
# NIFTy CG: direct translation from nifty/src/re/conjugate_gradient.py
# ============================================================================

const CG_N_RESET = 20  # residual restart frequency (NIFTy conjugate_gradient.py)

"""
    _nifty_cg(mat, j; kwargs...) -> (x, niter)

Conjugate gradient solver, direct translation of NIFTy's `_cg`
(nifty/src/re/conjugate_gradient.py:77-214).

Solves M·x = j where `mat(v)` computes M·v.

Key features matching NIFTy:
- Negative curvature detection (returns last good iterate)
- Energy-based termination (`absdelta`)
- Residual norm termination (`resnorm`, with configurable `norm_ord`)
- Minimum iteration count before early termination
- Residual restart every 20 iterations for numerical stability
- Polak-Ribière direction update with safeguard
"""
function _nifty_cg(mat, j::AbstractVector{T};
                   x0::Union{Nothing, AbstractVector{T}}=nothing,
                   absdelta::Union{Nothing, Real}=nothing,
                   resnorm::Union{Nothing, Real}=nothing,
                   norm_ord::Real=2,
                   tol::Real=1e-5,
                   atol::Real=0,
                   miniter::Union{Nothing, Int}=nothing,
                   maxiter::Union{Nothing, Int}=nothing) where {T<:AbstractFloat}
    n = length(j)
    maxiter_fallback = 20 * n
    _miniter = miniter === nothing ? min(6, maxiter === nothing ? maxiter_fallback : maxiter) : miniter
    _maxiter = maxiter === nothing ? max(min(200, maxiter_fallback), _miniter) : maxiter

    # Convergence criterion setup
    if absdelta === nothing && resnorm === nothing
        resnorm = max(tol * norm(j, norm_ord), atol)
    end

    eps_val = T(6) * eps(T)
    tiny_val = T(6) * floatmin(T)

    # Initial state
    if x0 === nothing
        pos = zeros(T, n)
        r = -j
        d = copy(r)
        energy = zero(T)
        nfev = 0
    else
        pos = copy(x0)
        r = mat(pos) .- j
        d = copy(r)
        energy = dot((r .- j) ./ 2, pos)
        nfev = 1
    end

    previous_gamma = dot(r, r)

    if previous_gamma == 0.0
        return pos, 0
    end

    info = -1
    niter = 0

    for i in 1:_maxiter
        niter = i
        q = mat(d)
        nfev += 1

        curv = dot(d, q)

        # Handle non-positive definite matrix
        if curv == 0.0
            info = 0
            break
        elseif curv < 0.0
            if i > 1
                info = 0
                break
            else
                pos .= (previous_gamma / (-curv)) .* (.-j)
                info = 0
                break
            end
        end

        alpha = previous_gamma / curv
        pos .-= alpha .* d

        # Restart residual every CG_N_RESET iterations (numerical stability)
        if i % CG_N_RESET == 0
            r .= mat(pos) .- j
            nfev += 1
        else
            r .-= q .* alpha
        end

        gamma = dot(r, r)

        # Check: residual too small
        if gamma >= 0.0 && gamma <= tiny_val
            info = 0
            break
        end

        # Check: residual norm convergence
        if resnorm !== nothing
            if norm(r, norm_ord) < resnorm && i >= _miniter
                info = 0
                break
            end
        end

        # Check: energy
        new_energy = dot((r .- j) ./ 2, pos)
        energy_diff = energy - new_energy
        neg_energy_eps = -eps_val * abs(new_energy)

        # Energy increased → indefinite
        if energy_diff < neg_energy_eps
            info = i
            break
        end

        # Absolute delta convergence
        if absdelta !== nothing && energy_diff < absdelta && i >= _miniter
            info = 0
            break
        end

        energy = new_energy

        # Polak-Ribière direction update with safeguard
        beta = max(zero(T), gamma / previous_gamma)
        d .= beta .* d .+ r
        previous_gamma = gamma
    end

    if info == -1
        info = niter
    end

    return pos, niter
end
