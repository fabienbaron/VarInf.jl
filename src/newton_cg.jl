# ============================================================================
# NIFTy Newton-CG: direct translation from nifty/src/re/optimize.py
# ============================================================================

"""
    _nifty_newton_cg(fun_and_grad, hessp, x0; kwargs...) -> x_opt

Newton-CG optimizer, direct translation of NIFTy's `_newton_cg`
(nifty/src/re/optimize.py:144-282).

Arguments:
- `fun_and_grad(x)`: returns `(energy, gradient)`, the objective and its gradient
- `hessp(x, v)`: returns `H·v`, the Hessian-vector product at x
- `x0`: initial position

Key features matching NIFTy:
- CG inner solve with adaptive tolerance (SciPy formula, L1 norm)
- Energy-based CG termination (energy_reduction_factor)
- Backtracking line search with 9 halvings
- Cauchy step fallback at halving 5
- Convergence on absolute energy change or gradient norm
- `xtol` compared to the raw L1 gradient norm (matches NIFTy `optimize.py`:
  `descent_norm ≤ xtol`, i.e. NOT scaled by the problem dimension)
"""
function _nifty_newton_cg(fun_and_grad, hessp, x0::AbstractVector{T};
                          maxiter::Int=200,
                          xtol::Real=1e-5,
                          absdelta::Union{Nothing, Real}=nothing,
                          energy_reduction_factor::Real=0.1,
                          convergence_level::Int=1,
                          miniter::Int=0,
                          cg_maxiter::Int=200,
                          custom_gradnorm=nothing,
                          verb::Bool=false) where {T<:AbstractFloat}
    pos = copy(x0)
    energy, g = fun_and_grad(pos)
    old_fval = T(Inf)

    gradnorm_func = custom_gradnorm === nothing ? (dd -> norm(dd, 1)) : custom_gradnorm

    if isnan(energy)
        verb && println("    Newton-CG: initial energy is NaN, aborting")
        return pos
    end

    ccount = 0  # convergence counter for absdelta (NIFTy AbsDeltaEnergyController)

    for i in 1:maxiter
        # CG tolerance from prior energy improvement (NIFTy optimize.py:188-191)
        if isfinite(old_fval) && energy_reduction_factor > 0 && old_fval > energy
            cg_absdelta = energy_reduction_factor * (old_fval - energy)
        else
            cg_absdelta = absdelta === nothing ? nothing : absdelta / 100.0
        end

        # Adaptive residual norm: SciPy formula, L1 (NIFTy optimize.py:192-197)
        mag_g = norm(g, 1)
        cg_resnorm = min(0.5, sqrt(mag_g)) * mag_g

        # CG solve: H·nat_g = g
        nat_g, cg_niter = _nifty_cg(v -> hessp(pos, v), g;
                                     absdelta=cg_absdelta,
                                     resnorm=cg_resnorm,
                                     norm_ord=1,
                                     maxiter=cg_maxiter)

        # Line search (NIFTy optimize.py:212-237)
        dd = nat_g
        grad_scaling = one(T)
        accepted = false
        new_energy = energy
        new_g = g
        naive_ls_it = -1

        for ls_iter in 0:8
            naive_ls_it = ls_iter
            # NIFTy halves at the START of each iteration (except first)
            if ls_iter > 0
                grad_scaling /= 2
            end

            new_pos = pos .- grad_scaling .* dd
            e_try, g_try = fun_and_grad(new_pos)

            if e_try <= energy
                new_energy = e_try
                new_g = g_try
                pos .= new_pos
                accepted = true
                break
            end

            # Cauchy step fallback at iteration 5 (NIFTy optimize.py:224-230)
            if ls_iter == 5
                gam = dot(g, g)
                Hg = hessp(pos, g)
                curv = dot(g, Hg)
                if curv > 0
                    grad_scaling = one(T)
                    dd = (gam / curv) .* g
                end
            end
        end

        if !accepted
            verb && @printf("    Newton %d: line search failed, stopping\n", i)
            break
        end

        energy_diff = energy - new_energy
        old_fval = energy
        energy = new_energy
        g = new_g

        descent_norm = grad_scaling * gradnorm_func(dd)

        verb && @printf("    Newton %d: E=%.4f, ΔE=%.2e, |d|=%.4f, α=%.3f, CG:%d\n",
                        i, energy, energy_diff, descent_norm, grad_scaling, cg_niter)

        if isnan(energy)
            verb && println("    Newton-CG: energy is NaN, aborting")
            break
        end

        # Convergence 1: absolute delta with convergence_level counter
        # (NIFTy AbsDeltaEnergyController: counter increments when ΔE < absdelta,
        #  decrements otherwise; converges when counter reaches convergence_level)
        if absdelta !== nothing
            if 0.0 <= energy_diff < absdelta && naive_ls_it < 2
                ccount += 1
            else
                ccount = max(0, ccount - 1)
            end
            if ccount >= convergence_level && i > miniter
                verb && @printf("    Converged: ΔE=%.2e < %.2e (%d consecutive)\n",
                                energy_diff, absdelta, ccount)
                break
            end
        end

        # Convergence 2: gradient norm (NIFTy optimize.py:263)
        if descent_norm <= xtol && i > miniter
            verb && @printf("    Converged: |d|=%.2e < %.2e\n", descent_norm, xtol)
            break
        end
    end

    return pos
end
