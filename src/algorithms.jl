# ============================================================================
# Generic VI algorithms (MAP / MGVI / GeoVI / Hybrid) and their helpers.
#
# Everything in this file operates on an AbstractInferenceProblem via the
# six protocol methods (energy_and_gradient, latent_size, transformation,
# right_sqrt_metric, left_sqrt_metric, data_size). Nothing here knows about
# sky models, NFFT, or any other domain.
#
# Helpers (private):
#   _draw_metric_sample, _posterior_metric_mul, _draw_samples_geovi,
#   _geovi_residual_vg, _geovi_metric_mul, _geovi_sampnorm,
#   _refine_sample_geovi, _kl_newton_cg, _kl_newton_cg_frozen
#
# Entry points (exported):
#   reconstruct_map(prob; ...)    → z_opt
#   reconstruct_map_with_info(prob; ...) → (z_opt, convergence info)
#   reconstruct_mgvi(prob; ...)   → (z_center, samples)
#   reconstruct_geovi(prob; ...)  → (z_center, samples)
#   reconstruct_hybrid(prob; ...) → (z_center, samples)
# ============================================================================

using Krylov
using LinearMaps
using OptimPackNextGen


# ============================================================================
# Per-iteration latent reporting (NIFTy minisanity analogue)
# ============================================================================

# Generic, problem-agnostic latent diagnostic, evaluated at the ABSOLUTE samples
# z + s (not the mean-zero offsets s), matching NIFTy's reduced_residual_stats
# over Samples.samples = pos + offset. Each block reports
# ⟨‖z+s‖²⟩/n = (‖μ‖² + tr Σ)/n → 1 and ⟨z+s⟩/n → 0 under a calibrated prior.
# Evaluating on s alone gives tr Σ/n, a posterior variance compared against a
# target meant for the full second moment, and drives the mean column to 0 by
# antithetic construction whatever the posterior does. Followed by the optional
# problem-specific `report_latents` hook. Both gated by `verb` at the call sites.
# Printf helper: one aligned "label  value  (note)" row, used by problem
# `report_latents` methods for decoded physical hyperparameters printed under
# the minisanity table (label column width = 18).
report_row(label::AbstractString, value::AbstractString, note::AbstractString="") =
    isempty(note) ? @printf("     %-18s %s\n", label, value) :
                    @printf("     %-18s %s   %s\n", label, value, note)

# Sample mean / std (Statistics is not a VarInf dependency).
_smean(v) = sum(v) / length(v)
_sstd(v)  = length(v) < 2 ? 0.0 :
            sqrt(sum(x -> (x - _smean(v))^2, v) / (length(v) - 1))

# A reduced-χ² cell ("mean ± std", right-justified to width 13), wrapped in an
# ANSI colour when stdout supports it: orange outside [½, 2], red outside
# [⅕, 5], matching NIFTy's minisanity thresholds.
function _rchi_cell(m::Real, s::Real)
    cell = lpad(@sprintf("%.2f ± %.2f", m, s), 13)
    get(stdout, :color, false) || return cell
    code = (m > 5 || m < 0.2) ? 31 : (m > 2 || m < 0.5) ? 33 : 0
    code == 0 ? cell : "\e[1;$(code)m$cell\e[0m"
end

# NIFTy-style minisanity table: per-domain reduced χ² (= ⟨standardized²⟩ → 1),
# mean (→ 0) and # dof, for the data residual and each latent block, followed
# by the optional problem-specific decoded report.
function _report_latents(prob::AbstractInferenceProblem,
                         z::AbstractVector{<:Real},
                         samples::AbstractVector{<:AbstractVector{<:Real}})
    isempty(samples) && (report_latents(prob, z, samples); return nothing)
    hdr = @sprintf("    %-18s%13s   %-14s%8s", "", "reduced χ²", "mean", "# dof")
    println("  ══ minisanity ", "═"^(length(hdr) - 16))
    println(hdr)

    _row(label, rc, mn, nd) = begin
        print("    ", rpad(label, 18), _rchi_cell(_smean(rc), _sstd(rc)), "   ")
        mn === nothing ? @printf("%-14s%8d\n", "—", nd) :
            @printf("%-14s%8d\n", (@sprintf("%+.2f ± %.2f", _smean(mn), _sstd(mn))), nd)
    end

    # Data residual row (normalized residual if whitened_data is available,
    # else the energy-identity reduced χ² with no mean).
    Nd = data_size(prob)
    wd = whitened_data(prob)
    if wd === nothing
        rc = [2 * (energy_and_gradient(prob, z .+ s)[1] -
                   prior_energy(prob, z .+ s)) / Nd for s in samples]
        _row("data residual", rc, nothing, Nd)
    else
        res = [transformation(prob, z .+ s) .- wd for s in samples]
        _row("data residual", [sum(abs2, r) / Nd for r in res],
             [sum(r) / length(r) for r in res], Nd)
    end

    # Latent-space rows, one per declared block. Whitened once per sample for
    # the whole vector (C^{-1/2}(z+s)), so the target stays 1 under a non-unit
    # prior covariance; with the default C = I this is just z + s. Exact per
    # block only when C is block-diagonal w.r.t. the declared blocks, which the
    # per-block split already assumes.
    whitened = [prior_inv_sqrt_covariance_mul(prob, z, z .+ s) for s in samples]
    total_latent = 0
    for (name, rng) in latent_blocks(prob)
        nd = length(rng)
        total_latent += nd
        _row(name, [sum(abs2, view(w, rng)) / nd for w in whitened],
             [sum(view(w, rng)) / nd for w in whitened], nd)
    end
    @printf("    %-18s%13s   %-14s%8d\n", "Σ latent (minimized)", "", "", total_latent)
    println("  ", "═"^(length(hdr) - 2))

    report_latents(prob, z, samples)   # decoded physical hypers (optional)
    return nothing
end

# ============================================================================
# GeoVI metric sampling helpers
# ============================================================================

"""
    _draw_metric_sample(prob, z) -> ms ~ N(0, M)

Draw a sample from the posterior metric M = J_T'·J_T + C⁻¹ (C = I by default).
"""
function _draw_metric_sample(prob::AbstractInferenceProblem,
                             z::AbstractVector{<:Real})
    T = eltype(z)
    w_data = randn(T, data_size(prob))
    w_latent = randn(T, latent_size(prob))
    lh_sample = left_sqrt_metric(prob, z, w_data)
    return lh_sample .+ prior_inv_sqrt_covariance_mul(prob, z, w_latent)
end

"""
    _posterior_metric_mul(prob, z, v) -> M·v

Apply the posterior metric M = J_T'·J_T + C⁻¹ to vector v (C = I by default).
"""
function _posterior_metric_mul(prob::AbstractInferenceProblem,
                               z::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    Rv = right_sqrt_metric(prob, z, v)
    LRv = left_sqrt_metric(prob, z, Rv)
    return LRv .+ prior_inv_covariance_mul(prob, z, v)
end

# ============================================================================
# Linear sampling
# ============================================================================

"""
    _draw_samples_geovi(prob, z, n_samples; kwargs...)
        -> (samples, metric_samples, noise)

Draw linear residual samples δ ~ N(0, M⁻¹) via CG inversion of metric samples.
Also returns the underlying white-noise pairs `noise[k] = (w_data, w_latent)`
so callers can recompute the metric sample at a different mean (NIFTy's
nonlinear_update keeps the white noise fixed and re-evaluates at the new mean).
"""
function _draw_samples_geovi(prob::AbstractInferenceProblem,
                             z::AbstractVector{<:Real},
                             n_samples::Int;
                             cg_maxiter=200, cg_tol=0.01, verb=false)
    T = eltype(z)
    n = latent_size(prob)
    n_data = data_size(prob)
    mat = v -> _posterior_metric_mul(prob, z, v)

    samples = Vector{Vector{T}}(undef, n_samples)
    metric_samples = Vector{Vector{T}}(undef, n_samples)
    noise = Vector{Tuple{Vector{T},Vector{T}}}(undef, n_samples)

    for k in 1:n_samples
        w_data = randn(T, n_data)
        w_latent = randn(T, n)
        lh_sample = left_sqrt_metric(prob, z, w_data)
        ms = lh_sample .+ w_latent
        delta, niter = _nifty_cg(mat, ms; x0=w_latent,
                                  maxiter=cg_maxiter, tol=cg_tol)
        samples[k] = delta
        metric_samples[k] = ms
        noise[k] = (w_data, w_latent)
        verb && @printf("    CG sample %d: |δ|=%.3f (%d iters)\n", k, norm(delta), niter)
    end
    return samples, metric_samples, noise
end

# ============================================================================
# GeoVI sample refinement: helper functions
# ============================================================================

"""
    _geovi_residual_vg(prob, x, center, T_at_e, metric_sample)
        -> (energy, neg_gradient)

GeoVI objective: 0.5 * ||ms - g(x)||² where g(x) = C^{-1/2}(x - e) + L(e, T(x) - T(e)),
so ∂g/∂x = C^{-1/2} + L(e,·)·R(x,·) and gᵀg carries the metric JᵀJ + C⁻¹.
C = I by default, which recovers g(x) = x - e + L(e, T(x) - T(e)).

Returns `(energy, gradient)`. NIFTy's `residual_vg` computes `ngrad = (∂g/∂x)ᵀr`
and returns `(res, -ngrad)`; since `∂E/∂x = -(∂g/∂x)ᵀr`, that negated value is
the true gradient, which is what `_nifty_newton_cg` expects.
"""
function _geovi_residual_vg(prob::AbstractInferenceProblem,
                            x::AbstractVector{<:Real}, center::AbstractVector{<:Real},
                            T_at_e::AbstractVector{<:Real},
                            metric_sample::AbstractVector{<:Real})
    T_at_x = transformation(prob, x)
    t = T_at_x .- T_at_e
    Lc_t = left_sqrt_metric(prob, center, t)
    g_val = prior_inv_sqrt_covariance_mul(prob, center, x .- center) .+ Lc_t
    r = metric_sample .- g_val
    energy = 0.5 * dot(r, r)

    # Gradient: ngrad = (∂g/∂x)ᵀ·r = C^{-1/2}r + L(x, R(e, r))  (NIFTy evi.py:167-168)
    Rc_r = right_sqrt_metric(prob, center, r)
    Lx_Rc_r = left_sqrt_metric(prob, x, Rc_r)
    neg_grad = prior_inv_sqrt_covariance_mul(prob, center, r) .+ Lx_Rc_r

    return energy, .-neg_grad  # NIFTy returns (res, -ngrad)
end

"""
    _geovi_metric_mul(prob, x, v, center) -> H·v

GeoVI Hessian-vector product (NIFTy evi.py:171-176), written as AᵀA·v for
A = ∂g/∂x = C^{-1/2} + L(e,·)·R(x,·):
H·v = C^{-1/2}(A·v) + L(x, R(e, A·v)),   A·v = C^{-1/2}v + L(e, R(x, v))
With C = I this is NIFTy's v + L(e,R(x,v)) + L(x,R(e, v + L(e,R(x,v)))) unchanged.
"""
function _geovi_metric_mul(prob::AbstractInferenceProblem,
                           x::AbstractVector{<:Real}, v::AbstractVector{<:Real},
                           center::AbstractVector{<:Real})
    Rxv = right_sqrt_metric(prob, x, v)
    LcRxv = left_sqrt_metric(prob, center, Rxv)
    Av = prior_inv_sqrt_covariance_mul(prob, center, v) .+ LcRxv
    Rc_Av = right_sqrt_metric(prob, center, Av)
    Lx_Rc_Av = left_sqrt_metric(prob, x, Rc_Av)
    return prior_inv_sqrt_covariance_mul(prob, center, Av) .+ Lx_Rc_Av
end

"""
    _geovi_sampnorm(prob, dd, center) -> Float64

Custom gradient norm for GeoVI convergence (NIFTy evi.py:178-181):
sqrt(||C^{-1/2}dd||² + ||R(e, dd)||²), i.e. NIFTy's sqrt(||dd||² + ||R(e,dd)||²)
when C = I.
"""
function _geovi_sampnorm(prob::AbstractInferenceProblem,
                         dd::AbstractVector{<:Real}, center::AbstractVector{<:Real})
    Rc_dd = right_sqrt_metric(prob, center, dd)
    Sdd = prior_inv_sqrt_covariance_mul(prob, center, dd)
    return sqrt(dot(Sdd, Sdd) + dot(Rc_dd, Rc_dd))
end

# ============================================================================
# GeoVI nonlinear sample refinement
# ============================================================================

"""
    _refine_sample_geovi(prob, center, delta_lin; kwargs...)
        -> delta_refined

Nonlinearly refine a linear MGVI sample via Newton-CG.
Uses `_nifty_newton_cg` with GeoVI-specific objective, Hessian, and gradient norm.
Falls back to linear sample if refinement diverges.
"""
function _refine_sample_geovi(prob::AbstractInferenceProblem,
                              center::AbstractVector{<:Real},
                              delta_lin::AbstractVector{<:Real};
                              metric_sample::Union{Nothing,AbstractVector{<:Real}}=nothing,
                              newton_maxiter=200,
                              cg_maxiter=200,
                              tol=1e-5,
                              absdelta=0.0005,
                              verb=false)
    T_at_e = transformation(prob, center)
    # Refine the linear sample tied to ITS OWN metric sample (NIFTy geoVI): the
    # linear δ = M⁻¹·ms is the starting point and Newton finds the nonlinear
    # preimage of the same ms. The caller forms the antithetic partner by calling
    # again with (−δ_lin, −ms): separate refinements sharing the negated ms (NIFTy
    # draw_residual), not a mirror of this δ (a mirror lands off a curved manifold).
    ms = metric_sample === nothing ? _draw_metric_sample(prob, center) : metric_sample
    x0 = center .+ delta_lin

    initial_energy, _ = _geovi_residual_vg(prob, x0, center, T_at_e, ms)

    fun_and_grad = x -> _geovi_residual_vg(prob, x, center, T_at_e, ms)

    hessp = (x, v) -> _geovi_metric_mul(prob, x, v, center)
    gradnorm = dd -> _geovi_sampnorm(prob, dd, center)

    x_opt = _nifty_newton_cg(fun_and_grad, hessp, x0;
                              maxiter=newton_maxiter,
                              cg_maxiter=cg_maxiter,
                              xtol=tol,
                              absdelta=absdelta,
                              custom_gradnorm=gradnorm,
                              verb=verb)

    delta = x_opt .- center

    final_energy, _ = _geovi_residual_vg(prob, x_opt, center, T_at_e, ms)
    if final_energy > initial_energy || isnan(final_energy)
        verb && @printf("    Refinement diverged (%.1f > %.1f), using linear sample\n",
                        final_energy, initial_energy)
        return delta_lin
    end
    return delta
end

# ============================================================================
# KL Newton-CG optimizer
# ============================================================================

"""
    _kl_newton_cg(prob, z, samples; kwargs...) -> z_opt

Newton-CG for sample-averaged KL energy, using `_nifty_newton_cg`.

Energy: E_KL(m) = (1/N) Σ_k E(m + δ_k)
Hessian: H_KL(m)·v = (1/N) Σ_k M(m + δ_k)·v
"""
function _kl_newton_cg(prob::AbstractInferenceProblem,
                       z::AbstractVector{<:Real},
                       samples::AbstractVector{<:AbstractVector{<:Real}};
                       maxiter=20,
                       cg_maxiter=200,
                       absdelta::Union{Nothing, Real}=nothing,
                       verb=false)
    T = eltype(z)
    n_smpls = length(samples)
    n = length(z)

    fun_and_grad = pos -> begin
        g = zeros(T, n)
        e_total = zero(T)
        for k in 1:n_smpls
            e_s, g_s = energy_and_gradient(prob, pos .+ samples[k])
            e_total += e_s
            g .+= g_s
        end
        (e_total / n_smpls, g ./ n_smpls)
    end

    hessp = (pos, v) -> begin
        Hv = zeros(T, n)
        for k in 1:n_smpls
            Hv .+= _posterior_metric_mul(prob, pos .+ samples[k], v)
        end
        Hv ./ n_smpls
    end

    verb && @printf("  KL Newton-CG: start E=%.4f\n", fun_and_grad(z)[1])

    kwargs = Dict{Symbol,Any}(:maxiter => maxiter, :cg_maxiter => cg_maxiter, :verb => verb)
    if absdelta !== nothing
        kwargs[:absdelta] = absdelta
    end

    return _nifty_newton_cg(fun_and_grad, hessp, z; kwargs...)
end

"""
    _kl_newton_cg_frozen(prob, z, samples, frozen_ranges; kwargs...)

Like `_kl_newton_cg` but zeros gradients and stiffens the metric for frozen latent directions.
"""
function _kl_newton_cg_frozen(prob::AbstractInferenceProblem,
                              z::AbstractVector{<:Real},
                              samples::AbstractVector{<:AbstractVector{<:Real}},
                              frozen_ranges::Vector{UnitRange{Int}};
                              maxiter=20,
                              cg_maxiter=200,
                              absdelta::Union{Nothing, Real}=nothing,
                              verb=false)
    T = eltype(z)
    n_smpls = length(samples)
    n = length(z)
    _zero!(v) = (for r in frozen_ranges; v[r] .= zero(T); end; v)
    _stiff!(v, u) = (for r in frozen_ranges; v[r] .= T(1e6) .* u[r]; end; v)

    fun_and_grad = pos -> begin
        g = zeros(T, n)
        e_total = zero(T)
        for k in 1:n_smpls
            z_s = pos .+ samples[k]
            _zero!(z_s)
            e_s, g_s = energy_and_gradient(prob, z_s)
            e_total += e_s
            g .+= g_s
        end
        g_avg = g ./ n_smpls
        _zero!(g_avg)
        (e_total / n_smpls, g_avg)
    end

    hessp = (pos, v) -> begin
        Hv = zeros(T, n)
        for k in 1:n_smpls
            z_s = pos .+ samples[k]
            _zero!(z_s)
            Hv .+= _posterior_metric_mul(prob, z_s, v)
        end
        Hv ./= n_smpls
        _stiff!(Hv, v)
        return Hv
    end

    verb && @printf("  KL Newton-CG: start E=%.4f\n", fun_and_grad(z)[1])

    kwargs = Dict{Symbol,Any}(:maxiter => maxiter, :cg_maxiter => cg_maxiter, :verb => verb)
    if absdelta !== nothing
        kwargs[:absdelta] = absdelta
    end

    return _nifty_newton_cg(fun_and_grad, hessp, z; kwargs...)
end
# ============================================================================
# Generic MAP: operates on any AbstractInferenceProblem
# ============================================================================

"""
    reconstruct_map(prob::AbstractInferenceProblem; kwargs...) -> z_opt

Generic MAP: minimize posterior energy `E(z) = -log p(z|data)` via L-BFGS.
Uses only `energy_and_gradient(prob, z)` and `latent_size(prob)`, so it works
on any problem implementing the protocol.

Keyword arguments:
- `z0`           : initial latent vector (default: 0.01·randn(latent_size(prob)))
- `maxiter`      : maximum L-BFGS iterations (default: 200)
- `gtol`         : gradient tolerance tuple for vmlmb (default: (1e-6, 1e-6))
- `verb`         : print start/final energy

Returns the optimized latent vector. Domain wrappers may post-process it
(e.g. compute an image cube, run domain-specific diagnostics).

Use [`reconstruct_map_with_info`] when you need to know whether the optimizer
actually converged: an unconverged MAP is silently optimiser-limited, and the
result can be far from the true optimum.
"""
reconstruct_map(prob::AbstractInferenceProblem; kwargs...) =
    first(reconstruct_map_with_info(prob; kwargs...))

"""
    reconstruct_map_with_info(prob; kwargs...) -> (z_opt, info)

As [`reconstruct_map`], but also returns a `NamedTuple` of convergence
diagnostics:

- `converged` : did the gradient-norm test pass (see below)?
- `gnorm`     : ‖∇E‖ at `z_opt`
- `gnorm0`    : ‖∇E‖ at the starting point
- `gtest`     : the threshold `gnorm` was compared against
- `n_evals`   : objective/gradient evaluations used
- `energy`    : E at `z_opt`
- `maxiter`   : the budget that was given

`converged` reproduces `vmlmb`'s own stopping test, `‖g‖ ≤ max(gtol[1],
gtol[2]·‖g₀‖)`, evaluated at the returned point; `vmlmb` computes a stopping
reason internally but does not return it, and its `printer` hook only fires when
`verb > 0`, so it cannot report anything for a silent run.

`converged == false` means the answer is **optimiser-limited**, not a MAP: raise
`maxiter`, loosen `gtol`, or improve the conditioning. This matters because an
optimiser-limited fit fails silently and plausibly — a badly conditioned prior
square root (a long-correlation-length Gaussian process, say) can leave χ²
*increasing* as the prior widens, which is impossible for a converged fit.
"""
function reconstruct_map_with_info(prob::AbstractInferenceProblem;
                                   z0::Union{Nothing, AbstractVector{<:Real}}=nothing,
                                   maxiter::Int=200,
                                   gtol::Tuple{Real,Real}=(1e-6, 1e-6),
                                   verb::Bool=true)
    n = latent_size(prob)
    T = z0 === nothing ? Float64 : float(eltype(z0))
    z = z0 === nothing ? T(0.01) .* randn(T, n) : Vector{T}(z0)

    e0, g0 = energy_and_gradient(prob, z)
    gnorm0 = norm(g0)
    verb && @printf("  MAP start energy: %.4f  (|g|=%.3e)\n", e0, gnorm0)

    n_evals = 0
    function crit(x, g)
        e, gv = energy_and_gradient(prob, x)
        g .= gv
        n_evals += 1
        return e
    end

    z_opt = OptimPackNextGen.vmlmb(crit, z; verb=verb, maxiter=maxiter,
                                    blmvm=false, gtol=gtol)

    e_final, g_final = energy_and_gradient(prob, z_opt)
    gnorm = norm(g_final)
    gtest = max(gtol[1], gtol[2] * gnorm0)
    converged = gnorm <= gtest
    # The old code printed `maxiter` here, so it always claimed it had used the
    # whole budget whether it converged at iteration 3 or hit the wall. vmlmb
    # does not expose its iteration count for a silent run; evaluations are what
    # we can count honestly.
    if verb
        @printf("  MAP final energy: %.4f  (|g|=%.3e, %d evals, %s)\n",
                e_final, gnorm, n_evals,
                converged ? "converged" : "NOT converged: optimiser-limited")
    end
    return z_opt, (; converged, gnorm, gnorm0, gtest, n_evals,
                     energy=e_final, maxiter)
end

# ============================================================================
# Generic MGVI helpers + entry point
# ============================================================================

"""
    reconstruct_mgvi(prob::AbstractInferenceProblem; kwargs...) -> (z_center, samples)

Generic MGVI: warm-start from MAP, then iterate (draw samples from the exact
Gauss-Newton metric M = JᵀJ + I via CG, minimize sample-averaged energy with
antithetic pairs).

Keyword arguments:
- `z0`            : initial latent vector (default: MAP warm-start)
- `n_iterations`  : outer MGVI iterations (default 6)
- `n_samples`     : per-iteration sample count (default 3)
- `map_maxiter`, `kl_maxiter`, `cg_maxiter`, `cg_tol`
- `iter_callback` : `(prob, z, samples, iter) -> nothing` called after each KL step
- `verb`          : print algorithm progress

Returns `(z_center, samples)`. Domain wrappers compute posterior statistics
(e.g. image_mean / image_std) by forward-modeling `z ± samples[k]`.
"""
function reconstruct_mgvi(prob::AbstractInferenceProblem;
                          z0::Union{Nothing, AbstractVector{<:Real}}=nothing,
                          n_iterations::Int=6,
                          n_samples::Int=3,
                          map_maxiter::Int=100,
                          kl_maxiter::Int=80,
                          cg_maxiter::Int=30,
                          cg_tol::Real=0.1,
                          kl_absdelta::Union{Nothing,Real}=0.5,
                          iter_callback=nothing,
                          verb::Bool=true)
    # Step 0: Warm-start from MAP (or use z0 if provided)
    if z0 === nothing
        verb && println("=== MGVI: MAP warm-start ===")
        z = reconstruct_map(prob; maxiter=map_maxiter, verb=verb)
    else
        z = Vector{float(eltype(z0))}(z0)
    end
    T = eltype(z)

    samples = Vector{Vector{T}}()

    for iter in 1:n_iterations
        verb && println("\n=== MGVI iteration $iter/$n_iterations (n_samples=$n_samples) ===")

        verb && println("  Drawing $n_samples metric samples (CG maxiter=$cg_maxiter)...")
        # Sample from the exact Gauss-Newton metric M = JᵀJ + I (NIFTy-style):
        # draw a metric sample ms ~ N(0, M), then solve M·δ = ms ⇒ δ ~ N(0, M⁻¹).
        # (The previous FD-Hessian path drew δ = H⁻¹·N(0,I) ~ N(0, H⁻²) and added
        # a damping·I that double-counted the prior; both biased the covariance.)
        lin_samples, _, _ = _draw_samples_geovi(prob, z, n_samples;
                                              cg_maxiter=cg_maxiter, cg_tol=cg_tol)
        # Exact antithetic pairs ±δ (NIFTy mirror_samples).
        samples = Vector{Vector{T}}(undef, 2 * n_samples)
        for k in 1:n_samples
            samples[2k-1] = lin_samples[k]
            samples[2k]   = -lin_samples[k]
        end

        if verb
            norms = [norm(s) for s in lin_samples]
            @printf("  Sample norms: min=%.3f, max=%.3f, mean=%.3f\n",
                    minimum(norms), maximum(norms), sum(norms)/length(norms))
        end

        # Mean update: Newton-CG on the sample-averaged energy (machine-precise on
        # a quadratic), matching NIFTy. Previously L-BFGS/vmlmb stopped at
        # gtol≈1e-5, giving mean error ~1e-4.
        z = _kl_newton_cg(prob, z, samples;
                          maxiter=kl_maxiter, cg_maxiter=cg_maxiter,
                          absdelta=kl_absdelta, verb=verb)

        verb && _report_latents(prob, z, samples)
        iter_callback === nothing || iter_callback(prob, z, samples, iter)
    end

    return z, samples
end
# ============================================================================
# Generic GeoVI reconstruction (operates on AbstractInferenceProblem)
# ============================================================================

"""
    reconstruct_geovi(prob::AbstractInferenceProblem; kwargs...) -> (z_center, samples)

Generic GeoVI: MAP warm-start, then iterate (draw linear samples, refine via
GeoVI Newton-CG with antithetic pairs, KL Newton-CG on sample-averaged energy).

Keyword arguments:
- `z0`               : initial latent (default: MAP warm-start of `prob`)
- `n_iterations`     : outer iterations (default 6)
- `n_samples`        : Int or `iter -> Int` (default `iter -> iter <= 2 ? 2 : 4`)
- `map_maxiter`, `kl_maxiter`, `kl_absdelta`, `cg_maxiter`, `cg_tol`,
  `geo_newton_maxiter`, `geo_cg_maxiter`, `geo_tol`
- `iter_callback`    : `(prob, z, samples, iter) -> nothing` after each KL step
- `verb`             : algorithm progress

Returns `(z_center, samples)` where `samples` are antithetic pairs `[δ₊_1, δ₋_1, …]`.
Domain wrappers post-process the samples (e.g. into a posterior image cube).
"""
function reconstruct_geovi(prob::AbstractInferenceProblem;
                           z0::Union{Nothing, AbstractVector{<:Real}}=nothing,
                           n_iterations::Int=6,
                           n_samples=iter -> iter <= 2 ? 2 : 4,
                           map_maxiter::Int=200,
                           kl_maxiter::Int=35,
                           kl_absdelta::Union{Nothing,Real}=0.5,
                           cg_maxiter::Int=100,
                           cg_tol::Real=0.01,
                           geo_newton_maxiter::Int=10,
                           geo_cg_maxiter::Int=50,
                           geo_tol::Real=1e-5,
                           iter_callback=nothing,
                           verb::Bool=true)
    if z0 === nothing
        verb && println("=== GeoVI: MAP warm-start ===")
        z = reconstruct_map(prob; maxiter=map_maxiter, verb=verb)
    else
        z = Vector{float(eltype(z0))}(z0)
    end
    T = eltype(z)

    samples = Vector{Vector{T}}()

    for iter in 1:n_iterations
        ns = n_samples isa Function ? n_samples(iter) : n_samples
        verb && println("\n=== GeoVI iteration $iter/$n_iterations (n_samples=$ns) ===")

        verb && println("  Drawing linear samples...")
        lin_samples, metric_samples, _ = _draw_samples_geovi(prob, z, ns;
                                              cg_maxiter=cg_maxiter,
                                              cg_tol=cg_tol, verb=verb)

        if verb
            norms = [norm(s) for s in lin_samples]
            @printf("  Linear sample norms: min=%.3f, max=%.3f, mean=%.3f\n",
                    minimum(norms), maximum(norms), sum(norms)/length(norms))
        end

        # GeoVI antithetic pair (NIFTy draw_residual): refine +δ_lin against +ms
        # and −δ_lin against −ms (the same metric sample, negated) as two
        # separate nonlinear refinements. (Mirroring one refined sample as −δ is
        # only correct for a symmetric posterior; for a curved/banana posterior the
        # mirror lands off the manifold. Sharing ms keeps the antithetic mean
        # cancellation; refining separately tracks the curvature on both branches.)
        verb && println("  Refining antithetic samples (Newton-CG)...")
        samples = Vector{Vector{T}}(undef, 2 * ns)
        for k in 1:ns
            verb && @printf("  Sample %d/%d (+):\n", k, ns)
            samples[2k-1] = _refine_sample_geovi(prob, z, lin_samples[k];
                metric_sample=metric_samples[k], newton_maxiter=geo_newton_maxiter,
                cg_maxiter=geo_cg_maxiter, tol=geo_tol, verb=verb)
            verb && @printf("  Sample %d/%d (−):\n", k, ns)
            samples[2k]   = _refine_sample_geovi(prob, z, -lin_samples[k];
                metric_sample=-metric_samples[k], newton_maxiter=geo_newton_maxiter,
                cg_maxiter=geo_cg_maxiter, tol=geo_tol, verb=verb)
            verb && @printf("    |δ_lin|=%.3f → |δ₊|=%.3f, |δ₋|=%.3f\n",
                            norm(lin_samples[k]), norm(samples[2k-1]), norm(samples[2k]))
        end

        z = _kl_newton_cg(prob, z, samples;
                          maxiter=kl_maxiter, cg_maxiter=cg_maxiter,
                          absdelta=kl_absdelta, verb=verb)

        verb && _report_latents(prob, z, samples)
        iter_callback === nothing || iter_callback(prob, z, samples, iter)
    end

    return z, samples
end

# ============================================================================
# Generic Hybrid MGVI→GeoVI reconstruction (operates on AbstractInferenceProblem)
# ============================================================================

"""
    reconstruct_hybrid(prob::AbstractInferenceProblem; kwargs...) -> (z_center, samples)

Generic hybrid MGVI→GeoVI matching NIFTy's `optimize_kl` strategy. Drives any
problem implementing the protocol.

Keyword arguments:
- `z0`               : initial latent (default: MAP warm-start of `prob`)
- `n_mgvi`, `n_geovi`: number of MGVI / GeoVI outer iterations (defaults 10, 6)
- `n_samples`        : Int or `iter -> Int`
- `sample_mode`      : `iter -> :linear_resample | :nonlinear_resample | :nonlinear_update`
                        (default mirrors NIFTy: linear during MGVI phase,
                        nonlinear update during GeoVI phase)
- `frozen_ranges`    : `Vector{UnitRange{Int}}` of latent indices to freeze
- `iter_callback`    : `(prob, z, samples, iter) -> nothing` after each KL step
- `map_maxiter`, `kl_maxiter`, `kl_absdelta`, `cg_maxiter`, `cg_tol`,
  `geo_newton_maxiter`, `geo_cg_maxiter`, `geo_tol`, `verb`

Returns `(z_center, samples)` where samples are antithetic pairs.
"""
function reconstruct_hybrid(prob::AbstractInferenceProblem;
                            z0::Union{Nothing, AbstractVector{<:Real}}=nothing,
                            n_mgvi::Int=10,
                            n_geovi::Int=6,
                            n_samples=iter -> iter <= 2 ? 2 : 4,
                            map_maxiter::Int=200,
                            kl_maxiter::Int=35,
                            kl_absdelta::Union{Nothing,Real}=0.5,
                            cg_maxiter::Int=100,
                            cg_tol::Real=0.01,
                            geo_newton_maxiter::Int=10,
                            geo_cg_maxiter::Int=50,
                            geo_tol::Real=1e-5,
                            frozen_ranges::Union{Nothing, Vector{UnitRange{Int}}}=nothing,
                            sample_mode=nothing,
                            iter_callback=nothing,
                            verb::Bool=true)
    n = latent_size(prob)
    n_total = n_mgvi + n_geovi

    has_frozen = frozen_ranges !== nothing && !isempty(frozen_ranges)
    function zero_frozen!(v)
        if has_frozen
            for r in frozen_ranges; v[r] .= 0.0; end
        end
        return v
    end
    function stiff_frozen!(v, u)
        if has_frozen
            for r in frozen_ranges; v[r] .= 1e6 .* u[r]; end
        end
        return v
    end
    n_frozen = has_frozen ? sum(length(r) for r in frozen_ranges) : 0

    if has_frozen && verb
        @printf("  Frozen latent ranges: %s (%d frozen, %d free)\n",
                join(["$(first(r)):$(last(r))" for r in frozen_ranges], ", "),
                n_frozen, n - n_frozen)
    end

    if sample_mode === nothing
        # iter == 1 can only reach the second branch when n_mgvi == 0 (GeoVI-only),
        # where :nonlinear_update has no previous samples to update; draw them
        # instead. The hybrid schedule (n_mgvi ≥ 1) is unchanged: its first GeoVI
        # iteration still updates the samples MGVI left behind, as NIFTy does.
        sample_mode = iter -> iter <= n_mgvi ? :linear_resample :
                              iter == 1      ? :nonlinear_resample :
                                               :nonlinear_update
    end

    if z0 === nothing
        verb && println("=== Hybrid: MAP warm-start ===")
        z = reconstruct_map(prob; maxiter=map_maxiter, verb=verb)
    else
        z = Vector{float(eltype(z0))}(z0)
    end
    zero_frozen!(z)
    T = eltype(z)

    samples = Vector{Vector{T}}()
    metric_samples = Vector{Vector{T}}()
    # White-noise pairs (w_data, w_latent) behind each sample, threaded across
    # iterations: :nonlinear_update keeps the white noise fixed and RECOMPUTES the
    # metric sample ms = L(z)ᵀ·w_data + w_latent at the new mean (L depends on the
    # mean), then re-refines: NIFTy's nonlinear_update, not a fresh draw.
    noise_pairs = Vector{Tuple{Vector{T},Vector{T}}}()

    for iter in 1:n_total
        mode = sample_mode(iter)
        ns = n_samples isa Function ? n_samples(iter) : n_samples
        # :nonlinear_update re-refines the existing pairs, so it uses however many
        # the last resample iteration drew; `ns` is inert there. Print the count
        # actually in use and flag the discrepancy rather than announcing `ns`.
        ns_reused = mode == :nonlinear_update && !isempty(noise_pairs)
        ns_eff = ns_reused ? length(noise_pairs) : ns
        verb && println("\n=== $(mode) iteration $iter/$n_total (n_samples=$ns_eff) ===")
        if verb && ns_reused && ns != ns_eff
            println("  Note: n_samples=$ns requested, but :nonlinear_update reuses the ",
                    "$ns_eff existing antithetic pair(s); use :nonlinear_resample ",
                    "for an iteration that should change the count.")
        end

        if mode == :linear_resample
            verb && println("  Drawing linear samples...")
            if has_frozen
                mat = v -> begin
                    mv = _posterior_metric_mul(prob, z, v)
                    stiff_frozen!(mv, v)
                    return mv
                end
                n_data = data_size(prob)
                lin_samples = Vector{Vector{T}}(undef, ns)
                noise_pairs = Vector{Tuple{Vector{T},Vector{T}}}(undef, ns)
                for k in 1:ns
                    w_data = randn(T, n_data)
                    w_latent = randn(T, n)
                    zero_frozen!(w_latent)
                    lh_sample = left_sqrt_metric(prob, z, w_data)
                    zero_frozen!(lh_sample)
                    ms = lh_sample .+ w_latent
                    delta, niter = _nifty_cg(mat, ms; x0=w_latent,
                                              maxiter=cg_maxiter, tol=cg_tol)
                    zero_frozen!(delta)
                    lin_samples[k] = delta
                    noise_pairs[k] = (w_data, w_latent)
                    verb && @printf("    CG sample %d: |δ|=%.3f (%d iters)\n", k, norm(delta), niter)
                end
            else
                lin_samples, _, noise_pairs = _draw_samples_geovi(prob, z, ns;
                                                      cg_maxiter=cg_maxiter,
                                                      cg_tol=cg_tol, verb=verb)
            end

            if verb
                norms = [norm(s) for s in lin_samples]
                @printf("  Linear sample norms: min=%.3f, max=%.3f, mean=%.3f\n",
                        minimum(norms), maximum(norms), sum(norms)/length(norms))
            end

            samples = Vector{Vector{T}}(undef, 2 * ns)
            for k in 1:ns
                samples[2k-1] = lin_samples[k]
                samples[2k] = -lin_samples[k]
            end

        elseif mode == :nonlinear_resample
            verb && println("  Drawing linear samples...")
            lin_samples, metric_samples, noise_pairs = _draw_samples_geovi(prob, z, ns;
                                                  cg_maxiter=cg_maxiter,
                                                  cg_tol=cg_tol, verb=verb)

            if verb
                norms = [norm(s) for s in lin_samples]
                @printf("  Linear sample norms: min=%.3f, max=%.3f, mean=%.3f\n",
                        minimum(norms), maximum(norms), sum(norms)/length(norms))
            end

            # Antithetic pair refined SEPARATELY against ±ms (NIFTy draw_residual):
            # +δ_lin→+ms, −δ_lin→−ms (same ms negated). Mirroring −δ would land off
            # a curved manifold; separate refinement tracks the curvature on both.
            verb && println("  Refining antithetic samples (Newton-CG)...")
            samples = Vector{Vector{T}}(undef, 2 * ns)
            for k in 1:ns
                verb && @printf("  Sample %d/%d (+):\n", k, ns)
                samples[2k-1] = _refine_sample_geovi(prob, z, lin_samples[k];
                    metric_sample=metric_samples[k], newton_maxiter=geo_newton_maxiter,
                    cg_maxiter=geo_cg_maxiter, tol=geo_tol, verb=verb)
                verb && @printf("  Sample %d/%d (−):\n", k, ns)
                samples[2k]   = _refine_sample_geovi(prob, z, -lin_samples[k];
                    metric_sample=-metric_samples[k], newton_maxiter=geo_newton_maxiter,
                    cg_maxiter=geo_cg_maxiter, tol=geo_tol, verb=verb)
                verb && @printf("    |δ_lin|=%.3f → |δ₊|=%.3f, |δ₋|=%.3f\n",
                                norm(lin_samples[k]), norm(samples[2k-1]), norm(samples[2k]))
            end

        elseif mode == :nonlinear_update
            if isempty(samples) || isempty(noise_pairs)
                error("sample_mode returned :nonlinear_update on iteration $iter with no samples " *
                      "to update. :nonlinear_update re-refines the samples that a previous " *
                      ":linear_resample / :nonlinear_resample iteration produced, so one of " *
                      "those must run first. Use :nonlinear_resample for this iteration (the " *
                      "default schedule does exactly that when n_mgvi == 0).")
            end

            verb && println("  Re-refining antithetic samples at the new mean (Newton-CG)...")
            n_base = length(noise_pairs)
            base_plus  = [samples[2k-1] for k in 1:n_base]
            base_minus = [samples[2k]   for k in 1:n_base]

            # NIFTy nonlinear_update: keep the white noise fixed, RECOMPUTE the
            # metric sample ms = L(z)ᵀ·w_data + w_latent at the NEW mean (L depends
            # on the mean), and re-refine both branches separately (+half→+ms,
            # −half→−ms) from their previous positions, not a mirror.
            samples = Vector{Vector{T}}(undef, 2 * n_base)
            for k in 1:n_base
                verb && @printf("  Sample %d/%d:\n", k, n_base)
                w_data, w_latent = noise_pairs[k]
                ms = left_sqrt_metric(prob, z, w_data) .+ w_latent
                has_frozen && zero_frozen!(ms)
                samples[2k-1] = _refine_sample_geovi(prob, z, base_plus[k];
                    metric_sample=ms, newton_maxiter=geo_newton_maxiter,
                    cg_maxiter=geo_cg_maxiter, tol=geo_tol, verb=verb)
                samples[2k]   = _refine_sample_geovi(prob, z, base_minus[k];
                    metric_sample=-ms, newton_maxiter=geo_newton_maxiter,
                    cg_maxiter=geo_cg_maxiter, tol=geo_tol, verb=verb)
                verb && @printf("    |δ₊|=%.3f, |δ₋|=%.3f\n",
                                norm(samples[2k-1]), norm(samples[2k]))
            end
        else
            error("Unknown sample_mode: $mode")
        end

        # KL optimization with Newton-CG
        if has_frozen
            z = _kl_newton_cg_frozen(prob, z, samples, frozen_ranges;
                                      maxiter=kl_maxiter,
                                      cg_maxiter=cg_maxiter, absdelta=kl_absdelta,
                                      verb=verb)
            zero_frozen!(z)
        else
            z = _kl_newton_cg(prob, z, samples;
                              maxiter=kl_maxiter, cg_maxiter=cg_maxiter,
                              absdelta=kl_absdelta, verb=verb)
        end

        verb && _report_latents(prob, z, samples)
        iter_callback === nothing || iter_callback(prob, z, samples, iter)
    end

    return z, samples
end


