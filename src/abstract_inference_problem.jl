# ============================================================================
# AbstractInferenceProblem protocol: abstract type + function stubs
# ============================================================================
#
# This file is included EARLY (before geovi.jl) so that algorithm helpers can
# type-annotate `prob::AbstractInferenceProblem` in their signatures. The
# concrete subtypes (InterferometricProblem, PointSourceProblem) and their
# method implementations live in src/inference_problem.jl, which is included
# LATE (after sky_model, observe, point_source, geovi, geovi_pointsource).
# ============================================================================

"""
    AbstractInferenceProblem

Abstract supertype for any inference problem that the generic VI machinery
can drive. The element type `T<:AbstractFloat` is chosen by the concrete problem
(e.g. from its data / config); every latent and model vector exchanged with the
solvers shares it, so the whole pipeline runs in Float32 or Float64 unchanged.
Subtypes must implement six methods (`T` = the problem's float type):

- `energy_and_gradient(prob, z) -> (T, Vector{T})`
- `latent_size(prob) -> Int`
- `transformation(prob, z) -> Vector{T}`         (whitened model T(z))
- `right_sqrt_metric(prob, z, v) -> Vector{T}`   (J_T(z) · v)
- `left_sqrt_metric(prob, z, v) -> Vector{T}`    (J_T(z)' · v)
- `data_size(prob) -> Int`

The `reconstruct_*` drivers infer `T` from the supplied `z0` (defaulting to
Float64); pass a `Float32` `z0` to run a problem end-to-end in single precision.
"""
abstract type AbstractInferenceProblem end

# Generic function declarations; subtypes attach methods.
function energy_and_gradient end
function latent_size end
function transformation end
function right_sqrt_metric end
function left_sqrt_metric end
function data_size end

"""
    report_latents(prob, z, samples) -> nothing

Optional per-iteration hook for problem-specific latent / hyperparameter
reporting (the analogue of NIFTy's per-domain `minisanity`). The iterative
`reconstruct_*` drivers call it once per iteration when `verb=true`, after
printing the generic latent-sample diagnostic. `z` is the current variational
mean and `samples` are the (antithetic, mean-zero) posterior samples.

The default does nothing; a problem type implements a method to decode and
print its own structured latents, e.g. correlated-field slope / fluctuation /
offset. Keep it to a few lines of `@printf`; it runs every iteration. It is
printed after the generic minisanity table.
"""
report_latents(::AbstractInferenceProblem, ::AbstractVector, ::Any) = nothing

"""
    latent_blocks(prob) -> Vector{Tuple{String, AbstractVector{Int}}}

Optional: name the sub-domains of the latent vector so the minisanity table
(printed by the `reconstruct_*` drivers under `verb=true`) can report the
standardized-latent reduced χ² (= ⟨ξ²⟩, target 1) and mean (target 0)
*per block*, NIFTy's per-domain latent diagnostic. The default is a single
block `"ξ"` spanning the whole latent.

A block index set may be any `AbstractVector{Int}`, not just a `UnitRange`: only
`length` and `view` are used. An explicit index vector lets you declare blocks
that are **not contiguous** in the latent layout, which is the only way to group
by a property the layout does not follow — e.g. splitting a surface map into
"well seen" and "barely seen" pixels, which is geometric rather than positional:

    latent_blocks(p::MyProblem) =
        [("well_seen", p.visible_idx), ("barely_seen", p.limb_idx)]

Correlated-field problems override this to expose each field/hyperparameter
(e.g. the excitation field, slope, fluct, offset) as its own row, so a row with
reduced χ² ≫ 1 flags a latent the data is pulling away from its N(0,1) prior,
and (for a 1-dof hyperparameter) the sign of `mean` shows which way.
"""
latent_blocks(prob::AbstractInferenceProblem) = [("ξ", 1:latent_size(prob))]

"""
    whitened_data(prob) -> Vector{Float64} | nothing

Optional: the whitened data `d/σ`, so the minisanity "data residual" row can be
computed from the normalized residual `transformation(prob, z) - whitened_data`
(NIFTy-style: reduced χ² = ⟨residual²⟩ → 1, mean → 0). The default `nothing`
makes the table fall back to the energy-identity reduced χ² (no mean column).
"""
whitened_data(::AbstractInferenceProblem) = nothing

# ============================================================================
# Optional non-unit prior covariance (centred / hierarchical parameterizations)
# ============================================================================
#
# VarInf's default contract is a standard-normal latent, prior covariance C = I.
# The three hooks below let a problem declare a different `C(z)` so that the
# NIFTy-style construction `amp = f(θ), y = amp ⊙ ξ` can be written *centred*
# (the learned spectrum in the prior) instead of as a product in the forward
# map, where it is Neal's funnel and MGVI's `JᵀJ + I` cannot see the curvature
# the funnel puts in the prior term.
#
# `C` may depend on `z`, since the hyper-latents that set it live in the same
# latent vector. All three default to the identity, so a problem that does not
# implement them behaves exactly as before.
#
# The two `_mul` hooks are always called ONCE on a full-length latent vector,
# never per block, so `C` need not be homogeneous: a MIXED parameterization is
# the expected case. A problem whose hyper-latents stay non-centred while only
# its field block is centred returns `v` unchanged on the hyper indices and the
# scaled values on the field indices, from that single call — e.g.
#
#     function VarInf.prior_inv_covariance_mul(p::MyProblem, z, v)
#         out = copy(v)                       # hyper-latents keep C = I
#         out[p.field_range] ./= amp(p, z).^2 # field block: C = diag(amp(z)²)
#         return out
#     end
#
# Division of labour: the PROBLEM owns the energy (`energy_and_gradient` must
# return the prior term matching its own `C`), VarInf owns the metric.
#
# Two things that are easy to get wrong, because both yield a plausible fit
# rather than an error:
#
#  * The centred energy is `0.5‖θ‖² + 0.5·yᵀC(θ)⁻¹y + 0.5·logdet C(θ)`. Omit the
#    log-det and nothing penalises `C → ∞`: the quadratic term vanishes and the
#    hyperparameters run away.
#  * `JᵀJ + C⁻¹` is not the full Fisher metric when `C` depends on `θ` — it drops
#    the prior's own Fisher block, `½·tr(C⁻¹C,ᵢC⁻¹C,ⱼ)`. This is the same
#    structure NIFTy uses and is expected to be adequate, but it is an
#    approximation, not an identity.

"""
    prior_inv_covariance_mul(prob, z, v) -> C(z)⁻¹·v

Optional: apply the inverse prior covariance. Used to build the posterior metric
`M = JᵀJ + C⁻¹`. Matrix-free; `C` may depend on `z`. Defaults to `v` (`C = I`).

Must be symmetric positive-definite and consistent with
[`prior_inv_sqrt_covariance_mul`]: `C^{-1/2}·C^{-1/2} = C⁻¹`.
"""
prior_inv_covariance_mul(::AbstractInferenceProblem, ::AbstractVector{<:Real},
                         v::AbstractVector{<:Real}) = v

"""
    prior_inv_sqrt_covariance_mul(prob, z, v) -> C(z)^{-1/2}·v

Optional: apply the inverse prior square root, i.e. whiten `v`. Used to draw the
metric sample `ms = Jᵀw_data + C^{-1/2}w_latent`, to form GeoVI's prior-whitened
coordinate displacement `C^{-1/2}(x - e)`, and to whiten the per-block minisanity
statistic so its target stays 1. Matrix-free; `C` may depend on `z`. Defaults to
`v` (`C = I`), and must be the symmetric square root consistent with
[`prior_inv_covariance_mul`].

Called once per use on a full-length latent vector, so `C` may differ between
index ranges (see the note above on mixed parameterizations). The minisanity
per-block split is exact only when `C` is block-diagonal with respect to the
blocks [`latent_blocks`] declares, which the split already assumes.
"""
prior_inv_sqrt_covariance_mul(::AbstractInferenceProblem, ::AbstractVector{<:Real},
                              v::AbstractVector{<:Real}) = v

"""
    prior_energy(prob, z) -> Real

Optional: the prior term of the energy, `0.5·zᵀC(z)⁻¹z` plus any normalization
(`0.5·logdet C(z)` when `C` depends on `z`). Defaults to `0.5·dot(z, z)`.

VarInf uses this only to subtract the prior contribution back out of
`energy_and_gradient` when forming the minisanity data-residual χ² for a problem
that does not implement [`whitened_data`]. It must agree with the prior term
`energy_and_gradient` actually adds, or that row will be wrong.
"""
prior_energy(::AbstractInferenceProblem, z::AbstractVector{<:Real}) = 0.5 * sum(abs2, z)
