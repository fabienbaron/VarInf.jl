# ============================================================================
# PhaseRetrievalLatentProblem: like PhaseRetrievalProblem (known object,
# unknown atmospheric phase screen) but the phase is a *learnable*
# correlated field: its amplitude-spectrum slope and fluctuation are
# hyperparameters inferred from the data, not a fixed Kolmogorov kernel.
#
# This is the variant that exercises the problem-specific `report_latents`
# hook; the fixed-kernel PhaseRetrievalProblem has no CF hyperparameters,
# so it only shows the generic latent diagnostics.
#
# Forward chain:
#     z ──► phase = CorrelatedField(z)        (slope/fluct in z's hyper-block)
#           U     = A · exp(i·phase)
#           PSF   = |fft(U)|²,  PSF_norm = PSF/∑PSF
#           obs   = real(ifft(fft(saturn) · fft(PSF_norm)))   (= saturn ⊛ PSF)
#     data | z ~ N(obs, σ²)
#
# The object (Saturn) is known. Piston (the phase DC mode) is unidentifiable
# (a constant phase leaves |fft(U)|² unchanged), so the phase CF uses
# `offset_prior=nothing`; that DC mode is a harmless flat direction held by
# its N(0,1) prior. Tip/tilt is identifiable here (it shifts the PSF relative
# to the known object), so it is left in the kernel.
#
# Latent layout: z = the CorrelatedField latent =
#     [xi_field (n²) ; xi_slope ; xi_fluct]    (use_iwp=false, use_offset=false)
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Statistics: mean, std
using Printf

struct PhaseRetrievalLatentProblem{T<:AbstractFloat,CF<:CorrelatedField{T},TP,TIP} <: AbstractInferenceProblem
    grid::FourierGridInfo{T}
    aperture::Matrix{T}            # n × n real aperture mask
    saturn::Matrix{T}              # known object on n × n grid
    saturn_fft::Matrix{Complex{T}}       # cached fft(saturn)
    cf_phase::CF                         # learnable CF prior on the phase
    data::Matrix{T}                # observed image
    sigma_noise::T
    # Cached out-of-place plans + scratch for the PSF/convolution FFTs; transient
    # transforms go through mul!(scratch, plan, ·) (single-threaded ⇒ safe). The
    # phase CF carries its own plans/buffers for the phase-generation FFTs.
    P::TP
    iP::TIP
    cs1::Matrix{Complex{T}}
    cs2::Matrix{Complex{T}}
end

Base.eltype(::PhaseRetrievalLatentProblem{T}) where {T} = T

VarInf.latent_size(p::PhaseRetrievalLatentProblem) = VarInf.latent_size(p.cf_phase)
VarInf.data_size(p::PhaseRetrievalLatentProblem)   = p.grid.npix^2

# Forward chain (returns intermediates for JVP/VJP reuse)

struct _PRLFwd{T<:AbstractFloat}
    phase::Matrix{T}
    U::Matrix{Complex{T}}
    F::Matrix{Complex{T}}
    PSF::Matrix{T}
    S::T
    PSF_norm::Matrix{T}
    data_clean::Matrix{T}
end

function _forward(z::AbstractVector{<:Real}, p::PhaseRetrievalLatentProblem)
    phase = p.cf_phase(z)   # phase-generation FFTs use CF buffers
    U     = p.aperture .* exp.(im .* phase)
    F     = p.P * U                          # persists (JVP/VJP use it) ⇒ fresh
    PSF   = abs2.(F)
    S     = sum(PSF)
    PSF_norm = PSF ./ S
    # data_clean = real(ifft(saturn_fft .* fft(PSF_norm))); transient via scratch
    p.cs1 .= PSF_norm
    mul!(p.cs2, p.P, p.cs1)
    p.cs2 .*= p.saturn_fft
    mul!(p.cs1, p.iP, p.cs2)
    data_clean = real.(p.cs1)
    return _PRLFwd(phase, U, F, PSF, S, PSF_norm, data_clean)
end

VarInf.transformation(p::PhaseRetrievalLatentProblem, z::AbstractVector{<:Real}) =
    vec(_forward(z, p).data_clean) ./ p.sigma_noise

# JVP

function VarInf.right_sqrt_metric(p::PhaseRetrievalLatentProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    fwd = _forward(z, p)
    # 1. v → d_phase through the correlated-field operator (field + hypers)
    d_phase = VarInf.mcf_jvp(p.cf_phase, z, v)
    # 2-6. phase → data sensitivity (identical to the fixed-kernel problem)
    dU   = im .* fwd.U .* d_phase
    dF   = p.P * dU                          # persists into step 4 ⇒ fresh
    dPSF = 2 .* real.(conj.(fwd.F) .* dF)
    dPSF_norm = (dPSF .- fwd.PSF_norm .* sum(dPSF)) ./ fwd.S
    # d_data_clean = real(ifft(saturn_fft .* fft(dPSF_norm))); transient via scratch
    p.cs1 .= dPSF_norm
    mul!(p.cs2, p.P, p.cs1)
    p.cs2 .*= p.saturn_fft
    mul!(p.cs1, p.iP, p.cs2)
    d_data_clean = real.(p.cs1)
    return vec(d_data_clean) ./ p.sigma_noise
end

# VJP

function VarInf.left_sqrt_metric(p::PhaseRetrievalLatentProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    fwd = _forward(z, p)
    n = p.grid.npix
    N = n * n
    w_2d = reshape(w ./ p.sigma_noise, n, n)
    # 1. data_clean ← w  →  g_PSF_norm (adjoint of conv with saturn); via scratch
    p.cs1 .= w_2d
    mul!(p.cs2, p.P, p.cs1)
    @. p.cs2 *= conj(p.saturn_fft)           # fused conj ⇒ no temporary
    mul!(p.cs1, p.iP, p.cs2)
    g_PSF_norm = real.(p.cs1)
    # 2. g_PSF_norm → g_PSF (adjoint of PSF/S normalization)
    inner_sum = dot(g_PSF_norm, fwd.PSF_norm)
    g_PSF = (g_PSF_norm .- inner_sum) ./ fwd.S
    # 3. g_PSF → g_F = 2 · g_PSF · F
    g_F = 2 .* g_PSF .* fwd.F
    # 4. g_F → g_U = N · ifft(g_F); g_U persists ⇒ fresh
    g_U = N .* (p.iP * g_F)
    # 5. g_U → g_phase
    g_phase = imag.(conj.(fwd.U) .* g_U)
    # 6. g_phase → g_z through the correlated-field adjoint
    return VarInf.mcf_vjp(p.cf_phase, z, g_phase)
end

function VarInf.energy_and_gradient(p::PhaseRetrievalLatentProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = vec(p.data) ./ p.sigma_noise
    resid = T_z .- d_white
    chi2  = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end

# Helpers

"""
    circular_aperture_latent(npix; radius_frac=0.45)

Circular binary aperture mask centered on the array.
"""
function circular_aperture_latent(npix::Int; radius_frac::Float64=0.45)
    r2 = (radius_frac * npix)^2
    mask = zeros(npix, npix)
    cx = (npix + 1) / 2; cy = (npix + 1) / 2
    @inbounds for j in 1:npix, i in 1:npix
        if (i - cx)^2 + (j - cy)^2 <= r2
            mask[i, j] = 1.0
        end
    end
    return mask
end

"""
    asymmetric_aperture_latent(npix; radius_frac=0.35, offset_frac=0.03,
                               notch_radius_frac=0.13, notch_dist_frac=0.30,
                               notch_dir=(-0.6, -0.8))

A **non-centrosymmetric** pupil for the learnable-CF demos: a near-centred
circle with a circular obscuration punched near the centre but off to one side
(like an off-axis secondary). Because the mask is not invariant under
reflection through the array centre, the twin-image ambiguity `φ → −φ_flipped`
(under which `|fft(A·exp(iφ))|²` is invariant for a *centrosymmetric* `A`) is
broken, so cold-start phase retrieval converges reliably to the true phase
instead of landing in the twin basin at random. Mirrors `asymmetric_aperture`
in `phase_retrieval_problem.jl`.
"""
function asymmetric_aperture_latent(npix::Int; radius_frac::Float64=0.35,
                                     offset_frac::Float64=0.03,
                                     notch_radius_frac::Float64=0.13,
                                     notch_dist_frac::Float64=0.30,
                                     notch_dir::NTuple{2,Float64}=(-0.6, -0.8))
    R  = radius_frac * npix
    cx = (npix + 1) / 2 + offset_frac * npix
    cy = (npix + 1) / 2 - offset_frac * npix
    nd = notch_dir ./ sqrt(notch_dir[1]^2 + notch_dir[2]^2)
    nrx = cx + nd[1] * notch_dist_frac * R   # obscuration near centre, off to one side
    nry = cy + nd[2] * notch_dist_frac * R
    nr2 = (notch_radius_frac * npix)^2
    mask = zeros(npix, npix)
    @inbounds for j in 1:npix, i in 1:npix
        in_disk  = (i - cx)^2 + (j - cy)^2 <= R^2
        in_notch = (i - nrx)^2 + (j - nry)^2 <= nr2
        mask[i, j] = (in_disk && !in_notch) ? 1.0 : 0.0
    end
    return mask
end

phase_from_latent(z::AbstractVector{<:Real}, p::PhaseRetrievalLatentProblem) =
    p.cf_phase(z)

# Decode the (slope, fluct) hyperparameters from a latent.
function phase_hypers(z::AbstractVector{<:Real}, p::PhaseRetrievalLatentProblem)
    cfg = p.cf_phase.cfgs[1]
    _, axes, _ = VarInf.latent_unpack(p.cf_phase, z)
    a = axes[1]
    slope = cfg.slope_mean + cfg.slope_std * a.xi_slope
    fluct = exp(cfg.fluct_mean + cfg.fluct_std * a.xi_fluct)
    return (slope, fluct)
end

"""
    make_phase_cf(npix; slope_prior, fluct_prior) -> CorrelatedField

Build the learnable phase CF: a single 2-D Fourier axis, no IWP, no offset
(piston is degenerate). `slope_prior`/`fluct_prior` are `(mean, std)` tuples;
nonzero std makes the hyperparameter learnable.
"""
function make_phase_cf(npix::Int; slope_prior=(-11/6, 0.4),
                        fluct_prior=(0.4, 0.12))
    cfg  = CorrFieldConfig(slope_prior=slope_prior, fluct_prior=fluct_prior)
    grid = FourierGridInfo(npix, one(eltype(cfg)))
    return CorrelatedField([grid], [cfg]; offset_prior=nothing)
end

"""
    generate_synthetic_phase_retrieval_latent(saturn, cf_inference; aperture,
        true_slope=-11/6, true_fluct=0.4, noise_frac=0.02, seed=42)
        -> (prob, phase_true, psf_true, data_clean, true_slope, true_fluct)

The TRUE phase screen is a fixed Kolmogorov realisation (slope and fluct
pinned at `true_slope`/`true_fluct`), generated *independently* of the
inference prior `cf_inference`. So the learnable-slope and slope-pinned demos,
called with the same `seed`, see IDENTICAL data and differ only in the
inference prior. The problem is then assembled with `cf_inference` (the prior
under which we recover the phase).
"""
function generate_synthetic_phase_retrieval_latent(saturn::AbstractMatrix{<:Real},
        cf_inference::CorrelatedField;
        aperture::AbstractMatrix{<:Real}=circular_aperture_latent(size(saturn, 1)),
        true_slope::Real=-11/6, true_fluct::Real=0.4,
        noise_frac::Real=0.02, seed::Int=42)
    T = eltype(cf_inference)
    Random.seed!(seed)
    n = size(saturn, 1)
    @assert size(aperture) == (n, n)
    @assert cf_inference.field_shape == (n, n)
    saturn = Matrix{T}(saturn); aperture = Matrix{T}(aperture)

    # Truth: a fixed Kolmogorov phase screen (slope + fluct pinned), so both
    # demos generate the same phase_true / data for a given seed.
    cf_truth = make_phase_cf(n; slope_prior=(true_slope, 0.0),
                                fluct_prior=(true_fluct, 0.0))
    phase_true = cf_truth(randn(VarInf.latent_size(cf_truth)))

    U_true = aperture .* exp.(im .* phase_true)
    PSF_true = abs2.(fft(U_true)); PSF_true ./= sum(PSF_true)
    data_clean = real.(ifft(fft(saturn) .* fft(PSF_true)))
    sigma_n = noise_frac * maximum(data_clean)
    data = data_clean .+ sigma_n .* randn(n, n)

    buf = zeros(Complex{T}, n, n)
    P   = plan_fft(buf;  flags=FFTW.MEASURE)
    iP  = plan_ifft(buf; flags=FFTW.MEASURE)
    prob = PhaseRetrievalLatentProblem(FourierGridInfo(n, one(T)), aperture, saturn,
                                        fft(saturn), cf_inference, Matrix{T}(data), T(sigma_n),
                                        P, iP, zeros(Complex{T}, n, n), zeros(Complex{T}, n, n))
    return prob, phase_true, PSF_true, data_clean, true_slope, true_fluct
end

# Minisanity hooks: latent blocks + whitened data, and decoded hypers

function VarInf.latent_blocks(prob::PhaseRetrievalLatentProblem)
    n2 = prob.grid.npix^2
    return [("ξ phase field", 1:n2), ("ξ slope", n2+1:n2+1), ("ξ fluct", n2+2:n2+2)]
end

VarInf.whitened_data(prob::PhaseRetrievalLatentProblem) =
    vec(prob.data) ./ prob.sigma_noise

function VarInf.report_latents(prob::PhaseRetrievalLatentProblem,
                               z::AbstractVector{<:Real}, samples)
    cfg = prob.cf_phase.cfgs[1]
    draws = [phase_hypers(z .+ s, prob) for s in samples]
    sl = [d[1] for d in draws]; fl = [d[2] for d in draws]
    pin = cfg.slope_std == 0 ? "(pinned, Kolmogorov)" : "(Kolmogorov −1.833)"
    report_row("phase CF slope", (@sprintf("%+.3f ± %.3f", mean(sl), std(sl))), pin)
    report_row("phase CF fluct", (@sprintf("%+.4f ± %.4f", mean(fl), std(fl))), "(≈ phase RMS, rad)")
    return nothing
end
