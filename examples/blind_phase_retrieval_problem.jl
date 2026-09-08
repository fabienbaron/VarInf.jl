# ============================================================================
# BlindPhaseRetrievalProblem: joint recovery of an atmospheric phase screen
# AND the underlying object, from a single noisy aberrated image.
#
# Extension of PhaseRetrievalProblem: instead of taking the object as known,
# we put a 2-D log-normal correlated-field prior on it and infer both
# (phase, log-object) jointly.
#
# Forward chain:
#     z_phase  ──► phase  = real(ifft(fft(z_phase) .* K_phase))
#     z_object ──► log_x  = CorrelatedField(z_object)
#                  x      = exp(log_x)               (positive object)
#                  U      = A · exp(i · phase)
#                  F      = fft(U); PSF = |F|²; PSF_norm = PSF / sum(PSF)
#                  obs    = real(ifft(fft(x) .* fft(PSF_norm)))   (= x ⊛ PSF)
#     data | z ~ N(obs, σ² · I)
#
# Why this version of "blind deconvolution" is well-conditioned:
#   * PSF support is bounded below by the aperture (diffraction-limited),
#     so the "δ-PSF + object=data" degeneracy is physically forbidden.
#   * PSF is auto-normalized to ∑=1 inside the forward, killing the
#     multiplicative scale ambiguity.
#   * Tip/tilt modes of the phase are zeroed in K_phase (alongside the
#     usual piston-free DC=0), which breaks the (object-shift, phase-tilt)
#     translation ambiguity.
#
# Latent layout: z = [z_phase; z_object], with
#     z_phase  ∈ R^{n²}             (real-space pre-image of the phase CF)
#     z_object ∈ R^{cf.latent_size}  (CorrelatedField latent: field+hypers+offset)
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Statistics: mean, std
using Printf

struct BlindPhaseRetrievalProblem{T<:AbstractFloat,CF<:CorrelatedField{T}} <: AbstractInferenceProblem
    grid::FourierGridInfo{T}
    aperture::Matrix{T}             # n × n real aperture mask
    phase_shape::Matrix{T}          # UNIT-strength Kolmogorov amp (DC + tip/tilt = 0)
    phase_fluct_mean::T             # log-strength prior μ: strength = exp(μ + σ·ξ_str)
    phase_fluct_std::T              # log-strength prior σ (0 ⇒ strength pinned)
    cf_object::CF                         # 2-D log-normal CF prior on the object
                                          # (its additive offset carries the level;
                                          #  concrete CF type ⇒ p.cf_object stable)
    data::Matrix{T}                 # observed image
    sigma_noise::T
    n_white::Int                          # phase white-field size (n²); ξ_strength at n_white+1
    n_object::Int
end

Base.eltype(::BlindPhaseRetrievalProblem{T}) where {T} = T

"""
    _kolmogorov_shape_blind(grid; zero_tip_tilt=true)

UNIT-strength Kolmogorov amplitude shape (K ∝ |k|^(-11/6), DC=0, tip/tilt=0,
normalized so a unit-strength field has RMS≈1). The TURBULENCE STRENGTH is now a
learnable scalar that multiplies this shape (the phase is the slope-pinned,
no-IWP, no-offset CorrelatedField specialized to keep the exact tip/tilt
projection). Tip/tilt (bin 2, the lowest non-zero |k|) is zeroed so a phase tilt
cannot absorb an object centroid shift (translation degeneracy).
"""
function _kolmogorov_shape_blind(grid::FourierGridInfo; zero_tip_tilt::Bool=true)
    T = eltype(grid)
    k_mat = grid.mode_lengths[grid.bin_index]
    K = similar(k_mat)
    ex = T(-11//6)
    @inbounds for i in eachindex(k_mat)
        K[i] = k_mat[i] == 0 ? zero(T) : k_mat[i]^ex
    end
    if zero_tip_tilt
        @inbounds for i in eachindex(K)
            grid.bin_index[i] == 2 && (K[i] = zero(T))
        end
    end
    K .*= grid.npix / sqrt(sum(abs2, K))   # unit strength ⇒ RMS ≈ 1
    return K
end

"""
    BlindPhaseRetrievalProblem(aperture, cf_object, data;
                                sigma_noise, phase_fluct_prior=(0.4, 0.3), dx=1.0)

Builds the unit Kolmogorov shape (DC + tip/tilt zeroed) and a LEARNABLE
turbulence strength with a value-space lognormal prior `phase_fluct_prior =
(mean, std)` (slope stays fixed at Kolmogorov −11/6; only the strength is
inferred: "we know the slope, not the strength"). Latent layout:
`[ξ_phase_white (n²) ; ξ_phase_strength (1) ; ξ_object (cf_object.latent_size)]`.

The object's overall level is carried by `cf_object`'s additive offset; build it
with `offset_prior=(log(mean(data)), σ)`.
"""
function BlindPhaseRetrievalProblem(aperture::AbstractMatrix{<:Real},
                                     cf_object::CorrelatedField,
                                     data::AbstractMatrix{<:Real};
                                     sigma_noise::Real,
                                     phase_fluct_prior::Tuple{Real,Real}=(0.4, 0.3),
                                     dx::Real=1.0)
    n = size(aperture, 1)
    @assert size(aperture) == size(data) == (n, n)
    @assert cf_object.field_shape == (n, n) "cf_object must produce an n×n field"
    T = eltype(cf_object)
    aperture = Matrix{T}(aperture); data = Matrix{T}(data)
    grid = FourierGridInfo(n, T(dx))
    shape = _kolmogorov_shape_blind(grid)
    μφ, σφ = VarInf._lognormal_params(T(phase_fluct_prior[1]), T(phase_fluct_prior[2]))
    n_white  = n^2
    n_object = VarInf.latent_size(cf_object)
    return BlindPhaseRetrievalProblem(grid, aperture, shape, μφ, σφ, cf_object,
                                       data, T(sigma_noise), n_white, n_object)
end

VarInf.latent_size(p::BlindPhaseRetrievalProblem) = p.n_white + 1 + p.n_object
VarInf.data_size(p::BlindPhaseRetrievalProblem)   = p.grid.npix^2

# Forward chain

struct _BPRFwd{T<:AbstractFloat}
    phase::Matrix{T}
    phase_strength::T               # turbulence strength = exp(μφ + σφ·ξ_str)
    phase_base::Matrix{T}           # unit-strength filtered field (phase = strength·base)
    log_object::Matrix{T}
    object::Matrix{T}
    object_fft::Matrix{Complex{T}}
    U::Matrix{Complex{T}}
    F::Matrix{Complex{T}}
    PSF::Matrix{T}
    S::T
    PSF_norm::Matrix{T}
    PSF_norm_fft::Matrix{Complex{T}}
    data_clean::Matrix{T}
end

# Latent: [ξ_white (n²) ; ξ_strength (1) ; ξ_object (n_object)].
function _split_z(p::BlindPhaseRetrievalProblem, z::AbstractVector{<:Real})
    nw = p.n_white
    return (view(z, 1:nw), z[nw+1], view(z, nw+2:nw+1+p.n_object))
end

function _forward(z::AbstractVector{<:Real}, p::BlindPhaseRetrievalProblem)
    n = p.grid.npix
    z_white, xi_str, z_object = _split_z(p, z)
    # Phase branch: strength·(Kolmogorov-filtered white field). Slope fixed
    # (phase_shape), strength learnable.
    strength = exp(p.phase_fluct_mean + p.phase_fluct_std * xi_str)
    base  = real.(ifft(fft(reshape(collect(z_white), n, n)) .* p.phase_shape))
    phase = strength .* base
    # Object branch (CF, which includes the additive offset, then exp)
    log_object = p.cf_object(collect(z_object))
    object = exp.(log_object)
    object_fft = fft(object)
    # Pupil → PSF
    U = p.aperture .* exp.(im .* phase)
    F = fft(U)
    PSF = abs2.(F)
    S = sum(PSF)
    PSF_norm = PSF ./ S
    PSF_norm_fft = fft(PSF_norm)
    # Convolution: x ⊛ PSF_norm
    data_clean = real.(ifft(object_fft .* PSF_norm_fft))
    return _BPRFwd(phase, strength, base, log_object, object, object_fft, U, F, PSF, S,
                   PSF_norm, PSF_norm_fft, data_clean)
end

VarInf.transformation(p::BlindPhaseRetrievalProblem, z::AbstractVector{<:Real}) =
    vec(_forward(z, p).data_clean) ./ p.sigma_noise

# JVP

function VarInf.right_sqrt_metric(p::BlindPhaseRetrievalProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    fwd = _forward(z, p)
    n = p.grid.npix
    v_white, v_str, v_object = _split_z(p, v)
    z_object_vec = view(z, p.n_white+2:p.n_white+1+p.n_object)
    # 1. Phase tangent: d(strength·base) = (strength·σφ·v_str)·base + strength·d_base
    d_base = real.(ifft(fft(reshape(collect(v_white), n, n)) .* p.phase_shape))
    d_phase = (fwd.phase_strength * p.phase_fluct_std * v_str) .* fwd.phase_base .+
              fwd.phase_strength .* d_base
    # 2. Object tangent (log-space, then through exp)
    d_log_object = VarInf.mcf_jvp(p.cf_object, z_object_vec, collect(v_object))
    d_object = fwd.object .* d_log_object       # chain through exp
    # 3. Phase → PSF_norm sensitivity (same as known-object problem)
    dU = im .* fwd.U .* d_phase
    dF = fft(dU)
    dPSF = 2 .* real.(conj.(fwd.F) .* dF)
    dPSF_norm = (dPSF .- fwd.PSF_norm .* sum(dPSF)) ./ fwd.S
    # 4. Combine at the convolution: d(x ⊛ k) = (dx ⊛ k) + (x ⊛ dk)
    d_data_clean = real.(ifft(fft(d_object) .* fwd.PSF_norm_fft .+
                              fwd.object_fft .* fft(dPSF_norm)))
    return vec(d_data_clean) ./ p.sigma_noise
end

# VJP

function VarInf.left_sqrt_metric(p::BlindPhaseRetrievalProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    fwd = _forward(z, p)
    n = p.grid.npix
    N = n * n
    w_2d = reshape(w ./ p.sigma_noise, n, n)
    F_w = fft(w_2d)
    # Adjoint of `data_clean = x ⊛ PSF_norm`:
    #   g_object   = w ⋆ PSF_norm  = ifft(fft(w) · conj(fft(PSF_norm)))
    #   g_PSF_norm = w ⋆ object    = ifft(fft(w) · conj(fft(object)))
    g_object_img = real.(ifft(F_w .* conj.(fwd.PSF_norm_fft)))
    g_PSF_norm   = real.(ifft(F_w .* conj.(fwd.object_fft)))

    # Object branch back to z_object
    # Adjoint of exp: g_log_object = object · g_object
    g_log_object = fwd.object .* g_object_img
    z_object_vec = view(z, p.n_white+2:p.n_white+1+p.n_object)
    g_z_object = VarInf.mcf_vjp(p.cf_object, z_object_vec, g_log_object)

    # Phase branch back to ξ_white and ξ_strength
    inner_sum = dot(g_PSF_norm, fwd.PSF_norm)
    g_PSF = (g_PSF_norm .- inner_sum) ./ fwd.S
    g_F   = 2 .* g_PSF .* fwd.F
    g_U   = N .* ifft(g_F)
    g_phase = imag.(conj.(fwd.U) .* g_U)
    # phase = strength·base ⇒ g_white = strength·(shape-filter adjoint of g_phase),
    # g_ξ_strength = σφ·⟨g_phase, phase⟩ (since ∂phase/∂ξ_str = strength·σφ·base).
    g_white_2d   = fwd.phase_strength .* real.(ifft(fft(g_phase) .* p.phase_shape))
    g_xi_strength = p.phase_fluct_std * dot(g_phase, fwd.phase)

    return vcat(vec(g_white_2d), g_xi_strength, g_z_object)
end

function VarInf.energy_and_gradient(p::BlindPhaseRetrievalProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = vec(p.data) ./ p.sigma_noise
    resid = T_z .- d_white
    chi2  = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end

# Convenience: extract phase/object from a latent

function phase_from_latent(z::AbstractVector{<:Real}, p::BlindPhaseRetrievalProblem)
    n = p.grid.npix
    z_white, xi_str, _ = _split_z(p, z)
    strength = exp(p.phase_fluct_mean + p.phase_fluct_std * xi_str)
    return strength .* real.(ifft(fft(reshape(collect(z_white), n, n)) .* p.phase_shape))
end

# Recovered turbulence strength from a latent (the learned Kolmogorov amplitude).
phase_strength_from_latent(z::AbstractVector{<:Real}, p::BlindPhaseRetrievalProblem) =
    exp(p.phase_fluct_mean + p.phase_fluct_std * z[p.n_white+1])

function object_from_latent(z::AbstractVector{<:Real}, p::BlindPhaseRetrievalProblem)
    _, _, z_object = _split_z(p, z)
    return exp.(p.cf_object(collect(z_object)))
end

# Synthetic data generator

"""
    mask_to_aperture(x, aperture) -> Matrix{Float64}

Return `x` with all out-of-aperture pixels set to `NaN`, so heatmaps display
only the meaningful (in-aperture) region of a phase quantity. The phase is
undefined/unconstrained outside the pupil, so showing it there is misleading.
"""
mask_to_aperture(x::AbstractMatrix, aperture::AbstractMatrix) =
    ifelse.(aperture .> 0.5, Float64.(x), NaN)

"""
    diffraction_cutoff_k(radius_frac; dx=1.0) -> Float64

Image-plane frequency (in `mode_lengths` / fftfreq units, cycles per pixel
for dx=1) above which the optical transfer function, the autocorrelation of
a pupil of radius `radius_frac·npix`, carries no power. Equals
`2·radius_frac/dx`. Object frequencies beyond this are absent from the data
and can only be supplied by the prior.
"""
diffraction_cutoff_k(radius_frac::Float64; dx::Float64=1.0) = 2 * radius_frac / dx

"""
    object_error_split(obj_est, obj_true, k_cut; dx=1.0) -> (err_below, err_above)

Relative L2 error of `obj_est` vs `obj_true` in the Fourier domain, split at
the diffraction cutoff `k_cut`. `err_below` measures the data-constrained
band; `err_above` measures the band that only the prior can fill, which is
where pinning the object's spectral slope is expected to matter.
"""
function object_error_split(obj_est::Matrix{Float64}, obj_true::Matrix{Float64},
                             k_cut::Float64; dx::Float64=1.0)
    n = size(obj_true, 1)
    kx = FFTW.fftfreq(n, 1.0 / dx)
    kr = [sqrt(kx[i]^2 + kx[j]^2) for i in 1:n, j in 1:n]
    F_est  = fft(obj_est)
    F_true = fft(obj_true)
    below = kr .<= k_cut
    above = .!below
    err_below = norm((F_est .- F_true)[below]) / norm(F_true[below])
    err_above = norm((F_est .- F_true)[above]) / norm(F_true[above])
    return err_below, err_above
end

"""
    circular_aperture_blind(npix; radius_frac=0.15)

Circular binary aperture. The default `radius_frac=0.15` gives a pupil
diameter of `0.30·npix`, i.e. an OTF cutoff at `0.6·Nyquist`, a genuine
diffraction limit well inside the band, so the object's high-frequency
content must come from the prior (the regime where a constrained spectral
slope helps). Larger fractions (→0.45) make the PSF a near-delta and the
deblurring trivial.
"""
function circular_aperture_blind(npix::Int; radius_frac::Float64=0.15)
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
    generate_synthetic_blind(object_true, cf_object;
                              aperture, kolmogorov_strength=0.4,
                              noise_frac=0.02, seed=42)
        -> (prob, z_true_phase_part, phase_true, object_true, psf_true, data_clean)

Build a synthetic blind problem with a *given* ground-truth object (e.g. a
real Saturn image, not necessarily drawn from `cf_object`'s prior). Draws
a random phase from the Kolmogorov prior, generates the aberrated noisy
observation, and returns the assembled problem.

The returned `z_true_phase_part` lets you recover the true phase via
`phase_from_latent([z_true_phase_part; zeros(n_object)], prob)`. There is
no "true z_object" because `object_true` was not drawn from `cf_object`.
"""
function generate_synthetic_blind(object_true::AbstractMatrix{<:Real},
                                   cf_object::CorrelatedField;
        aperture::AbstractMatrix{<:Real}=circular_aperture_blind(size(object_true, 1);
                                                          radius_frac=0.15),
        phase_fluct_prior::Tuple{Real,Real}=(0.4, 0.3),
        true_strength::Real=0.4,
        noise_frac::Real=0.02,
        seed::Int=42)
    T = eltype(cf_object)
    Random.seed!(seed)
    n = size(object_true, 1)
    @assert size(aperture) == (n, n)
    @assert cf_object.field_shape == (n, n)
    object_true = Matrix{T}(object_true); aperture = Matrix{T}(aperture)

    grid = FourierGridInfo(n, one(T))
    shape = _kolmogorov_shape_blind(grid)
    white_true = randn(T, n^2)
    phase_true = T(true_strength) .* real.(ifft(fft(reshape(white_true, n, n)) .* shape))
    U_true = aperture .* exp.(im .* phase_true)
    PSF_true = abs2.(fft(U_true))
    PSF_true ./= sum(PSF_true)
    data_clean = real.(ifft(fft(object_true) .* fft(PSF_true)))
    sigma_n = T(noise_frac) * maximum(data_clean)
    data = data_clean .+ sigma_n .* randn(T, n, n)

    # The object's level is carried by cf_object's additive offset; the caller
    # is expected to build cf_object with offset_prior=(log(mean data), σ).
    prob = BlindPhaseRetrievalProblem(aperture, cf_object, data;
                                       sigma_noise=sigma_n,
                                       phase_fluct_prior=phase_fluct_prior, dx=1.0)
    return prob, white_true, true_strength, phase_true, object_true, PSF_true, data_clean
end

# Latent / hyperparameter report (NIFTy-style)

"""
    report_latents(prob, z, samples)

Print the recovered object correlated-field hyperparameters as posterior
mean ± std over the GeoVI `samples` (the analogue of NIFTy's latent report),
plus a standardized-latent sanity check per block. The phase has no
hyperparameters (fixed Kolmogorov kernel); only the object CF does.

`std` of a latent block ≲ 1 means the data has constrained it below the
N(0,1) prior; values far above 1 flag a metric/scaling problem.
"""
# Minisanity latent blocks: phase field, object field, and the object CF
# hyperparameters (slope/fluct/offset), each their own row. Latent layout is
# [z_phase (n²); object: xi_field (n²), slope, fluct, offset].
function VarInf.latent_blocks(prob::BlindPhaseRetrievalProblem)
    nw = prob.n_white; nf = prod(prob.cf_object.field_shape)
    o = nw + 1 + nf            # object hypers start after white(nw)+strength(1)+field(nf)
    return [("ξ phase field",    1:nw),
            ("ξ phase strength", nw+1:nw+1),
            ("ξ object field",   nw+2:o),
            ("ξ slope",          o+1:o+1),
            ("ξ fluct",          o+2:o+2),
            ("ξ offset",         o+3:o+3)]
end

VarInf.whitened_data(prob::BlindPhaseRetrievalProblem) =
    vec(prob.data) ./ prob.sigma_noise

function VarInf.report_latents(prob::BlindPhaseRetrievalProblem,
                               z::AbstractVector{<:Real},
                               samples)
    cf  = prob.cf_object
    cfg = cf.cfgs[1]
    nw, no = prob.n_white, prob.n_object
    obj_latent(zz) = Vector{Float64}(view(zz, nw+2:nw+1+no))

    function hypers(zz)
        _, axes, _ = VarInf.latent_unpack(cf, obj_latent(zz))
        a = axes[1]
        slope     = cfg.slope_mean + cfg.slope_std * a.xi_slope
        fluct     = exp(cfg.fluct_mean + cfg.fluct_std * a.xi_fluct)
        # Realized object log-level = field mean (additive offset_mean + the lognormal-
        # azm DC fluctuation). Parameterization-agnostic; correct under the new
        # NIFTy-azm offset (the additive offset_mean + amp_kernel[DC]=azm·√P).
        logoffset = mean(cf(obj_latent(zz)))
        strength  = phase_strength_from_latent(zz, prob)   # learned turbulence strength
        return (slope, fluct, logoffset, strength)
    end

    draws = [hypers(z .+ s) for s in samples]
    sl = [d[1] for d in draws]; fl = [d[2] for d in draws]
    lo = [d[3] for d in draws]; st = [d[4] for d in draws]
    pin = cfg.slope_std == 0 ? "(pinned)" : ""
    pin_str = prob.phase_fluct_std == 0 ? "(pinned)" : "(learned)"

    # Decoded physical hyperparameters (NIFTy reports ξ-stats; we add the values).
    report_row("phase strength",    (@sprintf("%+.4f ± %.4f", mean(st), std(st))), pin_str)
    report_row("object CF slope",   (@sprintf("%+.3f ± %.3f", mean(sl), std(sl))), pin)
    report_row("object CF fluct",   (@sprintf("%+.4f ± %.4f", mean(fl), std(fl))))
    report_row("object log-offset", (@sprintf("%+.3f ± %.3f", mean(lo), std(lo))),
               (@sprintf("(level %.5f)", exp(mean(lo)))))
    return nothing
end
