# ============================================================================
# PhaseRetrievalProblem: recover an atmospheric phase screen from a single
# blurred-and-noisy image of a known object (Saturn) through a known
# aperture.
#
# Physics:
#   • Aperture A(x, y): real-valued mask (binary circle here).
#   • Phase φ(x, y) on the aperture: Kolmogorov-distributed, latent z is
#     the correlated-field input.
#   • Complex pupil:  U = A · exp(i · φ).
#   • Intensity PSF:  PSF = |fft(U)|², normalized to sum = 1.
#   • Observation:    data = (saturn ⊛ PSF) + N(0, σ²)  per pixel.
#
# Inference target: the latent z producing the phase φ that, after the
# nonlinear chain above, best explains the observed image.
#
# Prior: z ~ N(0, I); the correlated-field machinery shapes φ to have a
# Kolmogorov amplitude spectrum (slope = −11/6 for amplitude ⇒ power-law
# slope −11/3 for the phase power spectrum). The DC mode of the amplitude
# kernel is zeroed so the phase screen is piston-free.
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random

struct PhaseRetrievalProblem{T<:AbstractFloat,TP,TIP} <: AbstractInferenceProblem
    grid::FourierGridInfo{T}
    aperture::Matrix{T}            # n × n real aperture mask
    saturn::Matrix{T}              # known object on n × n grid
    saturn_fft::Matrix{Complex{T}}       # cached fft(saturn)
    data::Matrix{T}                # observed image (saturn ⊛ PSF + noise)
    sigma_noise::T
    amp_kernel::Matrix{T}          # Kolmogorov amp kernel (Fourier, real, DC=0)
    # Cached out-of-place FFT plans + two complex scratch buffers. Every forward/
    # JVP/VJP routes its *transient* transforms through `mul!(scratch, plan, ·)`
    # so those FFT outputs allocate nothing (single-threaded reconstruct ⇒ the
    # shared buffers are safe). Parameterising on the plan types keeps the struct
    # concrete (abstract plan fields would re-introduce dynamic dispatch).
    P::TP                                # plan_fft  (out-of-place)
    iP::TIP                              # plan_ifft (out-of-place, carries 1/N)
    cs1::Matrix{Complex{T}}              # complex scratch
    cs2::Matrix{Complex{T}}             # complex scratch
end

Base.eltype(::PhaseRetrievalProblem{T}) where {T} = T

"""
    _kolmogorov_amp_kernel(grid, strength) -> Matrix{Float64}

Build the Kolmogorov amplitude kernel on a 2D Fourier grid: K[k] ∝ |k|^(-11/6)
for |k| > 0, K[0] = 0 (piston-free). Normalized so that, under the forward
    phase = real.(ifft(fft(z) .* K))    with z ~ N(0, I),
the expected per-pixel variance of `phase` equals `strength²`, i.e.
RMS(phase) = strength.

Derivation: by Parseval, Var(phase) = (1/N) · Σ K², so we scale K to make
Σ K² = strength² · N (= strength² · npix²).
"""
function _kolmogorov_amp_kernel(grid::FourierGridInfo, strength::Real)
    T = eltype(grid)
    k_mat = grid.mode_lengths[grid.bin_index]      # |k| on the 2D grid
    K = similar(k_mat)
    ex = T(-11//6)
    @inbounds for i in eachindex(k_mat)
        K[i] = k_mat[i] == 0 ? zero(T) : k_mat[i]^ex
    end
    strength = T(strength)
    K .*= strength * grid.npix / sqrt(sum(abs2, K))
    return K
end

"""
    PhaseRetrievalProblem(aperture, saturn, data; sigma_noise, kolmogorov_strength,
                          dx=1.0)

Convenience constructor. Builds a Kolmogorov amplitude kernel (amp ∝ |k|^(-11/6),
DC=0, normalized so RMS(phase) ≈ `kolmogorov_strength`) and caches `fft(saturn)`.
"""
function PhaseRetrievalProblem(aperture::AbstractMatrix{<:Real},
                                saturn::AbstractMatrix{<:Real},
                                data::AbstractMatrix{<:Real};
                                sigma_noise::Real,
                                kolmogorov_strength::Real=2.4,
                                dx::Real=1.0)
    n = size(aperture, 1)
    @assert size(aperture) == size(saturn) == size(data) == (n, n)
    T = float(promote_type(eltype(aperture), eltype(saturn), eltype(data)))
    aperture = Matrix{T}(aperture); saturn = Matrix{T}(saturn); data = Matrix{T}(data)
    grid = FourierGridInfo(n, T(dx))
    amp_kernel = _kolmogorov_amp_kernel(grid, kolmogorov_strength)

    cb = zeros(Complex{T}, n, n)        # MEASURE: faster transforms; plan cached & reused
    P  = plan_fft(cb;  flags=FFTW.MEASURE)
    iP = plan_ifft(cb; flags=FFTW.MEASURE)
    return PhaseRetrievalProblem(grid, aperture, saturn,
                                  fft(saturn), data, T(sigma_noise), amp_kernel,
                                  P, iP, zeros(Complex{T}, n, n), zeros(Complex{T}, n, n))
end


VarInf.latent_size(p::PhaseRetrievalProblem) = p.grid.npix^2
VarInf.data_size(p::PhaseRetrievalProblem)   = p.grid.npix^2

# Forward chain (returns data_clean + cached intermediates)

struct _PRFwd{T<:AbstractFloat}
    phase::Matrix{T}
    U::Matrix{Complex{T}}
    F::Matrix{Complex{T}}
    PSF::Matrix{T}
    S::T
    PSF_norm::Matrix{T}
    data_clean::Matrix{T}
end

function _forward(z::AbstractVector{<:Real}, p::PhaseRetrievalProblem)
    n = p.grid.npix
    xi_2d = reshape(z, n, n)
    # phase = real(ifft(fft(ξ) .* K)); transient transforms via scratch
    p.cs1 .= xi_2d
    mul!(p.cs2, p.P, p.cs1)             # F_xi
    p.cs2 .*= p.amp_kernel
    mul!(p.cs1, p.iP, p.cs2)            # ifft(F_xi .* K)
    phase = real.(p.cs1)
    U   = p.aperture .* exp.(im .* phase)
    F   = p.P * U                       # persists (used in JVP/VJP) ⇒ fresh
    PSF = abs2.(F)
    S   = sum(PSF)
    PSF_norm = PSF ./ S
    # data_clean = real(ifft(saturn_fft .* fft(PSF_norm))); transient via scratch
    p.cs1 .= PSF_norm
    mul!(p.cs2, p.P, p.cs1)             # fft(PSF_norm)
    p.cs2 .*= p.saturn_fft
    mul!(p.cs1, p.iP, p.cs2)
    data_clean = real.(p.cs1)
    return _PRFwd(phase, U, F, PSF, S, PSF_norm, data_clean)
end

VarInf.transformation(p::PhaseRetrievalProblem, z::AbstractVector{<:Real}) =
    vec(_forward(z, p).data_clean) ./ p.sigma_noise

# JVP (right_sqrt_metric)

function VarInf.right_sqrt_metric(p::PhaseRetrievalProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    fwd = _forward(z, p)
    n = p.grid.npix
    # 1. v → d_phase (correlated-field forward); transient transforms via scratch
    p.cs1 .= reshape(v, n, n)
    mul!(p.cs2, p.P, p.cs1)
    p.cs2 .*= p.amp_kernel
    mul!(p.cs1, p.iP, p.cs2)
    d_phase = real.(p.cs1)
    # 2. d_phase → dU = i · U · d_phase
    dU = im .* fwd.U .* d_phase
    # 3. dU → dF
    dF = p.P * dU                       # persists into step 4 ⇒ fresh
    # 4. dF → dPSF = 2 · real(conj(F) · dF)
    dPSF = 2 .* real.(conj.(fwd.F) .* dF)
    # 5. dPSF → dPSF_norm
    dPSF_norm = (dPSF .- fwd.PSF_norm .* sum(dPSF)) ./ fwd.S
    # 6. dPSF_norm → d_data_clean  (linear conv with known saturn); via scratch
    p.cs1 .= dPSF_norm
    mul!(p.cs2, p.P, p.cs1)
    p.cs2 .*= p.saturn_fft
    mul!(p.cs1, p.iP, p.cs2)
    d_data_clean = real.(p.cs1)
    return vec(d_data_clean) ./ p.sigma_noise
end

# VJP (left_sqrt_metric)

function VarInf.left_sqrt_metric(p::PhaseRetrievalProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    fwd = _forward(z, p)
    n = p.grid.npix
    N = n * n
    w_2d = reshape(w ./ p.sigma_noise, n, n)
    # 1. data_clean ← w  →  g_PSF_norm (adjoint of conv with saturn); via scratch
    p.cs1 .= w_2d
    mul!(p.cs2, p.P, p.cs1)
    @. p.cs2 *= conj(p.saturn_fft)      # fused conj ⇒ no temporary
    mul!(p.cs1, p.iP, p.cs2)
    g_PSF_norm = real.(p.cs1)
    # 2. g_PSF_norm → g_PSF (adjoint of PSF/S normalization)
    inner_sum = dot(g_PSF_norm, fwd.PSF_norm)
    g_PSF = (g_PSF_norm .- inner_sum) ./ fwd.S
    # 3. g_PSF → g_F = 2 · g_PSF · F (complex)
    g_F = 2 .* g_PSF .* fwd.F
    # 4. g_F → g_U = N · ifft(g_F)  (adjoint of fft is N·ifft); g_U persists ⇒ fresh
    g_U = N .* (p.iP * g_F)
    # 5. g_U → g_phase  (adjoint of φ → A·exp(iφ): g_φ = imag(conj(U)·g_U))
    #    multiplied by A is already inside U, no extra mask needed
    g_phase = imag.(conj.(fwd.U) .* g_U)
    # 6. g_phase → g_xi via CF adjoint (amp_kernel real ⇒ self-adjoint); via scratch
    p.cs1 .= g_phase
    mul!(p.cs2, p.P, p.cs1)
    p.cs2 .*= p.amp_kernel
    mul!(p.cs1, p.iP, p.cs2)
    g_xi_2d = real.(p.cs1)
    return vec(g_xi_2d)
end

function VarInf.energy_and_gradient(p::PhaseRetrievalProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = vec(p.data) ./ p.sigma_noise
    resid = T_z .- d_white
    chi2  = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end

# Helpers: circular aperture + synthetic-data generator

"""
    circular_aperture(npix; radius_frac=0.45)

Return an `npix × npix` binary aperture mask centered on the array, with
radius `radius_frac · npix`. Centered ⇒ |fft(aperture)|² PSF lives at the
origin (index 1, 1).
"""
function circular_aperture(npix::Int; radius_frac::Float64=0.45)
    r2 = (radius_frac * npix)^2
    mask = zeros(npix, npix)
    cx = (npix + 1) / 2
    cy = (npix + 1) / 2
    @inbounds for j in 1:npix, i in 1:npix
        if (i - cx)^2 + (j - cy)^2 <= r2
            mask[i, j] = 1.0
        end
    end
    return mask
end

"""
    asymmetric_aperture(npix; radius_frac=0.35, offset_frac=0.03,
                        notch_radius_frac=0.13, notch_dist_frac=0.30,
                        notch_dir=(-0.6, -0.8))

A **non-centrosymmetric** pupil: a (near-centred) circle with a circular
obscuration punched **near the centre but off to one side** (like an off-axis
secondary), plus a small off-centre shift of the circle itself. Because the
mask is not invariant under reflection through the array centre, the twin-image
ambiguity `φ → −φ_flipped` (under which `|fft(A·exp(iφ))|²` is invariant for a
*centrosymmetric* `A`) is broken, so cold-start phase retrieval converges
reliably to the true phase instead of landing in the twin basin at random.
The off-centre obscuration is the main symmetry-breaker; `notch_dist_frac` sets
its centre distance from the pupil centre as a fraction of the radius (small ⇒
near the centre, 1 ⇒ on the rim).
"""
function asymmetric_aperture(npix::Int; radius_frac::Float64=0.35,
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

"""
    phase_from_latent(z, prob) -> Matrix{Float64}

Convenience: realise the phase screen from a latent `z` under `prob`'s
Kolmogorov amplitude kernel.
"""
function phase_from_latent(z::AbstractVector{<:Real}, p::PhaseRetrievalProblem)
    n = p.grid.npix
    p.cs1 .= reshape(z, n, n)
    mul!(p.cs2, p.P, p.cs1)
    p.cs2 .*= p.amp_kernel
    mul!(p.cs1, p.iP, p.cs2)
    return real.(p.cs1)
end

"""
    generate_synthetic_phase_retrieval(saturn; aperture, kolmogorov_strength=2.4,
                                       noise_frac=0.02, seed=42)
        -> (prob, z_true, phase_true, psf_true, data_clean)

Build a synthetic phase-retrieval problem: draw z_true ~ N(0, I), realise
the Kolmogorov phase, compute the aberrated PSF, convolve with the known
`saturn` image, add Gaussian noise, and assemble the problem.
"""
function generate_synthetic_phase_retrieval(saturn::AbstractMatrix{<:Real};
        aperture::AbstractMatrix{<:Real}=circular_aperture(size(saturn, 1)),
        kolmogorov_strength::Real=2.4,
        noise_frac::Real=0.02,
        seed::Int=42,
        T::Type{<:AbstractFloat}=Float64)
    Random.seed!(seed)
    n = size(saturn, 1)
    @assert size(aperture) == (n, n)
    saturn = Matrix{T}(saturn); aperture = Matrix{T}(aperture)

    # First build a stub problem so we can use its amp_kernel and forward.
    stub = PhaseRetrievalProblem(aperture, saturn, zeros(T, n, n);
                                  sigma_noise=one(T),
                                  kolmogorov_strength=kolmogorov_strength,
                                  dx=1.0)
    z_true = randn(T, n^2)
    fwd = _forward(z_true, stub)
    sigma_n = T(noise_frac) * maximum(fwd.data_clean)
    data = fwd.data_clean .+ sigma_n .* randn(T, n, n)

    prob = PhaseRetrievalProblem(aperture, saturn, data;
                                  sigma_noise=sigma_n,
                                  kolmogorov_strength=kolmogorov_strength,
                                  dx=1.0)
    return prob, z_true, fwd.phase, fwd.PSF_norm, fwd.data_clean
end
