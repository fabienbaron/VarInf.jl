# ============================================================================
# LogNormalFieldProblem: full-IWP correlated-field prior with a lognormal
# scaling, identity forward, Gaussian likelihood.
#
# This is VarInf's analogue of NIFTy's `0_intro.py`. It exercises the parts
# of VarInf that other examples don't:
#   * `amplitude_spectrum_jvp`  (learned slope / fluct / flex / asperity /
#                                 per-bin spectrum, all driven by latent z)
#   * `amplitude_spectrum_adjoint`
#   * the integrated Wiener process (`use_iwp = true` path)
#
# Forward chain:
#     z ──┐
#         │
#         │ unpack
#         ▼
#   xi_field      xi_slope, xi_fluct, xi_flex, xi_asp, xi_spectrum     xi_scaling
#         │              │                                                  │
#         │              ▼                                                  │
#         │      amp = amplitude_spectrum(...)                              │
#         │              │                                                  │
#         │              ▼                                                  │
#         │      amp_kernel = amp[grid.bin_index]                           │
#         ▼              ▼                                                  ▼
#       fft  →  fft(xi_field) .* amp_kernel  →  ifft  →  field             scaling
#                                                            │                │
#                                                            └─ exp(field) ──┘
#                                                                  │
#                                                                  ▼
#                                                       signal = scaling · exp(field)
#                                                                  │
#                                                                  ▼ (identity)
#                                                              data ~ N(signal, σ²)
#
# Latent layout (with IWP enabled):
#   z[1 : N]                    xi_field         (N = grid.npix²)
#   z[N + 1]                    xi_slope
#   z[N + 2]                    xi_fluct
#   z[N + 3]                    xi_flex
#   z[N + 4]                    xi_asp
#   z[N + 5 : N + 4 + 2·(n_bins-2)]  xi_spectrum (flattened, reshaped to
#                                                 (2, n_bins-2) inside)
#   z[end]                      xi_scaling
# total = N + 4 + 2·(n_bins-2) + 1 = N + 2·n_bins + 1.
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random

struct LogNormalFieldProblem{T<:AbstractFloat} <: AbstractInferenceProblem
    grid::FourierGridInfo{T}
    cfg::CorrFieldConfig{T}              # must have use_iwp == true here
    scaling_mean::T                      # lognormal prior on scaling: log(s) = mean + std·z
    scaling_std::T
    data::Matrix{T}
    sigma_noise::T
end

Base.eltype(::LogNormalFieldProblem{T}) where {T} = T

# Number of `xi_spectrum` entries (matrix is (2, n_bins-2), stored flat in z).
_xi_spectrum_size(p::LogNormalFieldProblem) = 2 * (p.grid.n_bins - 2)

function VarInf.latent_size(p::LogNormalFieldProblem)
    N = p.grid.npix^2
    # xi_field (N) + slope + fluct + flex + asp + xi_spectrum (n_spec)
    # + xi_offset (1) + xi_scaling (1).
    return N + 4 + _xi_spectrum_size(p) + 2
end

VarInf.data_size(p::LogNormalFieldProblem) = p.grid.npix^2

# Latent unpack / pack

struct _UnpackedLogNormal{T<:AbstractFloat}
    xi_field::Matrix{T}     # (npix, npix)
    xi_slope::T
    xi_fluct::T
    xi_flex::T
    xi_asp::T
    xi_spectrum::Matrix{T}  # (n_bins-2, 2): col 1 = pos noise, col 2 = vel noise
    xi_offset::T            # DC-bin amplitude latent
    xi_scaling::T
end

function _unpack(z::AbstractVector{<:Real}, p::LogNormalFieldProblem)
    Tz = eltype(p)
    n = p.grid.npix
    N = n * n
    n_spec = _xi_spectrum_size(p)

    xi_field = reshape(z[1:N], n, n)
    xi_slope = z[N + 1]
    xi_fluct = z[N + 2]
    xi_flex  = z[N + 3]
    xi_asp   = z[N + 4]
    xi_spectrum = reshape(z[N + 5 : N + 4 + n_spec], p.grid.n_bins - 2, 2)
    xi_offset  = z[N + 5 + n_spec]
    xi_scaling = z[end]
    return _UnpackedLogNormal(Matrix{Tz}(xi_field), Tz(xi_slope), Tz(xi_fluct),
                              Tz(xi_flex), Tz(xi_asp), Matrix{Tz}(xi_spectrum),
                              Tz(xi_offset), Tz(xi_scaling))
end

# Pack a gradient computed in unpacked form back into a flat vector.
function _pack_grad(g_xi_field::AbstractMatrix{<:Real},
                    g_xi_slope::Real, g_xi_fluct::Real,
                    g_xi_flex::Real,  g_xi_asp::Real,
                    g_xi_spectrum::AbstractMatrix{<:Real},
                    g_xi_offset::Real,
                    g_xi_scaling::Real,
                    p::LogNormalFieldProblem)
    n = p.grid.npix
    N = n * n
    n_spec = _xi_spectrum_size(p)
    out = Vector{eltype(p)}(undef, latent_size(p))
    out[1:N]          .= vec(g_xi_field)
    out[N + 1]         = g_xi_slope
    out[N + 2]         = g_xi_fluct
    out[N + 3]         = g_xi_flex
    out[N + 4]         = g_xi_asp
    out[N + 5 : N + 4 + n_spec] .= vec(g_xi_spectrum)
    out[N + 5 + n_spec] = g_xi_offset
    out[end]           = g_xi_scaling
    return out
end

# Forward (cached intermediates for JVP/VJP reuse)

struct _Fwd{T<:AbstractFloat}
    u::_UnpackedLogNormal{T}
    amp::Vector{T}          # length n_bins
    amp_kernel::Matrix{T}   # (npix, npix), real
    F_xi::Matrix{Complex{T}}      # fft(xi_field)
    field::Matrix{T}        # (npix, npix), real
    scaling::T
    signal::Matrix{T}       # scaling · exp(field)
end

function _forward(z::AbstractVector{<:Real}, p::LogNormalFieldProblem)
    u = _unpack(z, p)
    # The DC bin is controlled by the lognormal offset prior in p.cfg
    # (cfg.use_offset == true); `xi_offset` is the latent that drives it.
    amp = amplitude_spectrum(u.xi_slope, u.xi_fluct, u.xi_flex, u.xi_asp,
                             u.xi_spectrum, p.cfg, p.grid;
                             xi_offset=u.xi_offset)
    amp_kernel = amp[p.grid.bin_index]
    F_xi = fft(u.xi_field)
    field = real.(ifft(F_xi .* amp_kernel))
    scaling = exp(p.scaling_mean + p.scaling_std * u.xi_scaling)
    signal = scaling .* exp.(field)
    return _Fwd(u, amp, amp_kernel, F_xi, field, scaling, signal)
end

# Protocol implementations

function VarInf.transformation(p::LogNormalFieldProblem,
                                z::AbstractVector{<:Real})
    fwd = _forward(z, p)
    return vec(fwd.signal) ./ p.sigma_noise
end

# JVP: J · v
function VarInf.right_sqrt_metric(p::LogNormalFieldProblem,
                                   z::AbstractVector{<:Real},
                                   v::AbstractVector{<:Real})
    fwd = _forward(z, p)
    v_u = _unpack(v, p)

    # 1. Amplitude spectrum tangent (includes DC bin via xi_offset / v_offset)
    d_amp = amplitude_spectrum_jvp(v_u.xi_slope, v_u.xi_fluct,
                                    v_u.xi_flex,  v_u.xi_asp,
                                    v_u.xi_spectrum,
                                    fwd.u.xi_slope, fwd.u.xi_fluct,
                                    fwd.u.xi_flex,  fwd.u.xi_asp,
                                    fwd.u.xi_spectrum,
                                    p.cfg, p.grid;
                                    xi_offset=fwd.u.xi_offset,
                                    v_offset=v_u.xi_offset)
    d_amp_kernel = d_amp[p.grid.bin_index]

    # 2. Field tangent: d_field = real(ifft(fft(v_field)·amp_kernel + F_xi·d_amp_kernel))
    F_v_field = fft(v_u.xi_field)
    d_field = real.(ifft(F_v_field .* fwd.amp_kernel .+ fwd.F_xi .* d_amp_kernel))

    # 3. Scaling tangent: d_scaling = scaling · scaling_std · v_scaling
    d_scaling = fwd.scaling * p.scaling_std * v_u.xi_scaling

    # 4. Signal tangent:  signal = scaling · exp(field)
    #    d_signal = d_scaling · exp(field) + scaling · exp(field) · d_field
    #             = signal · (d_scaling / scaling + d_field)
    d_signal = fwd.signal .* ((d_scaling / fwd.scaling) .+ d_field)

    return vec(d_signal) ./ p.sigma_noise
end

# VJP: J' · w
function VarInf.left_sqrt_metric(p::LogNormalFieldProblem,
                                  z::AbstractVector{<:Real},
                                  w::AbstractVector{<:Real})
    fwd = _forward(z, p)
    n = p.grid.npix
    N2 = n * n
    w_scaled_2d = reshape(w ./ p.sigma_noise, n, n)

    # 1. Adjoint of signal = scaling · exp(field):
    #    g_signal_through = w_scaled_2d  (identity forward)
    #    g_field = g_signal · signal           (chain through exp(field))
    #    g_scaling = sum(g_signal · exp(field)) = sum(g_signal · signal / scaling)
    g_field_2d = w_scaled_2d .* fwd.signal
    g_scaling = sum(w_scaled_2d .* fwd.signal) / fwd.scaling
    # g_xi_scaling = g_scaling · ∂scaling/∂xi_scaling = g_scaling · scaling · scaling_std
    g_xi_scaling = g_scaling * fwd.scaling * p.scaling_std

    # 2. Adjoint of field = real(ifft(F_xi · amp_kernel)):
    #    The CF operator with real, even amp_kernel is self-adjoint w.r.t. xi_field:
    g_xi_field_2d = real.(ifft(fft(g_field_2d) .* fwd.amp_kernel))
    #    And the gradient w.r.t. amp_kernel (per Fourier mode):
    #    g_amp_kernel[k,l] = real(F_xi[k,l] · conj(fft(g_field)[k,l])) / N²
    F_g = fft(g_field_2d)
    g_amp_kernel_2d = real.(fwd.F_xi .* conj.(F_g)) ./ N2

    # 3. Sum-reduce g_amp_kernel_2d by bin_index → g_amp (length n_bins).
    #    The DC bin's gradient is now propagated through amplitude_spectrum's
    #    `xi_offset` latent; no manual zeroing needed.
    g_amp = zeros(eltype(p), p.grid.n_bins)
    @inbounds for j in 1:n, i in 1:n
        g_amp[p.grid.bin_index[i, j]] += g_amp_kernel_2d[i, j]
    end

    # 4. Backprop g_amp through amplitude_spectrum (6-tuple incl. g_xi_offset)
    g_xi_slope, g_xi_fluct, g_xi_flex, g_xi_asp, g_xi_spectrum, g_xi_offset =
        amplitude_spectrum_adjoint(g_amp,
                                    fwd.u.xi_slope, fwd.u.xi_fluct,
                                    fwd.u.xi_flex,  fwd.u.xi_asp,
                                    fwd.u.xi_spectrum,
                                    p.cfg, p.grid;
                                    xi_offset=fwd.u.xi_offset)

    return _pack_grad(g_xi_field_2d, g_xi_slope, g_xi_fluct, g_xi_flex,
                      g_xi_asp, g_xi_spectrum, g_xi_offset, g_xi_scaling, p)
end

function VarInf.energy_and_gradient(p::LogNormalFieldProblem,
                                     z::AbstractVector{<:Real})
    T_z = VarInf.transformation(p, z)
    d_white = vec(p.data) ./ p.sigma_noise
    resid = T_z .- d_white
    chi2 = dot(resid, resid) / 2
    prior = dot(z, z) / 2
    grad = VarInf.left_sqrt_metric(p, z, resid) .+ z
    return chi2 + prior, grad
end

# Convenience: synthetic data generator

"""
    generate_synthetic_lognormal(; npix, scaling_mean=2.0, scaling_std=0.5,
                                  cfg=..., noise_frac=0.05, seed=42)
        -> (prob, z_true, signal_true, amp_true)
"""
function generate_synthetic_lognormal(; npix::Int=64,
                                       dx::Real=1.0 / 64,
                                       scaling_mean::Real=2.0,
                                       scaling_std::Real=0.5,
                                       cfg::CorrFieldConfig=
                                           CorrFieldConfig(slope_prior=(-2.0, 0.5),
                                                            fluct_prior=(0.4, 0.04),
                                                            flex_prior=(1.0, 0.5),
                                                            asp_prior=(0.6, 0.06),
                                                            offset_prior=(0.0, 0.1)),
                                       noise_frac::Real=0.05,
                                       seed::Int=42,
                                       T::Type{<:AbstractFloat}=eltype(cfg))
    @assert cfg.use_iwp    "this example requires CorrFieldConfig with use_iwp = true"
    @assert cfg.use_offset "this example requires CorrFieldConfig with offset_prior set"
    Random.seed!(seed)

    grid = FourierGridInfo(npix, T(dx))
    stub = LogNormalFieldProblem(grid, cfg, T(scaling_mean), T(scaling_std),
                                  zeros(T, npix, npix), one(T))
    n_z = latent_size(stub)
    z_true = randn(T, n_z)
    fwd = _forward(z_true, stub)
    sigma_n = T(noise_frac) * maximum(fwd.signal)
    data = fwd.signal .+ sigma_n .* randn(T, npix, npix)

    prob = LogNormalFieldProblem(grid, cfg, T(scaling_mean), T(scaling_std),
                                  data, sigma_n)
    return prob, z_true, fwd.signal, fwd.amp
end
