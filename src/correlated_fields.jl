"""
    make_smoothing_kernel(nx, ny, dx, sigma) -> Matrix{Float64}

Precompute the Gaussian smoothing kernel in Fourier space.
`dx` and `sigma` must be in the same units (e.g. radians).
"""
function make_smoothing_kernel(nx::Int, ny::Int, dx::T, sigma::T) where {T<:AbstractFloat}
    kx = fftfreq(nx, inv(dx))   # cycles per unit
    ky = fftfreq(ny, inv(dx))
    c = 2 * T(π)^2
    kernel = [exp(-c * sigma^2 * (kx[i]^2 + ky[j]^2)) for i in 1:nx, j in 1:ny]
    return kernel
end

"""
    harmonic_smooth(image, kernel) -> Matrix{Float64}

Apply Gaussian smoothing via FFT convolution.
`kernel` is the precomputed Fourier-space kernel from `make_smoothing_kernel`.
The operation is self-adjoint (kernel is real and symmetric).
"""
function harmonic_smooth(image::AbstractMatrix{<:Real}, kernel::AbstractMatrix{<:Real})
    return real.(ifft(fft(image) .* kernel))
end

# ============================================================================
# Correlated Field Model (NIFTy-style)
# ============================================================================

"""
    CorrFieldConfig

Prior hyperparameters for a NIFTy-style correlated field. Each parameter maps a
`N(0,1)` latent to a physical value:
- `slope` (loglogavgslope): a **normal** prior, value `= mean + std·ξ`.
- `fluctuations`, `flexibility`, `asperity`: **lognormal** priors specified by
  their value-space `(mean, std)`, exactly like NIFTy's `LogNormalPrior`: value
  `= exp(μ + σ·ξ)` with `σ² = log1p((std/mean)²)`, `μ = log(mean) − σ²/2`.
  So you pass the value directly (e.g. `fluct_prior=(0.4, 0.1)`), not its log;
  `std=0` pins the value at `mean`. (Stored internally as `(μ, σ)`.)
"""
struct CorrFieldConfig{T<:AbstractFloat}
    slope_mean::T
    slope_std::T
    fluct_mean::T
    fluct_std::T
    flex_mean::T
    flex_std::T
    asp_mean::T
    asp_std::T
    offset_mean::T    # (mean, std) lognormal prior on amp[1] when use_offset
    offset_std::T
    use_iwp::Bool
    use_offset::Bool
end

Base.eltype(::CorrFieldConfig{T}) where {T} = T

"""
    CorrFieldConfig(; slope_prior, fluct_prior, flex_prior=nothing, asp_prior=nothing,
                     offset_prior=nothing)

Construct from `(mean, std)` prior tuples.
- `flex_prior = nothing` (default) disables the integrated Wiener process layer.
- `offset_prior = nothing` (default) keeps the legacy DC-bin convention
  `amplitude[1] = npix²`, which is convenient for masked / interferometric
  applications that already kill the mean elsewhere (e.g. via a limb-darkened
  disk weight + zero-mean projection in `OIVI`). For new applications that
  want to control the DC mode explicitly, pass `offset_prior=(mean, std)`.
  (Offset keeps its own convention, additive in `CorrelatedField`, log in the
  `amplitude_spectrum` DC path, and is *not* lognormal-converted here.)

`fluct_prior`/`flex_prior`/`asp_prior` are value-space `(mean, std)` (NIFTy
`LogNormalPrior`); `slope_prior` is a normal `(mean, std)`. See `CorrFieldConfig`.
"""
# NIFTy LogNormalPrior(mean, std) → underlying normal (μ, σ) for value = exp(μ+σξ).
# std = 0 ⇒ σ = 0, μ = log(mean) (pinned at mean).
function _lognormal_params(mean::T, std::T) where {T<:AbstractFloat}
    mean > 0 || throw(ArgumentError("lognormal prior mean must be > 0, got $mean"))
    σ² = log1p((std / mean)^2)
    return (log(mean) - σ² / 2, sqrt(σ²))
end

# Float type from a set of prior tuples (each entry may be Int/Float32/Float64).
_prior_eltype(ts...) = float(promote_type(map(t -> t === nothing ? Bool :
                                               promote_type(typeof(t[1]), typeof(t[2])),
                                               ts)...))

function CorrFieldConfig(; slope_prior::Tuple{Real,Real},
                           fluct_prior::Tuple{Real,Real},
                           flex_prior::Union{Nothing,Tuple{Real,Real}}=nothing,
                           asp_prior::Union{Nothing,Tuple{Real,Real}}=nothing,
                           offset_prior::Union{Nothing,Tuple{Real,Real}}=nothing)
    T = _prior_eltype(slope_prior, fluct_prior, flex_prior, asp_prior, offset_prior)
    use_iwp = flex_prior !== nothing && asp_prior !== nothing
    use_offset = offset_prior !== nothing
    fluct_m, fluct_s = _lognormal_params(T(fluct_prior[1]), T(fluct_prior[2]))
    flex_m, flex_s = use_iwp ? _lognormal_params(T(flex_prior[1]), T(flex_prior[2])) : (zero(T), one(T))
    asp_m,  asp_s  = use_iwp ? _lognormal_params(T(asp_prior[1]),  T(asp_prior[2]))  : (zero(T), one(T))
    return CorrFieldConfig(
        T(slope_prior[1]), T(slope_prior[2]),
        fluct_m, fluct_s,
        flex_m, flex_s,
        asp_m, asp_s,
        use_offset ? T(offset_prior[1]) : zero(T),
        use_offset ? T(offset_prior[2]) : zero(T),
        use_iwp, use_offset)
end

"""
    _spectral_latent_size(cfg, n_bins) -> Int

Number of latent variables for one correlated field's spectral parameters.
"""
function _spectral_latent_size(cfg::CorrFieldConfig, n_bins::Int)
    n = 2  # slope, fluct
    if cfg.use_iwp
        n += 2 + 2 * (n_bins - 2)  # flex, asp, spectrum
    end
    return n
end

"""
    fourier_mode_distributor(npix, dx)

Compute radial binning of 2D Fourier modes. Returns:
- `bin_index`: npix×npix matrix mapping each mode to its radial bin (1-based)
- `multiplicity`: number of modes per bin
- `mode_lengths`: sorted unique |k| values per bin
- `rel_log_mode_lengths`: log(k/k_ref), zero at k=0
- `log_volume`: log-spacing between consecutive nonzero bins (for IWP)
- `total_volume`: (npix*dx)² spatial volume
- `n_bins`: number of radial bins
"""
function fourier_mode_distributor(npix::Int, dx::T) where {T<:AbstractFloat}
    kx = fftfreq(npix, inv(dx))
    ky = fftfreq(npix, inv(dx))

    k_abs = [sqrt(kx[i]^2 + ky[j]^2) for i in 1:npix, j in 1:npix]

    # Round to avoid floating-point uniqueness issues
    k_rounded = [round(k_abs[i,j], sigdigits=12) for i in 1:npix, j in 1:npix]
    unique_k = sort(unique(vec(k_rounded)))
    n_bins = length(unique_k)

    # Assign each mode to a radial bin
    bin_index = Matrix{Int}(undef, npix, npix)
    for j in 1:npix, i in 1:npix
        bin_index[i,j] = searchsortedfirst(unique_k, k_rounded[i,j])
    end

    # Count multiplicity per bin
    multiplicity = zeros(T, n_bins)
    for b in vec(bin_index)
        multiplicity[b] += one(T)
    end

    mode_lengths = unique_k

    # Relative log mode lengths: log(k/k_ref), k_ref = first nonzero mode
    rel_log = zeros(T, n_bins)
    if n_bins > 1
        k_ref = mode_lengths[2]
        for b in 2:n_bins
            rel_log[b] = log(mode_lengths[b] / k_ref)
        end
    end

    # Log-spacing between consecutive nonzero bins (IWP step sizes)
    log_vol = T[]
    if n_bins > 2
        for b in 2:n_bins-1
            push!(log_vol, log(mode_lengths[b+1] / mode_lengths[b]))
        end
    end

    total_volume = (npix * dx)^2

    return bin_index, multiplicity, mode_lengths, rel_log, log_vol, total_volume, n_bins
end


# ============================================================================
# FourierGridInfo: bundle the radial-binning grid data into one struct.
#
# Duck-typed in `amplitude_spectrum*`: those functions only read .npix,
# .n_bins, .bin_index, .mode_multiplicity, .rel_log_mode_lengths, .log_volume,
# so they work transparently on either a FourierGridInfo or a SkyModelParams
# (the latter forwards those names to its embedded grid via getproperty).
# ============================================================================

"""
    FourierGridInfo

Holds the radial-binning of 2D Fourier modes for a square pixel grid.
Builds via `FourierGridInfo(npix, dx)` (which wraps `fourier_mode_distributor`).
"""
struct FourierGridInfo{T<:AbstractFloat}
    npix::Int
    dx::T
    n_bins::Int
    bin_index::Matrix{Int}
    mode_multiplicity::Vector{T}
    mode_lengths::Vector{T}
    rel_log_mode_lengths::Vector{T}
    log_volume::Vector{T}
    total_volume::T
end

Base.eltype(::FourierGridInfo{T}) where {T} = T

"""
    FourierGridInfo(npix, dx) -> FourierGridInfo

Construct a grid by calling `fourier_mode_distributor` and packaging the result.
The element type follows `dx` (pass `dx::Float32` for a Float32 grid).
"""
function FourierGridInfo(npix::Int, dx::T) where {T<:AbstractFloat}
    bin_index, mult, mode_lengths, rel_log, log_vol, total_volume, n_bins =
        fourier_mode_distributor(npix, dx)
    return FourierGridInfo(npix, dx, n_bins, bin_index, mult, mode_lengths,
                            rel_log, log_vol, total_volume)
end
FourierGridInfo(npix::Int, dx::Real) = FourierGridInfo(npix, float(dx))


# ============================================================================
# Axis1DInfo: 1-D sibling of FourierGridInfo for explicit per-axis composition
# in a multi-axis correlated-field model. Same field names as FourierGridInfo
# so the `amplitude_spectrum*` family can take either via duck typing.
# ============================================================================

"""
    axis_mode_distributor(npix, dx)

Compute |k|-binning of 1-D Fourier modes (returns analogous components to
`fourier_mode_distributor` but with a 1-D `bin_index`).
"""
function axis_mode_distributor(npix::Int, dx::T) where {T<:AbstractFloat}
    kx = fftfreq(npix, inv(dx))
    k_abs = abs.(kx)
    k_rounded = [round(k, sigdigits=12) for k in k_abs]
    unique_k = sort(unique(k_rounded))
    n_bins = length(unique_k)

    bin_index = Vector{Int}(undef, npix)
    for i in 1:npix
        bin_index[i] = searchsortedfirst(unique_k, k_rounded[i])
    end

    multiplicity = zeros(T, n_bins)
    for b in bin_index
        multiplicity[b] += one(T)
    end

    mode_lengths = unique_k

    rel_log = zeros(T, n_bins)
    if n_bins > 1
        k_ref = mode_lengths[2]
        for b in 2:n_bins
            rel_log[b] = log(mode_lengths[b] / k_ref)
        end
    end

    log_vol = T[]
    if n_bins > 2
        for b in 2:n_bins-1
            push!(log_vol, log(mode_lengths[b+1] / mode_lengths[b]))
        end
    end

    total_volume = npix * dx
    return bin_index, multiplicity, mode_lengths, rel_log, log_vol, total_volume, n_bins
end

"""
    Axis1DInfo

Holds the |k|-binning of a 1-D Fourier axis. `bin_index` is a
`Vector{Int}` (one entry per mode along this axis). Use this when composing
a multi-axis correlated field that has a purely 1-D axis (e.g. wavelength
in addition to a 2-D `FourierGridInfo` spatial axis); the
`amplitude_spectrum*` family accepts either grid type via duck typing.
"""
struct Axis1DInfo{T<:AbstractFloat}
    npix::Int
    dx::T
    n_bins::Int
    bin_index::Vector{Int}
    mode_multiplicity::Vector{T}
    mode_lengths::Vector{T}
    rel_log_mode_lengths::Vector{T}
    log_volume::Vector{T}
    total_volume::T
end

Base.eltype(::Axis1DInfo{T}) where {T} = T

"""
    Axis1DInfo(npix, dx) -> Axis1DInfo

Construct a 1-D axis grid by calling `axis_mode_distributor`.
The element type follows `dx`.
"""
function Axis1DInfo(npix::Int, dx::T) where {T<:AbstractFloat}
    bin_index, mult, mode_lengths, rel_log, log_vol, total_volume, n_bins =
        axis_mode_distributor(npix, dx)
    return Axis1DInfo(npix, dx, n_bins, bin_index, mult, mode_lengths,
                       rel_log, log_vol, total_volume)
end
Axis1DInfo(npix::Int, dx::Real) = Axis1DInfo(npix, float(dx))


# --- Integrated Wiener Process (IWP) for smooth spectral deviations ---

"""
    integrated_wiener_process(xi_spec, sigma, asp, dt) -> Vector{Float64}

Compute smooth deviations via an integrated Wiener process.
- `xi_spec`: (n_steps × 2) white noise matrix (position, velocity)
- `sigma`: flexibility (controls deviation amplitude)
- `asp`: asperity (roughness parameter)
- `dt`: log-spacing step sizes (n_steps vector)

Returns position vector of length n_steps + 1.
"""
function integrated_wiener_process(xi_spec::AbstractMatrix{<:Real},
                                   sigma::Real, asp::Real,
                                   dt::AbstractVector{<:Real})
    T = float(promote_type(eltype(xi_spec), typeof(sigma), typeof(asp), eltype(dt)))
    n_steps = length(dt)
    sqrt_dt = sqrt.(dt)
    asp_term = sqrt.(dt .^ 2 ./ T(12) .+ asp)

    # Scale noise
    pos_scale = sigma .* sqrt_dt .* asp_term
    vel_scale = sigma .* sqrt_dt

    # Combined noise with half-step coupling
    pos_noise = pos_scale .* xi_spec[:, 1] .+ T(1//2) .* dt .* vel_scale .* xi_spec[:, 2]
    vel_noise = vel_scale .* xi_spec[:, 2]

    # Integrate: prepend zero, cumsum velocity, drift, cumsum position
    vel_cs = cumsum(vel_noise)
    vwz = vcat(zero(T), vel_cs)  # velocity with initial zero, length n_steps + 1

    pwd = vcat(zero(T), pos_noise .+ dt .* vwz[1:n_steps])
    pos = cumsum(pwd)

    return pos
end

"""
    iwp_adjoint(g_pos, xi_spec, sigma, asp, dt) -> (g_xi_spec, g_sigma, g_asp)

Adjoint of `integrated_wiener_process`.
"""
function iwp_adjoint(g_pos::AbstractVector{<:Real},
                     xi_spec::AbstractMatrix{<:Real},
                     sigma::Real, asp::Real,
                     dt::AbstractVector{<:Real})
    T = float(promote_type(eltype(g_pos), eltype(xi_spec), typeof(sigma), typeof(asp), eltype(dt)))
    n_steps = length(dt)
    sqrt_dt = sqrt.(dt)
    asp_term = sqrt.(dt .^ 2 ./ T(12) .+ asp)

    pos_scale = sigma .* sqrt_dt .* asp_term
    vel_scale = sigma .* sqrt_dt

    # Adjoint of cumsum(pwd): reverse cumsum
    g_pwd = reverse(cumsum(reverse(g_pos)))

    # pwd = [0; pos_noise + dt .* vwz[1:n_steps]]
    g_pos_noise = g_pwd[2:end]
    g_vwz_drift = dt .* g_pwd[2:end]

    # vwz = [0; vel_cs], gradient flows to vel_cs
    g_vel_cs = zeros(T, n_steps)
    if n_steps > 1
        g_vel_cs[1:n_steps-1] .= g_vwz_drift[2:n_steps]
    end

    # Adjoint of cumsum(vel_noise)
    g_vel_noise = reverse(cumsum(reverse(g_vel_cs)))

    # Backprop through noise construction
    g_xi_1 = pos_scale .* g_pos_noise
    g_xi_2_from_pos = T(1//2) .* dt .* vel_scale .* g_pos_noise
    g_xi_2_from_vel = vel_scale .* g_vel_noise
    g_xi_spec = hcat(g_xi_1, g_xi_2_from_pos .+ g_xi_2_from_vel)

    # Backprop through scales
    g_pos_scale = xi_spec[:, 1] .* g_pos_noise
    g_vel_scale = xi_spec[:, 2] .* g_vel_noise .+ T(1//2) .* dt .* xi_spec[:, 2] .* g_pos_noise

    g_sigma = sum(sqrt_dt .* asp_term .* g_pos_scale) +
              sum(sqrt_dt .* g_vel_scale)

    # asp_term = sqrt(dt²/12 + asp), d/d(asp) = 0.5/asp_term
    g_asp_term = sigma .* sqrt_dt .* g_pos_scale
    g_asp = sum(g_asp_term .* T(1//2) ./ asp_term)

    return g_xi_spec, g_sigma, g_asp
end

"""
    iwp_jvp(v_spec, d_sigma, d_asp, xi_spec, sigma, asp, dt) -> d_pos

JVP of `integrated_wiener_process`.
"""
function iwp_jvp(v_spec::AbstractMatrix{<:Real},
                 d_sigma::Real, d_asp::Real,
                 xi_spec::AbstractMatrix{<:Real},
                 sigma::Real, asp::Real,
                 dt::AbstractVector{<:Real})
    T = float(promote_type(eltype(v_spec), typeof(d_sigma), typeof(d_asp),
                           eltype(xi_spec), typeof(sigma), typeof(asp), eltype(dt)))
    n_steps = length(dt)
    sqrt_dt = sqrt.(dt)
    asp_term = sqrt.(dt .^ 2 ./ T(12) .+ asp)
    d_asp_term = d_asp .* T(1//2) ./ asp_term

    pos_scale = sigma .* sqrt_dt .* asp_term
    vel_scale = sigma .* sqrt_dt
    d_pos_scale = d_sigma .* sqrt_dt .* asp_term .+ sigma .* sqrt_dt .* d_asp_term
    d_vel_scale = d_sigma .* sqrt_dt

    d_pos_noise = d_pos_scale .* xi_spec[:, 1] .+ pos_scale .* v_spec[:, 1] .+
                  T(1//2) .* dt .* (d_vel_scale .* xi_spec[:, 2] .+ vel_scale .* v_spec[:, 2])
    d_vel_noise = d_vel_scale .* xi_spec[:, 2] .+ vel_scale .* v_spec[:, 2]

    d_vel_cs = cumsum(d_vel_noise)
    d_vwz = vcat(zero(T), d_vel_cs)

    vel_noise = vel_scale .* xi_spec[:, 2]
    vel_cs = cumsum(vel_noise)
    vwz = vcat(zero(T), vel_cs)

    d_pwd = vcat(zero(T), d_pos_noise .+ dt .* d_vwz[1:n_steps])
    d_pos = cumsum(d_pwd)

    return d_pos
end

# --- Remove linear component (projection orthogonal to slope) ---

"""
    remove_slope(y, x) -> Vector{Float64}

Subtract the endpoint-anchored secant: the line through `(0, 0)` and
`(x[end], y[end])`, i.e. `y .- (y[end]/x[end]) .* x` (which zeroes the last
element). This matches NIFTy's `_remove_slope`, used to detrend the
integrated-Wiener-process spectral deviations. Linear in `y` but not self-adjoint;
its adjoint is [`remove_slope_adjoint`].

(Previously this projected out a *least-squares* line, which made the IWP spectral
deviations ~2× broader than NIFTy's; see comparison/step7.)
"""
function remove_slope(y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return y .- (y[end] / x[end]) .* x
end

"""
    remove_slope_adjoint(g, x) -> Vector{Float64}

Adjoint of `remove_slope(·, x)`. With `S y = y - (y[end]/x[end]) x`, the adjoint is
`(Sᵀ g)_j = g_j` for `j < end` and `g[end] - Σᵢ gᵢ xᵢ / x[end]` for the last entry.
"""
function remove_slope_adjoint(g::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    out = collect(float.(g))
    out[end] -= sum(g .* x) / x[end]
    return out
end

# --- Amplitude spectrum: power-law + optional IWP deviations ---

"""
    amplitude_spectrum(xi_slope, xi_fluct, xi_flex, xi_asp, xi_spectrum,
                       cfg::CorrFieldConfig, p; xi_offset=0.0) -> Vector{Float64}

Compute the radial amplitude spectrum from latent variables.
Prior hyperparameters come from `cfg`; Fourier grid data from `p`.

!!! warning "A learned amplitude is Neal's funnel"
    The NIFTy-style construction this implements — `amp = f(hyper-latents)` and
    then `y = amp ⊙ ξ` with `ξ ~ N(0,1)` — is a non-centred hierarchical model,
    i.e. Neal's funnel. `amp → c·amp, ξ → ξ/c` leaves `y` and therefore χ²
    pointwise unchanged, so only the prior term distinguishes those
    parameterizations, and MGVI's metric `JᵀJ + I` cannot see the curvature the
    funnel puts in the prior term. Measured on a 63-latent problem: χ² constant
    to four decimals while sweeping `c` from 0.7 to 3.0, the energy falling
    monotonically by 79, and the per-block latent diagnostic sliding straight
    through 1 on the way. GeoVI does not remove the drift (−67, −52, −63 across
    three schedules of increasing GeoVI content, with no ordering): it refines
    the samples about an expansion point, fixing the covariance, but it does not
    give the expansion point a unique home.

    Consequence for diagnostics: for a problem built this way the per-block
    reduced χ² of [`latent_blocks`] is a **coordinate on the funnel, not a
    calibration**, so a value far from 1 need not mean a mis-scaled prior.

    Scope: this is a property of *learned-amplitude* models specifically, not of
    VarInf or of correlated fields generally. A fixed prior covariance has no
    funnel; the funnel arrives with the hyperparameter that multiplies the
    excitation field. The same field reconstructed under a fixed prior
    covariance calibrates without any of this.

    To express the centred alternative instead — the learned spectrum in the
    prior rather than in the forward map — implement
    [`prior_inv_covariance_mul`], [`prior_inv_sqrt_covariance_mul`] and
    [`prior_energy`]; the metric then uses `JᵀJ + C⁻¹`. Note NIFTy has the same
    product (`op = ht(azm * corr * xi)`) and does not centre either.

!!! note "The grid argument is duck-typed"
    `amplitude_spectrum`, [`amplitude_spectrum_jvp`] and
    [`amplitude_spectrum_adjoint`] use only five fields of `p`: `npix`,
    `n_bins`, `mode_multiplicity`, `rel_log_mode_lengths` and `log_volume`.
    Anything supplying those works, so a new harmonic geometry needs no changes
    here. Spherical harmonics, for instance, are one bin per degree ℓ with
    multiplicity `2ℓ+1` and `rel_log_mode_lengths = log ℓ`, which matches NIFTy
    exactly (`LMSpace.get_k_length_array()` returns ℓ and `PowerSpace`'s `rho`
    is `2ℓ+1`). [`FourierGridInfo`] and [`Axis1DInfo`] are the two shipped
    implementations.

The DC bin (`amplitude[1]`) behavior depends on `cfg.use_offset`:
- `cfg.use_offset == false` (legacy): `amplitude[1] = npix²`. Appropriate when
  the caller projects the field's mean to zero downstream (e.g. via a mask).
- `cfg.use_offset == true`: `amplitude[1] = exp(offset_mean + offset_std · xi_offset)`,
  i.e. a lognormal prior on the DC amplitude. The `xi_offset` latent is
  passed by keyword so existing callers don't need to change.
"""
function amplitude_spectrum(xi_slope::Real, xi_fluct::Real,
                            xi_flex::Real, xi_asp::Real,
                            xi_spectrum::AbstractMatrix{<:Real},
                            cfg::CorrFieldConfig, p;
                            xi_offset::Real=0)
    T = eltype(cfg)
    n_bins = p.n_bins
    rel_log = p.rel_log_mode_lengths
    mult = p.mode_multiplicity
    n2 = T(p.npix)^2

    slope = cfg.slope_mean + cfg.slope_std * xi_slope
    fluct = exp(cfg.fluct_mean + cfg.fluct_std * xi_fluct)

    ln_A = slope .* rel_log

    if cfg.use_iwp
        flex = exp(cfg.flex_mean + cfg.flex_std * xi_flex)
        asp = exp(cfg.asp_mean + cfg.asp_std * xi_asp)
        iwp_out = integrated_wiener_process(xi_spectrum, flex, asp, p.log_volume)
        deviations = remove_slope(iwp_out, rel_log[2:end])
        ln_A = copy(ln_A)
        ln_A[2:end] .+= deviations
    end

    A = exp.(ln_A)

    norm_sq = sum(mult[b] * A[b]^2 for b in 2:n_bins)
    norm = sqrt(norm_sq)

    npx = T(p.npix)
    amplitude = Vector{T}(undef, n_bins)
    amplitude[1] = cfg.use_offset ?
                    exp(cfg.offset_mean + cfg.offset_std * xi_offset) :
                    n2
    for b in 2:n_bins
        amplitude[b] = fluct * npx / norm * A[b]
    end

    return amplitude
end

"""
    amplitude_spectrum(cfg, p) -> Vector{Float64}

Convenience: return the amplitude curve evaluated at all latent `xi = 0`,
i.e. at the prior means of slope/fluct (and flex/asp/spectrum if
`cfg.use_iwp`, and offset if `cfg.use_offset`). Useful when you want a
fixed power-law / Kolmogorov spectrum and aren't learning the
hyperparameters.
"""
function amplitude_spectrum(cfg::CorrFieldConfig, p)
    T = eltype(cfg)
    xi_spec = cfg.use_iwp ? zeros(T, p.n_bins - 2, 2) :
                            zeros(T, 0, 0)
    z = zero(T)
    return amplitude_spectrum(z, z, z, z, xi_spec, cfg, p)
end

"""
    amplitude_spectrum_adjoint(g_amp, xi_slope, xi_fluct, xi_flex, xi_asp,
                               xi_spectrum, cfg::CorrFieldConfig, p;
                               xi_offset=0.0)

Adjoint of `amplitude_spectrum`. Returns the 6-tuple
`(g_xi_slope, g_xi_fluct, g_xi_flex, g_xi_asp, g_xi_spectrum, g_xi_offset)`.

`g_xi_offset` is the gradient w.r.t. the DC-mode latent (only nonzero when
`cfg.use_offset == true`); legacy callers that destructured the previous
5-tuple `(g_xi_slope, g_xi_fluct, g_xi_flex, g_xi_asp, g_xi_spectrum)`
should add a trailing `_` (or `g_xi_offset`) to absorb the new entry.
"""
function amplitude_spectrum_adjoint(g_amp::AbstractVector{<:Real},
                                    xi_slope::Real, xi_fluct::Real,
                                    xi_flex::Real, xi_asp::Real,
                                    xi_spectrum::AbstractMatrix{<:Real},
                                    cfg::CorrFieldConfig, p;
                                    xi_offset::Real=0)
    T = eltype(cfg)
    n_bins = p.n_bins
    rel_log = p.rel_log_mode_lengths
    mult = p.mode_multiplicity
    npx = T(p.npix)

    slope = cfg.slope_mean + cfg.slope_std * xi_slope
    fluct = exp(cfg.fluct_mean + cfg.fluct_std * xi_fluct)

    ln_A = slope .* rel_log

    flex = zero(T)
    asp = zero(T)
    if cfg.use_iwp
        flex = exp(cfg.flex_mean + cfg.flex_std * xi_flex)
        asp = exp(cfg.asp_mean + cfg.asp_std * xi_asp)
        iwp_out = integrated_wiener_process(xi_spectrum, flex, asp, p.log_volume)
        deviations = remove_slope(iwp_out, rel_log[2:end])
        ln_A = copy(ln_A)
        ln_A[2:end] .+= deviations
    end

    A = exp.(ln_A)
    norm_sq = sum(mult[b] * A[b]^2 for b in 2:n_bins)
    norm = sqrt(norm_sq)
    c = fluct * npx / norm

    # --- Backward ---
    g_fluct_partial = zero(T)
    for b in 2:n_bins
        g_fluct_partial += g_amp[b] * npx / norm * A[b]
    end

    g_norm = zero(T)
    for b in 2:n_bins
        g_norm -= g_amp[b] * c * A[b] / norm
    end

    g_norm_sq = g_norm / (2 * norm)

    g_A = zeros(T, n_bins)
    for b in 2:n_bins
        g_A[b] = g_amp[b] * c + g_norm_sq * 2 * mult[b] * A[b]
    end

    g_ln_A = g_A .* A

    g_slope = sum(g_ln_A .* rel_log)
    g_xi_slope = g_slope * cfg.slope_std
    g_xi_fluct = g_fluct_partial * fluct * cfg.fluct_std

    g_xi_flex = zero(T)
    g_xi_asp = zero(T)
    g_xi_spectrum = zeros(T, 0, 2)

    if cfg.use_iwp
        g_deviations = g_ln_A[2:end]
        g_iwp_out = remove_slope_adjoint(g_deviations, rel_log[2:end])

        g_xi_spec_raw, g_flex, g_asp = iwp_adjoint(
            g_iwp_out, xi_spectrum, flex, asp, p.log_volume)
        g_xi_spectrum = g_xi_spec_raw

        g_xi_flex = g_flex * flex * cfg.flex_std
        g_xi_asp = g_asp * asp * cfg.asp_std
    end

    # Offset latent: amp[1] = exp(offset_mean + offset_std·xi_offset) when use_offset,
    # else amp[1] is a constant (npix²) ⇒ no latent dependence.
    g_xi_offset = if cfg.use_offset
        offset_val = exp(cfg.offset_mean + cfg.offset_std * xi_offset)
        g_amp[1] * offset_val * cfg.offset_std
    else
        zero(T)
    end

    return g_xi_slope, g_xi_fluct, g_xi_flex, g_xi_asp, g_xi_spectrum, g_xi_offset
end

"""
    amplitude_spectrum_jvp(v_slope, v_fluct, v_flex, v_asp, v_spectrum,
                           xi_slope, xi_fluct, xi_flex, xi_asp, xi_spectrum,
                           cfg::CorrFieldConfig, p;
                           xi_offset=0.0, v_offset=0.0) -> d_amp

JVP of `amplitude_spectrum`. The `xi_offset` / `v_offset` kwargs are only
meaningful when `cfg.use_offset == true`; otherwise the DC bin is a constant
and `d_amp[1] = 0`.
"""
function amplitude_spectrum_jvp(v_slope::Real, v_fluct::Real,
                                v_flex::Real, v_asp::Real,
                                v_spectrum::AbstractMatrix{<:Real},
                                xi_slope::Real, xi_fluct::Real,
                                xi_flex::Real, xi_asp::Real,
                                xi_spectrum::AbstractMatrix{<:Real},
                                cfg::CorrFieldConfig, p;
                                xi_offset::Real=0, v_offset::Real=0)
    T = eltype(cfg)
    n_bins = p.n_bins
    rel_log = p.rel_log_mode_lengths
    mult = p.mode_multiplicity
    npx = T(p.npix)

    slope = cfg.slope_mean + cfg.slope_std * xi_slope
    fluct = exp(cfg.fluct_mean + cfg.fluct_std * xi_fluct)
    d_slope = cfg.slope_std * v_slope
    d_fluct = fluct * cfg.fluct_std * v_fluct

    ln_A = slope .* rel_log
    d_ln_A = d_slope .* rel_log

    if cfg.use_iwp
        flex = exp(cfg.flex_mean + cfg.flex_std * xi_flex)
        asp = exp(cfg.asp_mean + cfg.asp_std * xi_asp)
        d_flex = flex * cfg.flex_std * v_flex
        d_asp = asp * cfg.asp_std * v_asp

        iwp_out = integrated_wiener_process(xi_spectrum, flex, asp, p.log_volume)
        d_iwp_out = iwp_jvp(v_spectrum, d_flex, d_asp, xi_spectrum, flex, asp, p.log_volume)

        deviations = remove_slope(iwp_out, rel_log[2:end])
        d_deviations = remove_slope(d_iwp_out, rel_log[2:end])

        ln_A = copy(ln_A)
        ln_A[2:end] .+= deviations
        d_ln_A = copy(d_ln_A)
        d_ln_A[2:end] .+= d_deviations
    end

    A = exp.(ln_A)
    d_A = A .* d_ln_A

    S = sum(mult[b] * A[b]^2 for b in 2:n_bins)
    d_S = sum(mult[b] * 2 * A[b] * d_A[b] for b in 2:n_bins)
    norm = sqrt(S)
    d_norm = d_S / (2 * norm)

    d_amp = Vector{T}(undef, n_bins)
    d_amp[1] = cfg.use_offset ?
                exp(cfg.offset_mean + cfg.offset_std * xi_offset) * cfg.offset_std * v_offset :
                zero(T)
    for b in 2:n_bins
        d_amp[b] = d_fluct * npx / norm * A[b] +
                   fluct * npx / norm * d_A[b] -
                   fluct * npx * A[b] / norm^2 * d_norm
    end

    return d_amp
end

# ============================================================================
# Matérn-kernel amplitude (port of NIFTy.re `matern_amplitude`)
# ============================================================================
#
# Ported from nifty/src/re/correlated_field.py: `matern_amplitude` (the
# `correlate` closure) and the prior wiring in
# `CorrelatedFieldMaker.add_fluctuations_matern`.
#
# NIFTy's `add_fluctuations_matern` DOCSTRING swaps `cutoff` and `loglogslope`:
# it describes `cutoff` as "Power law component exponent (a priori normal)" and
# `loglogslope` as "Amplitude of the non-power-law component (a priori
# log-normal)". The CODE does the opposite — `cutoff = lognormal_prior(*cutoff)`
# and `loglogslope = normal_prior(*loglogslope)` — and the code is right on both
# counts: a cutoff mode is a positive length scale, a spectral index may be
# negative. This port follows the code.
#
# `renormalize_amplitude` is deliberately NOT ported. Upstream emits
# `logger.warning("Renormalize amplidude is not yet tested!")` for that branch,
# and its normalization couples all three hyperparameter derivatives, so porting
# it would mean carrying untested upstream behaviour through our own JVP and
# adjoint. The `norm = 1` path (the default) is what is implemented here.

"""
    MaternConfig

Prior hyperparameters for a Matérn-kernel amplitude. Three latents map to

- `scale`       — **log-normal** prior, value-space `(mean, std)`
- `cutoff`      — **log-normal** prior, value-space `(mean, std)`
- `loglogslope` — **normal** prior, `(mean, std)` used directly

matching NIFTy's code (see the note above about its docstring). The spectrum is

```
ln A(k) = 0.25 · loglogslope · log1p((k / cutoff)²)
A(k)    = scale · √total_volume · exp(ln A(k))
A(k₀)   = total_volume                            (zero mode, set exactly)
```

and for `kind = :power` the whole vector is square-rooted afterwards, so the
Matérn kernel then describes the *power* spectrum rather than the amplitude.
"""
struct MaternConfig{T<:AbstractFloat}
    scale_mean::T       # log-space (μ, σ) of the lognormal scale prior
    scale_std::T
    cutoff_mean::T      # log-space (μ, σ) of the lognormal cutoff prior
    cutoff_std::T
    slope_mean::T       # value-space (mean, std) of the NORMAL loglogslope prior
    slope_std::T
    kind_power::Bool    # false = :amplitude, true = :power
end

Base.eltype(::MaternConfig{T}) where {T} = T

"""
    MaternConfig(; scale_prior, cutoff_prior, loglogslope_prior, kind=:amplitude)

`scale_prior` and `cutoff_prior` are value-space `(mean, std)` log-normal priors
(converted with the same `σ² = log1p((std/mean)²)`, `μ = log(mean) − σ²/2` as
[`CorrFieldConfig`]); `loglogslope_prior` is a `(mean, std)` normal prior used
directly. `kind` is `:amplitude` or `:power`.
"""
function MaternConfig(; scale_prior, cutoff_prior, loglogslope_prior,
                      kind::Symbol=:amplitude)
    kind in (:amplitude, :power) ||
        throw(ArgumentError("kind must be :amplitude or :power, got :$kind"))
    T = _prior_eltype(scale_prior, cutoff_prior, loglogslope_prior)
    s_m, s_s = _lognormal_params(T(scale_prior[1]), T(scale_prior[2]))
    c_m, c_s = _lognormal_params(T(cutoff_prior[1]), T(cutoff_prior[2]))
    return MaternConfig{T}(s_m, s_s, c_m, c_s,
                           T(loglogslope_prior[1]), T(loglogslope_prior[2]),
                           kind === :power)
end

# u = log1p(r²) and w = r²/(1+r²), both overflow-safe in `r = k/cutoff`.
# The direct forms are used wherever they are accurate (log1p is the better
# choice for r ≪ 1, and is what NIFTy evaluates); above the point where r²
# would overflow, log1p(r²) → 2·log(r) and w → 1 to full relative accuracy.
function _matern_terms(r::T) where {T<:AbstractFloat}
    isfinite(r) || return (T(Inf), one(T))
    r > sqrt(floatmax(T)) / 2 && return (2 * log(r), one(T))
    r2 = r * r
    return (log1p(r2), r2 / (1 + r2))
end

# Shared core, so value / JVP / adjoint cannot drift apart. Returns the
# pre-zero-mode, pre-sqrt amplitude together with the per-bin terms the
# derivatives need.
function _matern_core(xi_scale::Real, xi_cutoff::Real, xi_slope::Real,
                      cfg::MaternConfig{T}, p) where {T<:AbstractFloat}
    scale  = exp(cfg.scale_mean + cfg.scale_std * xi_scale)
    cutoff = exp(cfg.cutoff_mean + cfg.cutoff_std * xi_cutoff)
    slope  = cfg.slope_mean + cfg.slope_std * xi_slope
    k      = p.mode_lengths
    pref   = scale * sqrt(T(p.total_volume))

    n = length(k)
    A = Vector{T}(undef, n)
    u = Vector{T}(undef, n)
    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        u[i], w[i] = _matern_terms(T(k[i]) / cutoff)
        A[i] = pref * exp(T(0.25) * slope * u[i])
    end
    return A, u, w, slope
end

"""
    matern_amplitude(xi_scale, xi_cutoff, xi_slope, cfg::MaternConfig, p) -> Vector

Matérn amplitude spectrum over the bins of grid `p`, which must supply
`mode_lengths` and `total_volume` in addition to the fields
[`amplitude_spectrum`] needs.

The zero mode `A[1]` is set to `total_volume` exactly (NIFTy's
`spectrum.at[0].set(totvol)`), so it carries no dependence on the three
hyperparameter latents; `matern_amplitude_jvp` returns 0 there and
`matern_amplitude_adjoint` ignores `g_amp[1]`.
"""
function matern_amplitude(xi_scale::Real, xi_cutoff::Real, xi_slope::Real,
                          cfg::MaternConfig{T}, p) where {T<:AbstractFloat}
    A, = _matern_core(xi_scale, xi_cutoff, xi_slope, cfg, p)
    A[1] = T(p.total_volume)
    cfg.kind_power && (A .= sqrt.(A))
    return A
end

"""
    matern_amplitude_jvp(v_scale, v_cutoff, v_slope,
                         xi_scale, xi_cutoff, xi_slope, cfg, p) -> Vector

Directional derivative of [`matern_amplitude`] w.r.t. the three hyperparameter
latents, in direction `(v_scale, v_cutoff, v_slope)`.

With `u = log1p((k/cutoff)²)` and `w = (k/cutoff)²/(1 + (k/cutoff)²)`:

```
dA/A = σ_scale·v_scale + 0.25·u·σ_slope·v_slope − 0.5·slope·w·σ_cutoff·v_cutoff
```

the last term carrying the minus sign because raising the cutoff lowers `k/cutoff`.
"""
function matern_amplitude_jvp(v_scale::Real, v_cutoff::Real, v_slope::Real,
                              xi_scale::Real, xi_cutoff::Real, xi_slope::Real,
                              cfg::MaternConfig{T}, p) where {T<:AbstractFloat}
    A, u, w, slope = _matern_core(xi_scale, xi_cutoff, xi_slope, cfg, p)
    n = length(A)
    dA = Vector{T}(undef, n)
    @inbounds for i in 1:n
        dlog = cfg.scale_std * v_scale +
               T(0.25) * u[i] * cfg.slope_std * v_slope -
               T(0.5) * slope * w[i] * cfg.cutoff_std * v_cutoff
        dA[i] = A[i] * dlog
    end
    if cfg.kind_power
        # d(√A) = dA / (2√A), evaluated on the pre-override A.
        @inbounds for i in 2:n
            dA[i] /= 2 * sqrt(A[i])
        end
    end
    dA[1] = zero(T)        # zero mode is a constant
    return dA
end

"""
    matern_amplitude_adjoint(g_amp, xi_scale, xi_cutoff, xi_slope, cfg, p)
        -> (g_xi_scale, g_xi_cutoff, g_xi_slope)

Adjoint of [`matern_amplitude_jvp`]: pulls a gradient w.r.t. the amplitude
vector back to the three hyperparameter latents. `g_amp[1]` is ignored, since
the zero mode is a constant.
"""
function matern_amplitude_adjoint(g_amp::AbstractVector{<:Real},
                                  xi_scale::Real, xi_cutoff::Real, xi_slope::Real,
                                  cfg::MaternConfig{T}, p) where {T<:AbstractFloat}
    A, u, w, slope = _matern_core(xi_scale, xi_cutoff, xi_slope, cfg, p)
    g_scale  = zero(T)
    g_cutoff = zero(T)
    g_slope  = zero(T)
    @inbounds for i in 2:length(A)        # skip the constant zero mode
        gA = T(g_amp[i])
        cfg.kind_power && (gA /= 2 * sqrt(A[i]))
        gA *= A[i]
        g_scale  += gA * cfg.scale_std
        g_slope  += gA * T(0.25) * u[i] * cfg.slope_std
        g_cutoff -= gA * T(0.5) * slope * w[i] * cfg.cutoff_std
    end
    return (g_scale, g_cutoff, g_slope)
end
