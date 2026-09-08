# ============================================================================
# CorrelatedField: a multi-axis correlated-field prior.
#
# Composes N independent axes, each carrying its own `CorrFieldConfig`
# (slope / fluct / IWP) and its own Fourier grid (a `FourierGridInfo` for
# 2-D-radial axes such as spatial (npix, npix); an `Axis1DInfo` for 1-D
# linear axes such as wavelength). The total Fourier kernel is the outer
# product of the per-axis amplitude vectors evaluated at each axis's
# `bin_index`. A single global lognormal offset prior (`offset_prior`)
# controls the DC mode of the *composite* kernel; per-axis cfgs must
# leave `use_offset = false`.
#
# Latent layout (flat vector z):
#   z[1 : N_total]                       xi_field (the joint white-noise
#                                                   tensor, flattened in
#                                                   the order of axes)
#   for each axis i in 1..n_axes:
#     z[N_total + sum(m_j for j<i) + 1 .. N_total + sum(m_j for j<=i)]
#                                        per-axis hypers (slope, fluct,
#                                                          [flex, asp, xi_spec])
#   z[end]                               xi_offset (if use_offset)
#
# where N_total = prod(field_shape) and m_i = _spectral_latent_size(cfgs[i],
# grids[i].n_bins).
# ============================================================================

# --- Internal helpers ------------------------------------------------------

_axis_dims(g::FourierGridInfo) = (g.npix, g.npix)
_axis_dims(g::Axis1DInfo)      = (g.npix,)
_axis_ndims(g::FourierGridInfo) = 2
_axis_ndims(g::Axis1DInfo)      = 1

# How many latent entries each axis contributes (slope + fluct + IWP block).
function _axis_latent_size(cfg::CorrFieldConfig, g)
    @assert !cfg.use_offset "per-axis cfg must have use_offset=false; use the global offset_prior on CorrelatedField"
    return _spectral_latent_size(cfg, g.n_bins)
end

"""
    CorrelatedField

Multi-axis correlated-field prior. Build via

    CorrelatedField(grids::Vector, cfgs::Vector{CorrFieldConfig};
                    offset_prior = (offset_mean, azm)             # azm pinned
                                or (offset_mean, azm_mean, azm_std)) -> CorrelatedField

where `grids[i]` is either a `FourierGridInfo` (for a multi-dim
radially-binned axis group) or an `Axis1DInfo` (for a 1-D axis), and
`cfgs[i]` is the corresponding `CorrFieldConfig`. The composite field has
shape `(npix_i,...)` concatenated across the axes (in order).

Call the result on a latent vector to get the field:
    mcf = CorrelatedField([spatial, spectral], [sp_cfg, sc_cfg];
                          offset_prior=(0.0, 0.1))
    field = mcf(z)

# Offset semantics (NIFTy `set_amplitude_total_offset`)
`offset_prior` carries an **additive scalar** `offset_mean` plus a **lognormal
zero-mode amplitude** `azm` (NIFTy's `offset_std`):
- `(offset_mean, azm)`: `azm` pinned at that value.
- `(offset_mean, azm_mean, azm_std)`: `azm ~ LogNormal(azm_mean, azm_std)`
  (value-space), with `xi_azm` the last latent.

The field is `field = offset_mean + HT(amp · fft(ξ))` where the zero-mode of the
amplitude kernel is `amp[DC] = azm·√P` (P = #pixels). So the DC fluctuation is
`azm·N(0,1)` (driven by the field's own DC component) and the mean level is
`offset_mean`, exactly NIFTy's `offset_mean + azm·xi[DC]`. The non-DC
fluctuations are unaffected by `azm` (set by per-axis `fluctuations`). For a
lognormal field `exp(mcf(z))`, set `offset_mean = log(prior-mean level)`.

!!! note "Omitting `offset_prior`"
    With `offset_prior=nothing` there is no additive offset and the composite
    DC of the fluctuation kernel stays at 1 (each per-axis DC is normalised to
    1 by `_normalize_axis_dc`), i.e. a small zero-mean random DC mode. Pass
    `offset_prior` whenever you need to control/center the field's level.
    (OIVI's `sky_model.jl` uses `offset_prior=nothing` and projects out the
    field mean via a separate masking step.)
"""
const AxisGrid = Union{FourierGridInfo, Axis1DInfo}

struct CorrelatedField{T<:AbstractFloat,TP,TIP,TB<:AbstractArray{Complex{T}}}
    grids::Vector{AxisGrid}             # FourierGridInfo or Axis1DInfo entries
    cfgs::Vector{CorrFieldConfig{T}}
    use_offset::Bool
    offset_mean::T                 # additive scalar DC level (NIFTy offset_mean)
    azm_logmean::T                 # zero-mode amplitude lognormal: μ for azm = exp(μ + σ·ξ)
    azm_logstd::T                  # zero-mode amplitude lognormal: σ (NIFTy offset_std/azm)
    field_shape::Tuple{Vararg{Int}}
    n_axes::Int
    # Cached out-of-place FFT plans over the full field + three complex scratch
    # buffers, so the forward/JVP/VJP route their transient transforms through
    # `mul!(scratch, plan, ·)` and allocate no FFT outputs. Three buffers because
    # the JVP/VJP hold two spectra at once (F_xi and F_v/F_w). Safe because
    # reconstruct_* runs single-threaded; OIVI does not use this struct.
    P::TP                              # plan_fft  over field_shape (out-of-place)
    iP::TIP                            # plan_ifft over field_shape (carries 1/N)
    cs1::TB
    cs2::TB
    cs3::TB
end

Base.eltype(::CorrelatedField{T}) where {T} = T

function CorrelatedField(grids::AbstractVector, cfgs::AbstractVector{<:CorrFieldConfig};
                          offset_prior::Union{Nothing,Tuple{Real,Real},
                                              Tuple{Real,Real,Real}}=nothing)
    T = eltype(cfgs[1])
    @assert length(grids) == length(cfgs) "grids and cfgs must have equal length"
    @assert !isempty(grids) "must have at least one axis"
    for (i, (cfg, g)) in enumerate(zip(cfgs, grids))
        @assert !cfg.use_offset "per-axis cfg $i must have use_offset=false; pass offset_prior to CorrelatedField itself"
        # IWP needs at least one inter-bin step, i.e. n_bins ≥ 3 ⇒ log_volume
        # nonempty. NIFTy silently disables IWP in this case; we error out so
        # the user's latent layout (which counts on use_iwp at problem-build
        # time) doesn't desynchronize from what amplitude_spectrum produces.
        if cfg.use_iwp && length(g.log_volume) == 0
            error("axis $i has n_bins = $(g.n_bins) < 3, which cannot support " *
                  "the integrated Wiener process (log_volume is empty). " *
                  "Either rebuild cfg without `flex_prior`/`asp_prior` for " *
                  "this axis, or use a finer grid.")
        end
    end
    # offset_prior: (offset_mean, azm_value)               → azm pinned at azm_value
    #            or (offset_mean, azm_mean, azm_std)        → azm ~ LogNormal(azm_mean, azm_std)
    # offset_mean is the additive DC level; azm is the lognormal zero-mode amplitude
    # (NIFTy's offset_std). Stored as the underlying-normal (μ, σ).
    use_offset = offset_prior !== nothing
    offset_mean = use_offset ? T(offset_prior[1]) : zero(T)
    azm_logmean, azm_logstd =
        if !use_offset
            (zero(T), zero(T))
        elseif length(offset_prior) == 2
            _lognormal_params(T(offset_prior[2]), zero(T))        # pinned azm
        else
            _lognormal_params(T(offset_prior[2]), T(offset_prior[3]))
        end
    # Concatenate dims of each axis into the field shape
    dims_list = Int[]
    for g in grids
        append!(dims_list, _axis_dims(g))
    end
    field_shape = Tuple(dims_list)
    # MEASURE plans: 2-7× faster transforms than ESTIMATE. Built once here and
    # cached, so the one-time planning cost amortizes over the whole run. Applied
    # via mul! to the cs1/cs2/cs3 scratch buffers (same size/alignment, so valid).
    buf = zeros(Complex{T}, field_shape)
    P   = plan_fft(buf;  flags=FFTW.MEASURE)
    iP  = plan_ifft(buf; flags=FFTW.MEASURE)
    return CorrelatedField(Vector{AxisGrid}(grids), Vector{CorrFieldConfig{T}}(cfgs),
                            use_offset, offset_mean, azm_logmean, azm_logstd,
                            field_shape, length(grids),
                            P, iP, zeros(Complex{T}, field_shape),
                            zeros(Complex{T}, field_shape),
                            zeros(Complex{T}, field_shape))
end

"""
    latent_size(mcf::CorrelatedField) -> Int
"""
function latent_size(mcf::CorrelatedField)
    n = prod(mcf.field_shape)
    for (cfg, g) in zip(mcf.cfgs, mcf.grids)
        n += _axis_latent_size(cfg, g)
    end
    if mcf.use_offset
        n += 1
    end
    return n
end

# Per-axis offset within the latent at which axis i's hypers start.
function _axis_offset(mcf::CorrelatedField, i::Int)
    off = prod(mcf.field_shape)
    for j in 1:i-1
        off += _axis_latent_size(mcf.cfgs[j], mcf.grids[j])
    end
    return off
end

# Unpack the per-axis (slope, fluct, flex, asp, xi_spectrum) tuple from a
# slice of length `_axis_latent_size(cfg, g)`.
function _unpack_axis(z_slice::AbstractVector{<:Real}, cfg::CorrFieldConfig,
                      g)
    Tz = eltype(z_slice)
    xi_slope = z_slice[1]
    xi_fluct = z_slice[2]
    if cfg.use_iwp
        xi_flex = z_slice[3]
        xi_asp  = z_slice[4]
        n_spec = 2 * (g.n_bins - 2)
        xi_spectrum = reshape(z_slice[5 : 4 + n_spec], g.n_bins - 2, 2)
    else
        xi_flex = zero(Tz)
        xi_asp  = zero(Tz)
        xi_spectrum = zeros(Tz, 0, 2)
    end
    return xi_slope, xi_fluct, xi_flex, xi_asp, xi_spectrum
end

"""
    latent_unpack(mcf, z) -> (xi_field, per_axis::Vector{NamedTuple}, xi_offset)

Note: `per_axis` has an abstract element type. This cannot be tightened in
isolation: `mcf.grids` has the abstract element type `AxisGrid`, so `grids[i]`
is abstract and the per-axis `NamedTuple` field types (e.g. `xi_spectrum`'s
shape via `g.n_bins`) are not inferable here regardless of the container. Making
the forward/JVP type-stable would require parameterising `CorrelatedField` on a
*tuple* of concrete grid/cfg types and iterating axes with `map`/`ntuple`; the
runtime payoff is negligible (axes are few, FFTs dominate), so it is left as-is.
"""
function latent_unpack(mcf::CorrelatedField, z::AbstractVector{<:Real})
    N = prod(mcf.field_shape)
    xi_field = reshape(z[1:N], mcf.field_shape)
    per_axis = Vector{NamedTuple}(undef, mcf.n_axes)
    for i in 1:mcf.n_axes
        off = _axis_offset(mcf, i)
        m_i = _axis_latent_size(mcf.cfgs[i], mcf.grids[i])
        z_slice = z[off + 1 : off + m_i]
        xi_slope, xi_fluct, xi_flex, xi_asp, xi_spectrum =
            _unpack_axis(z_slice, mcf.cfgs[i], mcf.grids[i])
        per_axis[i] = (; xi_slope, xi_fluct, xi_flex, xi_asp, xi_spectrum)
    end
    xi_offset = mcf.use_offset ? z[end] : zero(eltype(z))
    return Array(xi_field), per_axis, xi_offset
end

# Reshape `piece` (an array whose shape matches axis i's dims) to broadcast
# into the field's full shape. Inserts singleton dims for other axes' dims.
function _reshape_for_axis(piece::AbstractArray, mcf::CorrelatedField, i::Int)
    target = ones(Int, length(mcf.field_shape))
    cumulative = 0
    for j in 1:mcf.n_axes
        nd = _axis_ndims(mcf.grids[j])
        if j == i
            for k in 1:nd
                target[cumulative + k] = size(piece, k)
            end
        end
        cumulative += nd
    end
    return reshape(piece, target...)
end

# Make a copy of the per-axis amp with amp[1] := 1, so the outer-product
# kernel has unit amplitude along each axis's DC slice. Without this
# substitution, the legacy `amplitude_spectrum` convention `amp[1] = npix²`
# bleeds through to every marginal-DC mode (e.g. (k_spatial=0, k_freq≠0))
# and the multi-axis kernel blows up by a factor `prod(npix_i^2)` at those
# modes, making `exp(field)` astronomical.
function _normalize_axis_dc(amp_per_axis::AbstractVector{<:AbstractVector})
    out = similar(amp_per_axis)
    for i in eachindex(amp_per_axis)
        a = copy(amp_per_axis[i])
        a[1] = one(eltype(a))
        out[i] = a
    end
    return out
end

"""
    _build_amp_kernel(amp_per_axis, mcf, xi_offset) -> Array{Float64}

Build the multi-D amplitude kernel as the outer product of per-axis
amplitude vectors indexed via each axis's `bin_index`. Each axis's DC
bin is normalised to 1 first (see `_normalize_axis_dc`). The composite
DC mode is then 1 (when `mcf.use_offset == false`) or, with `use_offset`,
the lognormal zero-mode amplitude `azm·√P` (azm = `exp(azm_logmean +
azm_logstd·xi_offset)`), so `xi_offset` is used here (it drives azm); the
additive `offset_mean` is added separately in the forward.
"""
function _build_amp_kernel(amp_per_axis::AbstractVector{<:AbstractVector},
                            mcf::CorrelatedField, xi_offset::Real)
    T = eltype(mcf)
    a = _normalize_axis_dc(amp_per_axis)
    pieces = [a[i][mcf.grids[i].bin_index] for i in 1:mcf.n_axes]
    amp_kernel = _reshape_for_axis(pieces[1], mcf, 1)
    for i in 2:mcf.n_axes
        amp_kernel = amp_kernel .* _reshape_for_axis(pieces[i], mcf, i)
    end
    amp_kernel = Array(amp_kernel)
    if mcf.use_offset
        # Zero-mode amplitude = azm·√P (azm = lognormal from the offset latent), so the
        # field DC fluctuation is azm·N(0,1) (matches NIFTy's azm·xi[DC]); the additive
        # offset_mean is added in the forward. (NIFTy's azm divides out of the non-DC
        # bins and survives only at the DC, so it lives entirely in this one entry.)
        dc_idx = CartesianIndex(ntuple(_ -> 1, length(mcf.field_shape))...)
        azm = exp(mcf.azm_logmean + mcf.azm_logstd * xi_offset)
        amp_kernel[dc_idx] = azm * sqrt(T(prod(mcf.field_shape)))
    end
    return amp_kernel
end

# Same idea, but builds the JVP tangent of `amp_kernel` via the product rule:
# d_amp_kernel = Σ_i (outer product where axis i is replaced by its tangent).
# Per-axis amp[1] is treated as a constant (1, after _normalize_axis_dc) ⇒
# d_amp[1] = 0 from amplitude_spectrum_jvp already, no override needed.
function _build_d_amp_kernel(amp_per_axis, d_amp_per_axis,
                              mcf::CorrelatedField,
                              xi_offset::Real, v_offset::Real)
    T = eltype(mcf)
    a = _normalize_axis_dc(amp_per_axis)
    pieces = [a[i][mcf.grids[i].bin_index] for i in 1:mcf.n_axes]
    d_pieces = [d_amp_per_axis[i][mcf.grids[i].bin_index] for i in 1:mcf.n_axes]
    d_amp_kernel = zeros(T, mcf.field_shape)
    for i in 1:mcf.n_axes
        # Term i: derivative of axis i, identity on others
        term = _reshape_for_axis(i == 1 ? d_pieces[1] : pieces[1], mcf, 1)
        for j in 2:mcf.n_axes
            piece_j = (j == i) ? d_pieces[j] : pieces[j]
            term = term .* _reshape_for_axis(piece_j, mcf, j)
        end
        d_amp_kernel .+= term
    end
    if mcf.use_offset
        # d(azm·√P)/dξ_azm = azm·σ·√P (chain rule of exp); the DC tangent flows
        # through the ifft product rule, not as a separate additive term.
        dc_idx = CartesianIndex(ntuple(_ -> 1, length(mcf.field_shape))...)
        azm = exp(mcf.azm_logmean + mcf.azm_logstd * xi_offset)
        d_amp_kernel[dc_idx] = azm * mcf.azm_logstd * v_offset * sqrt(T(prod(mcf.field_shape)))
    end
    return d_amp_kernel
end

# --- Callable forward -------------------------------------------------------

"""
    (mcf::CorrelatedField)(z) -> Array{Float64}

Forward: apply the multi-axis correlated-field operator to latent `z`.
Returns the field of shape `mcf.field_shape`. With `use_offset`, the additive
scalar `offset_mean` is added to every pixel and the zero-mode amplitude is
`azm·√P` (NIFTy `set_amplitude_total_offset`; see the type docstring).
"""
function (mcf::CorrelatedField)(z::AbstractVector{<:Real})
    xi_field, per_axis, xi_offset = latent_unpack(mcf, z)
    amp_per_axis = [amplitude_spectrum(per_axis[i].xi_slope, per_axis[i].xi_fluct,
                                        per_axis[i].xi_flex, per_axis[i].xi_asp,
                                        per_axis[i].xi_spectrum,
                                        mcf.cfgs[i], mcf.grids[i])
                    for i in 1:mcf.n_axes]
    amp_kernel = _build_amp_kernel(amp_per_axis, mcf, xi_offset)
    # field = real(ifft(fft(ξ) .* amp_kernel)); transient transforms via scratch
    mcf.cs1 .= xi_field
    mul!(mcf.cs2, mcf.P, mcf.cs1)          # F = fft(ξ)
    mcf.cs2 .*= amp_kernel
    mul!(mcf.cs1, mcf.iP, mcf.cs2)         # ifft(F .* amp_kernel)
    field = real.(mcf.cs1)
    if mcf.use_offset
        field .+= mcf.offset_mean   # additive DC level; DC fluctuation is in amp_kernel[DC]
    end
    return field
end

# --- JVP --------------------------------------------------------------------

"""
    mcf_jvp(mcf, z, v) -> Array{Float64}

JVP of the forward operator at `z` in the direction `v`. Returns the field-
shaped tangent.
"""
function mcf_jvp(mcf::CorrelatedField, z::AbstractVector{<:Real},
                  v::AbstractVector{<:Real})
    xi_field, per_axis, xi_offset = latent_unpack(mcf, z)
    v_field,  v_axis,   v_offset  = latent_unpack(mcf, v)
    amp_per_axis = [amplitude_spectrum(per_axis[i].xi_slope, per_axis[i].xi_fluct,
                                        per_axis[i].xi_flex, per_axis[i].xi_asp,
                                        per_axis[i].xi_spectrum,
                                        mcf.cfgs[i], mcf.grids[i])
                    for i in 1:mcf.n_axes]
    d_amp_per_axis = [amplitude_spectrum_jvp(
                        v_axis[i].xi_slope, v_axis[i].xi_fluct,
                        v_axis[i].xi_flex,  v_axis[i].xi_asp,
                        v_axis[i].xi_spectrum,
                        per_axis[i].xi_slope, per_axis[i].xi_fluct,
                        per_axis[i].xi_flex,  per_axis[i].xi_asp,
                        per_axis[i].xi_spectrum,
                        mcf.cfgs[i], mcf.grids[i])
                       for i in 1:mcf.n_axes]
    amp_kernel = _build_amp_kernel(amp_per_axis, mcf, xi_offset)
    d_amp_kernel = _build_d_amp_kernel(amp_per_axis, d_amp_per_axis, mcf,
                                        xi_offset, v_offset)
    # d_field = real(ifft(F_v·amp_kernel + F_xi·d_amp_kernel)); via scratch
    mcf.cs1 .= xi_field
    mul!(mcf.cs2, mcf.P, mcf.cs1)          # F_xi → cs2
    mcf.cs1 .= v_field
    mul!(mcf.cs3, mcf.P, mcf.cs1)          # F_v  → cs3
    @. mcf.cs3 = mcf.cs3 * amp_kernel + mcf.cs2 * d_amp_kernel
    mul!(mcf.cs1, mcf.iP, mcf.cs3)
    d_field = real.(mcf.cs1)
    # offset_mean is constant (no tangent); the azm/DC tangent rode through
    # d_amp_kernel[DC] in the ifft product rule above.
    return d_field
end

# --- VJP --------------------------------------------------------------------

"""
    mcf_vjp(mcf, z, w) -> Vector{Float64}

VJP of the forward operator at `z` against cotangent `w` (field-shaped or
flattened). Returns a latent-shaped gradient vector.
"""
function mcf_vjp(mcf::CorrelatedField, z::AbstractVector{<:Real},
                  w::AbstractArray{<:Real})
    T = eltype(mcf)
    xi_field, per_axis, xi_offset = latent_unpack(mcf, z)
    amp_per_axis = [amplitude_spectrum(per_axis[i].xi_slope, per_axis[i].xi_fluct,
                                        per_axis[i].xi_flex, per_axis[i].xi_asp,
                                        per_axis[i].xi_spectrum,
                                        mcf.cfgs[i], mcf.grids[i])
                    for i in 1:mcf.n_axes]
    amp_kernel = _build_amp_kernel(amp_per_axis, mcf, xi_offset)
    mcf.cs1 .= xi_field
    mul!(mcf.cs2, mcf.P, mcf.cs1)          # F_xi → cs2 (kept for g_amp_kernel)

    # w may be flattened or already field-shaped
    w_field = reshape(Array{T}(w), mcf.field_shape)

    # Adjoint of `field = real(ifft(F_xi · amp_kernel))`:
    #   g_xi_field = real(ifft(fft(w) · amp_kernel))   (CF op self-adjoint for real amp)
    #   g_amp_kernel = real(F_xi · conj(fft(w))) / N
    N_total = prod(mcf.field_shape)
    mcf.cs1 .= w_field
    mul!(mcf.cs3, mcf.P, mcf.cs1)          # F_w → cs3
    # g_amp_kernel needs both F_xi (cs2) and F_w (cs3); compute before clobbering cs3
    g_amp_kernel = real.(mcf.cs2 .* conj.(mcf.cs3)) ./ N_total
    mcf.cs3 .*= amp_kernel                  # now F_w · amp_kernel
    mul!(mcf.cs1, mcf.iP, mcf.cs3)
    g_xi_field = real.(mcf.cs1)

    # Zero-mode amplitude amp_kernel[DC] = azm·√P, azm = exp(μ + σ·ξ_azm), so the
    # DC kernel cotangent maps to ξ_azm via ∂amp_kernel[DC]/∂ξ_azm = azm·σ·√P.
    dc_idx = CartesianIndex(ntuple(_ -> 1, length(mcf.field_shape))...)
    g_xi_offset = zero(T)
    if mcf.use_offset
        azm = exp(mcf.azm_logmean + mcf.azm_logstd * xi_offset)
        g_xi_offset = g_amp_kernel[dc_idx] * azm * mcf.azm_logstd * sqrt(T(prod(mcf.field_shape)))
    end
    # Zero the composite DC bin before the per-axis contractions: the DC is set by
    # azm (handled above), not by the per-axis amplitudes, so it carries no per-axis
    # gradient. (When use_offset=false it is the normalized 1, also a constant.)
    g_amp_kernel_z = copy(g_amp_kernel)
    g_amp_kernel_z[dc_idx] = zero(T)

    # Per-axis VJP: contract g_amp_kernel_z with the product of the OTHER
    # axes' amp pieces, then sum-reduce over each axis's dims into the
    # axis's bin_index. (Use the same DC-normalised amps as the forward.)
    a_norm = _normalize_axis_dc(amp_per_axis)
    pieces = [a_norm[i][mcf.grids[i].bin_index] for i in 1:mcf.n_axes]
    g_amp_per_axis = Vector{Vector{T}}(undef, mcf.n_axes)
    # One reusable buffer for the per-axis weighted product (allocated once)
    weighted = similar(g_amp_kernel_z)
    for i in 1:mcf.n_axes
        # Restart from g_amp_kernel_z, then in-place multiply by other axes' pieces
        copy!(weighted, g_amp_kernel_z)
        for j in 1:mcf.n_axes
            j == i && continue
            weighted .*= _reshape_for_axis(pieces[j], mcf, j)
        end
        # Sum over all dims not belonging to axis i.
        start_i = sum(_axis_ndims(mcf.grids[j]) for j in 1:i-1; init=0)
        ndims_i = _axis_ndims(mcf.grids[i])
        all_dims = 1:length(mcf.field_shape)
        sum_dims = Tuple(setdiff(all_dims, start_i + 1 : start_i + ndims_i))
        contracted = isempty(sum_dims) ? weighted :
                     dropdims(sum(weighted; dims=sum_dims), dims=sum_dims)
        # Sum-reduce by bin_index (broadcasting over the axis's grid shape)
        g_amp_i = zeros(T, mcf.grids[i].n_bins)
        bin_idx = mcf.grids[i].bin_index
        for k in eachindex(bin_idx)
            g_amp_i[bin_idx[k]] += contracted[k]
        end
        g_amp_per_axis[i] = g_amp_i
    end

    # Backprop each axis's g_amp through amplitude_spectrum_adjoint
    g_z = zeros(T, latent_size(mcf))
    g_z[1:N_total] .= vec(g_xi_field)
    for i in 1:mcf.n_axes
        g_slope, g_fluct, g_flex, g_asp, g_spectrum, _g_offset_ignored =
            amplitude_spectrum_adjoint(g_amp_per_axis[i],
                                        per_axis[i].xi_slope, per_axis[i].xi_fluct,
                                        per_axis[i].xi_flex,  per_axis[i].xi_asp,
                                        per_axis[i].xi_spectrum,
                                        mcf.cfgs[i], mcf.grids[i])
        off = _axis_offset(mcf, i)
        g_z[off + 1] = g_slope
        g_z[off + 2] = g_fluct
        if mcf.cfgs[i].use_iwp
            g_z[off + 3] = g_flex
            g_z[off + 4] = g_asp
            n_spec = 2 * (mcf.grids[i].n_bins - 2)
            g_z[off + 5 : off + 4 + n_spec] .= vec(g_spectrum)
        end
    end
    if mcf.use_offset
        g_z[end] = g_xi_offset
    end
    return g_z
end
