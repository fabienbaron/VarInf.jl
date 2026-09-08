# ============================================================================
# Blind phase-retrieval with the OBJECT POWER-SPECTRUM SLOPE FIXED.
#
# Same problem and same synthetic data as `blind_phase_retrieval.jl`, but
# we first measure Saturn's radial power-spectrum slope and bake it into
# the CF prior as a hard constraint (slope_std = 0). The object's
# fluctuation amplitude and DC offset remain learnable.
#
# This is a cheap way to inject external knowledge: if you know what kind
# of object you're imaging (its rough spectral roll-off), pinning the slope
# kills one of the floppier directions of the joint posterior and the
# inference becomes substantially better-conditioned.
#
# Usage:
#   julia --project=examples examples/blind_phase_retrieval_constrained.jl
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Printf
using Statistics: mean, median, std
using FITSIO
using PyPlot
ion()        # interactive: figures update live; pause() pumps the GUI loop

include(joinpath(@__DIR__, "blind_phase_retrieval_problem.jl"))

# Config (mirrors blind_phase_retrieval.jl for direct comparability)
const SATURN_PATH = "/home/fabien/SOFTWARE/HyperspectralSpeckle.jl/data/saturn512.fits"
const N           = 128
const RADIUS_FRAC = 0.15        # must match blind_phase_retrieval.jl for comparability
const KOL_STRENGTH = 0.4
const NOISE_FRAC  = 0.02
const SEED        = 7
# Hybrid MGVI→GeoVI from the prior mean (no MAP seed): MGVI migrates the mean
# cheaply, GeoVI refines the covariance.
const N_MGVI  = 8
const N_GEOVI = 6
const N_TOTAL = N_MGVI + N_GEOVI
n_samples_schedule(iter) = iter <= 3 ? 4 : 6

# 1. Load and downsample Saturn
println("=== Blind phase retrieval — slope-constrained variant ===")
println("Loading $(SATURN_PATH) ...")
saturn_raw = Float64.(read(FITS(SATURN_PATH)[1]))
@assert size(saturn_raw) == (512, 512)
factor = size(saturn_raw, 1) ÷ N
saturn = zeros(N, N)
@inbounds for j in 1:N, i in 1:N
    block = view(saturn_raw, (i-1)*factor+1:i*factor, (j-1)*factor+1:j*factor)
    saturn[i, j] = mean(block)
end
saturn ./= maximum(saturn)
@printf("Saturn: %d×%d, range = [%.4f, %.4f], mean = %.4f\n",
        N, N, minimum(saturn), maximum(saturn), mean(saturn))

# 2. Fit Saturn's radial power-spectrum slope
#
# Bin |fft(saturn)|² into the CF prior's own radial bins, then do a weighted
# log-log fit of log(P_radial[b]) vs `rel_log_mode_lengths[b]`. Two choices
# that matter:
#   * We fit only the bins BELOW the diffraction cutoff (k ≤ k_cut). Those
#     are the frequencies the data actually constrains, the slope we measure
#     there is what the prior will extrapolate into the unmeasured band. (We
#     happen to have the ground-truth object here, but fitting only the
#     observable band keeps the demo honest about what you'd know in practice.)
#   * The fit is weighted by mode multiplicity: low-k bins average over few
#     modes and are noisy estimates of the spectrum, so they get less weight.
# Saturn is not a true power law (ring oscillations + a disk edge), so we
# also report R² to expose how good the single-slope approximation is. Since
# amp = √P, slope_amp = slope_power / 2.
function fit_object_slope(image::Matrix{Float64}, k_cut::Float64)
    n = size(image, 1)
    grid = FourierGridInfo(n, 1.0)
    P_2d = abs2.(fft(image))
    n_bins = grid.n_bins
    P_radial = zeros(n_bins)
    @inbounds for j in 1:n, i in 1:n
        P_radial[grid.bin_index[i, j]] += P_2d[i, j]
    end
    P_radial ./= grid.mode_multiplicity
    # Fit bins 2..b_cut, where b_cut is the last bin with |k| ≤ k_cut.
    b_cut = something(findlast(b -> grid.mode_lengths[b] <= k_cut, 1:n_bins), n_bins)
    b_range = 2:b_cut
    x = grid.rel_log_mode_lengths[b_range]
    y = log.(P_radial[b_range])
    w = grid.mode_multiplicity[b_range]                 # weights
    # Weighted least squares: y = a + slope_power · x
    sw  = sum(w); swx = sum(w .* x); swy = sum(w .* y)
    swxx = sum(w .* x .* x); swxy = sum(w .* x .* y)
    slope_power = (sw * swxy - swx * swy) / (sw * swxx - swx^2)
    intercept   = (swy - slope_power * swx) / sw
    # Weighted R²
    yhat = intercept .+ slope_power .* x
    ss_res = sum(w .* (y .- yhat).^2)
    ss_tot = sum(w .* (y .- swy/sw).^2)
    r2 = 1 - ss_res / ss_tot
    slope_amp = slope_power / 2
    return (; slope_amp, slope_power, intercept, r2, P_radial, grid, b_range, b_cut)
end

k_cut = diffraction_cutoff_k(RADIUS_FRAC)
fit = fit_object_slope(saturn, k_cut)
slope_amp_fit = fit.slope_amp
@printf("OTF cutoff k_cut = %.3f (%.0f%% of Nyquist) → fitting bins 2..%d\n",
        k_cut, 100 * k_cut / 0.5, fit.b_cut)
@printf("Saturn PSD slope_power ≈ %.3f (R² = %.3f)  ⇒  slope_amp ≈ %.3f\n",
        fit.slope_power, fit.r2, slope_amp_fit)
println("  (Unconstrained run learns the slope from slope_prior=(-2.0, 1.0);")
println("   here it's pinned at the measured value with std 0.)\n")

# Save the radial PSD plot, marking the fitted band and the diffraction cutoff
let
    grid = fit.grid; P_radial = fit.P_radial; nb = grid.n_bins
    xall = grid.rel_log_mode_lengths[2:nb]
    xf   = grid.rel_log_mode_lengths[fit.b_range]
    x_cut = log(k_cut / grid.mode_lengths[2])      # cutoff in rel-log units
    fig_psd = figure("saturn PSD"; figsize=(7, 5))
    ax = fig_psd.add_subplot(1, 1, 1)
    ax.plot(xall, log.(P_radial[2:nb]), "o"; ms=2, label="log P_radial (Saturn)")
    ax.plot(xf, fit.intercept .+ fit.slope_power .* xf, "-"; lw=2,
            label=(@sprintf("weighted fit (k≤k_cut), slope=%.3f", fit.slope_power)))
    ax.axvline(x_cut; ls="--", lw=2, color="k", label="diffraction cutoff")
    ax.set_xlabel("rel log |k|"); ax.set_ylabel("log |F(saturn)|² (radial avg)")
    ax.set_title(@sprintf("Saturn radial PSD — slope_power=%.3f, R²=%.3f",
                          fit.slope_power, fit.r2))
    ax.legend(loc="lower left")
    fig_psd.tight_layout()
    fig_psd.savefig(joinpath(@__DIR__, "saturn_radial_psd.png"); dpi=110)
    println("Saved: saturn_radial_psd.png")
end

# 3. Aperture + object CF prior with the slope pinned
aperture = circular_aperture_blind(N; radius_frac=RADIUS_FRAC)

cfg_obj  = CorrFieldConfig(slope_prior=(slope_amp_fit, 0.0),    # ← PINNED
                            fluct_prior=(1.0, 0.3))            #   fluct still learnable
# NIFTy-style additive offset carries the level: object(z=0) = exp(offset_mean)
# = mean(saturn); offset_std=0.3 lets the data adjust overall brightness.
cf_obj   = CorrelatedField([FourierGridInfo(N, 1.0)], [cfg_obj];
                            offset_prior=(log(mean(saturn)), 0.3))
@printf("Object-prior latent = %d (field %d + hypers %d)\n\n",
        latent_size(cf_obj), N^2, latent_size(cf_obj) - N^2)

# 4. Generate the same synthetic data (same seed)
prob, _, true_strength, phase_true, _, PSF_true, data_clean =
    generate_synthetic_blind(saturn, cf_obj;
        aperture=aperture, true_strength=KOL_STRENGTH,
        phase_fluct_prior=(KOL_STRENGTH, 0.4),   # strength learnable (slope fixed Kolmogorov)
        noise_frac=NOISE_FRAC, seed=SEED)

phase_rms_true = sqrt(mean(phase_true[aperture .> 0.5].^2))
@printf("True phase RMS inside aperture = %.3f rad (target ≈ %.2f)\n",
        phase_rms_true, KOL_STRENGTH)
@printf("Noise σ = %.5f, true turbulence strength = %.3f (learnable)\n", prob.sigma_noise, true_strength)
@printf("Latent dimension (variables minimized): %d  (phase field %d + strength 1 + object field %d + 3 hypers)\n\n",
        latent_size(prob), N^2, N^2)

# 5. Joint MAP (biased baseline) + GeoVI (the real answer)
# As in the unconstrained driver, joint MAP is kept only as the cautionary
# biased baseline; GeoVI runs from the prior mean and does the real work.
println("---- Joint MAP (biased baseline, NOT used to seed GeoVI) ----")
Random.seed!(11)
z_init = 0.01 .* randn(latent_size(prob))
@time z_map = reconstruct_map(prob; z0=z_init, maxiter=800,
                              gtol=(1e-9, 1e-9), verb=true)

phase_map  = phase_from_latent(z_map, prob)
object_map = object_from_latent(z_map, prob)
ph_rms_err_map = sqrt(mean((phase_map .- phase_true)[aperture .> 0.5].^2))
obj_rel_err_map = norm(object_map .- saturn) / norm(saturn)
obj_lo_map, obj_hi_map = object_error_split(object_map, saturn, k_cut)
@printf("MAP phase RMS error in aperture = %.3f rad\n", ph_rms_err_map)
@printf("MAP object rel err vs Saturn     = %.3f  (below cutoff %.3f / above %.3f)\n\n",
        obj_rel_err_map, obj_lo_map, obj_hi_map)

# GeoVI live panel (PyPlot, 3×4: phase / object / data rows)
const _FIG = figure("blind phase retrieval (constrained)"; figsize=(15, 11))

function _imp(i, img, ttl, cmap, vmn, vmx)
    ax = _FIG.add_subplot(3, 4, i)
    im = vmn === nothing ?
        ax.imshow(img; origin="lower", cmap=cmap, aspect="equal") :
        ax.imshow(img; origin="lower", cmap=cmap, vmin=vmn, vmax=vmx, aspect="equal")
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    _FIG.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
end

function _draw_panel_c(iter, fwd_mean, prob, phase_true, saturn, σ_phase, σ_object)
    apm(x) = mask_to_aperture(x, aperture)
    σp_max = maximum(σ_phase[aperture .> 0.5])
    dmax = maximum(prob.data)
    _FIG.clf()
    # Row 1: phase (aperture-masked)
    _imp(1, apm(phase_true),                    "True phase",       "RdBu_r", -2, 2)
    _imp(2, apm(fwd_mean.phase),                "GeoVI mean phase", "RdBu_r", -2, 2)
    _imp(3, apm(σ_phase),                       "GeoVI phase σ",    "magma",  0, max(0.1, σp_max))
    _imp(4, apm(fwd_mean.phase .- phase_true),  "phase residual",   "RdBu_r", -1, 1)
    # Row 2: object
    _imp(5, saturn,                             "True object (Saturn)", "viridis", 0, 1)
    _imp(6, fwd_mean.object,                    "GeoVI mean object",    "viridis", 0, 1)
    _imp(7, σ_object,                           "GeoVI object σ",       "magma", 0, max(0.01, maximum(σ_object)))
    _imp(8, fwd_mean.object .- saturn,          "object residual",      "RdBu_r", -0.3, 0.3)
    # Row 3: data space
    _imp(9,  prob.data,                         "observed data",  "viridis", 0, dmax)
    _imp(10, fwd_mean.data_clean,               "GeoVI model",    "viridis", 0, dmax)
    _imp(11, fwd_mean.data_clean .- prob.data,  "data residual",  "RdBu_r", nothing, nothing)
    _FIG.suptitle(@sprintf("%s iter %d/%d (slope pinned = %.3f)",
                           iter <= N_MGVI ? "MGVI" : "GeoVI",
                           iter, N_TOTAL, slope_amp_fit), fontsize=13)
    _FIG.tight_layout(rect=[0, 0, 1, 0.97])
    _FIG.savefig(joinpath(@__DIR__,
                 @sprintf("blind_phase_constrained_iter%02d.png", iter)); dpi=110)
    _FIG.canvas.draw(); pause(0.001)
end

function iter_callback_c(prob, z_mean, samples, iter)
    fwd_mean = _forward(z_mean, prob)
    phase_stack  = Array{Float64}(undef, N, N, length(samples))
    object_stack = Array{Float64}(undef, N, N, length(samples))
    for (k, s) in enumerate(samples)
        phase_stack[:,:,k]  = phase_from_latent(s, prob)
        object_stack[:,:,k] = object_from_latent(s, prob)
    end
    σ_phase  = dropdims(std(phase_stack; dims=3); dims=3)
    σ_object = dropdims(std(object_stack; dims=3); dims=3)

    ph_rms_err = sqrt(mean((fwd_mean.phase .- phase_true)[aperture .> 0.5].^2))
    obj_rel_err = norm(fwd_mean.object .- saturn) / norm(saturn)
    e_mean = 0.0
    for s in samples
        e_s, _ = VarInf.energy_and_gradient(prob, z_mean .+ s)
        e_mean += e_s
    end
    e_mean /= length(samples)
    @printf("  %s iter %d/%d:  E_KL ≈ %.3f,  phase RMS err = %.3f rad,  object rel err = %.3f\n",
            iter <= N_MGVI ? "MGVI" : "GeoVI", iter, N_TOTAL,
            e_mean, ph_rms_err, obj_rel_err)

    _draw_panel_c(iter, fwd_mean, prob, phase_true, saturn, σ_phase, σ_object)
end

println("---- Hybrid MGVI→GeoVI ($(N_MGVI)+$(N_GEOVI) iter, samples 4→6), from prior mean ----")
z0_prior = zeros(latent_size(prob))
# kl_/cg_/geo_ params left at hybrid defaults (= reconstruct_geovi's); sample_mode
# overridden so the GeoVI phase resamples each iter, matching standalone GeoVI.
@time z_geovi, samples_geovi = reconstruct_hybrid(
    prob; z0=z0_prior, map_maxiter=0,
    n_mgvi=N_MGVI, n_geovi=N_GEOVI, n_samples=n_samples_schedule,
    sample_mode = iter -> iter <= N_MGVI ? :linear_resample : :nonlinear_resample,
    verb=true, iter_callback=iter_callback_c)

fwd_geovi = _forward(z_geovi, prob)
ph_rms_err_geovi = sqrt(mean((fwd_geovi.phase .- phase_true)[aperture .> 0.5].^2))
obj_rel_err_geovi = norm(fwd_geovi.object .- saturn) / norm(saturn)
obj_lo_geovi, obj_hi_geovi = object_error_split(fwd_geovi.object, saturn, k_cut)
@printf("\nGeoVI posterior mean (slope pinned = %.3f):\n", slope_amp_fit)
@printf("    phase RMS err in aperture = %.3f rad\n", ph_rms_err_geovi)
@printf("    object rel err vs Saturn   = %.3f\n", obj_rel_err_geovi)
@printf("    object Fourier err  below cutoff = %.3f   above cutoff = %.3f\n",
        obj_lo_geovi, obj_hi_geovi)
# Recovery check: turbulence strength learned (phase slope fixed to Kolmogorov).
let str = [phase_strength_from_latent(z_geovi .+ s, prob) for s in samples_geovi]
    @printf("    turbulence strength: recovered %.3f ± %.3f  (truth %.3f) — learned, slope pinned\n",
            mean(str), std(str), true_strength)
end
println("    (The 'above cutoff' number is the one to compare against the")
println("     unconstrained run — lower here means the pinned slope helped.)")

println("\n================ joint MAP  vs  GeoVI-from-prior ================")
@printf("  phase RMS err (rad)   MAP %.3f    GeoVI %.3f\n",
        ph_rms_err_map, ph_rms_err_geovi)
@printf("  object rel err        MAP %.3f    GeoVI %.3f\n",
        obj_rel_err_map, obj_rel_err_geovi)
@printf("  object err above cut  MAP %.3f    GeoVI %.3f\n",
        obj_hi_map, obj_hi_geovi)

for (name, arr) in [
        ("blind_c_phase_geovi_mean.fits",   fwd_geovi.phase),
        ("blind_c_object_geovi_mean.fits",  fwd_geovi.object),
    ]
    path = joinpath(@__DIR__, name)
    isfile(path) && rm(path)
    FITS(path, "w") do f; write(f, arr); end
end
println("=== Slope-constrained blind demo complete ===")
