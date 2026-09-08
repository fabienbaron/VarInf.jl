# ============================================================================
# Blind phase-retrieval demo: jointly infer (Kolmogorov phase screen,
# log-normal object) from a single noisy aberrated image. Object is Saturn,
# downsampled from 512×512 to 128×128. Object is not drawn from our CF
# prior (it's a real image), so this is a fair test of the prior's
# robustness to model mismatch.
#
# Usage:
#   julia --project=examples examples/blind_phase_retrieval.jl
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

# Config
const SATURN_PATH = "/home/fabien/SOFTWARE/HyperspectralSpeckle.jl/data/saturn512.fits"
const N           = 128         # working grid size (saturn downsampled 4×)
const RADIUS_FRAC = 0.15        # pupil radius (→ OTF cutoff at 0.6·Nyquist)
const KOL_STRENGTH = 0.4        # phase RMS (rad)
const NOISE_FRAC  = 0.02        # noise σ as fraction of clean-data peak
const SEED        = 7
# Hybrid MGVI→GeoVI from the prior mean: cheap MGVI iterations migrate the mean
# a long way (GeoVI alone is slow to converge from scratch), then GeoVI refines
# the covariance. No MAP warm-start anywhere.
const N_MGVI  = 8
const N_GEOVI = 6
const N_TOTAL = N_MGVI + N_GEOVI
n_samples_schedule(iter) = iter <= 3 ? 4 : 6  # keep samples up from the start;
                                              # the marginalization is what avoids
                                              # the joint-MAP collapse.

# 1. Load and downsample Saturn
println("=== Blind phase retrieval (joint phase + object) ===")
println("Loading $(SATURN_PATH) ...")
saturn_raw = Float64.(read(FITS(SATURN_PATH)[1]))   # 512 × 512
@assert size(saturn_raw) == (512, 512)
# 4× block-average → 128×128
factor = size(saturn_raw, 1) ÷ N
saturn = zeros(N, N)
@inbounds for j in 1:N, i in 1:N
    block = view(saturn_raw, (i-1)*factor+1:i*factor, (j-1)*factor+1:j*factor)
    saturn[i, j] = mean(block)
end
saturn ./= maximum(saturn)        # peak = 1
@printf("Saturn: %d×%d, range = [%.4f, %.4f], mean = %.4f\n",
        N, N, minimum(saturn), maximum(saturn), mean(saturn))

# 2. Build aperture + object CF prior
aperture = circular_aperture_blind(N; radius_frac=RADIUS_FRAC)
fill_frac = sum(aperture) / N^2
k_cut = diffraction_cutoff_k(RADIUS_FRAC)        # OTF cutoff (cycles/pixel)
@printf("Pupil radius_frac = %.2f → OTF cutoff k_cut = %.3f (%.0f%% of Nyquist)\n",
        RADIUS_FRAC, k_cut, 100 * k_cut / 0.5)

grid_obj = FourierGridInfo(N, 1.0)
# Object prior with a GENUINELY BROAD slope: (-2.0 ± 1.0) lets the joint
# inference try to learn the spectral slope from the single blurred image
# (it covers the truth ≈ -1.57 well within ±1σ). This is the fair baseline
# for the slope-constrained variant, which instead pins the slope at the
# value measured from Saturn's own PSD. Moderate fluct, broad offset.
cfg_obj  = CorrFieldConfig(slope_prior=(-2.0, 1.0),
                            fluct_prior=(1.0, 0.3))
# NIFTy-style additive offset carries the object's level: at z=0 the object is
# exp(offset_mean) = mean(saturn). offset_std=0.3 lets the data adjust the
# overall brightness via a well-behaved O(1) latent.
cf_obj   = CorrelatedField([grid_obj], [cfg_obj];
                            offset_prior=(log(mean(saturn)), 0.3))

@printf("Aperture fill = %.1f%%, object-prior latent = %d (field %d + hypers %d)\n",
        100 * fill_frac, latent_size(cf_obj), N^2, latent_size(cf_obj) - N^2)

# 3. Generate synthetic noisy aberrated observation
prob, _, true_strength, phase_true, _, PSF_true, data_clean =
    generate_synthetic_blind(saturn, cf_obj;
        aperture=aperture, true_strength=KOL_STRENGTH,
        phase_fluct_prior=(KOL_STRENGTH, 0.4),   # strength learnable (prior wide, slope fixed)
        noise_frac=NOISE_FRAC, seed=SEED)

phase_rms_true = sqrt(mean(phase_true[aperture .> 0.5].^2))
@printf("True phase RMS inside aperture = %.3f rad (target ≈ %.2f)\n",
        phase_rms_true, KOL_STRENGTH)
@printf("Noise σ = %.5f, true turbulence strength = %.3f (learnable)\n", prob.sigma_noise, true_strength)
@printf("Latent dimension (variables minimized): %d  (phase field %d + strength 1 + object field %d + 3 hypers)\n\n",
        latent_size(prob), N^2, N^2)

# 4. Joint MAP: cautionary baseline only
# Joint MAP over (phase, object, hypers) is known to be biased for blind
# deblurring (Levin et al. 2011): the posterior DENSITY peaks at the trivial
# no-blur solution (sharp object, flat-ish phase) even though the posterior
# MASS does not. We compute it only to contrast against GeoVI; it is not used
# to warm-start the variational inference (doing so would drop GeoVI straight
# into this biased basin).
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
imshow(object_map)
@printf("MAP phase RMS error in aperture = %.3f rad\n", ph_rms_err_map)
@printf("MAP object rel err vs Saturn     = %.3f  (below cutoff %.3f / above %.3f)\n\n",
        obj_rel_err_map, obj_lo_map, obj_hi_map)

# 5. GeoVI posterior
# Per-iter live figure (PyPlot/matplotlib), 3×4: phase row (aperture-masked),
# object row, data row. One persistent figure, redrawn + saved each iteration.

const _FIG = figure("blind phase retrieval"; figsize=(15, 11))

# Draw one imshow panel into subplot slot `i` of the 3×4 grid.
function _imp(i, img, ttl, cmap, vmn, vmx)
    ax = _FIG.add_subplot(3, 4, i)
    im = vmn === nothing ?
        ax.imshow(img; origin="lower", cmap=cmap, aspect="equal") :
        ax.imshow(img; origin="lower", cmap=cmap, vmin=vmn, vmax=vmx, aspect="equal")
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    _FIG.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
end

function _draw_panel(iter, fwd_mean, prob, phase_true, saturn, σ_phase, σ_object)
    apm(x) = mask_to_aperture(x, aperture)
    σp_max = maximum(σ_phase[aperture .> 0.5])
    dmax = maximum(prob.data)
    _FIG.clf()
    # Row 1: phase (aperture-masked; NaN outside shows blank)
    _imp(1, apm(phase_true),                    "True phase",       "RdBu_r", -2, 2)
    _imp(2, apm(fwd_mean.phase),                "GeoVI mean phase", "RdBu_r", -2, 2)
    _imp(3, apm(σ_phase),                       "GeoVI phase σ",    "magma",  0, max(0.1, σp_max))
    _imp(4, apm(fwd_mean.phase .- phase_true),  "phase residual",   "RdBu_r", -1, 1)
    # Row 2: object (full frame)
    _imp(5, saturn,                             "True object (Saturn)", "viridis", 0, 1)
    _imp(6, fwd_mean.object,                    "GeoVI mean object",    "viridis", 0, 1)
    _imp(7, σ_object,                           "GeoVI object σ",       "magma", 0, max(0.01, maximum(σ_object)))
    _imp(8, fwd_mean.object .- saturn,          "object residual",      "RdBu_r", -0.3, 0.3)
    # Row 3: data space
    _imp(9,  prob.data,                         "observed data",  "viridis", 0, dmax)
    _imp(10, fwd_mean.data_clean,               "GeoVI model",    "viridis", 0, dmax)
    _imp(11, fwd_mean.data_clean .- prob.data,  "data residual",  "RdBu_r", nothing, nothing)
    _FIG.suptitle("$(iter <= N_MGVI ? "MGVI" : "GeoVI") iter $(iter)/$(N_TOTAL)",
                  fontsize=13)
    _FIG.tight_layout(rect=[0, 0, 1, 0.97])
    _FIG.savefig(joinpath(@__DIR__, @sprintf("blind_phase_iter%02d.png", iter)); dpi=110)
    _FIG.canvas.draw(); pause(0.001)
end

function iter_callback(prob, z_mean, samples, iter)
    fwd_mean = _forward(z_mean, prob)
    # Per-pixel phase σ + object σ from posterior samples
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
    # Sample-averaged KL energy at the current variational mean
    e_mean = 0.0
    for s in samples
        e_s, _ = VarInf.energy_and_gradient(prob, z_mean .+ s)
        e_mean += e_s
    end
    e_mean /= length(samples)
    @printf("  %s iter %d/%d:  E_KL ≈ %.3f,  phase RMS err = %.3f rad,  object rel err = %.3f\n",
            iter <= N_MGVI ? "MGVI" : "GeoVI", iter, N_TOTAL,
            e_mean, ph_rms_err, obj_rel_err)

    _draw_panel(iter, fwd_mean, prob, phase_true, saturn, σ_phase, σ_object)
end

# Hybrid MGVI→GeoVI does the real work, starting from the prior mean (z0 = zeros),
# with the internal MAP disabled. The mean update minimizes the sample-AVERAGED
# energy (a marginal estimate), which is what avoids the joint-MAP bias above;
# the MGVI phase just migrates the mean cheaply before GeoVI refines.
println("---- Hybrid MGVI→GeoVI ($(N_MGVI)+$(N_GEOVI) iter, samples 4→6), from prior mean ----")
z0_prior = zeros(latent_size(prob))
# Leave kl_/cg_/geo_ params at hybrid defaults; they equal reconstruct_geovi's,
# so the GeoVI phase matches standalone GeoVI. Override sample_mode so that
# phase RESAMPLES each iteration (fresh linear draw + refine), like standalone
# GeoVI, rather than the hybrid's default :nonlinear_update (sample reuse).
@time z_geovi, samples_geovi = reconstruct_hybrid(
    prob; z0=z0_prior, map_maxiter=0,
    n_mgvi=N_MGVI, n_geovi=N_GEOVI, n_samples=n_samples_schedule,
    sample_mode = iter -> iter <= N_MGVI ? :linear_resample : :nonlinear_resample,
    verb=true, iter_callback=iter_callback)

fwd_geovi = _forward(z_geovi, prob)
ph_rms_err_geovi = sqrt(mean((fwd_geovi.phase .- phase_true)[aperture .> 0.5].^2))
obj_rel_err_geovi = norm(fwd_geovi.object .- saturn) / norm(saturn)
obj_lo_geovi, obj_hi_geovi = object_error_split(fwd_geovi.object, saturn, k_cut)
@printf("\nGeoVI posterior mean: phase RMS err = %.3f rad, object rel err = %.3f\n",
        ph_rms_err_geovi, obj_rel_err_geovi)
@printf("    object Fourier err  below cutoff = %.3f   above cutoff = %.3f\n",
        obj_lo_geovi, obj_hi_geovi)
# Recovery check: the turbulence strength was LEARNED (slope fixed to Kolmogorov).
let str = [phase_strength_from_latent(z_geovi .+ s, prob) for s in samples_geovi]
    @printf("    turbulence strength: recovered %.3f ± %.3f  (truth %.3f) — learned, slope pinned\n",
            mean(str), std(str), true_strength)
end
println("    (Compare 'above cutoff' against the slope-constrained run — that's")
println("     the band the data can't see, where the spectral prior does the work.)")

# MAP-vs-GeoVI contrast (the point of keeping MAP around)
println("\n================ joint MAP  vs  GeoVI-from-prior ================")
@printf("  phase RMS err (rad)   MAP %.3f    GeoVI %.3f\n",
        ph_rms_err_map, ph_rms_err_geovi)
@printf("  object rel err        MAP %.3f    GeoVI %.3f\n",
        obj_rel_err_map, obj_rel_err_geovi)
@printf("  object err below cut  MAP %.3f    GeoVI %.3f\n",
        obj_lo_map, obj_lo_geovi)
@printf("  object err above cut  MAP %.3f    GeoVI %.3f\n",
        obj_hi_map, obj_hi_geovi)
println("  (Joint MAP is the biased baseline; the marginalizing GeoVI mean")
println("   should do better, especially on phase and the above-cutoff band.)")

# 6. Save outputs
for (name, arr) in [
        ("blind_phase_truth.fits",      phase_true),
        ("blind_object_truth.fits",     saturn),
        ("blind_data_observed.fits",    prob.data),
        ("blind_phase_map.fits",        phase_map),
        ("blind_object_map.fits",       object_map),
        ("blind_phase_geovi_mean.fits", fwd_geovi.phase),
        ("blind_object_geovi_mean.fits", fwd_geovi.object),
    ]
    path = joinpath(@__DIR__, name)
    isfile(path) && rm(path)
    FITS(path, "w") do f; write(f, arr); end
end
println("Saved 7 FITS files for inspection.")
println("=== Blind phase retrieval demo complete ===")
