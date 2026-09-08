# ============================================================================
# Phase retrieval with the phase CF's slope PINNED at Kolmogorov (−11/6).
# Known object (Saturn); only the fluctuation amplitude is learned. This is
# the slope-constrained counterpart of phase_retrieval_latent.jl, pinning
# the slope at the physically expected value tightens the inference.
#
# Usage:
#   julia --project=examples examples/phase_retrieval_constrained.jl
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Printf
using Statistics: mean, std
using FITSIO
using PyPlot
ion()

include(joinpath(@__DIR__, "phase_retrieval_latent_problem.jl"))

const SATURN_PATH = "/home/baron/SOFTWARE/HyperspectralSpeckle.jl/data/saturn512.fits"
const N            = 128
const NOISE_FRAC   = 0.02
const SEED         = 7
const N_MGVI    = 10   # hybrid: cheap MGVI migration then GeoVI refinement
const N_GEOVI   = 10
const N_TOTAL   = N_MGVI + N_GEOVI
const N_SAMPLES = 10   # enough samples to escape the non-convex (twin) basin
const TRUE_FLUCT   = 0.60   # at N=128 gives in-aperture RMS ≈ 0.55 (matched)
# Slope PINNED at Kolmogorov (std=0); only the fluctuation is learnable.
const SLOPE_PRIOR  = (-11/6, 0.0)
const FLUCT_PRIOR = (TRUE_FLUCT, 0.12)
const DEMO_LABEL   = "slope pinned (Kolmogorov)"
const PNG_PREFIX   = "phase_retrieval_constrained"

println("=== Phase retrieval — $(DEMO_LABEL) ===")

# 1. Load + downsample Saturn 256 → 64
raw = let f = FITS(SATURN_PATH); img = read(f[1]); close(f); img end
@assert size(raw) == (512, 512)
saturn_full = Float64.(raw)
saturn = let blk = 512 ÷ N
    coarse = zeros(N, N)
    for j in 1:N, i in 1:N
        s = 0.0
        for jj in 1:blk, ii in 1:blk
            s += saturn_full[blk*(i-1)+ii, blk*(j-1)+jj]
        end
        coarse[i, j] = s / (blk*blk)
    end
    coarse ./ maximum(coarse)
end

# Non-centrosymmetric pupil (off-axis central obscuration) breaks the twin-image
# ambiguity, so cold-start GeoVI converges to the true phase, not its twin.
aperture = asymmetric_aperture_latent(N; radius_frac=0.40)
mask = aperture .> 0
@printf("Saturn range [%.3f, %.3f], aperture fill %.1f%%\n",
        minimum(saturn), maximum(saturn), 100*mean(aperture))

# 2. Build the slope-pinned phase CF + synthetic observation
# Same fixed-Kolmogorov truth and SEED as phase_retrieval_latent.jl, so the two
# demos run on IDENTICAL data and differ only in the inference prior (here the
# slope is pinned at Kolmogorov; there it is learned).
cf_phase = make_phase_cf(N; slope_prior=SLOPE_PRIOR, fluct_prior=FLUCT_PRIOR)
prob, phase_true, psf_true, data_clean, true_slope, true_fluct =
    generate_synthetic_phase_retrieval_latent(saturn, cf_phase;
        aperture=aperture, true_fluct=TRUE_FLUCT, noise_frac=NOISE_FRAC, seed=SEED)

truth_rms = sqrt(mean(phase_true[mask] .^ 2))
@printf("Truth: slope = %+.3f (pinned), fluct = %.3f  (phase RMS in aperture = %.3f rad)\n",
        true_slope, true_fluct, truth_rms)
@printf("Noise σ = %.5f\n", prob.sigma_noise)
@printf("Latent dimension (variables minimized): %d  (phase field %d + slope + fluct)\n\n",
        latent_size(prob), N^2)

# NOTE: no MAP; see phase_retrieval_latent.jl. Phase retrieval is non-convex
# (twin-image ambiguity), so cold-start MAP is useless; GeoVI runs from the
# prior mean. The smooth posterior mean is the Wiener-shrunk estimate; high-k
# Kolmogorov structure lives in the posterior samples.

# 4. Live PyPlot panel (2×3)
phase_lim = maximum(abs.(phase_true[mask])) * 1.05
_apm(x) = ifelse.(mask, Float64.(x), NaN)
const _FIG = figure(PNG_PREFIX; figsize=(15, 9.5))

function _imp(i, img, ttl, cmap, vmn, vmx)
    ax = _FIG.add_subplot(2, 3, i)
    im = vmn === nothing ?
        ax.imshow(img; origin="lower", cmap=cmap, aspect="equal") :
        ax.imshow(img; origin="lower", cmap=cmap, vmin=vmn, vmax=vmx, aspect="equal")
    ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    _FIG.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
end

function _draw(phase_mean, phase_std, phase_sample, iter)
    rms_err = sqrt(mean((phase_mean[mask] .- phase_true[mask]).^2))
    residual = phase_mean .- phase_true
    res_lim = max(maximum(abs.(residual[mask]))*1.05, 1e-6)
    med_std = let v = phase_std[mask]; sort(v)[length(v)÷2 + 1]; end
    _FIG.clf()
    _imp(1, _apm(phase_true),   "truth φ (rad)", "RdBu_r", -phase_lim, phase_lim)
    _imp(2, _apm(phase_sample), "posterior sample φ (has high-k)", "RdBu_r", -phase_lim, phase_lim)
    _imp(3, _apm(phase_mean),   (@sprintf("GeoVI mean φ (iter %d, RMS err %.3f)", iter, rms_err)), "RdBu_r", -phase_lim, phase_lim)
    _imp(4, _apm(phase_std),    (@sprintf("GeoVI σ(φ) (median %.3f rad)", med_std)), "viridis", 0, nothing)
    _imp(5, prob.data,          "observed data", "inferno", nothing, nothing)
    _imp(6, _apm(residual),     "residual: GeoVI mean − truth (rad)", "RdBu_r", -res_lim, res_lim)
    _FIG.suptitle(@sprintf("Phase retrieval [%s] (truth RMS=%.3f, %s iter %d)",
                           DEMO_LABEL, truth_rms, iter <= N_MGVI ? "MGVI" : "GeoVI", iter),
                  fontsize=12)
    _FIG.tight_layout(rect=[0,0,1,0.97])
    _FIG.savefig(@sprintf("%s_iter%02d.png", PNG_PREFIX, iter); dpi=110)
    _FIG.canvas.draw(); pause(0.001)
    return rms_err
end

function iter_callback(_prob, z, samples, iter)
    phase_samples = [phase_from_latent(z .+ s, prob) for s in samples]
    stack = reduce((a,b)->cat(a,b;dims=3), [reshape(ph,N,N,1) for ph in phase_samples])
    pmean = dropdims(mean(stack,dims=3),dims=3)
    pstd  = dropdims(sqrt.(mean((stack .- pmean).^2;dims=3)),dims=3)
    rms_err = _draw(pmean, pstd, phase_samples[1], iter)
    e_kl = mean(VarInf.energy_and_gradient(prob, z .+ s)[1] for s in samples)
    @printf("  %s iter %d/%d:  E_KL ≈ %.3f,  VI-mean phase RMS = %.4f rad\n",
            iter <= N_MGVI ? "MGVI" : "GeoVI", iter, N_TOTAL, e_kl, rms_err)
end

# 5. Hybrid MGVI→GeoVI from the prior mean (no MAP)
println("---- Hybrid MGVI→GeoVI ($N_MGVI+$N_GEOVI iter × $N_SAMPLES samples), from prior mean ----")
@time z_geo, samples_geo = reconstruct_hybrid(
    prob; z0=zeros(latent_size(prob)), map_maxiter=0,
    n_mgvi=N_MGVI, n_geovi=N_GEOVI, n_samples=N_SAMPLES,
    sample_mode = iter -> iter <= N_MGVI ? :linear_resample : :nonlinear_resample,
    kl_maxiter=40, cg_maxiter=50, cg_tol=0.03,
    geo_newton_maxiter=6, geo_cg_maxiter=20, geo_tol=1e-6,
    iter_callback=iter_callback, verb=true)

# 6. Final summary + figure
phase_samples = [phase_from_latent(z_geo .+ s, prob) for s in samples_geo]
stack = reduce((a,b)->cat(a,b;dims=3), [reshape(ph,N,N,1) for ph in phase_samples])
pmean = dropdims(mean(stack,dims=3),dims=3)
pstd  = dropdims(sqrt.(mean((stack .- pmean).^2;dims=3)),dims=3)
phase_rms_geo = sqrt(mean((pmean[mask] .- phase_true[mask]).^2))
# Posterior mean of the hypers = average over samples (slope is pinned, so its
# value is constant; fluct is the learnable one).
_hd = [phase_hypers(z_geo .+ s, prob) for s in samples_geo]
geo_slope = mean(d[1] for d in _hd); geo_fluct = mean(d[2] for d in _hd)
@printf("\nGeoVI: slope = %+.3f (pinned), fluct = %.3f, posterior-mean phase RMS err = %.4f rad\n",
        geo_slope, geo_fluct, phase_rms_geo)

_draw(pmean, pstd, phase_samples[1], N_TOTAL)
_FIG.savefig("$(PNG_PREFIX).png"; dpi=110)
println("Saved figure: $(PNG_PREFIX).png")

for (name, arr) in [("$(PNG_PREFIX)_truth.fits", phase_true),
                    ("$(PNG_PREFIX)_geovi_mean.fits", pmean),
                    ("$(PNG_PREFIX)_geovi_std.fits", pstd)]
    isfile(name) && rm(name)
    FITS(name, "w") do f; write(f, arr); end
end
println("=== Phase retrieval ($(DEMO_LABEL)) complete ===")
