# ============================================================================
# Kolmogorov phase-screen retrieval from a single Saturn observation,
# with LIVE-updating multi-panel figure across GeoVI iterations.
#
# Setup (same physics as before):
#   • Object: known Saturn image (saturn512.fits), downsampled 512×512 → N×N.
#   • Aperture: asymmetric pupil (circle with a near-central off-axis
#     obscuration); its non-centrosymmetry breaks the twin-image ambiguity so
#     cold-start VI recovers the phase reliably (a plain centred circle gets
#     twin-trapped).
#   • Phase screen: Kolmogorov-distributed (slope = −11/3 on the phase PSD),
#     drawn from a correlated-field prior. Weak turbulence (RMS ≈ 0.5 rad).
#   • Observation: data = saturn ⊛ |fft(aperture · exp(iφ))|²  + N(0, σ²).
#   • Noise: σ = 2% of the noise-free observation's peak (≈30 dB S/N).
#
# Live-figure behaviour:
#   • A 2 × 3 PyPlot panel (truth / MAP / current GeoVI mean; posterior σ /
#     observed data / residual) is redrawn after every KL-Newton step inside
#     GeoVI. Live via ion()+pause; one PNG saved per iteration
#     (phase_retrieval_iter01.png, …) plus the final phase_retrieval.png.
#
# Usage:
#   julia --project=examples examples/phase_retrieval.jl
# ============================================================================

using VarInf
using FFTW
using LinearAlgebra
using Random
using Printf
using Statistics: mean
using FITSIO
using PyPlot
ion()        # interactive: figures update live; pause() pumps the GUI loop

include(joinpath(@__DIR__, "phase_retrieval_problem.jl"))

const SATURN_PATH = "/home/fabien/SOFTWARE/HyperspectralSpeckle.jl/data/saturn512.fits"
const N            = 128
const KOLMOGOROV_K = 0.6       # at N=128 gives in-aperture phase RMS ≈ 0.55 rad
                              # (matches the N=64 run at K=0.4 for a fair comparison)
const NOISE_FRAC   = 0.02
const SEED         = 7
# Inference: hybrid MGVI→GeoVI from the prior mean (no warm-start). MAP is run
# only as a cold-start REFERENCE to compare convergence (no latent hypers here,
# so this is the clean MAP-vs-hybrid comparison).
const N_MGVI       = 20
const N_GEOVI      = 10
const N_TOTAL      = N_MGVI + N_GEOVI
const N_SAMPLES    = 6   # antithetic pairs: +δ and −δ are refined separately
                         # (NIFTy draw_residual), so each is an independent draw;
                         # the asymmetric pupil breaks the twin, so this suffices

println("=== Kolmogorov phase-screen retrieval (live) ===")
@printf("Aperture grid %d×%d, Kolmogorov strength = %.2f, noise = %.0f%% of peak\n",
        N, N, KOLMOGOROV_K, 100 * NOISE_FRAC)

# 1. Load Saturn and downsample 256 → 64

println("Loading $SATURN_PATH ...")
raw = let f = FITS(SATURN_PATH)
    img = read(f[1])
    close(f)
    img
end
@assert size(raw) == (512, 512)
saturn_full = Float64.(raw)
saturn = let blk = 512 ÷ N
    coarse = zeros(N, N)
    for j in 1:N, i in 1:N
        s = 0.0
        for jj in 1:blk, ii in 1:blk
            s += saturn_full[blk*(i-1) + ii, blk*(j-1) + jj]
        end
        coarse[i, j] = s / (blk * blk)
    end
    coarse ./ maximum(coarse)
end

# Asymmetric pupil (circle + near-central off-axis obscuration): breaks the
# twin-image ambiguity so the cold-start VI converges reliably to the true phase
# (a centrosymmetric circle leaves φ→−φ_flipped degenerate and the inference
# lands in the twin basin at random). Recovery is reproducible (~90% phase var).
aperture = asymmetric_aperture(N; radius_frac=0.40)
mask = aperture .> 0
@printf("Saturn range: [%.3f, %.3f], aperture fill fraction = %.1f%%\n",
        minimum(saturn), maximum(saturn), 100 * mean(aperture))

# 2. Generate the true phase screen + observation

prob, z_true, phase_true, psf_true, data_clean =
    generate_synthetic_phase_retrieval(saturn;
        aperture=aperture, kolmogorov_strength=KOLMOGOROV_K,
        noise_frac=NOISE_FRAC, seed=SEED)

phase_in_aperture = phase_true[mask]
truth_rms = sqrt(mean(phase_in_aperture .^ 2))
@printf("Phase RMS inside aperture = %.3f rad (peak |φ| = %.3f rad)\n",
        truth_rms, maximum(abs.(phase_in_aperture)))
@printf("Noise σ = %.4f\n", prob.sigma_noise)
@printf("Latent dimension (variables minimized): %d  (phase field, no learnable hypers)\n",
        latent_size(prob))

# 3. MAP: cold-start reference only (not used to warm-start the VI)
# Phase retrieval is non-convex (twin-image ambiguity), so a cold-start MAP
# lands in a wrong basin. We compute it purely to compare convergence against
# the hybrid VI below; nothing is seeded from it.
println("\n---- MAP (L-BFGS) cold-start reference ----")
@time z_map = reconstruct_map(prob; z0=zeros(latent_size(prob)), maxiter=500,
                              gtol=(1e-9, 1e-9), verb=true)
phase_map = phase_from_latent(z_map, prob)
phase_rms_map = sqrt(mean((phase_map[mask] .- phase_true[mask]) .^ 2))
@printf("MAP (cold) phase RMS err inside aperture = %.4f rad\n", phase_rms_map)

# 4. Figure plumbing

_aperture_masked(arr, mask) = ifelse.(mask, Float64.(arr), NaN)

phase_lim = maximum(abs.(phase_true[mask])) * 1.05

const _FIG = figure("phase retrieval"; figsize=(15, 9.5))

# Draw one imshow panel into subplot slot `i` of the 2×3 grid.
function _imp(i, img, ttl, cmap, vmn, vmx)
    ax = _FIG.add_subplot(2, 3, i)
    im = vmn === nothing ?
        ax.imshow(img; origin="lower", cmap=cmap, aspect="equal") :
        ax.imshow(img; origin="lower", cmap=cmap, vmin=vmn, vmax=vmx, aspect="equal")
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    _FIG.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
end

# Convergence trace (RMS error vs iter)
const _conv_iters = Int[]
const _conv_rms   = Float64[]

function _draw_panel(phase_mean, phase_std, phase_sample, iter::Int)
    rms_err = sqrt(mean((phase_mean[mask] .- phase_true[mask]) .^ 2))
    residual = phase_mean .- phase_true
    res_lim = max(maximum(abs.(residual[mask])) * 1.05, 1e-6)
    med_std = let v = phase_std[mask]; sort(v)[length(v) ÷ 2 + 1]; end
    apm(x) = _aperture_masked(x, mask)
    _FIG.clf()
    # Row 1: truth, cold-start MAP (reference), hybrid posterior mean.
    _imp(1, apm(phase_true),   "truth φ (rad)",                  "RdBu_r", -phase_lim, phase_lim)
    _imp(2, apm(phase_map),    (@sprintf("MAP φ cold-ref (RMS err %.3f)", phase_rms_map)), "RdBu_r", -phase_lim, phase_lim)
    _imp(3, apm(phase_mean),   (@sprintf("VI mean φ (iter %d, RMS err %.3f)", iter, rms_err)), "RdBu_r", -phase_lim, phase_lim)
    # Row 2: one posterior sample (keeps high-k), uncertainty, residual.
    _imp(4, apm(phase_sample), "posterior sample φ (has high-k)", "RdBu_r", -phase_lim, phase_lim)
    _imp(5, apm(phase_std),    (@sprintf("VI σ(φ) (median %.3f rad)", med_std)), "viridis", 0, nothing)
    _imp(6, apm(residual),     "residual: VI mean − truth (rad)", "RdBu_r", -res_lim, res_lim)
    _FIG.suptitle(@sprintf("Phase retrieval [no latent vars] (truth RMS=%.3f, %s iter %d)",
                           truth_rms, iter <= N_MGVI ? "MGVI" : "GeoVI", iter), fontsize=12)
    _FIG.tight_layout(rect=[0, 0, 1, 0.97])
    _FIG.savefig(@sprintf("phase_retrieval_iter%02d.png", iter); dpi=110)
    _FIG.canvas.draw(); pause(0.001)
    return rms_err
end

# iter_callback fed to reconstruct_geovi, runs after every KL-Newton step.
function _live_callback(_prob, z, samples, iter)
    # Compute the posterior phase mean and std from the current sample set.
    phase_samples = [phase_from_latent(z .+ s, prob) for s in samples]
    stack = reduce((a, b) -> cat(a, b; dims=3),
                    [reshape(ph, N, N, 1) for ph in phase_samples])
    pmean = dropdims(mean(stack, dims=3), dims=3)
    pstd  = dropdims(sqrt.(mean((stack .- pmean) .^ 2; dims=3)), dims=3)

    rms_err = _draw_panel(pmean, pstd, phase_samples[1], iter)
    push!(_conv_iters, iter)
    push!(_conv_rms,   rms_err)

    # Energy summary at this z, samples (averaged sample-KL energy ≈ E_KL)
    e_kl = mean(VarInf.energy_and_gradient(prob, z .+ s)[1] for s in samples)
    @printf("  %s iter %d/%d:  E_KL ≈ %.3f,  VI-mean phase RMS = %.4f rad\n",
            iter <= N_MGVI ? "MGVI" : "GeoVI", iter, N_TOTAL, e_kl, rms_err)
    return nothing
end

# 5. Hybrid MGVI→GeoVI from the prior mean (no warm-start), live
println("\n---- Hybrid MGVI→GeoVI ($N_MGVI+$N_GEOVI iter × $N_SAMPLES samples), from prior mean ----")
@time z_geo, samples_geo = reconstruct_hybrid(
    prob; z0=zeros(latent_size(prob)), map_maxiter=0,
    n_mgvi=N_MGVI, n_geovi=N_GEOVI, n_samples=N_SAMPLES,
    sample_mode = iter -> iter <= N_MGVI ? :linear_resample : :nonlinear_resample,
    kl_maxiter=40, cg_maxiter=50, cg_tol=0.03,
    geo_newton_maxiter=6, geo_cg_maxiter=20, geo_tol=1e-6,
    iter_callback=_live_callback, verb=true)

# Final posterior (from the same z, samples that reached the callback)
phase_samples_final = [phase_from_latent(z_geo .+ s, prob) for s in samples_geo]
stack = reduce((a, b) -> cat(a, b; dims=3),
                [reshape(ph, N, N, 1) for ph in phase_samples_final])
phase_mean_arr = dropdims(mean(stack, dims=3), dims=3)
phase_std_arr  = dropdims(sqrt.(mean((stack .- phase_mean_arr) .^ 2; dims=3)),
                           dims=3)
phase_rms_geo  = sqrt(mean((phase_mean_arr[mask] .- phase_true[mask]) .^ 2))
median_post_std = let v = phase_std_arr[mask]
    sort(v)[length(v) ÷ 2 + 1]
end

@printf("\nHybrid VI posterior-mean phase RMS error inside aperture = %.4f rad\n",
        phase_rms_geo)
@printf("Median in-aperture posterior phase σ                    = %.4f rad\n",
        median_post_std)
@printf("(cold-start MAP reference was %.4f rad — VI vs MAP convergence)\n",
        phase_rms_map)

# 6. Final figure + convergence trace

_draw_panel(phase_mean_arr, phase_std_arr, phase_samples_final[1], N_TOTAL)
_FIG.savefig("phase_retrieval.png"; dpi=110)
println("\nSaved figure: phase_retrieval.png")

# Convergence comparison: hybrid VI trace vs the cold-start MAP reference line.
if length(_conv_iters) >= 2
    fig_c = figure("phase retrieval convergence"; figsize=(7, 4))
    ax = fig_c.add_subplot(1, 1, 1)
    ax.plot(_conv_iters, _conv_rms, "o-", label="hybrid VI (MGVI→GeoVI)")
    ax.axhline(phase_rms_map; ls="--", color="C3", label="cold-start MAP")
    ax.axvline(N_MGVI + 0.5; ls=":", color="gray", label="MGVI→GeoVI switch")
    ax.set_xlabel("VI iteration")
    ax.set_ylabel("posterior-mean phase RMS error (rad)")
    ax.set_title("Convergence: hybrid VI vs cold-start MAP (no latent vars)")
    ax.legend(fontsize=8)
    fig_c.tight_layout()
    fig_c.savefig("phase_retrieval_convergence.png"; dpi=110)
end

# 7. FITS outputs

function save_fits(filename::String, arr::AbstractMatrix{<:Real})
    f = FITS(filename, "w")
    write(f, Float64.(arr))
    close(f)
end

save_fits("phase_truth.fits",      phase_true)
save_fits("phase_geovi_mean.fits", phase_mean_arr)
save_fits("phase_geovi_std.fits",  phase_std_arr)
save_fits("phase_geovi_sample.fits", phase_samples_final[1])
save_fits("psf_truth.fits",        psf_true)
save_fits("data_observed.fits",    prob.data)
println("Saved 6 FITS files for inspection.")
println("=== Phase retrieval demo complete ===")
