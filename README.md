# VarInf.jl

A minimalistic variational-inference toolkit in Julia inspired by the NIFTy package
([GitHub](https://github.com/nifty-ppl/nifty), [manual](https://ift.pages.mpcdf.de/nifty/)).
It is built around the single abstract type:

```julia
abstract type AbstractInferenceProblem end
```

The following six methods are needed to set up a problem with MAP / MGVI / GeoVI:

| Method | Returns | Required for |
|---|---|---|
| `energy_and_gradient(prob, z)` | `(E::Float64, ∇E::Vector{Float64})` | MAP, MGVI |
| `latent_size(prob)` | `Int` (length of `z`) | all |
| `data_size(prob)` | `Int` (length of data) | GeoVI, hybrid |
| `transformation(prob, z)` | whitened model `T(z) = model(z) / σ` | GeoVI, hybrid |
| `right_sqrt_metric(prob, z, v)` | JVP `J_T(z) · v` | GeoVI, hybrid |
| `left_sqrt_metric(prob, z, v)` | VJP `J_T(z)' · v` | GeoVI, hybrid |

The latent `z` is assumed to have a standard-normal prior. One needs to map physical
parameters from `z` inside `transformation` (e.g. `μ = μ₀ + σ·z₁`, `F = exp(z₂)`).

## Worked examples

| Example | Latent | Forward op | Purpose |
|---|---|---|---|
| `deconv_tiny.jl` | 1 024 | linear FFT | smallest end-to-end demo |
| `deconv_saturn.jl` | 16 384 | linear FFT | demo 128×128 Saturn deconvolution |
| `two_gaussians.jl` | 6 | analytic Gaussians | test without correlated field |
| `point_sources.jl` | 9 | direct sum | 2D parametric astrometry/photometry |
| `phase_retrieval.jl` | 4 096 | `|FFT(A·exp(iφ))|² ⊛` | **nonlinear** chain (Kolmogorov phase screen) |

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
Among others, adjoint identities to 1e-9, finite-difference gradient checks, MAP convergence sanity, and finite-output checks.

## References

- J. Knollmüller and T. A. Enßlin, "Metric Gaussian Variational Inference", arXiv:1901.11033 (2019). [arXiv](https://arxiv.org/abs/1901.11033)
- P. Frank, R. Leike and T. A. Enßlin, "Geometric Variational Inference", *Entropy* **23**(7), 853 (2021). [doi:10.3390/e23070853](https://doi.org/10.3390/e23070853)
- P. Frank, "Geometric Variational Inference and Its Application to Bayesian Imaging", *Phys. Sci. Forum* **5**, 6 (2022). [doi:10.3390/psf2022005006](https://doi.org/10.3390/psf2022005006)
- G. Edenhofer, P. Frank, J. Roth, R. H. Leike, M. Guerdi, L. I. Scheel-Platz, M. Guardiani, V. Eberle, M. Westerkamp and T. A. Enßlin, "Re-Envisioning Numerical Information Field Theory (NIFTy.re): A Library for Gaussian Processes and Variational Inference", *Journal of Open Source Software* **9**(98), 6593 (2024). [doi:10.21105/joss.06593](https://doi.org/10.21105/joss.06593)
- V. Eberle, M. Guardiani, M. Westerkamp, P. Frank, J. Rüstig, J. Stadler and T. A. Enßlin, "J-UBIK: The JAX-accelerated Universal Bayesian Imaging Kit", *Journal of Open Source Software* **11**(120), 7768 (2026). [doi:10.21105/joss.07768](https://doi.org/10.21105/joss.07768), [arXiv:2409.10381](https://arxiv.org/abs/2409.10381)

## Acknowledgments

Thanks to Torsten Enßlin for productive discussions about NIFTy.
