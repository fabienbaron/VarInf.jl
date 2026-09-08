module VarInf

using FFTW
using LinearAlgebra
using Printf

include("abstract_inference_problem.jl")
include("cg.jl")
include("newton_cg.jl")
include("correlated_fields.jl")
include("multi_axis_correlated_field.jl")
include("algorithms.jl")

# Protocol
export AbstractInferenceProblem
export energy_and_gradient, latent_size, transformation
export right_sqrt_metric, left_sqrt_metric, data_size
export report_latents, report_row, latent_blocks, whitened_data
export prior_inv_covariance_mul, prior_inv_sqrt_covariance_mul, prior_energy

# Correlated-field building blocks
export FourierGridInfo, fourier_mode_distributor
export Axis1DInfo, axis_mode_distributor
export CorrFieldConfig
export amplitude_spectrum, amplitude_spectrum_jvp, amplitude_spectrum_adjoint
export MaternConfig, matern_amplitude, matern_amplitude_jvp, matern_amplitude_adjoint
export make_smoothing_kernel, harmonic_smooth

# Multi-axis correlated field
export CorrelatedField, latent_unpack, mcf_jvp, mcf_vjp

# Algorithms
export reconstruct_map, reconstruct_map_with_info
export reconstruct_mgvi, reconstruct_geovi, reconstruct_hybrid

end # module VarInf
