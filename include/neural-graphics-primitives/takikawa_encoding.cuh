/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   takikawa_encoding.cuh
 *  @author Thomas Müller, NVIDIA
 *  @brief  An implementation of an encoding similar to the one described in
 *          Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes
 *          by T. Takakawa et al.
 */

#pragma once

#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/random.h>


NGP_NAMESPACE_BEGIN

template <typename T, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_takikawa(
	const uint32_t num_elements,
	const uint32_t n_levels,
	const uint32_t starting_level,
	const tcnn::InterpolationType interpolation_type,
	const TriangleOctreeNode* octree_nodes,
	const TriangleOctreeDualNode* octree_dual_nodes,
	const T* __restrict__ grid,
	const tcnn::PitchedPtr<const float> data_in,
	tcnn::PitchedPtr<T> data_out,
	float* __restrict__ dy_dx
) {
	uint32_t n_features = N_FEATURES_PER_LEVEL * n_levels;

	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t encoded_index = i * n_features;
	if (encoded_index >= num_elements * n_features) return;

	int level = TriangleOctree::traverse(
		octree_nodes,
		octree_dual_nodes,
		n_levels + starting_level,
		{
			data_in(i)[0],
			data_in(i)[1],
			data_in(i)[2],
		},
		[&](const TriangleOctreeDualNode& node, uint32_t level, Eigen::Vector3f pos) {
			if (level < starting_level) {
				return;
			}
			level -= starting_level;

			Eigen::Vector3f pos_derivative;

			if (interpolation_type == tcnn::InterpolationType::Linear) {
				#pragma unroll
				for (uint32_t dim = 0; dim < 3; ++dim) {
					pos_derivative[dim] = 1.0f;
				}
			} else {
				#pragma unroll
				for (uint32_t dim = 0; dim < 3; ++dim) {
					pos_derivative[dim] = tcnn::smoothstep_derivative(pos[dim]);
					pos[dim] = tcnn::smoothstep(pos[dim]);
				}
			}

			// Tri-linear interpolation

			tcnn::vector_t<T, N_FEATURES_PER_LEVEL> result = {0};

			#pragma unroll
			for (uint32_t idx = 0; idx < 8; ++idx) {
				float weight = 1;

				#pragma unroll
				for (uint32_t dim = 0; dim < 3; ++dim) {
					if ((idx & (1<<dim)) == 0) {
						weight *= 1 - pos[dim];
					} else {
						weight *= pos[dim];
					}
				}

				int param_idx = node.vertices[idx] * N_FEATURES_PER_LEVEL;
				auto val = *(tcnn::vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[param_idx];

				// Read params
				#pragma unroll
				for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
					((T*)&result)[feature] += (T)(weight * (float)((T*)&val)[feature]);
				}
			}

			*(tcnn::vector_t<T, N_FEATURES_PER_LEVEL>*)&data_out(i)[level * N_FEATURES_PER_LEVEL] = result;

			// Gradient
			if (dy_dx) {
				const float scale = scalbnf(1.0f, level + starting_level);

				#pragma unroll
				for (uint32_t grad_dim = 0; grad_dim < 3; ++grad_dim) {
					tcnn::vector_fullp_t<N_FEATURES_PER_LEVEL> grad = {0};

					#pragma unroll
					for (uint32_t idx = 0; idx < 4; ++idx) {
						float weight = scale;
						uint32_t child_idx = 0;

						#pragma unroll
						for (uint32_t non_grad_dim = 0; non_grad_dim < 2; ++non_grad_dim) {
							const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

							if ((idx & (1<<non_grad_dim)) == 0) {
								weight *= 1 - pos[dim];
							} else {
								weight *= pos[dim];
								child_idx |= 1 << dim;
							}
						}

						int param_idx = node.vertices[child_idx] * N_FEATURES_PER_LEVEL;
						auto val_left = *(tcnn::vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[param_idx];

						child_idx |= 1 << grad_dim;
						param_idx = node.vertices[child_idx] * N_FEATURES_PER_LEVEL;
						auto val_right = *(tcnn::vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[param_idx];

						#pragma unroll
						for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
							((float*)&grad)[feature] += weight * ((float)((T*)&val_right)[feature] - (float)((T*)&val_left)[feature]) * pos_derivative[grad_dim];
						}
					}

					const uint32_t fan_out_grad = n_features * 3;
					*(tcnn::vector_fullp_t<N_FEATURES_PER_LEVEL>*)&dy_dx[i * fan_out_grad + level * N_FEATURES_PER_LEVEL + grad_dim * n_features] = grad;
				}
			}
		}
	);

	// Set output to zero for levels that were not reached
	level = max(0, level-(int)starting_level);
	for (; level < n_levels; ++level) {
		#pragma unroll
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			data_out(i)[level * N_FEATURES_PER_LEVEL + f] = (T)0.0f;
		}
	}
}

template <typename T>
__global__ void kernel_takikawa_backward_input(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const tcnn::PitchedPtr<const T> dL_dy,
	const float* __restrict__ dy_dx_pos,
	const float* __restrict__ dy_dx_oneblob,
	tcnn::PitchedPtr<float> dL_dx
) {
	const uint32_t input_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (input_index >= num_elements) return;

	const uint32_t fan_out_grad = num_grid_features * 3;

	const uint32_t i = input_index / 3;
	const uint32_t j = input_index  - i * 3;

	float result = 0;
	for (int k = 0; k < num_grid_features; ++k) {
		result += (float)dL_dy(i)[k] * dy_dx_pos[i * fan_out_grad + j * num_grid_features + k];
	}
	dL_dx(i)[j] = result;
}

template <typename T, typename GRAD_T, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_takikawa_backward(
	const uint32_t num_elements,
	const uint32_t n_levels,
	const uint32_t starting_level,
	const tcnn::InterpolationType interpolation_type,
	const TriangleOctreeNode* octree_nodes,
	const TriangleOctreeDualNode* octree_dual_nodes,
	GRAD_T* __restrict__ params_gradient,
	const tcnn::PitchedPtr<const float> data_in,
	const tcnn::PitchedPtr<const T> dL_dy
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t encoded_index = i * N_FEATURES_PER_LEVEL * n_levels;
	if (encoded_index >= num_elements * N_FEATURES_PER_LEVEL * n_levels) return;

	TriangleOctree::traverse(
		octree_nodes,
		octree_dual_nodes,
		n_levels + starting_level,
		{
			data_in(i)[0],
			data_in(i)[1],
			data_in(i)[2],
		},
		[&](const TriangleOctreeDualNode& node, uint32_t level, Eigen::Vector3f pos) {
			if (level < starting_level) {
				return;
			}
			level -= starting_level;

			if (interpolation_type == tcnn::InterpolationType::Smoothstep) {
				#pragma unroll
				for (uint32_t dim = 0; dim < 3; ++dim) {
					pos[dim] = tcnn::smoothstep(pos[dim]);
				}
			}

			auto grad = *(tcnn::vector_t<T, N_FEATURES_PER_LEVEL>*)&dL_dy(i)[N_FEATURES_PER_LEVEL * level];

			// Tri-linear interpolation

			#pragma unroll
			for (uint32_t idx = 0; idx < 8; ++idx) {
				float weight = 1;

				#pragma unroll
				for (uint32_t dim = 0; dim < 3; ++dim) {
					if ((idx & (1<<dim)) == 0) {
						weight *= 1 - pos[dim];
					} else {
						weight *= pos[dim];
					}
				}

				int param_idx = node.vertices[idx] * N_FEATURES_PER_LEVEL;

#if TCNN_MIN_GPU_ARCH >= 60 // atomicAdd(__half2) is only supported with compute capability 60 and above
				if (N_FEATURES_PER_LEVEL > 1 && std::is_same<GRAD_T, __half>::value) {
					#pragma unroll
					for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; feature += 2) {
						__half2 v = {(__half)((float)grad[feature] * weight), (__half)((float)grad[feature+1] * weight)};
						atomicAdd((__half2*)&params_gradient[param_idx + feature], v);
					}
				} else
#endif
				{
					#pragma unroll
					for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
						atomicAdd(&params_gradient[param_idx], (T)((float)grad[f] * weight));
					}
				}
			}
		}
	);
}

template <typename T, uint32_t N_FEATURES_PER_LEVEL=8>
class TakikawaEncoding : public tcnn::Encoding<T> {
public:
#if TCNN_MIN_GPU_ARCH >= 70
	// The GPUs that we tested this on do not have an efficient 1D fp16
	// atomicAdd feature. Thus, we accumulate gradients at fp32 if we're
	// forced to use 1D atomicAdds. As soon as 2D or higher is possible,
	// we can make use the efficient atomicAdd(half2) function.
	using grad_t = std::conditional_t<N_FEATURES_PER_LEVEL == 1, float, T>;
#else
	// atomicAdd(__half2) is only supported with compute capability 60 and above.
	// Since atomicAdd(__half) is relatively slow / doesn't exist for low compute
	// capabilities, accumulate in fp32 instead.
	using grad_t = float;
#endif

	TakikawaEncoding(uint32_t starting_level, bool sum_instead_of_concat, std::shared_ptr<TriangleOctree> octree, tcnn::InterpolationType interpolation_type)
		: m_starting_level{starting_level}, m_sum_instead_of_concat{sum_instead_of_concat}, m_octree{octree}, m_interpolation_type{interpolation_type} {

		if (m_starting_level >= m_octree->depth()) {
			throw std::runtime_error{"Starting level must be below octree depth."};
		}

		m_n_padded_output_dims = m_n_output_dims = N_FEATURES_PER_LEVEL * n_levels();

		if (N_FEATURES_PER_LEVEL != 1 && N_FEATURES_PER_LEVEL != 2 && N_FEATURES_PER_LEVEL != 4 && N_FEATURES_PER_LEVEL != 8) {
			throw std::runtime_error{"Number of features per level must be 1, 2, 4, or 8."};
		}
	}

	virtual ~TakikawaEncoding() { }

	void encode(
		cudaStream_t stream,
		const uint32_t num_elements,
		tcnn::PitchedPtr<const float> inputs,
		tcnn::PitchedPtr<T> outputs,
		float* dy_dx = nullptr,
		bool is_inference = false
	) const override {
		tcnn::linear_kernel(kernel_takikawa<T, N_FEATURES_PER_LEVEL>, 0, stream,
			num_elements,
			n_levels(),
			m_starting_level,
			m_interpolation_type,
			m_octree->nodes_gpu(),
			m_octree->dual_nodes_gpu(),
			is_inference ? m_params_inference : m_params,
			inputs,
			outputs,
			dy_dx
		);
	}

	void backward(
		cudaStream_t stream,
		const uint32_t num_elements,
		tcnn::PitchedPtr<const T> dL_dy, // Same shape as outputs
		const float* dy_dx, // encoded output dims x num_elements
		tcnn::PitchedPtr<float> dL_dx, // Same shape as inputs
		tcnn::PitchedPtr<const float> inputs,
		bool accumulate_param_gradients = false // whether to accumulate parameter gradients on top of the last backward() call
	) override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		{
			// We accumulate gradients with grad_t precision, which, for performance reasons, is not always T.
			// If not, accumulate in a temporary buffer and cast later.
			grad_t* params_gradient;
			if (!std::is_same<grad_t, T>::value) {
				params_gradient = (grad_t*)m_params_gradient_tmp.data();
			} else {
				params_gradient = (grad_t*)m_params_gradient;
			}

			if (!accumulate_param_gradients) {
				CUDA_CHECK_THROW(cudaMemsetAsync(params_gradient, 0, n_params() * sizeof(grad_t), stream));
			}

			tcnn::linear_kernel(kernel_takikawa_backward<T, grad_t, N_FEATURES_PER_LEVEL>, 0, stream,
				num_elements,
				n_levels(),
				m_starting_level,
				m_interpolation_type,
				m_octree->nodes_gpu(),
				m_octree->dual_nodes_gpu(),
				params_gradient,
				inputs,
				dL_dy
			);

			if (!std::is_same<grad_t, T>::value) {
				parallel_for_gpu(stream, n_params(), [grad=m_params_gradient, grad_tmp=params_gradient] __device__ (size_t i) {
					grad[i] = (T)grad_tmp[i];
				});
			}
		}

		// Gradient computation w.r.t. input
		if (dy_dx) {
			assert(dL_dx);

			tcnn::linear_kernel(kernel_takikawa_backward_input<T>, 0, stream,
				num_elements * num_dims_to_encode(),
				N_FEATURES_PER_LEVEL * n_levels(),
				dL_dy,
				dy_dx,
				dy_dx + 3 * N_FEATURES_PER_LEVEL * n_levels() * num_elements,
				dL_dx
			);
		}
	}

	uint32_t num_dims_to_encode() const override {
		return 3;
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return 3 * N_FEATURES_PER_LEVEL * n_levels();
	}

	void set_alignment(uint32_t alignment) override {
		if (m_n_output_dims != tcnn::next_multiple(m_n_output_dims, alignment)) {
			throw std::runtime_error{std::string{"TakikawaEncoding only supports number of output dims that divide into "} + std::to_string(alignment) + "; n_n_output_dims is " + std::to_string(m_n_output_dims)};
		}
	}

	uint32_t min_alignment() const override {
		return N_FEATURES_PER_LEVEL;
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		m_params = params;
		m_params_inference = inference_params;
		m_params_gradient = gradients;
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		set_params(params, inference_params, backward_params, gradients);

		// Initialize the encoding from the GPU, because the number of parameters can be quite large.
		tcnn::generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f, 1e-4f);

		// Only needs temporary storage if gradients are computed with different precision from T.
		if (!std::is_same<grad_t, T>::value) {
			m_params_gradient_tmp.resize(n_params());
		}
	}

	size_t n_params() const override {
		return N_FEATURES_PER_LEVEL * m_octree->n_vertices();
	}

	uint32_t n_levels() const {
		return m_octree->depth() - m_starting_level;
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		// Even though we have parameters, they can't really be considered a "layer".
		// So we return an empty array here.
		return {};
	}

private:
	uint32_t m_starting_level;
	bool m_sum_instead_of_concat;

	uint32_t m_n_dims_to_pass_through;

	// derived sizes
	uint32_t m_n_input_dims;
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;

	// Storage of params
	T* m_params;
	T* m_params_inference;

	tcnn::GPUMemory<grad_t> m_params_gradient_tmp;
	T* m_params_gradient;

	std::shared_ptr<TriangleOctree> m_octree;
	tcnn::InterpolationType m_interpolation_type;
};

NGP_NAMESPACE_END
