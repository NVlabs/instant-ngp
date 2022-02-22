/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A network that first processes 3D position to density and
 *          subsequently direction to color.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/network_with_input_encoding.h>

NGP_NAMESPACE_BEGIN

template <typename T>
__global__ void split_inputs_nerf(
	const uint32_t n_elements,
	const uint32_t n_partition_1_dims,
	const uint32_t n_partition_2_dims,
	const uint32_t stride,
	const T* __restrict__ inputs,
	T* __restrict__ partition_1,
	T* __restrict__ partition_2
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / stride;
	const uint32_t dim_idx = i - elem_idx * stride;

	T value = inputs[i];
	if (dim_idx < n_partition_1_dims) {
		partition_1[elem_idx * n_partition_1_dims + dim_idx] = value;
	} else {
		partition_2[elem_idx * n_partition_2_dims + dim_idx - n_partition_1_dims] = value;
	}
}

template <typename T>
__global__ void grab_density_network_output(
	const uint32_t n_elements,
	const uint32_t n_partition_1_dims,
	const uint32_t stride,
	const T* __restrict__ partition_1,
	T* __restrict__ inputs
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_partition_1_dims;
	const uint32_t dim_idx = i - elem_idx * n_partition_1_dims;

// #define RELU_NON_DENSITY
#ifdef RELU_NON_DENSITY
	// ReLU for non-density dims, i.e. dim_idx > 0
	inputs[elem_idx * stride + dim_idx] = (dim_idx == 0 || partition_1[i] > (T)0.0f) ? partition_1[i] : (T)0.0f;
#else
	// ReLU for non-density dims, i.e. dim_idx > 0
	inputs[elem_idx * stride + dim_idx] = partition_1[i];
#endif
}

template <typename T>
__global__ void assemble_rgbd(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgb_stride,
	const T* __restrict__ density,
	const T* __restrict__ rgb,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 4;
	const uint32_t dim_idx = i - elem_idx * 4;

	rgbd[i] = dim_idx < 3 ? rgb[elem_idx*rgb_stride + dim_idx] : density[elem_idx*density_stride];
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*4 + dim_idx];
}

template <typename T>
__global__ void extract_density_gradient(
	const uint32_t n_elements,
	const uint32_t width,
	const uint32_t stride,
	const T* __restrict__ rgbd,
	const T* __restrict__ rgb_in,
	const T* __restrict__ rgb_in_grad,
	T* __restrict__ density_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / width;
	const uint32_t dim_idx = i - elem_idx * width;

#ifdef RELU_NON_DENSITY
	// Backprop through ReLU if not the density dimension.
	float value = (dim_idx == 0 || rgb_in[i] > (T)0.0f) ? rgb_in_grad[elem_idx*stride + dim_idx] : (T)0.0f;
#else
	float value = rgb_in_grad[elem_idx*stride + dim_idx];
#endif
	if (dim_idx == 0) {
		value += (float)rgbd[elem_idx*4+3]; // Add density gradient
	}
	density_out[i] = (T)value;
}

template <typename T>
__global__ void extract_dir_gradient(
	const uint32_t n_elements,
	const uint32_t offset,
	const uint32_t width,
	const uint32_t stride,
	const T* __restrict__ rgb_in,
	T* __restrict__ density_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / width;
	const uint32_t dim_idx = i - elem_idx * width;

	density_out[i] = rgb_in[offset + elem_idx*stride + dim_idx];
}

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = rgb_network.contains("otype") && (tcnn::equals_case_insensitive(rgb_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(rgb_network["otype"], "MegakernelMLP")) ? 16u : 8u;
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		// Assume that row-major/SoA operation will be faster, so use it if supported.
		// This assumption is valid mostly for the (hash-)grid encoding, which is
		// typically used for the position.
		if (m_pos_encoding->supports_output_layout(tcnn::RM)) {
			m_pos_encoding->set_output_layout(tcnn::RM);
		}

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->num_encoded_dims();
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 16;
		}
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		m_rgb_network_input_width = tcnn::next_multiple(m_dir_encoding->num_encoded_dims() + m_density_network->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
	}

	virtual ~NerfNetwork() { }

	void inference(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output) override {
		tcnn::GPUMatrixDynamic<T> inference_network_output{4, input.n(), stream, output.layout()};
		inference_mixed_precision(stream, input, inference_network_output);
		tcnn::linear_kernel(tcnn::cast_from<T>, 0, stream, inference_network_output.n_elements(), inference_network_output.data(), output.data());
	}

	void inference_mixed_precision(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_matrices = true) override {
		if (input.layout() != tcnn::CM || output.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::inference_mixed_precision input and output must be in column major format.");
		}

		uint32_t batch_size = input.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->num_encoded_dims(), batch_size, stream, m_pos_encoding->output_layout()};
		tcnn::GPUMatrix<T> density_network_output{m_density_network->padded_output_width(), batch_size, stream};
		tcnn::GPUMatrix<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream};
		tcnn::GPUMatrix<T> rgb_network_output{m_rgb_network->padded_output_width(), batch_size, stream};

		// Perform directional encoding and density network query in parallel
		{
			tcnn::SyncedMultiStream synced_streams{stream, 2};

			m_pos_encoding->encode(
				synced_streams.get(0),
				batch_size,
				{input.data(), input.m()},
				{density_network_input.data(), density_network_input.m()},
				nullptr,
				use_inference_matrices
			);
			m_dir_encoding->encode(
				synced_streams.get(1),
				batch_size,
				{input.data() + m_dir_offset, input.m()},
				{rgb_network_input.data() + m_density_network->padded_output_width(), rgb_network_input.m()},
				nullptr,
				use_inference_matrices
			);

			m_density_network->inference_mixed_precision(synced_streams.get(0), density_network_input, density_network_output, use_inference_matrices);

			tcnn::linear_kernel(grab_density_network_output<T>, 0, synced_streams.get(0),
				density_network_output.n_elements(), density_network_output.m(), rgb_network_input.m() /* stride */, density_network_output.data(), rgb_network_input.data()
			);
		}

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_matrices);

		tcnn::linear_kernel(assemble_rgbd<T>, 0, stream,
			output.n_elements(), density_network_output.m(), rgb_network_output.m(), density_network_output.data(), rgb_network_output.data(), output.data()
		);
	}

	void density(cudaStream_t stream, const tcnn::PitchedPtr<const float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_matrices = true) {
		uint32_t batch_size = output.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->num_encoded_dims(), batch_size, stream, m_pos_encoding->output_layout()};
		m_pos_encoding->encode(
			stream,
			batch_size,
			input,
			{density_network_input.data(), density_network_input.m()},
			nullptr,
			use_inference_matrices
		);
		m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_matrices);
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_matrices = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		density(stream, {input.data(), input.m()}, output, use_inference_matrices);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	uint32_t num_encoded_dims() const {
		return m_pos_encoding->num_encoded_dims();
	}

	std::unique_ptr<tcnn::Context> forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) override {
		if (input.layout() != tcnn::CM || (output && output->layout() != tcnn::CM)) {
			throw std::runtime_error("NerfNetwork::forward input and output must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->num_encoded_dims(), batch_size, stream, m_pos_encoding->output_layout()};
		forward->rgb_network_input = tcnn::GPUMatrix<T>{m_rgb_network_input_width, batch_size, stream};

		if (prepare_input_gradients) {
			forward->pos_encoding_forward_gradient = tcnn::GPUMatrix<float>{m_pos_encoding->num_forward_gradient_dims(), batch_size, stream};
			forward->dir_encoding_forward_gradient = tcnn::GPUMatrix<float>{m_dir_encoding->num_forward_gradient_dims(), batch_size, stream};
		}

		m_pos_encoding->encode(
			stream,
			batch_size,
			{input.data(), input.m()},
			{forward->density_network_input.data(), forward->density_network_input.m()},
			prepare_input_gradients ? forward->pos_encoding_forward_gradient.data() : nullptr,
			use_inference_matrices
		);
		m_dir_encoding->encode(
			stream,
			batch_size,
			{input.data() + m_dir_offset, input.m()},
			{forward->rgb_network_input.data() + m_density_network->padded_output_width(), forward->rgb_network_input.m()},
			prepare_input_gradients ? forward->dir_encoding_forward_gradient.data() : nullptr,
			use_inference_matrices
		);

		forward->density_network_output = tcnn::GPUMatrix<T>{m_density_network->padded_output_width(), batch_size, stream};
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_matrices, prepare_input_gradients);

		tcnn::linear_kernel(grab_density_network_output<T>, 0, stream,
			forward->density_network_output.n_elements(), forward->density_network_output.m(), forward->rgb_network_input.m() /* stride */, forward->density_network_output.data(), forward->rgb_network_input.data()
		);

		forward->rgb_network_output = tcnn::GPUMatrix<T>{m_rgb_network->padded_output_width(), batch_size, stream};
		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, &forward->rgb_network_output, use_inference_matrices, prepare_input_gradients);

		if (output) {
			tcnn::linear_kernel(assemble_rgbd<T>, 0, stream,
				output->n_elements(), forward->density_network_output.m(), forward->rgb_network_output.m(), forward->density_network_output.data(), forward->rgb_network_output.data(), output->data()
			);
		}

		return forward;
	}

	void backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override {
		if (input.layout() != tcnn::CM || output.layout() != tcnn::CM || dL_doutput.layout() != tcnn::CM || (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
			throw std::runtime_error("NerfNetwork::backward input and output must be in column major format.");
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.data(), dL_drgb.data()
		);

		tcnn::GPUMatrix<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream};
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, forward.rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_matrices, compute_param_gradients);

		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			tcnn::GPUMatrix<T> dL_ddir_encoding_output{m_dir_encoding->num_encoded_dims(), batch_size, stream};

			tcnn::linear_kernel(extract_dir_gradient<T>, 0, stream,
				dL_ddir_encoding_output.n_elements(), m_density_network->padded_output_width(), dL_ddir_encoding_output.m(), dL_drgb_network_input.m(), dL_drgb_network_input.data(), dL_ddir_encoding_output.data()
			);

			m_dir_encoding->backward(
				stream,
				batch_size,
				{dL_ddir_encoding_output.data(), dL_ddir_encoding_output.m()},
				dL_dinput ? forward.dir_encoding_forward_gradient.data() : nullptr,
				dL_dinput ? tcnn::PitchedPtr<float>{dL_dinput->data() + m_dir_offset, dL_dinput->m()} : tcnn::PitchedPtr<float>{},
				{input.data() + m_dir_offset, input.m()}
			);
		}

		tcnn::GPUMatrix<T> dL_ddensity_network_output{m_density_network->padded_output_width(), batch_size, stream};
		tcnn::linear_kernel(extract_density_gradient<T>, 0, stream,
			dL_ddensity_network_output.n_elements(),
			dL_ddensity_network_output.m(),
			dL_drgb_network_input.m(),
			dL_doutput.data(),
			forward.density_network_output.data(),
			dL_drgb_network_input.data(),
			dL_ddensity_network_output.data()
		);

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->num_encoded_dims(), batch_size, stream, m_pos_encoding->output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_matrices, compute_param_gradients);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			m_pos_encoding->backward(
				stream,
				batch_size,
				{dL_ddensity_network_input.data(), dL_ddensity_network_input.m()},
				dL_dinput ? forward.pos_encoding_forward_gradient.data() : nullptr,
				dL_dinput ? tcnn::PitchedPtr<float>{dL_dinput->data(), dL_dinput->m()} : tcnn::PitchedPtr<float>{},
				{input.data(), input.m()}
			);
		}
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		m_density_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_density_network->n_params();

		m_rgb_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_dir_encoding->n_params();
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_density_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_density_network->n_params();

		m_rgb_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_dir_encoding->n_params();
	}

	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return 4;
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t n_extra_dims() const {
		return m_n_extra_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_pos_encoding->num_encoded_dims();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->width(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_rgb_network_input_width;
		} else {
			return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.density_network_input.data(), m_pos_encoding->output_layout()};
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
		}
	}

	const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	tcnn::json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
		};
	}

private:
	std::unique_ptr<tcnn::Network<T>> m_density_network;
	std::unique_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	// // Storage of forward pass data
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> density_network_input;
		tcnn::GPUMatrix<T> density_network_output;
		tcnn::GPUMatrix<T> rgb_network_input;
		tcnn::GPUMatrix<T> rgb_network_output;
		tcnn::GPUMatrix<float> pos_encoding_forward_gradient; // Only needed when computing input gradients
		tcnn::GPUMatrix<float> dir_encoding_forward_gradient; // Only needed when computing input gradients

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
	};
};

NGP_NAMESPACE_END
