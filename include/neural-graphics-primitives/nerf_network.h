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

namespace ngp {

template <typename T>
__global__ void extract_density(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ density,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rgbd[i * rgbd_stride] = density[i * density_stride];
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const uint32_t output_stride,
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*output_stride + dim_idx];
}

template <typename T>
__global__ void add_density_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

template <typename T>
class NerfNetwork : public Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = minimum_alignment(rgb_network);
		m_dir_encoding.reset(create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 16;
		}
		m_density_network.reset(create_network<T>(local_density_network_config));

		m_rgb_network_input_width = next_multiple(m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(create_network<T>(local_rgb_network_config));

		m_density_model = std::make_shared<NetworkWithInputEncoding<T>>(m_pos_encoding, m_density_network);
	}

	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();
		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

		auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == AoS ? density_network_output.stride() : 1,
			output.layout() == AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == AoS ? 1 : batch_size)
		);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->rgb_network_input = GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);

		auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->rgb_network_output = GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

		if (output) {
			linear_kernel(extract_density<T>, 0, stream,
				batch_size, m_dir_encoding->preferred_output_layout() == AoS ? forward->density_network_output.stride() : 1, padded_output_width(), forward->density_network_output.data(), output->data()+3
			);
		}

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

		const GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
		);

		GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void density(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_density_model->set_jit_fusion(this->jit_fusion());
		m_density_model->inference_mixed_precision(stream, input.slice_rows(0, m_pos_encoding->input_width()), output, use_inference_params);
	}

	std::unique_ptr<Context> density_forward(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
		if (input.layout() != CM) {
			throw std::runtime_error("NerfNetwork::density_forward input must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->density_network_output = GPUMatrixDynamic<T>{output->data(), m_density_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, output ? &forward->density_network_output : nullptr, use_inference_params, prepare_input_gradients);

		return forward;
	}

	void density_backward(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) {
		if (input.layout() != CM || (dL_dinput && dL_dinput->layout() != CM)) {
			throw std::runtime_error("NerfNetwork::density_backward input must be in column major format.");
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, output, dL_doutput, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		m_density_model->set_params(params, inference_params, gradients);

		size_t offset = 0;
		m_density_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_density_network->n_params();

		m_rgb_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_dir_encoding->n_params();
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		m_density_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_density_network->n_params();

		m_rgb_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_dir_encoding->n_params();
	}

	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
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
			return m_pos_encoding->padded_output_width();
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

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
		}
	}

	const std::shared_ptr<Encoding<T>>& pos_encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	const std::shared_ptr<Network<T>>& density_network() const {
		return m_density_network;
	}

	const std::shared_ptr<Network<T>>& rgb_network() const {
		return m_rgb_network;
	}

	json hyperparams() const override {
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

	std::string generate_device_function(const std::string& name) const override {
		std::string density_network = name + "_density_network";
		std::string rgb_network = name + "_rgb_network";
		std::string pos_encoding = name + "_pos_encoding";
		std::string dir_encoding = name + "_dir_encoding";

		std::ostringstream preamble;
		preamble
			<< m_density_network->generate_device_function(density_network) << "\n\n"
			<< m_rgb_network->generate_device_function(rgb_network) << "\n\n"
			<< m_pos_encoding->generate_device_function(pos_encoding) << "\n\n"
			<< m_dir_encoding->generate_device_function(dir_encoding) << "\n\n"
			;

		std::string body = dfmt(1, R"(
				auto pos_enc_out = {POS_ENC}(input.slice<0, {POS_ENC_DIMS_IN}>(), params + {POS_ENC_PARAMS_OFFSET}, fwd_ctx ? fwd_ctx + WARP_SIZE * {POS_ENC_FWD_CTX_OFFSET} : nullptr);

				{RGB_MLP_IN} rgb_mlp_in;
				rgb_mlp_in.slice<0, {DENSITY_MLP_DIMS_OUT}>() = {DENSITY_MLP}(pos_enc_out, params, fwd_ctx);
				rgb_mlp_in.slice<{DENSITY_MLP_DIMS_OUT}, {DIR_ENC_DIMS_OUT}>() = {DIR_ENC}(input.slice<{DIR_OFFSET}, {DIR_ENC_DIMS_IN}>(), params + {DIR_ENC_PARAMS_OFFSET}, fwd_ctx ? fwd_ctx + WARP_SIZE * {DIR_ENC_FWD_CTX_OFFSET} : nullptr);

				auto rgb_mlp_out = {RGB_MLP}(rgb_mlp_in, params + {RGB_MLP_PARAMS_OFFSET}, fwd_ctx ? fwd_ctx + WARP_SIZE * {RGB_MLP_FWD_CTX_OFFSET} : nullptr);

				return {{rgb_mlp_out[0], rgb_mlp_out[1], rgb_mlp_out[2], rgb_mlp_in[0]}};
			)",
			"POS_ENC"_a = pos_encoding,
			"POS_ENC_DIMS_IN"_a = m_pos_encoding->input_width(),
			"POS_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params(),
			"POS_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes(),
			"RGB_MLP"_a = rgb_network,
			"RGB_MLP_IN"_a = m_rgb_network->generate_vec_in(),
			"RGB_MLP_PARAMS_OFFSET"_a = m_density_network->n_params(),
			"RGB_MLP_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes(),
			"DENSITY_MLP"_a = density_network,
			"DENSITY_MLP_DIMS_OUT"_a = m_density_network->output_width(),
			"DIR_ENC"_a = dir_encoding,
			"DIR_ENC_DIMS_IN"_a = m_dir_encoding->input_width(),
			"DIR_ENC_DIMS_OUT"_a = m_dir_encoding->output_width(),
			"DIR_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params() + m_pos_encoding->n_params(),
			"DIR_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes() + m_pos_encoding->device_function_fwd_ctx_bytes(),
			"DIR_OFFSET"_a = m_dir_offset
		);

		return fmt::format("{}{}", preamble.str(), this->generate_device_function_from_body(name, body));
	}

	std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override {
		std::string density_network = name + "_density_network";
		std::string rgb_network = name + "_rgb_network";
		std::string pos_encoding = name + "_pos_encoding";
		std::string dir_encoding = name + "_dir_encoding";

		std::ostringstream preamble;
		preamble
			<< m_density_network->generate_backward_device_function(density_network, n_threads) << "\n\n"
			<< m_rgb_network->generate_backward_device_function(rgb_network, n_threads) << "\n\n"
			<< m_pos_encoding->generate_backward_device_function(pos_encoding, n_threads) << "\n\n"
			<< m_dir_encoding->generate_backward_device_function(dir_encoding, n_threads) << "\n\n"
			;

		std::string body = dfmt(1, R"(
				bool requires_pos_encoding_bwd = {POS_ENC_N_PARAMS} != 0 || dL_dx;
				bool requires_dir_encoding_bwd = {DIR_ENC_N_PARAMS} != 0 || dL_dx;

				{RGB_MLP_IN} dL_drgb_mlp_in;
				{RGB_MLP}(
					{RGB_MLP_OUT}(dL_dy.rgb()),
					params + {RGB_MLP_PARAMS_OFFSET},
					fwd_ctx + WARP_SIZE * {RGB_MLP_FWD_CTX_OFFSET},
					dL_dparams ? dL_dparams + {RGB_MLP_PARAMS_OFFSET} : nullptr,
					&dL_drgb_mlp_in
				);
				dL_drgb_mlp_in[0] = dL_drgb_mlp_in[0] + dL_dy[3];

				if (requires_dir_encoding_bwd) {{
					{DIR_ENC}(
						dL_drgb_mlp_in.slice<{DENSITY_MLP_DIMS_OUT}, {DIR_ENC_DIMS_OUT}>(),
						params + {DIR_ENC_PARAMS_OFFSET},
						fwd_ctx + WARP_SIZE * {DIR_ENC_FWD_CTX_OFFSET},
						dL_dparams ? dL_dparams + {DIR_ENC_PARAMS_OFFSET} : nullptr,
						dL_dx ? &dL_dx->slice<{DIR_OFFSET}, {DIR_ENC_DIMS_IN}>() : nullptr
					);
				}}

				{POS_ENC_OUT} dL_dpos_enc_out;
				{DENSITY_MLP}(
					dL_drgb_mlp_in.slice<0, {DENSITY_MLP_DIMS_OUT}>(),
					params,
					fwd_ctx,
					dL_dparams,
					requires_pos_encoding_bwd ? &dL_dpos_enc_out : nullptr
				);

				if (requires_pos_encoding_bwd) {{
					{POS_ENC}(
						dL_dpos_enc_out,
						params + {POS_ENC_PARAMS_OFFSET},
						fwd_ctx + WARP_SIZE * {POS_ENC_FWD_CTX_OFFSET},
						dL_dparams ? dL_dparams + {POS_ENC_PARAMS_OFFSET} : nullptr,
						dL_dx ? &dL_dx->slice<0, {POS_ENC_DIMS_IN}>() : nullptr
					);
				}}
			)",
			"POS_ENC"_a = pos_encoding,
			"POS_ENC_DIMS_IN"_a = m_pos_encoding->input_width(),
			"POS_ENC_OUT"_a = m_pos_encoding->generate_vec_out(),
			"POS_ENC_N_PARAMS"_a = m_pos_encoding->n_params(),
			"POS_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params(),
			"POS_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes(),
			"RGB_MLP"_a = rgb_network,
			"RGB_MLP_IN"_a = m_rgb_network->generate_vec_in(),
			"RGB_MLP_OUT"_a = m_rgb_network->generate_vec_out(),
			"RGB_MLP_PARAMS_OFFSET"_a = m_density_network->n_params(),
			"RGB_MLP_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes(),
			"DENSITY_MLP"_a = density_network,
			"DENSITY_MLP_DIMS_OUT"_a = m_density_network->output_width(),
			"DIR_ENC"_a = dir_encoding,
			"DIR_ENC_DIMS_IN"_a = m_dir_encoding->input_width(),
			"DIR_ENC_DIMS_OUT"_a = m_dir_encoding->output_width(),
			"DIR_ENC_N_PARAMS"_a = m_dir_encoding->n_params(),
			"DIR_ENC_PARAMS_OFFSET"_a = m_density_network->n_params() + m_rgb_network->n_params() + m_pos_encoding->n_params(),
			"DIR_ENC_FWD_CTX_OFFSET"_a = m_density_network->device_function_fwd_ctx_bytes() + m_rgb_network->device_function_fwd_ctx_bytes() + m_pos_encoding->device_function_fwd_ctx_bytes(),
			"DIR_OFFSET"_a = m_dir_offset
		);

		return fmt::format("{}{}", preamble.str(), this->generate_backward_device_function_from_body(name, body));
	}

	uint32_t device_function_fwd_ctx_bytes() const override {
		return
			m_density_network->device_function_fwd_ctx_bytes() +
			m_rgb_network->device_function_fwd_ctx_bytes() +
			m_pos_encoding->device_function_fwd_ctx_bytes() +
			m_dir_encoding->device_function_fwd_ctx_bytes()
			;
	}

	bool device_function_fwd_ctx_aligned_per_element() const override {
		return false;
	}

	uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, GradientMode param_gradients_mode) const override {
		return std::max(
			std::max(
				m_density_network->backward_device_function_shmem_bytes(n_threads, param_gradients_mode),
				m_rgb_network->backward_device_function_shmem_bytes(n_threads, param_gradients_mode)
			),
			std::max(
				m_pos_encoding->backward_device_function_shmem_bytes(n_threads, param_gradients_mode),
				m_dir_encoding->backward_device_function_shmem_bytes(n_threads, param_gradients_mode)
			)
		);
	}

	void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override {
		m_density_network->convert_params_to_jit_layout(stream, use_inference_params);
		m_rgb_network->convert_params_to_jit_layout(stream, use_inference_params);
		m_pos_encoding->convert_params_to_jit_layout(stream, use_inference_params);
		m_dir_encoding->convert_params_to_jit_layout(stream, use_inference_params);
	}

	void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override {
		m_density_network->convert_params_from_jit_layout(stream, use_inference_params);
		m_rgb_network->convert_params_from_jit_layout(stream, use_inference_params);
		m_pos_encoding->convert_params_from_jit_layout(stream, use_inference_params);
		m_dir_encoding->convert_params_from_jit_layout(stream, use_inference_params);
	}

private:
	std::shared_ptr<Network<T>> m_density_network;
	std::shared_ptr<Network<T>> m_rgb_network;
	std::shared_ptr<Encoding<T>> m_pos_encoding;
	std::shared_ptr<Encoding<T>> m_dir_encoding;

	// Aggregates m_pos_encoding and m_density_network
	std::shared_ptr<NetworkWithInputEncoding<T>> m_density_model;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	// // Storage of forward pass data
	struct ForwardContext : public Context {
		GPUMatrixDynamic<T> density_network_input;
		GPUMatrixDynamic<T> density_network_output;
		GPUMatrixDynamic<T> rgb_network_input;
		GPUMatrix<T> rgb_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
	};
};

}
