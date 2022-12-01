/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   trainable_buffer.cuh
 *  @author Thomas Müller, NVIDIA
 *  @brief  An implementation of a trainable N-channel buffer within the tcnn API.
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

NGP_NAMESPACE_BEGIN

template <uint32_t N_DIMS, uint32_t RANK, typename T>
class TrainableBuffer : public tcnn::DifferentiableObject<float, T, T> {
	using ResVector = Eigen::Matrix<int, RANK, 1>;

public:
	TrainableBuffer(const ResVector& resolution) : m_resolution{resolution} {
		m_params_gradient_weight.resize(n_params());
	}

	virtual ~TrainableBuffer() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_matrices = true) override {
		throw std::runtime_error{"The trainable buffer does not support inference(). Its content is meant to be used externally."};
	}

	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) override {
		throw std::runtime_error{"The trainable buffer does not support forward(). Its content is meant to be used externally."};
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		throw std::runtime_error{"The trainable buffer does not support backward(). Its content is meant to be used externally."};
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override { }

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override {
		// Initialize the buffer to zero from the GPU
		CUDA_CHECK_THROW(cudaMemset(params_full_precision, 0, n_params()*sizeof(float)));
	}

	size_t n_params() const override {
		return m_resolution.prod() * N_DIMS;
	}

	uint32_t input_width() const override {
		return RANK;
	}

	uint32_t padded_output_width() const override {
		return N_DIMS;
	}

	uint32_t output_width() const override {
		return N_DIMS;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return {};
	}

	T* gradient_weights() const {
		return m_params_gradient_weight.data();
	}

	tcnn::json hyperparams() const override {
		return {
			{"otype", "TrainableBuffer"},
		};
	}

private:
	ResVector m_resolution;
	tcnn::GPUMemory<T> m_params_gradient_weight;
};

NGP_NAMESPACE_END
