/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   sdf.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/gpu_memory.h>

NGP_NAMESPACE_BEGIN

struct SdfPayload {
	Eigen::Vector3f dir;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct RaysSdfSoa {
#ifdef __NVCC__
	void enlarge(size_t n_elements) {
		pos.enlarge(n_elements);
		normal.enlarge(n_elements);
		distance.enlarge(n_elements);
		prev_distance.enlarge(n_elements);
		total_distance.enlarge(n_elements);
		min_visibility.enlarge(n_elements);
		payload.enlarge(n_elements);
	}

	void copy_from_other_async(uint32_t n_elements, const RaysSdfSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(pos.data(), other.pos.data(), n_elements * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(normal.data(), other.normal.data(), n_elements * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(distance.data(), other.distance.data(), n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(prev_distance.data(), other.prev_distance.data(), n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(total_distance.data(), other.total_distance.data(), n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(min_visibility.data(), other.min_visibility.data(), n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload.data(), other.payload.data(), n_elements * sizeof(SdfPayload), cudaMemcpyDeviceToDevice, stream));
	}
#endif
	tcnn::GPUMemory<Eigen::Vector3f> pos;
	tcnn::GPUMemory<Eigen::Vector3f> normal;
	tcnn::GPUMemory<float> distance;
	tcnn::GPUMemory<float> prev_distance;
	tcnn::GPUMemory<float> total_distance;
	tcnn::GPUMemory<float> min_visibility;
	tcnn::GPUMemory<SdfPayload> payload;
};

struct BRDFParams {
	float metallic=0.f;
	float subsurface=0.f;
	float specular=1.f;
	float roughness=0.5f;
	float sheen=0.f;
	float clearcoat=0.f;
	float clearcoat_gloss=0.f;
	Eigen::Vector3f basecolor=Eigen::Vector3f(0.8f,0.8f,0.8f);
	Eigen::Vector3f ambientcolor=Eigen::Vector3f(0.f,0.f,0.f);
};

NGP_NAMESPACE_END
