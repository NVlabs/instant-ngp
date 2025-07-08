/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <neural-graphics-primitives/sdf_device.cuh>

#include <tiny-cuda-nn/gpu_memory.h>

namespace ngp {

struct RaysSdfSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(uint32_t n_elements, const RaysSdfSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(pos, other.pos, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(normal, other.normal, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(distance, other.distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(prev_distance, other.prev_distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(total_distance, other.total_distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(min_visibility, other.min_visibility, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, n_elements * sizeof(SdfPayload), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(vec3* pos, vec3* normal, float* distance, float* prev_distance, float* total_distance, float* min_visibility, SdfPayload* payload) {
		this->pos = pos;
		this->normal = normal;
		this->distance = distance;
		this->prev_distance = prev_distance;
		this->total_distance = total_distance;
		this->min_visibility = min_visibility;
		this->payload = payload;
	}

	vec3* pos;
	vec3* normal;
	float* distance;
	float* prev_distance;
	float* total_distance;
	float* min_visibility;
	SdfPayload* payload;
};

} // namespace ngp
