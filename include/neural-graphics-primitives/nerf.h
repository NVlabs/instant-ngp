/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */


#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/gpu_memory.h>


NGP_NAMESPACE_BEGIN

static constexpr __device__ uint32_t NERF_GRIDSIZE = 128; // size of the density/occupancy grid.

struct NerfPayload {
	Eigen::Vector3f origin;
	Eigen::Vector3f dir;
	float t;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct RaysNerfSoa {
	void enlarge(size_t n_elements) {
		rgba.enlarge(n_elements);
		payload.enlarge(n_elements);
	}
#ifdef __NVCC__
	void copy_from_other_async(const RaysNerfSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgba.data(), other.rgba.data(), rgba.get_bytes(), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload.data(), other.payload.data(), payload.get_bytes(), cudaMemcpyDeviceToDevice, stream));
	}
#endif
	tcnn::GPUMemory<Eigen::Array4f> rgba;
	tcnn::GPUMemory<NerfPayload> payload;
};


struct NerfPosition {
	NGP_HOST_DEVICE NerfPosition(const Eigen::Vector3f& pos, float dt) : p{pos} {}
	Eigen::Vector3f p;
};

struct NerfDirection {
	NGP_HOST_DEVICE NerfDirection(const Eigen::Vector3f& dir, float dt) : d{dir} {}
	Eigen::Vector3f d;
};

struct NerfCoordinate {
	NGP_HOST_DEVICE NerfCoordinate(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}
	NerfPosition pos;
	float dt;
	NerfDirection dir;
};

NGP_NAMESPACE_END
