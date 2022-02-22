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

// size of the density/occupancy grid in number of cells along an axis.
inline constexpr __device__ uint32_t NERF_GRIDSIZE() {
	return 128;
}

struct NerfPayload {
	Eigen::Vector3f origin;
	Eigen::Vector3f dir;
	float t;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct RaysNerfSoa {
#ifdef __NVCC__
	void copy_from_other_async(const RaysNerfSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgba, other.rgba, size * sizeof(Eigen::Array4f), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, size * sizeof(NerfPayload), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(Eigen::Array4f* rgba, NerfPayload* payload, size_t size) {
		this->rgba = rgba;
		this->payload = payload;
		this->size = size;
	}

	Eigen::Array4f* rgba;
	NerfPayload* payload;
	size_t size;
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
	NGP_HOST_DEVICE void set_with_optional_light_dir(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir, float dt, const Eigen::Vector3f& light_dir, uint32_t stride_in_bytes) {
		this->dt = dt;
		this->pos = NerfPosition(pos, dt);
		this->dir = NerfDirection(dir, dt);
		if (stride_in_bytes >= sizeof(Eigen::Vector3f) + sizeof(NerfCoordinate)) {
			*(Eigen::Vector3f*)(this+1) = light_dir;
		}
	}
	NGP_HOST_DEVICE void copy_with_optional_light_dir(const NerfCoordinate& inp, uint32_t stride_in_bytes) {
		*this = inp;
		if (stride_in_bytes >= sizeof(Eigen::Vector3f) + sizeof(NerfCoordinate)) {
			*(Eigen::Vector3f*)(this+1) = *(Eigen::Vector3f*)(&inp+1);
		}
	}

	NerfPosition pos;
	float dt;
	NerfDirection dir;
};

struct NerfCoordinateWithLightDir : NerfCoordinate {
	Eigen::Vector3f light_dir;
};

NGP_NAMESPACE_END
